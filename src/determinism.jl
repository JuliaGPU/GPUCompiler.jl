# Reproducibility fixes for the LLVM modules GPUCompiler generates.


## module layout workaround

# Julia <= 1.13.0-rc3 merges per-CodeInstance modules by iterating a pointer-keyed map.
# The 1.13 backport julia#62782 landed after rc3; Julia 1.14 uses a different codegen path.
#
# Generated-name counters are allocated in emission order. Sorting by them restores that
# order; uncountered declarations sort by name afterwards.

# The LLVM uniquing suffix a merge appends to a clashing local name.
const CODEGEN_UNIQUING_SUFFIX = r"\.[0-9]+$"

# The trailing codegen counter of a generated name, if any.
const CODEGEN_COUNTER_SUFFIX = r"[_#]([0-9]+)$"

function module_layout_key(name::String)
    base = replace(name, CODEGEN_UNIQUING_SUFFIX => "")
    m = match(CODEGEN_COUNTER_SUFFIX, base)
    counter = m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    return (counter, base, name)
end

# Merge auto-suffixed clones of a private constant back into the copy that kept the bare name.
# Only merge constants whose addresses have no identity and whose relevant attributes and
# context-uniqued initializers match.
function merge_constant_clones!(mod::LLVM.Module)
    mod_gvs = mod.globals
    merged = false
    for gv in collect(mod_gvs)
        name = gv.name
        m = match(CODEGEN_UNIQUING_SUFFIX, name)
        m === nothing && continue
        base_name = name[1:prevind(name, m.offset)]
        haskey(mod_gvs, base_name) || continue
        base = mod_gvs[base_name]
        mergeable(g) = g.linkage == LLVM.Linkage.Private && g.constant &&
                       g.unnamed_addr == LLVM.UnnamedAddr.Global && !isdeclaration(g)
        (mergeable(gv) && mergeable(base)) || continue
        gv.value_type == base.value_type || continue
        gv.global_value_type == base.global_value_type || continue
        gv.alignment == base.alignment || continue
        gv.section == base.section || continue
        gv.visibility == base.visibility || continue
        gv.dllstorage == base.dllstorage || continue
        gv.threadlocal_mode == base.threadlocal_mode || continue
        gv.externally_initialized == base.externally_initialized || continue
        gv.initializer == base.initializer || continue
        replace_uses!(gv, base)
        erase!(gv)
        merged = true
    end
    return merged
end

"""
    canonicalize_module_layout!(mod::LLVM.Module)

Restore emission order and merge equivalent private constants renamed during linking.
"""
function canonicalize_module_layout!(mod::LLVM.Module)
    merge_constant_clones!(mod)
    sort!(mod.functions; by = f -> module_layout_key(f.name))
    sort!(mod.globals; by = gv -> module_layout_key(gv.name))
    return mod
end


## compile-unit deduplication

# Repoint the operands of the metadata graph rooted at `md` according to `replacement`.
function repoint!(@nospecialize(md), replacement::Dict{LLVM.Metadata,LLVM.Metadata},
                  visited::Set{LLVM.Metadata})
    md isa LLVM.MDNode || return
    md in visited && return
    push!(visited, md)
    # iterate a copy, as replacing an operand re-uniques the node
    for (i, op) in enumerate(collect(md.operands))
        op isa LLVM.MDNode || continue
        repl = get(replacement, op, nothing)
        if repl !== nothing
            md.operands[i] = repl
        else
            repoint!(op, replacement, visited)
        end
    end
end

# LLVM does not merge distinct `DICompileUnit`s when linking. Group compile units by their
# uniqued operands, repoint references to the first unit in each group, and shrink
# `llvm.dbg.cu` to those canonical units.
function dedup_compile_units!(mod::LLVM.Module)
    mds = mod.metadata
    haskey(mds, "llvm.dbg.cu") || return false
    cus = mds["llvm.dbg.cu"].operands
    length(cus) <= 1 && return false

    canonical = Dict{Vector{Union{Nothing,LLVM.Metadata}},LLVM.Metadata}()
    canonical_cus = LLVM.Metadata[]
    replacement = Dict{LLVM.Metadata,LLVM.Metadata}()
    for cu in cus
        cu === nothing && continue
        key = Union{Nothing,LLVM.Metadata}[cu.operands...]
        canon = get!(canonical, key, cu)
        if canon == cu
            push!(canonical_cus, cu)
        else
            replacement[cu] = canon
        end
    end
    isempty(replacement) && return false

    # Metadata forms a graph, so walk every attachment reachable from module values.
    visited = Set{LLVM.Metadata}()
    has_records = LLVM.version() >= v"19"
    for f in mod.functions
        sp = f.subprogram
        sp === nothing || repoint!(sp, replacement, visited)
        for bb in f.blocks, inst in bb.instructions
            for (kind, md) in inst.metadata
                repoint!(md, replacement, visited)
            end
            # Debug records are not metadata attachments, and can hold the only reference to
            # the subprogram of an inlined function whose code was optimized away, or (through
            # the scope of the variable's type) to a compile unit.
            if has_records
                for record in inst.debug_records
                    repoint!(record.debug_location, replacement, visited)
                    record.kind == LLVM.API.LLVMDbgRecordLabel ||
                        repoint!(record.variable, replacement, visited)
                end
            end
        end
    end
    for gv in mod.globals
        for (kind, md) in gv.metadata
            repoint!(md, replacement, visited)
        end
    end

    # The verifier requires every reachable compile unit to be listed here.
    nmd = mds["llvm.dbg.cu"]
    empty!(nmd.operands)
    for cu in canonical_cus
        push!(nmd.operands, cu)
    end
    return true
end


## codegen name normalization

# Replace the names matching `codegen_name` in the strings of the metadata graph rooted at
# `md`, by `normalize`d names or, in the root node itself, by `replacement`.
function normalize_metadata!(@nospecialize(md), visited::Set{LLVM.MDNode}, codegen_name::Regex,
                             normalize, replacement=nothing)
    md isa LLVM.MDNode || return
    md in visited && return
    push!(visited, md)
    # (a copy, as replacing an operand can re-unique the node)
    for (i, op) in enumerate(collect(md.operands))
        if op isa LLVM.MDString
            str = convert(String, op)
            occursin(codegen_name, str) || continue
            new = replacement === nothing ? normalize(str) : replacement
            new == str || (md.operands[i] = LLVM.MDString(new))
        elseif op isa LLVM.MDNode
            normalize_metadata!(op, visited, codegen_name, normalize)
        end
    end
end

# Julia names each generated LLVM function `julia_<name>_<counter>`. Up to Julia 1.13, the
# counter is drawn from a process-global codegen sequence and so differs from one session to
# the next. Left anywhere in the emitted bitcode it makes a back-end's output (e.g., the AIR
# and the metallib wrapping it) non-reproducible across sessions, defeating byte-stable
# caching and content-keyed binary archives. And a symbol name is not the only place it
# appears: inlining a Julia function leaves its name behind in the block labels the inliner
# synthesizes, in the (now orphaned) `DISubprogram` its debug locations still point at, and
# in the alias-scope strings LLVM's inliner derives from it.
#
# So rewrite each of these occurrences: map each distinct codegen name to a deterministic
# module-local form (its rank in a fixed traversal), then substitute that map into symbol
# names, value names, and metadata strings. A subprogram belonging to a function instead
# adopts that function's current name, so an entry that `irgen.jl` already renamed to a
# stable mangled symbol keeps a `linkageName` matching the symbol it describes.
# The Metal and SPIR-V back-ends run it in `mcgen`, before writing the module to bitcode.
function normalize_julia_symbol_names!(mod::LLVM.Module)
    # The counter is the trailing digit group, so match the name part lazily. The word
    # boundaries keep the pattern from biting into a longer token: `myjulia_foo_1` is a user
    # symbol that merely ends this way, and `julia_foo_12bar` is one that merely contains it.
    # The name part must cover more than `\w`: mutating functions carry `!`
    # (`julia_record_exception!_18521`) and closures carry `#` (`julia_#kernel#123_456`),
    # and a missed name leaks the per-session counter into the bitcode — exactly the
    # byte-instability this pass exists to prevent.
    codegen_name = r"(?<!\w)julia_[\w!#]*?_[0-9]+(?!\w)"
    renames = Dict{String,String}()
    deterministic(match::AbstractString) = get!(renames, match) do
        replace(match, r"_[0-9]+$" => "") * "_$(length(renames) + 1)"
    end
    normalize(str::AbstractString) = replace(str, codegen_name => deterministic)
    # (unspecialized, and with `LLVM.name` rather than the `name` property, so that this is
    # compiled once rather than for every type of instruction)
    rename!(@nospecialize(value)) = let str = LLVM.name(value)
        occursin(codegen_name, str) && LLVM.name!(value, normalize(str))
    end

    # Metadata forms a graph (a debug location points at its subprogram, an alias scope at its
    # domain), so walk it, replacing every string that carries a name. Only nodes reachable
    # from a function or an instruction are visited, which is where inlining leaves its traces.
    visited = Set{LLVM.MDNode}()

    # the instruction metadata that can name a function: a debug location points at the
    # subprogram it came from (orphaned once that function is inlined away), and the alias
    # scopes the inliner creates are labelled with the function it inlined
    md_kinds = (MD_dbg, MD_alias_scope, MD_noalias, MD_tbaa, MD_tbaa_struct, MD_loop)

    # Symbols first, so that a subprogram can adopt its function's final name...
    for f in mod.functions
        isdeclaration(f) || rename!(f)
    end
    # ...and every surviving function adopts before any instruction walk runs, since a walk
    # can reach another function's subprogram through an inlined debug location and would
    # otherwise give it a rank name instead of that function's symbol.
    for f in mod.functions
        isdeclaration(f) && continue
        sp = f.subprogram
        sp === nothing || normalize_metadata!(sp, visited, codegen_name, normalize, f.name)
    end
    for f in mod.functions
        isdeclaration(f) && continue
        for bb in f.blocks
            rename!(bb)
            for inst in bb.instructions
                rename!(inst)
                md = inst.metadata
                for kind in md_kinds
                    haskey(md, kind) &&
                        normalize_metadata!(md[kind], visited, codegen_name, normalize)
                end
            end
        end
    end

    # LLVM's function-local value symbol table preserves insertion history, which is not
    # visible in textual IR but does affect bitcode string-table order. Transformations above
    # can create the same named values through pointer-keyed worklists in different orders.
    # Reinsert every local name in IR order so equivalent modules serialize identically.
    # (by reference, as LLVM.jl's setter would be compiled for every type of instruction)
    for f in mod.functions
        named_values = Pair{LLVM.API.LLVMValueRef,String}[]
        record!(@nospecialize(value)) = let name = LLVM.name(value)
            isempty(name) ||
                push!(named_values, Base.unsafe_convert(LLVM.API.LLVMValueRef, value) => name)
        end
        for value in f.parameters
            record!(value)
        end
        for bb in f.blocks
            record!(bb)
            for inst in bb.instructions
                record!(inst)
            end
        end
        for (ref, _) in named_values
            LLVM.API.LLVMSetValueName(ref, "")
        end
        for (ref, name) in named_values
            LLVM.API.LLVMSetValueName(ref, name)
        end
    end
    return
end
