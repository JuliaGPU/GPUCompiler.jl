# Reproducibility fixes for LLVM modules assembled from multiple codegen units.


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
    mod_gvs = globals(mod)
    merged = false
    for gv in collect(mod_gvs)
        name = LLVM.name(gv)
        m = match(CODEGEN_UNIQUING_SUFFIX, name)
        m === nothing && continue
        base_name = name[1:prevind(name, m.offset)]
        haskey(mod_gvs, base_name) || continue
        base = mod_gvs[base_name]
        mergeable(g) = linkage(g) == LLVM.API.LLVMPrivateLinkage && isconstant(g) &&
                       unnamed_addr(g) && !isdeclaration(g)
        (mergeable(gv) && mergeable(base)) || continue
        value_type(gv) == value_type(base) || continue
        global_value_type(gv) == global_value_type(base) || continue
        alignment(gv) == alignment(base) || continue
        section(gv) == section(base) || continue
        visibility(gv) == visibility(base) || continue
        dllstorage(gv) == dllstorage(base) || continue
        threadlocalmode(gv) == threadlocalmode(base) || continue
        isextinit(gv) == isextinit(base) || continue
        initializer(gv).ref == initializer(base).ref || continue
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
    sort!(functions(mod); by = f -> module_layout_key(LLVM.name(f)))
    sort!(globals(mod); by = gv -> module_layout_key(LLVM.name(gv)))
    return mod
end


## compile-unit deduplication

# LLVM does not merge distinct `DICompileUnit`s when linking. Group compile units by their
# uniqued operands, repoint references to the first unit in each group, and shrink
# `llvm.dbg.cu` to those canonical units.
function dedup_compile_units!(mod::LLVM.Module)
    mds = metadata(mod)
    haskey(mds, "llvm.dbg.cu") || return false
    cus = operands(mds["llvm.dbg.cu"])
    length(cus) <= 1 && return false

    canonical = Dict{Tuple,LLVM.Metadata}()
    canonical_cus = LLVM.Metadata[]
    replacement = Dict{LLVM.API.LLVMMetadataRef,LLVM.Metadata}()
    for cu in cus
        cu === nothing && continue
        key = Tuple(op === nothing ? LLVM.API.LLVMMetadataRef(C_NULL) : op.ref
                    for op in operands(cu))
        canon = get!(canonical, key, cu)
        if canon.ref == cu.ref
            push!(canonical_cus, cu)
        else
            replacement[cu.ref] = canon
        end
    end
    isempty(replacement) && return false

    # Metadata forms a graph, so walk every attachment reachable from module values.
    visited = Set{LLVM.API.LLVMMetadataRef}()
    function repoint!(@nospecialize(md))
        md isa LLVM.MDNode || return
        md.ref in visited && return
        push!(visited, md.ref)
        for (i, op) in enumerate(operands(md))
            op isa LLVM.MDNode || continue
            repl = get(replacement, op.ref, nothing)
            if repl !== nothing
                LLVM.replace_operand(md, i, repl)
            else
                repoint!(op)
            end
        end
    end

    function repoint_instruction!(inst)
        # LLVM.jl cannot iterate InstructionMetadataDict, so enumerate non-debug
        # attachments through LLVM's C API.
        md = metadata(inst)
        haskey(md, LLVM.MD_dbg) && repoint!(md[LLVM.MD_dbg])

        num_entries = Ref{Csize_t}()
        entries = LLVM.API.LLVMInstructionGetAllMetadataOtherThanDebugLoc(inst, num_entries)
        try
            for i in 1:num_entries[]
                ref = LLVM.API.LLVMValueMetadataEntriesGetMetadata(entries, i - 1)
                repoint!(LLVM.Metadata(ref))
            end
        finally
            LLVM.API.LLVMDisposeValueMetadataEntries(entries)
        end
    end

    for f in functions(mod)
        sp = LLVM.subprogram(f)
        sp === nothing || repoint!(sp)
        for bb in blocks(f), inst in instructions(bb)
            repoint_instruction!(inst)
        end
    end
    for gv in globals(mod)
        for (kind, md) in metadata(gv)
            repoint!(md)
        end
    end

    # The verifier requires every reachable compile unit to be listed here.
    nmd = mds["llvm.dbg.cu"]
    empty!(nmd)
    for cu in canonical_cus
        push!(nmd, cu)
    end
    return true
end
