# reproducible module layout
#
# Consumers content-key on the artifacts we emit: Metal.jl's `MTLBinaryArchive` tier keys
# native code on the exact metallib bytes, and package images persist the compiled results.
# Both are only useful when the same kernel compiles to the same bytes in every session, so
# nothing session-dependent may leak into the emitted code. Most of that is handled where it
# arises (content-derived relocation names in relocation.jl, back-end symbol normalization in
# metal.jl); the two sources below originate in how modules are assembled, and are handled here.


## module layout (Julia < 1.13.0-rc4)

# Julia codegen emits every CodeInstance into a module of its own and merges those into the
# module we receive. Before julia#62782 (1.13.0-rc4) and julia#60031 (1.14, which emits into a
# single module directly), that merge iterates a pointer-keyed map, so the resulting order of
# functions and globals is a different permutation of the same entities in every session.
# Everything downstream that depends on module order (rank-based symbol normalization,
# metadata numbering, `ConstantMerge` survivors) then shifts with it.
#
# The merge also scatters a second kind of session-dependence: a private constant emitted by
# several CodeInstances (`_j_const#3`, `_j_str_…#7`) exists once per per-CI module and is
# auto-suffixed on merge (`_j_const#3.1`), with the bare name going to whichever copy happened
# to be merged first.
#
# Both are repaired here. Codegen allocates the counter in a generated name (`julia_foo_123`,
# `jfptr_foo_124`, `jl_global#2916`, `+Core.Tuple#2912`, `_j_const#3`) from a monotonic
# sequence as it emits, so sorting by that counter recovers emission order — which is
# deterministic, and is exactly the layout Julia ≥ 1.14 hands us. Names without a counter
# (runtime and intrinsic declarations, `jl_nothing`) are unique and sort by name after them.

# The LLVM uniquing suffix a merge appends to a clashing local name.
const UNIQUING_SUFFIX = r"\.[0-9]+$"

# The trailing codegen counter of a generated name, if any.
const CODEGEN_COUNTER = r"[_#]([0-9]+)$"

function layout_key(name::String)
    base = replace(name, UNIQUING_SUFFIX => "")
    m = match(CODEGEN_COUNTER, base)
    counter = m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    return (counter, base, name)
end

# Merge auto-suffixed clones of a private constant back into the copy that kept the bare name.
# A clone and its original come from the same codegen constant, so they are identical by
# construction; the soundness of merging them is checked rather than assumed: both must be
# private, constant, `unnamed_addr` (so their addresses carry no identity), and of the same
# type, alignment and (context-uniqued, hence ref-comparable) initializer.
function merge_constant_clones!(mod::LLVM.Module)
    mod_gvs = globals(mod)
    merged = false
    for gv in collect(mod_gvs)
        name = LLVM.name(gv)
        m = match(UNIQUING_SUFFIX, name)
        m === nothing && continue
        base_name = name[1:prevind(name, m.offset)]
        haskey(mod_gvs, base_name) || continue
        base = mod_gvs[base_name]
        mergeable(g) = linkage(g) == LLVM.API.LLVMPrivateLinkage && isconstant(g) &&
                       unnamed_addr(g) && !isdeclaration(g)
        (mergeable(gv) && mergeable(base)) || continue
        value_type(gv) == value_type(base) || continue
        alignment(gv) == alignment(base) || continue
        initializer(gv).ref == initializer(base).ref || continue
        replace_uses!(gv, base)
        erase!(gv)
        merged = true
    end
    return merged
end

"""
    canonicalize_module_layout!(mod::LLVM.Module)

Put the functions and global variables of a freshly emitted module in emission order, and
merge the auto-suffixed private-constant clones the per-CodeInstance merge left behind. Makes
the layout of a module received from Julia < 1.13.0-rc4 session-independent; a no-op in
effect on later versions, whose codegen already emits in this order.
"""
function canonicalize_module_layout!(mod::LLVM.Module)
    merge_constant_clones!(mod)
    sort!(functions(mod); by = f -> layout_key(LLVM.name(f)))
    sort!(globals(mod); by = gv -> layout_key(LLVM.name(gv)))
    return mod
end


## compile-unit deduplication

# Julia's codegen dedups `DICompileUnit`s within one module, but every module we link in
# (deferred codegen, each runtime-library function) brings its own `distinct` copy, and LLVM's
# linker never merges distinct nodes. The final module then carries dozens of byte-identical
# CUs and each `DISubprogram`'s `unit:` edge lands on whichever copy its producing module
# happened to have — so the metadata graph, and its numbering in the emitted bitcode, depends
# on the order things were linked in. Repoint every reference to a duplicate at the canonical
# (first-listed) copy of its content group and list only those in `llvm.dbg.cu`. Content is
# compared by operand identity: the file, producer and flag operands are uniqued nodes, which
# distinguishes Julia's CUs from any vendor-library CU with different strings.
function dedup_compile_units!(mod::LLVM.Module)
    mds = metadata(mod)
    haskey(mds, "llvm.dbg.cu") || return false
    cus = operands(mds["llvm.dbg.cu"])
    length(cus) <= 1 && return false

    canonical = Dict{Vector{LLVM.API.LLVMMetadataRef},LLVM.Metadata}()
    canonical_cus = LLVM.Metadata[]
    replacement = Dict{LLVM.API.LLVMMetadataRef,LLVM.Metadata}()
    for cu in cus
        cu === nothing && continue
        key = LLVM.API.LLVMMetadataRef[op === nothing ? LLVM.API.LLVMMetadataRef(C_NULL) : op.ref
                                       for op in operands(cu)]
        canon = get!(canonical, key, cu)
        if canon.ref == cu.ref
            push!(canonical_cus, cu)
        else
            replacement[cu.ref] = canon
        end
    end
    isempty(replacement) && return false

    # Metadata forms a graph, so walk everything reachable from the module's values and
    # repoint any edge into a duplicate CU. The same kinds of instruction metadata that
    # `normalize_julia_symbol_names!` walks are the ones that can reach a subprogram.
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

    md_kinds = (LLVM.MD_dbg, LLVM.MD_alias_scope, LLVM.MD_noalias, LLVM.MD_tbaa,
                LLVM.MD_tbaa_struct, LLVM.MD_loop)
    for f in functions(mod)
        isdeclaration(f) && continue
        sp = LLVM.subprogram(f)
        sp === nothing || repoint!(sp)
        for bb in blocks(f), inst in instructions(bb)
            md = metadata(inst)
            for kind in md_kinds
                haskey(md, kind) && repoint!(md[kind])
            end
        end
    end
    for gv in globals(mod)
        for (kind, md) in metadata(gv)
            repoint!(md)
        end
    end

    # The duplicates are now unreferenced; list only the canonical set. (The verifier
    # requires every reachable CU to be listed, which the walk above guarantees.)
    nmd = mds["llvm.dbg.cu"]
    empty!(nmd)
    for cu in canonical_cus
        push!(nmd, cu)
    end
    return true
end
