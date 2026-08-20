# reproducible module layout
#
# Consumers content-key on the artifacts we emit: Metal.jl's `MTLBinaryArchive` tier keys
# native code on the exact metallib bytes, and package images persist the compiled results.
# Both are only useful when the same kernel compiles to the same bytes in every session, so
# nothing session-dependent may leak into the emitted code. Most of that is handled where it
# arises (content-derived relocation names in relocation.jl, back-end symbol normalization in
# metal.jl); the two sources below originate in how modules are assembled, and are handled here.


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
