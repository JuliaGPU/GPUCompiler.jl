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
        gv.initializer.ref == base.initializer.ref || continue
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

# LLVM does not merge distinct `DICompileUnit`s when linking. Group compile units by their
# uniqued operands, repoint references to the first unit in each group, and shrink
# `llvm.dbg.cu` to those canonical units.
function dedup_compile_units!(mod::LLVM.Module)
    mds = mod.metadata
    haskey(mds, "llvm.dbg.cu") || return false
    cus = mds["llvm.dbg.cu"].operands
    length(cus) <= 1 && return false

    canonical = Dict{Tuple,LLVM.Metadata}()
    canonical_cus = LLVM.Metadata[]
    replacement = Dict{LLVM.API.LLVMMetadataRef,LLVM.Metadata}()
    for cu in cus
        cu === nothing && continue
        key = Tuple(op === nothing ? LLVM.API.LLVMMetadataRef(C_NULL) : op.ref
                    for op in cu.operands)
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
        # iterate a copy, as replacing an operand re-uniques the node
        for (i, op) in enumerate(collect(md.operands))
            op isa LLVM.MDNode || continue
            repl = get(replacement, op.ref, nothing)
            if repl !== nothing
                md.operands[i] = repl
            else
                repoint!(op)
            end
        end
    end

    function repoint_instruction!(inst)
        # LLVM.jl cannot iterate InstructionMetadataDict, so enumerate non-debug
        # attachments through LLVM's C API.
        md = inst.metadata
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

    for f in mod.functions
        sp = f.subprogram
        sp === nothing || repoint!(sp)
        for bb in f.blocks, inst in bb.instructions
            repoint_instruction!(inst)
        end
    end
    for gv in mod.globals
        for (kind, md) in gv.metadata
            repoint!(md)
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
