# target-independent helpers for lowering LLVM atomics
#
# Back-ends without an LLVM target (Metal) or with one that mishandles some atomics (SPIR-V)
# need to inspect and rewrite LLVM's atomic instructions themselves; these helpers complement
# what LLVM.jl offers for that.

# does the ordering order other memory accesses, i.e., is it stronger than monotonic?
is_ordered(order::LLVM.AtomicOrdering.T) = is_stronger(order, LLVM.AtomicOrdering.Monotonic)

# (`atomicrmw` and `cmpxchg` are always atomic)
is_atomic_memop(inst::LLVM.Instruction) = inst isa LLVM.MemAccessInst && isatomic(inst)

# the type of the value an atomic memory operation accesses
atomic_value_type(inst::LLVM.LoadInst) = inst.value_type
atomic_value_type(inst::Union{LLVM.StoreInst,LLVM.AtomicRMWInst}) = inst.value_operand.value_type
atomic_value_type(inst::LLVM.AtomicCmpXchgInst) = inst.compare_operand.value_type

# LLVM's orderings order all memory, but front-ends can restrict an ordered atomic operation or
# fence to some memory, as MSL and OpenCL can, in the name of its synchronization scope (like
# AMDGPU's `agent-one-as`): `<scope>-mem-<memory>`, where `<memory>` is `none`, or `global`,
# `local`, `image` and `imageblock` joined by `+` in that order, e.g. `device-mem-global` or
# `workgroup-mem-local+image`. Unlike LLVM metadata, which optimizations can drop, the scope is
# preserved, so the memory can also include memory the target doesn't order by default.
const SYNCSCOPE_MEMORY = ("global", "local", "image", "imageblock")

# Split the name of a synchronization scope into the scope and the memory it orders, as a bit
# mask of `SYNCSCOPE_MEMORY` (`nothing` for all memory). Returns `nothing` for malformed names.
function split_syncscope(name::String)
    i = findfirst("-mem-", name)
    i === nothing && return (name, nothing)
    scope, memory = name[1:first(i)-1], name[last(i)+1:end]
    memory == "none" && return (scope, 0)
    mask, prev = 0, 0
    for class in split(memory, '+')
        j = findfirst(==(class), SYNCSCOPE_MEMORY)
        # (unknown, repeated or out of order)
        (j === nothing || j <= prev) && return nothing
        mask |= 1 << (j - 1)
        prev = j
    end
    return (scope, mask)
end

# rename the synchronization scopes of atomic instructions to the target's (see
# `llvm_syncscope`), dropping the memory they order on targets that order all memory (see
# `syncscope_memory`)
function lower_syncscopes!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    target = job.config.target
    changed = false
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        inst isa LLVM.FenceInst || is_atomic_memop(inst) || continue
        scope = inst.syncscope
        name = scope.name
        if !syncscope_memory(target)
            # (malformed names are passed on, for the back-end to reject)
            parts = split_syncscope(name)
            parts === nothing || (name = first(parts))
        end
        # (by ID, as a scope named "system" is called like the default one)
        new_scope = SyncScope(llvm_syncscope(target, name); context=context(inst))
        new_scope.id == scope.id && continue
        inst.syncscope = new_scope
        changed = true
    end
    return changed
end

# the ordering of an atomic memory operation, merging a compare-exchange's orderings
atomic_ordering(inst::LLVM.AtomicCmpXchgInst) = merged_ordering(inst)
atomic_ordering(inst::LLVM.Instruction) = inst.ordering
