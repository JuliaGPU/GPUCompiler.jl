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

# the size in bits of the value an atomic memory operation accesses, with pointers as large as
# the module's data layout says, or `nothing` if it isn't a scalar
function atomic_bits(inst::LLVM.Instruction)
    T = atomic_value_type(inst)
    T isa Union{LLVM.IntegerType,LLVM.FloatingPointType,LLVM.PointerType} || return nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    return Int(LLVM.bit_size(mod.datalayout, T))
end

# Does `ptr` point to the thread's own stack, i.e., is every object it can be derived from an
# `alloca`? That is the case for atomics on objects that Julia's `AllocOpt` moved to the stack,
# e.g., a non-escaping mutable struct with `@atomic` fields (GPUCompiler.jl#934). Metal and
# SPIR-V cannot express those (they only have atomics on device/global and threadgroup/local
# memory), but they don't need to be atomic: no other thread can access a thread's stack, even
# when it has a pointer to it (thread memory is private to every thread), so plain accesses
# behave the same. Anything this cannot trace back to an `alloca`, e.g., a function argument
# or a loaded pointer, is not known to be private.
function is_thread_private(ptr::LLVM.Value)
    seen = Set{LLVM.Value}()
    worklist = LLVM.Value[ptr]
    while !isempty(worklist)
        val = pop!(worklist)
        val in seen && continue
        push!(seen, val)
        if val isa LLVM.AllocaInst
            continue
        elseif val isa LLVM.GetElementPtrInst || val isa LLVM.BitCastInst ||
               val isa LLVM.AddrSpaceCastInst
            # (a cast to a generic pointer still points to the stack)
            push!(worklist, val.operands[1])
        elseif val isa LLVM.PHIInst
            append!(worklist, first.(val.incoming))
        elseif val isa LLVM.SelectInst
            push!(worklist, val.operands[2], val.operands[3])
        else
            return false
        end
    end
    return true
end

# Replace an atomic operation on the thread's own memory (see `is_thread_private`) by plain
# accesses. Its ordering and scope don't matter either: no other thread can observe the memory
# it accesses, so it cannot synchronize with any.
function demote_private_atomic!(inst::LLVM.Instruction)
    if inst isa LLVM.LoadInst || inst isa LLVM.StoreInst
        # (a plain access has the default, system scope)
        inst.syncscope = SyncScope("system")
        inst.ordering = LLVM.AtomicOrdering.NotAtomic
    else
        lower_atomic!(inst)
    end
    return
end

# Replace a `cmpxchg` of pointers with one of integers of the same size, like `AtomicExpand`'s
# `convertCmpXchgToIntegerType`, for targets that only support integer atomics (complementing
# LLVM.jl's `cast_atomic_to_integer!`). Returns the new instruction.
function cast_cmpxchg_to_integer!(inst::LLVM.AtomicCmpXchgInst)
    T = atomic_value_type(inst)
    T_int = LLVM.IntType(atomic_bits(inst))
    new = @dispose builder=IRBuilder() begin
        position!(builder, LLVM.before(inst))
        builder.debug_location = inst.debug_location
        cmp = ptrtoint!(builder, inst.compare_operand, T_int)
        desired = ptrtoint!(builder, inst.new_value_operand, T_int)
        new = atomic_cmpxchg!(builder, inst.pointer_operand, cmp, desired,
                              inst.success_ordering, inst.failure_ordering;
                              scope=inst.syncscope, align=inst.alignment,
                              volatile=inst.volatile, weak=inst.weak)
        copy_atomic_metadata!(new, inst)
        old = inttoptr!(builder, extract_value!(builder, new, 0), T)
        res = insert_value!(builder, UndefValue(inst.value_type), old, 0)
        res = insert_value!(builder, res, extract_value!(builder, new, 1), 1)
        replace_uses!(inst, res)
        new
    end
    erase!(inst)
    return new
end

# Replace an atomic operation on a value smaller than `word_size` bytes by masked operations
# on the word containing it, like `AtomicExpand` does for targets without atomics of that
# size. Like `AtomicExpand`, this assumes that the whole word can be accessed, which the
# target has to ensure.
function expand_partword_atomic!(inst::LLVM.Instruction, word_size::Integer)
    if inst isa LLVM.LoadInst
        # a word-sized load, shifted
        @dispose builder=IRBuilder() begin
            position!(builder, LLVM.before(inst))
            mask = partword_mask!(builder, inst.value_type, inst.pointer_operand;
                                  align=inst.alignment, word_size)
            word = load!(builder, mask.word_type, mask.aligned_addr)
            word.alignment = mask.aligned_addr_alignment
            word.ordering = inst.ordering
            word.syncscope = inst.syncscope
            word.volatile = inst.volatile
            copy_atomic_metadata!(word, inst)
            replace_uses!(inst, extract_masked_value!(builder, word, mask))
        end
        erase!(inst)
        return
    elseif inst isa LLVM.StoreInst
        # an exchange whose result is unused (`AtomicExpand`'s `expandAtomicStoreToXChg`),
        # which needs at least a monotonic ordering
        order = inst.ordering == LLVM.AtomicOrdering.Unordered ?
                LLVM.AtomicOrdering.Monotonic : inst.ordering
        rmw = @dispose builder=IRBuilder() begin
            position!(builder, LLVM.before(inst))
            atomic_rmw!(builder, LLVM.AtomicRMWBinOp.Xchg, inst.pointer_operand,
                        inst.value_operand, order, inst.syncscope)
        end
        rmw.alignment = inst.alignment
        rmw.volatile = inst.volatile
        copy_atomic_metadata!(rmw, inst)
        erase!(inst)
        inst = rmw
    end
    expand_partword!(inst, word_size)
    return
end
