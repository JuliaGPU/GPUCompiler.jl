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
atomic_value_type(inst::LLVM.AtomicCmpXchgInst) = inst.operands[2].value_type   # the comparand

# the ordering of an atomic memory operation, merging a compare-exchange's orderings
atomic_ordering(inst::LLVM.AtomicCmpXchgInst) = merged_ordering(inst)
atomic_ordering(inst::LLVM.Instruction) = inst.ordering

# the value of a read-modify-write operation, like LLVM.jl's `atomic_rmw_value!` (LLVM's
# `buildAtomicRMWValue`), except for `usub_sat`, which that computes with an `llvm.usub.sat`
# call that the Metal back-end has no lowering for
function atomicrmw_value!(builder::IRBuilder, op::LLVM.AtomicRMWBinOp.T, old::LLVM.Value,
                          val::LLVM.Value)
    if op == LLVM.AtomicRMWBinOp.USubSat
        return select!(builder, icmp!(builder, LLVM.IntPredicate.UGE, old, val),
                       sub!(builder, old, val), ConstantInt(old.value_type, 0))
    end
    return atomic_rmw_value!(builder, op, old, val)
end
