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
