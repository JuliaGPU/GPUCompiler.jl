# target-independent helpers for lowering LLVM atomics
#
# Back-ends without an LLVM target (Metal) or with one that mishandles some atomics (SPIR-V)
# need to inspect and rewrite LLVM's atomic instructions themselves; these helpers mirror the
# parts of LLVM's `AtomicExpand` and the atomic instruction classes that LLVM.jl lacks.

const AtomicOrdering = LLVM.AtomicOrdering.T

is_ordered(order::AtomicOrdering) =
    order ∉ (LLVM.AtomicOrdering.NotAtomic, LLVM.AtomicOrdering.Unordered,
             LLVM.AtomicOrdering.Monotonic)
is_acquire(order::AtomicOrdering) =
    order in (LLVM.AtomicOrdering.Acquire, LLVM.AtomicOrdering.AcquireRelease,
              LLVM.AtomicOrdering.SequentiallyConsistent)
is_release(order::AtomicOrdering) =
    order in (LLVM.AtomicOrdering.Release, LLVM.AtomicOrdering.AcquireRelease,
              LLVM.AtomicOrdering.SequentiallyConsistent)

is_atomic_memop(inst::LLVM.Instruction) =
    ((inst isa LLVM.LoadInst || inst isa LLVM.StoreInst) && isatomic(inst)) ||
    inst isa LLVM.AtomicRMWInst || inst isa LLVM.AtomicCmpXchgInst

atomic_pointer(inst::LLVM.StoreInst) = inst.operands[2]
atomic_pointer(inst::LLVM.Instruction) = inst.operands[1]

atomic_value_type(inst::LLVM.LoadInst) = inst.value_type
atomic_value_type(inst::LLVM.StoreInst) = inst.operands[1].value_type
atomic_value_type(inst::LLVM.Instruction) = inst.operands[2].value_type

is_volatile(inst::LLVM.Instruction) = LLVM.API.LLVMGetVolatile(inst) != 0

# LLVM.jl cannot name a synchronization scope, so read it from the textual form
function syncscope_name(inst::LLVM.Instruction)
    m = match(r"syncscope\(\"((?:[^\"\\]|\\.)*)\"\)", string(inst))
    return m === nothing ? "system" : repr(m.captures[1])
end

# the ordering that covers both of a compare-exchange's orderings
# (`AtomicCmpXchgInst::getMergedOrdering`)
function merged_ordering(inst::LLVM.AtomicCmpXchgInst)
    success, failure = inst.success_ordering, inst.failure_ordering
    failure == LLVM.AtomicOrdering.SequentiallyConsistent && return failure
    if failure == LLVM.AtomicOrdering.Acquire
        success == LLVM.AtomicOrdering.Monotonic && return failure
        success == LLVM.AtomicOrdering.Release &&
            return LLVM.AtomicOrdering.AcquireRelease
    end
    return success
end
atomic_ordering(inst::LLVM.AtomicCmpXchgInst) = merged_ordering(inst)
atomic_ordering(inst::LLVM.Instruction) = inst.ordering

# the failure ordering of a compare-exchange implementing an operation with the given
# ordering (`AtomicCmpXchgInst::getStrongestFailureOrdering`)
failure_ordering_for(order::AtomicOrdering) =
    order == LLVM.AtomicOrdering.AcquireRelease ? LLVM.AtomicOrdering.Acquire :
    order == LLVM.AtomicOrdering.Release ? LLVM.AtomicOrdering.Monotonic :
    order

# The operation of an `atomicrmw`. Parse it from the textual form, as the C API only knows the
# operations of the LLVM version its headers came from (LLVM 18 aborts on `uinc_wrap`).
function atomicrmw_op(inst::LLVM.AtomicRMWInst)
    m = match(r"\batomicrmw\s+(?:volatile\s+)?([a-z_]+)\s", string(inst))
    m === nothing && error("Unexpected atomicrmw instruction: $inst")
    return Symbol(m.captures[1])
end

# the value of a read-modify-write operation (`llvm::buildAtomicRMWValue`)
function atomicrmw_value!(builder::IRBuilder, op::Symbol, old::LLVM.Value, val::LLVM.Value)
    T = old.value_type
    minmax(pred) = select!(builder, icmp!(builder, pred, old, val), old, val)
    function intrinsic(name)
        mod = builder.insert_block.parent.parent
        intr = LLVM.Intrinsic(name)
        call!(builder, LLVM.FunctionType(intr, [T]), LLVM.Function(mod, intr, [T]), [old, val])
    end
    op == :xchg && return val
    op == :add && return add!(builder, old, val)
    op == :sub && return sub!(builder, old, val)
    op == :and && return and!(builder, old, val)
    op == :nand && return not!(builder, and!(builder, old, val))
    op == :or && return or!(builder, old, val)
    op == :xor && return xor!(builder, old, val)
    op == :max && return minmax(LLVM.IntPredicate.SGT)
    op == :min && return minmax(LLVM.IntPredicate.SLE)
    op == :umax && return minmax(LLVM.IntPredicate.UGT)
    op == :umin && return minmax(LLVM.IntPredicate.ULE)
    op == :fadd && return fadd!(builder, old, val)
    op == :fsub && return fsub!(builder, old, val)
    op == :fmax && return intrinsic("llvm.maxnum")
    op == :fmin && return intrinsic("llvm.minnum")
    op == :fmaximum && return intrinsic("llvm.maximum")
    op == :fminimum && return intrinsic("llvm.minimum")
    zero, one = ConstantInt(T, 0), ConstantInt(T, 1)
    if op == :uinc_wrap
        return select!(builder, icmp!(builder, LLVM.IntPredicate.UGE, old, val), zero,
                       add!(builder, old, one))
    elseif op == :udec_wrap
        wrap = or!(builder, icmp!(builder, LLVM.IntPredicate.EQ, old, zero),
                   icmp!(builder, LLVM.IntPredicate.UGT, old, val))
        return select!(builder, wrap, val, sub!(builder, old, one))
    elseif op == :usub_cond
        return select!(builder, icmp!(builder, LLVM.IntPredicate.UGE, old, val),
                       sub!(builder, old, val), old)
    elseif op == :usub_sat
        return select!(builder, icmp!(builder, LLVM.IntPredicate.UGE, old, val),
                       sub!(builder, old, val), zero)
    end
    error("Unsupported atomicrmw operation: $op")
end
