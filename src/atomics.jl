# target-independent helpers for lowering LLVM atomics
#
# Back-ends without an LLVM target (Metal) or with one that mishandles some atomics (SPIR-V)
# need to inspect and rewrite LLVM's atomic instructions themselves; these helpers mirror the
# parts of LLVM's `AtomicExpand` and the atomic instruction classes that LLVM.jl lacks.

const AtomicOrdering = LLVM.API.LLVMAtomicOrdering

is_ordered(order::AtomicOrdering) =
    order ∉ (LLVM.API.LLVMAtomicOrderingNotAtomic, LLVM.API.LLVMAtomicOrderingUnordered,
             LLVM.API.LLVMAtomicOrderingMonotonic)
is_acquire(order::AtomicOrdering) =
    order in (LLVM.API.LLVMAtomicOrderingAcquire, LLVM.API.LLVMAtomicOrderingAcquireRelease,
              LLVM.API.LLVMAtomicOrderingSequentiallyConsistent)
is_release(order::AtomicOrdering) =
    order in (LLVM.API.LLVMAtomicOrderingRelease, LLVM.API.LLVMAtomicOrderingAcquireRelease,
              LLVM.API.LLVMAtomicOrderingSequentiallyConsistent)

is_atomic_memop(inst::LLVM.Instruction) =
    ((inst isa LLVM.LoadInst || inst isa LLVM.StoreInst) && is_atomic(inst)) ||
    inst isa LLVM.AtomicRMWInst || inst isa LLVM.AtomicCmpXchgInst

atomic_pointer(inst::LLVM.StoreInst) = operands(inst)[2]
atomic_pointer(inst::LLVM.Instruction) = operands(inst)[1]

atomic_value_type(inst::LLVM.LoadInst) = value_type(inst)
atomic_value_type(inst::LLVM.StoreInst) = value_type(operands(inst)[1])
atomic_value_type(inst::LLVM.Instruction) = value_type(operands(inst)[2])

is_volatile(inst::LLVM.Instruction) = LLVM.API.LLVMGetVolatile(inst) != 0

# LLVM.jl cannot name a synchronization scope, so read it from the textual form
function syncscope_name(inst::LLVM.Instruction)
    m = match(r"syncscope\(\"((?:[^\"\\]|\\.)*)\"\)", string(inst))
    return m === nothing ? "system" : repr(m.captures[1])
end

# the ordering that covers both of a compare-exchange's orderings
# (`AtomicCmpXchgInst::getMergedOrdering`)
function merged_ordering(inst::LLVM.AtomicCmpXchgInst)
    success, failure = success_ordering(inst), failure_ordering(inst)
    failure == LLVM.API.LLVMAtomicOrderingSequentiallyConsistent && return failure
    if failure == LLVM.API.LLVMAtomicOrderingAcquire
        success == LLVM.API.LLVMAtomicOrderingMonotonic && return failure
        success == LLVM.API.LLVMAtomicOrderingRelease &&
            return LLVM.API.LLVMAtomicOrderingAcquireRelease
    end
    return success
end
atomic_ordering(inst::LLVM.AtomicCmpXchgInst) = merged_ordering(inst)
atomic_ordering(inst::LLVM.Instruction) = ordering(inst)

# the failure ordering of a compare-exchange implementing an operation with the given
# ordering (`AtomicCmpXchgInst::getStrongestFailureOrdering`)
failure_ordering_for(order::AtomicOrdering) =
    order == LLVM.API.LLVMAtomicOrderingAcquireRelease ? LLVM.API.LLVMAtomicOrderingAcquire :
    order == LLVM.API.LLVMAtomicOrderingRelease ? LLVM.API.LLVMAtomicOrderingMonotonic :
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
    T = value_type(old)
    minmax(pred) = select!(builder, icmp!(builder, pred, old, val), old, val)
    function intrinsic(name)
        mod = LLVM.parent(LLVM.parent(position(builder)))
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
    op == :max && return minmax(LLVM.API.LLVMIntSGT)
    op == :min && return minmax(LLVM.API.LLVMIntSLE)
    op == :umax && return minmax(LLVM.API.LLVMIntUGT)
    op == :umin && return minmax(LLVM.API.LLVMIntULE)
    op == :fadd && return fadd!(builder, old, val)
    op == :fsub && return fsub!(builder, old, val)
    op == :fmax && return intrinsic("llvm.maxnum")
    op == :fmin && return intrinsic("llvm.minnum")
    op == :fmaximum && return intrinsic("llvm.maximum")
    op == :fminimum && return intrinsic("llvm.minimum")
    zero, one = ConstantInt(T, 0), ConstantInt(T, 1)
    if op == :uinc_wrap
        return select!(builder, icmp!(builder, LLVM.API.LLVMIntUGE, old, val), zero,
                       add!(builder, old, one))
    elseif op == :udec_wrap
        wrap = or!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, old, zero),
                   icmp!(builder, LLVM.API.LLVMIntUGT, old, val))
        return select!(builder, wrap, val, sub!(builder, old, one))
    elseif op == :usub_cond
        return select!(builder, icmp!(builder, LLVM.API.LLVMIntUGE, old, val),
                       sub!(builder, old, val), old)
    elseif op == :usub_sat
        return select!(builder, icmp!(builder, LLVM.API.LLVMIntUGE, old, val),
                       sub!(builder, old, val), zero)
    end
    error("Unsupported atomicrmw operation: $op")
end
