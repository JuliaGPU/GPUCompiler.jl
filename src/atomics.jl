# target-independent helpers for lowering LLVM atomics
#
# Back-ends without an LLVM target (Metal) or with one that mishandles some atomics (SPIR-V)
# need to inspect and rewrite LLVM's atomic instructions themselves; these helpers mirror the
# parts of LLVM's `AtomicExpand` and the atomic instruction classes that LLVM.jl lacks.

const AtomicOrdering = LLVM.API.LLVMAtomicOrdering

# LLVM.jl cannot name a synchronization scope, so read it from the textual form
function syncscope_name(inst::LLVM.Instruction)
    m = match(r"syncscope\(\"((?:[^\"\\]|\\.)*)\"\)", string(inst))
    return m === nothing ? "system" : repr(m.captures[1])
end
