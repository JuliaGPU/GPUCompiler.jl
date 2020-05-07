# native target for CPU execution

## target

export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUFeatures())
end

llvm_triple(::NativeCompilerTarget) = Sys.MACHINE

function llvm_machine(target::NativeCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple)

    tm = TargetMachine(t, triple, target.cpu, target.features)
    asm_verbosity!(tm, true)

    return tm
end


## job

runtime_slug(job::CompilerJob{NativeCompilerTarget}) = "native_$(job.target.cpu)-$(hash(job.target.features))"
