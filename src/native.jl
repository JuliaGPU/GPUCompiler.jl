# native target for CPU execution

## target

export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String=unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=""
end

llvm_triple(::NativeCompilerTarget) = Sys.MACHINE

function llvm_machine(target::NativeCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple)

    tm = TargetMachine(t, triple, target.cpu, target.features)
    asm_verbosity!(tm, true)

    return tm
end

module NativeRuntime
    malloc(sz) =  return
    report_oom(sz) = return
end

runtime_module(target::NativeCompilerTarget) = NativeRuntime


## job

export NativeCompilerJob

Base.@kwdef struct NativeCompilerJob <: AbstractCompilerJob
    target::NativeCompilerTarget
    source::FunctionSpec
end

target(job::NativeCompilerJob) = job.target
source(job::NativeCompilerJob) = job.source

Base.similar(job::NativeCompilerJob, source::FunctionSpec) =
    NativeCompilerJob(target=job.target, source=source)

function Base.show(io::IO, job::NativeCompilerJob)
    print(io, "Native CompilerJob of ", source(job))
    print(io, " for $(target(job).cpu) $(target(job).features)")
end

runtime_slug(job::NativeCompilerJob) = "native_$(target(job).cpu)$(target(job).features)"

add_lowering_passes!(::NativeCompilerJob, pm::LLVM.PassManager) = return
