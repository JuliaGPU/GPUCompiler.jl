export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String
    features::String=""
end

llvm_triple(::NativeCompilerTarget) = triple()

llvm_datalayout(::NativeCompilerTarget) =  nothing

function llvm_machine(target::NativeCompilerTarget)
    t = Target(llvm_triple(target))
    @show t 
    tm = TargetMachine(t, llvm_triple(target), target.cpu, target.features)
    asm_verbosity!(tm, true)

    return tm
end

module NativeRuntime
    # the GPU runtime library
    signal_exception() = return
    malloc(sz) =  return
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
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

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::NativeCompilerJob) = "native_$(target(job).cpu)$(target(job).features)"