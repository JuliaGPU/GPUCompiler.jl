# native target for CPU execution

## target

export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String=LLVM.host_cpu_name()
    features::String=LLVM.host_cpu_features()
    llvm_always_inline::Bool=false # will mark the job function as always inline
    jlruntime::Bool=false # Use Julia runtime for throwing errors, instead of the GPUCompiler support
end
llvm_triple(::NativeCompilerTarget) = Sys.MACHINE

function llvm_machine(target::NativeCompilerTarget)
    triple = llvm_triple(target)

    t = LLVM.Target(triple=triple)

    tm = LLVM.TargetMachine(t, triple; target.cpu, target.features)
    LLVM.asm_verbosity!(tm, true)

    return tm
end

function finish_module!(job::CompilerJob{NativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    if job.config.target.llvm_always_inline
        push!(entry.function_attributes, EnumAttribute(:alwaysinline))
    end

    return entry
end

## job

uses_julia_runtime(job::CompilerJob{NativeCompilerTarget}) = job.config.target.jlruntime
can_vectorize(job::CompilerJob{NativeCompilerTarget}) = true
