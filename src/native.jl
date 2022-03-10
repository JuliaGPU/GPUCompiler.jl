# native target for CPU execution

## target

export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUFeatures())
    always_inline::Bool=false # will mark the job function as always inline
    jlruntime::Bool=true # Use Julia runtime for throwing errors, instead of the GPUCompiler support
end

llvm_triple(::NativeCompilerTarget) = Sys.MACHINE

function llvm_machine(target::NativeCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple=triple)

    tm = TargetMachine(t, triple, target.cpu, target.features)
    asm_verbosity!(tm, true)

    return tm
end

function process_entry!(job::CompilerJob{NativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)
    if job.target.always_inline
        push!(function_attributes(entry), EnumAttribute("alwaysinline", 0; ctx))
    end
    invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)
end

## job

runtime_slug(job::CompilerJob{NativeCompilerTarget}) = "native_$(job.target.cpu)-$(hash(job.target.features))$(job.target.jlruntime ? "-jlrt" : "")"
uses_julia_runtime(job::CompilerJob{NativeCompilerTarget}) = job.target.jlruntime
