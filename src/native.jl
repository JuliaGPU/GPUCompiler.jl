# native target for CPU execution

## target

export NativeCompilerTarget

Base.@kwdef struct NativeCompilerTarget <: AbstractCompilerTarget
    cpu::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUName())
    features::String=(LLVM.version() < v"8") ? "" : unsafe_string(LLVM.API.LLVMGetHostCPUFeatures())
    llvm_always_inline::Bool=false # will mark the job function as always inline
    jlruntime::Bool=false # Use Julia runtime for throwing errors, instead of the GPUCompiler support
end
llvm_triple(::NativeCompilerTarget) = Sys.MACHINE

function llvm_machine(target::NativeCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple=triple)

    tm = TargetMachine(t, triple, target.cpu, target.features)
    asm_verbosity!(tm, true)

    return tm
end

function finish_module!(job::CompilerJob{NativeCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)

    if job.config.target.llvm_always_inline
        push!(function_attributes(entry), EnumAttribute("alwaysinline", 0; ctx))
    end

    return entry
end

## job

runtime_slug(job::CompilerJob{NativeCompilerTarget}) = "native_$(job.config.target.cpu)-$(hash(job.config.target.features))$(job.config.target.jlruntime ? "-jlrt" : "")"
uses_julia_runtime(job::CompilerJob{NativeCompilerTarget}) = job.config.target.jlruntime

function llvm_debug_info(@nospecialize(job::CompilerJob{NativeCompilerTarget}))
    if uses_julia_runtime(job)
        Base.default_debug_info_kind()
    else
        invoke(llvm_debug_info, Tuple{CompilerJob}, job)
    end
end

llvm_gnu_pubnames(@nospecialize(job::CompilerJob{NativeCompilerTarget})) = true
