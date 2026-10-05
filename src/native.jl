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

# code for the host, which shouldn't see the overlays for devices
method_table_view(@nospecialize(job::CompilerJob{NativeCompilerTarget})) =
    stack_method_tables(job.world, method_tables(job)...)

# LLVM's CPU back-ends only distinguish the system scope from `singlethread`, and X86 even
# treats other scopes like `singlethread`, so use the system scope for them, like Clang does
llvm_syncscope(::NativeCompilerTarget, name::String) =
    name == "singlethread" ? name : "system"

## job

uses_julia_runtime(job::CompilerJob{NativeCompilerTarget}) = job.config.target.jlruntime
can_vectorize(job::CompilerJob{NativeCompilerTarget}) = true
