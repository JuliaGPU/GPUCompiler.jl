@testsetup module Native

using GPUCompiler


# create a native test compiler, and generate reflection methods for it

include("runtime.jl")

# local method table for device functions
Base.Experimental.@MethodTable(test_method_table)

struct CompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    use_jlplt::Bool
    method_table

    CompilerParams(entry_safepoint::Bool=false, use_jlplt::Bool=true, method_table=test_method_table) =
        new(entry_safepoint, use_jlplt, method_table)
end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}
GPUCompiler.runtime_module(::NativeCompilerJob) = TestRuntime

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint
@static if VERSION > v"1.11.0-DEV.398"
    GPUCompiler.codegen_params(@nospecialize(job::NativeCompilerJob)) = (;use_jlplt=job.config.params.use_jlplt)
end

function create_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                    entry_abi=:specfunc, entry_safepoint::Bool=false, use_jlplt::Bool=true, always_inline=false,
                    method_table=test_method_table, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = NativeCompilerTarget()
    params = CompilerParams(entry_safepoint, use_jlplt, method_table)
    config = CompilerConfig(target, params; kernel, entry_abi, always_inline)
    CompilerJob(source, config), kwargs
end

function code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    method = Symbol("$(method)")
    @eval begin
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

# simulates codegen for a kernel function: validates by default
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end

end
