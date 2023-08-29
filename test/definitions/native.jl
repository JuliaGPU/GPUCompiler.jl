using GPUCompiler

if !@isdefined(TestRuntime)
    include("../testhelpers.jl")
end


# create a native test compiler, and generate reflection methods for it

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(test_method_table)
else
const test_method_table = nothing
end

struct NativeCompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table

    NativeCompilerParams(entry_safepoint::Bool=false, method_table=test_method_table) =
        new(entry_safepoint, method_table)
end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,NativeCompilerParams}

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint
GPUCompiler.runtime_module(::NativeCompilerJob) = TestRuntime

function native_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                    entry_abi=:specfunc, entry_safepoint::Bool=false, always_inline=false,
                    method_table=test_method_table, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = NativeCompilerTarget()
    params = NativeCompilerParams(entry_safepoint, method_table)
    config = CompilerConfig(target, params; kernel, entry_abi, always_inline)
    CompilerJob(source, config), kwargs
end

function native_code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function native_code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function native_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function native_code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    native_method = Symbol("native_$(method)")
    @eval begin
        $native_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $native_method(stdout, func, types; kwargs...)
    end
end

# simulates codegen for a kernel function: validates by default
function native_code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end
