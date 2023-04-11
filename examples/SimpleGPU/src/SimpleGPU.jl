module SimpleGPU
using GPUCompiler
struct NativeCompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table

    NativeCompilerParams(entry_safepoint::Bool=false, method_table=test_method_table) =
        new(entry_safepoint, method_table)
end

const test_method_table = nothing

function native_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                    entry_abi=:specfunc, entry_safepoint::Bool=false, always_inline=false,
                    method_table=test_method_table, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types))
    target = NativeCompilerTarget()
    params = NativeCompilerParams(entry_safepoint, method_table)
    config = CompilerConfig(target, params; kernel, entry_abi, always_inline)
    CompilerJob(source, config), kwargs
end

function precompile_simple(f, t)
    job, _ = native_job(f, t)
    GPUCompiler.precompile_gpucompiler(job)
end

end # module SimpleGPU
