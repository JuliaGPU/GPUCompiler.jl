is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0
struct NativeCompilerParams <: AbstractCompilerParams end

function native_job(@nospecialize(f_type), @nospecialize(types); kernel::Bool=false, entry_abi=:specfunc, kwargs...)
    source = FunctionSpec(f_type, Base.to_tuple_type(types), kernel)
    target = NativeCompilerTarget(always_inline=true)
    params = NativeCompilerParams()
    CompilerJob(target, source, params, entry_abi), kwargs
end

g(x) = 12

function precompile_native(F, tt)
    job, _ = native_job(F, tt)
    @show job.source.world
    @show Base.get_world_counter()
    method_instance, _ = GPUCompiler.emit_julia(job)
    @show hash(method_instance)
    precompile(g, (Int, ))
    @show Base.get_world_counter()
    # populate the cache
    cache = GPUCompiler.ci_cache(job)
    mt = GPUCompiler.method_table(job)
    interp = GPUCompiler.get_interpreter(job)
    if GPUCompiler.ci_cache_lookup(cache, method_instance, job.source.world, typemax(Cint)) === nothing
        GPUCompiler.ci_cache_populate(interp, cache, mt, method_instance, job.source.world, typemax(Cint))
    end
end

function reload_cache()
    if !is_precompiling()
        merge!(GPUCompiler.GLOBAL_CI_CACHES, MY_CACHE);
        # have to go through and kill anthing that has a finite range
    end
end

f(x) = 2
precompile_native(f, (Int, ))
f(x) = 4
precompile_native(f, (Int, ))

function snapshot()
    new_cache = IdDict()
    for key in keys(GPUCompiler.GLOBAL_CI_CACHES)
        new_cache[key] = GPUCompiler.CodeCache(GPUCompiler.GLOBAL_CI_CACHES[key])
    end
    return new_cache
end

const MY_CACHE = snapshot()
my_cache = nothing
