is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0
struct NativeCompilerParams <: AbstractCompilerParams end

"""
Copied from the testing env, returns the unique job for this function

TODO: is this right? What are params supposed to be
"""
function native_job(@nospecialize(f_type), @nospecialize(types); kernel::Bool=false, entry_abi=:specfunc, kwargs...)
    source = FunctionSpec(f_type, Base.to_tuple_type(types), kernel)
    target = NativeCompilerTarget(always_inline=true)
    params = NativeCompilerParams()
    CompilerJob(target, source, params, entry_abi), kwargs
end

"""
Given a function and param types caches the function to the global cache
"""
function precompile_native(F, tt)
    job, _ = native_job(F, tt)
    method_instance, _ = GPUCompiler.emit_julia(job)
    # populate the cache
    cache = GPUCompiler.ci_cache(job)
    mt = GPUCompiler.method_table(job)
    interp = GPUCompiler.get_interpreter(job)
    if GPUCompiler.ci_cache_lookup(cache, method_instance, job.source.world, typemax(Cint)) === nothing
        GPUCompiler.ci_cache_populate(interp, cache, mt, method_instance, job.source.world, typemax(Cint))
    end
end

"""
Reloads Global Cache from global variable which stores the previous
cached results
"""
function reload_cache()
    if !is_precompiling()
        # MY_CACHE already only has infinite ranges at this point
        merge!(GPUCompiler.GLOBAL_CI_CACHES, MY_CACHE);
    end
end

"""
Takes a snapshot of the current status of the cache

The cache returned is a deep copy with finite world age endings removed
"""
function snapshot()
    cleaned_cache_to_save = IdDict()
    for key in keys(GPUCompiler.GLOBAL_CI_CACHES)
        # Will only keep those elements with infinite ranges
        cleaned_cache_to_save[key] = GPUCompiler.CodeCache(GPUCompiler.GLOBAL_CI_CACHES[key])
    end
    return cleaned_cache_to_save
end

# Do I need to put this in a function?
const MY_CACHE = snapshot()
my_cache = nothing
