# cached compilation

const cache_lock = ReentrantLock()

"""
    cached_compilation(cache::Dict{UInt}, cfg::CompilerConfig, ft::Type, tt::Type,
                       compiler, linker)

Compile a method instance, identified by its function type `ft` and argument types `tt`,
using `compiler` and `linker`, and store the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using a `UInt` and store
whatever the `linker` function returns. The `compiler` function should take a `CompilerJob`
and return data that can be cached across sessions (e.g., LLVM IR). This data is then
forwarded, along with the `CompilerJob`, to the `linker` function which is allowed to create
session-dependent objects (e.g., a `CuModule`).
"""
function cached_compilation(cache::AbstractDict{UInt,V},
                            cfg::CompilerConfig,
                            ft::Type, tt::Type,
                            compiler::Function, linker::Function) where {V}
    # fast path: a simple cache that's indexed by the current world age, the function and
    #            argument types, and the compiler configuration

    world = tls_world_age()
    key = hash(ft)
    key = hash(tt, key)
    key = hash(world, key)
    key = hash(cfg, key)

    # NOTE: no use of lock(::Function)/@lock/get! to avoid try/catch and closure overhead
    lock(cache_lock)
    obj = get(cache, key, nothing)
    unlock(cache_lock)

    LLVM.Interop.assume(isassigned(compile_hook))
    if obj === nothing || compile_hook[] !== nothing
        obj = actual_compilation(cfg, ft, tt, world, compiler, linker)::V
        lock(cache_lock)
        cache[key] = obj
        unlock(cache_lock)
    end
    return obj::V
end

@noinline function actual_compilation(cfg::CompilerConfig, ft::Type, tt::Type, world::UInt,
                                      compiler::Function, linker::Function)
    src = methodinstance(ft, tt)
    job = CompilerJob(src, cfg, world)

    # somewhat fast path: intersect the requested world age with the cached codeinstances
    cache = ci_cache(job)
    ci = ci_cache_lookup(cache, src, world, world)
    if ci !== nothing && haskey(cache.obj_for_ci, ci)
        return cache.obj_for_ci[ci]
    end

    # slow path: compile and link
    # TODO: consider loading the assembly from an on-disk cache here
    asm = compiler(job)
    obj = linker(job, asm)
    ci = ci_cache_lookup(cache, src, world, world)
    cache.obj_for_ci[ci] = obj

    return obj
end
