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
    # NOTE: we only use the codegen world age for invalidation purposes;
    #       actual compilation happens at the current world age.
    world = codegen_world_age(ft, tt)
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
        obj = actual_compilation(cache, key, cfg, ft, tt, compiler, linker)::V
    end
    return obj::V
end

@noinline function actual_compilation(cache::AbstractDict, key::UInt,
                                      cfg::CompilerConfig, ft::Type, tt::Type,
                                      compiler::Function, linker::Function)
    src = methodinstance(ft, tt)
    job = CompilerJob(src, cfg)

    asm = nothing
    # TODO: consider loading the assembly from an on-disk cache here

    # compile
    if asm === nothing
        asm = compiler(job)
    end

    # link (but not if we got here because of forced compilation,
    # in which case the cache will already be populated)
    lock(cache_lock) do
        haskey(cache, key) && return cache[key]

        obj = linker(job, asm)
        cache[key] = obj
        obj
    end
end
