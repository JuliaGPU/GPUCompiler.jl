# Legacy and deprecated functionality.
#
# This file houses code paths that are kept for Julia 1.10 (which lacks the integrated
# CodeInstance cache and `analysis_results`) plus older interfaces retained for the 2.0
# breaking release. Everything here is expected to go away once 1.10 falls off the LTS
# line and downstream packages have migrated.

# `WorldView` is used by the integrated-cache JIT lookup callback (1.11–1.13) and by the
# legacy `CodeCache` interface (1.10). Moved out of `Core.Compiler` on 1.14+.
@static if VERSION < v"1.14-"
    using Core.Compiler: WorldView
end


## Julia 1.10: in-process `CodeCache` + WorldView

@static if !HAS_INTEGRATED_CACHE

"""
    CodeCache

Per-job CodeInstance store used on Julia 1.10, where there's no integrated cache and no
`cache_owner` partition on `CodeInstance`. Inference deposits CIs via a `WorldView` over
this store; invalidation callbacks attached to each `MethodInstance` clip `max_world` on
stale CIs when methods are redefined.
"""
struct CodeCache
    dict::IdDict{MethodInstance, Vector{CodeInstance}}

    CodeCache() = new(IdDict{MethodInstance, Vector{CodeInstance}}())
end

function Base.show(io::IO, ::MIME"text/plain", cc::CodeCache)
    print(io, "CodeCache with $(mapreduce(length, +, values(cc.dict); init=0)) entries")
    if !isempty(cc.dict)
        print(io, ": ")
        for (mi, cis) in cc.dict
            println(io)
            print(io, "  ")
            show(io, mi)
            for ci in cis
                println(io)
                worldstr = if ci.min_world == typemax(UInt)
                    "empty world range"
                elseif ci.max_world == typemax(UInt)
                    "worlds $(Int(ci.min_world))+"
                else
                    "worlds $(Int(ci.min_world)) to $(Int(ci.max_world))"
                end
                print(io, "    CodeInstance for ", worldstr)
            end
        end
    end
end

Base.empty!(cc::CodeCache) = empty!(cc.dict)

# Session-local registry of `CodeCache`s, keyed by the configuration that produced them.
# 1.10 has no per-CI partitioning, so we partition by `cache_owner(job)` ourselves.
const GLOBAL_CI_CACHES = Dict{Any, CodeCache}()
const GLOBAL_CI_CACHES_LOCK = ReentrantLock()

function get_code_cache(@nospecialize(job::CompilerJob))
    key = cache_owner(job)
    Base.@lock GLOBAL_CI_CACHES_LOCK get!(GLOBAL_CI_CACHES, key) do
        CodeCache()
    end
end

# Invalidation: keep `max_world` honest when callees in a CI's edge graph change.
struct CodeCacheCallback
    cache::CodeCache
end

function CC.setindex!(cache::CodeCache, ci::CodeInstance, mi::MethodInstance)
    add_codecache_callback!(cache, mi)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

@static if VERSION ≥ v"1.11.0-DEV.798"

function add_codecache_callback!(cache::CodeCache, mi::MethodInstance)
    callback = CodeCacheCallback(cache)
    CC.add_invalidation_callback!(callback, mi)
end
function (callback::CodeCacheCallback)(replaced::MethodInstance, max_world::UInt32)
    cis = get(callback.cache.dict, replaced, nothing)
    cis === nothing && return
    for ci in cis
        if ci.max_world == ~0 % Csize_t
            @assert ci.min_world - 1 <= max_world "attempting to set illogical constraints"
@static if VERSION >= v"1.11.0-DEV.1390"
            @atomic ci.max_world = max_world
else
            ci.max_world = max_world
end
        end
        @assert ci.max_world <= max_world
    end
end

else

function add_codecache_callback!(cache::CodeCache, mi::MethodInstance)
    callback = CodeCacheCallback(cache)
    if !isdefined(mi, :callbacks)
        mi.callbacks = Any[callback]
    elseif !in(callback, mi.callbacks)
        push!(mi.callbacks, callback)
    end
end
function (callback::CodeCacheCallback)(replaced::MethodInstance, max_world::UInt32,
                                       seen::Set{MethodInstance}=Set{MethodInstance}())
    push!(seen, replaced)

    cis = get(callback.cache.dict, replaced, nothing)
    if cis !== nothing
        for ci in cis
            if ci.max_world == ~0 % Csize_t
                @assert ci.min_world - 1 <= max_world "attempting to set illogical constraints"
                ci.max_world = max_world
            end
            @assert ci.max_world <= max_world
        end
    end

    # recurse into backedges to update their valid range too
    if isdefined(replaced, :backedges)
        backedges = filter(replaced.backedges) do @nospecialize(mi)
            if mi isa MethodInstance
                mi ∉ seen
            elseif mi isa Type
                # an `invoke` call, which is a `(sig, MethodInstance)` pair.
                # let's ignore the `sig` and process the `MethodInstance` next.
                false
            else
                error("invalid backedge")
            end
        end

        # Don't touch/empty backedges — `invalidate_method_instance` in C does that later.

        for mi in backedges
            callback(mi::MethodInstance, max_world, seen)
        end
    end
end

end

## 1.10 cache view interface

function CC.haskey(wvc::WorldView{CodeCache}, mi::MethodInstance)
    CC.get(wvc, mi, nothing) !== nothing
end

function CC.get(wvc::WorldView{CodeCache}, mi::MethodInstance, default)
    for ci in get!(wvc.cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            return ci
        end
    end
    return default
end

function CC.getindex(wvc::WorldView{CodeCache}, mi::MethodInstance)
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{CodeCache}, ci::CodeInstance, mi::MethodInstance)
    CC.setindex!(wvc.cache, ci, mi)
end


## 1.10 StackedMethodTable
#
# Priority-lookup method table. The 1.11+ version lives in CompilerCaching; here we keep
# the 1.10 copy, which has to produce the older `MethodMatchResult` shape from
# `CC.findall` / `CC.findsup`.
struct StackedMethodTable{MTV<:CC.MethodTableView} <: CC.MethodTableView
    world::UInt
    mt::Core.MethodTable
    parent::MTV
end
StackedMethodTable(world::UInt, mt::Core.MethodTable) =
    StackedMethodTable(world, mt, CC.InternalMethodTable(world))
StackedMethodTable(world::UInt, mt::Core.MethodTable, parent::Core.MethodTable) =
    StackedMethodTable(world, mt, StackedMethodTable(world, parent))

CC.isoverlayed(::StackedMethodTable) = true

function CC.findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = CC._findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing # too many matches
    nr = CC.length(result)
    if nr ≥ 1 && CC.getindex(result, nr).fully_covers
        return CC.MethodMatchResult(result, true)
    end

    parent_result = CC.findall(sig, table.parent; limit)::Union{Nothing, CC.MethodMatchResult}
    parent_result === nothing && return nothing # too many matches

    overlayed = parent_result.overlayed | !CC.isempty(result)
    parent_result = parent_result.matches::CC.MethodLookupResult

    return CC.MethodMatchResult(
        CC.MethodLookupResult(
            CC.vcat(result.matches, parent_result.matches),
            CC.WorldRange(
                CC.max(result.valid_worlds.min_world, parent_result.valid_worlds.min_world),
                CC.min(result.valid_worlds.max_world, parent_result.valid_worlds.max_world)),
            result.ambig | parent_result.ambig),
        overlayed)
end

function CC.findsup(@nospecialize(sig::Type), table::StackedMethodTable)
    match, valid_worlds = CC._findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds, true
    parent_match, parent_valid_worlds, overlayed = CC.findsup(sig, table.parent)
    return (
        parent_match,
        CC.WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds.min_world),
            min(valid_worlds.max_world, parent_valid_worlds.max_world)),
        overlayed)
end


## 1.10 `cached_results`
#
# Session-local storage for the per-job results structs; on 1.11+ these live on the
# `CodeInstance`s of Julia's integrated cache instead (see `interface.jl`). The key
# mirrors the 1.11+ semantics — the full job identity, i.e. method instance, world,
# and entire config — but nothing persists across sessions, and redefinition
# protection comes from the world age in the key rather than from CI invalidation.
const job_results = Dict{Any,Any}()
const job_results_lock = ReentrantLock()

# specialized for the launch hot path, mirroring the 1.11+ implementation
function cached_results(::Type{V}, job::CompilerJob) where {V}
    # NOTE: store the MethodInstance's objectid (not the MI) to avoid an expensive
    #       boxing allocation; the MI is kept alive by its method specializations.
    key = (V, objectid(job.source), job.world, job.config)
    Base.@lock job_results_lock begin
        get!(job_results, key) do
            V()
        end::V
    end
end


end # !HAS_INTEGRATED_CACHE


## Legacy `cached_compilation` (1.10+)

# A session-local, MI-keyed kernel cache modeled after the pre-CompilerCaching API. Used
# by back-ends that haven't migrated to the `CacheView`-based flow (and by all back-ends
# on Julia 1.10, where the new flow doesn't apply because there's no integrated cache /
# `analysis_results`).

"""
    cached_compilation(cache::AbstractDict, src::MethodInstance, cfg::CompilerConfig,
                       compiler, linker)

Compile a method instance `src` with configuration `cfg`, by invoking `compiler` and
`linker` and storing the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using any value and store
whatever the `linker` function returns. The `compiler` function should take a
`CompilerJob` and return data that the `linker` function then turns into a session-local
artifact (e.g. a `CuModule`).

This is the legacy caching API used before GPUCompiler 2.0. New code on Julia 1.11+
should prefer `CompilerCaching.CacheView`-based caching (see the package extension).
"""
function cached_compilation(cache::AbstractDict{<:Any,V},
                            src::MethodInstance, cfg::CompilerConfig,
                            compiler::Function, linker::Function) where {V}
    world = tls_world_age()
    key = (objectid(src), world, cfg)
    # NOTE: store the MethodInstance's objectid (not the MI) to avoid an expensive boxing
    # allocation. Base does this with a multi-level lookup; we use a single-level dict.

    Base.@lock cached_compilation_lock begin
        obj = get(cache, key, nothing)
    end

    if obj === nothing || compile_hook[] !== nothing
        obj = actual_compilation(cache, src, world, cfg, compiler, linker)::V
        Base.@lock cached_compilation_lock begin
            cache[key] = obj
        end
    end
    obj::V
end

const cached_compilation_lock = ReentrantLock()

@noinline function actual_compilation(cache::AbstractDict, src::MethodInstance, world::UInt,
                                      cfg::CompilerConfig, compiler::Function,
                                      linker::Function)
    job = CompilerJob(src, cfg, world)
    asm = nothing

    # provide a hook to use in tools / debuggers
    if compile_hook[] !== nothing
        Base.invokelatest(compile_hook[], job)
    end

    asm = compiler(job)
    obj = linker(job, asm)
    obj
end

@public cached_compilation
