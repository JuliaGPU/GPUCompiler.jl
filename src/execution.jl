# reusable functionality to implement code execution

export split_kwargs, assign_args!


## macro tools

# split keyword arguments expressions into groups. returns vectors of keyword argument
# values, one more than the number of groups (unmatched keywords in the last vector).
# intended for use in macros; the resulting groups can be used in expressions.
# can be used at run time, but not in performance critical code.
function split_kwargs(kwargs, kw_groups...)
    # called by launch macros with differently shaped arguments, so don't compile it for each
    @nospecialize
    kwarg_groups = ntuple(_->[], length(kw_groups) + 1)
    for kwarg in kwargs
        # decode
        if Meta.isexpr(kwarg, :(=))
            # use in macros
            key, val = kwarg.args
        elseif kwarg isa Pair{Symbol,<:Any}
            # use in functions
            key, val = kwarg
        else
            throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        end
        isa(key, Symbol) || throw(ArgumentError("non-symbolic keyword '$key'"))

        # find a matching group
        group = length(kwarg_groups)
        for (i, kws) in enumerate(kw_groups)
            if key in kws
                group = i
                break
            end
        end
        push!(kwarg_groups[group], kwarg)
    end

    return kwarg_groups
end

# assign arguments to variables, handle splatting
function assign_args!(code, _args)
    nargs = length(_args)

    # handle splatting
    splats = Vector{Bool}(undef, nargs)
    args = Vector{Any}(undef, nargs)
    for i in 1:nargs
        splats[i] = Meta.isexpr(_args[i], :(...))
        args[i] = splats[i] ? _args[i].args[1] : _args[i]
    end

    # assign arguments to variables
    vars = Vector{Symbol}(undef, nargs)
    for i in 1:nargs
        vars[i] = gensym()
        push!(code.args, :($(vars[i]) = $(args[i])))
    end

    # convert the arguments, compile the function and call the kernel
    # while keeping the original arguments alive
    var_exprs = Vector{Any}(undef, nargs)
    for i in 1:nargs
        var_exprs[i] = splats[i] ? Expr(:(...), vars[i]) : vars[i]
    end

    return vars, var_exprs
end


## kernel lookup cache

"""
    cached_lookup(lookup, V, F, tt, key)::V

Return the result of `lookup()`, reusing an earlier result for the same kernel signature
(`F`, `tt`) and `key`. This is meant for the functions that back-ends call on every kernel
launch to look up the compiled kernel (e.g. `cufunction`), which otherwise have to resolve
the method instance, build a `CompilerJob` and query the compiler's caches every time:

```julia
function cufunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    fun, state = GPUCompiler.cached_lookup(Tuple{CuFunction,KernelState}, F, tt,
                                           (context(), values(kwargs))) do
        cufunction_link(methodinstance(F, tt); kwargs...)
    end
    return HostKernel{F,tt}(f, fun, state)
end
```

An earlier result is reused when it was looked up by the same type of `lookup`, in the same
world age (see `Base.tls_world_age`), and with a key that is `isequal` to `key`. Matching the
world exactly is what keeps this correct when methods are added or redefined: compiled code
can still be valid in a newer world, but the kernel call may now dispatch to another method.
In another world `lookup` runs again, which normally finds the compiled code in the
compiler's cache.

Everything that affects the result of `lookup` besides the kernel signature and the world
must be part of `key`, e.g. the context or device and the compiler keyword arguments. Use
values with stable `isequal` semantics: mutating an object referenced by an earlier key does
not invalidate the results stored with it. `V` is the type of the result, which should be
concrete and should only contain compiled artifacts or session handles: do not return the
kernel callable `f` itself, as it holds the values captured by the kernel, which need to
come from each launch. The cache keeps neither `lookup` nor `f` alive, but it does keep
the keys and results of a few lookups per signature.

Results are not stored while a compile hook is active (e.g. by the `@device_code_*` macros),
so that the hook sees every lookup (see [`run_compile_hook`](@ref)), or while generating
output during precompilation, as session handles cannot be serialized. Exceptions thrown by
`lookup` propagate, and nothing is stored.

Hits don't take a lock. Concurrent misses may each run `lookup`, so it has to be thread-safe
itself, and stored results may be discarded at any point.
"""
@inline function cached_lookup(lookup::L, ::Type{V}, ::Type{F}, ::Type{TT},
                               key::K) where {L,V,F,TT,K}
    cache = lookup_cache(L, F, TT, K, V)
    entry = @atomic :acquire cache.head
    if entry !== nothing && entry.world == tls_world_age() && isequal(entry.key, key) &&
       !compile_hook_active()
        return entry.value
    end
    return cached_lookup_slow(cache, lookup, key)
end

@public cached_lookup

# The cache is an immutable list of results, newest first, so that hits don't need a lock.
struct LookupCacheEntry{K,V}
    world::UInt
    key::K
    value::V
    next::Union{Nothing,LookupCacheEntry{K,V}}
end

mutable struct LookupCache{K,V}
    @atomic head::Union{Nothing,LookupCacheEntry{K,V}}
    LookupCache{K,V}() where {K,V} = new{K,V}(nothing)
end

# enough for a signature that is looked up for a couple of devices, or in a couple of worlds
const LOOKUP_CACHE_SIZE = 4

# A cache per lookup and signature, as a constant in the code of every caller. The generator
# only creates an empty cache; it may run several times for the same signature, creating
# several caches, which only costs some missed hits. Callers that are serialized during
# precompilation serialize the cache too, which is why it is never written to then.
@generated function lookup_cache(::Type{L}, ::Type{F}, ::Type{TT}, ::Type{K},
                                 ::Type{V}) where {L,F,TT,K,V}
    cache = LookupCache{K,V}()
    return :($cache)
end

@noinline compile_hook_active() = active_compile_hook[] !== nothing

# Don't specialize on the lookup, which is a different closure for every kernel. The world
# is the one `lookup` runs in, as it's called from the same task.
@noinline Base.@nospecializeinfer function cached_lookup_slow(cache::LookupCache{K,V},
                                                              @nospecialize(lookup),
                                                              key::K) where {K,V}
    world = tls_world_age()
    head = @atomic :acquire cache.head
    store = !compile_hook_active() && ccall(:jl_generating_output, Cint, ()) != 1
    if store
        # the head was checked by the caller
        entry = head === nothing ? nothing : head.next
        while entry !== nothing
            if entry.world == world && isequal(entry.key, key)
                return entry.value
            end
            entry = entry.next
        end
    end

    value = lookup()::V
    if store
        # keep the most recent results, except for one with the same world and key that
        # another task may have stored meanwhile
        kept = LookupCacheEntry{K,V}[]
        entry = @atomic :acquire cache.head
        while entry !== nothing && length(kept) < LOOKUP_CACHE_SIZE - 1
            if entry.world != world || !isequal(entry.key, key)
                push!(kept, entry)
            end
            entry = entry.next
        end
        tail = nothing
        for entry in Iterators.reverse(kept)
            tail = LookupCacheEntry{K,V}(entry.world, entry.key, entry.value, tail)
        end
        @atomic :release cache.head = LookupCacheEntry{K,V}(world, key, value, tail)
    end
    return value
end
