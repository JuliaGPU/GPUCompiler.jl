# reusable functionality to implement code execution

export split_kwargs, assign_args!


## macro tools

# split keyword arguments expressions into groups. returns vectors of keyword argument
# values, one more than the number of groups (unmatched keywords in the last vector).
# intended for use in macros; the resulting groups can be used in expressions.
function split_kwargs(kwargs, kw_groups...)
    kwarg_groups = ntuple(_->[], length(kw_groups) + 1)
    for kwarg in kwargs
        # decode
        Meta.isexpr(kwarg, :(=)) || throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        key, val = kwarg.args
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


## cached compilation

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
    if compile_hook[] === nothing
        ci = ci_cache_lookup(cache, src, world, world)
        if ci !== nothing && haskey(cache.obj_for_ci, ci)
            return cache.obj_for_ci[ci]
        end
    end

    # slow path: compile and link
    # TODO: consider loading the assembly from an on-disk cache here
    asm = compiler(job)
    obj = linker(job, asm)
    ci = ci_cache_lookup(cache, src, world, world)
    cache.obj_for_ci[ci] = obj

    return obj
end
