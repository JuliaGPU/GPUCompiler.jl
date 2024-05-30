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

mutable struct CacheEntry
    const ci::Core.CodeInstance
    const obj::Any
    @atomic next::Union{CacheEntry, Nothing}
end

@inline function lookup(entry::CacheEntry, world::UInt)::Any
    while entry !== nothing
        ci = entry.ci
        if ci.min_world <= world <= ci.max_world
            return entry.obj
        end
        entry = entry.next
    end
    return nothing
end

@inline function insert!(entry, nentry)
    success = false
    while !success
        next = entry.next
        if next === nothing
            entry, success = @atomicreplace entry.next nothing => nentry
        else
            entry = next
        end
    end
    return
end

"""
    cached_compilation(cache::Dict{Any}, src::MethodInstance, cfg::CompilerConfig,
                       compiler, linker)

Compile a method instance `src` with configuration `cfg`, by invoking `compiler` and
`linker` and storing the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using any value and store
whatever the `linker` function returns. The `compiler` function should take a `CompilerJob`
and return data that can be cached across sessions (e.g., LLVM IR). This data is then
forwarded, along with the `CompilerJob`, to the `linker` function which is allowed to create
session-dependent objects (e.g., a `CuModule`).
"""
function cached_compilation(cache::AbstractDict{<:Any,V},
                            src::MethodInstance, cfg::CompilerConfig,
                            compiler::Function, linker::Function) where {V}
    key = (objectid(src), cfg)
    # NOTE: we store the MethodInstance's objectid to avoid an expensive allocation.
    # NOTE: no use of lock(::Function)/@lock/get! to avoid try/catch and closure overhead
    lock(cache_lock)
    entry = get(cache, key, nothing)
    unlock(cache_lock)

    success = entry !== nothing
    world = tls_world_age()
    obj = success ? lookup(entry::CacheEntry, world) : nothing

    if obj !== nothing
        return obj::V
    end

    ci, obj = actual_compilation(src, world, cfg, compiler, linker)::V

    new_entry = CacheEntry(ci, obj, nothing)
    if success
        insert!(entry::CacheEntry, new_entry)
        return obj::V
    end

    lock(cache_lock)
    # Need to re-obtain entry in case we raced.
    entry = get(cache, key, nothing)
    if entry === nothing
        cache[key] = new_entry
    else
        insert!(entry::CacheEntry, new_entry)
    end
    unlock(cache_lock)
    return obj::V
end

@noinline function actual_compilation(src::MethodInstance, world::UInt,
                                      cfg::CompilerConfig, compiler::Function, linker::Function)
    job = CompilerJob(src, cfg, world)
    obj = nothing

    ci = ci_cache_lookup(ci_cache(job), src, world, world)::Union{Nothing,CodeInstance}
    if obj === nothing 
        # TODO: consider loading the assembly from an on-disk cache here
        asm = compiler(job)
        obj = linker(job, asm)

        if ci === nothing
            ci = ci_cache_lookup(ci_cache(job), src, world, world)::CodeInstance
        end
    end

    return ci, obj
end
