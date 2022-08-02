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
disk_cache() = parse(Bool, @load_preference("disk_cache", "false"))

"""
    enable_cache!(state::Bool=true)

Activate the GPUCompiler disk cache in the current environment.
You will need to restart your Julia environment for it to take effect.

!!! note
    The cache functionality requires Julia 1.11
"""
function enable_cache!(state::Bool=true)
    @set_preferences!("disk_cache"=>string(state))
end

cache_path() = @get_scratch!("cache")
clear_disk_cache!() = rm(cache_path(); recursive=true, force=true)

const cache_lock = ReentrantLock()

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
    # NOTE: we index the cach both using (mi, world, cfg) keys, for the fast look-up,
    #       and using CodeInfo keys for the slow look-up. we need to cache both for
    #       performance, but cannot use a separate private cache for the ci->obj lookup
    #       (e.g. putting it next to the CodeInfo's in the CodeCache) because some clients
    #       expect to be able to wipe the cache (e.g. CUDA.jl's `device_reset!`)

    # fast path: index the cache directly for the *current* world + compiler config

    world = tls_world_age()
    key = (objectid(src), world, cfg)
    # NOTE: we store the MethodInstance's objectid to avoid an expensive allocation.
    #       Base does this with a multi-level lookup, first keyed on the mi,
    #       then a linear scan over the (typically few) entries.

    # NOTE: no use of lock(::Function)/@lock/get! to avoid try/catch and closure overhead
    lock(cache_lock)
    obj = get(cache, key, nothing)
    unlock(cache_lock)

    if obj === nothing || compile_hook[] !== nothing
        obj = actual_compilation(cache, src, world, cfg, compiler, linker)::V
        lock(cache_lock)
        cache[key] = obj
        unlock(cache_lock)
    end
    return obj::V
end

@noinline function cache_file(ci::CodeInstance, cfg::CompilerConfig)
    @static if isdefined(Base, :object_build_id)
        id = Base.object_build_id(ci)
        if id === nothing # CI is from a runtime compilation, not worth caching on disk
            return nothing
        else
            id = id % UInt64 # The upper 64bit are a checksum, unavailable during precompilation
        end
    else
        id = Base.objectid(ci)
    end

    gpucompiler_buildid = Base.module_build_id(@__MODULE__)
    if (gpucompiler_buildid >> 64) % UInt64 == 0xffffffffffffffff
        return nothing # Don't cache during precompilation of GPUCompiler
    end

    return joinpath(
        cache_path(),
        # bifurcate the cache by build id of GPUCompiler
        string(gpucompiler_buildid),
        string(hash(cfg, hash(id)), ".jls"))
end

@noinline function actual_compilation(cache::AbstractDict, src::MethodInstance, world::UInt,
                                      cfg::CompilerConfig, compiler::Function, linker::Function)
    job = CompilerJob(src, cfg, world)
    obj = nothing

    # fast path: find an applicable CodeInstance and see if we have compiled it before
    ci = ci_cache_lookup(ci_cache(job), src, world, world)::Union{Nothing,CodeInstance}
    if ci !== nothing
        key = (ci, cfg)
        obj = get(cache, key, nothing)
    end

    # slow path: compile and link
    if obj === nothing || compile_hook[] !== nothing
        asm = nothing
        path = nothing
        ondisk_hit = false
        @static if VERSION >= v"1.11.0-"
            # Don't try to hit the disk cache if we are for a *compile* hook
            if ci !== nothing && obj === nothing && disk_cache() # TODO: (Should we allow backends to opt out?)
                path = cache_file(ci, cfg)
                @debug "Looking for on-disk cache" job path
                if path !== nothing && isfile(path)
                    ondisk_hit = true
                    try
                        @debug "Loading compiled kernel" job path
                        asm = deserialize(path)
                    catch ex
                        @warn "Failed to load compiled kernel" job path exception=(ex, catch_backtrace())
                    end
                end
            end
        end

        if asm === nothing || compile_hook[] !== nothing
            # Run the compiler in-case we need to hook it.
            asm = compiler(job)
        end
        if obj !== nothing
            # we got here because of a *compile* hook; don't bother linking
            return obj
        end

        @static if VERSION >= v"1.11.0-"
            if !ondisk_hit && path !== nothing && disk_cache()
                @debug "Writing out on-disk cache" job path
                # TODO: Do we want to serialize some more metadata to make sure the asm matches?
                tmppath, io = mktemp(;cleanup=false)
                serialize(io, asm)
                close(io)
                # atomic move
                mkpath(dirname(path))
                Base.rename(tmppath, path, force=true)
            end
        end

        obj = linker(job, asm)

        if ci === nothing
            ci = ci_cache_lookup(ci_cache(job), src, world, world)::CodeInstance
            key = (ci, cfg)
        end
        cache[key] = obj
    end

    return obj
end
