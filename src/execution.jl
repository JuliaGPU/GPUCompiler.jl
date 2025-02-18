# reusable functionality to implement code execution

export split_kwargs, assign_args!


## macro tools

# split keyword arguments expressions into groups. returns vectors of keyword argument
# values, one more than the number of groups (unmatched keywords in the last vector).
# intended for use in macros; the resulting groups can be used in expressions.
# can be used at run time, but not in performance critical code.
function split_kwargs(kwargs, kw_groups...)
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


## cached compilation

### Notes on interactions with package images and disk cache.
# Julia uses package images (pkgimg) to cache both the result of inference,
# and the result of native code emissions. Up until Julia v1.11 neither the
# inferred nor the nativce code of foreign abstract interpreters was cached
# across sessions. Julia v1.11 allows for caching of inference results across
# sessions as long as those inference results are created during precompilation.
#
# Julia cache hierarchy is roughly as follows:
# Function (name of a thing)
# -> Method (particular piece of code to dispatch to with a signature)
#  -> MethodInstance (A particular Method + particular signature)
#    -> CodeInstance (A MethodInstance compiled for a world)
#
# In order to cache code across sessions we need to insert CodeInstance(owner=GPUCompilerCacheToken)
# into the internal cache. Once we have done so we know that a particular CodeInstance is unique in
# the system. (During pkgimg loading conflicts will be resolved).
#
# When a pkgimg is loaded we check it's validity, this means checking that all depdencies are the same,
# the pkgimg was created for the right set of compiler flags, and that all source code that was used
# to create this pkgimg is the same. When a CodeInstance is inside a pkgimg we can extend the chain of
# validity even for GPU code, we cannot verify a "runtime" CodeInstance in the same way.
#
# Therefore when we see a compilation request for a CodeInstance that is originating from a pkgimg
# we can use it as part of the hash for the on-disk cache. (see `cache_file`)

"""
    disk_cache_enabled()

Query if caching to disk is enabled.
"""
disk_cache_enabled() = parse(Bool, @load_preference("disk_cache", "false"))

"""
    enable_disk_cache!(state::Bool=true)

Activate the GPUCompiler disk cache in the current environment.
You will need to restart your Julia environment for it to take effect.

!!! note
    The cache functionality requires Julia 1.11
"""
function enable_disk_cache!(state::Bool=true)
    @set_preferences!("disk_cache"=>string(state))
end

disk_cache_path() = @get_scratch!("disk_cache")
clear_disk_cache!() = rm(disk_cache_path(); recursive=true, force=true)

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
    h = hash(Base.objectid(ci))
    @static if isdefined(Base, :object_build_id)
        bid = Base.object_build_id(ci)
        if bid === nothing # CI is from a runtime compilation, not worth caching on disk
            return nothing
        else
            bid = bid % UInt64 # The upper 64bit are a checksum, unavailable during precompilation
        end
        h = hash(bid, h)
    end
    h = hash(cfg, h)

    gpucompiler_buildid = Base.module_build_id(@__MODULE__)
    if (gpucompiler_buildid >> 64) % UInt64 == 0xffffffffffffffff
        return nothing # Don't cache during precompilation of GPUCompiler
    end

    return joinpath(
        disk_cache_path(),
        # bifurcate the cache by build id of GPUCompiler
        string(gpucompiler_buildid),
        string(h, ".jls"))
end

struct DiskCacheEntry
    src::Type # Originally MethodInstance, but upon deserialize they were not uniqued...
    cfg::CompilerConfig
    asm
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
            # TODO:
            #  - Sould we hit disk cache if Base.generating_output()
            #  - Should we allow backend to opt out?
            if ci !== nothing && obj === nothing && disk_cache_enabled()
                path = cache_file(ci, cfg)
                @debug "Looking for on-disk cache" job path
                if path !== nothing && isfile(path)
                    ondisk_hit = true
                    try
                        @debug "Loading compiled kernel" job path
                        # The MI we deserialize here didn't get uniqued...
                        entry = deserialize(path)::DiskCacheEntry
                        if entry.src == src.specTypes && entry.cfg == cfg
                            asm = entry.asm
                        else
                            @show entry.src == src.specTypes
                            @show entry.cfg == cfg
                            @warn "Cache missmatch" src.specTypes cfg entry.src entry.cfg
                        end
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
            if !ondisk_hit && path !== nothing && disk_cache_enabled()
                @debug "Writing out on-disk cache" job path
                mkpath(dirname(path))
                entry = DiskCacheEntry(src.specTypes, cfg, asm)

                # atomic write to disk
                tmppath, io = mktemp(dirname(path); cleanup=false)
                serialize(io, entry)
                close(io)
                @static if VERSION >= v"1.12.0-DEV.1023"
                    mv(tmppath, path; force=true)
                else
                    Base.rename(tmppath, path, force=true)
                end
            end
        end

        obj = linker(job, asm)

        if ci === nothing
            ci = ci_cache_lookup(ci_cache(job), src, world, world)
            if ci === nothing
                error("""Did not find CodeInstance for $job.

                         Pleaase make sure that the `compiler` function passed to `cached_compilation`
                         invokes GPUCompiler with exactly the same configuration as passed to the API.

                         Note that you should do this by calling `GPUCompiler.compile`, and not by
                         using reflection functions (which alter the compiler configuration).""")
            end
            key = (ci, cfg)
        end
        cache[key] = obj
    end

    return obj
end
