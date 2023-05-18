const CACHE_NAME = gensym(:CACHE) # is now a const symbol (not a variable)
is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0

export ci_cache_snapshot, ci_cache_delta, ci_cache_insert, precompile_gpucompiler

function ci_cache_snapshot()
    cleaned_cache_to_save = IdDict()
    for key in keys(GPUCompiler.GLOBAL_CI_CACHES)
        # Will only keep those elements with infinite ranges
        # copy constructor
        cleaned_cache_to_save[key] = GPUCompiler.CodeCache(GPUCompiler.GLOBAL_CI_CACHES[key])
    end

    return cleaned_cache_to_save
end

function ci_cache_delta(previous_snapshot)
    current_snapshot = ci_cache_snapshot()
    delta_snapshot = IdDict{Tuple{DataType, Core.Compiler.InferenceParams, Core.Compiler.OptimizationParams}, GPUCompiler.CodeCache}()
    for (cachekey, codecache) in current_snapshot # iterate through all caches
        if cachekey in keys(previous_snapshot)
            for (mi, civ) in codecache.dict # iterate through all mi
                if mi in keys(previous_snapshot[cachekey].dict)
                    for ci in civ
                        if !(ci in previous_snapshot[cachekey].dict[mi])
                            if !(cachekey in keys(delta_snapshot))
                                delta_snapshot[cachekey] = GPUCompiler.CodeCache()
                                delta_snapshot[cachekey].dict[mi] = Vector{CodeInstance}()
                                if haskey(codecache.asm, mi)
                                    delta_snapshot[cachekey].asm[mi] = codecache.asm[mi]
                                end
                            elseif !(mi in keys(delta_snapshot[cachekey].dict))
                                delta_snapshot[cachekey].dict[mi] = Vector{CodeInstance}()
                                if haskey(codecache.asm, mi)
                                    delta_snapshot[cachekey].asm[mi] = codecache.asm[mi]
                                end
                            end

                            push!(delta_snapshot[cachekey].dict[mi], ci)
                        end
                    end
                else
                    # this whole cache is not present in the previous snapshot, can add all
                    if !(cachekey in keys(delta_snapshot))
                        delta_snapshot[cachekey] = GPUCompiler.CodeCache()
                    end
                    
                    if haskey(codecache.asm, mi)
                        delta_snapshot[cachekey].asm[mi] = codecache.asm[mi]
                    end
                    delta_snapshot[cachekey].dict[mi] = civ
                end
            end
        else
            delta_snapshot[cachekey] = current_snapshot[cachekey]
        end
    end

    return delta_snapshot
end

function print_keys(caches)
    println("************")
    for (key, cache) in caches
        for (mi, civ) in cache.dict
            println("$mi -> $(length(civ))")
        end
    end
    println("************")
end
function ci_cache_insert(cache)
    if !is_precompiling()
        # need to merge caches at the code instance level
        for (key, local_cache) in cache
            if haskey(GPUCompiler.GLOBAL_CI_CACHES, key)
                global_cache = GPUCompiler.GLOBAL_CI_CACHES[key]
                for (mi, civ) in (local_cache.dict)
                    # this should be one since there is only one range that is infinite
                    @assert length(civ) == 1
                    # add all code instances to global cache
                    # could move truncating code to set index
                    Core.Compiler.setindex!(global_cache, civ[1], mi)
                    #@assert haskey(local_cache.asm, mi)
                    if haskey(local_cache.asm, mi)
                        global_cache.asm[mi] = local_cache.asm[mi]
                    end
                end
            else
                # no conflict at cache level
                GPUCompiler.GLOBAL_CI_CACHES[key] = cache[key]
            end
        end
    end
end

"""
Given a function and param types caches the function to the global cache
"""
function precompile_gpucompiler(job)
    # populate the cache
    cache = GPUCompiler.ci_cache(job)
    mt = GPUCompiler.method_table(job)
    interp = GPUCompiler.get_interpreter(job)
    if GPUCompiler.ci_cache_lookup(cache, job.source, job.world, typemax(Cint)) === nothing
        GPUCompiler.ci_cache_populate(interp, cache, mt, job.source, job.world, typemax(Cint))
    end
end

"""
Generate a precompile file for the current state of the cache
"""
function generate_precompilation_file(snapshot, filename, precompilation_function)
    method_instances = []
    for (cachekey, cache) in snapshot
        for (mi, civ) in cache.dict
            push!(method_instances, mi)
        end
    end

    precompile_statements = join(["$precompilation_function($(mi.specTypes.parameters[1]), Core.$(mi.specTypes.parameters[2:length(mi.specTypes.parameters)]))" for mi in method_instances], '\n')
    open(filename, "w") do file
        write(file, precompile_statements)
    end
end
