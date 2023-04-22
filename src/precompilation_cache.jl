const CACHE_NAME = gensym(:CACHE) # is now a const symbol (not a variable)
is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0

export ci_cache_snapshot, ci_cache_delta, ci_cache_insert, precompile_gpucompiler

function ci_cache_snapshot()
    cleaned_cache_to_save = IdDict()
    for key in keys(GPUCompiler.GLOBAL_CI_CACHES)
        # Will only keep those elements with infinite ranges
        cleaned_cache_to_save[key] = GPUCompiler.CodeCache(GPUCompiler.GLOBAL_CI_CACHES[key])
    end
    return cleaned_cache_to_save
end

function ci_cache_delta(previous_snapshot)
    current_snapshot = ci_cache_snapshot()
    delta_snapshot = IdDict{Tuple{DataType, Core.Compiler.InferenceParams, Core.Compiler.OptimizationParams}, GPUCompiler.CodeCache}()
    for (cachekey, codecache) in current_snapshot
        if cachekey in keys(previous_snapshot)
            for (mi, civ) in codecache.dict
                if mi in keys(previous_snapshot[cachekey].dict)
                    for ci in civ
                        if !(ci in previous_snapshot[cachekey].dict[mi])
                            if !(cachekey in keys(delta_snapshot))
                                delta_snapshot[cachekey] = GPUCompiler.CodeCache()
                                delta_snapshot[cachekey].dict[mi] = Vector{CodeInstance}()
                            elseif !(mi in keys(delta_snapshot[cachekey].dict))
                                delta_snapshot[cachekey].dict[mi] = Vector{CodeInstance}()
                            end

                            push!(delta_snapshot[cachekey].dict[mi], ci)
                        end
                    end
                else
                    # this whole cache is not present in the previous snapshot, can add all
                    if !(cachekey in keys(delta_snapshot))
                        delta_snapshot[cachekey] = GPUCompiler.CodeCache()
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

#=function ci_cache_insert(caches)
    empty!(GPUCompiler.GLOBAL_CI_CACHES)
    for (key, cache) in caches
        GPUCompiler.GLOBAL_CI_CACHES[key] = GPUCompiler.CodeCache(cache)
    end
end=#

function ci_cache_insert(cache)
    if !is_precompiling()
        #first clean the cache
        cleaned_cache = IdDict()
        for (key, c) in cache
            usedCache = false
            newCodeCache = GPUCompiler.CodeCache()
            for (mi, civ) in c.dict
                new_civ = Vector()
                for ci in civ
                    if ci.min_world <= ci.max_world
                        push!(new_civ, ci)
                    end
                end
                if length(new_civ) > 0
                    usedCache = true
                    newCodeCache.dict[mi] = new_civ
                end
            end
            if usedCache
                cleaned_cache[key] = newCodeCache
            end
        end

        # need to merge caches at the code instance level
        for (key, local_cache) in cleaned_cache
            if haskey(GPUCompiler.GLOBAL_CI_CACHES, key)
                global_cache = GPUCompiler.GLOBAL_CI_CACHES[key]
                #local_cache = cache[key]
                for (mi, civ) in (local_cache.dict)
                    # this should be one since there is only one range that is infinite
                    @assert length(civ) == 1
                    # add all code instances to global cache
                    # could move truncating code to set index
                    ci = civ[1]
                    if haskey(global_cache.dict, mi)
                        gciv = global_cache.dict[mi]
                        # truncation cod3
                        # sort by min world age, then make sure no age ranges overlap // this part is uneeded
                        sort(gciv, by=x->x.min_world)
                        if ci.min_world > gciv[length(gciv)].min_world
                            invalidate_code_cache(global_cache, mi, ci.min_world - 1)
                            Core.Compiler.setindex!(global_cache, ci, mi)
                        else
                            println("Should not get here?")
                            @assert false
                        end
                    else
                        # occurs if we kill everything in the parent and then need to store in child
                        Core.Compiler.setindex!(global_cache, ci, mi)
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
Reloads Global Cache from global variable which stores the previous
cached results
"""
function reinit_cache(LOCAL_CACHE)
    if !is_precompiling()
        # need to merge caches at the code instance level
        for key in keys(LOCAL_CACHE)
            if haskey(GPUCompiler.GLOBAL_CI_CACHES, key)
                global_cache = GPUCompiler.GLOBAL_CI_CACHES[key]
                local_cache = LOCAL_CACHE[key]
                for (mi, civ) in (local_cache.dict)
                    # this should be one since there is only one range that is infinite
                    @assert length(civ) == 1
                    # add all code instances to global cache
                    # could move truncating code to set index
                    ci = civ[1]
                    if haskey(global_cache.dict, mi)
                        gciv = global_cache.dict[mi]
                        # truncation cod3
                        # sort by min world age, then make sure no age ranges overlap // this part is uneeded
                        sort(gciv, by=x->x.min_world)
                        if ci.min_world > gciv[length(gciv)].min_world
                            invalidate_code_cache(global_cache, mi, ci.min_world - 1)
                            Core.Compiler.setindex!(global_cache, ci, mi)
                        else
                            println("Should not get here?")
                            @assert false
                        end
                    else
                        # occurs if we kill everything in the parent and then need to store in child
                        Core.Compiler.setindex!(global_cache, ci, mi)
                    end
                end
            else
                # no conflict at cache level
                GPUCompiler.GLOBAL_CI_CACHES[key] = LOCAL_CACHE[key]
            end
        end
    end
end

"""
Takes a snapshot of the current status of the cache

The cache returned is a deep copy with finite world age endings removed
"""
function snapshot_cache(LOCAL_CACHE)
    cleaned_cache_to_save = IdDict()
    for key in keys(GPUCompiler.GLOBAL_CI_CACHES)
        # Will only keep those elements with infinite ranges
        cleaned_cache_to_save[key] = GPUCompiler.CodeCache(GPUCompiler.GLOBAL_CI_CACHES[key])
    end
    global MY_CACHE #technically don't need the global
    #empty insert
    empty!(LOCAL_CACHE)
    merge!(LOCAL_CACHE, cleaned_cache_to_save)
end
