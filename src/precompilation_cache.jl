const CACHE_NAME = gensym(:CACHE) # is now a const symbol (not a variable)
is_precompiling() = ccall(:jl_generating_output, Cint, ()) != 0

export @declare_cache, @snapshot_cache, @reinit_cache, @get_cache
export reinit_cache, snapshot_cache

macro declare_cache()
    var = esc(CACHE_NAME) #this will esc variable from our const symbol
    quote
        #const $esc(CACHE_NAME) function esc is executed when macro is executed, not when code is defined
        # dollar sign means will have the value of esc cachename here
        const $var = $IdDict()
    end
end

macro snapshot_cache()
    var = esc(CACHE_NAME)
    quote
        $snapshot_cache($var)
    end
end

macro reinit_cache()
    var = esc(CACHE_NAME)
    quote
        # will need to keep track of this is CUDA so that GPUCompiler caches are not overfilled
        $reinit_cache($var)
    end
end

macro get_cache()
    var = esc(CACHE_NAME)
    quote
        $var
    end
end

function declare_cache()
    return IdDict()
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
