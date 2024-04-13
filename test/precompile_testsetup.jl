@testsetup module

using ReTestItems

export precompile_test_harness, check_presence

function precompile_test_harness(@nospecialize(f), testset::String)
    @testset "$testset" begin
        precompile_test_harness(f, true)
    end
end
function precompile_test_harness(@nospecialize(f), separate::Bool)
    load_path = mktempdir()
    load_cache_path = separate ? mktempdir() : load_path
    try
        pushfirst!(LOAD_PATH, load_path)
        pushfirst!(DEPOT_PATH, load_cache_path)
        f(load_path)
    finally
        try
            rm(load_path, force=true, recursive=true)
        catch err
            @show err
        end
        if separate
            try
                rm(load_cache_path, force=true, recursive=true)
            catch err
                @show err
            end
        end
        filter!((≠)(load_path), LOAD_PATH)
        separate && filter!((≠)(load_cache_path), DEPOT_PATH)
    end
    nothing
end

function check_presence(mi, token)
    found = false
    ci = isdefined(mi, :cache) ? mi.cache : nothing
    while ci !== nothing
        if ci.owner === token && ci.max_world == typemax(UInt)
            found = true
            break
        end
        ci = isdefined(ci, :next) ? ci.next : nothing
    end
    return found
end

end
