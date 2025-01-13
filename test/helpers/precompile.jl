function precompile_test_harness(@nospecialize(f), testset::String)
    @testset "$testset" begin
        precompile_test_harness(f, true)
    end
end
function precompile_test_harness(@nospecialize(f), separate::Bool)
    # XXX: clean-up may fail on Windows, because opened files are not deletable.
    #      fix this by running the harness in a separate process, such that the
    #      compilation cache files are not opened?
    load_path = mktempdir(cleanup=true)
    load_cache_path = separate ? mktempdir(cleanup=true) : load_path
    try
        pushfirst!(LOAD_PATH, load_path)
        pushfirst!(DEPOT_PATH, load_cache_path)
        f(load_path)
    finally
        popfirst!(DEPOT_PATH)
        popfirst!(LOAD_PATH)
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

function create_standalone(load_path, name::String, file)
    code = :(
        module $(Symbol(name))

        using GPUCompiler

        include($(joinpath(@__DIR__, "runtime.jl")))
        include($(joinpath(@__DIR__, file)))

        end
    )

    # Write out the test setup as a micro package
    write(joinpath(load_path, "$name.jl"), string(code))
    Base.compilecache(Base.PkgId(name))
end
