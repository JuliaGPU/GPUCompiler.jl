@testitem "Precompile" begin

using GPUCompiler
using ReTestItems
using Test

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

precompile_test_harness("Inference caching") do load_path
    TS_Native = include("native_testsetup.jl")
    cp("runtime.jl", joinpath(load_path, "runtime.jl"))

    # Write out the Native test harness as a micro package
    write(joinpath(load_path, "Native.jl"), string(:(module Native $(TS_Native.code) end)))
    Base.compilecache(Base.PkgId("Native"))

    write(joinpath(load_path, "InferenceCaching.jl"), :(module InferenceCaching
        import Native
        import GPUCompiler

        function kernel()
            return
        end

        let
            job, _ = Native.create_job(kernel, ())
            GPUCompiler.code_typed(job)
        end
        
        # identity is foreign
        # Maybe https://github.com/JuliaLang/julia/pull/49391
        job, _ = Native.create_job(identity, (Int,))
        GPUCompiler.code_typed(job)
    end) |> string)

    Base.compilecache(Base.PkgId("InferenceCaching"))
    @eval let
        import Native

        # Check that no cached entry is present
        identity_mi = GPUCompiler.methodinstance(typeof(identity), Tuple{Int})

        token = let
            job, _ = Native.create_job(identity, (Int,))
            GPUCompiler.ci_cache_token(job)
        end
        ci = isdefined(identity_mi, :cache) ? identity_mi.cache : nothing
        while ci !== nothing
            @test ci.owner === nothing
            @test ci.owner !== token
            ci = isdefined(ci, :next) ? ci.next : nothing
        end

        using InferenceCaching

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(InferenceCaching.kernel), Tuple{})
        @test check_presence(kernel_mi, token)

        # check that identity survived
        @test_broken check_presence(identity_mi, token)
    end
end

end # testitem
