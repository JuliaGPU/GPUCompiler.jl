precompile_test_harness("Inference caching") do load_path
    # Write out the PTX test helpers as a micro package
    create_standalone(load_path, "TestCompiler", "ptx.jl")

    write(joinpath(load_path, "InferenceCaching.jl"), :(module InferenceCaching
        import TestCompiler
        using PrecompileTools

        function kernel()
            return
        end

        let
            job, _ = TestCompiler.PTX.create_job(kernel, ())
            precompile(job)
        end

        # identity is foreign
        @setup_workload begin
            job, _ = TestCompiler.PTX.create_job(identity, (Int,))
            @compile_workload begin
                precompile(job)
            end
        end
    end) |> string)

    Base.compilecache(Base.PkgId("InferenceCaching"))
    @eval let
        import TestCompiler

        # Check that no cached entry is present
        identity_mi = GPUCompiler.methodinstance(typeof(identity), Tuple{Int})

        token = let
            job, _ = PTX.create_job(identity, (Int,))
            GPUCompiler.ci_cache_token(job)
        end
        ci = isdefined(identity_mi, :cache) ? identity_mi.cache : nothing
        while ci !== nothing
            @test ci.owner !== token
            ci = isdefined(ci, :next) ? ci.next : nothing
        end

        using InferenceCaching

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(InferenceCaching.kernel), Tuple{})
        @test check_presence(kernel_mi, token)

        # check that identity survived
        @test check_presence(identity_mi, token) broken=VERSION>=v"1.12.0-DEV.1268"
    end
end
