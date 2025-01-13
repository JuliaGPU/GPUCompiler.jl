precompile_test_harness("Inference caching") do load_path
    # Write out the PTX test helpers as a micro package
    create_standalone(load_path, "PTXCompiler", "ptx.jl")

    write(joinpath(load_path, "PTXBackend.jl"), :(
        module PTXBackend
        import PTXCompiler
        using PrecompileTools

        function kernel()
            return
        end

        let
            job, _ = PTXCompiler.PTX.create_job(kernel, ())
            precompile(job)
        end

        # identity is foreign
        @setup_workload begin
            job, _ = PTXCompiler.PTX.create_job(identity, (Int,))
            @compile_workload begin
                precompile(job)
            end
        end
    end) |> string)

    Base.compilecache(Base.PkgId("PTXBackend"))
    @eval let
        import PTXCompiler

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

        using PTXBackend

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(PTXBackend.kernel), Tuple{})
        @test check_presence(kernel_mi, token)

        # check that identity survived
        @test check_presence(identity_mi, token) broken=VERSION>=v"1.12.0-DEV.1268"
    end
end
