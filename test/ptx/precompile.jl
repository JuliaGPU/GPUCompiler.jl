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

    Base.compilecache(Base.PkgId("PTXBackend"), stderr, stdout)
    @eval let
        import PTXCompiler

        # Check that no cached entry is present
        identity_mis = Any[
            GPUCompiler.methodinstance(typeof(identity), Tuple{Int}),
            GPUCompiler.CompilerCaching.method_instance(identity, (Int,)),
        ]
        unique!(identity_mis)

        # NOTE: use the standalone package's module to construct the token — the
        #       sandbox's own PTX helper module defines a distinct CompilerParams
        #       type, whose token can never match the precompiled CIs.
        token = let
            job, _ = PTXCompiler.PTX.create_job(identity, (Int,))
            GPUCompiler.cache_owner(job)
        end
        for identity_mi in identity_mis
            ci = isdefined(identity_mi, :cache) ? identity_mi.cache : nothing
            while ci !== nothing
                @test ci.owner !== token
                ci = isdefined(ci, :next) ? ci.next : nothing
            end
        end

        using PTXBackend

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(PTXBackend.kernel), Tuple{})
        @test check_presence(kernel_mi, token)

        # check that identity survived
        # NOTE: external CIs from the workload survive only flakily on 1.13
        #       (the 1.13.0-beta3 backport did not fully fix this), so skip the
        #       check there.
        ext_cis_lost = v"1.12.0-DEV.1268" <= VERSION < v"1.12.5" ||
                       v"1.14.0-" <= VERSION < v"1.14.0-DEV.1843"
        ext_cis_flaky = v"1.13.0-" <= VERSION < v"1.14-"
        if !ext_cis_flaky
            @test any(mi -> check_presence(mi, token), identity_mis) broken=ext_cis_lost
        end
    end
end
