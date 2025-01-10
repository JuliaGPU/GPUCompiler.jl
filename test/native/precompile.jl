

precompile_test_harness("Inference caching") do load_path
    # Write out the Native test setup as a micro package
    create_standalone(load_path, "TestCompiler", "native.jl")

    write(joinpath(load_path, "InferenceCaching.jl"), :(module InferenceCaching
        import TestCompiler
        using PrecompileTools

        function kernel(A, x)
            A[1] = x
            return
        end

        let
            job, _ = TestCompiler.Native.create_job(kernel, (Vector{Int}, Int))
            precompile(job)
        end

        # identity is foreign
        @setup_workload begin
            job, _ = TestCompiler.Native.create_job(identity, (Int,))
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
            job, _ = TestCompiler.Native.create_job(identity, (Int,))
            GPUCompiler.ci_cache_token(job)
        end
        @test !check_presence(identity_mi, token)

        using InferenceCaching

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(InferenceCaching.kernel), Tuple{Vector{Int}, Int})
        @test check_presence(kernel_mi, token)

        # check that identity survived
        @test check_presence(identity_mi, token) broken=VERSION>=v"1.12.0-DEV.1268"

        GPUCompiler.clear_disk_cache!()
        @test GPUCompiler.disk_cache_enabled() == false

        GPUCompiler.enable_disk_cache!()
        @test GPUCompiler.disk_cache_enabled() == true

        job, _ = TestCompiler.Native.create_job(InferenceCaching.kernel, (Vector{Int}, Int))
        @assert job.source == kernel_mi
        ci = GPUCompiler.ci_cache_lookup(GPUCompiler.ci_cache(job), job.source, job.world, job.world)
        @assert ci !== nothing
        @assert ci.inferred !== nothing
        path = GPUCompiler.cache_file(ci, job.config)
        @test path !== nothing
        @test !ispath(path)
        TestCompiler.Native.cached_execution(InferenceCaching.kernel, (Vector{Int}, Int))
        @test ispath(path)
        GPUCompiler.clear_disk_cache!()
        @test !ispath(path)

        GPUCompiler.enable_disk_cache!(false)
        @test GPUCompiler.disk_cache_enabled() == false
    end
end
