

precompile_test_harness("Inference caching") do load_path
    # Write out the Native test setup as a micro package
    create_standalone(load_path, "NativeCompiler", "native.jl")

    write(joinpath(load_path, "NativeBackend.jl"), :(
        module NativeBackend
        import NativeCompiler
        using PrecompileTools

        function kernel(A, x)
            A[1] = x
            return
        end

        # a kernel that makes no calls, so its CodeInstance has no inference edges
        leaf_kernel(A) = nothing

        function kernel_w_global(A, x, sym)
            if sym == :A
                A[1] = x
            end
            return
        end

        function square(x)
            return x*x
        end

        function checked_convert(x)
            return Int(x)
        end

        mutable struct Results
            artifact::Union{Nothing,String}
            Results() = new(nothing)
        end

        portable_kernel(x) = x + 1
        session_kernel(x) = x + 2

        # Attach representative back-end artifacts while the package image is built. The
        # portable entry should survive serialization; the session-dependent one should be
        # removed by GPUCompiler's pre-output atexit hook.
        let
            job, _ = NativeCompiler.Native.create_job(portable_kernel, (Int,))
            precompile(job)
            NativeCompiler.GPUCompiler.cached_results(Results, job).artifact = "portable"
        end
        let
            job, _ = NativeCompiler.Native.create_job(session_kernel, (Int,))
            precompile(job)
            NativeCompiler.GPUCompiler.cached_results(Results, job).artifact = "session"
            NativeCompiler.GPUCompiler.mark_session_dependent!(job)
        end

        let
            job, _ = NativeCompiler.Native.create_job(kernel, (Vector{Int}, Int))
            precompile(job)
        end

        let
            job, _ = NativeCompiler.Native.create_job(leaf_kernel, (Vector{Int},))
            precompile(job)
        end

        let
            job, _ = NativeCompiler.Native.create_job(kernel_w_global, (Vector{Int}, Int, Symbol))
            precompile(job)
        end

        let
            # Emit the func abi to box the return
            job, _ = NativeCompiler.Native.create_job(square, (Float64,), entry_abi=:func)
            precompile(job)
        end

        # identity is foreign
        @setup_workload begin
            job, _ = NativeCompiler.Native.create_job(identity, (Int,))
            @compile_workload begin
                precompile(job)
            end
        end

        @setup_workload begin
            job, _ = NativeCompiler.Native.create_job(checked_convert, (UInt,); validate=false)
            @compile_workload begin
                precompile(job)
            end
        end
    end) |> string)

    Base.compilecache(Base.PkgId("NativeBackend"), stderr, stdout)
    @eval let
        import NativeCompiler

        # Check that no cached entry is present
        identity_mis = Any[
            GPUCompiler.methodinstance(typeof(identity), Tuple{Int}),
            GPUCompiler.CompilerCaching.method_instance(identity, (Int,)),
        ]
        unique!(identity_mis)

        token = let
            job, _ = NativeCompiler.Native.create_job(identity, (Int,))
            GPUCompiler.cache_owner(job)
        end
        @test all(!check_presence(mi, token) for mi in identity_mis)

        using NativeBackend

        portable_job, _ = NativeCompiler.Native.create_job(NativeBackend.portable_kernel, (Int,))
        portable_res = GPUCompiler.cached_results(NativeBackend.Results, portable_job)
        @test portable_res !== nothing
        @test portable_res.artifact == "portable"

        session_job, _ = NativeCompiler.Native.create_job(NativeBackend.session_kernel, (Int,))
        session_res = GPUCompiler.cached_results(NativeBackend.Results, session_job)
        @test session_res !== nothing
        @test session_res.artifact === nothing

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(NativeBackend.kernel), Tuple{Vector{Int}, Int})
        @test check_presence(kernel_mi, token)

        kernel_w_global_mi = GPUCompiler.methodinstance(typeof(NativeBackend.kernel_w_global), Tuple{Vector{Int}, Int, Symbol})
        @test check_presence(kernel_w_global_mi, token)

        # a CodeInstance without inference edges (a kernel making no calls) is
        # serialized with the revalidation sentinel but excluded from the edge
        # verification list on Julia 1.12 (jl_record_edges skips empty-edge CIs),
        # so it deserializes permanently invalid. Fixed by the serialization
        # rework in 1.13; 1.11 uses the old scheme and is unaffected.
        leaf_edges_lost = v"1.12-" <= VERSION < v"1.13-"
        leaf_kernel_mi = GPUCompiler.methodinstance(typeof(NativeBackend.leaf_kernel), Tuple{Vector{Int}})
        @test check_presence(leaf_kernel_mi, token) broken=leaf_edges_lost

        square_mi = GPUCompiler.methodinstance(typeof(NativeBackend.square), Tuple{Float64})
        @test check_presence(square_mi, token)

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

        # Recompiling a foreign method after loading precompiled owner-token CIs
        # may also surface a native owner-less CI for the same MethodInstance.
        # GPUCompiler should prefer the owner-token CI instead of recording both.
        job, _ = NativeCompiler.Native.create_job(identity, (Int,))
        JuliaContext() do ctx
            _, meta = GPUCompiler.compile(:llvm, job)
            @test haskey(meta.compiled, job.source)
        end

        job, _ = NativeCompiler.Native.create_job(NativeBackend.checked_convert, (UInt,); validate=false)
        JuliaContext() do ctx
            _, meta = GPUCompiler.compile(:llvm, job)
            @test haskey(meta.compiled, job.source)
        end
    end
end
