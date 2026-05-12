

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

        let
            job, _ = NativeCompiler.Native.create_job(kernel, (Vector{Int}, Int))
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
        identity_mi = GPUCompiler.methodinstance(typeof(identity), Tuple{Int})

        token = let
            job, _ = NativeCompiler.Native.create_job(identity, (Int,))
            GPUCompiler.cache_owner(job)
        end
        @test !check_presence(identity_mi, token)

        using NativeBackend

        # Check that kernel survived
        kernel_mi = GPUCompiler.methodinstance(typeof(NativeBackend.kernel), Tuple{Vector{Int}, Int})
        @test check_presence(kernel_mi, token)

        kernel_w_global_mi = GPUCompiler.methodinstance(typeof(NativeBackend.kernel_w_global), Tuple{Vector{Int}, Int, Symbol})
        @test check_presence(kernel_w_global_mi, token)

        square_mi = GPUCompiler.methodinstance(typeof(NativeBackend.square), Tuple{Float64})
        @test check_presence(square_mi, token)

        # check that identity survived
        @test check_presence(identity_mi, token) broken=(v"1.12.0-DEV.1268" <= VERSION < v"1.12.5" || v"1.13.0-" <= VERSION < v"1.13.0-beta3"|| v"1.14.0-" <= VERSION < v"1.14.0-DEV.1843")

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
