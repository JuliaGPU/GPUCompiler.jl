using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    precompile_module = @eval module $(gensym())
        using ..GPUCompiler

        module DummyRuntime
            # dummy methods
            signal_exception() = return
            malloc(sz) = C_NULL
            report_oom(sz) = return
            report_exception(ex) = return
            report_exception_name(ex) = return
            report_exception_frame(idx, func, file, line) = return
        end

        struct DummyCompilerParams <: AbstractCompilerParams end
        const DummyCompilerJob = CompilerJob{NativeCompilerTarget, DummyCompilerParams}

        GPUCompiler.runtime_module(::DummyCompilerJob) = DummyRuntime
    end

    kernel() = nothing

    @compile_workload begin
        source = methodinstance(typeof(kernel), Tuple{})
        target = NativeCompilerTarget()
        params = precompile_module.DummyCompilerParams()
        # XXX: on Windows, compiling the GPU runtime leaks GPU code in the native cache,
        #      so prevent building the runtime library (see JuliaGPU/GPUCompiler.jl#601)
        config = CompilerConfig(target, params; libraries=false)
        job = CompilerJob(source, config)

        JuliaContext() do ctx
            GPUCompiler.compile(:asm, job)
        end
    end

    # reset state that was initialized during precompilation
    __llvm_initialized[] = false
end
