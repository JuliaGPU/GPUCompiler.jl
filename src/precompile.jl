using PrecompileTools: @setup_workload, @compile_workload

# Load REPL so that interactive use doesn't invalidate the precompiled compiler
# pipeline (fixed on Julia 1.14, see JuliaLang/julia#61714).
import REPL

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

        # the runtime library isn't linked (see below), so don't reject calls to it
        GPUCompiler.isintrinsic(::DummyCompilerJob, fn::String) = startswith(fn, "gpu_")

        # a minimal device array, so that kernels contain the bounds checks, exceptions and
        # memory accesses that back-ends' array types generate. (back-ends override Base's
        # `throw_boundserror`, which allocates the error, to throw without allocating.)
        struct DeviceArray{T} <: DenseVector{T}
            ptr::Ptr{T}
            len::Int
        end
        Base.size(a::DeviceArray) = (a.len,)
        @noinline throw_boundserror() = throw(nothing)
        Base.@propagate_inbounds function Base.getindex(a::DeviceArray, i::Int)
            @boundscheck 1 <= i <= a.len || throw_boundserror()
            unsafe_load(a.ptr, i)
        end
        Base.@propagate_inbounds function Base.setindex!(a::DeviceArray, x, i::Int)
            @boundscheck 1 <= i <= a.len || throw_boundserror()
            unsafe_store!(a.ptr, x, i)
            return
        end

        # exercise the parts of the pipeline that back-ends' kernels typically need: bounds
        # checks lowered to exceptions, loops, Float16 arithmetic and math intrinsics
        function kernel(a::DeviceArray{Float32}, b::DeviceArray{Float16}, n::Int)
            acc = 0f0
            for i in 1:n
                x = a[i]
                acc += fma(sqrt(abs(x)), x, floor(x))
                b[i] = b[i] * Float16(2) + Float16(x)
            end
            a[1] = min(acc, 1f0)
            return
        end
    end

    @compile_workload begin
        target = NativeCompilerTarget()
        params = precompile_module.DummyCompilerParams()
        # XXX: on Windows, compiling the GPU runtime leaks GPU code in the native cache,
        #      so prevent building the runtime library (see JuliaGPU/GPUCompiler.jl#601)
        config = CompilerConfig(target, params; libraries=false)

        tt = Tuple{precompile_module.DeviceArray{Float32},
                   precompile_module.DeviceArray{Float16}, Int}
        source = methodinstance(typeof(precompile_module.kernel), tt)
        job = CompilerJob(source, config)

        JuliaContext() do ctx
            _, meta = GPUCompiler.compile(:asm, job)
            dispose(meta.ir)
        end

        # helpers for back-ends' launch macros, which are expanded in the user's session
        split_kwargs((:(a = 1), :(b = 2)), [:a], [:c])
        assign_args!(quote end, Any[:x, :(y...)])
    end

    # reset state that was initialized during precompilation
    __llvm_initialized[] = false
end
