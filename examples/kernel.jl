using GPUCompiler

module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

kernel() = nothing

function main()
    source = methodinstance(typeof(kernel), Tuple{})
    target = NativeCompilerTarget()
    params = TestCompilerParams()
    config = CompilerConfig(target, params)
    job = CompilerJob(source, config)

    output = JuliaContext() do ctx
        GPUCompiler.compile(:asm, job)
    end

    println(output[1])
end

isinteractive() || main()
