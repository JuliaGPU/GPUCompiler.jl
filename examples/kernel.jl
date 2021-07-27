using GPUCompiler

module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return

    # for validation
    sin(x) = Base.sin(x)
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::Compiler{<:Any,TestCompilerParams}) = TestRuntime

kernel() = nothing

function main()
    target = NativeCompilerTarget()
    params = TestCompilerParams()
    compiler = Compiler(target, params)
    source = FunctionSpec(kernel)
    job = CompilerJob(compiler, source)

    println(GPUCompiler.compile(:asm, job)[1])
end

isinteractive() || main()
