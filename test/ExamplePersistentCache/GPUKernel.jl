module GPUKernel
using GPUCompiler
using TestRuntime
snapshot = GPUCompiler.ci_cache_snapshot()

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

kernel() = nothing
function main()
    source = methodinstance(typeof(kernel), Tuple{})
    target = NativeCompilerTarget()
    params = TestCompilerParams()
    config = CompilerConfig(target, params)
    job = CompilerJob(source, config)

    println(GPUCompiler.compile(:asm, job)[1])
end

main()
const persistent_cache = GPUCompiler.ci_cache_delta(snapshot)

function __init__()
    GPUCompiler.ci_cache_insert(persistent_cache)
end
end # module GPUKernel
