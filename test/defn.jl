@testset "definitions" begin

struct ConcreteCompilerParams <: AbstractCompilerParams
end

function sqexp(x::Float64)
    return exp(x)*exp(x)
end

fspec =  GPUCompiler.FunctionSpec(sqexp,Tuple{Float64}, false)

job = GPUCompiler.CompilerJob(
    GPUCompiler.NativeCompilerTarget(),
    fspec,
    ConcreteCompilerParams())

method_instance = GPUCompiler.emit_julia(job)[1]

for (_, mi) in GPUCompiler.compile_method_instance(job, method_instance)[3]
    @test mi.def in methods(Base.exp) || mi.def in methods(sqexp)
end

end
