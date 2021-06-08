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

map = GPUCompiler.compile_method_instance(job, method_instance)[2]
@test method_instance in keys(map)
@test method_instance.def in methods(sqexp)

seen = false
for mi in keys(map)
    if mi != method_instance
        @test mi.def in methods(Base.exp)
        seen = true
    end
end
@test seen

end
