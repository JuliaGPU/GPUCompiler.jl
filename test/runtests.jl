using Test, Base.CoreLogging
import Base.CoreLogging: Info

using GPUCompiler

using LLVM, LLVM.Interop

include("util.jl")

@testset "GPUCompiler" begin

GPUCompiler.reset_runtime()

GPUCompiler.enable_timings()

include("native.jl")
include("ptx.jl")
include("spirv.jl")
include("gcn.jl")

include("examples.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

end
