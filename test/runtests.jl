using Test, Base.CoreLogging
import Base.CoreLogging: Info

using GPUCompiler

import LLVM

include("util.jl")

@testset "GPUCompiler" begin

GPUCompiler.reset_runtime()

GPUCompiler.enable_timings()

include("native.jl")
include("ptx.jl")
include("gcn.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

end
