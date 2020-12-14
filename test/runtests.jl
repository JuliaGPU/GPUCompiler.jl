using Test, Base.CoreLogging
import Base.CoreLogging: Info

import InteractiveUtils
InteractiveUtils.versioninfo(verbose=true)

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
include("bpf.jl")

include("examples.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

end
