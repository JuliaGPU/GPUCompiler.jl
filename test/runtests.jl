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

Base.libllvm_version < v"12" && include("native.jl")    # TODO(vchuravy): wrap ORCv2
include("ptx.jl")
Base.libllvm_version < v"12" && include("spirv.jl")     # TODO(maleadt): rebuild SPIRV tools
Base.libllvm_version < v"12" && include("gcn.jl")       # TODO(jpsamaroo): test failures
include("bpf.jl")
include("examples.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

end
