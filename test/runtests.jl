using Test, Base.CoreLogging
import Base.CoreLogging: Info

using GPUCompiler

using LLVM, LLVM.Interop

using Pkg

include("util.jl")

@testset "GPUCompiler" begin

GPUCompiler.reset_runtime()

GPUCompiler.enable_timings()

include("native.jl")
include("ptx.jl")
if VERSION >= v"1.4"
  Pkg.add(["SPIRV_LLVM_Translator_jll", "SPIRV_Tools_jll"])
  include("spirv.jl")
end
#include("gcn.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

end
