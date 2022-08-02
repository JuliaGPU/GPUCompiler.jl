using Test, Base.CoreLogging
import Base.CoreLogging: Info

import InteractiveUtils
InteractiveUtils.versioninfo(verbose=true)

using GPUCompiler

using LLVM, LLVM.Interop

include("testhelpers.jl")

@testset "GPUCompiler" begin

GPUCompiler.reset_runtime()

GPUCompiler.enable_timings()

include("util.jl")
include("native.jl")
include("ptx.jl")
include("spirv.jl")
include("bpf.jl")
if VERSION >= v"1.8-"
    include("gcn.jl")
    include("metal.jl")
end
include("examples.jl")

haskey(ENV, "CI") && GPUCompiler.timings()

@testset "Disk cache" begin
    @test GPUCompiler.disk_cache == false

    cmd = Base.julia_cmd()
    if Base.JLOptions().project != C_NULL
        cmd = `$cmd --project=$(unsafe_string(Base.JLOptions().project))`
    end

    withenv("JULIA_LOAD_PATH" => "$(get(ENV, "JULIA_LOAD_PATH", "")):$(joinpath(@__DIR__, "CacheEnv"))") do
        @test success(pipeline(`$cmd cache.jl true`, stderr=stderr, stdout=stdout))
        @test success(pipeline(`$cmd cache.jl false`, stderr=stderr, stdout=stdout))
    end
end

end
