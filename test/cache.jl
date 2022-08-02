using GPUCompiler
using Test

const TOTAL_KERNELS = 1

clear = parse(Bool, ARGS[1])

@test GPUCompiler.disk_cache == true

if clear
    GPUCompiler.clear_disk_cache!()
    @test length(readdir(GPUCompiler.cache_path())) == 0
else
    @test length(readdir(GPUCompiler.cache_path())) == TOTAL_KERNELS
end

using LLVM, LLVM.Interop

include("util.jl")
include("definitions/native.jl")

kernel() = return

const runtime_cache = Dict{UInt, Any}()

function compiler(job)
    return GPUCompiler.compile(:asm, job)
end

function linker(job, asm)
    asm
end

let (job, kwargs) = native_job(kernel, Tuple{})
    GPUCompiler.cached_compilation(runtime_cache, job, compiler, linker)
end

@test length(readdir(GPUCompiler.cache_path())) == TOTAL_KERNELS