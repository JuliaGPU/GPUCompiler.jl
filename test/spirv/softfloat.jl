@testset "binary64 emulation in SPIR-V" begin
    mod = @eval module $(gensym())
        function kernel(out::Core.LLVMPtr{Float64,1}, a::Float64, b::Float64)
            unsafe_store!(out, a + b, 1)
            unsafe_store!(out, a * b, 2)
            unsafe_store!(out, a / b, 3)
            unsafe_store!(out, sqrt(abs(a)), 4)
            unsafe_store!(out, fma(a, b, 1.0), 5)
            unsafe_store!(out, a < b ? a : b, 6)
            nothing
        end
    end
    types = (Core.LLVMPtr{Float64,1}, Float64, Float64)
    # The LLVM backend legalizes the nonstandard integer widths introduced by
    # optimization. The Khronos translator currently rejects e.g. i11 here.
    job, _ = SPIRV.create_job(mod.kernel, types; backend=:llvm, kernel=true,
                             supports_fp64=false, emulate_fp64=true)
    assembly = GPUCompiler.JuliaContext() do ctx
        first(GPUCompiler.compile(:asm, job))
    end
    @test occursin("OpCapability Int64", assembly)
    @test !occursin("OpCapability Float64", assembly)
    @test !occursin(r"OpTypeFloat\s+64", assembly)
    @test occursin("OpIAdd", assembly)
    @test occursin("OpStore", assembly)
    @test !occursin(r"LinkageAttributes[^\n]*gpu_softfloat[^\n]*Import", assembly)

    native_job, _ = SPIRV.create_job(mod.kernel, types; backend=:llvm, kernel=true)
    @test isempty(GPUCompiler.device_library_providers(native_job))
end
