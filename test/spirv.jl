using SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

@testset "SPIR-V" begin

include("definitions/spirv.jl")

############################################################################################

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    kernel() = return

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("spir_kernel", ir)

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("spir_kernel", ir)
end

@testset "byval workaround" begin
    kernel(x) = return

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{Tuple{Int}}))
    @test occursin(r"@.*julia_.+_kernel.+\(({ i64 }|\[1 x i64\])\*", ir)

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{Tuple{Int}}; kernel=true))
    @test occursin(r"@.*julia_.+_kernel.+\({ ({ i64 }|\[1 x i64\]) }\*.+byval", ir)
end
end

end

############################################################################################

@testset "asm" begin

@testset "trap removal" begin
    function kernel(x)
        x && error()
        return
    end

    asm = sprint(io->spirv_code_native(io, kernel, Tuple{Bool}; kernel=true))
    @test occursin(r"OpFunctionCall %void %julia_error", asm)
end

end

############################################################################################

end
