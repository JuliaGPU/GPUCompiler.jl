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
end

end

############################################################################################

end
