@testitem "BPF" setup=[BPF, Helpers] begin

############################################################################################

@testset "No-op" begin
    kernel() = 0

    output = sprint(io->BPF.code_native(io, kernel, ()))
    @test occursin("\tr0 = 0\n\texit", output)
end
@testset "Return argument" begin
    kernel(x) = x

    output = sprint(io->BPF.code_native(io, kernel, (UInt64,)))
    @test occursin("\tr0 = r1\n\texit", output)
end
@testset "Addition" begin
    kernel(x) = x+1

    output = sprint(io->BPF.code_native(io, kernel, (UInt64,)))
    @test occursin("\tr0 = r1\n\tr0 += 1\n\texit", output)
end
@testset "Errors" begin
    kernel(x) = fakefunc(x)

    @test_throws GPUCompiler.InvalidIRError BPF.code_execution(kernel, (UInt64,))
end
@testset "Function Pointers" begin
    @testset "valid" begin
        goodcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = goodcall(x)

        output = sprint(io->BPF.code_native(io, kernel, (Int,)))
        @test occursin(r"\tcall .*\n\texit", output)
    end
    @testset "invalid" begin
        badcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3000 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = badcall(x)

        @test_throws GPUCompiler.InvalidIRError BPF.code_execution(kernel, (Int,))
    end
end

end
