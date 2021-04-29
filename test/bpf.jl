@testset "eBPF" begin

include("definitions/bpf.jl")

############################################################################################

@testset "No-op" begin
    kernel() = 0

    output = bpf_code_native(kernel, ())[1]
    @test occursin("\tr0 = 0\n\texit", output)
end
@testset "Return argument" begin
    kernel(x) = x

    output = bpf_code_native(kernel, (UInt64,); strip=true)[1]
    @test occursin("\tr0 = r1\n\texit", output)
end
@testset "Addition" begin
    kernel(x) = x+1

    output = bpf_code_native(kernel, (UInt64,); strip=true)[1]
    @test occursin("\tr0 = r1\n\tr0 += 1\n\texit", output)
end
@testset "Errors" begin
    kernel(x) = fakefunc(x)

    @test_throws GPUCompiler.InvalidIRError bpf_code_native(kernel, (UInt64,); strip=true)[1]
end
@testset "Function Pointers" begin
    @testset "valid" begin
        goodcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = goodcall(x)

        output = bpf_code_native(kernel, (Int,); strip=true)[1]
        @test occursin("\tcall 3\n\texit", output)
    end
    @testset "invalid" begin
        badcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3000 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = badcall(x)

        @test_throws GPUCompiler.InvalidIRError bpf_code_native(kernel, (Int,); strip=true)[1]
    end
end

end
