@testset "No-op" begin
    kernel() = 0

    @test @filecheck begin
        check"CHECK: r0 = 0"
        check"CHECK-NEXT: exit"
        BPF.code_native(kernel, ())
    end
end
@testset "Return argument" begin
    kernel(x) = x

    @test @filecheck begin
        check"CHECK: r0 = r1"
        check"CHECK-NEXT: exit"
        BPF.code_native(kernel, (UInt64,))
    end
end
@testset "Addition" begin
    kernel(x) = x+1

    @test @filecheck begin
        check"CHECK: r0 = r1"
        check"CHECK-NEXT: r0 += 1"
        check"CHECK-NEXT: exit"
        BPF.code_native(kernel, (UInt64,))
    end
end
@testset "Errors" begin
    kernel(x) = fakefunc(x)

    @test_throws GPUCompiler.InvalidIRError BPF.code_execution(kernel, (UInt64,))
end
@testset "Function Pointers" begin
    @testset "valid" begin
        goodcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = goodcall(x)

        @test @filecheck begin
            check"CHECK: call"
            check"CHECK-NEXT: exit"
            BPF.code_native(kernel, (Int,))
        end
    end
    @testset "invalid" begin
        badcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3000 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
        kernel(x) = badcall(x)

        @test_throws GPUCompiler.InvalidIRError BPF.code_execution(kernel, (Int,))
    end
end
