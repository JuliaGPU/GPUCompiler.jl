@testset "No-op" begin
    mod = @eval module $(gensym())
        kernel() = 0
    end

    @test @filecheck begin
        check"CHECK-LABEL: julia_kernel_{{[0-9_]*}}:"
        check"CHECK: r0 = 0"
        check"CHECK-NEXT: exit"
        BPF.code_native(mod.kernel, ())
    end
end
@testset "Return argument" begin
    mod = @eval module $(gensym())
        kernel(x) = x
    end

    @test @filecheck begin
        check"CHECK-LABEL: julia_kernel_{{[0-9_]*}}:"
        check"CHECK: r0 = r1"
        check"CHECK-NEXT: exit"
        BPF.code_native(mod.kernel, (UInt64,))
    end
end
@testset "Addition" begin
    mod = @eval module $(gensym())
        kernel(x) = x+1
    end

    @test @filecheck begin
        check"CHECK-LABEL: julia_kernel_{{[0-9_]*}}:"
        check"CHECK: r0 = r1"
        check"CHECK-NEXT: r0 += 1"
        check"CHECK-NEXT: exit"
        BPF.code_native(mod.kernel, (UInt64,))
    end
end
@testset "Errors" begin
    mod = @eval module $(gensym())
        kernel(x) = fakefunc(x)
    end

    @test_throws GPUCompiler.InvalidIRError BPF.code_execution(mod.kernel, (UInt64,))
end
@testset "Function Pointers" begin
    @testset "valid" begin
        mod = @eval module $(gensym())
            goodcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
            kernel(x) = goodcall(x)
        end

        @test @filecheck begin
            check"CHECK-LABEL: julia_kernel_{{[0-9_]*}}:"
            check"CHECK: call"
            check"CHECK-NEXT: exit"
            BPF.code_native(mod.kernel, (Int,))
        end
    end

    @testset "invalid" begin
        mod = @eval module $(gensym())
            badcall(x) = Base.llvmcall("%2 = call i64 inttoptr (i64 3000 to i64 (i64)*)(i64 %0)\nret i64 %2", Int, Tuple{Int}, x)
            kernel(x) = badcall(x)
        end

        @test_throws GPUCompiler.InvalidIRError BPF.code_execution(mod.kernel, (Int,))
    end
end
