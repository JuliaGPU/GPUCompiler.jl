for backend in (:khronos, :llvm)

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    kernel() = return

    @test @filecheck begin
        check"CHECK-NOT: spir_kernel"
        SPIRV.code_llvm(kernel, Tuple{}; backend, dump_module=true)
    end

    @test @filecheck begin
        check"CHECK: spir_kernel"
        SPIRV.code_llvm(kernel, Tuple{}; backend, dump_module=true, kernel=true)
    end
end

@testset "byval workaround" begin
    mod = @eval module $(gensym())
        export kernel
        kernel(x) = return
    end

    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}([1 x i64]*"
        check"OPAQUE: @{{.*kernel.*}}(ptr"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend)
    end

    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}({ [1 x i64] }* byval"
        check"OPAQUE: @{{.*kernel.*}}(ptr byval"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend, kernel=true)
    end
end

@testset "byval bug" begin
    # byval added alwaysinline, which could conflict with noinline and fail verification
    @noinline kernel() = return
    SPIRV.code_llvm(devnull, kernel, Tuple{}; backend, kernel=true)
    @test "We did not crash!" != ""
end
end

@testset "unsupported type detection" begin
    mod = @eval module $(gensym())
        export kernel
        function kernel(ptr, val)
            unsafe_store!(ptr, val)
            return
        end
    end

    @test @filecheck begin
        check"CHECK: store half"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float16}, Float16}; backend)
    end

    @test @filecheck begin
        check"CHECK: store float"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float32}, Float32}; backend)
    end

    @test @filecheck begin
        check"CHECK: store double"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float64}, Float64}; backend)
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_execution(mod.kernel, Tuple{Ptr{Float16}, Float16};
                                              backend, supports_fp16=false)) do msg
        occursin("unsupported use of half value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin("[2] kernel", msg)
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_execution(mod.kernel, Tuple{Ptr{Float64}, Float64};
                                              backend, supports_fp64=false)) do msg
        occursin("unsupported use of double value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin("[2] kernel", msg)
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

    @test @filecheck begin
        check"CHECK: OpFunctionCall %void %{{(julia|j)_error}}"
        SPIRV.code_native(kernel, Tuple{Bool}; backend, kernel=true)
    end
end

end

end
