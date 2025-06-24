for backend in (:khronos, :llvm)

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        check"CHECK-NOT: spir_kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, dump_module=true)
    end

    @test @filecheck begin
        check"CHECK: spir_kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, dump_module=true, kernel=true)
    end
end

@testset "byval workaround" begin
    mod = @eval module $(gensym())
        kernel(x) = return
    end

    @test @filecheck begin
        check"CHECK-LABEL: define void @{{.*kernel.*}}("
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define spir_kernel void @{{.*kernel.*}}("
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend, kernel=true)
    end
end

@testset "byval bug" begin
    # byval added alwaysinline, which could conflict with noinline and fail verification
    mod = @eval module $(gensym())
        @noinline kernel() = return
    end
    SPIRV.code_llvm(devnull, mod.kernel, Tuple{}; backend, kernel=true)
    @test "We did not crash!" != ""
end
end

@testset "unsupported type detection" begin
    mod = @eval module $(gensym())
        function kernel(ptr, val)
            unsafe_store!(ptr, val)
            return
        end
    end

    @test @filecheck begin
        check"CHECK-LABEL: define {{.*}} @{{.*kernel.*}}("
        check"CHECK: store half"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float16}, Float16}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define {{.*}} @{{.*kernel.*}}("
        check"CHECK: store float"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float32}, Float32}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define {{.*}} @{{.*kernel.*}}("
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
    mod = @eval module $(gensym())
        function kernel(x)
            x && error()
            return
        end
    end

    @test @filecheck begin
        check"CHECK: {{.*kernel.*}}"
        SPIRV.code_native(mod.kernel, Tuple{Bool}; backend, kernel=true)
    end
end

end

end
