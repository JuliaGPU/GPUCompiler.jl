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
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define spir_kernel void @_Z6kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend, kernel=true)
    end
end

@testset "byval bug" begin
    # byval added alwaysinline, which could conflict with noinline and fail verification
    mod = @eval module $(gensym())
        @noinline kernel() = return
    end
    @test @filecheck begin
        check"CHECK-LABEL: define spir_kernel void @_Z6kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, kernel=true)
    end
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
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        check"CHECK: store half"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float16}, Float16}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        check"CHECK: store float"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float32}, Float32}; backend)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        check"CHECK: store double"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float64}, Float64}; backend)
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_execution(mod.kernel, Tuple{Ptr{Float16}, Float16};
                                              backend, supports_fp16=false)) do msg
        occursin("unsupported use of half value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin(r"\[\d+\] kernel", msg)
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_execution(mod.kernel, Tuple{Ptr{Float64}, Float64};
                                              backend, supports_fp64=false)) do msg
        occursin("unsupported use of double value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin(r"\[\d+\] kernel", msg)
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
        check"CHECK: %_Z6kernel4Bool = OpFunction %void None"
        SPIRV.code_native(mod.kernel, Tuple{Bool}; backend, kernel=true)
    end
end

end

@testset "replace i128 allocas" begin
    mod = @eval module $(gensym())
        # reimplement some of SIMD.jl
        struct Vec{N, T}
            data::NTuple{N, Core.VecElement{T}}
        end
        @generated function fadd(x::Vec{N, Float32}, y::Vec{N, Float32}) where {N}
            quote
                Vec(Base.llvmcall($"""
                    %ret = fadd <$N x float> %0, %1
                    ret <$N x float> %ret
                """, NTuple{N, Core.VecElement{Float32}}, NTuple{2, NTuple{N, Core.VecElement{Float32}}}, x.data, y.data))
            end
        end
        kernel(x...) = @noinline fadd(x...)
    end

    @test @filecheck begin
        # TODO: should structs of `NTuple{VecElement{T}}` be passed by value instead of sret?
        check"CHECK-NOT: i128"
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        @static VERSION >= v"1.12" && check"CHECK: alloca <2 x i64>, align 16"
        SPIRV.code_llvm(mod.kernel, NTuple{2, mod.Vec{4, Float32}}; backend, dump_module=true)
    end

    @test @filecheck begin
        check"CHECK-NOT: i128"
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        @static VERSION >= v"1.12" && check"CHECK: alloca [2 x <2 x i64>], align 16"
        SPIRV.code_llvm(mod.kernel, NTuple{2, mod.Vec{8, Float32}}; backend, dump_module=true)
    end
end

end
