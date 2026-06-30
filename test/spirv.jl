for backend in (:khronos, :llvm)

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "spir_kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, dump_module=true)
    end

    @test @filecheck begin
        @check "spir_kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, dump_module=true, kernel=true)
    end
end

@testset "byval workaround" begin
    mod = @eval module $(gensym())
        kernel(x) = return
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend)
    end

    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
        SPIRV.code_llvm(mod.kernel, Tuple{Tuple{Int}}; backend, kernel=true)
    end
end

@testset "byval bug" begin
    # byval added alwaysinline, which could conflict with noinline and fail verification
    mod = @eval module $(gensym())
        @noinline kernel() = return
    end
    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
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
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "store half"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float16}, Float16}; backend)
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "store float"
        SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Float32}, Float32}; backend)
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "store double"
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

    @static if isdefined(Core, :BFloat16)
        @test @filecheck begin
            @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
            @check "store bfloat"
            SPIRV.code_llvm(mod.kernel, Tuple{Ptr{Core.BFloat16}, Core.BFloat16};
                            backend, supports_bfloat16=true)
        end

        @test_throws_message(InvalidIRError,
                             SPIRV.code_execution(mod.kernel, Tuple{Ptr{Core.BFloat16}, Core.BFloat16};
                                                  backend, supports_bfloat16=false)) do msg
            occursin("unsupported use of bfloat value", msg) &&
            occursin("[1] unsafe_store!", msg) &&
            occursin(r"\[\d+\] kernel", msg)
        end
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

    # at the IR level, `lower_unreachable_control_flow!` must have stripped the device-side
    # `llvm.trap` and lowered the throw's `unreachable` into a clean `ret`.
    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
        @check_not "llvm.trap"
        @check_not "unreachable"
        @check "ret void"
        SPIRV.code_llvm(mod.kernel, Tuple{Bool}; backend, kernel=true)
    end

    # and at the SPIR-V level, no `OpUnreachable` (UB if reached) should survive.
    @test @filecheck begin
        @check "%_Z6kernel4Bool = OpFunction %void None"
        @check_not "OpUnreachable"
        SPIRV.code_native(mod.kernel, Tuple{Bool}; backend, kernel=true)
    end
end

@testset "inlining of throwing callees" begin
    mod = @eval module $(gensym())
        @noinline function guard(x)
            x || error()
            return
        end
        function kernel(x)
            guard(x)
            return
        end
    end

    # `guard` throws on one path and returns on the other; rewriting its `unreachable` into a
    # `ret` is only sound if `guard` is inlined into the kernel first (otherwise the kernel would
    # resume after the call on the throwing path). even though `guard` is `@noinline`, the lowering
    # must have force-inlined it: the throw's `signal_exception` now lives in the kernel's own body
    # (it would sit in `guard` had it stayed out-of-line), with the trap/unreachable lowered away.
    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
        @check "gpu_signal_exception"
        @check_not "llvm.trap"
        @check_not "unreachable"
        @check "ret void"
        SPIRV.code_llvm(mod.kernel, Tuple{Bool}; backend, kernel=true)
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
        @check_not "i128"
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check cond=(v"1.12" <= VERSION < v"1.12.5") "alloca <2 x i64>, align 16"
        @check cond=(VERSION >= v"1.12.5") "alloca [2 x i64], align 16"
        SPIRV.code_llvm(mod.kernel, NTuple{2, mod.Vec{4, Float32}}; backend, dump_module=true)
    end

    @test @filecheck begin
        @check_not "i128"
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check cond=(v"1.12" <= VERSION < v"1.12.5") "alloca [2 x <2 x i64>], align 16"
        @check cond=(VERSION >= v"1.12.5") "alloca [4 x i64], align 16"
        SPIRV.code_llvm(mod.kernel, NTuple{2, mod.Vec{8, Float32}}; backend, dump_module=true)
    end
end

@testset "stack allocation intrinsic" begin
    mod = @eval module $(gensym())
        import ..GPUCompiler

        function scratch(x)
            p = GPUCompiler.alloca(Float32, Val(8))
            @inbounds unsafe_store!(p, x, 1)
            @inbounds unsafe_store!(p, x, 8)
            return @inbounds unsafe_load(p, 1) + unsafe_load(p, 8)
        end

        # zero-element scratch yields a (null) pointer without emitting an alloca
        empty_scratch() = GPUCompiler.alloca(Float32, Val(0)) === reinterpret(Ptr{Float32}, C_NULL)
    end

    # the intrinsic is materialized as a single entry-block `alloca [32 x i8]`,
    # and no `julia.gpu.alloca` call/declaration survives lowering.
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check "alloca [32 x i8], align 4"
        @check_not "julia.gpu.alloca"
        SPIRV.code_llvm(mod.scratch, Tuple{Float32}; backend, optimize=false)
    end

    # once optimized the slot is promoted away entirely (result is x + x).
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check_not "alloca"
        @check_not "julia.gpu.alloca"
        SPIRV.code_llvm(mod.scratch, Tuple{Float32}; backend)
    end

    # a zero-byte allocation lowers to a null pointer rather than a degenerate alloca.
    @test @filecheck begin
        @check_label "define {{.*}}@{{(julia|j)_empty_scratch_[0-9]+}}"
        @check_not "alloca"
        @check_not "julia.gpu.alloca"
        SPIRV.code_llvm(mod.empty_scratch, Tuple{}; backend)
    end
end

end
