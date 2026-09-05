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

@testset "exceptions without a device heap" begin
    mod = @eval module $(gensym())
        function kernel(out, x)
            unsafe_store!(out, exponent(x))
            return
        end
    end
    for T in (Float32, Float64)
        source = methodinstance(typeof(mod.kernel), Tuple{Core.LLVMPtr{Int,1},T},
                                Base.get_world_counter())
        target = SPIRVCompilerTarget(; backend, validate=true)
        job = CompilerJob(source, CompilerConfig(target, NoHeapCompilerParams(); kernel=true))
        # exponent's DomainError paths require boxing. Validate the final module,
        # including runtime linkage, even though valid inputs do not allocate (#906).
        JuliaContext() do ctx
            code, _ = GPUCompiler.compile(:asm, job)
            @test !isempty(code)
        end
    end
end

@testset "exception strings" begin
    # the exception name and backtrace strings are globals in the cross-workgroup address
    # space, so the reporting runtime should accept them there without a cast.
    mod = @eval module $(gensym())
        kernel() = throw(DivideError())
    end
    # Keep the IR unoptimized because the test runtime ignores and otherwise drops the strings.
    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
        @check "gpu_report_exception_name("
        @check_same cond=opaque_ptrs "ptr addrspace(1) @exception"
        @check_same cond=typed_ptrs "i8 addrspace(1)* getelementptr inbounds ({{.*}} @exception"
        @check "gpu_report_exception_frame(i32 1,"
        @check_same cond=opaque_ptrs "ptr addrspace(1) @di_func"
        @check_same cond=opaque_ptrs "ptr addrspace(1) @di_file"
        @check_same cond=typed_ptrs "i8 addrspace(1)* getelementptr inbounds ({{.*}} @di_func"
        @check_same cond=typed_ptrs "i8 addrspace(1)* getelementptr inbounds ({{.*}} @di_file"
        SPIRV.code_llvm(mod.kernel, Tuple{}; backend, kernel=true, debug_level=2, optimize=false)
    end

    # Exercise translation too, and ensure this does not introduce generic pointers.
    @test @filecheck begin
        @check_not "OpCapability GenericPointer"
        @check "OpEntryPoint Kernel"
        @check_not "OpPtrCastToGeneric"
        SPIRV.code_native(mod.kernel, Tuple{}; backend, kernel=true,
                          debug_level=2, optimize=false)
    end
end

@testset "failed runtime boxing" begin
    mod = @eval module $(gensym())
        import ..GPUCompiler

        function kernel(x::Float32)
            throw(GPUCompiler.Runtime.box_float32(x))
        end
    end

    # Call the helper directly because Julia's lowering of boxed exception fields varies by
    # version. The allocator-less SPIR-V runtime should take a real OOM path before the stores.
    @test @filecheck begin
        @check_label "define spir_kernel void @_Z6kernel"
        @check "gpu_malloc"
        @check "icmp eq"
        @check "br i1"
        @check "gpu_report_oom"
        @check "gpu_signal_exception"
        @check "store"
        @check "store"
        @check "gpu_report_exception"
        @check "gpu_signal_exception"
        SPIRV.code_llvm(mod.kernel, Tuple{Float32}; backend, kernel=true, dump_module=true)
    end
end

@testset "baked boxed relocation cleanup" begin
    mod = @eval module $(gensym())
        @noinline produce(cond::Bool, value::Int32) = cond ? value : 1.5
        function kernel(out::Core.LLVMPtr{UInt,1}, cond::Bool, value::Int32)
            x = produce(cond, value)
            Base.unsafe_store!(out, UInt(x isa Float64))
            return
        end
    end

    # Baking an interior relocation can expose a dead pointer component of an isbits-union
    # result. It must be folded before SPIR-V translation, which otherwise emits a reference
    # to the now-unused box without defining it.
    _, meta = SPIRV.code_execution(
        mod.kernel, (Core.LLVMPtr{UInt,1}, Bool, Int32); backend)
    @test all(!endswith(LLVM.name(gv), "_box") for gv in globals(meta.ir))
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
        @check "OpEntryPoint Kernel %[[KERNEL:[^ ]+]] \"_Z6kernel4Bool\""
        @check "%[[KERNEL]] = OpFunction %void None"
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
        kernel(x, y) = @noinline fadd(x, y)
    end

    @test @filecheck begin
        # TODO: should structs of `NTuple{VecElement{T}}` be passed by value instead of sret?
        @check_not "i128"
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check cond=(v"1.12" <= VERSION < v"1.12.5") "alloca <2 x i64>, align 16"
        @check cond=(VERSION >= v"1.12.5") "alloca [2 x i64], align 16"
        SPIRV.code_llvm(mod.kernel, Tuple{mod.Vec{4, Float32}, mod.Vec{4, Float32}};
                        backend, dump_module=true)
    end

    @test @filecheck begin
        @check_not "i128"
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check cond=(v"1.12" <= VERSION < v"1.12.5") "alloca [2 x <2 x i64>], align 16"
        @check cond=(VERSION >= v"1.12.5") "alloca [4 x i64], align 16"
        SPIRV.code_llvm(mod.kernel, Tuple{mod.Vec{8, Float32}, mod.Vec{8, Float32}};
                        backend, dump_module=true)
    end
end

end
