@testset "IR" begin

@testset "kernel functions" begin
@testset "byref aggregates" begin
    kernel(x) = return

    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}({{(\{ i64 \}|\[1 x i64\])}}*"
        check"OPAQUE: @{{.*kernel.*}}(ptr"
        Metal.code_llvm(kernel, Tuple{Tuple{Int}})
    end

    # for kernels, every pointer argument needs to take an address space
    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}({{(\{ i64 \}|\[1 x i64\])}} addrspace(1)*"
        check"OPAQUE: @{{.*kernel.*}}(ptr addrspace(1)"
        Metal.code_llvm(kernel, Tuple{Tuple{Int}}; kernel=true)
    end
end

@testset "byref primitives" begin
    kernel(x) = return

    @test @filecheck begin
        check"CHECK: @{{.*kernel.*}}(i64 "
        Metal.code_llvm(kernel, Tuple{Int})
    end

    # for kernels, every pointer argument needs to take an address space
    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}(i64 addrspace(1)*"
        check"OPAQUE: @{{.*kernel.*}}(ptr addrspace(1)"
        Metal.code_llvm(kernel, Tuple{Int}; kernel=true)
    end
end

@testset "module metadata" begin
    kernel() = return

    @test @filecheck begin
        check"CHECK: air.version"
        check"CHECK: air.language_version"
        check"CHECK: air.max_device_buffers"
        Metal.code_llvm(kernel, Tuple{}; dump_module=true, kernel=true)
    end
end

@testset "argument metadata" begin
    kernel(x) = return

    @test @filecheck begin
        check"CHECK: air.buffer"
        Metal.code_llvm(kernel, Tuple{Int}; dump_module=true, kernel=true)
    end

    # XXX: perform more exhaustive testing of argument passing metadata here,
    #      or just defer to execution testing in Metal.jl?
end

@testset "input arguments" begin
    function kernel(ptr)
        idx = ccall("extern julia.air.thread_position_in_threadgroup.i32", llvmcall, UInt32, ()) + 1
        unsafe_store!(ptr, 42, idx)
        return
    end

    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}({{.*}} addrspace(1)* %{{.+}})"
        check"OPAQUE: @{{.*kernel.*}}(ptr addrspace(1) %{{.+}})"
        check"CHECK: call i32 @julia.air.thread_position_in_threadgroup.i32"
        Metal.code_llvm(kernel, Tuple{Core.LLVMPtr{Int,1}})
    end

    @test @filecheck begin
        check"TYPED: @{{.*kernel.*}}({{.*}} addrspace(1)* %{{.+}}, i32 %thread_position_in_threadgroup)"
        check"OPAQUE: @{{.*kernel.*}}(ptr addrspace(1) %{{.+}}, i32 %thread_position_in_threadgroup)"
        check"CHECK-NOT: call i32 @julia.air.thread_position_in_threadgroup.i32"
        Metal.code_llvm(kernel, Tuple{Core.LLVMPtr{Int,1}}; kernel=true)
    end
end

@testset "vector intrinsics" begin
    foo(x, y) = ccall("llvm.smax.v2i64", llvmcall, NTuple{2, VecElement{Int64}},
                      (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}}), x, y)

    @test @filecheck begin
        check"CHECK: air.max.s.v2i64"
        Metal.code_llvm(foo, (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}}))
    end
end

@testset "unsupported type detection" begin
    function kernel1(ptr)
        buf = reinterpret(Ptr{Float32}, ptr)
        val = unsafe_load(buf)
        dval = Cdouble(val)
        # ccall("extern metal_os_log", llvmcall, Nothing, (Float64,), dval)
        Base.llvmcall(("""
        declare void @llvm.va_start(i8*)
        declare void @llvm.va_end(i8*)
        declare void @air.os_log(i8*, i64)

        define void @metal_os_log(...) {
            %1 = alloca i8*
            %2 = bitcast i8** %1 to i8*
            call void @llvm.va_start(i8* %2)
            %3 = load i8*, i8** %1
            call void @air.os_log(i8* %3, i64 8)
            call void @llvm.va_end(i8* %2)
            ret void
        }

        define void @entry(double %val) #0 {
            call void (...) @metal_os_log(double %val)
            ret void
        }

        attributes #0 = { alwaysinline }""", "entry"),
        Nothing, Tuple{Float64}, dval)
        return
    end

    @test @filecheck begin
        check"CHECK: @metal_os_log"
        Metal.code_llvm(kernel1, Tuple{Core.LLVMPtr{Float32,1}}; validate=true)
    end

    function kernel2(ptr)
        val = unsafe_load(ptr)
        res = val * val
        unsafe_store!(ptr, res)
        return
    end

    @test_throws_message(InvalidIRError, Metal.code_execution(kernel2, Tuple{Core.LLVMPtr{Float64,1}})) do msg
        occursin("unsupported use of double value", msg)
    end
end

@testset "constant globals" begin
    mod = @eval module $(gensym())
        const xs = (1.0f0, 2f0)

        function kernel(ptr, i)
            unsafe_store!(ptr, xs[i])

            return
        end
    end

    @test @filecheck begin
        check"CHECK: addrspace(2) constant [2 x float]"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Float32,1}, Int}; dump_module=true, kernel=true)
    end
end

end

end
