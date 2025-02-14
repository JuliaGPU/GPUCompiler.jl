@testset "IR" begin

@testset "kernel functions" begin
@testset "byref aggregates" begin
    kernel(x) = return

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Tuple{Int}}))
    @test occursin(r"@\w*kernel\w*\(({ i64 }|\[1 x i64\])\*", ir) ||
          occursin(r"@\w*kernel\w*\(ptr", ir)

    # for kernels, every pointer argument needs to take an address space
    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Tuple{Int}}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(({ i64 }|\[1 x i64\]) addrspace\(1\)\*", ir) ||
          occursin(r"@\w*kernel\w*\(ptr addrspace\(1\)", ir)
end

@testset "byref primitives" begin
    kernel(x) = return

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Int}))
    @test occursin(r"@\w*kernel\w*\(i64 ", ir)

    # for kernels, every pointer argument needs to take an address space
    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Int}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(i64 addrspace\(1\)\*", ir) ||
          occursin(r"@\w*kernel\w*\(ptr addrspace\(1\)", ir)
end

@testset "module metadata" begin
    kernel() = return

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("air.version", ir)
    @test occursin("air.language_version", ir)
    @test occursin("air.max_device_buffers", ir)
end

@testset "argument metadata" begin
    kernel(x) = return

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Int};
                                    dump_module=true, kernel=true))
    @test occursin("air.buffer", ir)

    # XXX: perform more exhaustive testing of argument passing metadata here,
    #      or just defer to execution testing in Metal.jl?
end

@testset "input arguments" begin
    function kernel(ptr)
        idx = ccall("extern julia.air.thread_position_in_threadgroup.i32", llvmcall, UInt32, ()) + 1
        unsafe_store!(ptr, 42, idx)
        return
    end

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Core.LLVMPtr{Int,1}}))
    @test occursin(r"@\w*kernel\w*\(.* addrspace\(1\)\* %.+\)", ir) ||
          occursin(r"@\w*kernel\w*\(ptr addrspace\(1\) %.+\)", ir)
    @test occursin(r"call i32 @julia.air.thread_position_in_threadgroup.i32", ir)

    ir = sprint(io->Metal.code_llvm(io, kernel, Tuple{Core.LLVMPtr{Int,1}}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(.* addrspace\(1\)\* %.+, i32 %thread_position_in_threadgroup\)", ir) ||
          occursin(r"@\w*kernel\w*\(ptr addrspace\(1\) %.+, i32 %thread_position_in_threadgroup\)", ir)
    @test !occursin(r"call i32 @julia.air.thread_position_in_threadgroup.i32", ir)
end

@testset "vector intrinsics" begin
    foo(x, y) = ccall("llvm.smax.v2i64", llvmcall, NTuple{2, VecElement{Int64}},
                      (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}}), x, y)

    ir = sprint(io->Metal.code_llvm(io, foo, (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}})))
    @test occursin("air.max.s.v2i64", ir)
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

    ir = sprint(io->Metal.code_llvm(io, kernel1, Tuple{Core.LLVMPtr{Float32,1}}; validate=true))
    @test occursin("@metal_os_log", ir)

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

    ir = sprint(io->Metal.code_llvm(io, mod.kernel, Tuple{Core.LLVMPtr{Float32,1}, Int};
                                    dump_module=true, kernel=true))
    @test occursin("addrspace(2) constant [2 x float]", ir)
end

end

end
