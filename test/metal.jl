using Metal_LLVM_Tools_jll, LLVM

include("definitions/metal.jl")

@testset "Metal" begin

############################################################################################

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    kernel() = return

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("cc103", ir)

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("cc103", ir)
end

@testset "byref aggregates" begin
    kernel(x) = return

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Tuple{Int}}))
    @test occursin(r"@\w*kernel\w*\(({ i64 }|\[1 x i64\])\*", ir)

    # for kernels, every pointer argument needs to take an address space
    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Tuple{Int}}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(({ i64 }|\[1 x i64\]) addrspace\(1\)\*", ir)
end

@testset "byref primitives" begin
    kernel(x) = return

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Int}))
    @test occursin(r"@\w*kernel\w*\(i64 ", ir)

    # for kernels, every pointer argument needs to take an address space
    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Int}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(i64 addrspace\(1\)\*", ir)
end

@testset "module metadata" begin
    kernel() = return

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("air.version", ir)
    @test occursin("air.language_version", ir)
    @test occursin("air.max_device_buffers", ir)
end

@testset "argument metadata" begin
    kernel(x) = return

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Int};
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

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Core.LLVMPtr{Int,1}}))
    @test occursin(r"@\w*kernel\w*\(.* addrspace\(1\)\* %0\)", ir)
    @test occursin(r"call i32 @julia.air.thread_position_in_threadgroup.i32", ir)

    ir = sprint(io->metal_code_llvm(io, kernel, Tuple{Core.LLVMPtr{Int,1}}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(.* addrspace\(1\)\* %0, i32 %thread_position_in_threadgroup\)", ir)
    @test !occursin(r"call i32 @julia.air.thread_position_in_threadgroup.i32", ir)
end

end

end

############################################################################################

Sys.isapple() && @testset "asm" begin

@testset "smoke test" begin
    kernel() = return

    asm = sprint(io->metal_code_native(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("[header]", asm)
    @test occursin("[program]", asm)
    @test occursin(r"name: \w*kernel\w*", asm)
    @test occursin(r"define void @\w*kernel\w*", asm)
end

end

end
