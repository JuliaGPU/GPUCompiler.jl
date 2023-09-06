@testitem "SPIRV" setup=[SPIRV, Helpers] begin

using SPIRV_LLVM_Translator_unified_jll, SPIRV_Tools_jll

############################################################################################

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    kernel() = return

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("spir_kernel", ir)

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("spir_kernel", ir)
end

@testset "byval workaround" begin
    kernel(x) = return

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{Tuple{Int}}))
    @test occursin(r"@\w*kernel\w*\(({ i64 }|\[1 x i64\])\*", ir)

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{Tuple{Int}}; kernel=true))
    @test occursin(r"@\w*kernel\w*\(.*{ ({ i64 }|\[1 x i64\]) }\*.+byval", ir)
end

@testset "byval bug" begin
    # byval added alwaysinline, which could conflict with noinline and fail verification
    @noinline kernel() = return
    SPIRV.code_llvm(devnull, kernel, Tuple{}; kernel=true)
    @test "We did not crash!" != ""
end
end

@testset "unsupported type detection" begin
    function kernel(ptr, val)
        unsafe_store!(ptr, val)
        return
    end

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{Ptr{Float16}, Float16}; validate=true))
    @test occursin("store half", ir)

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{Ptr{Float32}, Float32}; validate=true))
    @test occursin("store float", ir)

    ir = sprint(io->SPIRV.code_llvm(io, kernel, Tuple{Ptr{Float64}, Float64}; validate=true))
    @test occursin("store double", ir)

    @test_throws_message(InvalidIRError,
                         SPIRV.code_llvm(devnull, kernel, Tuple{Ptr{Float16}, Float16};
                                         supports_fp16=false, validate=true)) do msg
        occursin("unsupported unsupported use of half value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin(r"\[2\] .*kernel", msg)
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_llvm(devnull, kernel, Tuple{Ptr{Float64}, Float64};
                                         supports_fp64=false, validate=true)) do msg
        occursin("unsupported unsupported use of double value", msg) &&
        occursin("[1] unsafe_store!", msg) &&
        occursin(r"\[2\] .*kernel", msg)
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

    asm = sprint(io->SPIRV.code_native(io, kernel, Tuple{Bool}; kernel=true))
    @test occursin(r"OpFunctionCall %void %julia_error", asm)
end

end

############################################################################################

end
