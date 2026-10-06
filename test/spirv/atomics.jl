@testset "extensions" begin
    float_add = "SPV_EXT_shader_atomic_float_add"
    float16_add = "SPV_EXT_shader_atomic_float16_add"
    extensions(; extensions="", kwargs...) =
        GPUCompiler.spirv_extensions(SPIRVCompilerTarget(; extensions,
                                                         atomics=SPIRVAtomics(; kwargs...)))

    # the extensions the atomics need are added to the user's
    @test extensions() == ""
    @test extensions(; extensions="+SPV_KHR_expect_assume") == "+SPV_KHR_expect_assume"
    @test extensions(; fadd_f32_global=true) == "+$float_add"
    @test extensions(; fadd_f64_local=true, extensions="+SPV_KHR_expect_assume") ==
          "+SPV_KHR_expect_assume,+$float_add"
    # (the half-precision extension extends the single and double-precision one)
    @test extensions(; fadd_f16_local=true) == "+$float_add,+$float16_add"

    # unless the user already enabled them
    @test extensions(; fadd_f32_global=true, extensions="+$float_add") == "+$float_add"
    @test extensions(; fadd_f16_global=true, extensions="+all") == "+all"
    @test extensions(; fadd_f16_global=true, extensions="-all,+$float_add,+$float16_add") ==
          "-all,+$float_add,+$float16_add"

    # disabling them is an error
    @test_throws "$float_add, which the atomics" extensions(; fadd_f32_global=true,
                                                            extensions="-$float_add")
    @test_throws "$float_add, which the atomics" extensions(; fadd_f32_global=true,
                                                            extensions="+$float_add,-all")
    @test extensions(; fadd_f32_global=true, extensions="-all,+$float_add") ==
          "-all,+$float_add"
    @test_throws "$float16_add, which the atomics" extensions(; fadd_f16_global=true,
                                                              extensions="-all,+$float_add")
    @test extensions(; extensions="-$float_add") == "-$float_add"

    # which is reported when compiling, before translation
    mod = @eval module $(gensym())
        kernel() = return
    end
    @test_throws "which the atomics" SPIRV.code_llvm(devnull, mod.kernel, Tuple{};
        backend=:khronos, extensions="-$float_add",
        atomics=SPIRVAtomics(; fadd_f32_global=true))
end

for backend in (:khronos, :llvm)
@testset "$backend" begin

@testset "extension declarations" begin
    # the back-ends only declare the extensions a module uses, so enabling them for the
    # device's atomics doesn't affect modules without atomics
    mod = @eval module $(gensym())
        kernel(p::Core.LLVMPtr{Float32,1}) = (unsafe_store!(p, 1f0); return)
    end
    @test @filecheck begin
        @check_not "OpExtension"
        @check "OpEntryPoint"
        SPIRV.code_native(mod.kernel, Tuple{Core.LLVMPtr{Float32,1}}; backend, kernel=true,
                          atomics=SPIRVAtomics(; fadd_f16_global=true, fadd_f32_global=true))
    end
end

end
end
