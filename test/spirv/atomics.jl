# a job for a SPIR-V target with the given atomics, to process textual IR with
function atomics_job(; backend=:llvm, atomics=SPIRVAtomics(), validate=true)
    source = methodinstance(typeof(identity), Tuple{Int}, Base.get_world_counter())
    target = SPIRVCompilerTarget(; backend, atomics, validate=true)
    config = CompilerConfig(target, SPIRV.CompilerParams(); kernel=true, validate)
    CompilerJob(source, config)
end

# a kernel with global, local, generic, private and constant pointers
atomics_kernel(body) = """
    target datalayout = "$(GPUCompiler.llvm_datalayout(SPIRVCompilerTarget()))"
    define spir_kernel void @kernel(ptr addrspace(1) %g, ptr addrspace(3) %l, ptr %p,
                                    ptr addrspace(2) %c) {
      %a = addrspacecast ptr addrspace(1) %g to ptr addrspace(4)
    $body
      ret void
    }
    """

# the reasons `validate_ir` gives for rejecting the atomics in `body`
function atomics_errors(body; kwargs...)
    job = atomics_job(; kwargs...)
    Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, atomics_kernel(body))
        map(first, GPUCompiler.validate_ir(job, mod))
    end
end

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

@testset "validation" begin
    # supported operations
    for body in ["%r = atomicrmw add ptr addrspace(1) %g, i32 1 monotonic, align 4",
                 "%r = atomicrmw umax ptr addrspace(3) %l, i64 1 syncscope(\"workgroup\") acq_rel, align 8",
                 "%r = atomicrmw nand ptr addrspace(4) %a, i32 1 syncscope(\"device\") seq_cst, align 4",
                 "%r = atomicrmw fadd ptr addrspace(1) %g, float 1.0 syncscope(\"device-mem-global\") release, align 4",
                 "%r = atomicrmw fmax ptr addrspace(1) %g, double 1.0 monotonic, align 8",
                 "%r = atomicrmw xchg ptr addrspace(1) %g, ptr null syncscope(\"subgroup\") monotonic, align 8",
                 "%r = cmpxchg weak ptr addrspace(1) %g, i64 0, i64 1 syncscope(\"singlethread\") release acquire, align 8",
                 "%r = cmpxchg ptr addrspace(3) %l, ptr null, ptr null monotonic seq_cst, align 8",
                 "%r = load atomic double, ptr addrspace(1) %g syncscope(\"system\") acquire, align 8",
                 # (atomics on the thread's own memory are demoted)
                 "%s = alloca half\n  store atomic half 0xH0000, ptr %s monotonic, align 2",
                 "fence syncscope(\"workgroup-mem-local+image\") release",
                 "fence syncscope(\"singlethread\") acquire"]
        @test atomics_errors(body) == []
    end
    # (with a native half-precision addition)
    @test atomics_errors("%r = atomicrmw fadd ptr addrspace(1) %g, half 0xH3C00 monotonic, align 2";
                         atomics=SPIRVAtomics(; fadd_f16_global=true)) == []
    # (a native double-precision addition doesn't need 64-bit integer atomics)
    @test atomics_errors("%r = atomicrmw fsub ptr addrspace(3) %l, double 1.0 monotonic, align 8";
                         atomics=SPIRVAtomics(; int64=false, fadd_f64_local=true)) == []

    # unsupported operations
    for (body, reason) in [
        "%r = atomicrmw add ptr addrspace(1) %g, i8 1 monotonic, align 1" =>
            "8-bit atomic operation",
        "%r = atomicrmw xchg ptr addrspace(1) %g, i16 1 monotonic, align 2" =>
            "16-bit atomic operation",
        "%r = cmpxchg ptr addrspace(1) %g, i16 0, i16 1 monotonic monotonic, align 2" =>
            "16-bit atomic operation",
        "%r = atomicrmw fadd ptr addrspace(1) %g, bfloat 1.0 monotonic, align 2" =>
            "16-bit atomic operation",
        "%r = atomicrmw xchg ptr addrspace(1) %g, half 0xH3C00 monotonic, align 2" =>
            "half-precision atomic operation",
        "%r = load atomic half, ptr addrspace(1) %g monotonic, align 2" =>
            "half-precision atomic operation",
        "%r = atomicrmw fadd ptr addrspace(1) %g, half 0xH3C00 monotonic, align 2" =>
            "half-precision atomic operation",
        "%r = atomicrmw fmin ptr addrspace(1) %g, half 0xH3C00 monotonic, align 2" =>
            "half-precision atomic operation",
        "%r = atomicrmw add ptr addrspace(1) %g, i128 1 monotonic, align 16" =>
            "atomic operation on a i128 value",
        "%r = atomicrmw add ptr %p, i32 1 monotonic, align 4" =>
            "atomic operation in address space 0",
        "%r = load atomic i32, ptr addrspace(2) %c monotonic, align 4" =>
            "atomic operation in address space 2",
        "%r = atomicrmw add ptr addrspace(1) %g, i32 1 syncscope(\"agent\") monotonic, align 4" =>
            "atomic operation with synchronization scope \"agent\"",
        "%r = atomicrmw add ptr addrspace(1) %g, i32 1 syncscope(\"device-mem-imageblock\") release, align 4" =>
            "atomic operation with synchronization scope \"device-mem-imageblock\"",
        "%r = atomicrmw add ptr addrspace(1) %g, i32 1 syncscope(\"device-mem-local+global\") release, align 4" =>
            "atomic operation with synchronization scope \"device-mem-local+global\"",
        "fence syncscope(\"agent\") seq_cst" =>
            "fence with synchronization scope \"agent\"",
        "%r = atomicrmw add ptr addrspace(1) %g, i32 1 monotonic, align 2" =>
            "misaligned atomic operation",
        "%r = atomicrmw volatile add ptr addrspace(1) %g, i32 1 monotonic, align 4" =>
            "volatile atomic operation"]
        # (bfloat values are rejected separately)
        @test any(startswith(reason), atomics_errors(body))
    end
    if LLVM.version() >= v"17"
        @test atomics_errors("%r = atomicrmw fadd ptr addrspace(1) %g, <2 x float> zeroinitializer monotonic, align 8") ==
              ["atomic operation on a <2 x float> value"]
    end

    # without 64-bit integer atomics, only a native double-precision addition remains
    atomics = SPIRVAtomics(; int64=false, fadd_f64_global=true)
    @test atomics_errors("%r = atomicrmw fadd ptr addrspace(1) %g, double 1.0 monotonic, align 8";
                         atomics) == []
    for body in ["%r = atomicrmw add ptr addrspace(1) %g, i64 1 monotonic, align 8",
                 "%r = cmpxchg ptr addrspace(3) %l, i64 0, i64 1 monotonic monotonic, align 8",
                 "%r = atomicrmw xchg ptr addrspace(1) %g, ptr null monotonic, align 8",
                 "%r = load atomic double, ptr addrspace(1) %g monotonic, align 8",
                 "%r = atomicrmw fmax ptr addrspace(1) %g, double 1.0 monotonic, align 8",
                 # (a generic pointer needs the native addition for both global and local memory)
                 "%r = atomicrmw fadd ptr addrspace(4) %a, double 1.0 monotonic, align 8"]
        @test atomics_errors(body; atomics) ==
              ["64-bit atomic operation (the target does not support 64-bit integer atomics)"]
    end
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
