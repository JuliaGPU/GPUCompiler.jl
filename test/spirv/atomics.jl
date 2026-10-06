using .SPIRV: atomics_job, atomics_kernel, atomics_errors, lower_atomics

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
                 "%r = atomicrmw nand ptr addrspace(4) %generic, i32 1 syncscope(\"device\") seq_cst, align 4",
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
        "%r = load atomic i32, ptr addrspace(2) %constant monotonic, align 4" =>
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
                 "%r = atomicrmw fadd ptr addrspace(4) %generic, double 1.0 monotonic, align 8"]
        @test atomics_errors(body; atomics) ==
              ["64-bit atomic operation (the target does not support 64-bit integer atomics)"]
    end
end

# check `output` against FileCheck directives, e.g. "CHECK: OpAtomicIAdd"
filecheck(output, directives...) =
    FileCheck.filecheck(_ -> output, join(directives, "\n"))

# the mangled name of the `__spirv_*` builtin of an atomic operation on a value of LLVM type `T`
# in address space `as`, e.g. `_Z18__spirv_AtomicIAddPU3AS1Vijji`
function builtin(name, T, as; operands=1)
    m = Dict("i32" => "i", "i64" => "l", "half" => "Dh", "float" => "f", "double" => "d")[T]
    name = "__spirv_Atomic$name"
    "_Z$(length(name))$(name)PU3AS$(as)V$(m)jj" * (name == "__spirv_AtomicCompareExchange" ? "j" : "") *
        m^operands
end

const MEMORY = 0x380    # SubgroupMemory | WorkgroupMemory | CrossWorkgroupMemory

@testset "32-bit pointers" begin
    # pointers are as large as the data layout says, so with 32-bit pointers, their atomics
    # are 32-bit integer ones (and need no 64-bit integer atomics)
    datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
    job = atomics_job(; atomics=SPIRVAtomics(; int64=false))
    ir = Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, atomics_kernel("""
            %a = atomicrmw xchg ptr addrspace(1) %g, ptr %p syncscope("device") monotonic, align 4
            %b = cmpxchg ptr addrspace(1) %g, ptr null, ptr %p syncscope("device") monotonic monotonic, align 4
            %c = load atomic ptr, ptr addrspace(1) %g syncscope("device") monotonic, align 4
            """))
        mod.datalayout = datalayout
        @test isempty(GPUCompiler.validate_ir(job, mod))
        GPUCompiler.lower_atomics!(job, mod)
        string(mod)
    end
    @test filecheck(ir,
        "CHECK: [[P:%[0-9]+]] = ptrtoint ptr %p to i32",
        "CHECK: call i32 @$(builtin("Exchange", "i32", 1))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 [[P]])",
        "CHECK: call i32 @$(builtin("CompareExchange", "i32", 1; operands=2))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 $MEMORY, i32 {{%[0-9]+}}, i32 0)",
        "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $MEMORY)")
end

for backend in (:khronos, :llvm)
@testset "$backend" begin

@testset "read-modify-write" begin
    # every integer operation SPIR-V has an instruction for, in every address space
    ops = ["xchg" => "Exchange", "add" => "IAdd", "sub" => "ISub", "and" => "And",
           "or" => "Or", "xor" => "Xor", "max" => "SMax", "min" => "SMin",
           "umax" => "UMax", "umin" => "UMin"]
    for (as, ptr, T) in ((1, "%g", "i32"), (3, "%l", "i64"), (4, "%generic", "i32"))
        align = T == "i32" ? 4 : 8
        ir, asm = lower_atomics(join(["%$op = atomicrmw $op ptr addrspace($as) $ptr, $T 1 syncscope(\"device\") monotonic, align $align"
                                      for (op, _) in ops], "\n"); backend)
        @test filecheck(ir, ["CHECK: call $T @$(builtin(name, T, as))(ptr addrspace($as) $ptr, i32 1, i32 $MEMORY, $T 1)"
                             for (_, name) in ops]...)
        @test filecheck(asm, ["CHECK: OpAtomic$name %u$(T == "i32" ? "int" : "long") {{%.+}} 1 $MEMORY 1"
                              for (_, name) in ops]...)
    end
end

@testset "scopes" begin
    # Invocation is not a valid scope for atomics in the OpenCL environment, so a
    # single-thread atomic on memory other threads can access uses the workgroup scope
    scopes = ["singlethread" => 2, "subgroup" => 3, "workgroup" => 2, "device" => 1,
              "system" => 0, "" => 0]
    syncscope(scope) = isempty(scope) ? "" : "syncscope(\"$scope\") "
    ir, asm = lower_atomics(join(["%r$i = atomicrmw add ptr addrspace(1) %g, i32 1 $(syncscope(scope))monotonic, align 4"
                                  for (i, (scope, _)) in enumerate(scopes)], "\n"); backend)
    @test filecheck(asm, ["CHECK: OpAtomicIAdd %uint %g $id $MEMORY 1" for (_, id) in scopes]...)

    # fences can use it, and also order image memory
    scopes[1] = "singlethread" => 4
    ir, asm = lower_atomics(join(["fence $(syncscope(scope))seq_cst" for (scope, _) in scopes], "\n");
                            backend)
    @test filecheck(ir, ["CHECK: call void @_Z21__spirv_MemoryBarrierjj(i32 $id, i32 $(0x800 | MEMORY | 0x10))"
                         for (_, id) in scopes]...)
    @test filecheck(asm, ["CHECK: OpMemoryBarrier $id $(0x800 | MEMORY | 0x10)" for (_, id) in scopes]...)
end

@testset "orderings" begin
    # the MemorySemantics of an ordered operation orders all memory, whatever the address
    # space it accesses, e.g., so that a release store to a flag in local memory publishes
    # earlier stores to global memory; relaxed operations name the memory too, like DPC++
    orderings = ["monotonic" => 0x0, "acquire" => 0x2, "release" => 0x4, "acq_rel" => 0x8,
                 "seq_cst" => 0x10]
    ir, asm = lower_atomics(join(["%rmw_$order = atomicrmw add ptr addrspace(3) %l, i64 1 syncscope(\"workgroup\") $order, align 8"
                                  for (order, _) in orderings], "\n"); backend)
    @test filecheck(asm, ["CHECK: OpAtomicIAdd %ulong %l 2 $(MEMORY | bits) 1"
                          for (_, bits) in orderings]...)

    ir, asm = lower_atomics("""
        %a = load atomic i32, ptr addrspace(1) %g syncscope("device") monotonic, align 4
        %b = load atomic i32, ptr addrspace(1) %g syncscope("device") acquire, align 4
        %c = load atomic i32, ptr addrspace(3) %l syncscope("workgroup") seq_cst, align 4
        store atomic i32 %a, ptr addrspace(1) %g syncscope("device") monotonic, align 4
        store atomic i32 %b, ptr addrspace(3) %l syncscope("workgroup") release, align 4
        store atomic i32 %c, ptr addrspace(1) %g seq_cst, align 4
        fence syncscope("device") acquire
        fence syncscope("device") release
        fence syncscope("device") acq_rel
        """; backend)
    @test filecheck(ir,
        "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $MEMORY)",
        "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $(MEMORY | 0x2))",
        "CHECK: call i32 @$(builtin("Load", "i32", 3; operands=0))(ptr addrspace(3) %l, i32 2, i32 $(MEMORY | 0x10))",
        "CHECK: call void @$(builtin("Store", "i32", 1))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 %",
        "CHECK: call void @$(builtin("Store", "i32", 3))(ptr addrspace(3) %l, i32 2, i32 $(MEMORY | 0x4), i32 %",
        "CHECK: call void @$(builtin("Store", "i32", 1))(ptr addrspace(1) %g, i32 0, i32 $(MEMORY | 0x10), i32 %")
    @test filecheck(asm,
        "CHECK: OpAtomicLoad %uint %g 1 $MEMORY",
        "CHECK: OpAtomicLoad %uint %g 1 $(MEMORY | 0x2)",
        "CHECK: OpAtomicLoad %uint %l 2 $(MEMORY | 0x10)",
        "CHECK: OpAtomicStore %g 1 $MEMORY",
        "CHECK: OpAtomicStore %l 2 $(MEMORY | 0x4)",
        "CHECK: OpAtomicStore %g 0 $(MEMORY | 0x10)",
        "CHECK: OpMemoryBarrier 1 $(0x800 | MEMORY | 0x2)",
        "CHECK: OpMemoryBarrier 1 $(0x800 | MEMORY | 0x4)",
        "CHECK: OpMemoryBarrier 1 $(0x800 | MEMORY | 0x8)")

    # front-ends can restrict the memory an operation orders in its synchronization scope
    ir, asm = lower_atomics("""
        %a = atomicrmw add ptr addrspace(1) %g, i32 1 syncscope("device-mem-global") release, align 4
        %b = load atomic i32, ptr addrspace(3) %l syncscope("workgroup-mem-local") acquire, align 4
        %c = atomicrmw add ptr addrspace(1) %g, i32 1 syncscope("device-mem-none") monotonic, align 4
        fence syncscope("device-mem-global+image") seq_cst
        fence syncscope("workgroup-mem-local") acq_rel
        """; backend)
    @test filecheck(asm,
        "CHECK: OpAtomicIAdd %uint %g 1 $(0x200 | 0x4) 1",
        "CHECK: OpAtomicLoad %uint %l 2 $(0x100 | 0x2)",
        "CHECK: OpAtomicIAdd %uint %g 1 0 1",
        "CHECK: OpMemoryBarrier 1 $(0x200 | 0x800 | 0x10)",
        "CHECK: OpMemoryBarrier 2 $(0x100 | 0x8)")
end

@testset "compare-exchange" begin
    # SPIR-V requires the failure ordering to be no stronger than the success one, so the
    # success ordering is strengthened to cover it
    orderings = [("monotonic", "monotonic") => (0x0, 0x0),
                 ("acquire", "acquire") => (0x2, 0x2),
                 ("release", "monotonic") => (0x4, 0x0),
                 ("acq_rel", "acquire") => (0x8, 0x2),
                 ("seq_cst", "seq_cst") => (0x10, 0x10),
                 ("monotonic", "acquire") => (0x2, 0x2),
                 ("release", "acquire") => (0x8, 0x2),
                 ("monotonic", "seq_cst") => (0x10, 0x10)]
    ir, asm = lower_atomics(join(["%r$i = cmpxchg ptr addrspace(1) %g, i32 $i, i32 0 syncscope(\"device\") $success $failure, align 4"
                                  for (i, ((success, failure), _)) in enumerate(orderings)], "\n");
                            backend)
    @test filecheck(asm, ["CHECK: OpAtomicCompareExchange %uint %g 1 $(MEMORY | eq) $(MEMORY | neq) 0 $i"
                          for (i, (_, (eq, neq))) in enumerate(orderings)]...)

    # a weak compare-exchange may fail spuriously, so a strong one is fine too. the success
    # flag is derived from the old value.
    ir, asm = lower_atomics("""
        %x = cmpxchg weak ptr addrspace(3) %l, i64 1, i64 2 syncscope("workgroup") acquire monotonic, align 8
        %old = extractvalue { i64, i1 } %x, 0
        %success = extractvalue { i64, i1 } %x, 1
        %y = select i1 %success, i64 %old, i64 0
        store i64 %y, ptr addrspace(3) %l
        """; backend)
    @test filecheck(ir,
        "CHECK: [[OLD:%[0-9a-z]+]] = call i64 @$(builtin("CompareExchange", "i64", 3; operands=2))(ptr addrspace(3) %l, i32 2, i32 $(MEMORY | 0x2), i32 $MEMORY, i64 2, i64 1)",
        "CHECK: icmp eq i64 [[OLD]], 1")
    @test filecheck(asm,
        "CHECK: OpAtomicCompareExchange %ulong %l 2 $(MEMORY | 0x2) $MEMORY 2 1",
        "CHECK-NOT: OpAtomicCompareExchangeWeak")

    # pointers are compared as integers
    ir, asm = lower_atomics("""
        %x = cmpxchg ptr addrspace(1) %g, ptr null, ptr %p syncscope("device") monotonic monotonic, align 8
        %old = extractvalue { ptr, i1 } %x, 0
        store ptr %old, ptr addrspace(1) %g
        """; backend)
    @test filecheck(ir,
        "CHECK: [[NEW:%[0-9a-z]+]] = ptrtoint ptr %p to i64",
        "CHECK: [[OLD:%[0-9a-z]+]] = call i64 @$(builtin("CompareExchange", "i64", 1; operands=2))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 $MEMORY, i64 [[NEW]], i64 0)",
        "CHECK: inttoptr i64 {{%[0-9]+}} to ptr")
    @test filecheck(asm, "CHECK: OpAtomicCompareExchange %ulong")
end

@testset "floating-point and pointer accesses" begin
    # loads, stores and exchanges of floating-point numbers and pointers are integer ones
    ir, asm = lower_atomics("""
        %a = load atomic float, ptr addrspace(1) %g syncscope("device") acquire, align 4
        store atomic double 1.0, ptr addrspace(3) %l syncscope("workgroup") release, align 8
        %b = atomicrmw xchg ptr addrspace(1) %g, float %a syncscope("device") monotonic, align 4
        %c = atomicrmw xchg ptr addrspace(4) %generic, ptr %p syncscope("device") monotonic, align 8
        """; backend)
    @test filecheck(ir,
        "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $(MEMORY | 0x2))",
        "CHECK: call void @$(builtin("Store", "i64", 3))(ptr addrspace(3) %l, i32 2, i32 $(MEMORY | 0x4), i64 4607182418800017408)",
        "CHECK: call i32 @$(builtin("Exchange", "i32", 1))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 %",
        "CHECK: [[P:%[0-9a-z]+]] = ptrtoint ptr %p to i64",
        "CHECK: call i64 @$(builtin("Exchange", "i64", 4))(ptr addrspace(4) %generic, i32 1, i32 $MEMORY, i64 [[P]])")
    @test filecheck(asm,
        "CHECK: OpAtomicLoad %uint %g 1 $(MEMORY | 0x2)",
        "CHECK: OpAtomicStore %l 2 $(MEMORY | 0x4) 4607182418800017408",
        "CHECK: OpAtomicExchange %uint %g 1 $MEMORY",
        "CHECK: OpAtomicExchange %ulong {{%.+}} 1 $MEMORY")
end

@testset "floating-point addition" begin
    body = """
        %a = atomicrmw fadd ptr addrspace(1) %g, float 1.0 syncscope("device") monotonic, align 4
        %b = atomicrmw fsub ptr addrspace(3) %l, float 1.0 syncscope("workgroup") monotonic, align 4
        %c = atomicrmw fadd ptr addrspace(4) %generic, double 1.0 syncscope("device") monotonic, align 8
        """

    # without native instructions, these become compare-exchange loops, which need no
    # extension
    ir, asm = lower_atomics(body; backend)
    @test filecheck(ir,
        "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $MEMORY)",
        "CHECK: atomicrmw.start:",
        "CHECK: fadd float",
        "CHECK: call i32 @$(builtin("CompareExchange", "i32", 1; operands=2))(ptr addrspace(1) %g, i32 1, i32 $MEMORY, i32 $MEMORY,",
        "CHECK: fsub float",
        "CHECK: call i32 @$(builtin("CompareExchange", "i32", 3; operands=2))(ptr addrspace(3) %l, i32 2, i32 $MEMORY, i32 $MEMORY,",
        "CHECK: fadd double",
        "CHECK: call i64 @$(builtin("CompareExchange", "i64", 4; operands=2))(ptr addrspace(4) %generic, i32 1, i32 $MEMORY, i32 $MEMORY,")
    @test filecheck(asm,
        "CHECK-NOT: OpExtension",
        "CHECK-NOT: OpAtomicFAddEXT",
        "CHECK: OpAtomicCompareExchange %uint %g",
        "CHECK: OpAtomicCompareExchange %uint %l",
        "CHECK: OpAtomicCompareExchange %ulong")

    # with them, they are selected for the memory and types the device supports, which for
    # generic pointers means global and local memory
    ir, asm = lower_atomics(body; backend, atomics=SPIRVAtomics(; fadd_f32_global=true,
                                                                fadd_f64_global=true,
                                                                fadd_f64_local=true))
    @test filecheck(asm,
        "CHECK: OpCapability AtomicFloat32AddEXT",
        "CHECK: OpCapability AtomicFloat64AddEXT",
        "CHECK: OpExtension \"SPV_EXT_shader_atomic_float_add\"",
        "CHECK: OpAtomicFAddEXT %float %g 1 $MEMORY %float_1",
        "CHECK-NOT: OpAtomicFAddEXT %float",
        "CHECK: OpAtomicCompareExchange %uint %l",
        "CHECK: OpAtomicFAddEXT %double {{%.+}} 1 $MEMORY %double_1")
    # (subtraction is the addition of the negated value)
    ir, asm = lower_atomics(body; backend, atomics=SPIRVAtomics(; fadd_f32_local=true,
                                                                fadd_f64_global=true))
    @test filecheck(ir,
        "CHECK: call i32 @$(builtin("CompareExchange", "i32", 1; operands=2))",
        "CHECK: call float @$(builtin("FAddEXT", "float", 3))(ptr addrspace(3) %l, i32 2, i32 $MEMORY, float -1.0",
        "CHECK: call i64 @$(builtin("CompareExchange", "i64", 4; operands=2))")
    @test filecheck(asm,
        "CHECK: OpAtomicCompareExchange %uint %g",
        "CHECK: OpAtomicFAddEXT %float %l 2 $MEMORY %float_n1",
        "CHECK-NOT: OpAtomicFAddEXT",
        "CHECK: OpAtomicCompareExchange %ulong")

    # half-precision addition needs its own extension, which extends the other one
    body = """
        %a = atomicrmw fadd ptr addrspace(1) %g, half 0xH3C00 syncscope("device") monotonic, align 2
        %b = atomicrmw fadd ptr addrspace(1) %g, float 1.0 syncscope("device") monotonic, align 4
        """
    ir, asm = lower_atomics(body; backend, atomics=SPIRVAtomics(; fadd_f16_global=true))
    @test filecheck(asm,
        "CHECK: OpCapability AtomicFloat16AddEXT",
        "CHECK: OpExtension \"SPV_EXT_shader_atomic_float16_add\"",
        "CHECK: OpAtomicFAddEXT %half %g 1 $MEMORY",
        "CHECK-NOT: OpAtomicFAddEXT",
        "CHECK: OpAtomicCompareExchange %uint")
end

@testset "compare-exchange loops" begin
    # read-modify-write operations without an instruction (like floating-point min and max,
    # whose native instructions may return either zero) become compare-exchange loops
    Op = LLVM.AtomicRMWBinOp
    ops = ["nand" => Op.Nand, "fmax" => Op.FMax, "fmin" => Op.FMin,
           "uinc_wrap" => Op.UIncWrap, "udec_wrap" => Op.UDecWrap,
           "usub_cond" => Op.USubCond, "usub_sat" => Op.USubSat,
           "fmaximum" => Op.FMaximum, "fminimum" => Op.FMinimum,
           "fmaximumnum" => Op.FMaximumNum, "fminimumnum" => Op.FMinimumNum]
    @testset "$op" for (op, binop) in ops
        # (operations the LLVM in use doesn't support can't occur)
        LLVM.isavailable(binop) || continue
        T, val = startswith(op, "f") ? ("float", "1.0") : ("i32", "1")
        ir, asm = lower_atomics("%r = atomicrmw $op ptr addrspace(1) %g, $T $val syncscope(\"device\") seq_cst, align 4";
                                backend, atomics=SPIRVAtomics(; fadd_f32_global=true))
        @test filecheck(ir,
            "CHECK: call i32 @$(builtin("Load", "i32", 1; operands=0))(ptr addrspace(1) %g, i32 1, i32 $MEMORY)",
            "CHECK: atomicrmw.start:",
            "CHECK: call i32 @$(builtin("CompareExchange", "i32", 1; operands=2))(ptr addrspace(1) %g, i32 1, i32 $(MEMORY | 0x10), i32 $(MEMORY | 0x10),")
        @test filecheck(asm,
            "CHECK-NOT: OpAtomicFM{{in|ax}}EXT",
            "CHECK: OpAtomicCompareExchange %uint %g 1 $(MEMORY | 0x10) $(MEMORY | 0x10)")
        # the loops of floating-point min/max compute it like LLVM, then fix up NaNs and
        # signed zeros (see `lower_minimum_maximum!`)
        if T == "float"
            @test filecheck(asm, "CHECK: OpExtInst %float {{%.+}} $(op[1:4])")
        end
    end
end

@testset "thread-private memory" begin
    # atomics on the thread's own memory become plain accesses
    ir, asm = lower_atomics("""
        %s = alloca i32
        store atomic i32 0, ptr %s syncscope("device") release, align 4
        %x = atomicrmw add ptr %s, i32 1 seq_cst, align 4
        %y = cmpxchg ptr %s, i32 1, i32 2 acquire monotonic, align 4
        %t = alloca i32
        %generic_t = addrspacecast ptr %t to ptr addrspace(4)
        %z = atomicrmw add ptr addrspace(4) %generic_t, i32 1 syncscope("device") monotonic, align 4
        """; backend)
    @test !occursin("__spirv_Atomic", ir)
    @test !occursin("OpAtomic", asm)
end

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

@testset "Julia code" begin
    # end-to-end, from Julia code emitting LLVM atomics (like UnsafeAtomics does), which also
    # exercises typed pointers on Julia versions that still use them
    mod = @eval module $(gensym())
        import ..SPIRV: Atomics
        const Op = Atomics.Op
        const Ordering = Atomics.Ordering

        # (recursing rather than iterating, which Julia 1.10 doesn't unroll)
        function add!(p, scope::Val, scopes::Val...)
            Atomics.modify!(p, one(eltype(p)), Val(Op.Add), Val(Ordering.Monotonic), scope)
            add!(p, scopes...)
        end
        add!(p) = nothing

        @noinline generic_add!(p::Core.LLVMPtr{Int32,4}) =
            Atomics.modify!(p, Int32(1), Val(Op.Add), Val(Ordering.Monotonic), Val(:device))
        function generic(p::Core.LLVMPtr{Int32,1})
            generic_add!(reinterpret(Core.LLVMPtr{Int32,4}, p))
            return
        end

        function publish(data::Core.LLVMPtr{Float32,1}, flag::Core.LLVMPtr{Int32,3})
            unsafe_store!(data, 1f0)
            Atomics.store!(flag, Int32(1), Val(Ordering.Release), Val(:workgroup))
            return
        end

        function fadd!(p::Core.LLVMPtr{Float32,1})
            Atomics.modify!(p, 1f0, Val(Op.FAdd), Val(Ordering.Monotonic), Val(:device))
            return
        end
    end

    # the LLVM back-end caches the IDs of synchronization scopes across compilations, which
    # used to give later kernels the scopes of earlier ones
    Ptr = Core.LLVMPtr{Int32,1}
    for scopes in [(:device,), (:workgroup,), (:subgroup,), (:workgroup, :device),
                   (:subgroup, :workgroup, :device), (:device,)]
        asm = sprint(io -> SPIRV.code_native(io, mod.add!, Tuple{Ptr, map(s -> Val{s}, scopes)...};
                                             backend, kernel=true))
        ids = Dict(:subgroup => 3, :workgroup => 2, :device => 1)
        @test filecheck(replace(asm, r"%uint_(\d+)\b" => s"\1"),
                        ["CHECK: OpAtomicIAdd %uint {{%.+}} $(ids[scope]) $MEMORY"
                         for scope in scopes]...)
    end

    # generic pointers, which the optimizer can't make specific across a call
    # (output is captured before checking it, as a large module can block the pipe
    #  `@filecheck` reads it from while the compiler holds Julia's codegen lock)
    tt = Tuple{Core.LLVMPtr{Int32,1}}
    ir = sprint(io -> SPIRV.code_llvm(io, mod.generic, tt; backend, kernel=true,
                                      dump_module=true))
    @test filecheck(ir, "CHECK-LABEL: define {{.*}} @{{.*}}generic_add",
                    "CHECK: @_Z18__spirv_AtomicIAddPU3AS4Vijji({{.+}}, i32 1, i32 896, i32 1)")
    asm = sprint(io -> SPIRV.code_native(io, mod.generic, tt; backend, kernel=true,
                                         dump_module=true))
    @test filecheck(asm, "CHECK: OpAtomicIAdd %uint {{%.+}} %uint_1 %uint_896 %uint_1")

    # a release to a flag in local memory also publishes global memory
    asm = sprint(io -> SPIRV.code_native(io, mod.publish,
                                         Tuple{Core.LLVMPtr{Float32,1}, Core.LLVMPtr{Int32,3}};
                                         backend, kernel=true))
    @test filecheck(asm, "CHECK: OpAtomicStore {{%.+}} %uint_2 %uint_900 %uint_1")

    tt = Tuple{Core.LLVMPtr{Float32,1}}
    asm = sprint(io -> SPIRV.code_native(io, mod.fadd!, tt; backend, kernel=true))
    @test filecheck(asm, "CHECK-NOT: OpAtomicFAddEXT", "CHECK: OpAtomicCompareExchange")
    asm = sprint(io -> SPIRV.code_native(io, mod.fadd!, tt; backend, kernel=true,
                                         atomics=SPIRVAtomics(; fadd_f32_global=true)))
    @test filecheck(asm, "CHECK: OpAtomicFAddEXT")
end

@testset "operation matrix" begin
    # the Khronos translator used to exit the process on some of these, so if any still
    # reached it unlowered, that would take down the test worker: run them in another process
    if backend === :khronos
        script = """
            using GPUCompiler, LLVM, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll
            include($(repr(joinpath(@__DIR__, "..", "helpers", "runtime.jl"))))
            include($(repr(joinpath(@__DIR__, "..", "helpers", "spirv.jl"))))
            failures, rejected = SPIRV.check_atomics_matrix(:khronos)
            foreach(println, failures)
            exit(isempty(failures) && rejected > 0 ? 0 : 1)
            """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        @test success(pipeline(cmd; stdout, stderr))
    else
        failures, rejected = SPIRV.check_atomics_matrix(backend)
        @test isempty(failures) || failures
        # (8- and 16-bit operations other than native half-precision additions)
        @test rejected > 0
    end
end

@testset "without validation" begin
    # the lowering reports unsupported atomics too, instead of passing them to the back-ends
    job = atomics_job(; backend, validate=false)
    Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, atomics_kernel("""
            %a = atomicrmw add ptr addrspace(1) %g, i8 1 monotonic, align 1
            %b = atomicrmw add ptr addrspace(1) %g, i32 1 monotonic, align 4
            """))
        err = try
            GPUCompiler.lower_atomics!(job, mod)
            nothing
        catch err
            err
        end
        @test err isa GPUCompiler.InvalidIRError
        @test startswith(only(err.errors)[1], "8-bit atomic operation")
    end

    # e.g. when reflecting, which compiles without validation. the Khronos translator would
    # exit the process on a half-precision addition it has no extension for, so try that in
    # another process
    script = """
        using GPUCompiler, LLVM
        include($(repr(joinpath(@__DIR__, "..", "helpers", "runtime.jl"))))
        include($(repr(joinpath(@__DIR__, "..", "helpers", "spirv.jl"))))
        import .SPIRV: Atomics

        kernel(p) = (Atomics.modify!(p, Float16(1), Val(Atomics.Op.FAdd)); return)
        job, _ = SPIRV.create_job(kernel, (Core.LLVMPtr{Float16,1},); backend=:$backend,
                                  kernel=true, validate=false)
        try
            GPUCompiler.code_native(devnull, job)
        catch err
            err isa GPUCompiler.InvalidIRError &&
                occursin("half-precision atomic operation", sprint(showerror, err)) && exit(42)
            rethrow()
        end
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test run(pipeline(ignorestatus(cmd); stdout, stderr)).exitcode == 42
end

end
end
