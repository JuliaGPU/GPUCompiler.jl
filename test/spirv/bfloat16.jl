# `bfloat` on targets that support it for storage and conversions only (SPV_KHR_bfloat16),
# where `widen_bfloat16!` performs arithmetic on `float` and rounds conversions once

# a job for a SPIR-V target with `bfloat` storage, to process textual IR with. `validate`
# runs `spirv-val` on the output.
function bfloat16_job(; backend=:llvm, validate=true, kwargs...)
    source = methodinstance(typeof(identity), Tuple{Int}, Base.get_world_counter())
    target = SPIRVCompilerTarget(; backend, supports_bfloat16=true, validate,
                                   extensions="+SPV_KHR_bfloat16", kwargs...)
    config = CompilerConfig(target, SPIRV.CompilerParams(); kernel=true)
    CompilerJob(source, config)
end

# a kernel storing the results of `body` (`%r1` to `%r$nresults`), which can use the
# `bfloat` values `%x`, `%y` and `%z`, the `double` `%d` and the integers `%i` and `%l`
function bfloat16_kernel(body; decls="", results=String[])
    stores = join(["  store $T %r$i, ptr addrspace(1) %p" for (i, T) in enumerate(results)],
                  "\n")
    """
    target datalayout = "$(GPUCompiler.llvm_datalayout(SPIRVCompilerTarget()))"
    $decls
    define spir_kernel void @kernel(ptr addrspace(1) %p, bfloat %x, bfloat %y, bfloat %z,
                                    double %d, i32 %i, i64 %l) {
    $body
    $stores
      ret void
    }
    """
end

# the floating-point operations on `bfloat` values in `mod`, which need
# SPV_INTEL_bfloat16_arithmetic
function bfloat16_arithmetic(mod::LLVM.Module)
    isbf(T) = (T isa LLVM.VectorType ? T.element_type : T) isa LLVM.BFloatType
    found = String[]
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        arith = inst isa Union{LLVM.FNegInst, LLVM.FAddInst, LLVM.FSubInst, LLVM.FMulInst,
                               LLVM.FDivInst, LLVM.FRemInst, LLVM.FCmpInst} ||
                (inst isa LLVM.CallInst && LLVM.isintrinsic(inst.called_operand))
        if arith && (isbf(inst.value_type) || any(op -> isbf(op.value_type), inst.operands))
            push!(found, string(inst))
        end
    end
    return found
end

# widen `body` like `mcgen` does for the job, after lowering the minimum and maximum
# intrinsics like `finish_ir!` does. returns the IR, and the SPIR-V the job's back-end
# compiles it to (validated by `spirv-val`), unless `codegen` is false.
function widen_bfloat16(body; decls="", results=String[], codegen=true, kwargs...)
    job = bfloat16_job(; kwargs...)
    Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, bfloat16_kernel(body; decls, results))
        GPUCompiler.lower_minimum_maximum!(mod)
        GPUCompiler.widen_bfloat16!(job, mod)
        verify(mod)
        ir = string(mod)
        job.config.target.supports_bfloat16_arithmetic || @test bfloat16_arithmetic(mod) == []
        codegen || return ir, nothing

        mod.triple = GPUCompiler.llvm_triple(job.config.target)
        # (before Julia 1.12, `mcgen` releases the type-inference lock its caller holds)
        locked = VERSION < v"1.12.0-DEV.769"
        locked && ccall(:jl_typeinf_lock_begin, Cvoid, ())
        asm = try
            GPUCompiler.mcgen(job, mod)
        finally
            locked && ccall(:jl_typeinf_lock_end, Cvoid, ())
        end
        ir, replace(asm, "\r\n" => "\n")
    end
end

for backend in (:khronos, :llvm)
@testset "$backend" begin

@testset "arithmetic" begin
    ir, asm = widen_bfloat16("""
          %r1 = fadd bfloat %x, %y
          %a = fmul fast bfloat %r1, %z
          %r2 = fneg bfloat %a
          %r3 = fcmp olt bfloat %x, %y
          %v = insertelement <2 x bfloat> poison, bfloat %x, i32 0
          %r4 = fdiv <2 x bfloat> %v, %v
        """; results=["bfloat", "bfloat", "i1", "<2 x bfloat>"], backend)

    @test @filecheck begin
        @check "[[X:%[0-9]+]] = fpext bfloat %x to float"
        @check "[[Y:%[0-9]+]] = fpext bfloat %y to float"
        @check "[[SUM:%[0-9]+]] = fadd float [[X]], [[Y]]"
        @check "%r1 = fptrunc float [[SUM]] to bfloat"
        @check "fmul fast float"
        # (`fneg` on the bits, as conversions may change NaNs)
        @check "xor i16 {{%[0-9]+}}, -32768"
        @check "fcmp olt float"
        @check "fdiv <2 x float>"
        @check "fptrunc <2 x float> {{%[0-9]+}} to <2 x bfloat>"
        ir
    end

    @test @filecheck begin
        @check "OpCapability BFloat16TypeKHR"
        @check_not "BFloat16Arithmetic"
        @check "OpFAdd %float"
        @check "OpFConvert %bfloat16"
        @check "OpBitwiseXor %ushort"
        @check "OpFDiv %v2float"
        asm
    end
end

@testset "intrinsics" begin
    fns = ["fabs", "copysign", "minnum", "minimum", "maxnum", "floor", "sqrt", "sin",
           "fma", "fmuladd"]
    arity = Dict("copysign" => 2, "minnum" => 2, "minimum" => 2, "maxnum" => 2, "fma" => 3,
                 "fmuladd" => 3)
    body = join(map(enumerate(fns)) do (i, fn)
        args = join(["bfloat %" * "xyz"[j] for j in 1:get(arity, fn, 1)], ", ")
        "  %r$i = call bfloat @llvm.$fn.bf16($args)"
    end, "\n")
    decls = join(map(fns) do fn
        "declare bfloat @llvm.$fn.bf16($(join(fill("bfloat", get(arity, fn, 1)), ", ")))"
    end, "\n")
    n = length(fns)
    body *= """
          %r$(n+1) = call i1 @llvm.is.fpclass.bf16(bfloat %x, i32 1)
          %r$(n+2) = call i32 @llvm.fptosi.sat.i32.bf16(bfloat %x)
          %v = insertelement <2 x bfloat> poison, bfloat %x, i32 0
          %r$(n+3) = call <2 x bfloat> @llvm.fabs.v2bf16(<2 x bfloat> %v)
        """
    decls *= """
        declare i1 @llvm.is.fpclass.bf16(bfloat, i32)
        declare i32 @llvm.fptosi.sat.i32.bf16(bfloat)
        declare <2 x bfloat> @llvm.fabs.v2bf16(<2 x bfloat>)
        """
    results = [fill("bfloat", n); "i1"; "i32"; "<2 x bfloat>"]
    ir, asm = widen_bfloat16(body; decls, results, backend)

    @test @filecheck begin
        # (`fabs` and `copysign` on the bits, as conversions may change NaNs)
        @check "%[[XB:[0-9]+]] = bitcast bfloat %x to i16"
        @check "and i16 %[[XB]], 32767"
        @check "%r1 = bitcast i16 {{%[0-9]+}} to bfloat"
        @check "and i16 {{%[0-9]+}}, -32768"
        @check "%r2 = bitcast i16 {{%[0-9]+}} to bfloat"
        # (`lower_minimum_maximum!` fixes up the signs of zeros, comparing `bfloat` values)
        @check "call nsz float @llvm.minnum.f32"
        @check "fcmp oeq float"
        @check "call float @llvm.minimum.f32"
        @check "call float @llvm.floor.f32"
        @check "call float @llvm.sqrt.f32"
        @check "call float @llvm.sin.f32"
        # `fma` in `double`, rounded to odd
        @check "fmul double"
        @check "fadd double"
        @check "fptrunc double {{%[0-9]+}} to float"
        # `fmuladd` unfused
        @check "fmul float"
        @check "fptrunc float {{%[0-9]+}} to bfloat"
        @check "fadd float"
        # the bits of the value, keeping signaling NaNs
        @check "bitcast bfloat %x to i16"
        @check "shl i32"
        @check "call i1 @llvm.is.fpclass.f32"
        @check "call i32 @llvm.fptosi.sat.i32.f32"
        @check "and <2 x i16> {{%[0-9]+}}, {{splat .i16 32767.|<i16 32767, i16 32767>}}"
        ir
    end

    @test @filecheck begin
        @check_not "BFloat16Arithmetic"
        @check "OpExtInst %float {{%[0-9]+}} fmin"
        @check "OpExtInst %float {{%[0-9]+}} floor"
        @check "OpExtInst %float {{%[0-9]+}} sqrt"
        asm
    end
end

@testset "conversions" begin
    # SPV_KHR_bfloat16 converts through `float`, so conversions from `double` and wide
    # integers are rounded to odd first; the others are left alone
    ir, asm = widen_bfloat16("""
          %r1 = fptrunc double %d to bfloat
          %r2 = sitofp i32 %i to bfloat
          %r3 = uitofp i64 %l to bfloat
          %s = trunc i32 %i to i16
          %r4 = sitofp i16 %s to bfloat
          %f = fpext bfloat %x to float
          %r5 = fptrunc float %f to bfloat
          %r6 = fpext bfloat %x to double
          %r7 = fptosi bfloat %x to i64
        """; results=["bfloat", "bfloat", "bfloat", "bfloat", "bfloat", "double", "i64"],
        backend)

    @test @filecheck begin
        @check "fptrunc double %d to float"
        @check "or i32"
        @check "%r1 = fptrunc float {{%[0-9]+}} to bfloat"
        @check "call i32 @llvm.ctlz.i32"
        @check "uitofp i32 {{%[0-9]+}} to float"
        @check "%r2 = fptrunc float {{%[0-9]+}} to bfloat"
        @check "call i64 @llvm.ctlz.i64"
        @check "uitofp i64 {{%[0-9]+}} to float"
        @check "%r3 = fptrunc float {{%[0-9]+}} to bfloat"
        @check "%r4 = sitofp i16 %s to bfloat"
        @check "%r5 = fptrunc float %f to bfloat"
        @check "%r6 = fpext bfloat %x to double"
        @check "%r7 = fptosi bfloat %x to i64"
        ir
    end

    # (vectors one element at a time)
    ir, _ = widen_bfloat16("""
          %v = insertelement <2 x double> poison, double %d, i32 0
          %r1 = fptrunc <2 x double> %v to <2 x bfloat>
        """; results=["<2 x bfloat>"], backend)
    @test count("fptrunc double", ir) == 2
end

@testset "native arithmetic" begin
    # only the conversions that would round twice change
    ir, _ = widen_bfloat16("""
          %r1 = fadd bfloat %x, %y
          %r2 = fptrunc double %d to bfloat
        """; results=["bfloat", "bfloat"], backend, supports_bfloat16_arithmetic=true,
        # (SPIRV-Tools doesn't know SPV_INTEL_bfloat16_arithmetic)
        codegen=false)
    @test occursin("%r1 = fadd bfloat %x, %y", ir)
    @test occursin("fptrunc double %d to float", ir)
end

@testset "unsupported operations" begin
    # e.g., ordered reductions, which would round differently
    body = "%r1 = call bfloat @llvm.vector.reduce.fadd.v2bf16(bfloat %x, <2 x bfloat> zeroinitializer)"
    decls = "declare bfloat @llvm.vector.reduce.fadd.v2bf16(bfloat, <2 x bfloat>)"
    job = bfloat16_job(; backend)
    Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, bfloat16_kernel(body; decls, results=["bfloat"]))
        errors = GPUCompiler.validate_ir(job, mod)
        @test startswith(only(errors)[1], "call to llvm.vector.reduce.fadd.v2bf16 on bfloat values")

        # `fma` without double precision, as it would round twice in `float`
        fma = parse(LLVM.Module, bfloat16_kernel(
            "%r1 = call bfloat @llvm.fma.bf16(bfloat %x, bfloat %y, bfloat %z)";
            decls="declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)", results=["bfloat"]))
        errors = GPUCompiler.validate_ir(bfloat16_job(; backend, supports_fp64=false), fma)
        @test startswith(only(errors)[1], "call to llvm.fma.bf16 (the target supports neither")

        # without validation, compilation fails before reaching the back-end
        @test_throws GPUCompiler.InvalidIRError GPUCompiler.widen_bfloat16!(job, mod)

        # unless the target supports bfloat arithmetic
        native_job = bfloat16_job(; backend, supports_bfloat16_arithmetic=true)
        @test isempty(GPUCompiler.validate_ir(native_job, mod))
    end
end

if isdefined(Core, :BFloat16)
@testset "kernels" begin
    mod = @eval module $(gensym())
        const BF16 = Core.BFloat16
        function kernel(p::Ptr{BF16}, a::BF16, d::Float64)
            x = unsafe_load(p)
            unsafe_store!(p, Base.add_float(Base.mul_float(a, x), x), 1)
            unsafe_store!(p, Base.lt_float(x, a) ? x : a, 2)
            unsafe_store!(p, Base.abs_float(x), 3)
            unsafe_store!(p, Base.fptrunc(BF16, d), 4)
            return
        end
        reduce(p::Ptr{BF16}, x::NTuple{2,VecElement{BF16}}) =
            (unsafe_store!(p, ccall("llvm.vector.reduce.fadd.v2bf16", llvmcall, BF16,
                                    (BF16, NTuple{2,VecElement{BF16}}), reinterpret(BF16, 0x0000), x));
             return)
    end
    BF16 = Core.BFloat16

    # InstCombine runs right before `mcgen`, so the widening happens in there
    @test @filecheck begin
        @check "OpCapability BFloat16TypeKHR"
        @check_not "BFloat16Arithmetic"
        @check "OpFMul %float"
        @check "OpFAdd %float"
        @check "OpFConvert %bfloat16"
        SPIRV.code_native(mod.kernel, Tuple{Ptr{BF16}, BF16, Float64}; backend, kernel=true,
                          supports_bfloat16=true, extensions="+SPV_KHR_bfloat16")
    end

    @test_throws_message(InvalidIRError,
                         SPIRV.code_execution(mod.reduce, Tuple{Ptr{BF16}, NTuple{2,VecElement{BF16}}};
                                              backend, supports_bfloat16=true,
                                              extensions="+SPV_KHR_bfloat16")) do msg
        occursin("call to llvm.vector.reduce.fadd.v2bf16 on bfloat values", msg) &&
        occursin(r"\[\d+\] reduce", msg)
    end
end
end

end
end

# the conversions, evaluated on the host
if isdefined(Core, :BFloat16) && (Sys.ARCH === :x86_64 || Base.libllvm_version >= v"19")
@testset "rounding" begin
    # compile `op` on `T` values, widened, to a host function returning the bits of the
    # `bfloat` result
    # (passing `bfloat` values as `i16`, as the host ABI may not support them)
    function widened(op, Ts...)
        args = join(["$(T == "bfloat" ? "i16" : T) %$(i-1)" for (i, T) in enumerate(Ts)], ", ")
        casts = join(["%a$(i-1) = bitcast i16 %$(i-1) to bfloat"
                      for (i, T) in enumerate(Ts) if T == "bfloat"], "\n")
        ir = Context(; opaque_pointers=true) do ctx
            mod = parse(LLVM.Module, """
                declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)
                define i16 @entry($args) {
                  $casts
                  $op
                  %bits = bitcast bfloat %r to i16
                  ret i16 %bits
                }""")
            GPUCompiler.widen_bfloat16!(bfloat16_job(), mod)
            replace(string(mod), r"^(;|source_filename|target|attributes).*$"m => "")
        end
        jlT = Dict("double" => Float64, "i32" => Int32, "i64" => Int64, "bfloat" => UInt16)
        argtypes = Tuple{(jlT[T] for T in Ts)...}
        args = [Symbol("a$i") for i in 1:length(Ts)]
        f = @eval ($(args...),) -> Base.llvmcall(($ir, "entry"), UInt16, $argtypes, $(args...))
        bits(x) = x isa Core.BFloat16 ? reinterpret(UInt16, x) : x
        (args...) -> Base.invokelatest(f, map(bits, args)...)
    end

    # the bits of the `bfloat` nearest to `x` (ties to even), for values in the normal range
    function nearest(x)
        r = setprecision(BigFloat, 8) do
            BigFloat(x)
        end
        (reinterpret(UInt32, Float32(r)) >> 16) % UInt16
    end
    bf16(bits::UInt16) = reinterpret(Core.BFloat16, bits)
    bf16(x::Float32) = bf16((reinterpret(UInt32, x) >> 16) % UInt16)   # (exact values)
    tobig(x::Core.BFloat16) = BigFloat(reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16))

    @testset "double" begin
        f = widened("%r = fptrunc double %0 to bfloat", "double")
        # values whose nearest `float` is halfway between two `bfloat`s
        values = [1 + 2.0^-8 + 2.0^-30, 1 + 2.0^-8 - 2.0^-30, -(1 + 2.0^-8 + 2.0^-30),
                  3 * 2.0^100 + 2.0^94 + 2.0^60, 1 + 2.0^-8, 1 + 3 * 2.0^-8]
        for _ in 1:1000
            push!(values, randn() * 2.0^rand(-100:100))
            # near halfway between two `bfloat`s
            h = (1 + rand(0:127) / 128 + 2.0^-8) * 2.0^rand(-100:100)
            push!(values, h + h * 2.0^-rand(25:52) * rand((-1, 1)))
        end
        for x in values
            @test f(x) == nearest(x)
        end
        # around the largest `bfloat`, and halfway to the next power of two
        for x in (0x1.fe8p127, 0x1.ffp127 - 0x1p90, 0x1.ffp127, 0x1.ffp127 + 0x1p90)
            @test f(x) == nearest(x)
        end
        @test f(1e39) == 0x7f80
        @test f(-1e39) == 0xff80
        @test f(Inf) == 0x7f80
        @test f(-0.0) == 0x8000
        @test f(NaN) & 0x7fff > 0x7f80
    end

    @testset "$T" for (T, irT) in ((Int32, "i32"), (Int64, "i64"))
        fs = widened("%r = sitofp $irT %0 to bfloat", irT)
        fu = widened("%r = uitofp $irT %0 to bfloat", irT)
        U = unsigned(T)
        values = T[typemin(T), typemax(T), 0, 1, -1, T(2)^30 + T(2)^22 + 1,
                   -(T(2)^30 + T(2)^22 + 1), rand(T, 10000)...]
        for x in values
            @test fs(x) == nearest(x)
            @test fu(x) == nearest(reinterpret(U, x))
        end
    end

    @testset "fma" begin
        f = widened("%r = call bfloat @llvm.fma.bf16(bfloat %a0, bfloat %a1, bfloat %a2)",
                    "bfloat", "bfloat", "bfloat")
        # a product halfway between two `bfloat`s, and a tiny addend
        values = [(bf16(1.0625f0), bf16(1.0625f0), bf16(2f0^-60)),
                  (bf16(1.0625f0), bf16(1.0625f0), bf16(-2f0^-60))]
        for _ in 1:1000
            # (normal values, of either sign)
            a, b, c = (bf16(rand(UInt16) & 0xbfff | 0x3000) for _ in 1:3)
            push!(values, (a, b, c))
        end
        for (a, b, c) in values
            exact = setprecision(BigFloat, 256) do
                tobig(a) * tobig(b) + tobig(c)
            end
            @test f(a, b, c) == nearest(exact)
        end
    end
end
end
