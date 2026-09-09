# LLVM-level tests of the binary64 legalization pass

using GPUCompiler.SoftFloat: legalize_module!, validate_module
using LLVM, Test

# The IR below is written with opaque pointers, which LLVM 15/16 contexts do not accept
# unless asked; the typed-pointer path of the pass is covered by the Metal tests.

@testset "module legalization" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        ir = """
        %node = type { double, ptr }
        @values = constant [2 x double] [double -0.0, double 0x7FF8000000000042]
        @vector = constant <2 x double> <double 1.0, double -0.0>

        declare double @llvm.sqrt.f64(double)
        declare double @llvm.fma.f64(double, double, double)
        declare double @llvm.fabs.f64(double)

        define double @transport(double %a, double %b, ptr byval(double) %ignored) {
        entry:
          %slot = alloca %node
          %field = getelementptr %node, ptr %slot, i32 0, i32 0
          store double %a, ptr %field
          %loaded = load double, ptr %field
          %sum = fadd double %loaded, %b
          %product = fmul double %sum, %b
          %root = call double @llvm.sqrt.f64(double %product)
          %fused = call double @llvm.fma.f64(double %root, double %a, double 1.5)
          %negative = fneg double %fused
          %absolute = call double @llvm.fabs.f64(double %negative)
          %cmp = fcmp ule double %absolute, %a
          %selected = select i1 %cmp, double %absolute, double %a
          br i1 %cmp, label %left, label %right
        left:
          br label %exit
        right:
          br label %exit
        exit:
          %phi = phi double [ %selected, %left ], [ 0x8000000000000000, %right ]
          ret double %phi
        }

        define <2 x double> @vectors(<2 x double> %a, <2 x double> %b) {
          %x = fdiv <2 x double> %a, %b
          ret <2 x double> %x
        }

        !air.kernel = !{!0}
        !0 = !{ptr @transport}
        """
        mod = parse(LLVM.Module, ir)
        entry = legalize_module!(mod, LLVM.functions(mod)["transport"])
        output = string(mod)
        @test isempty(validate_module(mod))
        @test LLVM.verify(mod) === nothing
        @test LLVM.function_type(entry) == LLVM.FunctionType(
            LLVM.Int64Type(), [LLVM.Int64Type(), LLVM.Int64Type(), LLVM.PointerType()])
        @test occursin("@values = constant [2 x i64] [i64 -9223372036854775808, i64 9221120237041090626]", output)
        @test occursin("%node.softfloat64 = type { i64, ptr }", output)
        @test occursin("byval(i64)", output)
        @test occursin("!air.kernel", output)
        @test !occursin(r"\bdouble\b", output)
        @test !occursin(r"\bi128\b", output)
        dispose(mod)
    end
end

@testset "nested constant arrays" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        @table = constant [3 x [2 x double]] [
          [2 x double] zeroinitializer,
          [2 x double] [double 1.0, double -0.0],
          [2 x double] undef]
        @cube = constant [1 x [2 x [2 x double]]] [[2 x [2 x double]] [
          [2 x double] [double 2.0, double 3.0],
          [2 x double] [double 4.0, double 5.0]]]
        define double @lookup(i64 %i, i64 %j) {
          %ptr = getelementptr [3 x [2 x double]], ptr @table, i64 0, i64 %i, i64 %j
          %x = load double, ptr %ptr
          ret double %x
        }
        """)
        legalize_module!(mod, LLVM.functions(mod)["lookup"])
        @test LLVM.verify(mod) === nothing
        @test isempty(validate_module(mod))
        table = LLVM.initializer(LLVM.globals(mod)["table"])
        @test LLVM.value_type(table) == LLVM.ArrayType(LLVM.ArrayType(LLVM.Int64Type(), 2), 3)
        elements = collect(LLVM.operands(table))
        @test elements[1] isa LLVM.ConstantAggregateZero
        @test string(elements[2]) == "[2 x i64] [i64 4607182418800017408, i64 -9223372036854775808]"
        @test elements[3] isa LLVM.UndefValue
        cube = LLVM.initializer(LLVM.globals(mod)["cube"])
        @test LLVM.value_type(cube) == LLVM.ArrayType(LLVM.ArrayType(LLVM.ArrayType(LLVM.Int64Type(), 2), 2), 1)
        @test occursin("getelementptr [3 x [2 x i64]]", string(mod))
        dispose(mod)
    end
end

@testset "call-site types and metadata" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        define void @callee(ptr byval(double) %x) { ret void }
        define double @caller(double %x, ptr %p) {
          call void @callee(ptr byval(double) %p)
          %y = fadd double %x, 1.0, !fpmath !0
          ret double %y
        }
        !0 = !{float 2.5}
        """)
        legalize_module!(mod, LLVM.functions(mod)["caller"])
        @test LLVM.verify(mod) === nothing
        @test occursin("call void @callee(ptr byval(i64)", string(mod))
        @test !occursin("!fpmath", string(mod))
        dispose(mod)
    end
end

@testset "external binary64 ABI is not silently changed" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        declare double @external(double)
        define double @caller(double %x) {
          %y = call double @external(double %x)
          ret double %y
        }
        """)
        @test_throws "unsupported binary64 call" legalize_module!(mod, LLVM.functions(mod)["caller"])
        dispose(mod)
    end
    for attribute in ("byval", "sret"), indirect in (false, true)
        LLVM.Context(; opaque_pointers=true) do ctx
            callee = indirect ? "%fn" : "@external"
            mod = parse(LLVM.Module, """
            %payload = type { double }
            declare void @external(ptr $attribute(%payload))
            define void @caller(ptr %fn, ptr %x) {
              call void $callee(ptr $attribute(%payload) %x)
              ret void
            }
            """)
            @test LLVM.verify(mod) === nothing
            @test_throws "unsupported binary64 call" legalize_module!(mod, LLVM.functions(mod)["caller"])
            dispose(mod)
        end
    end
end

@testset "opaque-pointer storage still requires legalization" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        define ptr @storage() {
          %slot = alloca double
          %next = getelementptr double, ptr %slot, i64 1
          ret ptr %next
        }
        """)
        @test LLVM.verify(mod) === nothing
        @test count(contains("retains double"), validate_module(mod)) == 2
        legalize_module!(mod, LLVM.functions(mod)["storage"])
        @test occursin("alloca i64", string(mod))
        @test occursin("getelementptr i64", string(mod))
        @test isempty(validate_module(mod))
        dispose(mod)
    end
end

@testset "unrelated equal-width mapping remains generic" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        remapper = GPUCompiler.SoftFloat.Float64TypeRemapper()
        @test GPUCompiler.SoftFloat.remap_type(remapper, LLVM.HalfType()) == LLVM.HalfType()
        @test GPUCompiler.SoftFloat.remap_type(remapper,
            LLVM.ArrayType(LLVM.DoubleType(), 3)) == LLVM.ArrayType(LLVM.Int64Type(), 3)
    end
end

@testset "native floating-point operations survive" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        define float @native_float(i1 %flag, float %x) {
          %converted = uitofp i1 %flag to float
          %sum = fadd float %converted, %x
          ret float %sum
        }
        """)
        legalize_module!(mod, LLVM.functions(mod)["native_float"])
        output = string(mod)
        @test isempty(validate_module(mod))
        @test occursin("uitofp i1", output)
        @test occursin("fadd float", output)
        dispose(mod)
    end
end

if LLVM.version() >= v"17"
    @testset "floating-point class attributes" begin
        LLVM.Context(; opaque_pointers=true) do ctx
            mod = parse(LLVM.Module, """
            define nofpclass(nan inf) double @callee(double nofpclass(nan) %x,
                                                    float nofpclass(nan) %y) {
              ret double %x
            }
            define double @caller(double %x, float %y) {
              %z = call nofpclass(nan inf) double @callee(double nofpclass(nan) %x,
                                                          float nofpclass(nan) %y)
              ret double %z
            }
            """)
            @test LLVM.verify(mod) === nothing
            legalize_module!(mod, LLVM.functions(mod)["caller"])
            @test LLVM.verify(mod) === nothing
            output = string(mod)
            @test !occursin(r"nofpclass\([^)]*\) i64|i64 nofpclass", output)
            @test count("float nofpclass(nan)", output) == 2
            dispose(mod)
        end
    end
end

@testset "widening preserves input bits" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        define double @widen(half %x) {
          %y = fpext half %x to double
          ret double %y
        }
        """)
        legalize_module!(mod, LLVM.functions(mod)["widen"])
        @test LLVM.verify(mod) === nothing
        @test occursin("call i64 @gpu_softfloat_f16_to_f64(i16", string(mod))
        @test !occursin("fpext", string(mod))
        dispose(mod)
    end
end
