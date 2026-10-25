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
        legalize_module!(mod)
        entry = LLVM.functions(mod)["transport"]
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
        legalize_module!(mod)
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
        legalize_module!(mod)
        @test LLVM.verify(mod) === nothing
        @test occursin("call void @callee(ptr byval(i64)", string(mod))
        @test !occursin("!fpmath", string(mod))
        dispose(mod)
    end
end

@testset "outlined calls have a debug location" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
        define i1 @caller(double %x) !dbg !3 {
          %y = fadd double %x, 1.0, !dbg !4
          %z = fcmp une double %y, 0.0
          ret i1 %z
        }

        !llvm.dbg.cu = !{!0}
        !llvm.module.flags = !{!5}
        !0 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !1, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
        !1 = !DIFile(filename: "julia", directory: ".")
        !2 = !DISubroutineType(types: !{})
        !3 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: null, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
        !4 = !DILocation(line: 2, scope: !3)
        !5 = !{i32 2, !"Debug Info Version", i32 3}
        """)
        legalize_module!(mod)
        @test LLVM.verify(mod) === nothing
        caller = LLVM.functions(mod)["caller"]
        calls = [inst for block in LLVM.blocks(caller) for inst in LLVM.instructions(block)
                 if inst isa LLVM.CallInst]
        @test !isempty(calls)
        locations = [call.debug_location for call in calls]
        @test all(loc -> loc !== nothing && LLVM.scope(loc) == caller.subprogram, locations)
        # the call replacing the `fcmp` gets a line-0 location in the caller
        @test any(loc -> loc !== nothing && LLVM.line(loc) == 0, locations)
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
        @test_throws "unsupported binary64 call" legalize_module!(mod)
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
            @test_throws "unsupported binary64 call" legalize_module!(mod)
            dispose(mod)
        end
    end
end

@testset "unsupported binary64 intrinsic diagnostic" begin
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, """
            declare double @llvm.ldexp.f64.i32(double, i32)
            define double @caller(double %value, i32 %power) {
                %result = call double @llvm.ldexp.f64.i32(double %value, i32 %power)
                ret double %result
            }
            """)
        try
            @test_throws "unsupported binary64" legalize_module!(mod)
        finally
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
        legalize_module!(mod)
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
        legalize_module!(mod)
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
            legalize_module!(mod)
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
        legalize_module!(mod)
        @test LLVM.verify(mod) === nothing
        @test occursin("call i64 @gpu_softfloat_f16_to_f64(i16", string(mod))
        @test !occursin("fpext", string(mod))
        dispose(mod)
    end
end

if LLVM.version() < v"19"
    # LLVM 19 removed `icmp` constant expressions; before, Base's `sin` contained one
    @testset "unrelated constant expressions" begin
        LLVM.Context(; opaque_pointers=true) do ctx
            mod = parse(LLVM.Module, """
            @flag = global i8 0

            define double @select_by_address(double %x) {
              %y = fadd double %x, 1.0
              %z = select i1 icmp ult (i64 ptrtoint (ptr @flag to i64), i64 4096), double %x, double %y
              ret double %z
            }
            """)
            legalize_module!(mod)
            @test LLVM.verify(mod) === nothing
            @test isempty(validate_module(mod))
            @test occursin("icmp ult (i64 ptrtoint (ptr @flag to i64), i64 4096)", string(mod))
            dispose(mod)
        end
    end
end
