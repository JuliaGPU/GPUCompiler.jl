@testset "module layout canonicalization" begin
    # Restore codegen-counter order and merge only equivalent constant clones.
    JuliaContext() do ctx
        p = typed_ptrs ? "i64*" : "ptr"
        @dispose mod=parse(LLVM.Module, """
            @"_j_const#2" = private unnamed_addr constant i64 2, align 8
            @"_j_const#2.1" = private unnamed_addr constant i64 2, align 4
            @"_j_const#1.1" = private unnamed_addr constant i64 1, align 8
            @jl_nothing = external global i64
            @"_j_str_x#3.1" = private unnamed_addr constant i64 4, align 8
            @"jl_global#20" = external global i64
            @"_j_const#1" = private unnamed_addr constant i64 1, align 8
            @"_j_str_x#3" = private unnamed_addr constant i64 3, align 8
            @"+Core.Tuple#15" = external global i64

            declare void @llvm.trap()

            define i64 @julia_b_12() {
              %v = load i64, $p @"_j_const#1.1"
              ret i64 %v
            }

            define void @jfptr_a_11() {
              ret void
            }

            declare void @ijl_throw()

            define i64 @julia_a_10() {
              %v = load i64, $p @"_j_const#1"
              ret i64 %v
            }

            define i64 @julia_c_13() {
              %v = load i64, $p @"_j_str_x#3.1"
              ret i64 %v
            }
            """) begin
            GPUCompiler.canonicalize_module_layout!(mod)
            @test (verify(mod); true)

            # emission order, then uncountered declarations by name
            @test [f.name for f in mod.functions] ==
                  ["julia_a_10", "jfptr_a_11", "julia_b_12", "julia_c_13", "ijl_throw", "llvm.trap"]
            @test [g.name for g in mod.globals] ==
                  ["_j_const#1", "_j_const#2", "_j_const#2.1", "_j_str_x#3", "_j_str_x#3.1",
                   "+Core.Tuple#15", "jl_global#20", "jl_nothing"]

            # the identical clone was folded into the copy that kept the bare name...
            @test !haskey(mod.globals, "_j_const#1.1")
            @test occursin("@\"_j_const#1\"", string(mod.functions["julia_b_12"]))
            # ...while a same-named constant with different content is left alone
            @test occursin("@\"_j_str_x#3.1\"", string(mod.functions["julia_c_13"]))

            # idempotent
            before = string(mod)
            GPUCompiler.canonicalize_module_layout!(mod)
            @test string(mod) == before
        end
    end
end

@testset "compile unit deduplication" begin
    # Linking modules concatenates their `distinct` DICompileUnits. Identical copies must be
    # folded onto the first-listed one, references repointed, and the list shrunk, while a CU
    # with different content (another producer) is left as is. The result must still verify.
    JuliaContext() do ctx
        @dispose mod=parse(LLVM.Module, """
            define void @f() !dbg !4 {
              ret void, !dbg !7, !custom !12
            }

            define void @g() !dbg !8 {
              ret void
            }

            define void @h() !dbg !9 {
              ret void
            }

            !llvm.dbg.cu = !{!0, !1, !2}
            !llvm.module.flags = !{!10, !11}

            !0 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
            !1 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
            !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "vendor", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
            !3 = !DIFile(filename: "julia", directory: ".")
            !4 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)
            !5 = !DISubroutineType(types: !6)
            !6 = !{}
            !7 = !DILocation(line: 1, scope: !4)
            !8 = distinct !DISubprogram(name: "g", linkageName: "g", scope: null, file: !3, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !2)
            !9 = distinct !DISubprogram(name: "h", linkageName: "h", scope: null, file: !3, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
            !10 = !{i32 2, !"Dwarf Version", i32 4}
            !11 = !{i32 2, !"Debug Info Version", i32 3}
            !12 = !{!1}
            """) begin
            @test GPUCompiler.dedup_compile_units!(mod)
            @test (verify(mod); true)
            ir = string(mod)

            # Two CUs remain even though the duplicate was also reachable through custom metadata.
            @test count("distinct !DICompileUnit", ir) == 2
            @test occursin(r"!llvm\.dbg\.cu = !\{!\d+, !\d+\}", ir)

            # both Julia subprograms now share the canonical CU; the vendor one is untouched
            function unit_producer(fname)
                m = match(Regex("!DISubprogram\\(name: \"$fname\".*?unit: !(\\d+)"), ir)
                m === nothing && return nothing
                cu = match(Regex("^!$(m.captures[1]) = distinct !DICompileUnit\\(.*?producer: \"([^\"]+)\"", "m"), ir)
                cu === nothing ? nothing : cu.captures[1]
            end
            @test unit_producer("f") == "julia"
            @test unit_producer("h") == "julia"
            @test unit_producer("g") == "vendor"
            @test occursin(r"!DISubprogram\(name: \"f\".*?unit: !(\d+)", ir) &&
                  match(r"!DISubprogram\(name: \"f\".*?unit: !(\d+)", ir).captures[1] ==
                  match(r"!DISubprogram\(name: \"h\".*?unit: !(\d+)", ir).captures[1]

            # nothing left to do
            @test !GPUCompiler.dedup_compile_units!(mod)
        end
    end

    # Debug records can hold the only reference to a compile unit: here, only a record is
    # left of the inlined `g`, so it holds the only reference to `g`'s subprogram, and the
    # type of `y` is scoped to a compile unit (as Clang does for types at file scope).
    LLVM.version() >= v"19" && JuliaContext() do ctx
        @dispose mod=parse(LLVM.Module, """
            define void @f() !dbg !3 {
              #dbg_value(i32 0, !6, !DIExpression(), !8)
              #dbg_value(i32 1, !12, !DIExpression(), !9)
              ret void
            }

            !llvm.dbg.cu = !{!0, !1, !14}
            !llvm.module.flags = !{!10, !11}

            !0 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
            !1 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
            !2 = !DIFile(filename: "julia", directory: ".")
            !3 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !2, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
            !4 = !DISubroutineType(types: !5)
            !5 = !{}
            !6 = !DILocalVariable(name: "x", scope: !7, file: !2, line: 2)
            !7 = distinct !DISubprogram(name: "g", linkageName: "g", scope: null, file: !2, line: 2, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !1)
            !8 = !DILocation(line: 2, scope: !7, inlinedAt: !9)
            !9 = !DILocation(line: 1, scope: !3)
            !10 = !{i32 2, !"Dwarf Version", i32 4}
            !11 = !{i32 2, !"Debug Info Version", i32 3}
            !12 = !DILocalVariable(name: "y", scope: !3, file: !2, line: 1, type: !13)
            !13 = !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !14, file: !2, size: 64, elements: !5)
            !14 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
            """) begin
            @test GPUCompiler.dedup_compile_units!(mod)
            @test (verify(mod); true)
            ir = string(mod)
            @test count("distinct !DICompileUnit", ir) == 1
            @test count("#dbg_value", ir) == 2
        end
    end
end

@testset "codegen name normalization" begin
    # Only a debug record is left of the inlined `julia_g_123`, so it holds the only reference
    # to the subprogram that carries its name.
    LLVM.version() >= v"19" && JuliaContext() do ctx
        @dispose mod=parse(LLVM.Module, """
            define void @julia_f_100() !dbg !3 {
              #dbg_value(i32 0, !6, !DIExpression(), !8)
              ret void
            }

            !llvm.dbg.cu = !{!0}
            !llvm.module.flags = !{!10, !11}

            !0 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
            !2 = !DIFile(filename: "julia", directory: ".")
            !3 = distinct !DISubprogram(name: "f", linkageName: "julia_f_100", scope: null, file: !2, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
            !4 = !DISubroutineType(types: !5)
            !5 = !{}
            !6 = !DILocalVariable(name: "x", scope: !7, file: !2, line: 2)
            !7 = distinct !DISubprogram(name: "g", linkageName: "julia_g_123", scope: null, file: !2, line: 2, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
            !8 = !DILocation(line: 2, scope: !7, inlinedAt: !9)
            !9 = !DILocation(line: 1, scope: !3)
            !10 = !{i32 2, !"Dwarf Version", i32 4}
            !11 = !{i32 2, !"Debug Info Version", i32 3}
            """) begin
            GPUCompiler.normalize_julia_symbol_names!(mod)
            @test (verify(mod); true)
            ir = string(mod)
            @test occursin("define void @julia_f_1()", ir)
            @test occursin("linkageName: \"julia_g_2\"", ir)
            @test !occursin("julia_g_123", ir)
        end
    end
end
