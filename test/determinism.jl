@testset "compile unit deduplication" begin
    # Linking modules concatenates their `distinct` DICompileUnits. Identical copies must be
    # folded onto the first-listed one, references repointed, and the list shrunk, while a CU
    # with different content (another producer) is left as is. The result must still verify.
    JuliaContext() do ctx
        mod = parse(LLVM.Module, """
            define void @f() !dbg !4 {
              ret void, !dbg !7
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
            """)
        @test GPUCompiler.dedup_compile_units!(mod)
        @test (verify(mod); true)
        ir = string(mod)

        # two CUs remain: the canonical Julia one and the vendor one
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
        dispose(mod)
    end
end
