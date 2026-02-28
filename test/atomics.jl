function run_pass(backend, pass, mod)
    GPUCompiler.initialize_llvm()

    fake_job, _ = backend.create_job(identity, (Int,))

    # TODO: Set DL?
    asm, meta = JuliaContext(opaque_pointers=true) do ctx
        ir = parse(LLVM.Module, mod)
        ir = pass(fake_job, ir)
        GPUCompiler.emit_asm(fake_job, ir, LLVM.API.LLVMAssemblyFile)
    end
    write(stdout, asm)
end

@testset "PTX" begin
    # PTX backend doesn't support larger than i64 atomics
    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i128, ptr %a seq_cst, align 16
        store atomic i128 %1, ptr %a seq_cst, align 16
        ret void
        }
        """
        check"CHECK: LLVM error: Undefined external symbol \"__sync_val_compare_and_swap_16\""

        run_pass(PTX, (_, ir)-> ir, mod)
    end

    # Note: Unordered gets eliminated here

    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i64, ptr %a monotonic, align 8
        store atomic i64 %1, ptr %a monotonic, align 8
        ret void
        }
        """
        check"CHECK: .target sm_70"
        check"CHECK: ld.volatile.u64"
        check"CHECK: st.volatile.u64"

        run_pass(PTX, (_, ir)-> ir, mod)
    end
    
    # Note: PTX backend doesn't support store/release yet
    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i64, ptr %a acquire, align 8
        store atomic i64 %1, ptr %a release, align 8
        ret void
        }
        """
        check"CHECK: LLVM error: Cannot select: 0x{{[0-9_a-z]*}}: ch = AtomicStore<(store release (s64)"

        run_pass(PTX, (_, ir)-> ir, mod)
    end

    # Note: PTX backend doesn't support seq_cst yet
    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
          %1 = load atomic i64, ptr %a seq_cst, align 8
          store atomic i64 %1, ptr %a seq_cst, align 8
          ret void
        }
        """
        check"CHECK: LLVM error: Cannot select: 0x{{[0-9_a-z]*}}: ch = AtomicStore<(store seq_cst (s64)"

        run_pass(PTX, (_, ir)-> ir, mod)
    end

    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i32, ptr %a monotonic, align 4
        store atomic i32 %1, ptr %a monotonic, align 4
        ret void
        }
        """
        check"CHECK: .target sm_70"
        check"CHECK: ld.volatile.u32"
        check"CHECK: st.volatile.u32"

        run_pass(PTX, (_, ir)-> ir, mod)
    end

    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i16, ptr %a monotonic, align 2
        store atomic i16 %1, ptr %a monotonic, align 2
        ret void
        }
        """
        check"CHECK: .target sm_70"
        check"CHECK: ld.volatile.u16"
        check"CHECK: st.volatile.u16"

        run_pass(PTX, (_, ir)-> ir, mod)
    end

    @test @filecheck begin
        mod = """define void @test(ptr %a) nounwind {
        %1 = load atomic i8, ptr %a monotonic, align 1
        store atomic i8 %1, ptr %a monotonic, align 1
        ret void
        }
        """
        check"CHECK: .target sm_70"
        check"CHECK: ld.volatile.u8"
        check"CHECK: st.volatile.u8"

        run_pass(PTX, (_, ir)-> ir, mod)
    end

end # PTX