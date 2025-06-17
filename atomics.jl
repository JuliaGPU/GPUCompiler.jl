using GPUCompiler
using LLVM

function initialize()
    if !GPUCompiler.__llvm_initialized[]
        InitializeAllTargets()
        InitializeAllTargetInfos()
        InitializeAllAsmPrinters()
        InitializeAllAsmParsers()
        InitializeAllTargetMCs()
        GPUCompiler.__llvm_initialized[] = true
    end
end

# include all helpers
include(joinpath("test", "helpers", "runtime.jl"))
include(joinpath("test", "helpers", "ptx.jl"))

job, _ = PTX.create_job(identity, (Int,))

includet("src/atomic_legalization.jl")

# mod = """
# define void @test(ptr %a) nounwind {
#   %1 = load atomic i128, ptr %a seq_cst, align 16
#   store atomic i128 %1, ptr %a seq_cst, align 16
#   ret void
# }
# """
# => __sync_val_compare_and_swap_16


# mod = """
# define void @test(ptr %a) nounwind {
#   %1 = load atomic i8, ptr %a seq_cst, align 16
#   store atomic i8 %1, ptr %a seq_cst, align 16
#   ret void
# }
# """

# Cannot select: 0x67a0660: ch = AtomicStore<(store seq_cst (s8) into %ir.a, align 16)> 0x67a05f0:1, 0x67a0580, 0x67a05f0
#   0x67a0580: i64,ch = load<(dereferenceable invariant load (s64) from `ptr addrspace(101) null`, addrspace 101)> 0x7125d30, TargetExternalSymbol:i64'test_param_0', undef:i64
#     0x67a0200: i64 = TargetExternalSymbol'test_param_0'
#     0x67a02e0: i64 = undef
#   0x67a05f0: i16,ch = AtomicLoad<(load seq_cst (s8) from %ir.a, align 16)> 0x7125d30, 0x67a0580
#     0x67a0580: i64,ch = load<(dereferenceable invariant load (s64) from `ptr addrspace(101) null`, addrspace 101)> 0x7125d30, TargetExternalSymbol:i64'test_param_0', undef:i64
#       0x67a0200: i64 = TargetExternalSymbol'test_param_0'
#       0x67a02e0: i64 = undef

mod = """
define i8 @test(ptr %a) nounwind {
  %1 = load atomic i8, ptr %a seq_cst, align 16
  ret i8 %1
}
"""

asm, meta = JuliaContext(opaque_pointers=true) do ctx
    initialize()
    ir = parse(LLVM.Module, mod)
    ir = legalize_atomics!(job, ir)
    GPUCompiler.emit_asm(job, ir, LLVM.API.LLVMAssemblyFile)
end


