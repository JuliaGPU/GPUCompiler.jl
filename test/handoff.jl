# the late hand-off to a standalone LLVM library (see helpers/handoff.jl for how to select one)

@testset "configuration" begin
    if haskey(ENV, "GPUCOMPILER_TEST_HANDOFF")
        @test GPUCompiler.default_handoff() === Symbol(ENV["GPUCOMPILER_TEST_HANDOFF"])
    end

    source = methodinstance(typeof(identity), Tuple{Nothing})
    native = NativeCompilerTarget()
    ptx = PTXCompilerTarget(; cap=v"7.0")

    @test CompilerConfig(native, PTX.CompilerParams(); handoff=:none).handoff === :none
    @test_throws "does not support the late hand-off" CompilerConfig(native, PTX.CompilerParams(); handoff=:late)
    @test_throws "Unknown handoff" CompilerConfig(ptx, PTX.CompilerParams(); handoff=:early)

    # nested jobs are linked into their parent before it is handed off
    config = CompilerConfig(ptx, PTX.CompilerParams(); handoff=:late)
    @test config.handoff === :late
    @test CompilerConfig(config; toplevel=false).handoff === :none
    @test CompilerConfig(config; toplevel=false, target=native).handoff === :none

    # the preference only applies to targets that support the hand-off
    @test CompilerConfig(native, PTX.CompilerParams()).handoff === :none
    @test CompilerConfig(ptx, PTX.CompilerParams()).handoff === GPUCompiler.default_handoff()

    # the results of the hand-off depend on a library that package images don't track
    job = CompilerJob(source, CompilerConfig(ptx, PTX.CompilerParams(true); handoff=:late))
    @test !GPUCompiler.can_persist_results(job)
end

const library_available =
    GPUCompiler.isavailable(GPUCompiler.LLVMGPU_jll) || haskey(ENV, "GPUCOMPILER_LIBLLVMGPU")
haskey(ENV, "GPUCOMPILER_TEST_HANDOFF") && @test library_available

if library_available

kernel(p::Core.LLVMPtr{Float32,1}, x::Float32) =
    (unsafe_store!(p, unsafe_load(p) * x + 1f0); return)
const tt = Tuple{Core.LLVMPtr{Float32,1}, Float32}

@testset "IR" begin
    is_handoff = GPUCompiler.default_handoff() === :late
    job, _ = PTX.create_job(kernel, tt; kernel=true, handoff=:late)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test ir.library === LLVM.Library(:gpu)
        @test LLVM.version(ir) >= v"23"

        # owning objects are usable as is, values in the context of the library
        @test occursin("ptx_kernel", string(ir))
        context!(context(ir)) do
            @test meta.entry in ir.functions
            @test meta.entry.callconv == LLVM.CallConv.PTXKernel
        end

        # by default, as the preference says
        default_job, _ = PTX.create_job(kernel, tt; kernel=true)
        ir2, _ = GPUCompiler.compile(:llvm, default_job)
        @test is_handoff == (ir2.library === LLVM.Library(:gpu))
        dispose(ir2)

        # the IR can be compiled after the fact, also when its library is the current one
        ir2, meta2 = GPUCompiler.compile(:llvm, job)
        asm, _ = GPUCompiler.emit_asm(job, ir, meta.relocations,
                                      LLVM.CodeGenFileType.Assembly)
        @test occursin(".entry", asm)
        LLVM.library!(ir2.library) do
            asm, _ = GPUCompiler.emit_asm(job, ir2, meta2.relocations,
                                          LLVM.CodeGenFileType.Assembly)
            @test occursin(".entry", asm)
        end
        dispose(ir2)

        # the block owns one context of the library, for all its compilations
        ir2, _ = GPUCompiler.compile(:llvm, job)
        @test context(ir2) == context(ir)
        dispose(ir2)
        dispose(ir)
    end

    # that context is owned by the `JuliaContext` block
    ThreadSafeContext() do ts_ctx
        context!(context(ts_ctx)) do
            @test_throws "inside a `JuliaContext() do ... end` block" GPUCompiler.compile(:llvm, job)
        end
    end

    # reflection prints the IR of the library
    ir = sprint(io -> PTX.code_llvm(io, kernel, tt; kernel=true, handoff=:late))
    @test occursin("define ptx_kernel void", ir)
end

@testset "code generation" begin
    # PTX
    asm = sprint(io -> PTX.code_native(io, kernel, tt; kernel=true, handoff=:late, cap=v"8.0",
                                       ptx=v"7.8"))
    @test occursin(".target sm_80", asm)
    @test occursin(".version 7.8", asm)
    @test occursin("fma.rn.f32", asm)
    @test_throws "Unsupported processor sm_12" PTX.code_native(devnull, kernel, tt;
        kernel=true, handoff=:late, cap=v"1.2")
    @test_throws "Unsupported PTX ISA version 1.0" PTX.code_native(devnull, kernel, tt;
        kernel=true, handoff=:late, ptx=v"1.0")
    @test_throws "needs at least PTX ISA version 7.8" PTX.code_native(devnull, kernel, tt;
        kernel=true, handoff=:late, cap=v"9.0", ptx=v"7.0")

    # GCN
    asm = sprint(io -> GCN.code_native(io, kernel, tt; kernel=true, handoff=:late))
    @test occursin("amdhsa.kernels", asm)
    @test_throws "Unsupported processor gfx1" GCN.code_native(devnull, kernel, tt;
        kernel=true, handoff=:late, dev_isa="gfx1")

    # SPIR-V, with both back-ends
    for backend in (:llvm, :khronos)
        asm = sprint(io -> SPIRV.code_native(io, kernel, tt; kernel=true, handoff=:late, backend))
        @test occursin("OpEntryPoint Kernel", asm)
    end
end

@testset "SPIR-V extensions" begin
    # every target machine gets exactly the extensions of its job, regardless of what the
    # jobs before it enabled
    fadd(p::Core.LLVMPtr{Float32,1}) = (SPIRV.Atomics.modify!(p, 1f0,
        Val(SPIRV.Atomics.Op.FAdd), Val(SPIRV.Atomics.Ordering.Monotonic), Val(:device)); return)
    native(atomics) = sprint(io -> SPIRV.code_native(io, fadd, Tuple{Core.LLVMPtr{Float32,1}};
                                                     kernel=true, handoff=:late, backend=:llvm,
                                                     atomics))
    with_fadd = SPIRVAtomics(; fadd_f32_global=true)
    with_both = SPIRVAtomics(; fadd_f16_global=true, fadd_f32_global=true)
    for atomics in (with_fadd, SPIRVAtomics(), with_fadd, with_both, SPIRVAtomics(), with_fadd)
        asm = native(atomics)
        @test occursin("OpAtomicFAddEXT", asm) == (atomics != SPIRVAtomics())
    end

    @test_throws "Unknown SPIR-V extension \"SPV_FOO\"" SPIRV.code_native(devnull, kernel, tt;
        kernel=true, handoff=:late, backend=:llvm, extensions="+SPV_FOO")
end

@testset "relocations" begin
    # relocations are recorded by name, so they survive the hand-off
    sym_kernel(p::Core.LLVMPtr{Int,1}, s::Symbol) = (unsafe_store!(p, s === :foo ? 1 : 2); return)
    job, _ = PTX.create_job(sym_kernel, Tuple{Core.LLVMPtr{Int,1}, Symbol}; kernel=true,
                            handoff=:late, patch=true)
    JuliaContext() do ctx
        asm, meta = GPUCompiler.compile(:asm, job)
        @test !isempty(meta.relocations.records)
        for rec in meta.relocations.records
            @test occursin(rec.name, asm)
        end
        dispose(meta.ir)
    end
end

end
