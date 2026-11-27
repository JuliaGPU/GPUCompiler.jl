# the toolchain selected by GPUCompiler's `toolchain` preference (see src/toolchain.jl)

if haskey(ENV, "GPUCOMPILER_TEST_TOOLCHAIN")
    @test GPUCompiler.TOOLCHAIN === Symbol(ENV["GPUCOMPILER_TEST_TOOLCHAIN"])
end

kernel(p::Core.LLVMPtr{Float32,1}, x::Float32) =
    (unsafe_store!(p, unsafe_load(p) * x + 1f0); return)
const tt = Tuple{Core.LLVMPtr{Float32,1}, Float32}

# PTX jobs whose kernels are replaced by copies after linking, and that record the LLVM
# library they are optimized with
struct HookParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{PTXCompilerTarget,HookParams}) = PTX.PTXTestRuntime
function GPUCompiler.finish_linked_module!(
        @nospecialize(job::CompilerJob{PTXCompilerTarget,HookParams}), mod::LLVM.Module)
    for f in GPUCompiler.kernels(mod)
        GPUCompiler.replace_function!(f, LLVM.clone(f))
    end
end
const optimized_with = Ref{Symbol}()
GPUCompiler.optimize_module!(
        @nospecialize(job::CompilerJob{PTXCompilerTarget,HookParams}), mod::LLVM.Module) =
    (optimized_with[] = LLVM.library().name; return)

# a target defined elsewhere
struct OtherTarget <: AbstractCompilerTarget end

# the library of the IR that `job` compiles to
function ir_library(job)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        lib = ir.library
        dispose(ir)
        lib.name
    end
end

@testset "LLVM library" begin
    gpu = llvmgpu ? :gpu : :julia
    @test ir_library(first(PTX.create_job(kernel, tt; kernel=true))) === gpu
    @test ir_library(first(GCN.create_job(kernel, tt; kernel=true))) === gpu
    for backend in (:llvm, :khronos)
        @test ir_library(first(SPIRV.create_job(kernel, tt; kernel=true, backend))) === gpu
    end
    @test ir_library(first(BPF.create_job(identity, Tuple{Int}))) === gpu
    @test ir_library(first(Metal.create_job(kernel, tt; kernel=true))) === gpu
    if PTX.nvvm_available()
        @test ir_library(first(PTX.create_job(kernel, tt; kernel=true, backend=:nvvm))) === gpu
    end

    # host code stays with Julia's LLVM
    @test ir_library(first(Native.create_job(kernel, tt; kernel=true))) === :julia
    @test GPUCompiler.llvm_library(NativeCompilerTarget()) === LLVM.Library(:julia)
    # as does that of targets defined elsewhere, unless they opt in (e.g. Enzyme's, which
    # wraps the native target)
    @test GPUCompiler.llvm_library(Enzyme.EnzymeTarget()) === LLVM.Library(:julia)
    @test GPUCompiler.llvm_library(OtherTarget()) === LLVM.Library(:julia)

    # nested jobs are linked into their parent before it moves to libLLVMGPU
    job, _ = PTX.create_job(kernel, tt; kernel=true)
    job = CompilerJob(job; config=CompilerConfig(job.config; toplevel=false))
    @test ir_library(job) === :julia

    # the module moves before optimization, and back-ends can replace the entry point
    # before that (OpenCL.jl does, adding input arguments in `finish_linked_module!`)
    job, _ = PTX.create_job(kernel, tt; kernel=true)
    job = CompilerJob(job; config=CompilerConfig(job.config; params=HookParams()))
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test optimized_with[] === gpu
        context!(context(ir)) do
            @test meta.entry in ir.functions
            @test meta.entry.callconv == LLVM.CallConv.PTXKernel

            # Julia code can be compiled while inspecting the IR, into the contexts of
            # another block
            JuliaContext() do ctx2
                ir2, _ = GPUCompiler.compile(:llvm, job)
                @test ir2.library.name === gpu
                @test context(ir2) != context(ir)
                dispose(ir2)
            end
        end
        dispose(ir)
    end

    # the do-block form runs its block in the context of the IR, of either library
    for (job, lib) in ((first(PTX.create_job(kernel, tt; kernel=true)), gpu),
                       (first(Native.create_job(kernel, tt; kernel=true)), :julia))
        res = GPUCompiler.compile(:llvm, job) do ir, meta
            @test context() == context(ir)
            ir.library.name, meta.entry.name
        end
        @test res[1] === lib && occursin("kernel", res[2])
        # (without the llvmgpu toolchain, PTX code needs NVPTX_LLVM_Backend_jll, which
        # isn't available everywhere)
        job.config.target isa PTXCompilerTarget && !llvmgpu &&
            !NVPTX_LLVM_Backend_jll.is_available() && continue
        asm, entry = GPUCompiler.compile(:asm, job) do asm, meta
            @test meta.entry in meta.ir.functions
            asm, meta.entry.name
        end
        @test asm isa String && entry == res[2]
    end
    @test_throws "boom" GPUCompiler.compile((_, _) -> error("boom"), :llvm,
                                            first(PTX.create_job(kernel, tt; kernel=true)))
    # (which leaves no context of either library behind)
    @test context(; throw_error=false) === nothing
    if llvmgpu
        @test LLVM.library!(() -> context(; throw_error=false),
                            GPUCompiler.llvmgpu_state().library) === nothing
    end

    # the IR keeps GPUCompiler's datalayout (e.g. for the host's layout of `Int128`)
    for job in (first(PTX.create_job(kernel, tt; kernel=true)),
                first(SPIRV.create_job(kernel, tt; kernel=true, backend=:llvm)))
        JuliaContext() do ctx
            expected = GPUCompiler.llvm_datalayout(job.config.target)
            ir, _ = GPUCompiler.compile(:llvm, job)
            @test string(LLVM.datalayout(ir)) == expected
            dispose(ir)
        end
    end
end

if llvmgpu

@testset "defaults" begin
    # (whether SPIRV_LLVM_Backend_jll is loaded or not)
    @test SPIRVCompilerTarget().backend === :llvm

    # the version of the LLVM that generates code, also in Julia's context
    lib = GPUCompiler.llvmgpu_state().library
    for target in (PTXCompilerTarget(; cap=v"7.0"), GCNCompilerTarget(; dev_isa="gfx900"),
                   SPIRVCompilerTarget(), BPFCompilerTarget())
        @test GPUCompiler.llvm_library(target) === lib
        @test GPUCompiler.backend_llvm_version(target) == LLVM.version(lib)
    end
end

@testset "IR" begin
    job, _ = PTX.create_job(kernel, tt; kernel=true)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test LLVM.version(ir) >= v"23"
        context!(context(ir)) do
            @test meta.entry in ir.functions
            @test meta.entry.callconv == LLVM.CallConv.PTXKernel

            # Julia code can't be compiled in that context
            @test_throws "in a context of Julia's LLVM" GPUCompiler.compile(:llvm, job)
        end

        # the block owns one context of the library, for all its compilations
        ir2, meta2 = GPUCompiler.compile(:llvm, job)
        @test context(ir2) == context(ir)

        # the IR can be compiled after the fact
        asm, _ = GPUCompiler.emit_asm(job, ir2, meta2.relocations,
                                      LLVM.CodeGenFileType.Assembly)
        @test occursin(".entry", asm)
        dispose(ir2)
        dispose(ir)
    end

    # that context is owned by the `JuliaContext` block, of the task that runs it
    ThreadSafeContext() do ts_ctx
        context!(context(ts_ctx)) do
            @test_throws "No active context of libLLVMGPU" GPUCompiler.compile(:llvm, job)
        end
    end
    JuliaContext() do ctx
        msg = fetch(@async ThreadSafeContext() do ts_ctx
            context!(context(ts_ctx)) do
                try
                    GPUCompiler.compile(:llvm, job)
                    ""
                catch err
                    sprint(showerror, err)
                end
            end
        end)
        @test occursin("No active context of libLLVMGPU", msg)
    end

    # the move keeps Julia's non-integral address spaces, which optimization relies on
    JuliaContext() do ctx
        config = CompilerConfig(job.config; optimize=false, validate=false)
        ir, _ = GPUCompiler.compile(:llvm, CompilerJob(job; config))
        @test ir.library.name === :gpu
        @test occursin("ni:10:11:12:13", string(LLVM.datalayout(ir)))
        dispose(ir)
    end

    # Julia's LLVM can use typed pointers (on 1.10 and 1.11 by default), as host code keeps
    # using Julia's passes there, and the module moves to libLLVMGPU as bitcode
    if VERSION < v"1.12"
        JuliaContext(; opaque_pointers=false) do ctx
            ir, meta = GPUCompiler.compile(:llvm, first(Native.create_job(kernel, tt)))
            @test supports_typed_pointers(context(ir))
            dispose(ir)
            ir, meta = GPUCompiler.compile(:llvm, job)
            @test ir.library.name === :gpu
            dispose(ir)
        end
    end

    # reflection prints the IR of the library
    ir = sprint(io -> PTX.code_llvm(io, kernel, tt; kernel=true))
    @test occursin("define ptx_kernel void", ir)
end

@testset "code generation" begin
    asm = sprint(io -> PTX.code_native(io, kernel, tt; kernel=true, cap=v"8.0", ptx=v"7.8"))
    @test occursin(".target sm_80", asm)
    @test occursin(".version 7.8", asm)
    @test occursin("fma.rn.f32", asm)
    @test_throws "Unsupported processor sm_12" PTX.code_native(devnull, kernel, tt;
        kernel=true, cap=v"1.2")
    @test_throws "needs at least PTX ISA version 7.8" PTX.code_native(devnull, kernel, tt;
        kernel=true, cap=v"9.0", ptx=v"7.0")

    asm = sprint(io -> GCN.code_native(io, kernel, tt; kernel=true))
    @test occursin("amdhsa.kernels", asm)
    @test_throws "only supports `backend=:external`" GCN.code_native(devnull, kernel, tt;
        kernel=true, backend=:inprocess)
    @test_throws "Unsupported processor gfx1" GCN.code_native(devnull, kernel, tt;
        kernel=true, dev_isa="gfx1")

    asm = sprint(io -> SPIRV.code_native(io, kernel, tt; kernel=true, backend=:llvm))
    @test occursin("OpEntryPoint Kernel", asm)
    @test_throws "Invalid SPIR-V extensions \"+SPV_FOO\"" SPIRV.code_native(devnull, kernel, tt;
        kernel=true, backend=:llvm, extensions="+SPV_FOO")
    # (disabled extensions would stay disabled for every later target machine)
    @test_throws "Cannot disable SPIR-V extensions" SPIRV.code_native(devnull, kernel, tt;
        kernel=true, backend=:llvm, extensions="all,-SPV_KHR_bit_instructions")
    # extensions without a sign are enabled, as with SPIRV_LLVM_Backend_jll
    for extensions in ("SPV_KHR_bit_instructions", "")
        asm = sprint(io -> SPIRV.code_native(io, kernel, tt; kernel=true, backend=:llvm,
                                             extensions))
        @test occursin("OpEntryPoint Kernel", asm)
    end

    # the Khronos translator reads libLLVMGPU's bitcode (the job validates the SPIR-V)
    job, _ = SPIRV.create_job(kernel, tt; kernel=true, backend=:khronos)
    lib = GPUCompiler.compile(:obj, job) do obj, meta
        meta.ir.library.name
    end
    @test lib === :gpu

    asm = sprint(io -> BPF.code_native(io, identity, Tuple{Int}))
    @test occursin("exit", asm)
end

end
