# Nothing may combine or canonicalize instructions after target finalization (`finish_ir!`),
# as back-ends rely on its output reaching code generation as it is. Check that for every
# GPU target with a canary added at the end of `finish_ir!`, an exported function multiplying
# by two, which InstCombine would turn into a shift: it has to reach `mcgen` unchanged.

module Finalization

using GPUCompiler, LLVM, LLVM.IR, LLVM.Build, Test
using NVPTX_LLVM_Backend_jll, AMDGPU_LLVM_Backend_jll
using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll
import ..TestRuntime

struct CanaryParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,CanaryParams}) = TestRuntime

const canary = "gpucompiler_canary"
const canary_reached_mcgen = Ref(false)

function add_canary!(mod::LLVM.Module)
    context!(context(mod)) do
        T = LLVM.Int32Type()
        f = LLVM.Function(mod, canary, LLVM.FunctionType(T, [T]))
        @dispose builder=IRBuilder() begin
            position!(builder, LLVM.at_end(BasicBlock(f, "entry")))
            ret!(builder, mul!(builder, f.parameters[1], ConstantInt(T, 2)))
        end
    end
end

has_canary(mod::LLVM.Module) =
    haskey(mod.functions, canary) &&
    any(inst -> LLVM.opcode(inst) == LLVM.Opcode.Mul,
        only(mod.functions[canary].blocks).instructions)

for T in (PTXCompilerTarget, GCNCompilerTarget, SPIRVCompilerTarget, MetalCompilerTarget)
    @eval begin
        function GPUCompiler.finish_ir!(job::CompilerJob{$T,CanaryParams},
                                        mod::LLVM.Module, entry::LLVM.Function)
            entry = invoke(GPUCompiler.finish_ir!,
                           Tuple{CompilerJob{$T}, LLVM.Module, LLVM.Function},
                           job, mod, entry)
            haskey(mod.functions, canary) || add_canary!(mod)
            return entry
        end
        function GPUCompiler.mcgen(job::CompilerJob{$T,CanaryParams}, mod::LLVM.Module,
                                   format)
            canary_reached_mcgen[] = has_canary(mod)
            invoke(GPUCompiler.mcgen, Tuple{CompilerJob{$T}, LLVM.Module, typeof(format)},
                   job, mod, format)
        end
    end
end

function kernel(out::Core.LLVMPtr{Int32,1}, x::Int32)
    unsafe_store!(out, x)
    return
end

# an isbits union whose boxed constant SPIR-V and Metal demote onto the stack during
# finalization, which then cleans up after itself
@noinline produce(cond::Bool, value::Int32) = cond ? value : 1.5
function union_kernel(out::Core.LLVMPtr{Int32,1}, cond::Bool, value::Int32)
    x = produce(cond, value)
    unsafe_store!(out, Int32(x isa Float64))
    return
end

targets = Pair{String,Any}["Metal" => MetalCompilerTarget(; macos=v"12.2", metal=v"3.0", air=v"3.0")]
NVPTX_LLVM_Backend_jll.is_available() && push!(targets, "PTX" => PTXCompilerTarget(; cap=v"7.0"))
if SPIRV_LLVM_Backend_jll.is_available() && SPIRV_LLVM_Translator_jll.is_available() &&
   SPIRV_Tools_jll.is_available()
    push!(targets, "SPIR-V (LLVM)" => SPIRVCompilerTarget(; backend=:llvm),
                   "SPIR-V (Khronos)" => SPIRVCompilerTarget(; backend=:khronos))
end
if AMDGPU_LLVM_Backend_jll.is_available() && !LLVM.is_asserts()
    push!(targets, "GCN" => GCNCompilerTarget(; dev_isa="gfx900"))
end

@testset "$name" for (name, target) in targets, f in (kernel, union_kernel),
                     optimize in (true, false)
    # (Metal rejects boxed constants before LLVM 17, and SPIR-V unoptimized ones)
    f === union_kernel && !(LLVM.version() >= v"17" && target isa MetalCompilerTarget ||
                            (optimize || LLVM.version() >= v"17") &&
                            target isa SPIRVCompilerTarget) &&
        continue
    tt = f === kernel ? Tuple{Core.LLVMPtr{Int32,1}, Int32} :
                        Tuple{Core.LLVMPtr{Int32,1}, Bool, Int32}
    source = methodinstance(typeof(f), tt)
    # (unoptimized code still contains calls that validation rejects, e.g. on 1.13)
    config = CompilerConfig(target, CanaryParams(); kernel=true, optimize, validate=optimize)
    job = CompilerJob(source, config)
    canary_reached_mcgen[] = false
    JuliaContext() do ctx
        _, meta = GPUCompiler.compile(:asm, job)
        dispose(meta.ir)
    end
    @test canary_reached_mcgen[]
end

end
