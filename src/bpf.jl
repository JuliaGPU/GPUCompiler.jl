# implementation of the GPUCompiler interfaces for generating eBPF code

## target

export BPFCompilerTarget

Base.@kwdef struct BPFCompilerTarget <: AbstractCompilerTarget
    prog_section::String="prog" # section for kernel to be placed in
    license::String="" # license for kernel and source code
    function_pointers::UnitRange{Int}=1:1000 # set of valid function "pointers"
end

llvm_triple(::BPFCompilerTarget) = "bpf-bpf-bpf"
llvm_datalayout(::BPFCompilerTarget) = "e-m:e-p:64:64-i64:64-n32:64-S128"

function llvm_machine(target::BPFCompilerTarget)
    triple = llvm_triple(target)
    t = Target(;triple=triple)

    cpu = ""
    feat = ""
    tm = TargetMachine(t, triple, cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end


## job

runtime_slug(job::CompilerJob{BPFCompilerTarget}) = "bpf"

const bpf_intrinsics = () # TODO
isintrinsic(::CompilerJob{BPFCompilerTarget}, fn::String) = in(fn, bpf_intrinsics)

function finish_module!(job::CompilerJob{BPFCompilerTarget}, mod::LLVM.Module)
    for func in LLVM.functions(mod)
        if LLVM.name(func) == "gpu_signal_exception"
            throw(KernelError(job, "eBPF does not support exceptions"))
        end
        # Set entry section for loaders like libbpf
        LLVM.section!(func, job.target.prog_section)
    end

    # Set license
    license = job.target.license
    if license != ""
        ctx = LLVM.context(mod)
        i8 = LLVM.Int8Type(ctx)
        glob = GlobalVariable(mod, LLVM.ArrayType(i8, length(license)+1), "_license")
        linkage!(glob, LLVM.API.LLVMExternalLinkage)
        constant!(glob, true)
        section!(glob, "license")
        initializer!(glob, ConstantArray(i8, vcat(Vector{UInt8}(license),0x0)))
    end

    # Set all map definitions as external linkage
    for gv in filter(x->(section(x)=="maps")||(section(x)==".maps"), collect(LLVM.globals(mod)))
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
    end
end

valid_function_pointer(job::CompilerJob{BPFCompilerTarget}, ptr::Ptr{Cvoid}) =
    reinterpret(UInt, ptr) in job.target.function_pointers
