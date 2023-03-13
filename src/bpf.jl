# implementation of the GPUCompiler interfaces for generating eBPF code

## target

export BPFCompilerTarget

Base.@kwdef struct BPFCompilerTarget <: AbstractCompilerTarget
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

valid_function_pointer(job::CompilerJob{BPFCompilerTarget}, ptr::Ptr{Cvoid}) =
    reinterpret(UInt, ptr) in job.config.target.function_pointers
