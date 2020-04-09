# implementation of the GPUCompiler interfaces for generating PTX code

## target

export PTXCompilerTarget

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    runtime_module::Base.Module

    cap::VersionNumber

    # hooks for exposing functionality to CUDAnative.jl
    # FIXME: this isn't particularly nice, duplicating fields and functions
    #        from the abstract interface over this job to CUDAnative...
    rewrite_ir_hook::Union{Nothing,Function}=nothing
    link_library_hook::Union{Nothing,Function}=nothing
    isintrinsic_hook::Union{Nothing,Function}=nothing
end

llvm_triple(::PTXCompilerTarget) = Int===Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda"

llvm_datalayout(::PTXCompilerTarget) = Int===Int64 ?
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64" :
    "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

function llvm_machine(target::PTXCompilerTarget)
    InitializeNVPTXTarget()
    InitializeNVPTXTargetInfo()
    t = Target(llvm_triple(target))

    InitializeNVPTXTargetMC()
    cpu = "sm_$(target.cap.major)$(target.cap.minor)"
    feat = "+ptx60" # we only support CUDA 9.0+ and LLVM 6.0+
    tm = TargetMachine(t, llvm_triple(target), cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end

runtime_module(target::PTXCompilerTarget) = target.runtime_module

rewrite_ir!(target::PTXCompilerTarget, mod::LLVM.Module) =
    target.rewrite_ir_hook !== nothing && target.rewrite_ir_hook(target, mod)
link_libraries!(target::PTXCompilerTarget, mod::LLVM.Module, undefined_fns::Vector{String}) =
    target.link_library_hook !== nothing && target.link_library_hook(target, mod, undefined_fns)

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(target::PTXCompilerTarget, fn::String) =
    in(fn, ptx_intrinsics) || (target.isintrinsic_hook!==nothing && target.isintrinsic_hook(fn))


## job

export PTXCompilerJob

Base.@kwdef struct PTXCompilerJob <: AbstractCompilerJob
    target::PTXCompilerTarget
    source::FunctionSpec

    # optional
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing
end

target(job::PTXCompilerJob) = job.target
source(job::PTXCompilerJob) = job.source

PTXCompilerJob(target, source; kwargs...) =
    PTXCompilerJob(target=target, source=source; kwargs...)

Base.similar(job::PTXCompilerJob, source::FunctionSpec) =
    PTXCompilerJob(target=job.target, source=source,
                   minthreads=job.minthreads, maxthreads=job.maxthreads,
                   blocks_per_sm=job.blocks_per_sm, maxregs=job.maxregs)

function Base.show(io::IO, job::PTXCompilerJob)
    print(io, "PTX CompilerJob of ", source(job))
    print(io, " for sm_$(target(job).cap.major)$(target(job).cap.minor)")

    job.minthreads !== nothing && print(io, ", minthreads=$(job.minthreads)")
    job.maxthreads !== nothing && print(io, ", maxthreads=$(job.maxthreads)")
    job.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.blocks_per_sm)")
    job.maxregs !== nothing && print(io, ", maxregs=$(job.maxregs)")
end

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::PTXCompilerJob) = "ptx-sm_$(job.target.cap.major)$(job.target.cap.minor)"
