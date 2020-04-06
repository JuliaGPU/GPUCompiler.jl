export AbstractCompilerJob, PTXCompilerJob

abstract type AbstractCompilerJob end
# expected fields:
# - target::AbstractCompilerTarget
# - f::Base.Callable
# - tt::DataType
# - kernel::Bool
# - name::Union{Nothing,String}

# expected methods:
# - similar(::AbstractCompilerJob, f, tt, kernel=true; name=nothing)

# global job reference
# FIXME: thread through `job` everywhere (deadlocks the Julia compiler when doing so with
#        LLVM passes implemented in Julia)
current_job = nothing

function signature(job::AbstractCompilerJob)
    fn = something(job.name, nameof(job.f))
    args = join(job.tt.parameters, ", ")
    return "$fn($(join(job.tt.parameters, ", ")))"
end


#
# PTX
#

Base.@kwdef struct PTXCompilerJob <: AbstractCompilerJob
    target::PTXCompilerTarget
    f::Base.Callable
    tt::DataType
    kernel::Bool
    name::Union{Nothing,String} = nothing

    # PTX specific
    cap::VersionNumber
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing
end

PTXCompilerJob(target, f, tt, cap, kernel; kwargs...) =
    PTXCompilerJob(target=target, f=f, tt=tt, kernel=kernel, cap=cap; kwargs...)

Base.similar(job::PTXCompilerJob, f, tt, kernel=true; name=nothing) =
    PTXCompilerJob(target=job.target, f=f, tt=tt, kernel=kernel, name=name,
                   cap=job.cap,
                   minthreads=job.minthreads, maxthreads=job.maxthreads,
                   blocks_per_sm=job.blocks_per_sm, maxregs=job.maxregs)

function Base.show(io::IO, job::AbstractCompilerJob)
    print(io, "CUDA CompilerJob for $(signature(job))")

    print(io, " (cap=$(job.cap.major).$(job.cap.minor)")
    job.kernel && print(io, ", kernel=true")
    job.minthreads !== nothing && print(io, ", minthreads=$(job.minthreads)")
    job.maxthreads !== nothing && print(io, ", maxthreads=$(job.maxthreads)")
    job.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.blocks_per_sm)")
    job.maxregs !== nothing && print(io, ", maxregs=$(job.maxregs)")
    job.name !== nothing && print(io, ", name=$(job.name)")
    print(io, ")")
end
