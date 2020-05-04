using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a PTX-based test compiler, and generate reflection methods for it

struct PTXTestCompilerTarget <: CompositeCompilerTarget
    parent::PTXCompilerTarget

    PTXTestCompilerTarget(cap::VersionNumber) = new(PTXCompilerTarget(cap))
end

Base.parent(target::PTXTestCompilerTarget) = target.parent

struct PTXTestCompilerJob <: CompositeCompilerJob
    parent::AbstractCompilerJob
end

GPUCompiler.runtime_module(target::PTXTestCompilerTarget) = TestRuntime

PTXTestCompilerJob(target::AbstractCompilerTarget, source::FunctionSpec; kwargs...) =
    PTXTestCompilerJob(PTXCompilerJob(target, source; kwargs...))

Base.similar(job::PTXTestCompilerJob, source::FunctionSpec; kwargs...) =
    PTXTestCompilerJob(similar(job.parent, source; kwargs...))

Base.parent(job::PTXTestCompilerJob) = job.parent

ptx_cap = v"7.0"
for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    ptx_method = Symbol("ptx_$(method)")

    @eval begin
        function $ptx_method(io::IO, @nospecialize(func), @nospecialize(types);
                             kernel::Bool=false, minthreads=nothing, maxthreads=nothing,
                             blocks_per_sm=nothing, maxregs=nothing, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = PTXTestCompilerTarget($ptx_cap)
            job = PTXTestCompilerJob(target, source;
                                     minthreads=minthreads, maxthreads=maxthreads,
                                     blocks_per_sm=blocks_per_sm, maxregs=maxregs)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $ptx_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $ptx_method(stdout, func, types; kwargs...)
    end
end
