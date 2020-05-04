using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a GCN-based test compiler, and generate reflection methods for it

struct GCNTestCompilerTarget <: CompositeCompilerTarget
    parent::GCNCompilerTarget

    GCNTestCompilerTarget(dev_isa::String) = new(GCNCompilerTarget(dev_isa))
end

Base.parent(target::GCNTestCompilerTarget) = target.parent

struct GCNTestCompilerJob <: CompositeCompilerJob
    parent::AbstractCompilerJob
end

GPUCompiler.runtime_module(target::GCNTestCompilerTarget) = TestRuntime

GCNTestCompilerJob(target::AbstractCompilerTarget, source::FunctionSpec; kwargs...) =
    GCNTestCompilerJob(GCNCompilerJob(target, source; kwargs...))

Base.similar(job::GCNTestCompilerJob, source::FunctionSpec; kwargs...) =
    GCNTestCompilerJob(similar(job.parent, source; kwargs...))

Base.parent(job::GCNTestCompilerJob) = job.parent

gcn_dev_isa = "gfx900"
for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    gcn_method = Symbol("gcn_$(method)")

    @eval begin
        function $gcn_method(io::IO, @nospecialize(func), @nospecialize(types);
                             kernel::Bool=false, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = GCNTestCompilerTarget($gcn_dev_isa)
            job = GCNTestCompilerJob(target, source)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $gcn_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $gcn_method(stdout, func, types; kwargs...)
    end
end
