using GPUCompiler

if !@isdefined(TestRuntime)
    include("../testhelpers.jl")
end


# create a GCN-based test compiler, and generate reflection methods for it

function gcn_job(@nospecialize(func), @nospecialize(types);
                 kernel::Bool=false, always_inline=false, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = GCNCompilerTarget(dev_isa="gfx900")
    params = TestCompilerParams()
    config = CompilerConfig(target, params; kernel, always_inline)
    CompilerJob(source, config), kwargs
end

function gcn_code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = gcn_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function gcn_code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = gcn_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function gcn_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = gcn_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function gcn_code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = gcn_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# simulates codegen for a kernel function: validates by default
function gcn_code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = gcn_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end
