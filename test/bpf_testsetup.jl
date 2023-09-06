@testsetup module BPF

using GPUCompiler


# create a native test compiler, and generate reflection methods for it

include("runtime.jl")
struct CompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,CompilerParams}) = TestRuntime

function create_job(@nospecialize(func), @nospecialize(types);
                    kernel::Bool=false, always_inline=false, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = BPFCompilerTarget()
    params = CompilerParams()
    config = CompilerConfig(target, params; kernel, always_inline)
    CompilerJob(source, config), kwargs
end

function code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# simulates codegen for a kernel function: validates by default
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end

end
