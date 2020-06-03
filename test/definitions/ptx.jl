using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a PTX-based test compiler, and generate reflection methods for it

function ptx_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                 minthreads=nothing, maxthreads=nothing, blocks_per_sm=nothing,
                 maxregs=nothing, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = PTXCompilerTarget(cap=v"7.0",
                               minthreads=minthreads, maxthreads=maxthreads,
                               blocks_per_sm=blocks_per_sm, maxregs=maxregs)
    params = TestCompilerParams()
    CompilerJob(target, source, params), kwargs
end

function ptx_code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = ptx_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function ptx_code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = ptx_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function ptx_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = ptx_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function ptx_code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = ptx_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    ptx_method = Symbol("ptx_$(method)")
    @eval begin
        $ptx_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $ptx_method(stdout, func, types; kwargs...)
    end
end

# simulates codegen for a kernel function: validates by default
function ptx_code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = ptx_job(func, types; kernel=true, kwargs...)
    GPUCompiler.compile(:asm, job; kwargs...)
end
