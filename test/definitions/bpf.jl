using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a native test compiler, and generate reflection methods for it

function bpf_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = BPFCompilerTarget()
    params = TestCompilerParams()
    CompilerJob(target, source, params), kwargs
end

function bpf_code_llvm(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = bpf_job(func, types; kwargs...)
    GPUCompiler.compile(:llvm, job; kwargs...)
end

function bpf_code_native(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = bpf_job(func, types; kwargs...)
    GPUCompiler.compile(:asm, job; kwargs...)
end
