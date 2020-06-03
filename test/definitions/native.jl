using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a native test compiler, and generate reflection methods for it

function native_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = NativeCompilerTarget()
    params = TestCompilerParams()
    CompilerJob(target, source, params), kwargs
end

function native_code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function native_code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function native_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function native_code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    native_method = Symbol("native_$(method)")
    @eval begin
        $native_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $native_method(stdout, func, types; kwargs...)
    end
end

# simulates codegen for a kernel function: validates by default
function native_code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = native_job(func, types; kernel=true, kwargs...)
    GPUCompiler.compile(:asm, job; kwargs...)
end
