using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a SPIRV-based test compiler, and generate reflection methods for it

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    spirv_method = Symbol("spirv_$(method)")

    @eval begin
        function $spirv_method(io::IO, @nospecialize(func), @nospecialize(types);
                               kernel::Bool=false, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = SPIRVCompilerTarget()
            params = TestCompilerParams()
            job = CompilerJob(target, source, params)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $spirv_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $spirv_method(stdout, func, types; kwargs...)
    end
end
