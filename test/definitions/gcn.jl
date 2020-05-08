using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a GCN-based test compiler, and generate reflection methods for it

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    gcn_method = Symbol("gcn_$(method)")

    @eval begin
        function $gcn_method(io::IO, @nospecialize(func), @nospecialize(types);
                             kernel::Bool=false, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = GCNCompilerTarget("gfx900")
            params = TestCompilerParams()
            job = CompilerJob(target, source, params)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $gcn_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $gcn_method(stdout, func, types; kwargs...)
    end
end
