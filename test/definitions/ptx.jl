using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a PTX-based test compiler, and generate reflection methods for it

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    ptx_method = Symbol("ptx_$(method)")

    @eval begin
        function $ptx_method(io::IO, @nospecialize(func), @nospecialize(types);
                             kernel::Bool=false, minthreads=nothing, maxthreads=nothing,
                             blocks_per_sm=nothing, maxregs=nothing, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = PTXCompilerTarget(cap=v"7.0",
                                       minthreads=minthreads, maxthreads=maxthreads,
                                       blocks_per_sm=blocks_per_sm, maxregs=maxregs)
            params = TestCompilerParams()
            job = CompilerJob(target, source, params)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $ptx_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $ptx_method(stdout, func, types; kwargs...)
    end
end
