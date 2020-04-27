using GPUCompiler
import LLVM

if !@isdefined(TestRuntime)
    include("../util.jl")
end

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)
    native_method = Symbol("native_$(method)")

    @eval begin
        function $native_method(io::IO, @nospecialize(func), @nospecialize(types);
                             kernel::Bool=false, minthreads=nothing, maxthreads=nothing,
                             blocks_per_sm=nothing, maxregs=nothing, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = NativeCompilerTarget()
            job = NativeCompilerJob(target=target, source=source)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $native_method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $native_method(stdout, func, types; kwargs...)
    end
end

