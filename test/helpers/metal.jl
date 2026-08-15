module Metal

using ..GPUCompiler
import ..TestRuntime

struct CompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,CompilerParams}) = TestRuntime

struct ThreadedRuntimeCompilerParams <: AbstractCompilerParams end

module ThreadedRuntime
    using ..GPUCompiler

    signal_exception() = return
    malloc(sz) = C_NULL
    function report_oom(sz)
        ccall("extern julia.air.thread_position_in_grid.v3i32", llvmcall,
              NTuple{3, VecElement{UInt32}}, ())
        return
    end
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

ThreadedRuntimeCompilerJob = CompilerJob{MetalCompilerTarget,ThreadedRuntimeCompilerParams}
GPUCompiler.runtime_module(::ThreadedRuntimeCompilerJob) = ThreadedRuntime

# Selects the `:table` relocation strategy, as Metal.jl does, so relocation words are read out
# of a table reached through the kernel state instead of being baked. Used to test that delivery
# path (and the Metal target's `relocation_table_pointer`) without depending on Metal.jl.
struct TableKernelState
    reloc_table::Core.LLVMPtr{UInt64, 1}
end

struct TableCompilerParams <: AbstractCompilerParams end
TableCompilerJob = CompilerJob{MetalCompilerTarget,TableCompilerParams}
GPUCompiler.runtime_module(::TableCompilerJob) = TestRuntime
# as Metal.jl does: only a kernel has the state a table can be reached through, so reflection
# on a plain device function falls back to session-local resolution
GPUCompiler.relocation_lowering(job::TableCompilerJob) =
    job.config.kernel ? (:table) : (:bake)
GPUCompiler.kernel_state_type(::TableCompilerJob) = TableKernelState

function create_table_job(@nospecialize(func), @nospecialize(types); kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = MetalCompilerTarget(; macos=v"12.2", metal=v"3.0", air=v"3.0")
    config = CompilerConfig(target, TableCompilerParams(); kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function code_native_table(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_table_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end
code_native_table(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_native_table(stdout, func, types; kwargs...)

function create_job(@nospecialize(func), @nospecialize(types); kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = MetalCompilerTarget(; macos=v"12.2", metal=v"3.0", air=v"3.0")
    params = CompilerParams()
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function create_threaded_runtime_job(@nospecialize(func), @nospecialize(types); kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = MetalCompilerTarget(; macos=v"12.2", metal=v"3.0", air=v"3.0")
    params = ThreadedRuntimeCompilerParams()
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function code_llvm_threaded_runtime(io::IO, @nospecialize(func), @nospecialize(types);
                                    kwargs...)
    job, kwargs = create_threaded_runtime_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    method = Symbol("$(method)")
    @eval begin
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

code_llvm_threaded_runtime(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_llvm_threaded_runtime(stdout, func, types; kwargs...)

# simulates codegen for a kernel function: validates by default
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end

end
