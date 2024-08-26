@testsetup module PTX

using GPUCompiler


# create a PTX-based test compiler, and generate reflection methods for it

include("runtime.jl")
struct CompilerParams <: AbstractCompilerParams end

PTXCompilerJob = CompilerJob{PTXCompilerTarget,CompilerParams}

struct PTXKernelState
    data::Int64
end
GPUCompiler.kernel_state_type(@nospecialize(job::PTXCompilerJob)) = PTXKernelState
@inline @generated kernel_state() = GPUCompiler.kernel_state_value(PTXKernelState)

function mark(x)
    ccall("gpucompiler.mark", llvcmall, Nothing, (Int,), x)
end

function remove_mark!(@nospecialize(job), intrinsic, mod::LLVM.Module)
    changed = false

    for use in uses(intrinsic)
        val = user(use)
        if isempty(uses(val))
            unsafe_delete!(LLVM.parent(val), val)
            changed = true
        else
            # the validator will detect this
        end
    end

    return changed
end

GPUCompiler.PIPELINE_CALLBACKS["gpucompiler.mark"] = remove_mark!

# a version of the test runtime that has some side effects, loading the kernel state
# (so that we can test if kernel state arguments are appropriately optimized away)
module PTXTestRuntime
    using ..GPUCompiler
    import ..PTXKernelState

    function signal_exception()
        kernel_state()
        return
    end

    # dummy methods
    # HACK: if malloc returns 0 or traps, all calling functions (like jl_box_*)
    #       get reduced to a trap, which really messes with our test suite.
    malloc(sz) = Ptr{Cvoid}(Int(0xDEADBEEF))
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end
GPUCompiler.runtime_module(::PTXCompilerJob) = PTXTestRuntime

function create_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                    minthreads=nothing, maxthreads=nothing, blocks_per_sm=nothing,
                    maxregs=nothing, always_inline=false, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = PTXCompilerTarget(;cap=v"7.0",
                               minthreads, maxthreads,
                               blocks_per_sm, maxregs)
    params = CompilerParams()
    config = CompilerConfig(target, params; kernel, always_inline)
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

function code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# simulates codegen for a kernel function: validates by default
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end

end
