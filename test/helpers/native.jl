module Native

using ..GPUCompiler
using LLVM
import ..TestRuntime

# local method table for device functions
Base.Experimental.@MethodTable(test_method_table)

struct CompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table
    relocatable::Bool

    CompilerParams(entry_safepoint::Bool=false, method_table=test_method_table,
                   relocatable::Bool=false) =
        new(entry_safepoint, method_table, relocatable)
end

module Runtime end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}
GPUCompiler.runtime_module(::NativeCompilerJob) = Runtime

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint
function GPUCompiler.lower_host_references!(@nospecialize(job::NativeCompilerJob),
                                            mod::LLVM.Module,
                                            refs::GPUCompiler.HostReferences)
    if job.config.params.relocatable
        GPUCompiler.emit_host_reference_declarations!(mod, refs)
    else
        invoke(GPUCompiler.lower_host_references!,
               Tuple{CompilerJob,LLVM.Module,GPUCompiler.HostReferences}, job, mod, refs)
    end
end

function create_job(@nospecialize(func), @nospecialize(types);
                    entry_safepoint::Bool=false, method_table=test_method_table,
                    relocatable::Bool=false, kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = NativeCompilerTarget(;jlruntime=true)
    params = CompilerParams(entry_safepoint, method_table, relocatable)
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function load(obj::Vector{UInt8}, entry::String, refs::GPUCompiler.HostReferences)
    lljit = LLJIT()
    try
        jd = JITDylib(lljit)
        prefix = LLVM.get_prefix(lljit)
        add!(jd, LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix))

        cells = Vector{UInt}(undef, length(refs.slots))
        roots = Any[]
        pairs = LLVM.API.LLVMOrcCSymbolMapPair[]
        for (i, (name, ref)) in enumerate(refs.slots)
            cells[i] = GPUCompiler.resolve_host_reference(ref)
            ref isa GPUCompiler.JuliaValueRef && push!(roots, ref.value)
            symbol = LLVM.API.LLVMJITEvaluatedSymbol(
                reinterpret(UInt, pointer(cells, i)),
                LLVM.API.LLVMJITSymbolFlags(
                    LLVM.API.LLVMJITSymbolGenericFlagsExported, 0))
            push!(pairs, LLVM.API.LLVMOrcCSymbolMapPair(mangle(lljit, name), symbol))
        end
        isempty(pairs) || LLVM.define(jd, LLVM.absolute_symbols(pairs))

        add!(lljit, jd, MemoryBuffer(obj))
        addr = lookup(lljit, entry)
        return pointer(addr), (lljit, cells, roots)
    catch
        dispose(lljit)
        rethrow()
    end
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

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    method = Symbol("$(method)")
    @eval begin
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

# simulates codegen for a kernel function: validates by default
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        GPUCompiler.compile(:asm, job; kwargs...)
    end
end

end
