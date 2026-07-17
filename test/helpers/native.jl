module Native

using ..GPUCompiler
using LLVM
import ..TestRuntime

# local method table for device functions
Base.Experimental.@MethodTable(test_method_table)

struct CompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table
    jit::Bool

    CompilerParams(entry_safepoint::Bool=false, method_table=test_method_table,
                   jit::Bool=false) =
        new(entry_safepoint, method_table, jit)
end

module Runtime end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}
GPUCompiler.runtime_module(::NativeCompilerJob) = Runtime

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint
function GPUCompiler.lower_relocations!(@nospecialize(job::NativeCompilerJob),
                                            mod::LLVM.Module,
                                            relocs::GPUCompiler.Relocations)
    if job.config.params.jit
        GPUCompiler.emit_imported_relocations!(mod, relocs)
    else
        invoke(GPUCompiler.lower_relocations!,
               Tuple{CompilerJob,LLVM.Module,GPUCompiler.Relocations}, job, mod, relocs)
    end
end

function GPUCompiler.mcgen(@nospecialize(job::NativeCompilerJob), mod::LLVM.Module,
                           format=LLVM.API.LLVMAssemblyFile)
    if job.config.params.jit
        target = job.config.target
        @dispose tm=JITTargetMachine(GPUCompiler.llvm_triple(target), target.cpu,
                                     target.features) begin
            return String(emit(tm, mod, format))
        end
    else
        return invoke(GPUCompiler.mcgen, Tuple{CompilerJob,LLVM.Module,Any},
                      job, mod, format)
    end
end

function create_job(@nospecialize(func), @nospecialize(types);
                    entry_safepoint::Bool=false, method_table=test_method_table,
                    jit::Bool=false, kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = NativeCompilerTarget(;jlruntime=true)
    params = CompilerParams(entry_safepoint, method_table, jit)
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function load(obj::Vector{UInt8}, entry::String, relocs::GPUCompiler.Relocations,
              ir::LLVM.Module)
    lljit = LLJIT(; tm=JITTargetMachine())
    try
        jd = JITDylib(lljit)
        prefix = LLVM.get_prefix(lljit)
        add!(jd, LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix))

        relocations = GPUCompiler.resolved_relocations(relocs)
        declarations = [(site, value) for (site, value) in relocations.sites
                        if isdeclaration(globals(ir)[site.name])]
        cells = Vector{UInt}(undef, length(declarations))
        pairs = LLVM.API.LLVMOrcCSymbolMapPair[]
        for (i, (site, value)) in enumerate(declarations)
            cells[i] = value
            symbol = LLVM.API.LLVMJITEvaluatedSymbol(
                reinterpret(UInt, pointer(cells, i)),
                LLVM.API.LLVMJITSymbolFlags(
                    LLVM.API.LLVMJITSymbolGenericFlagsExported, 0))
            push!(pairs, LLVM.API.LLVMOrcCSymbolMapPair(mangle(lljit, site.name), symbol))
        end
        isempty(pairs) || LLVM.define(jd, LLVM.absolute_symbols(pairs))

        add!(lljit, jd, MemoryBuffer(obj))
        for (site, value) in relocations.sites
            isdeclaration(globals(ir)[site.name]) && continue
            addr = lookup(lljit, site.name)
            unsafe_store!(Ptr{UInt}(pointer(addr) + site.offset), value)
        end
        addr = lookup(lljit, entry)
        return pointer(addr), (lljit, cells, relocations.roots)
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
