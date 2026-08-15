module Native

using ..GPUCompiler
using LLVM
import ..TestRuntime

# local method table for device functions
Base.Experimental.@MethodTable(test_method_table)

struct CompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table
    relocations::Symbol

    CompilerParams(entry_safepoint::Bool=false, method_table=test_method_table,
                   relocations::Symbol=:bake) =
        new(entry_safepoint, method_table, relocations)
end

module Runtime end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,CompilerParams}
GPUCompiler.runtime_module(::NativeCompilerJob) = Runtime

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint

# Every mode ends up in an ORC JIT: `patch` emits definitions for `load` to write after adding
# the object, while `table` delivers the words as run-time data, reached through the single
# patchable global below. The latter is a stand-in for a platform that offers no access to
# loaded code at all (Metal), letting that strategy be tested off-device.
GPUCompiler.relocation_lowering(@nospecialize(job::NativeCompilerJob)) =
    job.config.params.relocations

# The `:table` back-end contract: hand out a pointer to the table base. A real back-end reads
# it out of per-dispatch state; here one patchable global holds it for the whole object, which
# `load` fills in after adding the object to the JIT.
const RELOC_TABLE_BASE = "__reloc_table_base"

function GPUCompiler.relocation_table_pointer(@nospecialize(job::NativeCompilerJob),
                                              builder::LLVM.IRBuilder, fun::LLVM.Function)
    mod = LLVM.parent(fun)
    T_word = GPUCompiler.relocation_word_type()
    gv = if haskey(globals(mod), RELOC_TABLE_BASE)
        globals(mod)[RELOC_TABLE_BASE]
    else
        gv = GlobalVariable(mod, T_word, RELOC_TABLE_BASE)
        initializer!(gv, LLVM.ConstantInt(T_word, 0))
        extinit!(gv, true)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
        set_used!(mod, gv)
        gv
    end
    return inttoptr!(builder, load!(builder, T_word, gv), LLVM.PointerType(T_word))
end

function GPUCompiler.mcgen(@nospecialize(job::NativeCompilerJob), mod::LLVM.Module,
                           format=LLVM.API.LLVMAssemblyFile)
    if job.config.params.relocations !== :bake
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
                    relocations::Symbol=:bake, kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = NativeCompilerTarget(;jlruntime=true)
    params = CompilerParams(entry_safepoint, method_table, relocations)
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

# Add an object to a fresh ORC JIT and supply its relocation words, the loader in miniature.
# Under `:patch` each word is written into the loaded image by name and offset (CUDA does the
# same with `cuModuleGetGlobal` + `cuMemcpyHtoD`); under `:table` one word table is allocated
# and its address written into the module's single table-base global (Metal passes the same
# address in the kernel state). Pass empty `relocs` for objects that need neither.
#
# The returned table must stay rooted for as long as the code is callable.
function load(obj::Vector{UInt8}, entry::String, relocs::GPUCompiler.Relocations;
              table::Bool=false)
    lljit = LLJIT(; tm=JITTargetMachine())
    try
        jd = JITDylib(lljit)
        prefix = LLVM.get_prefix(lljit)
        add!(jd, LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix))

        add!(lljit, jd, MemoryBuffer(obj))
        words = UInt[]
        if table
            words = GPUCompiler.resolved_relocation_table(relocs)
            if !isempty(words)
                base = lookup(lljit, RELOC_TABLE_BASE)
                unsafe_store!(Ptr{UInt}(pointer(base)), UInt(pointer(words)))
            end
        else
            for (rec, value) in GPUCompiler.resolved_relocations(relocs)
                addr = lookup(lljit, rec.name)
                unsafe_store!(Ptr{UInt}(pointer(addr) + rec.offset), value)
            end
        end
        addr = lookup(lljit, entry)
        return pointer(addr), lljit, words
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
