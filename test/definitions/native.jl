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

module LazyCodegen
    using LLVM
    using GPUCompiler

    import ..native_job

    # We have one global JIT and TM
    const jit = Ref{LLVM.OrcJIT}()
    const tm  = Ref{LLVM.TargetMachine}()

    function __init__()
        optlevel = LLVM.API.LLVMCodeGenLevelDefault

        tm[] = GPUCompiler.JITTargetMachine(optlevel=optlevel)
        LLVM.asm_verbosity!(tm[], true)

        jit[] = LLVM.OrcJIT(tm[]) # takes ownership of tm
        atexit() do
            dispose(jit[])
        end
    end

    import GPUCompiler: deferred_codegen_jobs, CompilerJob
    mutable struct CallbackContext
        job::CompilerJob
        stub::String
        compiled::Bool
    end

    const outstanding = IdDict{CallbackContext, Nothing}()

    # Setup the lazy callback for creating a module
    function callback(orc_ref::LLVM.API.LLVMOrcJITStackRef, callback_ctx::Ptr{Cvoid})
        orc = LLVM.OrcJIT(orc_ref)
        cc = Base.unsafe_pointer_to_objref(callback_ctx)::CallbackContext

        @assert !cc.compiled
        job = cc.job

        ir, entry = GPUCompiler.codegen(:llvm, job; validate=false)
        entry_name = name(entry)

        jitted_mod = compile!(orc, ir)

        addr = addressin(orc, jitted_mod, entry_name)
        ptr  = pointer(addr)

        cc.compiled = true
        delete!(outstanding, cc)

        # 4. Update the stub pointer to point to the recently compiled module
        set_stub!(orc, cc.stub, ptr)

        # 5. Return the address of the implementation, since we are going to call it now
        return UInt64(reinterpret(UInt, ptr))
    end

    @generated function deferred_codegen(::Val{f}, ::Val{tt}) where {f,tt}
        job, _ = native_job(f, tt)

        cc = CallbackContext(job, String(gensym(:trampoline)), false)
        outstanding[cc] = nothing

        c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

        orc = jit[]
        initial_addr = callback!(orc, c_callback, pointer_from_objref(cc))
        create_stub!(orc, cc.stub, initial_addr)
        addr = address(orc, cc.stub)
        trampoline = pointer(addr)
        id = Base.reinterpret(Int, trampoline)

        deferred_codegen_jobs[id] = job

        quote
            ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
        end
    end
end