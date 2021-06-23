using GPUCompiler

if !@isdefined(TestRuntime)
    include("../util.jl")
end


# create a native test compiler, and generate reflection methods for it

function native_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false, kwargs...)
    source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
    target = NativeCompilerTarget(always_inline=true)
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

@static Base.libllvm_version < v"12" && module LazyCodegen
    using LLVM
    using LLVM.Interop
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

        ir, meta = GPUCompiler.codegen(:llvm, job; validate=false)
        entry_name = name(meta.entry)

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
            ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
            assume(ptr != C_NULL)
            return ptr
        end
    end

    @generated function abi_call(f::Ptr{Cvoid}, rt::Type{RT}, tt::Type{T}, args::Vararg{Any, N}) where {T, RT, N}
        argtt    = tt.parameters[1]
        rettype  = rt.parameters[1]
        argtypes = DataType[argtt.parameters...]

        argexprs = Union{Expr, Symbol}[]
        ccall_types = DataType[]

        before = :()
        after = :(ret)

        # Note this follows: emit_call_specfun_other
        LLVM.Interop.JuliaContext() do ctx
            T_jlvalue = LLVM.StructType(LLVMType[], ctx)
            T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

            for (source_i, source_typ) in enumerate(argtypes)
                if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                    continue
                end

                argexpr = :(args[$source_i])

                isboxed = GPUCompiler.deserves_argbox(source_typ)
                et = isboxed ? T_prjlvalue : convert(LLVMType, source_typ, ctx)

                if isboxed
                    push!(ccall_types, Any)
                elseif isa(et, LLVM.SequentialType) # et->isAggregateType
                    push!(ccall_types, Ptr{source_typ})
                    argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                else
                    push!(ccall_types, source_typ)
                end
                push!(argexprs, argexpr)
            end

            if GPUCompiler.isghosttype(rettype) || Core.Compiler.isconstType(rettype)
                # Do nothing...
                # In theory we could set `rettype` to `T_void`, but ccall will do that for us
            # elseif jl_is_uniontype?
            elseif !GPUCompiler.deserves_retbox(rettype)
                rt = convert(LLVMType, rettype, ctx)
                if !isa(rt, LLVM.VoidType) && GPUCompiler.deserves_sret(rettype, rt)
                    before = :(sret = Ref{$rettype}())
                    pushfirst!(argexprs, :(sret))
                    pushfirst!(ccall_types, Ptr{rettype})
                    rettype = Nothing
                    after = :(sret[])
                end
            else
                # rt = T_prjlvalue
            end
        end

        quote
            $before
            ret = ccall(f, $rettype, ($(ccall_types...),), $(argexprs...))
            $after
        end
    end

    @inline function call_delayed(f::F, args...) where F
        tt = Tuple{map(Core.Typeof, args)...}
        rt = Core.Compiler.return_type(f, tt)
        ptr = deferred_codegen(Val(f), Val(tt))
        abi_call(ptr, rt, tt, args...)
    end
end
