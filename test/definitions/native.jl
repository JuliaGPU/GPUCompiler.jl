using GPUCompiler

if !@isdefined(TestRuntime)
    include("../testhelpers.jl")
end


# create a native test compiler, and generate reflection methods for it

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(test_method_table)
else
const test_method_table = nothing
end

struct NativeCompilerParams <: AbstractCompilerParams
    entry_safepoint::Bool
    method_table

    NativeCompilerParams(entry_safepoint::Bool=false, method_table=test_method_table) =
        new(entry_safepoint, method_table)
end

NativeCompilerJob = CompilerJob{NativeCompilerTarget,NativeCompilerParams}

GPUCompiler.method_table(@nospecialize(job::NativeCompilerJob)) = job.config.params.method_table
GPUCompiler.can_safepoint(@nospecialize(job::NativeCompilerJob)) = job.config.params.entry_safepoint
GPUCompiler.runtime_module(::NativeCompilerJob) = TestRuntime

function native_job(@nospecialize(func), @nospecialize(types); kernel::Bool=false,
                    entry_abi=:specfunc, entry_safepoint::Bool=false, always_inline=false,
                    method_table=test_method_table, kwargs...)
    source = methodinstance(typeof(func), Base.to_tuple_type(types))
    target = NativeCompilerTarget()
    params = NativeCompilerParams(entry_safepoint, method_table)
    config = CompilerConfig(target, params; kernel, entry_abi, always_inline)
    CompilerJob(source, config), kwargs
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
    using LLVM.Interop
    using GPUCompiler

    import ..native_job

    @static if Base.libllvm_version < v"12"
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

            name, jitted_mod = JuliaContext() do ctx
                ir, meta = GPUCompiler.compile(:llvm, job; validate=false, ctx)
                name(meta.entry), compile!(orc, ir)
            end

            addr = addressin(orc, jitted_mod, name)
            ptr  = pointer(addr)

            cc.compiled = true
            delete!(outstanding, cc)

            # 4. Update the stub pointer to point to the recently compiled module
            set_stub!(orc, cc.stub, ptr)

            # 5. Return the address of the implementation, since we are going to call it now
            return UInt64(reinterpret(UInt, ptr))
        end

        function get_trampoline(job)
            cc = CallbackContext(job, String(gensym(:trampoline)), false)
            outstanding[cc] = nothing

            c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

            orc = jit[]
            initial_addr = callback!(orc, c_callback, pointer_from_objref(cc))
            create_stub!(orc, cc.stub, initial_addr)
            return address(orc, cc.stub)
        end
    else # LLVM >=12

        function absolute_symbol_materialization(name, ptr)
            address = LLVM.API.LLVMOrcJITTargetAddress(reinterpret(UInt, ptr))
            flags = LLVM.API.LLVMJITSymbolFlags(LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
            symbol = LLVM.API.LLVMJITEvaluatedSymbol(address, flags)
            gv = LLVM.API.LLVMJITCSymbolMapPair(name, symbol)

            return LLVM.absolute_symbols(Ref(gv))
        end

        function define_absolute_symbol(jd, name)
            ptr = LLVM.find_symbol(name)
            if ptr !== C_NULL
                LLVM.define(jd, absolute_symbol_materialization(name, ptr))
                return true
            end
            return false
        end

        struct CompilerInstance
            jit::LLVM.LLJIT
            lctm::LLVM.LazyCallThroughManager
            ism::LLVM.IndirectStubsManager
        end
        const jit = Ref{CompilerInstance}()

        function __init__()
            optlevel = LLVM.API.LLVMCodeGenLevelDefault
            tm = GPUCompiler.JITTargetMachine(optlevel=optlevel)
            LLVM.asm_verbosity!(tm, true)

            lljit = LLJIT(;tm)

            jd_main = JITDylib(lljit)

            prefix = LLVM.get_prefix(lljit)
            dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
            add!(jd_main, dg)
            if Sys.iswindows() && Int === Int64
                # TODO can we check isGNU?
                define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
            end

            es = ExecutionSession(lljit)

            lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
            ism = LLVM.LocalIndirectStubsManager(triple(lljit))

            jit[] = CompilerInstance(lljit, lctm, ism)
            atexit() do
                ci = jit[]
                dispose(ci.ism)
                dispose(ci.lctm)
                dispose(ci.jit)
            end
        end

        function get_trampoline(job)
            compiler = jit[]
            lljit = compiler.jit
            lctm  = compiler.lctm
            ism   = compiler.ism

            # We could also use one dylib per job
            jd = JITDylib(lljit)

            entry_sym = String(gensym(:entry))
            target_sym = String(gensym(:target))
            flags = LLVM.API.LLVMJITSymbolFlags(
                LLVM.API.LLVMJITSymbolGenericFlagsCallable |
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
            entry = LLVM.API.LLVMOrcCSymbolAliasMapPair(
                mangle(lljit, entry_sym),
                LLVM.API.LLVMOrcCSymbolAliasMapEntry(
                    mangle(lljit, target_sym), flags))

            mu = LLVM.reexports(lctm, ism, jd, Ref(entry))
            LLVM.define(jd, mu)

            # 2. Lookup address of entry symbol
            addr = lookup(lljit, entry_sym)

            # 3. add MU that will call back into the compiler
            sym = LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, target_sym), flags)

            function materialize(mr)
                JuliaContext() do ctx
                    ir, meta = GPUCompiler.compile(:llvm, job; validate=false, ctx)

                    # Rename entry to match target_sym
                    LLVM.name!(meta.entry, target_sym)

                    # So 1. serialize the module
                    buf = convert(MemoryBuffer, ir)

                    # 2. deserialize and wrap by a ThreadSafeModule
                    ThreadSafeContext() do ctx
                        mod = parse(LLVM.Module, buf; ctx=context(ctx))
                        tsm = ThreadSafeModule(mod; ctx)

                        il = LLVM.IRTransformLayer(lljit)
                        LLVM.emit(il, mr, tsm)
                    end
                end

                return nothing
            end

            function discard(jd, sym)
            end

            mu = LLVM.CustomMaterializationUnit(entry_sym, Ref(sym), materialize, discard)
            LLVM.define(jd, mu)
            return addr
        end
    end

    import GPUCompiler: deferred_codegen_jobs
    import ..NativeCompilerParams
    @generated function deferred_codegen(f::F, ::Val{tt}, ::Val{world}) where {F,tt,world}
        # manual version of native_job because we have a function type
        source = methodinstance(F, Base.to_tuple_type(tt), world)
        target = NativeCompilerTarget(; jlruntime=true, llvm_always_inline=true)
        # XXX: do we actually require the Julia runtime?
        #      with jlruntime=false, we reach an unreachable.
        params = NativeCompilerParams()
        config = CompilerConfig(target, params; kernel=false)
        job = CompilerJob(source, config, world)
        # XXX: invoking GPUCompiler from a generated function is not allowed!
        #      for things to work, we need to forward the correct world, at least.

        addr = get_trampoline(job)
        trampoline = pointer(addr)
        id = Base.reinterpret(Int, trampoline)

        deferred_codegen_jobs[id] = job

        quote
            ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
            assume(ptr != C_NULL)
            return ptr
        end
    end

    @generated function abi_call(f::Ptr{Cvoid}, rt::Type{RT}, tt::Type{T}, func::F, args::Vararg{Any, N}) where {T, RT, F, N}
        argtt    = tt.parameters[1]
        rettype  = rt.parameters[1]
        argtypes = DataType[argtt.parameters...]

        argexprs = Union{Expr, Symbol}[]
        ccall_types = DataType[]

        before = :()
        after = :(ret)


        # Note this follows: emit_call_specfun_other
        JuliaContext() do ts_ctx
            ctx = GPUCompiler.unwrap_context(ts_ctx)
            if !isghosttype(F) && !Core.Compiler.isconstType(F)
                isboxed = GPUCompiler.deserves_argbox(F)
                argexpr = :(func)
                if isboxed
                    push!(ccall_types, Any)
                else
                    et = convert(LLVMType, func; ctx)
                    if isa(et, LLVM.SequentialType) # et->isAggregateType
                        push!(ccall_types, Ptr{F})
                        argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                    else
                        push!(ccall_types, F)
                    end
                end
                push!(argexprs, argexpr)
            end

            T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
            T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

            for (source_i, source_typ) in enumerate(argtypes)
                if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                    continue
                end

                argexpr = :(args[$source_i])

                isboxed = GPUCompiler.deserves_argbox(source_typ)
                et = isboxed ? T_prjlvalue : convert(LLVMType, source_typ; ctx)

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
                rt = convert(LLVMType, rettype; ctx)
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
        world = GPUCompiler.codegen_world_age(F, tt)
        ptr = deferred_codegen(f, Val(tt), Val(world))
        abi_call(ptr, rt, tt, f, args...)
    end
end
