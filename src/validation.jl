# validation of properties and code

export InvalidIRError

function method_matches(@nospecialize(tt::Type{<:Tuple}); world=Base.get_world_counter())
    ms = Core.MethodMatch[]
    for m in Base._methods_by_ftype(tt, -1, world)::Vector
        m = m::Core.MethodMatch
        push!(ms, m)
    end

    return ms
end

function return_type(m::Core.MethodMatch;
                     interp = Core.Compiler.NativeInterpreter(world))
    ty = Core.Compiler.typeinf_type(interp, m.method, m.spec_types, m.sparams)
    return something(ty, Any)
end


function check_method(@nospecialize(job::CompilerJob))
    isa(job.source.f, Core.Builtin) && throw(KernelError(job, "function is not a generic function"))

    # get the method
    world = job.source.world
    ms = method_matches(typed_signature(job); world)
    isempty(ms)   && throw(KernelError(job, "no method found"))
    length(ms)!=1 && throw(KernelError(job, "no unique matching method"))

    # kernels can't return values
    if job.source.kernel
        cache = ci_cache(job)
        mt = method_table(job)
        interp = GPUInterpreter(cache, mt, world)
        rt = return_type(only(ms); interp)

        if rt != Nothing
            throw(KernelError(job, "kernel returns a value of type `$rt`",
                """Make sure your kernel function ends in `return`, `return nothing` or `nothing`.
                   If the returned value is of type `Union{}`, your Julia code probably throws an exception.
                   Inspect the code with `@device_code_warntype` for more details."""))
        end
    end

    return
end

# The actual check is rather complicated
# and might change from version to version...
function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function explain_nonisbits(@nospecialize(dt), depth=1; maxdepth=10)
    dt===Module && return ""    # work around JuliaLang/julia#33347
    depth > maxdepth && return ""
    hasfieldcount(dt) || return ""
    msg = ""
    for (ft, fn) in zip(fieldtypes(dt), fieldnames(dt))
        if !isbitstype(ft)
            msg *= "  "^depth * ".$fn is of type $ft which is not isbits.\n"
            msg *= explain_nonisbits(ft, depth+1)
        end
    end
    return msg
end

function check_invocation(@nospecialize(job::CompilerJob))
    # make sure any non-isbits arguments are unused
    real_arg_i = 0

    sig = typed_signature(job)

    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        Core.Compiler.isconstType(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            throw(KernelError(job, "passing and using non-bitstype argument",
                """Argument $arg_i to your kernel function is of type $dt, which is not isbits:
                    $(explain_nonisbits(dt))"""))
        end
    end

    return
end


## IR validation

const IRError = Tuple{String, StackTraces.StackTrace, Any} # kind, bt, meta

struct InvalidIRError <: Exception
    job::CompilerJob
    errors::Vector{IRError}
end

const RUNTIME_FUNCTION = "call to the Julia runtime"
const UNKNOWN_FUNCTION = "call to an unknown function"
const POINTER_FUNCTION = "call through a literal pointer"
const DELAYED_BINDING  = "use of an undefined name"
const DYNAMIC_CALL     = "dynamic function invocation"

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling ", err.job.source, " resulted in invalid LLVM IR")
    for (kind, bt, meta) in err.errors
        print(io, "\nReason: unsupported $kind")
        if meta !== nothing
            if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION || kind == POINTER_FUNCTION || kind == DYNAMIC_CALL
                print(io, " (call to ", meta, ")")
            elseif kind == DELAYED_BINDING
                print(io, " (use of '", meta, "')")
            end
        end
        Base.show_backtrace(io, bt)
    end
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": catch this exception as `err` and call `code_typed(err; interactive = true)` to",
        " introspect the erronous code";
        color = :cyan,
    )
    return
end

function check_ir(job, args...)
    errors = check_ir!(job, IRError[], args...)
    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(job, errors))
    end

    return
end

function check_ir!(job, errors::Vector{IRError}, mod::LLVM.Module)
    for f in functions(mod)
        check_ir!(job, errors, f)
    end

    return errors
end

function check_ir!(job, errors::Vector{IRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            check_ir!(job, errors, inst)
        end
    end

    return errors
end

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.CallInst)
    bt = backtrace(inst)
    dest = called_value(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # some special handling for runtime functions that we don't implement
        if fn == "jl_get_binding_or_error" || fn == "ijl_get_binding_or_error"
            try
                m, sym = arguments(inst)
                sym = first(operands(sym::ConstantExpr))::ConstantInt
                sym = convert(Int, sym)
                sym = Ptr{Cvoid}(sym)
                sym = Base.unsafe_pointer_to_objref(sym)
                push!(errors, (DELAYED_BINDING, bt, sym))
            catch e
                @debug "Decoding arguments to jl_get_binding_or_error failed" inst bb=LLVM.parent(inst)
                push!(errors, (DELAYED_BINDING, bt, nothing))
            end
        elseif fn == "jl_invoke" || fn == "ijl_invoke"
            try
                f, args, nargs, meth = arguments(inst)
                meth = first(operands(meth::ConstantExpr))::ConstantInt
                meth = convert(Int, meth)
                meth = Ptr{Cvoid}(meth)
                meth = Base.unsafe_pointer_to_objref(meth)::Core.MethodInstance
                push!(errors, (DYNAMIC_CALL, bt, meth.def))
            catch e
                @debug "Decoding arguments to jl_invoke failed" inst bb=LLVM.parent(inst)
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end
        elseif fn == "jl_apply_generic" || fn == "ijl_apply_generic"
            try
                f, args, nargs = arguments(inst)
                f = first(operands(f))::ConstantInt # get rid of inttoptr
                f = convert(Int, f)
                f = Ptr{Cvoid}(f)
                f = Base.unsafe_pointer_to_objref(f)
                push!(errors, (DYNAMIC_CALL, bt, f))
            catch e
                @debug "Decoding arguments to jl_apply_generic failed" inst bb=LLVM.parent(inst)
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end

        # detect calls to undefined functions
        elseif isdeclaration(dest) && !LLVM.isintrinsic(dest) && !isintrinsic(job, fn)
            # figure out if the function lives in the Julia runtime library
            if libjulia[] == C_NULL
                paths = filter(Libdl.dllist()) do path
                    name = splitdir(path)[2]
                    startswith(name, "libjulia")
                end
                libjulia[] = Libdl.dlopen(first(paths))
            end

            if Libdl.dlsym_e(libjulia[], fn) != C_NULL
                push!(errors, (RUNTIME_FUNCTION, bt, LLVM.name(dest)))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, LLVM.name(dest)))
            end
        end

    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM

    elseif isa(dest, ConstantExpr)
        # detect calls to literal pointers
        if opcode(dest) == LLVM.API.LLVMIntToPtr
            # extract the literal pointer
            ptr_arg = first(operands(dest))
            @compiler_assert isa(ptr_arg, ConstantInt) job
            ptr_val = convert(Int, ptr_arg)
            ptr = Ptr{Cvoid}(ptr_val)

            if !valid_function_pointer(job, ptr)
                # look it up in the Julia JIT cache
                frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint,), ptr, 0)
                # XXX: what if multiple frames are returned? rare, but happens
                if length(frames) == 1
                    fn, file, line, linfo, fromC, inlined = last(frames)
                    push!(errors, (POINTER_FUNCTION, bt, fn))
                else
                    push!(errors, (POINTER_FUNCTION, bt, nothing))
                end
            end
        end
    end

    return errors
end
