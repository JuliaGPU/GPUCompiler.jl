# validation of properties and code

export InvalidIRError

# TODO: upstream
function method_matches(@nospecialize(tt::Type{<:Tuple}); world::Integer)
    methods = Core.MethodMatch[]
    matches = Base._methods_by_ftype(tt, -1, world)
    matches === nothing && return methods
    for match in matches::Vector
        push!(methods, match::Core.MethodMatch)
    end
    return methods
end

function typeinf_type(mi::MethodInstance; interp::CC.AbstractInterpreter)
    ty = Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals)
    return something(ty, Any)
end

function check_method(@nospecialize(job::CompilerJob))
    ft = job.source.specTypes.parameters[1]
    ft <: Core.Builtin && error("$(unsafe_function_from_type(ft)) is not a generic function")

    for sparam in job.source.sparam_vals
        if sparam isa TypeVar
            throw(KernelError(job, "method captures typevar '$sparam' (you probably use an unbound type variable)"))
        end
    end

    # kernels can't return values
    if job.config.kernel
        rt = typeinf_type(job.source; interp=get_interpreter(job))

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
    sig = job.source.specTypes
    ft = sig.parameters[1]
    tt = Tuple{sig.parameters[2:end]...}

    Base.isdispatchtuple(tt) || error("$tt is not a dispatch tuple")

    # make sure any non-isbits arguments are unused
    real_arg_i = 0

    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        Core.Compiler.isconstType(dt) && continue
        real_arg_i += 1

        # XXX: can we support these for CPU targets?
        if dt <: Core.OpaqueClosure
            throw(KernelError(job, "passing an opaque closure",
                """Argument $arg_i to your kernel function is an opaque closure.
                   This is a CPU-only object not supported by GPUCompiler."""))
        end

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
const CCALL_FUNCTION   = "call to an external C function"
const LAZY_FUNCTION    = "call to a lazy-initialized function"
const DELAYED_BINDING  = "use of an undefined name"
const DYNAMIC_CALL     = "dynamic function invocation"

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling ", err.job.source, " resulted in invalid LLVM IR")
    for (kind, bt, meta) in err.errors
        printstyled(io, "\nReason: unsupported $kind"; color=:red)
        if meta !== nothing
            if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION || kind == POINTER_FUNCTION || kind == DYNAMIC_CALL || kind == CCALL_FUNCTION || kind == LAZY_FUNCTION
                printstyled(io, " (call to ", meta, ")"; color=:red)
            elseif kind == DELAYED_BINDING
                printstyled(io, " (use of '", meta, "')"; color=:red)
            end
        end
        Base.show_backtrace(io, bt)
    end
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": catch this exception as `err` and call `code_typed(err; interactive = true)` to",
        " introspect the erronous code with Cthulhu.jl";
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

    # custom validation
    append!(errors, validate_ir(job, mod))

    return errors
end

function check_ir!(job, errors::Vector{IRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            check_ir!(job, errors, inst)
        elseif isa(inst, LLVM.LoadInst)
            check_ir!(job, errors, inst)
        end
    end

    return errors
end

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.LoadInst)
    bt = backtrace(inst)
    src = operands(inst)[1]
    if src isa ConstantExpr
        if opcode(src) == LLVM.API.LLVMBitCast
            src = operands(src)[1]
        end
    end
    if src isa GlobalVariable
        name = LLVM.name(src)
        if startswith(name, "jlplt_")
            try
                rx = r"jlplt_(.*)_\d+_got"
                name = match(rx, name).captures[1]
                push!(errors, (LAZY_FUNCTION, bt, name))
            catch e
                @safe_debug "Decoding name of PLT entry failed" inst bb=LLVM.parent(inst)
                push!(errors, (LAZY_FUNCTION, bt, nothing))
            end
        end
    end
    return errors
end

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.CallInst)
    bt = backtrace(inst)
    dest = called_operand(inst)
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
                @safe_debug "Decoding arguments to jl_get_binding_or_error failed" inst bb=LLVM.parent(inst)
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
                @safe_debug "Decoding arguments to jl_invoke failed" inst bb=LLVM.parent(inst)
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
                @safe_debug "Decoding arguments to jl_apply_generic failed" inst bb=LLVM.parent(inst)
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end

        elseif fn == "jl_load_and_lookup" || fn == "ijl_load_and_lookup"
            try
                f_lib, f_name, hnd = arguments(inst)
                f_name = first(operands(f_name))::GlobalVariable # get rid of the GEP
                name_init = LLVM.initializer(f_name)::ConstantDataSequential
                name_value = map(collect(name_init)) do char
                    convert(UInt8, char)
                end |> String
                name_value = name_value[1:end-1] # remove trailing \0
                push!(errors, (CCALL_FUNCTION, bt, name_value))
            catch e
                @safe_debug "Decoding arguments to jl_load_and_lookup failed" inst bb=LLVM.parent(inst)
                push!(errors, (CCALL_FUNCTION, bt, nothing))
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

# helper function to check if a LLVM module uses values of a certain type
function check_ir_values(mod::LLVM.Module, T_bad::LLVMType)
    errors = IRError[]

    for fun in functions(mod), bb in blocks(fun), inst in instructions(bb)
        if value_type(inst) == T_bad || any(param->value_type(param) == T_bad, operands(inst))
            bt = backtrace(inst)
            push!(errors, ("unsupported use of $(string(T_bad)) value", bt, inst))
        end
    end

    return errors
end
