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
    @static if hasmethod(Core.Compiler.typeinf_type, Tuple{CC.AbstractInterpreter, MethodInstance})
        rt = Core.Compiler.typeinf_type(interp, mi)
    else
        # Julia 1.10: only the 4-arg form exists; reconstruct it from the MI.
        method = mi.def::Method
        rt = Core.Compiler.typeinf_type(interp, method, mi.specTypes, mi.sparam_vals)
    end
    return something(rt, Any)
end

function check_method(@nospecialize(job::CompilerJob))
    ft = job.source.specTypes.parameters[1]
    ft <: Core.Builtin && error("$(unsafe_function_from_type(ft)) is not a generic function")

    for sparam in job.source.sparam_vals
        if sparam isa (__has_internal_julia_change(v"1.14-alpha", :svectvar) ? Core.SimpleVector : TypeVar)
            throw(KernelError(job, "method captures typevar '$sparam' (you probably use an unbound type variable)"))
        end
    end

    # kernels can't return values
    if job.config.kernel
        rt = typeinf_type(job.source; interp=get_interpreter(job))

        if rt != Nothing && rt != Union{}
            throw(KernelError(job, "kernel returns a value of type `$rt`",
                """Make sure your kernel function ends in `return`, `return nothing` or `nothing`."""))
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

        # Before `Core.TypeEgal`, `Type{T}` is only a singleton when `T` has a unique
        # representation, so e.g. `Type{Union{Missing, Int}}` or `Type{Vector}` would be
        # passed as a boxed host pointer.
        if Base.isType(dt)
            throw(KernelError(job, "passing a non-singleton type argument",
                """Argument $arg_i to your kernel function is the type $(dt.parameters[1]), which
                   cannot be passed to a GPU kernel on this version of Julia.
                   Pass `Val($(dt.parameters[1]))` instead, or a value of that type."""))
        end

        # If an object doesn't have fields, it can only be used by identity, so we can allow
        # them to be passed to the GPU (this also applies to e.g. Symbols).
        if fieldcount(dt) == 0
            continue
        end

        if !isbitstype(dt)
            throw(KernelError(job, "passing non-bitstype argument",
                """Argument $arg_i to your kernel function is of type $dt, which is not a bitstype:
                   $(explain_nonisbits(dt))

                   Only bitstypes, which are "plain data" types that are immutable
                   and contain no references to other values, can be used in GPU kernels.
                   For more information, see the `Base.isbitstype` function."""))
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
const NONCONST_GLOBAL  = "use of a non-constant global"
const DYNAMIC_CALL     = "dynamic function invocation"
const UNKNOWN_INTRINSIC = "call to an unknown LLVM intrinsic"
const UNSUPPORTED_ALLOCATION = "allocation of an object with references"

function show_reason(io::IO, (kind, bt, meta)::IRError)
    prefix = kind == STATIC_ASSERTION ? "Reason: $kind" : "Reason: unsupported $kind"
    printstyled(io, "\n$prefix"; color=:red)
    if meta !== nothing
        if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION || kind == POINTER_FUNCTION || kind == DYNAMIC_CALL || kind == CCALL_FUNCTION || kind == LAZY_FUNCTION
            printstyled(io, " (call to ", meta, ")"; color=:red)
        elseif kind == DELAYED_BINDING
            printstyled(io, " (use of '", meta, "')"; color=:red)
        elseif kind == NONCONST_GLOBAL
            printstyled(io, " (", meta, ")"; color=:red)
        elseif kind == STATIC_ASSERTION
            printstyled(io, " (", meta, ")"; color=:red)
        elseif kind == UNSUPPORTED_ALLOCATION
            printstyled(io, " (", meta, ")"; color=:red)
        end
    end
    Base.show_backtrace(io, bt)
end

# Calls that codegen emits into Julia's runtime support on its own, e.g., for exception
# handling, GC frames or boxing. They are invalid too, but where they occur alongside another
# error, that one is usually what needs fixing, and they are not worth listing one by one.
function is_runtime_call((kind, bt, meta)::IRError)
    kind in (RUNTIME_FUNCTION, UNKNOWN_FUNCTION, POINTER_FUNCTION, LAZY_FUNCTION,
             CCALL_FUNCTION) || return false
    (meta isa AbstractString || meta isa Symbol) || return false
    name = String(meta)
    return startswith(name, "jl_") || startswith(name, "ijl_") || startswith(name, "julia.")
end

# The frames of the call in the outermost function that an error originates from, or `nothing`
# if the backtrace does not reach that function.
function error_origin(bt::StackTraces.StackTrace)
    isempty(bt) && return nothing
    any(frame -> frame.func === Symbol("multiple call sites"), bt) && return nothing
    return bt[max(end-1, 1):end]
end

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling ", err.job.source, " resulted in invalid LLVM IR")

    # group errors by where they originate from, keeping them in order
    groups = Dict{Any,Vector{IRError}}()
    origins = []
    for error in err.errors
        origin = error_origin(error[2])
        key = origin === nothing ? nothing : [(frame.func, frame.file, frame.line) for frame in origin]
        if !haskey(groups, key)
            groups[key] = IRError[]
            push!(origins, (key, origin))
        end
        push!(groups[key], error)
    end

    has_other_errors = any(!is_runtime_call, err.errors)
    collapsed = 0
    for (key, origin) in origins
        errors = groups[key]
        # only collapse runtime calls where there are other errors to look at
        others = filter(!is_runtime_call, errors)
        if isempty(others) && (origin !== nothing || !has_other_errors)
            foreach(error -> show_reason(io, error), errors)
            continue
        end
        foreach(error -> show_reason(io, error), others)
        runtime_calls = filter(is_runtime_call, errors)
        isempty(runtime_calls) && continue
        names = unique(String(error[3]) for error in runtime_calls)
        shown = names[1:min(end, 5)]
        printstyled(io, "\nReason: unsupported calls into the Julia runtime from the same code (",
                    join(shown, ", "), length(names) > length(shown) ? ", …" : "", ")";
                    color=:red)
        if origin === nothing
            print(io, "\nin functions with several callers")
        else
            Base.show_backtrace(io, origin)
        end
        collapsed += length(runtime_calls)
    end
    if collapsed > 0
        print(io, "\n\n", collapsed, " of the ", length(err.errors),
              " errors were summarized; they are all listed in the `errors` field of this exception.")
    end

    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": catch this exception as `err` and call `code_typed(err; interactive = true)` to",
        " introspect the erroneous code with Cthulhu.jl";
        color = :cyan,
    )
    return
end

# `show` via `showerror`, avoiding the default field-dump that derefs disposed IR
Base.show(io::IO, err::InvalidIRError) = showerror(io, err)

function check_ir(job, mod::LLVM.Module, relocs::Relocations=Relocations())
    errors = check_ir!(job, IRError[], mod, relocs)
    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(job, errors))
    end

    return
end

function check_ir!(job, errors::Vector{IRError}, mod::LLVM.Module, relocs::Relocations)
    for f in mod.functions
        check_ir!(job, errors, f, relocs)
    end

    # custom validation
    append!(errors, validate_ir(job, mod))

    return errors
end

function check_ir!(job, errors::Vector{IRError}, f::LLVM.Function, relocs::Relocations)
    dl = f.parent.datalayout
    for bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.CallInst)
            check_ir!(job, errors, inst, relocs)
        elseif isa(inst, LLVM.LoadInst)
            check_ir!(job, errors, inst)
        end
        if (isa(inst, LLVM.LoadInst) || isa(inst, LLVM.StoreInst)) && is_binding_access(inst)
            binding = accessed_binding(inst, relocs, dl)
            if binding === nothing
                @safe_debug "Decoding the binding of a global access failed" inst bb=inst.parent
                push!(errors, (NONCONST_GLOBAL, backtrace(inst), nothing))
            else
                gr = binding.globalref
                push!(errors, (global_access_error(job, gr), backtrace(inst), gr))
            end
        end
    end

    return errors
end

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.LoadInst)
    bt = backtrace(inst)
    src = inst.operands[1]
    if src isa ConstantExpr
        if src.opcode == LLVM.Opcode.BitCast
            src = src.operands[1]
        end
    end
    if src isa GlobalVariable
        name = src.name
        if startswith(name, "jlplt_")
            try
                rx = r"jlplt_(.*)_\d+_got"
                name = match(rx, name).captures[1]
                push!(errors, (LAZY_FUNCTION, bt, name))
            catch e
                @safe_debug "Decoding name of PLT entry failed" inst bb=inst.parent
                push!(errors, (LAZY_FUNCTION, bt, nothing))
            end
        end
    end
    return errors
end

# Codegen accesses a global at run time when it is not a defined constant in the job's world.
# Tell apart names that are undefined from globals that are defined but not constant.
function global_access_error(@nospecialize(job::CompilerJob), gr::GlobalRef)
    defined = Base.invoke_in_world(job.world, isdefined, gr.mod, gr.name)
    return defined ? NONCONST_GLOBAL : DELAYED_BINDING
end

const BINDING_VALUE_OFFSET = fieldoffset(Core.Binding, Base.fieldindex(Core.Binding, :value))

# Codegen reads and writes non-constant globals through a pointer to the binding's value,
# without calling into the runtime. Often nothing else gives the access away: Julia 1.10 does
# not check the read of a global that was assigned at compile time, and on targets that cannot
# throw, the check for an undefined value is lowered to an exception like any other. Codegen
# tags these accesses with a TBAA type it uses for nothing else.
is_binding_access(inst::LLVM.Instruction) = tbaa_type(inst) == "jtbaa_binding"

# The binding whose value a load or store accesses, or `nothing` if it cannot be identified.
function accessed_binding(inst::Union{LLVM.LoadInst,LLVM.StoreInst}, relocs::Relocations,
                          dl::LLVM.DataLayout)
    ptr = inst.pointer_operand
    offset = 0
    while true
        ptr = strip_pointer_casts(ptr)
        ptr isa LLVM.GetElementPtrInst ||
            (ptr isa ConstantExpr && ptr.opcode == LLVM.Opcode.GetElementPtr) || break
        delta = LLVM.constant_offset(Int, ptr, dl)
        delta === nothing && return nothing
        offset += delta
        ptr = ptr.operands[1]
    end

    obj = if ptr isa ConstantExpr && ptr.opcode == LLVM.Opcode.IntToPtr
        # a literal address, as emitted by Julia 1.10 or resolved from a relocation
        addr = first(ptr.operands)
        addr isa ConstantInt || return nothing
        ref = object_at(convert(UInt, addr) + (offset - BINDING_VALUE_OFFSET) % UInt, relocs)
        ref === nothing ? nothing : something(ref)
    elseif ptr isa LLVM.LoadInst && offset == BINDING_VALUE_OFFSET
        # a relocation slot
        ref = referenced_object(ptr, relocs)
        ref === nothing ? nothing : something(ref)
    else
        nothing
    end
    return obj isa Core.Binding ? obj : nothing
end

# the name of the TBAA type of a memory access, or `nothing`
function tbaa_type(inst::LLVM.Instruction)
    md = LLVM.metadata(inst)
    haskey(md, LLVM.MD_tbaa) || return nothing
    tag = md[LLVM.MD_tbaa]
    # struct-path tags are `!{base type, access type, offset}`, and types `!{name, ...}`
    ops = LLVM.operands(tag)
    length(ops) >= 2 || return nothing
    access = ops[2]
    access isa LLVM.MDNode || return nothing
    name = first(LLVM.operands(access))
    name isa LLVM.MDString || return nothing
    return convert(String, name)
end

# the contents of a constant string global, or `nothing`
function constant_string(val::LLVM.Value)
    while val isa LLVM.ConstantExpr
        val = first(val.operands)
    end
    val isa LLVM.GlobalVariable || return nothing
    init = val.initializer
    init === nothing && return nothing
    isstring(init) || return nothing
    return rstrip(String(init), '\0')
end

# Julia's codegen replaces an `llvmcall` of an intrinsic it doesn't know, e.g. one that was
# removed from LLVM, with a call to `jl_error`, deferring the error to run time.
function is_unknown_intrinsic_error(call::LLVM.CallInst)
    dest = call.called_operand
    dest isa LLVM.Function || return false
    dest.name in ("jl_error", "ijl_error") || return false
    args = call.arguments
    length(args) == 1 || return false
    return constant_string(args[1]) == "llvmcall only supports intrinsic calls"
end

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.CallInst, relocs::Relocations)
    bt = backtrace(inst)
    dest = inst.called_operand
    if isa(dest, LLVM.Function)
        fn = dest.name

        # some special handling for runtime functions that we don't implement
        if fn == STATIC_ASSERT_MARKER
            push!(errors, (STATIC_ASSERTION, bt, static_assert_message(inst)))
        elseif is_unknown_intrinsic_error(inst)
            push!(errors, (UNKNOWN_INTRINSIC, bt, nothing))
        elseif fn == UNSUPPORTED_ALLOCATION_MARKER
            push!(errors, (UNSUPPORTED_ALLOCATION, bt, static_assert_message(inst)))
        elseif fn == "jl_get_binding_or_error" || fn == "ijl_get_binding_or_error"
            try
                m, sym = inst.arguments
                ref = referenced_object(sym, relocs)
                ref === nothing && error("Unknown binding")
                push!(errors, (DELAYED_BINDING, bt, something(ref)))
            catch e
                @safe_debug "Decoding arguments to jl_get_binding_or_error failed" inst bb=inst.parent
                push!(errors, (DELAYED_BINDING, bt, nothing))
            end
        elseif fn == "jl_reresolve_binding_value_seqcst" || fn == "ijl_reresolve_binding_value_seqcst" ||
               fn == "jl_get_binding_value_seqcst" || fn == "ijl_get_binding_value_seqcst"
            try
                # pry the binding from the IR
                ref = referenced_object(inst.arguments[1], relocs)
                ref === nothing && error("Unknown binding")
                gr = something(ref).globalref
                push!(errors, (global_access_error(job, gr), bt, gr))
            catch e
                @safe_debug "Decoding arguments to jl_reresolve_binding_value_seqcst failed" inst bb=inst.parent
                push!(errors, (DELAYED_BINDING, bt, nothing))
            end
        elseif startswith(fn, "tojlinvoke")
            try
                fun, args, nargs = inst.arguments
                ref = referenced_object(fun, relocs)
                ref === nothing && error("Unknown function")
                fun = something(ref)::Base.Function
                push!(errors, (DYNAMIC_CALL, bt, fun))
                # XXX: an invoke trampoline happens when codegen doesn't have access to code
                #      which suggests a GPUCompiler.jl bug. throw an error instead?
            catch e
                @safe_debug "Decoding arguments to jl_invoke failed" inst bb = inst.parent
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end
        elseif fn == "jl_invoke" || fn == "ijl_invoke"
            # most invokes are contained in a trampoline handled above,
            # but some direct ones remain (e.g., with `@nospecialize`)
            # XXX: this shouldn't be true on 1.12+ anymore; jl_invoke is always trampolined
            caller = inst.parent.parent
            if startswith(caller.name, "tojlinvoke")
                return
            end
            try
                fun, args, nargs, meth = inst.arguments
                ref = referenced_object(meth, relocs)
                ref === nothing && error("Unknown method instance")
                meth = something(ref)::Core.MethodInstance
                push!(errors, (DYNAMIC_CALL, bt, meth.def))
            catch e
                @safe_debug "Decoding arguments to jl_invoke failed" inst bb=inst.parent
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end
        elseif fn == "jl_apply_generic" || fn == "ijl_apply_generic"
            try
                f, args, nargs = inst.arguments
                ref = referenced_object(f, relocs)
                ref === nothing && error("Unknown function")
                f = something(ref)
                push!(errors, (DYNAMIC_CALL, bt, f))
            catch e
                @safe_debug "Decoding arguments to jl_apply_generic failed" inst bb=inst.parent
                push!(errors, (DYNAMIC_CALL, bt, nothing))
            end

        elseif fn == "jl_load_and_lookup" || fn == "ijl_load_and_lookup"
            try
                f_lib, f_name, hnd = inst.arguments
                name_value = constant_string(f_name)
                name_value === nothing && error("Unknown function name")
                push!(errors, (CCALL_FUNCTION, bt, String(name_value)))
            catch e
                @safe_debug "Decoding arguments to jl_load_and_lookup failed" inst bb=inst.parent
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
                push!(errors, (RUNTIME_FUNCTION, bt, dest.name))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, dest.name))
            end
        end

    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM

    elseif isa(dest, ConstantExpr)
        # detect calls to literal pointers
        if dest.opcode == LLVM.Opcode.IntToPtr
            # extract the literal pointer
            ptr_arg = first(dest.operands)
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

# helper function to check for illegal values in an LLVM module
function check_ir_values(mod::LLVM.Module, predicate, msg="value")
    errors = IRError[]
    for fun in mod.functions, bb in fun.blocks, inst in bb.instructions
        if predicate(inst) || any(predicate, inst.operands)
            bt = backtrace(inst)
            # snapshot to a string: the error may outlive the module, and showing a
            # disposed LLVM value segfaults
            push!(errors, (msg, bt, string(inst)))
        end
    end
    return errors
end
## shorthand to check for illegal value types
function check_ir_values(mod::LLVM.Module, T_bad::LLVMType)
    check_ir_values(mod, val -> val.value_type == T_bad, "use of $(string(T_bad)) value")
end
