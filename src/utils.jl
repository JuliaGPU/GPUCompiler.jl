## julia compat
if VERSION >= v"1.12"
    __has_internal_julia_change(version_or::VersionNumber, feature::Symbol) =
        Base.__has_internal_change(version_or, feature)
else
    __has_internal_julia_change(version_or::VersionNumber, feature::Symbol) =
        false
end


## `public` keyword compat

"""
    @public foo, bar

Declare `foo, bar` as public API. Lowers to `public foo, bar` on 1.11+ (where `public`
is keyword syntax) and to a no-op on 1.10.
"""
macro public(symbols_expr)
    syms = symbols_expr isa Symbol ? [symbols_expr] :
           symbols_expr.head === :tuple ? [a isa Symbol ? a : a.args[1] for a in symbols_expr.args] :
           [symbols_expr.args[1]]
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, syms...))
    else
        nothing
    end
end


## debug verification

should_verify() = ccall(:jl_is_debugbuild, Cint, ()) == 1 ||
                  Base.JLOptions().debug_level >= 2 ||
                  something(tryparse(Bool, get(ENV, "CI", "false")), true)

isdebug(group, mod=GPUCompiler) =
    Base.CoreLogging.current_logger_for_env(Base.CoreLogging.Debug, group, mod) !== nothing


## lazy module loading

using UUIDs

struct LazyModule
    pkg::Base.PkgId
    LazyModule(name, uuid) = new(Base.PkgId(uuid, name))
end

isavailable(lazy_mod::LazyModule) = haskey(Base.loaded_modules, getfield(lazy_mod, :pkg))

function Base.getproperty(lazy_mod::LazyModule, sym::Symbol)
    pkg = getfield(lazy_mod, :pkg)
    mod = get(Base.loaded_modules, pkg, nothing)
    if mod === nothing
        error("This functionality requires the $(pkg.name) package, which should be installed and loaded first.")
    end
    getfield(mod, sym)
end


## external back-ends

# The LLVM back-ends for PTX, GCN and SPIR-V, the Khronos SPIR-V translator and the
# bitcode downgrader ship as symbol-hidden shared libraries with a small C API modelled
# after llvm-c: status as the return value, an error message through an out-pointer on
# failure, results as opaque memory buffers, and diagnostics that the tools used to print
# to stderr delivered through a callback. The libraries share the shape of this API,
# differing only in the prefix of their entry points, so this is implemented once against
# function pointers.

struct ExternalBackend
    library::String     # path to the shared library, as exported by its JLL
    prefix::String      # e.g. "NVPTX" for `NVPTXCompile`
end

function api(backend::ExternalBackend, name::String)
    # `dlopen` returns the handle of an already-loaded library, so this is only a lookup
    Libdl.dlsym(Libdl.dlopen(backend.library), Symbol(backend.prefix * name))
end

function external_diagnostic(severity::Cint, message::Cstring, ctx::Ptr{Cvoid})
    diagnostics = unsafe_pointer_to_objref(ctx)::Vector{String}
    # errors are folded into the failure message by the back-end; only keep what would
    # otherwise be lost (warnings, remarks and notes).
    severity == 0 || push!(diagnostics, unsafe_string(message))
    return nothing
end

# compile `input` (bitcode) with the back-end's `Compile` entry point. `options` is a
# reference to the back-end's option struct, whose string fields the caller keeps alive.
function external_compile(backend::ExternalBackend, input::Vector{UInt8}, options::Ref,
                          what::String; warn=(msg)->@warn(msg))
    diagnostics = String[]
    buffer = Ref{Ptr{Cvoid}}(C_NULL)
    message = Ref{Cstring}(C_NULL)
    compile = api(backend, "Compile")
    handler = @cfunction(external_diagnostic, Cvoid, (Cint, Cstring, Ptr{Cvoid}))
    status = GC.@preserve diagnostics begin
        @ccall $compile(input::Ptr{UInt8}, length(input)::Csize_t, options::Ptr{Cvoid},
                        handler::Ptr{Cvoid}, pointer_from_objref(diagnostics)::Ptr{Cvoid},
                        buffer::Ptr{Ptr{Cvoid}}, message::Ptr{Cstring})::Cint
    end
    external_result(backend, status, what, message[], diagnostics, input; warn)
    return take_buffer(backend, buffer[])
end

# report the outcome of a back-end invocation: raise `what` on failure with the error
# message and a bitcode file to attach, and surface collected diagnostics otherwise.
function external_result(backend::ExternalBackend, status, what::String, message::Cstring,
                         diagnostics::Vector{String}, input::Vector{UInt8};
                         warn=(msg)->@warn(msg))
    if status != 0
        path = tempname(cleanup=false) * ".bc"
        write(path, input)
        msg = what
        message == C_NULL || (msg *= ":\n" * take_message(backend, message))
        isempty(diagnostics) || (msg *= "\n" * join(diagnostics, "\n"))
        msg *= "\nIf you think this is a bug, please file an issue and attach $(path)."
        error(msg)
    elseif !isempty(diagnostics)
        warn("The $(backend.prefix) back-end reported:\n" * join(diagnostics, "\n"))
    end
    return
end

# copy out and dispose of an opaque memory buffer
function take_buffer(backend::ExternalBackend, buffer::Ptr{Cvoid})
    start = @ccall $(api(backend, "GetBufferStart"))(buffer::Ptr{Cvoid})::Ptr{UInt8}
    size = @ccall $(api(backend, "GetBufferSize"))(buffer::Ptr{Cvoid})::Csize_t
    data = copy(unsafe_wrap(Array, start, size))
    @ccall $(api(backend, "DisposeMemoryBuffer"))(buffer::Ptr{Cvoid})::Cvoid
    return data
end

# copy out and dispose of a message
function take_message(backend::ExternalBackend, message::Cstring)
    str = unsafe_string(message)
    @ccall $(api(backend, "DisposeMessage"))(message::Cstring)::Cvoid
    return str
end

# serialize a module to bitcode
function bitcode(mod::LLVM.Module)
    io = IOBuffer()
    write(io, mod)
    take!(io)
end


## safe logging

using Logging

const STDERR_HAS_COLOR = Ref{Bool}(false)

# Call into the latest world: custom loggers are usually defined after the compiler,
# which may run in an older world (e.g. through `invoke_in_world`). As a dynamic call,
# combined with @nospecialize, it also avoids invalidation by recording no backedges.
function _invoked_min_enabled_level(@nospecialize(logger))
    return Base.invokelatest(Logging.min_enabled_level, logger)::LogLevel
end

# define safe loggers for use in generated functions (where task switches are not allowed)
for level in [:debug, :info, :warn, :error]
    @eval begin
        macro $(Symbol("safe_$level"))(ex...)
            macrocall = :(@placeholder $(ex...) _file=$(String(__source__.file)) _line=$(__source__.line))
            # NOTE: `@placeholder` in order to avoid hard-coding @__LINE__ etc
            macrocall.args[1] = Symbol($"@$level")
            quote
                io = IOContext(Core.stderr, :color=>STDERR_HAS_COLOR[])
                # ideally we call Logging.shouldlog() here, but that is likely to yield,
                # so instead we rely on the min_enabled_level of the logger.
                # in the case of custom loggers that may be an issue, because,
                # they may expect Logging.shouldlog() getting called, so we use
                # the global_logger()'s min level which is more likely to be usable.
                min_level = _invoked_min_enabled_level(global_logger())
                safe_logger = Logging.ConsoleLogger(io, min_level)
                # using with_logger would create a closure, which is incompatible with
                # generated functions, so instead we reproduce its implementation here
                safe_logstate = Base.CoreLogging.LogState(safe_logger)
                @static if VERSION < v"1.11-"
                    t = current_task()
                    old_logstate = t.logstate
                    try
                        t.logstate = safe_logstate
                        $(esc(macrocall))
                    finally
                        t.logstate = old_logstate
                    end
                else
                    Base.ScopedValues.@with(
                        Base.CoreLogging.CURRENT_LOGSTATE => safe_logstate, $(esc(macrocall))
                    )
                end
            end
        end
    end
end

macro safe_show(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args,
              :(println(Core.stdout, $(sprint(Base.show_unquoted,ex)*" = "),
                                     repr(begin local value = $(esc(ex)) end))))
    end
    isempty(exs) || push!(blk.args, :value)
    return blk
end


## safe deprecation warnings

const depwarn_lock = Threads.SpinLock()
# the frame is a `Ptr{Cvoid}` for compiled frames, or a `Base.InterpreterIP` for
# interpreted ones (e.g., when the deprecated function is called from top level)
const depwarn_seen = Set{Tuple{Union{Ptr{Cvoid},Base.InterpreterIP},Symbol}}()

"""
    safe_depwarn(msg, funcsym; force=false)

A `Base.depwarn` that does not switch tasks, so it can be used where task switches are
illegal: `@locked` regions holding the typeinf lock, generated functions, or abstract
interpreter callbacks. `Base.depwarn` logs through the active logger, whose I/O may
yield; this version writes the warning synchronously to `Core.stderr`, like `@safe_warn`
does. It still attributes the warning to the caller, warns only once per call site, and
throws under `--depwarn=error`. Custom loggers are bypassed, though.
"""
function safe_depwarn(msg, funcsym; force::Bool=false)
    @static if VERSION >= v"1.12.0-DEV.769"
        # compilation does not hold the typeinf lock, so we can warn regularly
        return Base.depwarn(msg, funcsym; force)
    else
        opts = Base.JLOptions()
        if opts.depwarn == 2
            throw(ErrorException(msg))
        end
        force || opts.depwarn == 1 || return

        # respect the verbosity of the global logger, like `@safe_warn` does
        Logging.Warn >= _invoked_min_enabled_level(global_logger()) || return

        # attribute the warning to the caller, like `Base.depwarn`
        # (`backtrace` and `firstcaller` do not switch tasks)
        bt = Base.backtrace()
        frame, caller = Base.firstcaller(bt, funcsym)

        # only warn once per call site. we can't use the logger's `maxlog` for this,
        # since we construct a fresh logger every time
        Base.@lock depwarn_lock begin
            (frame, funcsym) in depwarn_seen && return
            push!(depwarn_seen, (frame, funcsym))
        end

        linfo = caller.linfo
        mod = if linfo isa Core.MethodInstance
            def = linfo.def
            def isa Module ? def : def.module
        else
            Core
        end

        # emit synchronously; writes to `Core.stderr` do not switch tasks
        io = IOContext(Core.stderr, :color => STDERR_HAS_COLOR[])
        logger = Logging.ConsoleLogger(io)
        Logging.handle_message(logger, Logging.Warn, msg, mod, :depwarn,
                               (frame, funcsym), String(caller.file), caller.line;
                               caller)
        return
    end
end


## codegen locking

# lock codegen to prevent races on the LLVM context.
#
# XXX: it's not allowed to switch tasks while under this lock, can we guarantee that?
#      its probably easier to start using our own LLVM context when that's possible.
macro locked(ex)
    if VERSION >= v"1.12.0-DEV.769"
        # no need to handle locking; it's taken care of by the engine
        # as long as we use a correct cache owner token.
        return esc(ex)
    end

    def = splitdef(ex)
    def[:body] = quote
        ccall(:jl_typeinf_lock_begin, Cvoid, ())
        try
            $(def[:body])
        finally
            ccall(:jl_typeinf_lock_end, Cvoid, ())
        end
    end
    esc(combinedef(def))
end

# HACK: temporarily unlock again to perform a task switch
macro unlocked(ex)
    if VERSION >= v"1.12.0-DEV.769"
        return esc(ex)
    end

    def = splitdef(ex)
    def[:body] = quote
        ccall(:jl_typeinf_lock_end, Cvoid, ())
        try
            $(def[:body])
        finally
            ccall(:jl_typeinf_lock_begin, Cvoid, ())
        end
    end
    esc(combinedef(def))
end


## replacing a global with a runtime value

# Replace every use of `gv` with the function-local value `replacement(f)`, then erase it.
# Plain `replace_uses!` can't do this: an instruction is not a valid operand of the constant
# expressions/aggregates the global's address may be folded into (a `getelementptr` onto it,
# an isbits union's `{ptr, i8}` return value), so those constants are expanded into
# instructions first (phi operands materialize in their incoming block). `replacement` is
# invoked once per using function (memoize per-function state such as an entry-block alloca
# in the callback).
function replace_global_with_local!(gv::LLVM.GlobalVariable, replacement)
    convert_users_to_instructions!([gv])
    for use in collect(gv.uses)
        inst = use.user
        inst isa LLVM.Instruction ||
            error("Unexpected use of global '$(gv.name)': $inst")
        f = inst.parent.parent
        # an instruction is visited once per use, but all its uses are replaced at once
        any(==(gv), inst.operands) && replace!(inst.operands, gv => replacement(f))
    end
    @assert isempty(gv.uses) "global '$(gv.name)' still has uses after replacement"
    erase!(gv)
    return
end


## function-signature rewriting

# Several passes need to change a function's signature, which LLVM can't do in place: you create a
# new function, move the body over, and fix up the callers (the same shape as LLVM's
# ArgumentPromotion). These two helpers capture the mechanical scaffolding shared by those passes;
# the parts that genuinely differ between them -- which parameters change, how each is reconstructed
# on entry, attribute handling, and rewriting call sites -- stay in the caller.

# Clone `f` into a new function whose parameter types come from `new_types` (one entry per
# parameter of `f`; `nothing` leaves that parameter's type unchanged). For each changed parameter a
# fresh entry block reconstructs the value the body expects -- of the *original* type -- via
# `reconstruct(builder, new_param, i)`, and the body is cloned to use it, so the body is unchanged.
# The old function is left in place; the caller fixes up attributes and call sites and then drops it
# with `replace_function!`. `changes` is forwarded to `clone_into!`.
function clone_with_converted_args!(mod::LLVM.Module, f::LLVM.Function, new_types::Vector, reconstruct;
                                    changes = LLVM.CloneFunctionChangeType.GlobalChanges)
    ft = f.function_type
    param_types = ft.parameters
    @assert length(new_types) == length(param_types)
    new_ptypes = LLVM.LLVMType[something(new_types[i], pty) for (i, pty) in enumerate(param_types)]
    new_ft = LLVM.FunctionType(ft.return_type, new_ptypes)

    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    new_f.callconv = f.callconv
    for (arg, new_arg) in zip(f.parameters, new_f.parameters)
        new_arg.name = arg.name
    end

    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, LLVM.at_end(entry))
        body_values = LLVM.Value[
            new_types[i] === nothing ? new_f.parameters[i] :
                                       reconstruct(builder, new_f.parameters[i], i)
            for i in 1:length(param_types)]
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => body_values[i] for (i, param) in enumerate(f.parameters))
        value_map[f] = new_f
        clone_into!(new_f, f; value_map, changes)
        br!(builder, new_f.blocks[2])  # fall through to the cloned entry block
    end

    return new_f
end

# Replace `f` with `new_f` once every value use of `f` has been rewritten away. Drops dead
# constant-expression uses on both sides -- including the dead `bitcast(new_f -> old type)` that
# `clone_into!` leaves behind when the signature changes -- hands the name and metadata to `new_f`,
# and erases `f`.
function replace_function!(f::LLVM.Function, new_f::LLVM.Function)
    remove_dead_constant_users!(f)
    @assert isempty(f.uses)
    replace_metadata_uses!(f, new_f)
    take_name!(new_f, f)
    erase!(f)
    remove_dead_constant_users!(new_f)
    return new_f
end


## kernel metadata handling

# kernels are encoded in the IR using the julia.kernel metadata.

# IDEA: don't only mark kernels, but all jobs, and save all attributes of the CompileJob
#       so that we can reconstruct the CompileJob instead of setting it globally

# mark a function as kernel
function mark_kernel!(f::LLVM.Function)
    mod = f.parent
    push!(get!(mod.metadata, "julia.kernel").operands, MDNode([f]))
    return f
end

# iterate over all kernels in the module
function kernels(mod::LLVM.Module)
    vals = LLVM.Function[]
    if haskey(mod.metadata, "julia.kernel")
        kernels_md = mod.metadata["julia.kernel"]
        for kernel_md in kernels_md.operands
            push!(vals, LLVM.Value(kernel_md.operands[1]))
        end
    end
    return vals
end

@static if VERSION < v"1.13.0-DEV.623"
    import Libdl

    const HAS_LLVM_GVS_GLOBALS = Libdl.dlsym(
        unsafe_load(cglobal(:jl_libjulia_handle, Ptr{Cvoid})), :jl_get_llvm_gvs_globals, throw_error=false) !== nothing

    const AL_N_INLINE = 29

    # Mirrors arraylist_t
    mutable struct ArrayList
        len::Csize_t
        max::Csize_t
        items::Ptr{Ptr{Cvoid}}
        _space::NTuple{AL_N_INLINE, Ptr{Cvoid}}

        function ArrayList()
            list = new(0, AL_N_INLINE, Ptr{Ptr{Cvoid}}(C_NULL), ntuple(_ -> Ptr{Cvoid}(C_NULL), AL_N_INLINE))
            list.items = Base.pointer_from_objref(list) + fieldoffset(typeof(list), 4)

            finalizer(list) do list
                if list.items != Base.pointer_from_objref(list) + fieldoffset(typeof(list), 4)
                    Libc.free(list.items)
                end
            end
            return list
        end
    end

    function get_llvm_global_vars(native_code::Ptr{Cvoid})
        gvs_list = ArrayList()
        GC.@preserve gvs_list begin
            p_gvs = Base.pointer_from_objref(gvs_list)
            @ccall jl_get_llvm_gvs_globals(native_code::Ptr{Cvoid}, p_gvs::Ptr{Cvoid})::Nothing
            gvs = Vector{Ptr{LLVM.API.LLVMOpaqueValue}}(undef, gvs_list.len)
            items = Base.unsafe_convert(Ptr{Ptr{LLVM.API.LLVMOpaqueValue}}, gvs_list.items)
            for i in 1:gvs_list.len
                gvs[i] = unsafe_load(items, i)
            end
        end
        return gvs
    end

    function get_llvm_global_inits(native_code::Ptr{Cvoid})
        inits_list = ArrayList()
        GC.@preserve inits_list begin
            p_inits = Base.pointer_from_objref(inits_list)
            @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, p_inits::Ptr{Cvoid})::Nothing
            inits = Vector{Ptr{Cvoid}}(undef, inits_list.len)
            for i in 1:inits_list.len
                inits[i] = unsafe_load(inits_list.items, i)
            end
        end
        return inits
    end
end

"""Whether Julia exposes enough global-variable metadata to emit relocatable IR."""
supports_relocatable_ir() = @static if VERSION >= v"1.13.0-DEV.623"
    true
else
    # `jl_get_llvm_gvs_globals` was backported to 1.10, so the symbol alone is not enough:
    # 1.10's codegen still embeds Julia addresses (as `inttoptr` constants) in the JIT
    # (non-imaging) mode we compile in, instead of emitting the relocatable global
    # declarations the relocation machinery collects. Only 1.11+ emits those declarations.
    VERSION >= v"1.11-" && HAS_LLVM_GVS_GLOBALS
end
