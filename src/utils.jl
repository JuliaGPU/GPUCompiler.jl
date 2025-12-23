defs(mod::LLVM.Module)  = filter(f -> !isdeclaration(f), collect(functions(mod)))
decls(mod::LLVM.Module) = filter(f ->  isdeclaration(f) && !LLVM.isintrinsic(f),
                                 collect(functions(mod)))

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


## safe logging

using Logging

const STDERR_HAS_COLOR = Ref{Bool}(false)

# Prevent invalidation when packages define custom loggers
# Using invoke in combination with @nospecialize eliminates backedges to these methods
function _invoked_min_enabled_level(@nospecialize(logger))
    return invoke(Logging.min_enabled_level, Tuple{typeof(logger)}, logger)::LogLevel
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


## constant expression pruning

# for some reason, after cloning the LLVM IR can contain unused constant expressions.
# these result in false positives when checking that values are unused and can be deleted.
# this helper function removes such unused constant expression uses of a value.
# the process needs to be recursive, as constant expressions can refer to one another.
function prune_constexpr_uses!(root::LLVM.Value)
    for use in uses(root)
        val = user(use)
        if val isa ConstantExpr
            prune_constexpr_uses!(val)
            isempty(uses(val)) && LLVM.unsafe_destroy!(val)
        end
    end
end


## kernel metadata handling

# kernels are encoded in the IR using the julia.kernel metadata.

# IDEA: don't only mark kernels, but all jobs, and save all attributes of the CompileJob
#       so that we can reconstruct the CompileJob instead of setting it globally

# mark a function as kernel
function mark_kernel!(f::LLVM.Function)
    mod = LLVM.parent(f)
    push!(metadata(mod)["julia.kernel"], MDNode([f]))
    return f
end

# iterate over all kernels in the module
function kernels(mod::LLVM.Module)
    vals = LLVM.Function[]
    if haskey(metadata(mod), "julia.kernel")
        kernels_md = metadata(mod)["julia.kernel"]
        for kernel_md in operands(kernels_md)
            push!(vals, LLVM.Value(operands(kernel_md)[1]))
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
## method overrides

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)
using ExprTools: splitdef, combinedef
macro device_function(rt, ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that warns and returns the expected type
    # FIXME: The type may not have a default constructor, what do we do then?
    #        Currently we are using the constructor with an Int64(1) as an argument.
    # NOTE: using Int64(1) is a bit odd. This is because Ptr(Int64(0)) == C_NULL, and julia code lowering
    #       seems to get rid of this automatically.
    def[:body] = quote
        @warn "This function is not intended for use on the CPU something may have gone wrong"
        $rt(1)
    end

    esc(quote
        $(combinedef(def))

        # NOTE: no use of `@consistent_overlay` here because the regular function errors
        Base.Experimental.@overlay($(GPUCompiler).GLOBAL_METHOD_TABLE, $ex)
    end)
end


