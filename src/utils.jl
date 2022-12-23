defs(mod::LLVM.Module)  = filter(f -> !isdeclaration(f), collect(functions(mod)))
decls(mod::LLVM.Module) = filter(f ->  isdeclaration(f) && !LLVM.isintrinsic(f),
                                 collect(functions(mod)))

## timings

const to = TimerOutput()

timings() = (TimerOutputs.print_timer(to); println())

enable_timings() = (TimerOutputs.enable_debug_timings(GPUCompiler); return)


## debug verification

should_verify() = ccall(:jl_is_debugbuild, Cint, ()) == 1 ||
                  Base.JLOptions().debug_level >= 2 ||
                  parse(Bool, get(ENV, "CI", "false"))


## lazy module loading

using UUIDs

struct LazyModule
    pkg::Base.PkgId
    LazyModule(name, uuid) = new(Base.PkgId(uuid, name))
end

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

# define safe loggers for use in generated functions (where task switches are not allowed)
for level in [:debug, :info, :warn, :error]
    @eval begin
        macro $(Symbol("safe_$level"))(ex...)
            macrocall = :(@placeholder $(ex...))
            # NOTE: `@placeholder` in order to avoid hard-coding @__LINE__ etc
            macrocall.args[1] = Symbol($"@$level")
            quote
                old_logger = global_logger()
                io = IOContext(Core.stderr, :color=>STDERR_HAS_COLOR[])
                min_level = Logging.min_enabled_level(old_logger)
                global_logger(Logging.ConsoleLogger(io, min_level))
                ret = $(esc(macrocall))
                global_logger(old_logger)
                ret
            end
        end
    end
end



## codegen locking

# lock codegen to prevent races on the LLVM context.
#
# XXX: it's not allowed to switch tasks while under this lock, can we guarantee that?
#      its probably easier to start using our own LLVM context when that's possible.
macro locked(ex)
    def = splitdef(ex)
    def[:body] = quote
        if VERSION >= v"1.9.0-DEV.1308"
            ccall(:jl_typeinf_lock_begin, Cvoid, ())
        else
            ccall(:jl_typeinf_begin, Cvoid, ())
        end
        try
            $(def[:body])
        finally
            if VERSION >= v"1.9.0-DEV.1308"
                ccall(:jl_typeinf_lock_end, Cvoid, ())
            else
                ccall(:jl_typeinf_end, Cvoid, ())
            end
        end
    end
    esc(combinedef(def))
end

# HACK: temporarily unlock again to perform a task switch
macro unlocked(ex)
    def = splitdef(ex)
    def[:body] = quote
        if VERSION >= v"1.9.0-DEV.1308"
            ccall(:jl_typeinf_lock_end, Cvoid, ())
        else
            ccall(:jl_typeinf_end, Cvoid, ())
        end
        try
            $(def[:body])
        finally
            if VERSION >= v"1.9.0-DEV.1308"
                ccall(:jl_typeinf_lock_begin, Cvoid, ())
            else
                ccall(:jl_typeinf_begin, Cvoid, ())
            end
        end
    end
    esc(combinedef(def))
end

function callsite_attribute!(call, attributes)
    # TODO: Make a nice API for this in LLVM.jl
    for attribute in attributes
        LLVM.API.LLVMAddCallSiteAttribute(call, LLVM.API.LLVMAttributeFunctionIndex, attribute)
    end
end
