defs(mod::LLVM.Module)  = filter(f -> !isdeclaration(f), collect(functions(mod)))
decls(mod::LLVM.Module) = filter(f ->  isdeclaration(f) && !LLVM.isintrinsic(f),
                                 collect(functions(mod)))


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

# define safe loggers for use in generated functions (where task switches are not allowed)
for level in [:debug, :info, :warn, :error]
    @eval begin
        macro $(Symbol("safe_$level"))(ex...)
            macrocall = :(@placeholder $(ex...))
            # NOTE: `@placeholder` in order to avoid hard-coding @__LINE__ etc
            macrocall.args[1] = Symbol($"@$level")
            quote
                old_logger = global_logger()
                # FIXME: Core.stderr supports colors
                io = IOContext(Core.stderr, :color=>stderr[:color])
                global_logger(Logging.ConsoleLogger(io, old_logger.min_level))
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
        ccall(:jl_typeinf_begin, Cvoid, ())
        try
            $(def[:body])
        finally
            ccall(:jl_typeinf_end, Cvoid, ())
        end
    end
    esc(combinedef(def))
end

# HACK: temporarily unlock again to perform a task switch
macro unlocked(ex)
    def = splitdef(ex)
    def[:body] = quote
        ccall(:jl_typeinf_end, Cvoid, ())
        try
            $(def[:body])
        finally
            ccall(:jl_typeinf_begin, Cvoid, ())
        end
    end
    esc(combinedef(def))
end
