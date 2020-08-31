const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
    function getsym(arg)
        isa(arg, Symbol) && return arg
        @assert isa(arg, GlobalRef)
        return arg.name
    end

    f = get(__bodyfunction__, mnokw, nothing)
    if f === nothing
        fmod = mnokw.module
        # The lowered code for `mnokw` should look like
        #   %1 = mkw(kwvalues..., #self#, args...)
        #        return %1
        # where `mkw` is the name of the "active" keyword body-function.
        ast = Base.uncompressed_ast(mnokw)
        if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
            callexpr = ast.code[end-1]
            if isa(callexpr, Expr) && callexpr.head == :call
                fsym = callexpr.args[1]
                if isa(fsym, Symbol)
                    f = getfield(fmod, fsym)
                elseif isa(fsym, GlobalRef)
                    if fsym.mod === Core && fsym.name === :_apply
                        f = getfield(mnokw.module, getsym(callexpr.args[2]))
                    elseif fsym.mod === Core && fsym.name === :_apply_iterate
                        f = getfield(mnokw.module, getsym(callexpr.args[3]))
                    else
                        f = getfield(fsym.mod, fsym.name)
                    end
                else
                    f = missing
                end
            else
                f = missing
            end
        else
            f = missing
        end
        __bodyfunction__[mnokw] = f
    end
    return f
end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    @assert precompile(Tuple{typeof(GPUCompiler.assign_args!),Expr,Vector{Any}})
    @assert precompile(Tuple{typeof(GPUCompiler.hide_trap!),LLVM.Module})
    @assert precompile(Tuple{typeof(GPUCompiler.hide_unreachable!),LLVM.Function})
    @assert precompile(Tuple{typeof(GPUCompiler.lower_gc_frame!),LLVM.Function})
    @assert precompile(Tuple{typeof(GPUCompiler.lower_throw!),LLVM.Module})
    @assert precompile(Tuple{typeof(GPUCompiler.split_kwargs),Tuple{},Vector{Symbol},Vararg{Vector{Symbol}, N} where N})
    let fbody = try __lookup_kwbody__(which(GPUCompiler.compile, (Symbol,GPUCompiler.CompilerJob,))) catch missing end
        if !ismissing(fbody)
            @assert precompile(fbody, (Bool,Bool,Bool,Bool,Bool,Bool,typeof(GPUCompiler.compile),Symbol,GPUCompiler.CompilerJob,))
        end
    end
end
