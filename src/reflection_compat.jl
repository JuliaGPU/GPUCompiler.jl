# The content of this file should be upstreamed to Julia proper

@inline function typed_signature(@nospecialize(job::CompilerJob))
    u = Base.unwrap_unionall(job.source.tt)
    return Base.rewrap_unionall(Tuple{job.source.f, u.parameters...}, job.source.tt)
end

function method_instances(@nospecialize(tt::Type), world::UInt=Base.get_world_counter())
    return map(Core.Compiler.specialize_method, method_matches(tt; world))
end

function code_warntype_by_type(io::IO, @nospecialize(tt);
                       debuginfo::Symbol=:default, optimize::Bool=false, kwargs...)
    debuginfo = Base.IRShow.debuginfo(debuginfo)
    lineprinter = Base.IRShow.__debuginfo[debuginfo]
    for (src, rettype) in Base.code_typed_by_type(tt; optimize, kwargs...)
        if !(src isa Core.CodeInfo)
            println(io, src)
            println(io, "  failed to infer")
            continue
        end
        lambda_io::IOContext = io
        p = src.parent
        nargs::Int = 0
        if p isa Core.MethodInstance
            println(io, p)
            print(io, "  from ")
            println(io, p.def)
            p.def isa Method && (nargs = p.def.nargs)
            if !isempty(p.sparam_vals)
                println(io, "Static Parameters")
                sig = p.def.sig
                warn_color = Base.warn_color() # more mild user notification
                for i = 1:length(p.sparam_vals)
                    sig = sig::UnionAll
                    name = sig.var.name
                    val = p.sparam_vals[i]
                    print_highlighted(io::IO, v::String, color::Symbol) =
                        if highlighting[:warntype]
                            Base.printstyled(io, v; color)
                        else
                            Base.print(io, v)
                        end
                    if val isa TypeVar
                        if val.lb === Union{}
                            print(io, "  ", name, " <: ")
                            print_highlighted(io, "$(val.ub)", warn_color)
                        elseif val.ub === Any
                            print(io, "  ", sig.var.name, " >: ")
                            print_highlighted(io, "$(val.lb)", warn_color)
                        else
                            print(io, "  ")
                            print_highlighted(io, "$(val.lb)", warn_color)
                            print(io, " <: ", sig.var.name, " <: ")
                            print_highlighted(io, "$(val.ub)", warn_color)
                        end
                    elseif val isa typeof(Vararg)
                        print(io, "  ", name, "::")
                        print_highlighted(io, "Int", warn_color)
                    else
                        print(io, "  ", sig.var.name, " = ")
                        print_highlighted(io, "$(val)", :cyan) # show the "good" type
                    end
                    println(io)
                    sig = sig.body
                end
            end
        end
        if src.slotnames !== nothing
            slotnames = Base.sourceinfo_slotnames(src)
            lambda_io = IOContext(lambda_io, :SOURCE_SLOTNAMES => slotnames)
            slottypes = src.slottypes
            nargs > 0 && println(io, "Arguments")
            for i = 1:length(slotnames)
                if i == nargs + 1
                    println(io, "Locals")
                end
                print(io, "  ", slotnames[i])
                if isa(slottypes, Vector{Any})
                    InteractiveUtils.warntype_type_printer(io, slottypes[i], true)
                end
                println(io)
            end
        end
        print(io, "Body")
        InteractiveUtils.warntype_type_printer(io, rettype, true)
        println(io)
@static if VERSION < v"1.7.0"
        Base.IRShow.show_ir(lambda_io, src, lineprinter(src), InteractiveUtils.warntype_type_printer)
else
        irshow_config = Base.IRShow.IRShowConfig(lineprinter(src), InteractiveUtils.warntype_type_printer)
        Base.IRShow.show_ir(lambda_io, src, irshow_config)
end
        println(io)
    end
    nothing
end
