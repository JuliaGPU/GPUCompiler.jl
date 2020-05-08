# error handling

export KernelError, InternalCompilerError

struct KernelError <: Exception
    job::CompilerJob
    message::String
    help::Union{Nothing,String}
    bt::StackTraces.StackTrace

    KernelError(job::CompilerJob, message::String, help=nothing;
                bt=StackTraces.StackTrace()) =
        new(job, message, help, bt)
end

function Base.showerror(io::IO, err::KernelError)
    println(io, "GPU compilation of ", err.job.source, " failed")
    println(io, "KernelError: $(err.message)")
    println(io)
    println(io, something(err.help, "Try inspecting the generated code with any of the @device_code_... macros."))
    Base.show_backtrace(io, err.bt)
end


struct InternalCompilerError <: Exception
    job::CompilerJob
    message::String
    meta::Dict
    InternalCompilerError(job, message; kwargs...) = new(job, message, kwargs)
end

function Base.showerror(io::IO, err::InternalCompilerError)
    println(io, """GPUCompiler.jl encountered an unexpected internal error.
                   Please file an issue attaching the following information, including the backtrace,
                   as well as a reproducible example (if possible).""")

    println(io, "\nInternalCompilerError: $(err.message)")

    println(io, "\nCompiler invocation: ", err.job)

    if !isempty(err.meta)
        println(io, "\nAdditional information:")
        for (key,val) in err.meta
            println(io, " - $key = $(repr(val))")
        end
    end

    let Pkg = Base.require(Base.PkgId(Base.UUID((0x44cfe95a1eb252ea, 0xb672e2afdf69b78f)), "Pkg"))
        println(io, "\nInstalled packages:")
        for (pkg,ver) in Pkg.installed()
            println(io, " - $pkg = $(repr(ver))")
        end
    end

    println(io)
    versioninfo(io)
end

macro compiler_assert(ex, job, kwargs...)
    msg = "$ex, at $(__source__.file):$(__source__.line)"
    return :($(esc(ex)) ? $(nothing)
                        : throw(InternalCompilerError($(esc(job)), $msg;
                                                      $(map(esc, kwargs)...)))
            )
end
