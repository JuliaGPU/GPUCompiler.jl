# Julia's LLVM passes, from Julia's code generator or as ported to Julia by JuliaIR.jl

# The default of `CompilerConfig`'s `julia_passes`. Set with
# `Preferences.set_preferences!(GPUCompiler, "julia_passes" => "ports")`.
const JULIA_PASSES = Symbol(@load_preference("julia_passes", "cpp"))
JULIA_PASSES in (:cpp, :ports) ||
    error("Invalid julia_passes preference $(repr(String(JULIA_PASSES))); expected \"cpp\" or \"ports\"")

# The ports of Julia's passes, supplied by the JuliaIRExt extension: a function registering
# them with a `PassBuilder`, and the pipeline name of the port of each of Julia's passes by
# the name of the `LLVM.Interop` constructor of the C++ pass (`nothing` if not ported).
struct PassPorts
    register!::Function
    names::Dict{Symbol,Union{Nothing,String}}
end
const pass_ports = Ref{Union{Nothing,PassPorts}}(nothing)

# The default of `CompilerConfig`'s `julia_passes`: the preference, except while
# precompiling without JuliaIR.jl loaded, e.g., in the workload of a package that does not
# depend on it. That uses the C++ passes; since the setting is part of the configuration,
# what it caches is not used for configurations that select the ports.
function default_julia_passes()
    if JULIA_PASSES === :ports && pass_ports[] === nothing &&
       ccall(:jl_generating_output, Cint, ()) == 1
        return :cpp
    end
    return JULIA_PASSES
end

function check_julia_passes(@nospecialize(job::CompilerJob))
    job.config.julia_passes === :ports || return
    pass_ports[] === nothing &&
        error("julia_passes=:ports requires JuliaIR.jl to be loaded (`using JuliaIR`)")
    supports_typed_pointers(context()) &&
        error("julia_passes=:ports requires a context with opaque pointers; create it with `JuliaContext(; opaque_pointers=true)`")
    return
end

# The `opaque_pointers` argument for contexts to compile in with `julia_passes`: the ports
# only support opaque pointers, which Julia 1.10 and 1.11 do not use by default.
context_opaque_pointers(julia_passes::Symbol) = julia_passes === :ports ? true : nothing

"""
    julia_pass(job, name::Symbol)

The Julia pass `name` (the name of its constructor in `LLVM.Interop`, e.g.
`:AllocOptPass`) for use in the optimization pipeline of `job`: Julia's C++ pass, or its
port in JuliaIR.jl if `job.config.julia_passes == :ports`.

Passes that JuliaIR.jl does not port yet run as the C++ pass in either mode; the
JuliaIRExt extension lists them.
"""
function julia_pass(@nospecialize(job::CompilerJob), name::Symbol)
    if job.config.julia_passes === :ports
        ports = pass_ports[]::PassPorts
        haskey(ports.names, name) || error("Unknown Julia pass $name")
        port = ports.names[name]
        port === nothing || return port
    end
    return getfield(LLVM.Interop, name)()
end

# make the ports usable by name in the pipeline of `pb`
function register_julia_passes!(pb::PassBuilder, @nospecialize(job::CompilerJob))
    job.config.julia_passes === :ports || return
    Base.invokelatest((pass_ports[]::PassPorts).register!, pb)
    return
end
