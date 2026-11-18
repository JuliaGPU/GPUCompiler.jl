# Julia's LLVM passes, from Julia's code generator or as ported to Julia by JuliaIR.jl

# The default of `CompilerConfig`'s `julia_passes`. Set with
# `Preferences.set_preferences!(GPUCompiler, "julia_passes" => "ports")`.
const JULIA_PASSES = Symbol(@load_preference("julia_passes", "cpp"))
JULIA_PASSES in (:cpp, :ports) ||
    error("Invalid julia_passes preference $(repr(String(JULIA_PASSES))); expected \"cpp\" or \"ports\"")

function check_julia_passes(@nospecialize(job::CompilerJob))
    job.config.julia_passes === :ports || return
    supports_typed_pointers(context()) &&
        error("julia_passes=:ports requires a context with opaque pointers; create it with `JuliaContext(; opaque_pointers=true)`")
    return
end

# The `opaque_pointers` argument for contexts to compile in with `julia_passes`: the ports
# only support opaque pointers, which Julia 1.10 and 1.11 do not use by default.
context_opaque_pointers(julia_passes::Symbol) = julia_passes === :ports ? true : nothing

# The port of each of Julia's passes in the pipeline, by the name of the `LLVM.Interop`
# constructor of the C++ pass (or the name it would have, see `CPP_PASS_NAMES`).
const PASS_PORTS = Dict{Symbol,String}(
    :AllocOptPass                 => "jl-alloc-opt",
    :CancellationLoweringPass     => "jl-cancellation-lowering",
    :CombineMulAddPass            => "jl-combine-muladd",
    :DemoteFloat16Pass            => "jl-demote-float16",
    :ExpandAtomicModifyPass       => "jl-expand-atomic-modify",
    :FinalLowerGCPass             => "jl-final-lower-gc",
    :GCInvariantVerifierPass      => "jl-gc-invariant-verifier",
    :JuliaLICMPass                => "jl-licm",
    :LateLowerGCPass              => "jl-late-lower-gc-frame",
    :LowerExcHandlersPass         => "jl-lower-exc-handlers",
    :LowerPTLSPass                => "jl-lower-ptls",
    :LowerSIMDLoopPass            => "jl-lower-simdloop",
    :PropagateJuliaAddrspacesPass => "jl-propagate-addrspaces",
    :RemoveJuliaAddrspacesPass    => "jl-remove-addrspaces",
    :RemoveNIPass                 => "jl-remove-ni",
)

"""
    julia_pass(job, name::Symbol)

The Julia pass `name` (the name of its constructor in `LLVM.Interop`, e.g.
`:AllocOptPass`) for use in the optimization pipeline of `job`: Julia's C++ pass, or its
port in JuliaIR.jl if `job.config.julia_passes == :ports`. Passes that `LLVM.Interop` has
no constructor for are named as if it had one, e.g. `:CancellationLoweringPass`.
"""
function julia_pass(@nospecialize(job::CompilerJob), name::Symbol)
    if job.config.julia_passes === :ports
        haskey(PASS_PORTS, name) || error("Unknown Julia pass $name")
        return PASS_PORTS[name]
    end
    isdefined(LLVM.Interop, name) && return getfield(LLVM.Interop, name)()
    haskey(CPP_PASS_NAMES, name) || error("Unknown Julia pass $name")
    return CPP_PASS_NAMES[name]
end

# Julia's passes that `LLVM.Interop` has no constructor for: their names in the pipelines
# of Julia's code generator, which registers them with every `PassBuilder`.
const CPP_PASS_NAMES = Dict{Symbol,String}(
    :CancellationLoweringPass => "CancellationLowering",
)

# make the ports usable by name in the pipeline of `pb`
function register_julia_passes!(pb::PassBuilder, @nospecialize(job::CompilerJob))
    job.config.julia_passes === :ports || return
    JuliaIR.register!(pb)
    return
end
