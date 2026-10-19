# Julia's LLVM passes as ported to Julia by JuliaIR.jl, for jobs that select them with
# `julia_passes=:ports`. Loading this extension alone does not change generated code.
module JuliaIRExt

import GPUCompiler
import JuliaIR

# The port of each of Julia's passes in GPUCompiler's pipeline, by the name of the
# `LLVM.Interop` constructor of the C++ pass. The ones JuliaIR.jl does not port yet are
# `nothing`, and run as the C++ pass.
const PORTS = Dict{Symbol,Union{Nothing,String}}(
    :AllocOptPass                 => "jl-alloc-opt",
    :CombineMulAddPass            => "jl-combine-muladd",
    :DemoteFloat16Pass            => "jl-demote-float16",
    :ExpandAtomicModifyPass       => "jl-expand-atomic-modify",
    :GCInvariantVerifierPass      => "jl-gc-invariant-verifier",
    :LateLowerGCPass              => "jl-late-lower-gc-frame",
    :LowerExcHandlersPass         => "jl-lower-exc-handlers",
    :LowerSIMDLoopPass            => "jl-lower-simdloop",
    :PropagateJuliaAddrspacesPass => "jl-propagate-addrspaces",
    :RemoveNIPass                 => "jl-remove-ni",

    :FinalLowerGCPass             => nothing,
    :JuliaLICMPass                => nothing,
    :LowerPTLSPass                => nothing,
    :RemoveJuliaAddrspacesPass    => nothing,
)

function __init__()
    available = JuliaIR.passes()
    for port in values(PORTS)
        port === nothing || port in available ||
            error("JuliaIR.jl does not provide the pass $port")
    end
    GPUCompiler.pass_ports[] = GPUCompiler.PassPorts(pb -> JuliaIR.register!(pb), PORTS)
end

end # module JuliaIRExt
