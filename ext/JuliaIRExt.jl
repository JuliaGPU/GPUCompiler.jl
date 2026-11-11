# Julia's LLVM passes as ported to Julia by JuliaIR.jl, for jobs that select them with
# `julia_passes=:ports`. Loading this extension alone does not change generated code.
module JuliaIRExt

import GPUCompiler
import JuliaIR

# The port of each of Julia's passes in GPUCompiler's pipeline, by the name of the
# `LLVM.Interop` constructor of the C++ pass (or the name it would have, see
# `GPUCompiler.CPP_PASS_NAMES`).
const PORTS = Dict{Symbol,String}(
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

function __init__()
    available = JuliaIR.passes()
    for port in values(PORTS)
        port in available || error("JuliaIR.jl does not provide the pass $port")
    end
    GPUCompiler.pass_ports[] = GPUCompiler.PassPorts(pb -> JuliaIR.register!(pb), PORTS)
end

end # module JuliaIRExt
