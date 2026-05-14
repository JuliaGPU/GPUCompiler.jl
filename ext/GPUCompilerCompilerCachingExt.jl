module GPUCompilerCompilerCachingExt

# GPUCompiler ↔ CompilerCaching glue.
#
# CompilerCaching attaches a `CachedResult{V}` to each newly inferred `CodeInstance`'s
# `analysis_results` chain so back-ends can recover typed results on cache hits. Julia
# only allows writing `CodeInstance.analysis_results` during the `CC.finish!` step of
# inference (it's a const field afterwards on 1.12), so the typed-results attachment
# has to live on a `CC.finish!` override.
#
# We define that override here, parametrically over `GPUInterpreter{V, MTV}`'s results
# type `V`. When `V === Nothing` (the default — no consumer override of `results_type`)
# this is a no-op pass-through to the default `CC.finish!`, so loading CompilerCaching
# alongside a no-results back-end has no inference-time cost.
#
# The extension only loads on Julia ≥ 1.11 (CompilerCaching's `__init__` errors on
# 1.10), so the integrated cache and `analysis_results` are guaranteed to exist here.

using GPUCompiler: GPUInterpreter
using CompilerCaching: CachedResult
const CC = Core.Compiler

@static if hasmethod(CC.finish!, Tuple{CC.AbstractInterpreter, CC.InferenceState, UInt, UInt64})
    function CC.finish!(interp::GPUInterpreter{V, MTV}, caller::CC.InferenceState,
                        validation_world::UInt, time_before::UInt64) where {MTV, V}
        V === Nothing || CC.stack_analysis_result!(caller.result, CachedResult{V}(V()))
        @invoke CC.finish!(interp::CC.AbstractInterpreter, caller::CC.InferenceState,
                           validation_world::UInt, time_before::UInt64)
    end
else
    function CC.finish!(interp::GPUInterpreter{V, MTV},
                        caller::CC.InferenceState) where {MTV, V}
        V === Nothing || CC.stack_analysis_result!(caller.result, CachedResult{V}(V()))
        @invoke CC.finish!(interp::CC.AbstractInterpreter, caller::CC.InferenceState)
    end
end

end # module
