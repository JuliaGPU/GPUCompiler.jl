module Example
using GPUCompiler
using SimpleGPU
SimpleGPU.@declare_cache()

f(x) = 1
SimpleGPU.precompile_simple(f, (Int, ))

function __init__()
    SimpleGPU.@reinit_cache()
end

SimpleGPU.@snapshot_cache()

end # module Example
