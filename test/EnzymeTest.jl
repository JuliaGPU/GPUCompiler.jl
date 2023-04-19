module EnzymeTest
using GPUCompiler
using Enzyme

f1(x) = x*x
autodiff_wrapper(f) = first(autodiff(Reverse, f, Active(1.0)))

println("precompilation!")



const cache = let
    cache_snapshot = GPUCompiler.ci_cache_snapshot()
    autodiff_wrapper(f1)
    GPUCompiler.ci_cache_delta(cache_snapshot)
end

__init__() = GPUCompiler.ci_cache_insert(cache)
end # module EnzymeTest
