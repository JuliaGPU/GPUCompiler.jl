module GPUCompiler

using LLVM
using LLVM.Interop

using TimerOutputs

using ExprTools: splitdef, combinedef

using Libdl

const to = TimerOutput()

timings() = (TimerOutputs.print_timer(to); println())

enable_timings() = (TimerOutputs.enable_debug_timings(GPUCompiler); return)

include("utils.jl")

# compiler interface and implementations
include("interface.jl")
include("error.jl")
include("native.jl")
include("ptx.jl")
include("gcn.jl")
include("spirv.jl")
include("bpf.jl")
include("metal.jl")

include("runtime.jl")

# compiler implementation
include("jlgen.jl")
include("irgen.jl")
include("optim.jl")
include("validation.jl")
include("rtlib.jl")
include("mcgen.jl")
include("debug.jl")
include("driver.jl")

# other reusable functionality
include("cache.jl")
include("execution.jl")
include("reflection.jl")

include("precompile.jl")
include("precompile_native.jl")
_precompile_()

# Users of CUDA.jl will need there own partial cache ( why we need keys)
#=
Have a big cache, can load a package after already cached code, what should we 
be putting into the cache after loading package. Need to change how caching works
so we can filter correctly.

Where a function came: package it came from (who entered this thing originally into our cache)
=#
# could be done here
function __init__()
    println("init called")
    @show MY_CACHE
    if !is_precompiling()
        @show MY_CACHE
        reload_cache()
    else
        atexit(snapshot) # might have to do outside the init
    end
    STDERR_HAS_COLOR[] = get(stderr, :color, false)
end

end # module
