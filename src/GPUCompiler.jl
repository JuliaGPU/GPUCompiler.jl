module GPUCompiler

using LLVM
using LLVM.Interop

using DataStructures

using TimerOutputs

using Libdl

const to = TimerOutput()

timings() = (TimerOutputs.print_timer(to); println())

enable_timings() = (TimerOutputs.enable_debug_timings(GPUCompiler); return)

include("utils.jl")

include("interface.jl")
include("error.jl")
include("ptx.jl")

include("runtime.jl")

include("irgen.jl")
include("optim.jl")
include("validation.jl")
include("rtlib.jl")
include("mcgen.jl")
include("debug.jl")
include("driver.jl")
include("cache.jl")
include("reflection.jl")

function __init__()
    TimerOutputs.reset_timer!(to)
    InitializeAllTargets()
    InitializeAllTargetInfos()
    InitializeAllAsmPrinters()
    InitializeAllTargetMCs()

    return
end

end # module
