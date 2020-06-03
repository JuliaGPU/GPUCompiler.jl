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

# compiler interface and implementations
include("interface.jl")
include("error.jl")
include("native.jl")
include("ptx.jl")
include("gcn.jl")

include("runtime.jl")

# compiler implementation
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

function __init__()
    TimerOutputs.reset_timer!(to)
    InitializeAllTargets()
    InitializeAllTargetInfos()
    InitializeAllAsmPrinters()
    InitializeAllAsmParsers()
    InitializeAllTargetMCs()

    return
end

end # module
