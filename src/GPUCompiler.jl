module GPUCompiler

using LLVM
using LLVM.Interop

using TimerOutputs

using ExprTools: splitdef, combinedef

using Libdl

using Scratch: @get_scratch!

const CC = Core.Compiler
using Core: MethodInstance, CodeInstance, CodeInfo

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
include("execution.jl")
include("reflection.jl")


include("precompile.jl")
include("precompilation_cache.jl")

_precompile_()

function __init__()
    STDERR_HAS_COLOR[] = get(stderr, :color, false)

    global compile_cache = @get_scratch!("compiled")
end

end # module
