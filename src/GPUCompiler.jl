module GPUCompiler

using LLVM
using LLVM.Interop

using TimerOutputs

using ExprTools: splitdef, combinedef

using Libdl

using Serialization
using Scratch: @get_scratch!
using Preferences

const CC = Core.Compiler
using Core: MethodInstance, CodeInstance, CodeInfo

const use_newpm = LLVM.has_newpm()

include("utils.jl")
include("mangling.jl")

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
_precompile_()



compile_cache = "" # defined in __init__()

function __init__()
    STDERR_HAS_COLOR[] = get(stderr, :color, false)

    dir = @get_scratch!("compiled")
    ## add the Julia version
    dir = joinpath(dir, "v$(VERSION.major).$(VERSION.minor)")
    if VERSION > v"1.9"
        ## also add the package version
        pkgver = Base.pkgversion(GPUCompiler)
        dir = joinpath(dir, "v$(pkgver.major).$(pkgver.minor)")
    end
    mkpath(dir)
    global compile_cache = dir
end

end # module
