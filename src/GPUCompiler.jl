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

compile_cache = nothing # set during __init__()
const pkgver = Base.pkgversion(GPUCompiler)

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

function __init__()
    STDERR_HAS_COLOR[] = get(stderr, :color, false)
    create_compile_cache()
end

function create_compile_cache()
    dir = @get_scratch!("compiled")
    ## add the Julia version
    dir = joinpath(dir, "v$(VERSION.major).$(VERSION.minor)")
    ## also add the package version
    if pkgver !== nothing
        # XXX: `Base.pkgversion` is buggy and sometimes returns `nothing`, see e.g.
        #       JuliaLang/PackageCompiler.jl#896 and JuliaGPU/GPUCompiler.jl#593
        dir = joinpath(dir, "v$(pkgver.major).$(pkgver.minor)")
    end
    mkpath(dir)
    global compile_cache = dir
end

# remove the existing cache
# NOTE: call this function from global scope, so any change triggers recompilation.
function reset_compile_cache()
    dir = @get_scratch!("compiled")
    lock(runtime_lock) do
        rm(dir; force=true, recursive=true)
        create_compile_cache()
    end

    return
end
# backwards compatibility
const reset_runtime = reset_compile_cache

end # module
