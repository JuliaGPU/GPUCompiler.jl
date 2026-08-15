module GPUCompiler

using LLVM
using LLVM.Interop


using ExprTools: splitdef, combinedef

using Libdl

using Preferences

const ENABLE_TRACY = parse(Bool, @load_preference("tracy", "false"))

"""
    enable_tracy!(state::Bool=true)

Activate tracy in the current environment.
You will need to restart your Julia environment for it to take effect.
"""
function enable_tracy!(state::Bool=true)
    @set_preferences!("tracy"=>string(state))
end

if ENABLE_TRACY
    using Tracy
else
    macro tracepoint(name, expr)
        return esc(expr)
    end
end

const CC = Core.Compiler
using Core: MethodInstance, CodeInstance, CodeInfo

# `HAS_INTEGRATED_CACHE` distinguishes 1.11+ (owner-keyed `Core.Compiler.InternalCodeCache`)
# from 1.10 (per-interpreter `CodeCache` IdDict + invalidation callbacks). The two have
# disjoint shapes; everything cache-related fans on this flag.
const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

# Loads as an empty shell on 1.10; on 1.11+ provides `CacheView`, `typeinf!`,
# `get_codeinfos`, `lookup`, `results`, etc. We `import` the module name (not its
# exports) to avoid clashing with `LLVM.lookup`; internal call sites qualify with
# `CompilerCaching.`.
import CompilerCaching

# Optional callback invoked from `compile(...)` / `cached_compilation(...)` before
# compilation runs. Set by `@device_code_*` reflection macros. Defined here (early)
# so the legacy `cached_compilation` in deprecated.jl can reference it.
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

include("utils.jl")
include("mangling.jl")

# compiler interface and implementations
include("interface.jl")
include("relocation.jl")
include("error.jl")
include("native.jl")
include("ptx.jl")
include("gcn.jl")
include("spirv.jl")
include("bpf.jl")
include("metal.jl")

include("runtime.jl")

# compiler implementation
include("deprecated.jl")
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
include("static_assert.jl")
include("reflection.jl")

include("precompile.jl")

function __init__()
    STDERR_HAS_COLOR[] = get(stderr, :color, false)
    empty!(session_results_cache)

    @static if !HAS_INTEGRATED_CACHE
        # CodeInstances created by GPUCompiler's precompile workload are process-local.
        empty!(GLOBAL_CI_CACHES)
    end

    @static if ENABLE_TRACY
        Tracy.@register_tracepoints()
    end
    register_deferred_codegen()
end

end # module
