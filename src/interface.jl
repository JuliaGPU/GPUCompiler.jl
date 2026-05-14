# interfaces for defining new compilers

# the definition of a new GPU compiler is typically split in two:
# - a generic compiler that lives in GPUCompiler.jl (e.g., emitting PTX, SPIR-V, etc)
# - a more specific version in a package that targets an environment (e.g. CUDA, ROCm, etc)
#
# the first level of customizability is found in the AbstractCompilerTarget hierarchy,
# with methods and interfaces that can only be implemented within GPUCompiler.jl.
#
# further customization should be put in a concrete instance of the AbstractCompilerParams
# type, and can be used to customize interfaces defined on CompilerJob.


## target

export AbstractCompilerTarget

# container for state handled by targets defined in GPUCompiler.jl

abstract type AbstractCompilerTarget end

source_code(@nospecialize(target::AbstractCompilerTarget)) = "text"

llvm_triple(@nospecialize(target::AbstractCompilerTarget)) = error("Not implemented")

# may return nothing if the target is not support by the current version of LLVM.
function llvm_machine(@nospecialize(target::AbstractCompilerTarget))
    triple = llvm_triple(target)

    t = Target(triple=triple)

    tm = TargetMachine(t, triple)
    asm_verbosity!(tm, true)

    return tm
end

llvm_datalayout(target::AbstractCompilerTarget) = DataLayout(llvm_machine(target))

# a custom `TargetTransformInfo` for targets that don't have (or can't rely on) a
# `TargetMachine`-supplied TTI. Return `nothing` to fall back to LLVM's native TTI.
llvm_targetinfo(@nospecialize(target::AbstractCompilerTarget)) = nothing

# the target's datalayout, with Julia's non-integral address spaces added to it
function julia_datalayout(@nospecialize(target::AbstractCompilerTarget))
    dl = llvm_datalayout(target)
    dl === nothing && return nothing
    DataLayout(string(dl) * "-ni:10:11:12:13")
end

have_fma(@nospecialize(target::AbstractCompilerTarget), T::Type) = false

dwarf_version(target::AbstractCompilerTarget) = Int32(4) # It seems every target supports v4 bar cuda

# If your target performs nested compilation, this function should reconstruct your target with a new inner target
nest_target(target::AbstractCompilerTarget, parent::AbstractCompilerTarget) = target

## params

export AbstractCompilerParams

# container for state handled by external users of GPUCompiler.jl

abstract type AbstractCompilerParams end

nest_params(params::AbstractCompilerParams, parent::AbstractCompilerParams) = params


## config

export CompilerConfig

# the configuration of the compiler

const CONFIG_KWARGS = [:kernel, :name, :entry_abi, :always_inline, :opt_level,
                       :libraries, :optimize, :cleanup, :validate, :strip]

"""
    CompilerConfig(target, params; kernel=true, entry_abi=:specfunc, name=nothing,
                                   always_inline=false)

Construct a `CompilerConfig` that will be used to drive compilation for the given `target`
and `params`.

Several keyword arguments can be used to customize the compilation process:

- `kernel`: specifies if the function should be compiled as a kernel (the default) or as a
   plain function. This toggles certain optimizations, rewrites and validations.
- `name`: the name that will be used for the entrypoint function. If `nothing` (the
   default), the name will be generated automatically.
- `entry_abi`: can be either `:specfunc` (the default), or `:func`.
   - `:specfunc` expects the arguments to be passed in registers, simple return values are
     returned in registers as well, and complex return values are returned on the stack
     using `sret`, the calling convention is `fastcc`.
   - The `:func` abi is simpler with a calling convention of the first argument being the
     function itself (to support closures), the second argument being a pointer to a vector
     of boxed Julia values and the third argument being the number of values, the return
     value will also be boxed. The `:func` abi will internally call the `:specfunc` abi, but
     is generally easier to invoke directly.
- `always_inline` specifies if the Julia front-end should inline all functions into one if
   possible.
- `opt_level`: the optimization level to use (default: 2)
- `debug_level`: the amount of debug information to emit and the verbosity of device-side
   exception reporting (0, 1 or 2; default: the running session's `-g` level).
- `libraries`: link the GPU runtime and `libdevice` libraries (default: true)
- `optimize`: optimize the code (default: true)
- `cleanup`: run cleanup passes on the code (default: true)
- `validate`: enable optional validation of input and outputs (default: true)
- `strip`: strip non-functional metadata and debug information (default: false)
"""
struct CompilerConfig{T,P}
    target::T
    params::P

    kernel::Bool
    name::Union{Nothing,String}
    entry_abi::Symbol
    always_inline::Bool
    opt_level::Int
    debug_level::Int
    libraries::Bool
    optimize::Bool
    cleanup::Bool
    validate::Bool
    strip::Bool

    # internal
    toplevel::Bool
    only_entry::Bool

    function CompilerConfig(target::AbstractCompilerTarget, params::AbstractCompilerParams;
                            kernel=true, name=nothing, entry_abi=:specfunc, toplevel=true,
                            always_inline=false, opt_level=2,
                            debug_level=Base.JLOptions().debug_level, optimize=toplevel,
                            libraries=toplevel, cleanup=toplevel, validate=toplevel,
                            strip=false, only_entry=false)
        if entry_abi ∉ (:specfunc, :func)
            error("Unknown entry_abi=$entry_abi")
        end
        new{typeof(target), typeof(params)}(target, params, kernel, name, entry_abi,
                                            always_inline, opt_level, debug_level, libraries,
                                            optimize, cleanup, validate, strip, toplevel,
                                            only_entry)
    end
end

# copy constructor
function CompilerConfig(cfg::CompilerConfig; target=cfg.target, params=cfg.params,
                        kernel=cfg.kernel, name=cfg.name, entry_abi=cfg.entry_abi,
                        always_inline=cfg.always_inline, opt_level=cfg.opt_level,
                        debug_level=cfg.debug_level, libraries=cfg.libraries,
                        optimize=cfg.optimize, cleanup=cfg.cleanup,
                        validate=cfg.validate, strip=cfg.strip, toplevel=cfg.toplevel,
                        only_entry=cfg.only_entry)
    # deriving a non-toplevel job disables certain features
    # XXX: should we keep track if any of these were set explicitly in the first place?
    #      see how PkgEval does that.
    if !toplevel
        optimize = false
        libraries = false
        cleanup = false
        validate = false
    end
    CompilerConfig(target, params; kernel, entry_abi, name, always_inline, opt_level,
                   debug_level, libraries, optimize, cleanup, validate, strip, toplevel,
                   only_entry)
end

function Base.show(io::IO, @nospecialize(cfg::CompilerConfig{T})) where {T}
    print(io, "CompilerConfig for ", T)
end

function Base.hash(cfg::CompilerConfig, h::UInt)
    h = hash(cfg.target, h)
    h = hash(cfg.params, h)

    h = hash(cfg.kernel, h)
    h = hash(cfg.name, h)
    h = hash(cfg.entry_abi, h)
    h = hash(cfg.always_inline, h)
    h = hash(cfg.opt_level, h)
    h = hash(cfg.debug_level, h)
    h = hash(cfg.libraries, h)
    h = hash(cfg.optimize, h)
    h = hash(cfg.cleanup, h)
    h = hash(cfg.validate, h)
    h = hash(cfg.strip, h)
    h = hash(cfg.toplevel, h)
    h = hash(cfg.only_entry, h)

    return h
end


## job

export CompilerJob

using Core: MethodInstance

# a specific invocation of the compiler, bundling everything needed to generate code

"""
    CompilerJob(source::MethodInstance, config::CompilerConfig, [world=tls_world_age()])

Construct a `CompilerJob` that will be used to drive compilation for the given `source` and
`config` in a given `world`.
"""
struct CompilerJob{T,P}
    source::MethodInstance
    config::CompilerConfig{T,P}
    world::UInt

    CompilerJob(source::MethodInstance, config::CompilerConfig{T,P},
                world=tls_world_age()) where {T,P} =
        new{T,P}(source, config, world)
end

# copy constructor
CompilerJob(job::CompilerJob; source=job.source, config=job.config, world=job.world) =
    CompilerJob(source, config, world)

function Base.hash(job::CompilerJob, h::UInt)
    h = hash(job.source, h)
    h = hash(job.config, h)
    h = hash(job.world, h)

    return h
end


## default definitions that can be overridden to influence GPUCompiler's behavior

# Has the runtime available and does not require special handling
uses_julia_runtime(@nospecialize(job::CompilerJob)) = false

# Optional toggles consulted by the optimization pipeline. Override this method to return
# a `NamedTuple` with any of the following keys (defaults shown):
#
# - `instcombine::Bool = true`: when `false`, the pipeline substitutes `InstSimplifyPass`
#   for `InstCombinePass`, retaining only the simplification subset of the peephole
#   transforms (useful e.g. for downstream rewriters like Enzyme that get confused by
#   InstCombine's more aggressive rewrites).
#
# Returning a `NamedTuple` keeps this single extension point lightweight: downstream
# users add new keys without GPUCompiler having to grow an interface method per option.
optimization_options(@nospecialize(job::CompilerJob)) = (;)

# Is it legal to run vectorization passes on this target
can_vectorize(@nospecialize(job::CompilerJob)) = false

# Should emit PTLS lookup that can be relocated
dump_native(@nospecialize(job::CompilerJob)) = false

# the Julia module to look up target-specific runtime functions in (this includes both
# target-specific functions from the GPU runtime library, like `malloc`, but also
# replacements functions for operations like `Base.sin`)
runtime_module(@nospecialize(job::CompilerJob)) = error("Not implemented")

# check if a function is an intrinsic that can assumed to be always available
isintrinsic(@nospecialize(job::CompilerJob), fn::String) = false

# provide a specific interpreter to use.
@static if HAS_INTEGRATED_CACHE
function get_interpreter(@nospecialize(job::CompilerJob))
    V = results_type(job)
    GPUInterpreter{V}(job.world;
                     method_table_view=maybe_cached(method_table_view(job)),
                     owner=cache_owner(job),
                     inf_params=inference_params(job),
                     opt_params=optimization_params(job))
end
else
function get_interpreter(@nospecialize(job::CompilerJob))
    V = results_type(job)
    cache = get_code_cache(job)
    GPUInterpreter{V}(job.world;
                     method_table_view=maybe_cached(method_table_view(job)),
                     code_cache=cache,
                     inf_params=inference_params(job),
                     opt_params=optimization_params(job))
end
end

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(@nospecialize(job::CompilerJob)) = uses_julia_runtime(job)

# does this target support loading from Julia safepoints?
# if not, safepoints at function entry will not be emitted
can_safepoint(@nospecialize(job::CompilerJob)) = uses_julia_runtime(job)

# the type of the kernel state object, or Nothing if this back-end doesn't need one.
#
# the generated code will be rewritten to include an object of this type as the first
# argument to each kernel, and pass that object to every function that accesses the kernel
# state (possibly indirectly) via the `kernel_state_pointer` function.
kernel_state_type(@nospecialize(job::CompilerJob)) = Nothing

# Does the target need to pass kernel arguments by value?
pass_by_value(@nospecialize(job::CompilerJob)) = true

# Should the target use byref instead of byval+lower_byval for kernel arguments?
# When true, aggregate arguments are passed as pointers with the byref attribute,
# allowing the backend to load fields directly from the argument memory (e.g. kernarg
# segment on AMDGPU) instead of materializing the entire struct via first-class aggregates.
pass_by_ref(@nospecialize(job::CompilerJob)) = false

# whether pointer is a valid call target
valid_function_pointer(@nospecialize(job::CompilerJob), ptr::Ptr{Cvoid}) = false

# Cache partitioning. On Julia 1.11+, the owner is stored on every `CodeInstance` and
# compared via `jl_egal`, so it (and every field) must be immutable for cross-session
# matches (e.g. via package precompilation); custom `target` / `params` types must be
# `struct`s, not `mutable struct`s.
#
# On Julia 1.10, where there's no per-CI owner field, this token only identifies the
# session-local cache partition (see `cached_compilation` / `GLOBAL_CI_CACHES` in
# `deprecated.jl`); the immutability requirement is still useful for stable hashing.
#
# Care is required for anything that impacts:
#   - method_table
#   - inference_params
#   - optimization_params
# The default covers the full target+params instances (so backends with version- or
# arch-specific knobs partition cleanly), `always_inline` (which feeds optimization_params),
# and the method table.
struct GPUCompilerCacheToken{T<:AbstractCompilerTarget, P<:AbstractCompilerParams}
    target::T
    params::P
    always_inline::Bool
    method_table::Core.MethodTable
end

cache_owner(@nospecialize(job::CompilerJob)) =
    GPUCompilerCacheToken(job.config.target, job.config.params,
                          job.config.always_inline, method_table(job))

# The consumer's results struct type, stored on each `CodeInstance` (1.11+) via the
# `CompilerCaching` extension. Override to attach session-portable artifacts (e.g. IR or
# object bytes) and session-local handles (e.g. `CuModule`, `MTLComputePipelineState`).
# The struct must be a `mutable struct` with a zero-arg constructor.
#
# When the consumer hasn't overridden, the default `Nothing` opts out: no results struct
# is attached during inference, and the `analysis_results` chain is untouched. This is the
# right default for reflection paths, precompile workloads, and the legacy 1.10 flow.
results_type(@nospecialize(job::CompilerJob)) = Nothing

"""
    bitcode(results) -> Union{Nothing, Vector{UInt8}}
    bitcode!(results, bytes::Vector{UInt8}) -> Nothing

Optional consumer hooks for stashing post-irgen LLVM bitcode on the results struct.
Used by [`emit_function!`](@ref) to memoize each runtime function's bitcode on its
`CodeInstance` (1.11+, via `analysis_results`); back-ends can populate the slot from
any compile they want to memoize at the LLVM stage.

Override both on your [`results_type`](@ref) to opt in. The default pair (`bitcode` →
`nothing`, `bitcode!` → no-op) means no LLVM-stage caching happens.

The LLVM context's pointer mode (opaque vs. typed) is assumed fixed for the lifetime of
a session — and across precompile/load pairs of the same Julia version, since pkgimages
are invalidated when the toolchain changes. Back-ends don't need to gate caching on
that mode.

On Julia 1.10 these hooks are never invoked (no integrated cache to stash bitcode on);
runtime libraries fall back to the session-local `runtime_libs` cache.
"""
bitcode(@nospecialize(results)) = nothing
bitcode!(@nospecialize(results), bytes::Vector{UInt8}) = nothing

@static if HAS_INTEGRATED_CACHE
    """
        cache_view(job::CompilerJob) -> CompilerCaching.CacheView

    Construct a `CacheView{typeof(cache_owner(job)), results_type(job)}` over Julia's
    integrated `CodeInstance` cache at `job.world`. The handle back-ends pass to
    `CompilerCaching.lookup` / `CompilerCaching.results` when populating their own
    per-CI artifacts.
    """
    function cache_view(@nospecialize(job::CompilerJob))
        owner = cache_owner(job)
        CompilerCaching.CacheView{typeof(owner), results_type(job)}(owner, job.world)
    end
end

@public GPUCompilerCacheToken, cache_owner, results_type, bitcode, bitcode!
@static if HAS_INTEGRATED_CACHE
    @public cache_view
end

# the method table to use
# deprecate method_table on next-breaking release
method_table(@nospecialize(job::CompilerJob)) = GLOBAL_METHOD_TABLE
method_table_view(@nospecialize(job::CompilerJob)) = get_method_table_view(job.world, method_table(job))

# the inference parameters to use when constructing the GPUInterpreter
function inference_params(@nospecialize(job::CompilerJob))
    if VERSION >= v"1.12.0-DEV.1017"
        CC.InferenceParams()
    else
        CC.InferenceParams(; unoptimize_throw_blocks=false)
    end
end

# the optimization parameters to use when constructing the GPUInterpreter
function optimization_params(@nospecialize(job::CompilerJob))
    kwargs = NamedTuple()

    if job.config.always_inline
        kwargs = (kwargs..., inline_cost_threshold=Int(CC.MAX_INLINE_COST))
    end

    return CC.OptimizationParams(;kwargs...)
end

# how much debuginfo to emit
function llvm_debug_info(@nospecialize(job::CompilerJob))
    if job.config.debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif job.config.debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif job.config.debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
end


## extension points at important stages of compilation

# prepare the environment for compilation of a job. this can involve, e.g.,
# priming the cache with entries that cannot be easily inferred.
prepare_job!(@nospecialize(job::CompilerJob)) = return

# early extension point used to link-in external bitcode files.
# this is typically used by downstream packages to link vendor libraries.
link_libraries!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = return

# finalization of the module, before deferred codegen and optimization
finish_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function) =
    entry

# finalization of linked modules, after deferred codegen but before optimization
finish_linked_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = return

# post-Julia optimization processing of the module
optimize_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = return

# final processing of the IR, right before validation and machine-code generation
finish_ir!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function) =
    entry

# whether an LLVM function is valid for this back-end
validate_ir(@nospecialize(job::CompilerJob), mod::LLVM.Module) = IRError[]
