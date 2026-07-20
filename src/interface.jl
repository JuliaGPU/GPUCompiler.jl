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


## cache owner

# Inference results are shared by jobs with the same target, params, and inlining policy.
# On 1.11+, store this token on the CompilerConfig (as `Any`) so the immutable value is
# boxed once when the config is created, instead of on every integrated-cache lookup.
struct GPUCompilerCacheToken{T<:AbstractCompilerTarget, P<:AbstractCompilerParams}
    target::T
    params::P
    always_inline::Bool
end

"""
    cache_owner(target, params, always_inline)

Construct the immutable token that partitions inference for a compiler configuration. On
Julia 1.11+, the job-level `cache_owner(job)` returns the pre-boxed token stored on its
`CompilerConfig`.

The default token covers the full `target` and `params` instances plus `always_inline`.
That is sufficient because the inference inputs derived from a job — `method_table`,
`method_table_view`, `inference_params` and `optimization_params` — must be pure
functions of those values, so back-ends normally leave this untouched.

When overriding, the returned value must match under `===`/`jl_egal` after package-image
deserialization: use immutable containers, and only reference mutable objects (like method
tables) that are module-level singletons.
"""
cache_owner(target::AbstractCompilerTarget, params::AbstractCompilerParams,
            always_inline::Bool) =
    GPUCompilerCacheToken(target, params, always_inline)


## config

export CompilerConfig

# the configuration of the compiler

# Keep the 1.10 layout concrete: an `Any` field makes otherwise-inline configs escape when
# used in the legacy results key. Integrated caching needs the pre-boxed owner on 1.11+.
const CompilerConfigCacheOwner = @static if HAS_INTEGRATED_CACHE
    Any
else
    Nothing
end

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

    # Boxed once so calls into Julia's Any-typed integrated cache don't allocate on every hit.
    cache_owner::CompilerConfigCacheOwner

    function CompilerConfig(target::AbstractCompilerTarget, params::AbstractCompilerParams;
                            kernel=true, name=nothing, entry_abi=:specfunc, toplevel=true,
                            always_inline=false, opt_level=2,
                            debug_level=Base.JLOptions().debug_level, optimize=toplevel,
                            libraries=toplevel, cleanup=toplevel, validate=toplevel,
                            strip=false, only_entry=false)
        if entry_abi ∉ (:specfunc, :func)
            error("Unknown entry_abi=$entry_abi")
        end
        owner = @static if HAS_INTEGRATED_CACHE
            cache_owner(target, params, always_inline)
        else
            nothing
        end
        new{typeof(target), typeof(params)}(target, params, kernel, name, entry_abi,
                                            always_inline, opt_level, debug_level, libraries,
                                            optimize, cleanup, validate, strip, toplevel,
                                            only_entry, owner)
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
    # `cache_owner` deliberately is not hashed: its target/params/inlining components were
    # already included above, and a method-table-only difference may safely collide. Hashing
    # the full token here would duplicate the most expensive parts of every config hash.
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
    GPUInterpreter(job.world;
                   method_table_view=maybe_cached(method_table_view(job)),
                   owner=cache_owner(job),
                   inf_params=inference_params(job),
                   opt_params=optimization_params(job))
end
else
function get_interpreter(@nospecialize(job::CompilerJob))
    GPUInterpreter(job.world;
                   method_table_view=maybe_cached(method_table_view(job)),
                   code_cache=get_code_cache(job),
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
# compared via `jl_egal`; custom `target` / `params` containers must be immutable so
# equivalent values can match across package-image deserialization.
#
# On Julia 1.10, where there's no per-CI owner field, this token only identifies the
# session-local `GLOBAL_CI_CACHES` partition in `deprecated.jl`; the immutability
# requirement is still useful for stable hashing.
#
# The token is created and boxed by the CompilerConfig constructor. Returning the boxed field
# is important: passing a freshly-constructed immutable token to the Any-typed C cache API
# allocates on every kernel-cache hit.
@static if HAS_INTEGRATED_CACHE
    cache_owner(job::CompilerJob) = job.config.cache_owner
else
    # Preserve the allocation-free 1.10 path: constructing this specialized immutable token
    # is cheaper than storing an Any-typed box on every CompilerConfig.
    cache_owner(job::CompilerJob) =
        cache_owner(job.config.target, job.config.params, job.config.always_inline)
end

"""
    cached_results(::Type{V}, job::CompilerJob) -> Union{Nothing,V}

Retrieve the compilation-results struct of type `V` for `job`. Returns `nothing` when no
code has been compiled (or inferred) for the job yet; once code exists, an empty `V` is
created on first access and the same struct is returned ever after. `V` must be a
`mutable struct` with a zero-arg constructor; back-ends define one such struct holding
the artifacts of each compilation stage, check whether the relevant fields are
populated, and fill them in (running the compiler) when not:

```julia
mutable struct MetalResults
    metallib::Union{Nothing,Vector{UInt8}}
    entry::Union{Nothing,String}
    pipelines::Vector{Tuple{MTLDevice,MTLComputePipelineState}}  # session-local
    MetalResults() = new(nothing, nothing, [])
end

res = GPUCompiler.cached_results(MetalResults, job)
if res === nothing || res.metallib === nothing || GPUCompiler.compile_hook[] !== nothing
    artifacts = ...compile...
    res = @something res GPUCompiler.cached_results(MetalResults, job)
    ...populate res from artifacts...
end
```

Compiling the job (through `GPUCompiler.compile`) populates Julia's code cache, so the
post-compile lookup in the example is guaranteed to succeed. To attach results without
generating code — e.g. from an inference-only precompilation workload — run
`precompile(job)` first.

Results are keyed by the *full* compiler job: its method instance, world age, and entire
`CompilerConfig` (two jobs differing only in, say, the kernel `name` get distinct
structs). Storage differs per Julia version, transparently to the back-end:

- On Julia 1.11+, the struct lives on the `CodeInstance`'s `analysis_results` chain in
  Julia's integrated code cache, partitioned by [`cache_owner`](@ref) and keyed by
  config within the per-CI [`JobResults`](@ref) container. Method redefinition
  invalidates the CI — and with it the attached results. Artifacts stored during
  precompilation are serialized into the package image along with the CI: populate only
  session-portable values (bytes, strings) during precompile workloads, and keep
  session-local handles (device modules, pipeline objects) in fields that remain empty
  until first use at run time.

- On Julia 1.10, the struct lives in a session-local store keyed by the foreign
  `CodeInstance` and config. Nothing persists across sessions.

Thread safety: concurrent calls for the same job return the same struct, but
GPUCompiler does not serialize back-end *compilation*; take a back-end lock around
the check-and-compile sequence (all back-ends already do, e.g. `mtlfunction_lock`).
"""
function cached_results end

@static if HAS_INTEGRATED_CACHE

"""
    JobResults

Per-CodeInstance container mapping a `CompilerConfig` to the back-end's results struct.

A `CodeInstance` is shared by every job whose [`cache_owner`](@ref) matches — the owner
token only covers what affects *inference* (target, params, `always_inline`). The
remaining `CompilerConfig` fields (`kernel`, `name`, `entry_abi`,
`opt_level`, …) do affect *codegen*, so artifacts must not be shared across them.
Entries are matched by `===` (`jl_egal`): `CompilerConfig` is an immutable struct, so
structurally identical configs compare equal, including against configs deserialized
from package images.
"""
# `CompilerConfig` is an abstract UnionAll here. Keeping it in a tuple stored inline in a
# Vector boxes the config again on every iteration. A non-isbits entry object is stored by
# reference, so both fields are boxed once and hot-path scans allocate nothing.
struct JobResultEntry
    config::CompilerConfig
    value::Any
end

mutable struct JobResults
    entries::Vector{JobResultEntry}
    JobResults() = new(JobResultEntry[])
end

const cached_results_lock = ReentrantLock()

# NOTE: like `cache_owner`, specialized for the launch hot path (bounded number of
#       instantiations: one per back-end and results type).
function job_results(::Type{V}, ci::CodeInstance, config::CompilerConfig) where {V}
    jr = CompilerCaching.results(JobResults, ci)
    Base.@lock cached_results_lock begin
        for entry in jr.entries
            entry.config === config && entry.value isa V && return entry.value::V
        end
        v = V()
        push!(jr.entries, JobResultEntry(config, v))
        return v
    end
end

function cache_view(@nospecialize(job::CompilerJob))
    # `cache_owner` is deliberately stored as Any on CompilerConfig: preserve that box in the
    # CacheView instead of re-specializing and re-boxing the immutable token for the ccall.
    CompilerCaching.CacheView{Any,JobResults}(cache_owner(job), job.world)
end

# Fetch the CodeInstance backing `job` from the integrated cache. On 1.14+, inference
# caches vararg methods under their compilable (vararg-widened) MethodInstance, which
# for such methods differs from the fully-specialized `job.source`: retry there.
function job_code_instance(@nospecialize(job::CompilerJob))
    cache = cache_view(job)
    ci = get(cache, job.source, nothing)
    @static if VERSION >= v"1.14-"
        if ci === nothing
            mi = ccall(:jl_normalize_to_compilable_mi, Any, (Any,),
                       job.source)::MethodInstance
            mi === job.source || (ci = get(cache, mi, nothing))
        end
    end
    return ci
end

# Deliberately lookup-only: when no CI exists, back-ends run codegen (which drives
# inference itself) and re-fetch, instead of us running a standalone inference walk here
# that the compiler's own walk would immediately repeat.
function cached_results(::Type{V}, job::CompilerJob) where {V}
    ci = job_code_instance(job)
    ci === nothing && return nothing
    return job_results(V, ci, job.config)
end

## session-dependent results
#
# Some compilation results embed session-specific data: `relocate_gvs!` bakes absolute
# pointers into the IR of toplevel jobs that reference `julia.constgv` globals (except
# for slots it can materialize as session-portable device constants), and any
# artifact a back-end derives from that IR (metallib, SPIR-V, ...) inherits them. Such
# results must not survive into a package image, while remaining available for
# within-session lookups during the precompilation process itself. Julia wipes its own
# session-dependent CodeInstance state during serialization (staticdata.c); we
# approximate that with an `atexit` hook, which the runtime invokes *before*
# `jl_write_compiler_output`: right before the image is written, the entries of jobs
# marked session-dependent are deleted from their `JobResults` container, so a later
# session simply recompiles them.

const session_dependent_jobs = Vector{CompilerJob}()
const session_dependent_lock = ReentrantLock()

function mark_session_dependent!(@nospecialize(job::CompilerJob))
    ccall(:jl_generating_output, Cint, ()) == 1 || return
    Base.@lock session_dependent_lock begin
        if isempty(session_dependent_jobs)
            atexit(wipe_session_dependent_results)
        end
        push!(session_dependent_jobs, job)
    end
    return
end

function wipe_session_dependent_results()
    Base.@lock session_dependent_lock begin
        for job in session_dependent_jobs
            ci = job_code_instance(job)
            ci === nothing && continue
            jr = CompilerCaching.results(JobResults, ci)
            Base.@lock cached_results_lock begin
                filter!(entry -> entry.config !== job.config, jr.entries)
            end
        end
        empty!(session_dependent_jobs)
    end
    return
end

end # HAS_INTEGRATED_CACHE

@public GPUCompilerCacheToken, cache_owner, cached_results

# the method table to use
#
# NOTE: these (like `inference_params` and `optimization_params` below) may only depend on
#       the job's world and its config's `target`/`params` values (+ `always_inline`);
#       [`cache_owner`](@ref) relies on that to partition inference results correctly.
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

# Whether to emit gnu_pubnames
llvm_gnu_pubnames(@nospecialize(job::CompilerJob)) = false

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

# post-runtime-intrinsic-lowering processing of the module
finish_runtime_intrinsics!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = false

# final processing of the IR, right before validation and machine-code generation
finish_ir!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function) =
    entry

# whether an LLVM function is valid for this back-end
validate_ir(@nospecialize(job::CompilerJob), mod::LLVM.Module) = IRError[]
