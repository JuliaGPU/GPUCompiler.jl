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
                            always_inline=false, opt_level=2, optimize=toplevel,
                            libraries=toplevel, cleanup=toplevel, validate=toplevel,
                            strip=false, only_entry=false)
        if entry_abi ∉ (:specfunc, :func)
            error("Unknown entry_abi=$entry_abi")
        end
        new{typeof(target), typeof(params)}(target, params, kernel, name, entry_abi,
                                            always_inline, opt_level, libraries, optimize,
                                            cleanup, validate, strip, toplevel, only_entry)
    end
end

# copy constructor
function CompilerConfig(cfg::CompilerConfig; target=cfg.target, params=cfg.params,
                        kernel=cfg.kernel, name=cfg.name, entry_abi=cfg.entry_abi,
                        always_inline=cfg.always_inline, opt_level=cfg.opt_level,
                        libraries=cfg.libraries, optimize=cfg.optimize, cleanup=cfg.cleanup,
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
                   libraries, optimize, cleanup, validate, strip, toplevel, only_entry)
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
if VERSION >= v"1.11.0-DEV.1552"
get_interpreter(@nospecialize(job::CompilerJob)) =
    GPUInterpreter(job.world; method_table_view=maybe_cached(method_table_view(job)),
                   token=ci_cache_token(job), inf_params=inference_params(job),
                   opt_params=optimization_params(job))
else
get_interpreter(@nospecialize(job::CompilerJob)) =
    GPUInterpreter(job.world; method_table_view=maybe_cached(method_table_view(job)),
                   code_cache=ci_cache(job), inf_params=inference_params(job),
                   opt_params=optimization_params(job))
end

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(@nospecialize(job::CompilerJob)) = uses_julia_runtime(job)

# does this target support loading from Julia safepoints?
# if not, safepoints at function entry will not be emitted
can_safepoint(@nospecialize(job::CompilerJob)) = uses_julia_runtime(job)

# generate a string that represents the type of compilation, for selecting a compiled
# instance of the runtime library. this slug should encode everything that affects
# the generated code of this compiler job (with exception of the function source)
runtime_slug(@nospecialize(job::CompilerJob)) = error("Not implemented")

# the type of the kernel state object, or Nothing if this back-end doesn't need one.
#
# the generated code will be rewritten to include an object of this type as the first
# argument to each kernel, and pass that object to every function that accesses the kernel
# state (possibly indirectly) via the `kernel_state_pointer` function.
kernel_state_type(@nospecialize(job::CompilerJob)) = Nothing

# Does the target need to pass kernel arguments by value?
pass_by_value(@nospecialize(job::CompilerJob)) = true

# whether pointer is a valid call target
valid_function_pointer(@nospecialize(job::CompilerJob), ptr::Ptr{Cvoid}) = false

# Care is required for anything that impacts:
#   - method_table
#   - inference_params
#   - optimization_params
# By default that is just always_inline
# the cache token is compared with jl_egal
struct GPUCompilerCacheToken
    target_type::Type
    always_inline::Bool
    method_table::Core.MethodTable
end

ci_cache_token(@nospecialize(job::CompilerJob)) =
    GPUCompilerCacheToken(typeof(job.config.target), job.config.always_inline, method_table(job))

# the codeinstance cache to use -- should only be used for the constructor
if VERSION >= v"1.11.0-DEV.1552"
    # Soft deprecated user should use `CC.code_cache(get_interpreter(job))`
    ci_cache(@nospecialize(job::CompilerJob)) = CC.code_cache(get_interpreter(job))
else
function ci_cache(@nospecialize(job::CompilerJob))
    lock(GLOBAL_CI_CACHES_LOCK) do
        cache = get!(GLOBAL_CI_CACHES, job.config) do
            CodeCache()
        end
        return cache
    end
end
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
    if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
end


## extension points at important stages of compilation

# prepare the environment for compilation of a job. this can involve, e.g.,
# priming the cache with entries that cannot be easily inferred.
prepare_job!(@nospecialize(job::CompilerJob)) = return

# early extension point used to link-in external bitcode files.
# this is typically used by downstream packages to link vendor libraries.
link_libraries!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                undefined_fns::Vector{String}) = return

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

# deprecated
struct DeprecationMarker end
process_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = DeprecationMarker()
process_entry!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function) =
    DeprecationMarker()
