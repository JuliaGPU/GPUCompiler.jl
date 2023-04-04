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


## params

export AbstractCompilerParams

# container for state handled by external users of GPUCompiler.jl

abstract type AbstractCompilerParams end


## config

export CompilerConfig

# the configuration of the compiler

"""
    CompilerConfig(target, params; kernel=true, entry_abi=:specfunc, name=nothing,
                                   always_inline=false)

Construct a `CompilerConfig` that will be used to drive compilation for the given `target`
and `params`.

Several keyword arguments can be used to customize the compilation process:

- `kernel`: specifies if the function should be compiled as a kernel, or as a regular
   function. This is used to determine the calling convention and for validation purposes.
- `entry_abi`: can be either `:specfunc` the default, or `:func`. `:specfunc` expects the
  arguments to be passed in registers, simple return values are returned in registers as
   well, and complex return values are returned on the stack using `sret`, the calling
   convention is `fastcc`. The `:func` abi is simpler with a calling convention of the first
   argument being the function itself (to support closures), the second argument being a
   pointer to a vector of boxed Julia values and the third argument being the number of
   values, the return value will also be boxed. The `:func` abi will internally call the
   `:specfunc` abi, but is generally easier to invoke directly.
- `name`: the name that will be used for the entrypoint function. If `nothing` (the
   default), the name will be generated automatically.
- `always_inline` specifies if the Julia front-end should inline all functions into one if
   possible.
"""
struct CompilerConfig{T,P}
    target::T
    params::P

    kernel::Bool
    name::Union{Nothing,String}
    entry_abi::Symbol
    always_inline::Bool

    function CompilerConfig(target::AbstractCompilerTarget,
                            params::AbstractCompilerParams;
                            kernel=true,
                            name=nothing,
                            entry_abi=:specfunc,
                            always_inline=false)
        if entry_abi ∉ (:specfunc, :func)
            error("Unknown entry_abi=$entry_abi")
        end
        new{typeof(target), typeof(params)}(target, params, kernel, name, entry_abi,
                                            always_inline)
    end
end

# copy constructor
CompilerConfig(cfg::CompilerConfig; target=cfg.target, params=cfg.params,
               kernel=cfg.kernel, name=cfg.name, entry_abi=cfg.entry_abi,
               always_inline=cfg.always_inline) =
    CompilerConfig(target, params; kernel, entry_abi, name, always_inline)

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

    return h
end


## job

export CompilerJob

using Core: MethodInstance

# a specific invocation of the compiler, bundling everything needed to generate code

struct CompilerJob{T,P}
    source::MethodInstance
    config::CompilerConfig{T,P}
    world::UInt

    CompilerJob(src::MethodInstance, cfg::CompilerConfig{T,P},
                world=tls_world_age()) where {T,P} =
        new{T,P}(src, cfg, world)
end


## contexts

if VERSION >= v"1.9.0-DEV.516"
    const JuliaContextType = ThreadSafeContext
else
    const JuliaContextType = Context
end


## interfaces and fallback definitions

# Has the runtime available and does not require special handling
uses_julia_runtime(@nospecialize(job::CompilerJob)) = false

# Should emit PTLS lookup that can be relocated
dump_native(@nospecialize(job::CompilerJob)) = false

# the Julia module to look up target-specific runtime functions in (this includes both
# target-specific functions from the GPU runtime library, like `malloc`, but also
# replacements functions for operations like `Base.sin`)
runtime_module(@nospecialize(job::CompilerJob)) = error("Not implemented")

# check if a function is an intrinsic that can assumed to be always available
isintrinsic(@nospecialize(job::CompilerJob), fn::String) = false

# provide a specific interpreter to use.
get_interpreter(@nospecialize(job::CompilerJob)) =
    GPUInterpreter(ci_cache(job), method_table(job), job.world,
                   inference_params(job), optimization_params(job))

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

# early processing of the newly generated LLVM IR module
process_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = return

# the type of the kernel state object, or Nothing if this back-end doesn't need one.
#
# the generated code will be rewritten to include an object of this type as the first
# argument to each kernel, and pass that object to every function that accesses the kernel
# state (possibly indirectly) via the `kernel_state_pointer` function.
kernel_state_type(@nospecialize(job::CompilerJob)) = Nothing

# Does the target need to pass kernel arguments by value?
needs_byval(@nospecialize(job::CompilerJob)) = true

# early processing of the newly identified LLVM kernel function
function process_entry!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                        entry::LLVM.Function)
    ctx = context(mod)

    if job.config.kernel && needs_byval(job)
        # pass all bitstypes by value; by default Julia passes aggregates by reference
        # (this improves performance, and is mandated by certain back-ends like SPIR-V).
        args = classify_arguments(job, function_type(entry))
        for arg in args
            if arg.cc == BITS_REF
                attr = if LLVM.version() >= v"12"
                    TypeAttribute("byval", eltype(arg.codegen.typ); ctx)
                else
                    EnumAttribute("byval", 0; ctx)
                end
                push!(parameter_attributes(entry, arg.codegen.i), attr)
            end
        end
    end

    return entry
end

# post-Julia optimization processing of the module
optimize_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module) = return

# whether an LLVM function is valid for this back-end
validate_module(@nospecialize(job::CompilerJob), mod::LLVM.Module) = IRError[]

# finalization of the module, before deferred codegen and optimization
function finish_module!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function)
    return entry
end

# final processing of the IR, right before validation and machine-code generation
function finish_ir!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function)
    return entry
end

add_lowering_passes!(@nospecialize(job::CompilerJob), pm::LLVM.PassManager) = return

link_libraries!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                undefined_fns::Vector{String}) = return

# whether pointer is a valid call target
valid_function_pointer(@nospecialize(job::CompilerJob), ptr::Ptr{Cvoid}) = false

# the codeinfo cache to use
function ci_cache(@nospecialize(job::CompilerJob))
    lock(GLOBAL_CI_CACHES_LOCK) do
        cache = get!(GLOBAL_CI_CACHES, (typeof(job.config.target), inference_params(job), optimization_params(job))) do
            CodeCache()
        end
        return cache
    end
end

# the method table to use
method_table(@nospecialize(job::CompilerJob)) = GLOBAL_METHOD_TABLE

# the inference parameters to use when constructing the GPUInterpreter
function inference_params(@nospecialize(job::CompilerJob))
    return InferenceParams(;unoptimize_throw_blocks=false)
end

# the optimization parameters to use when constructing the GPUInterpreter
function optimization_params(@nospecialize(job::CompilerJob))
    kwargs = NamedTuple()

    if VERSION < v"1.8.0-DEV.486"
        kwargs = (kwargs..., unoptimize_throw_blocks=false)
    end

    if job.config.always_inline
        kwargs = (kwargs..., inline_cost_threshold=typemax(Int))
    end

    return OptimizationParams(;kwargs...)
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
