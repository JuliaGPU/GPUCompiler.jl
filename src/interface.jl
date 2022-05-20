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


## function specification

export FunctionSpec

# what we'll be compiling

struct FunctionSpec{F,TT}
    f::Type{F}
    tt::Type{TT}
    kernel::Bool
    name::Union{Nothing,String}
    world_age::UInt
end


function Base.hash(spec::FunctionSpec, h::UInt)
    h = hash(spec.f, h)
    h = hash(spec.tt, h)
    h = hash(spec.kernel, h)
    h = hash(spec.name, h)
    h = hash(spec.world_age, h)
    h
end

# put the function and argument types in typevars
# so that we can access it from generated functions
# XXX: the default value of 0xffffffffffffffff is a hack, because we don't properly perform
#      world age intersection when querying the compilation cache. once we do, callers
#      should probably provide the world age of the calling code (!= the current world age)
#      so that querying the cache from, e.g. `cufuncton` is a fully static operation.
FunctionSpec(f::Type, tt=Tuple{}, kernel=true, name=nothing, world_age=-1%UInt) =
    FunctionSpec{f,tt}(f, tt, kernel, name, world_age)

FunctionSpec(f, tt=Tuple{}, kernel=true, name=nothing, world_age=-1%UInt) =
    FunctionSpec(Core.Typeof(f), tt, kernel, name, world_age)

function Base.getproperty(@nospecialize(spec::FunctionSpec), sym::Symbol)
    if sym == :world
        # NOTE: this isn't used by the call to `hash` in `check_cache`,
        #       so we still use the raw world age there.
        age = spec.world_age
        return age == -1%UInt ? Base.get_world_counter() : age
    else
        return getfield(spec, sym)
    end
end

function signature(@nospecialize(spec::FunctionSpec))
    fn = something(spec.name, nameof(spec.f))
    args = join(spec.tt.parameters, ", ")
    return "$fn($(join(spec.tt.parameters, ", ")))"
end

function Base.show(io::IO, @nospecialize(spec::FunctionSpec))
    spec.kernel ? print(io, "kernel ") : print(io, "function ")
    print(io, signature(spec))
end


## job

export CompilerJob

# a specific invocation of the compiler, bundling everything needed to generate code

"""
    CompilerJob(target, source, params, entry_abi)

Construct a `CompilerJob` for `source` that will be used to drive compilation for
the given `target` and `params`. The `entry_abi` can be either `:specfunc` the default,
or `:func`. `:specfunc` expects the arguments to be passed in registers, simple
return values are returned in registers as well, and complex return values are returned
on the stack using `sret`, the calling convention is `fastcc`. The `:func` abi is simpler
with a calling convention of the first argument being the function itself (to support closures),
the second argument being a pointer to a vector of boxed Julia values and the third argument
being the number of values, the return value will also be boxed. The `:func` abi
will internally call the `:specfunc` abi, but is generally easier to invoke directly.
"""
struct CompilerJob{T,P,F}
    target::T
    source::F
    params::P
    entry_abi::Symbol

    function CompilerJob(target::AbstractCompilerTarget, source::FunctionSpec, params::AbstractCompilerParams, entry_abi::Symbol)
        if entry_abi ∉ (:specfunc, :func)
            error("Unknown entry_abi=$entry_abi")
        end
        new{typeof(target), typeof(params), typeof(source)}(target, source, params, entry_abi)
    end
end
CompilerJob(target::AbstractCompilerTarget, source::FunctionSpec, params::AbstractCompilerParams; entry_abi=:specfunc) =
    CompilerJob(target, source, params, entry_abi)

Base.similar(@nospecialize(job::CompilerJob), @nospecialize(source::FunctionSpec)) =
    CompilerJob(job.target, source, job.params, job.entry_abi)

function Base.show(io::IO, @nospecialize(job::CompilerJob{T})) where {T}
    print(io, "CompilerJob of ", job.source, " for ", T)
end

function Base.hash(job::CompilerJob, h::UInt)
    h = hash(job.target, h)
    h = hash(job.source, h)
    h = hash(job.params, h)
    h = hash(job.entry_abi, h)
    h
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
    GPUInterpreter(ci_cache(job), method_table(job), job.source.world)

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(@nospecialize(job::CompilerJob)) = uses_julia_runtime(job)

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

    if job.source.kernel && needs_byval(job)
        # pass all bitstypes by value; by default Julia passes aggregates by reference
        # (this improves performance, and is mandated by certain back-ends like SPIR-V).
        args = classify_arguments(job, eltype(llvmtype(entry)))
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
ci_cache(@nospecialize(job::CompilerJob)) = GLOBAL_CI_CACHE

# the method table to use
method_table(@nospecialize(job::CompilerJob)) = GLOBAL_METHOD_TABLE

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
