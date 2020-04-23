# target-specific interface for packages to implement

# you should create concrete structures that subtype the abstract types in this file,
# and implement all accessor methods to return fields from your structures.
#
# you can add additional fields to these structures for use in target-specific sections
# of the compiler, or add hooks for more specific uses (or to avoid package dependencies).


## target

# a target represents a logical (sub)device to generate code for, and contains state and
# data that is unlikely to change across compiler invocations. it is expected you only
# need to instantiate a handful of these in your package.

export AbstractCompilerTarget

abstract type AbstractCompilerTarget end

llvm_triple(::AbstractCompilerTarget) = error("Not implemented")

function llvm_machine(target::AbstractCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple)

    tm = TargetMachine(t, triple)
    asm_verbosity!(tm, true)

    return tm
end

llvm_datalayout(target::AbstractCompilerTarget) = DataLayout(llvm_machine(target))

# the Julia module to look up target-specific runtime functions in (this includes both
# target-specific functions from the GPU runtime library, like `malloc`, but also
# replacements functions for operations like `Base.sin`)
runtime_module(::AbstractCompilerTarget) = error("Not implemented")

# check if a function is an intrinsic that can assumed to be always available
isintrinsic(::AbstractCompilerTarget, fn::String) = false

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(::AbstractCompilerTarget) = false


## function specification

# what we'll be compiling

export FunctionSpec

struct FunctionSpec{F,TT}
    f::Base.Callable
    tt::DataType
    kernel::Bool
    name::Union{Nothing,String}
end

# put the function and argument types in typevars
# so that we can access it from generated functions
FunctionSpec(f, tt=Tuple{}, kernel=true, name=nothing) =
    FunctionSpec{typeof(f),tt}(f, tt, kernel, name)

function signature(spec::FunctionSpec)
    fn = something(spec.name, nameof(spec.f))
    args = join(spec.tt.parameters, ", ")
    return "$fn($(join(spec.tt.parameters, ", ")))"
end

function Base.show(io::IO, spec::FunctionSpec)
    spec.kernel ? print(io, "kernel ") : print(io, "function ")
    print(io, signature(spec))
end


## job

# a compiler job encodes a specific invocation of the compiler, and together with the
# compiler target contains all necessary information to generate code.

export AbstractCompilerJob

abstract type AbstractCompilerJob end

# link to the AbstractCompilerTarget
target(::AbstractCompilerJob) = error("Not implemented")

# link to the FunctionSpec
source(::AbstractCompilerJob) = error("Not implemented")

# generate a string that represents the type of compilation, for selecting a compiled
# instance of the runtime library. this slug should encode everything that affects
# the generated code of this compiler job (with exception of the function source)
runtime_slug(::AbstractCompilerJob) = error("Not implemented")

# early processing of the newly generated LLVM IR module
process_module!(::AbstractCompilerJob, mod::LLVM.Module) = return

# early processing of the newly identified LLVM kernel function
process_kernel!(::AbstractCompilerJob, mod::LLVM.Module, kernel::LLVM.Function) = return

add_lowering_passes!(::AbstractCompilerJob, pm::LLVM.PassManager) = return

add_optimization_passes!(::AbstractCompilerJob, pm::LLVM.PassManager) = return

link_libraries!(::AbstractCompilerJob, mod::LLVM.Module, undefined_fns::Vector{String}) = return


## inheritance through composition

# downstream packages likely need to extend the above compiler functionality.
# to facilitate that, the abstract Composite... types below can be used,
# where downstream functionality only needs to implement `Base.parent`
# to make sure non-overloaded functionality is dispatched to the parent.

export CompositeCompilerTarget, CompositeCompilerJob

Base.parent(target::AbstractCompilerTarget) = target
Base.parent(job::AbstractCompilerJob) = job

abstract type CompositeCompilerTarget <: AbstractCompilerTarget end
llvm_triple(target::CompositeCompilerTarget) = llvm_triple(Base.parent(target))
llvm_machine(target::CompositeCompilerTarget) = llvm_machine(Base.parent(target))
llvm_datalayout(target::CompositeCompilerTarget) = llvm_datalayout(Base.parent(target))
runtime_module(target::CompositeCompilerTarget) = runtime_module(Base.parent(target))
isintrinsic(target::CompositeCompilerTarget, fn::String) = isintrinsic(Base.parent(target), fn)
can_throw(target::CompositeCompilerTarget) = can_throw(Base.parent(target))

abstract type CompositeCompilerJob <: AbstractCompilerJob end
target(job::CompositeCompilerJob) = target(Base.parent(job))
source(job::CompositeCompilerJob) = source(Base.parent(job))
runtime_slug(job::CompositeCompilerJob) = runtime_slug(Base.parent(job))
process_module!(job::CompositeCompilerJob, mod::LLVM.Module) =
    process_module!(Base.parent(job), mod)
process_kernel!(job::CompositeCompilerJob, mod::LLVM.Module, kernel::LLVM.Function) =
    process_kernel!(Base.parent(job), mod, kernel)
add_lowering_passes!(job::CompositeCompilerJob, pm::LLVM.PassManager) =
    add_lowering_passes!(Base.parent(job), pm)
add_optimization_passes!(job::CompositeCompilerJob, pm::LLVM.PassManager) =
    add_optimization_passes!(Base.parent(job), pm)
link_libraries!(job::CompositeCompilerJob, mod::LLVM.Module, undefined_fns::Vector{String}) =
    link_libraries!(Base.parent(job), mod, undefined_fns)
