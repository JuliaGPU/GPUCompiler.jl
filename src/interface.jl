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

llvm_triple(::AbstractCompilerTarget) = error("Not implemented")

function llvm_machine(target::AbstractCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple)

    tm = TargetMachine(t, triple)
    asm_verbosity!(tm, true)

    return tm
end

llvm_datalayout(target::AbstractCompilerTarget) = DataLayout(llvm_machine(target))


## params

export AbstractCompilerParams

# container for state handled by external users of GPUCompiler.jl

abstract type AbstractCompilerParams end


## function specification

export FunctionSpec

# what we'll be compiling

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

export CompilerJob

# a specific invocation of the compiler, bundling everything needed to generate code

Base.@kwdef struct CompilerJob{T,P}
    target::T
    source::FunctionSpec
    params::P

    CompilerJob(target::AbstractCompilerTarget, source::FunctionSpec, params::AbstractCompilerParams) =
        new{typeof(target), typeof(params)}(target, source, params)
end

Base.similar(job::CompilerJob, source::FunctionSpec) =
    CompilerJob(target=job.target, source=source, params=job.params)

function Base.show(io::IO, job::CompilerJob{T}) where {T}
    print(io, "CompilerJob of ", job.source, " for ", T)
end


## interfaces and fallback definitions

# the Julia module to look up target-specific runtime functions in (this includes both
# target-specific functions from the GPU runtime library, like `malloc`, but also
# replacements functions for operations like `Base.sin`)
runtime_module(::CompilerJob) = error("Not implemented")

# check if a function is an intrinsic that can assumed to be always available
isintrinsic(::CompilerJob, fn::String) = false

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(::CompilerJob) = false

# generate a string that represents the type of compilation, for selecting a compiled
# instance of the runtime library. this slug should encode everything that affects
# the generated code of this compiler job (with exception of the function source)
runtime_slug(::CompilerJob) = error("Not implemented")

# early processing of the newly generated LLVM IR module
process_module!(::CompilerJob, mod::LLVM.Module) = return

# early processing of the newly identified LLVM kernel function
process_kernel!(::CompilerJob, mod::LLVM.Module, kernel::LLVM.Function) = return kernel

# final processing of the IR module, right before validation and machine-code generation
finish_module!(::CompilerJob, mod::LLVM.Module) = return

add_lowering_passes!(::CompilerJob, pm::LLVM.PassManager) = return

add_optimization_passes!(::CompilerJob, pm::LLVM.PassManager) = return

link_libraries!(::CompilerJob, mod::LLVM.Module, undefined_fns::Vector{String}) = return
