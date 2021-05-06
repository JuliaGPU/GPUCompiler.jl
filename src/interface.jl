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

source_code(::AbstractCompilerTarget) = "text"

llvm_triple(::AbstractCompilerTarget) = error("Not implemented")

function llvm_machine(target::AbstractCompilerTarget)
    triple = llvm_triple(target)

    t = Target(triple=triple)

    tm = TargetMachine(t, triple)
    asm_verbosity!(tm, true)

    return tm
end

llvm_datalayout(target::AbstractCompilerTarget) = DataLayout(llvm_machine(target))

# the target's datalayout, with Julia's non-integral address spaces added to it
function julia_datalayout(target::AbstractCompilerTarget)
    dl = llvm_datalayout(target)
    dl === nothing && return nothing
    DataLayout(string(dl) * "-ni:10:11:12:13")
end


## params

export AbstractCompilerParams

# container for state handled by external users of GPUCompiler.jl

abstract type AbstractCompilerParams end


## function specification

export FunctionSpec

# what we'll be compiling

struct FunctionSpec{F,TT}
    f::F
    tt::DataType
    kernel::Bool
    name::Union{Nothing,String}
    world_age::UInt
end

# put the function and argument types in typevars
# so that we can access it from generated functions
# XXX: the default value of 0xffffffffffffffff is a hack, because we don't properly perform
#      world age intersection when querying the compilation cache. once we do, callers
#      should probably provide the world age of the calling code (!= the current world age)
#      so that querying the cache from, e.g. `cufuncton` is a fully static operation.
FunctionSpec(f, tt=Tuple{}, kernel=true, name=nothing, world_age=-1%UInt) =
    FunctionSpec{typeof(f),tt}(f, tt, kernel, name, world_age)

function Base.getproperty(spec::FunctionSpec, sym::Symbol)
    if sym == :world
        # NOTE: this isn't used by the call to `hash` in `check_cache`,
        #       so we still use the raw world age there.
        age = spec.world_age
        return age == -1%UInt ? Base.get_world_counter() : age
    else
        return getfield(spec, sym)
    end
end

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

Base.@kwdef struct CompilerJob{T,P,F}
    target::T
    source::F
    params::P

    CompilerJob(target::AbstractCompilerTarget, source::FunctionSpec, params::AbstractCompilerParams) =
        new{typeof(target), typeof(params), typeof(source)}(target, source, params)
end

Base.similar(@nospecialize(job::CompilerJob), source::FunctionSpec) =
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

# provide a specific interpreter to use.
get_interpreter(job::CompilerJob) = GPUInterpreter(ci_cache(job), method_table(job),
                                                   job.source.world)

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
function process_entry!(job::CompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)

    if job.source.kernel
        # pass all bitstypes by value; by default Julia passes aggregates by reference
        # (this improves performance, and is mandated by certain back-ends like SPIR-V).
        args = classify_arguments(job, entry)
        for arg in args
            if arg.cc == BITS_REF
                push!(parameter_attributes(entry, arg.codegen.i), EnumAttribute("byval", 0, ctx))
            end
        end
    end

    return entry
end

# post-Julia optimization processing of the module
optimize_module!(::CompilerJob, mod::LLVM.Module) = return

# final processing of the IR module, right before validation and machine-code generation
finish_module!(::CompilerJob, mod::LLVM.Module) = return

add_lowering_passes!(::CompilerJob, pm::LLVM.PassManager) = return

link_libraries!(::CompilerJob, mod::LLVM.Module, undefined_fns::Vector{String}) = return

# whether pointer is a valid call target
valid_function_pointer(::CompilerJob, ptr::Ptr{Cvoid}) = false

# the codeinfo cache to use
ci_cache(::CompilerJob) = GLOBAL_CI_CACHE

# the method table to use
method_table(::CompilerJob) = GLOBAL_METHOD_TABLE

# how much debuginfo to emit
function llvm_debug_info(::CompilerJob)
    if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
end
