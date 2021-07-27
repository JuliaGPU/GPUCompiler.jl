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

source_code(target::AbstractCompilerTarget) = "text"

llvm_triple(target::AbstractCompilerTarget) = error("Not implemented")

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


## compiler

export Compiler

# this definition is parametric so that we can use it for overriding parts of the compiler

struct Compiler{T,P}
    target::T
    params::P
end

Compiler(target::T, params::P) where {T<:AbstractCompilerTarget, P<:AbstractCompilerParams} =
    Compiler{T,P}(target, params)

function Base.hash(compiler::Compiler, h::UInt)
    h = hash(compiler.target, h)
    h = hash(compiler.params, h)
    h
end


## function specification

export FunctionSpec

# this definition isn't parametric to avoid specializing the compiler on what we compile

struct FunctionSpec
    f::Any
    tt::Type
    kernel::Bool
    name::Union{Nothing,String}
    world_age::UInt

    # XXX: the default value of 0xffffffffffffffff is a hack, because we don't properly perform
    #      world age intersection when querying the compilation cache. once we do, callers
    #      should probably provide the world age of the calling code (!= the current world age)
    #      so that querying the cache from, e.g. `cufuncton` is a fully static operation.
    FunctionSpec(f, tt=Tuple{}, kernel=true, name=nothing, world_age=-1%UInt) =
        new(f, tt, kernel, name, world_age)
end

function Base.hash(spec::FunctionSpec, h::UInt)
    h = hash(spec.f, h)
    h = hash(spec.tt, h)
    h = hash(spec.kernel, h)
    h = hash(spec.name, h)
    h = hash(spec.world_age, h)
    h
end

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

# because we lack a `@noinfer`, we can't just pass the `Compiler` and a `FunctionSpec` to
# the GPUCompiler entry-points, or we'd re-infer everything based on the concrete compiler.
#
# instead, for APIs that shouldn't re-specialize, we bundle those objects in a `CompilerJob`
# and pass that instead. when calling code that can be overridden, like most of the
# interfaces here, we pass the compiler and its source separately (but do so using
# `@invokelatest` to avoid invalidating the parent).

struct CompilerJob
    compiler::Compiler
    source::FunctionSpec
end

Base.similar(job::CompilerJob, source::FunctionSpec) = CompilerJob(job.compiler, source)

function Base.show(io::IO, job::CompilerJob)
    print(io, "CompilerJob of ", job.source, " using ", job.compiler)
end

function Base.hash(job::CompilerJob, h::UInt)
    h = hash(job.compiler, h)
    h = hash(job.source, h)
    h
end


## interfaces and fallback definitions

# the Julia module to look up target-specific runtime functions in (this includes both
# target-specific functions from the GPU runtime library, like `malloc`, but also
# replacements functions for operations like `Base.sin`)
runtime_module(compiler::Compiler) = error("Not implemented")

# check if a function is an intrinsic that can assumed to be always available
isintrinsic(compiler::Compiler, fn::String) = false

# provide a specific interpreter to use.
get_interpreter(compiler::Compiler, source::FunctionSpec) =
    GPUInterpreter(@invokelatest(ci_cache(compiler)),
                   @invokelatest(method_table(compiler)),
                   source.world)

# does this target support throwing Julia exceptions with jl_throw?
# if not, calls to throw will be replaced with calls to the GPU runtime
can_throw(compiler::Compiler) = false

# generate a string that represents the type of compilation, for selecting a compiled
# instance of the runtime library. this slug should encode everything that affects
# the generated code of this compiler job (with exception of the function source)
runtime_slug(compiler::Compiler) = error("Not implemented")

# early processing of the newly generated LLVM IR module
process_module!(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module) = return

# early processing of the newly identified LLVM kernel function
function process_entry!(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module,
                        entry::LLVM.Function)
    ctx = context(mod)

    if source.kernel
        # pass all bitstypes by value; by default Julia passes aggregates by reference
        # (this improves performance, and is mandated by certain back-ends like SPIR-V).
        args = classify_arguments(CompilerJob(compiler, source), entry)
        for arg in args
            if arg.cc == BITS_REF
                push!(parameter_attributes(entry, arg.codegen.i), EnumAttribute("byval", 0; ctx))
            end
        end
    end

    return entry
end

# post-Julia optimization processing of the module
optimize_module!(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module) = return

# final processing of the IR module, right before validation and machine-code generation
finish_module!(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module) = return

add_lowering_passes!(compiler::Compiler, source::FunctionSpec, pm::LLVM.PassManager) = return

link_libraries!(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module,
                undefined_fns::Vector{String}) = return

# whether pointer is a valid call target
valid_function_pointer(compiler::Compiler, ptr::Ptr{Cvoid}) = false

# the codeinfo cache to use
ci_cache(compiler::Compiler) = GLOBAL_CI_CACHE

# the method table to use
method_table(compiler::Compiler) = GLOBAL_METHOD_TABLE

# how much debuginfo to emit
function llvm_debug_info(compiler::Compiler)
    if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
end

function emit_assembly(compiler::Compiler, source::FunctionSpec, mod::LLVM.Module, format)
    tm = @invokelatest llvm_machine(compiler.target)
    return String(emit(tm, mod, format))
end

# sometimes there exist tools for better displaying native code, like `spirv-dis`.
function show_native_code(compiler::Compiler, source::FunctionSpec, io::IO;
                          raw::Bool, dump_module::Bool)
    asm, meta = codegen(:asm, CompilerJob(compiler, source);
                        strip=!raw, only_entry=!dump_module, validate=false)
    highlight(io, asm, @invokelatest(source_code(compiler.target)))
end
