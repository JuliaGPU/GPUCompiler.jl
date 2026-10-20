# device libraries: definitions linked into kernels on demand

"""
    DeviceLibraryMethod(f, return_type, argument_types; llvm_name,
                        llvm_return_type=nothing, llvm_types=nothing)

Describe an exported Julia specialization and its explicit LLVM ABI. Type generators
are evaluated in the compilation's LLVM context.
"""
function DeviceLibraryMethod(def::Function, return_type::Type, types::Tuple;
                             name=nameof(def), llvm_name="gpu_$name",
                             llvm_return_type=nothing, llvm_types=nothing)
    Runtime.RuntimeMethodInstance(def, return_type, types, name,
                                  llvm_return_type, llvm_types, llvm_name)
end

"""
    DeviceLibrary(name; methods, optimize=false, always_inline=optimize)
    DeviceLibrary(name; bitcode, always_inline=false)

Describe a library of device functions: Julia specializations (a tuple of
[`DeviceLibraryMethod`](@ref)s), or LLVM bitcode (a path or bytes). Bitcode is read once
per session, so files must not change while Julia is running.

Libraries are linked on demand: only definitions the kernel references, and their
dependencies, are imported. Julia libraries are only compiled when used.

`optimize=true` optimizes a Julia library on its own, and requires it to be standalone
device code without Julia runtime calls, address spaces, relocations, mutable globals or
indirect calls. `always_inline=true` forces imported definitions to be inlined;
otherwise the inliner decides.
"""
struct DeviceLibrary
    name::Symbol
    methods::Tuple
    bitcode::Union{Nothing,String,Vector{UInt8}}
    optimize::Bool
    always_inline::Bool
    # export manifest, computed on first use
    exports::Base.RefValue{Union{Nothing,Set{String}}}

    function DeviceLibrary(name::Symbol; methods=nothing, bitcode=nothing,
                           optimize=false, always_inline=optimize)
        (methods === nothing) != (bitcode === nothing) ||
            throw(ArgumentError("Specify exactly one of methods and bitcode"))
        if methods !== nothing
            methods isa Tuple || throw(ArgumentError("methods must be a tuple"))
            all(method -> method isa Runtime.RuntimeMethodInstance, methods) ||
                throw(ArgumentError("Expected DeviceLibraryMethod descriptors"))
            allunique(method.llvm_name for method in methods) ||
                throw(ArgumentError("Duplicate exports in $name"))
            exports = Set(method.llvm_name for method in methods)
        else
            optimize && throw(ArgumentError("Only Julia libraries can be optimized"))
            bitcode isa Union{String,Vector{UInt8}} ||
                throw(ArgumentError("bitcode must be a path or Vector{UInt8}"))
            exports = nothing
        end
        new(name, something(methods, ()), bitcode, optimize, always_inline,
            Ref{Union{Nothing,Set{String}}}(exports))
    end
end

"""
    runtime_library(mod::Module)

Describe the target's implementations of GPUCompiler's runtime functions (e.g.
`signal_exception` or `malloc`) defined in `mod`, using GPUCompiler's signatures.
"""
function runtime_library(mod::Module)
    methods = Tuple(DeviceLibraryMethod(getfield(mod, method.def),
                                        method.return_type, method.types;
                                        method.name, method.llvm_name,
                                        method.llvm_return_type, method.llvm_types)
                    for method in runtime_methods()
                    if method.def isa Symbol && isdefined(mod, method.def))
    DeviceLibrary(nameof(mod); methods)
end

"""
    device_libraries(job) -> Tuple

Libraries to resolve the job's external references with, in search order. The default
uses `runtime_module(job)`; back-ends overriding this hook should include
`runtime_library(MyRuntime)` themselves. GPUCompiler's own runtime is added for targets
that do not use Julia's runtime.

The selection must only depend on the job's configuration, as it is not part of any cache
key, and Julia library methods must be resolvable through the job's method table.
"""
device_libraries(@nospecialize(job::CompilerJob)) =
    uses_julia_runtime(job) ? () : (runtime_library(runtime_module(job)),)

function selected_device_libraries(@nospecialize(job::CompilerJob))
    libraries = device_libraries(job)
    libraries isa Tuple || error("device_libraries must return a tuple")
    uses_julia_runtime(job) && return libraries
    core = DeviceLibrary(:gpu_runtime;
                         methods=Tuple(method for method in runtime_methods()
                                       if !(method.def isa Symbol)))
    return (core, libraries...)
end


## bitcode libraries

const bitcode_files = Dict{String,Vector{UInt8}}()
const bitcode_files_lock = ReentrantLock()

function library_bytes(lib::DeviceLibrary)
    lib.bitcode isa Vector{UInt8} && return lib.bitcode
    path = lib.bitcode::String
    Base.@lock bitcode_files_lock get!(() -> read(path), bitcode_files, path)
end

# the names a library can define
function library_exports(lib::DeviceLibrary)
    exports = lib.exports[]
    exports === nothing || return exports
    @dispose ctx=Context() mod=parse(LLVM.Module, MemoryBuffer(library_bytes(lib)); lazy=true) begin
        exports = Set{String}()
        for value in Iterators.flatten((mod.functions, mod.globals, mod.aliases))
            value isa LLVM.GlobalAlias || !isdeclaration(value) || continue
            value.linkage in (LLVM.Linkage.Private, LLVM.Linkage.Internal,
                              LLVM.Linkage.AvailableExternally) && continue
            push!(exports, value.name)
        end
    end
    lib.exports[] = exports
end


## Julia libraries

# Julia library artifacts are cached per session. Artifacts linked before optimization
# keep the original ABI; those linked afterwards have been lowered by `lower_intrinsics!`.
const device_libs = Dict{Any,RuntimeLibrary}()
const device_libs_lock = ReentrantLock()

@locked function load_julia_library(lib::DeviceLibrary, @nospecialize(job::CompilerJob);
                                    lowered::Bool)
    config = runtime_config(job)
    # a `Vector{Any}` key avoids compiling `Dict` methods for every library's tuple type
    key = Any[lib.name, lib.optimize, lib.always_inline, config,
              !supports_typed_pointers(context()), lowered,
              lib.optimize ? job.config.opt_level : 0]
    for method in lib.methods
        push!(key, method.def, method.return_type, method.types, method.name,
              method.llvm_return_type, method.llvm_types, method.llvm_name)
    end
    cached = Base.@lock device_libs_lock begin
        cached = get(device_libs, key, nothing)
        if cached === nothing || !device_library_valid(cached, job, lib.methods)
            cached = build_device_library(job, config, lib; lowered)
            device_libs[key] = cached
        end
        cached
    end
    parse(LLVM.Module, MemoryBuffer(cached.bytes)), cached.relocations
end

function build_device_library(@nospecialize(job::CompilerJob), config::CompilerConfig,
                              lib::DeviceLibrary; lowered::Bool=false)
    @dispose mod=LLVM.Module(string(lib.name)) begin
        sources = MethodInstance[]
        instances = CodeInstance[]
        relocs = Relocations()
        for method in lib.methods
            source = runtime_method_instance(job, method)
            source === nothing && error("No method for $(method.name) in library $(lib.name)")
            push!(sources, source)
            push!(instances, emit_function!(mod, relocs, config, source, method, job.world))
        end
        if lib.optimize
            standalone = CompilerJob(first(sources),
                CompilerConfig(config; libraries=false, opt_level=job.config.opt_level),
                job.world)
            optimize!(standalone, mod, relocs; opt_level=job.config.opt_level)
            prune_dead_relocations!(mod, relocs)
            validate_device_library(mod, relocs, lib)
        end
        if lowered
            lower_intrinsics!(CompilerJob(first(sources), config, job.world), mod)
            for method in lib.methods
                haskey(mod.functions, method.llvm_name) ||
                    error("Lowering library $(lib.name) removed export $(method.llvm_name)")
            end
        end
        if lib.always_inline
            for method in lib.methods
                attrs = mod.functions[method.llvm_name].function_attributes
                delete!(attrs, EnumAttribute("noinline"))
                push!(attrs, EnumAttribute("alwaysinline"))
            end
        end
        io = IOBuffer()
        write(io, mod)
        return RuntimeLibrary(take!(io), sources, instances, job.world, relocs)
    end
end

function validate_device_library(mod, relocs, lib)
    verify(mod)
    isempty(relocs) || error("Optimized library $(lib.name) contains relocations")
    runtime_names = Set(method.llvm_name for method in runtime_methods())
    function check_type(typ, seen=Set{LLVMType}())
        typ in seen && return
        push!(seen, typ)
        if typ isa LLVM.PointerType
            typ.addrspace in 10:13 &&
                error("Optimized library $(lib.name) contains Julia address spaces")
            supports_typed_pointers(context()) && check_type(typ.element_type, seen)
        elseif typ isa LLVM.FunctionType
            check_type(typ.return_type, seen)
            foreach(typ -> check_type(typ, seen), typ.parameters)
        elseif typ isa Union{LLVM.ArrayType,LLVM.VectorType}
            check_type(typ.element_type, seen)
        elseif typ isa LLVM.StructType && !isopaque(typ)
            foreach(typ -> check_type(typ, seen), typ.elements)
        end
    end
    for global_var in mod.globals
        (global_var.constant && global_var.linkage == LLVM.Linkage.Private) ||
            error("Optimized library $(lib.name) contains globals that are not private constants")
        check_type(global_var.global_value_type)
    end
    for fun in mod.functions
        check_type(fun.function_type)
        if isdeclaration(fun) && !isempty(fun.uses) &&
           (startswith(fun.name, "julia.") || startswith(fun.name, "jl_") ||
            startswith(fun.name, "ijl_") || fun.name in runtime_names)
            error("Optimized library $(lib.name) contains runtime operation $(fun.name)")
        end
        for block in fun.blocks, inst in block.instructions
            check_type(inst.value_type)
            foreach(value -> check_type(value.value_type), inst.operands)
            inst isa LLVM.CallBase && inst.called_function === nothing &&
                !(inst.called_operand isa LLVM.InlineAsm) &&
                error("Optimized library $(lib.name) contains an indirect call")
        end
    end
    return
end


## resolution

# Whether a definition of type `actual` can be called through a declaration of type
# `expected`. Typed-pointer element types and struct names are not part of the ABI.
abi_compatible(expected::LLVMType, actual::LLVMType) = expected == actual
abi_compatible(expected::LLVM.PointerType, actual::LLVM.PointerType) =
    expected.addrspace == actual.addrspace
function abi_compatible(expected::LLVM.FunctionType, actual::LLVM.FunctionType)
    LLVM.isvararg(expected) == LLVM.isvararg(actual) &&
        abi_compatible(expected.return_type, actual.return_type) &&
        length(expected.parameters) == length(actual.parameters) &&
        all(abi_compatible(e, a) for (e, a) in zip(expected.parameters, actual.parameters))
end
function abi_compatible(expected::LLVM.StructType, actual::LLVM.StructType)
    expected == actual && return true
    (isopaque(expected) || isopaque(actual)) && return false
    ispacked(expected) == ispacked(actual) &&
        length(expected.elements) == length(actual.elements) &&
        all(abi_compatible(e, a) for (e, a) in zip(expected.elements, actual.elements))
end
function abi_compatible(expected::T, actual::T) where {T<:Union{LLVM.ArrayType,LLVM.VectorType}}
    length(expected) == length(actual) &&
        abi_compatible(expected.element_type, actual.element_type)
end

needed_symbols(mod::LLVM.Module) =
    Set(value.name for value in Iterators.flatten((mod.functions, mod.globals))
        if isdeclaration(value) && !isempty(value.uses))

# Import the definitions `mod` references from the job's device libraries, until no
# library provides any remaining reference. Libraries are searched in order. Returns the
# names of imported functions and aliases.
function link_device_libraries!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                relocs::Relocations; lowered::Bool)
    libraries = selected_device_libraries(job)
    definitions() = Iterators.flatten((Iterators.filter(!isdeclaration, mod.functions),
                                       mod.aliases))
    defined = Set(value.name for value in definitions())
    needed = needed_symbols(mod)
    progress = true
    while progress && !isempty(needed)
        progress = false
        for lib in libraries
            names = intersect(needed, library_exports(lib))
            isempty(names) && continue
            library, library_relocs = if lib.bitcode === nothing
                load_julia_library(lib, job; lowered)
            else
                parse(LLVM.Module, MemoryBuffer(library_bytes(lib)); lazy=true), Relocations()
            end
            try
                for name in names
                    expected = get(mod.functions, name, nothing)
                    actual = get(library.functions, name, nothing)
                    actual = something(actual, get(library.aliases, name, nothing), Some(nothing))
                    (expected === nothing || actual === nothing) && continue
                    actual_type = LLVM.global_value_type(actual)
                    abi_compatible(expected.function_type, actual_type) ||
                        error("Device library $(lib.name) defines $name as $actual_type, incompatible with its use as $(expected.function_type)")
                end
                if lib.always_inline && lib.bitcode !== nothing
                    for fun in library.functions
                        isdeclaration(fun) && continue
                        attrs = fun.function_attributes
                        haskey(attrs, :noinline) || push!(attrs, EnumAttribute("alwaysinline"))
                    end
                end
            catch
                dispose(library)
                rethrow()
            end
            # avoid warnings about mismatching triples and data layouts
            library.triple = mod.triple
            library.datalayout = mod.datalayout
            link_relocatable!(mod, relocs, library, library_relocs; only_needed=true)
            needed = needed_symbols(mod)
            progress = true
        end
    end
    return [value.name for value in definitions() if !(value.name in defined)]
end

# Before optimization: import every referenced definition, so that it gets optimized
# along with the kernel.
function link_device_libraries_early!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                      relocs::Relocations)
    @tracepoint "device libraries" link_device_libraries!(job, mod, relocs; lowered=false)
    @tracepoint "target libraries" link_libraries!(job, mod)
    return
end

# After optimization: lower what the target cannot select, and import definitions for the
# references this introduces. Definitions imported before optimization that are still
# around are reused: the optimizer does not change their signatures, only their calling
# convention, which `call_declared!` follows.
function lower_and_link_device_libraries!(@nospecialize(job::CompilerJob),
                                          mod::LLVM.Module, relocs::Relocations)
    lower_intrinsics!(job, mod)
    return link_device_libraries_late!(job, mod, relocs)
end

# Import definitions for references introduced after optimization. Returns whether
# anything was imported.
function link_device_libraries_late!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                     relocs::Relocations)
    imported = link_device_libraries!(job, mod, relocs; lowered=true)
    link_libraries!(job, mod)
    for name in imported
        value = something(get(mod.functions, name, nothing), get(mod.aliases, name, nothing))
        value.linkage = LLVM.Linkage.Internal
    end
    # repeat what imports before optimization went through
    CPUFeatures(job)(mod)
    lower_syncscopes!(job, mod)
    isempty(imported) || finish_imports!(job, mod, imported)
    return !isempty(imported)
end

@public DeviceLibrary, DeviceLibraryMethod, device_libraries, runtime_library
