"""
    DeviceLibraryMethod(f, return_type, argument_types; llvm_name,
                        llvm_return_type=nothing, llvm_types=nothing)

Describe an exported Julia specialization and its explicit LLVM ABI. Type generators
are evaluated in the compilation's LLVM context. Descriptors do not register methods
globally.
"""
function DeviceLibraryMethod(def::Function, return_type::Type, types::Tuple;
                             name=nameof(def), llvm_name="gpu_$name",
                             llvm_return_type=nothing, llvm_types=nothing)
    Runtime.RuntimeMethodInstance(def, return_type, types, name,
                                  llvm_return_type, llvm_types, llvm_name)
end

"""
    DeviceLibrary(name; methods, optimize=false, always_inline=optimize)
    DeviceLibrary(name; bitcode)

An immutable description of either Julia exports or external bitcode (a path or bytes).
Libraries are resolved once, in selection order. Imports must be supplied by a later
library or recognized by `isintrinsic`. Unused Julia libraries are never compiled.
Optimized Julia libraries must contain standalone device IR: no Julia address spaces,
runtime operations, relocations, indirect calls, or mutable globals.
Optimized exports are inlined after linking by default; `always_inline=false` keeps
larger routines out of line.
"""
struct DeviceLibrary
    name::Symbol
    methods::Tuple
    bitcode::Union{Nothing,String,Vector{UInt8}}
    digest::Union{Nothing,NTuple{32,UInt8}}
    optimize::Bool
    always_inline::Bool
    function DeviceLibrary(name::Symbol; methods=nothing, bitcode=nothing, optimize=false,
                           always_inline=optimize)
        (methods === nothing) != (bitcode === nothing) ||
            throw(ArgumentError("Specify exactly one of methods and bitcode"))
        if methods !== nothing
            methods isa Tuple || throw(ArgumentError("methods must be a tuple"))
            all(method -> method isa Runtime.RuntimeMethodInstance, methods) ||
                throw(ArgumentError("Expected DeviceLibraryMethod descriptors"))
            names = map(method -> method.llvm_name, methods)
            allunique(names) || throw(ArgumentError("Duplicate exports in $name"))
        else
            (optimize || always_inline) &&
                throw(ArgumentError("build policy applies only to Julia libraries"))
            bitcode isa Union{String,Vector{UInt8}} ||
                throw(ArgumentError("bitcode must be a path or Vector{UInt8}"))
        end
        new(name, something(methods, ()),
            bitcode isa Vector{UInt8} ? copy(bitcode) : bitcode,
            bitcode isa Vector{UInt8} ? Tuple(sha256(bitcode)) : nothing,
            optimize, always_inline)
    end
end

"""
    runtime_library(mod::Module)

Describe the target implementations of GPUCompiler's required runtime imports.
Only functions defined in `mod` are exported; signatures come from GPUCompiler.
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

Ordered library selection for a job. The default adapts `runtime_module(job)`;
back-ends overriding this hook should include `runtime_library` explicitly when needed.
GPUCompiler prepends its core runtime for targets without Julia's runtime.
"""
device_libraries(@nospecialize(job::CompilerJob)) =
    uses_julia_runtime(job) ? () : (runtime_library(runtime_module(job)),)

runtime_imports() = Tuple(method for method in runtime_methods() if method.def isa Symbol)

function selected_device_libraries(@nospecialize(job::CompilerJob))
    libraries = device_libraries(job)
    libraries isa Tuple || error("device_libraries must return a tuple")
    uses_julia_runtime(job) && return libraries
    core = DeviceLibrary(:gpu_runtime;
                         methods=Tuple(method for method in runtime_methods()
                                       if !(method.def isa Symbol)))
    return (core, libraries...)
end

mutable struct BitcodeLibrary
    bytes::Vector{UInt8}
    digest::NTuple{32,UInt8}
    exports::Union{Nothing,Set{String}}
end

const bitcode_files = Dict{String,Tuple{Tuple{Float64,Float64,Int64},BitcodeLibrary}}()
const bitcode_buffers = Dict{NTuple{32,UInt8},BitcodeLibrary}()
const bitcode_library_lock = ReentrantLock()

function bitcode_library(lib::DeviceLibrary)
    Base.@lock bitcode_library_lock begin
        if lib.bitcode isa String
            path = abspath(lib.bitcode)
            info = stat(path)
            stamp = (info.mtime, info.ctime, info.size)
            cached = get(bitcode_files, path, nothing)
            if cached === nothing || first(cached) != stamp
                bytes = read(path)
                artifact = BitcodeLibrary(bytes, Tuple(sha256(bytes)), nothing)
                bitcode_files[path] = (stamp, artifact)
                return artifact
            end
            return last(cached)
        end
        bytes = lib.bitcode::Vector{UInt8}
        return get!(bitcode_buffers, lib.digest) do
            BitcodeLibrary(bytes, lib.digest, nothing)
        end
    end
end

library_bytes(lib::DeviceLibrary) = bitcode_library(lib).bytes

library_method_key(method) =
    (method.def, method.return_type, method.types, method.name,
     method.llvm_return_type, method.llvm_types, method.llvm_name)

library_key(lib::DeviceLibrary, snapshot=nothing) =
    (lib.name, map(library_method_key, lib.methods), lib.optimize, lib.always_inline,
     lib.bitcode === nothing ? nothing :
     (snapshot === nothing ? bitcode_library(lib) : snapshot).digest)

const device_libs = Dict{Any,RuntimeLibrary}()
const device_libs_lock = ReentrantLock()

mutable struct DeviceDependentResults{V}
    generation::UInt
    value::V
    DeviceDependentResults{V}() where {V} = new{V}(0, V())
end

mutable struct DeviceLibraryDependencies
    selection::Any
    methods::Vector{Tuple{Runtime.RuntimeMethodInstance,MethodInstance,CodeInstance}}
    generation::UInt
    DeviceLibraryDependencies() = new(nothing, [], 0)
end

mutable struct DeviceLibrarySelection
    world::UInt
    libraries::Tuple
    signature::Tuple
end

mutable struct DeviceLibraryValidation
    world::UInt
    selection::Union{Nothing,DeviceLibrarySelection}
    DeviceLibraryValidation() = new(0, nothing)
end

mutable struct DeviceLibraryLookup{V}
    result::Union{Nothing,DeviceDependentResults{V}}
    dependencies::Union{Nothing,DeviceLibraryDependencies}
    validation::Union{Nothing,DeviceLibraryValidation}
    DeviceLibraryLookup{V}() where {V} = new{V}(nothing, nothing, nothing)
end

const library_dependencies_lock = ReentrantLock()
const library_selections = Dict{CompilerConfig,DeviceLibrarySelection}()
const library_selections_lock = ReentrantLock()

function library_signature(@nospecialize(job::CompilerJob), validation=nothing)
    job.config.libraries && job.config.toplevel || return ()
    Base.@lock library_selections_lock begin
        selection = validation === nothing ? nothing : validation.selection
        if selection === nothing || selection.world != job.world
            selection = get(library_selections, job.config, nothing)
        end
        if selection === nothing || selection.world != job.world
            libraries = selected_device_libraries(job)
            selection = DeviceLibrarySelection(job.world, libraries, map(library_key, libraries))
            library_selections[job.config] = selection
        else
            for (library, key) in zip(selection.libraries, selection.signature)
                library.bitcode === nothing && continue
                if bitcode_library(library).digest != last(key)
                    selection.signature = map(library_key, selection.libraries)
                    break
                end
            end
        end
        validation === nothing || (validation.selection = selection)
        return selection.signature
    end
end

function device_dependent_results(::Type{V}, ci, job; persistent=false) where {V}
    lookup = session_results(DeviceLibraryLookup{V}, ci, job.config)
    dependencies, validation, result = Base.@lock library_dependencies_lock begin
        if lookup.result === nothing
            get_results(typ) = persistent ? persistent_results(typ, ci, job.config) :
                                           session_results(typ, ci, job.config)
            lookup.dependencies = get_results(DeviceLibraryDependencies)
            lookup.validation = session_results(DeviceLibraryValidation, ci, job.config)
            lookup.result = get_results(DeviceDependentResults{V})
        end
        (lookup.dependencies::DeviceLibraryDependencies,
         lookup.validation::DeviceLibraryValidation, lookup.result::DeviceDependentResults{V})
    end
    selection = library_signature(job, validation)
    Base.@lock library_dependencies_lock begin
        valid = isequal(selection, dependencies.selection)
        if valid && validation.world != job.world
            valid = all(dependencies.methods) do (method, source, instance)
                instance.min_world <= job.world <= instance.max_world &&
                    runtime_method_instance(job, method) === source
            end
        end
        if !valid && dependencies.selection !== nothing
            dependencies.generation += 1
            dependencies.selection = nothing
            empty!(dependencies.methods)
        end
        validation.world = job.world
        if result.generation != dependencies.generation
            result.generation = dependencies.generation
            result.value = V()
        end
        return result.value
    end
end

function record_library_dependencies!(job, selection, methods)
    ci = job_code_instance(job)
    ci === nothing && return
    dependencies = can_persist_results(job) ?
        persistent_results(DeviceLibraryDependencies, ci, job.config) :
        session_results(DeviceLibraryDependencies, ci, job.config)
    validation = session_results(DeviceLibraryValidation, ci, job.config)
    Base.@lock library_dependencies_lock begin
        dependencies.selection = selection
        dependencies.methods = methods
        validation.world = job.world
    end
end

device_library_key(lib, job) =
    (library_key(lib), runtime_config(job), !supports_typed_pointers(context()),
     lib.optimize ? job.config.opt_level : 0)

function load_device_library(lib::DeviceLibrary, @nospecialize(job::CompilerJob))
    mod, relocs, _ = load_device_library_record(lib, job)
    return mod, relocs
end

@locked function load_device_library_record(lib::DeviceLibrary, @nospecialize(job::CompilerJob),
                                            snapshot=nothing)
    if lib.bitcode !== nothing
        bytes = snapshot === nothing ? library_bytes(lib) : snapshot.bytes
        return parse(LLVM.Module, MemoryBuffer(bytes)), Relocations(), nothing
    end
    config = runtime_config(job)
    key = device_library_key(lib, job)
    cached = Base.@lock device_libs_lock begin
        cached = get(device_libs, key, nothing)
        if cached === nothing || !device_library_valid(cached, job, lib.methods)
            bytes, sources, instances, relocs = build_device_library(job, config, lib)
            cached = RuntimeLibrary(bytes, sources, instances, job.world, relocs)
            device_libs[key] = cached
        end
        cached
    end
    parse(LLVM.Module, MemoryBuffer(cached.bytes)), cached.relocations, cached
end

function build_device_library(@nospecialize(job::CompilerJob), config, lib::DeviceLibrary)
    @dispose mod=LLVM.Module(string(lib.name)) begin
        sources = MethodInstance[]
        instances = CodeInstance[]
        relocs = Relocations()
        for method in lib.methods
            source = runtime_method_instance(job, method)
            push!(sources, source)
            push!(instances, emit_function!(mod, relocs, config, source, method, job.world))
        end
        if !isempty(lib.methods)
            entry_name = first(lib.methods).llvm_name
            library_job = CompilerJob(first(sources), config, job.world)
            if lib.optimize
                standalone = CompilerJob(first(sources),
                    CompilerConfig(config; libraries=false, opt_level=job.config.opt_level),
                    job.world)
                optimize!(standalone, mod, relocs; opt_level=job.config.opt_level)
            end
            lower_intrinsics!(library_job, mod, mod.functions[entry_name])
            for method in lib.methods
                haskey(mod.functions, method.llvm_name) ||
                    error("Lowering library $(lib.name) removed export $(method.llvm_name)")
            end
            if lib.optimize
                prune_dead_relocations!(mod, relocs)
                validate_device_library(mod, relocs, lib)
            end
            if lib.always_inline
                for method in lib.methods
                    attrs = mod.functions[method.llvm_name].function_attributes
                    delete!(attrs, EnumAttribute("noinline"))
                    push!(attrs, EnumAttribute("alwaysinline"))
                end
            end
        end
        io = IOBuffer()
        write(io, mod)
        return take!(io), sources, instances, relocs
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

function library_exports(lib::DeviceLibrary, snapshot=nothing)
    lib.bitcode === nothing && return Set(method.llvm_name for method in lib.methods)
    artifact = snapshot === nothing ? bitcode_library(lib) : snapshot
    Base.@lock bitcode_library_lock begin
        if artifact.exports === nothing
            @dispose mod=parse(LLVM.Module, MemoryBuffer(artifact.bytes); lazy=true) begin
                artifact.exports = Set(value.name for value in
                    Iterators.flatten((mod.functions, mod.globals))
                    if !isdeclaration(value) &&
                       value.linkage ∉ (LLVM.Linkage.Private, LLVM.Linkage.Internal))
            end
        end
        return artifact.exports
    end
end

function validate_library_imports(job, library, mod, relocs, owners, index, legacy_linker)
    for value in Iterators.flatten((mod.functions, mod.globals))
        isdeclaration(value) && !isempty(value.uses) || continue
        name = value.name
        value isa LLVM.GlobalVariable && find_relocation(relocs, name) !== nothing && continue
        value isa LLVM.Function && value.intrinsic !== nothing && continue
        isintrinsic(job, name) && continue
        if library.bitcode === nothing && !library.optimize &&
           (startswith(name, "jl_") || startswith(name, "ijl_") ||
            startswith(name, "julia."))
            continue
        end
        owner = get(owners, name, nothing)
        owner === nothing && legacy_linker && continue
        owner === nothing && library.name === :gpu_runtime &&
            any(method -> method.llvm_name == name, runtime_imports()) && continue
        owner === nothing &&
            error("Library $(library.name) imports $name, but no later library exports it")
        owner == index &&
            error("Library $(library.name) imports its own export $name")
        owner > index ||
            error("Library $(library.name) imports $name from an earlier library; reorder device_libraries")
    end
end

function resolve_device_libraries!(@nospecialize(job::CompilerJob), mod, relocs;
                                   entry_name=nothing)
    job.config.libraries || return
    entries = [fun.name for fun in mod.functions if !isdeclaration(fun) &&
               fun.linkage == LLVM.Linkage.External]
    if entry_name === nothing
        isempty(entries) && return finish_runtime_intrinsics!(job, mod)
        entry_name = first(entries)
    end
    entry = lower_intrinsics!(job, mod, mod.functions[entry_name])
    entry.name == entry_name || error("lower_intrinsics! must preserve the entry name")
    existing = Set(fun.name for fun in mod.functions if !isdeclaration(fun))
    preserved = [value.name for value in Iterators.flatten((mod.functions, mod.globals))
                 if !isdeclaration(value)]
    libraries = selected_device_libraries(job)
    snapshots = map(lib -> lib.bitcode === nothing ? nothing : bitcode_library(lib), libraries)
    selection = map(library_key, libraries, snapshots)
    legacy_linker = uses_legacy_linker(job)
    dependencies = Tuple{Runtime.RuntimeMethodInstance,MethodInstance,CodeInstance}[]
    manifests = map(library_exports, libraries, snapshots)
    owners = Dict{String,Int}()
    for (index, exports) in enumerate(manifests), name in exports
        haskey(owners, name) && error("Duplicate device-library export $name")
        owners[name] = index
    end
    for (index, library) in enumerate(libraries)
        needed = [value.name for value in Iterators.flatten((mod.functions, mod.globals))
                  if isdeclaration(value) && !isempty(value.uses) &&
                     value.name in manifests[index]]
        isempty(needed) && continue
        artifact, artifact_relocs, cached = load_device_library_record(library, job,
                                                                     snapshots[index])
        if library.bitcode === nothing
            append!(dependencies, zip(library.methods, cached.sources, cached.code_instances))
        end
        try
            for name in needed
                expected = get(mod.functions, name, nothing)
                definition = get(artifact.functions, name, nothing)
                expected === nothing && continue
                definition === nothing &&
                    error("Library $(library.name) does not define advertised export $name")
                expected.function_type == definition.function_type ||
                    error("Library $(library.name) has an incompatible ABI for $name")
            end
            append!(needed, [global_var.name for global_var in artifact.globals
                             if global_var.externally_initialized])
            for value in Iterators.flatten((artifact.functions, artifact.globals))
                if value.name in needed && !isdeclaration(value) &&
                   value.linkage in (LLVM.Linkage.LinkOnceAny, LLVM.Linkage.LinkOnceODR)
                    value.linkage = LLVM.Linkage.External
                end
            end
            with_llvm_machine(job.config.target) do tm
                run!(InternalizePass(; preserved_gvs=needed), artifact, tm)
                run!(GlobalDCEPass(), artifact, tm)
            end
            validate_library_imports(job, library, artifact, artifact_relocs, owners, index,
                                     legacy_linker)
        catch
            dispose(artifact)
            rethrow()
        end
        artifact.triple = mod.triple
        artifact.datalayout = mod.datalayout
        link_relocatable!(mod, relocs, artifact, artifact_relocs; only_needed=true)
    end
    link_libraries!(job, mod)
    imported = [fun.name for fun in mod.functions if !isdeclaration(fun) &&
                fun.name ∉ existing]
    finish_imports!(job, mod, imported)
    lower_syncscopes!(job, mod)
    append!(preserved, (record.name for record in relocs.records))
    with_llvm_machine(job.config.target) do tm
        run!(InternalizePass(; preserved_gvs=preserved), mod, tm)
    end
    finish_runtime_intrinsics!(job, mod)
    record_library_dependencies!(job, selection, dependencies)
    return
end

@public DeviceLibrary, DeviceLibraryMethod, device_libraries, runtime_library
@public runtime_imports, load_device_library
