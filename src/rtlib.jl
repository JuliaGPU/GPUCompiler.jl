# compiler support for working with run-time libraries

link_library!(mod::LLVM.Module, lib::LLVM.Module) = link_library!(mod, [lib])
function link_library!(mod::LLVM.Module, libs::Vector{LLVM.Module})
    # linking is destructive, so copy the libraries
    libs = [LLVM.Module(lib) for lib in libs]

    for lib in libs
        link!(mod, lib)
    end
end


#
# GPU run-time library
#

# get the path to a directory where we can put cache files (machine-specific, ephemeral)
# NOTE: maybe we should use XDG_CACHE_PATH/%LOCALAPPDATA%, but other Julia cache files
#       are put in .julia anyway so let's just follow suit for now.
function cachedir(depot=DEPOT_PATH[1])
    # this mimicks Base.compilecache. we can't just call the function, or we might actually
    # _generate_ a cache file, e.g., when running with `--compiled-modules=no`.
    entrypath, entryfile = Base.cache_file_entry(Base.PkgId(GPUCompiler))
    abspath(depot, entrypath, entryfile)
end


## higher-level functionality to work with runtime functions

function LLVM.call!(builder, rt::Runtime.RuntimeMethodInstance, args=LLVM.Value[];
                    state::Type=Nothing)
    bb = position(builder)
    f = LLVM.parent(bb)
    mod = LLVM.parent(f)
    ctx = context(mod)

    # get or create a function prototype
    if haskey(functions(mod), rt.llvm_name)
        f = functions(mod)[rt.llvm_name]
        ft = eltype(llvmtype(f))
    else
        ft = convert(LLVM.FunctionType, rt; ctx, state)
        f = LLVM.Function(mod, rt.llvm_name, ft)
    end

    # we may be calling this function after kernel state lowering,
    # in which case we need to manually get and pass the state.
    args = Value[args...]
    if state !== Nothing
        T_state = convert(LLVMType, state; ctx)

        state_intr = kernel_state_intr(mod, T_state)
        state_val = call!(builder, state_intr, Value[], "state")
        pushfirst!(args, state_val)
    end

    # runtime functions are written in Julia, while we're calling from LLVM,
    # this often results in argument type mismatches. try to fix some here.
    for (i,arg) in enumerate(args)
        if llvmtype(arg) != parameters(ft)[i]
            if (llvmtype(arg) isa LLVM.PointerType) &&
               (parameters(ft)[i] isa LLVM.IntegerType)
                # Julia pointers are passed as integers
                args[i] = ptrtoint!(builder, args[i], parameters(ft)[i])
            else
                error("Don't know how to convert ", arg, " argument to ", parameters(ft)[i])
            end
        end
    end

    call!(builder, f, args)
end


## functionality to build the runtime library

function emit_function!(mod, @nospecialize(job::CompilerJob), f, method)
    tt = Base.to_tuple_type(method.types)
    new_mod, meta = codegen(:llvm, similar(job, FunctionSpec(f, tt, #=kernel=# false));
                            optimize=false, libraries=false)
    ft = eltype(llvmtype(meta.entry))
    expected_ft = convert(LLVM.FunctionType, method; ctx=context(new_mod))
    if return_type(ft) != return_type(expected_ft)
        error("Invalid return type for runtime function '$(method.name)': expected $(return_type(expected_ft)), got $(return_type(ft))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    pm = ModulePassManager()
    strip_dead_prototypes!(pm)
    run!(pm, new_mod)
    dispose(pm)

    temp_name = LLVM.name(meta.entry)
    # FIXME: on 1.6, there's no single global LLVM context anymore,
    #        but there's no API yet to pass a context to codegen.
    # round-trip the module through serialization to get it in the proper context.
    buf = convert(MemoryBuffer, new_mod)
    new_mod = parse(LLVM.Module, buf; ctx=context(mod))
    @assert context(mod) == context(new_mod)
    link!(mod, new_mod)
    entry = functions(mod)[temp_name]

    # if a declaration already existed, replace it with the function to avoid aliasing
    # (and getting function names like gpu_signal_exception1)
    name = method.llvm_name
    if haskey(functions(mod), name)
        decl = functions(mod)[name]
        @assert llvmtype(decl) == llvmtype(entry)
        replace_uses!(decl, entry)
        unsafe_delete!(mod, decl)
    end
    LLVM.name!(entry, name)
end

function build_runtime(@nospecialize(job::CompilerJob); ctx)
    mod = LLVM.Module("GPUCompiler run-time library"; ctx)

    # the compiler job passed into here is identifies the job that requires the runtime.
    # derive a job that represents the runtime itself (notably with kernel=false).
    source = FunctionSpec(identity, Tuple{Nothing}, false, nothing, job.source.world_age)
    job = CompilerJob(job.target, source, job.params)

    for method in values(Runtime.methods)
        def = if isa(method.def, Symbol)
            isdefined(runtime_module(job), method.def) || continue
            getfield(runtime_module(job), method.def)
        else
            method.def
        end
        emit_function!(mod, job, def, method)
    end

    # we cannot optimize the runtime library, because the code would then be optimized again
    # during main compilation (and optimizing twice isn't safe). for example, optimization
    # removes Julia address spaces, which would then lead to type mismatches when using
    # functions from the runtime library from IR that has not been stripped of AS info.

    mod
end

const runtime_lock = ReentrantLock()

@locked function load_runtime(@nospecialize(job::CompilerJob); ctx)
    lock(runtime_lock) do
        # find the first existing cache directory (for when dealing with layered depots)
        cachedirs = [cachedir(depot) for depot in DEPOT_PATH]
        filter!(isdir, cachedirs)
        input_dir = if isempty(cachedirs)
            nothing
        else
            first(cachedirs)
        end

        # we are only guaranteed to be able to write in the current depot
        output_dir = cachedir()

        # if both aren't equal, copy pregenerated runtime libraries to our depot
        # NOTE: we don't just lazily read from the one and write to the other, because
        #       once we generate additional runtimes in the output dir we don't know if
        #       it's safe to load from other layers (since those could have been invalidated)
        if input_dir !== nothing && input_dir != output_dir
            mkpath(dirname(output_dir))
            cp(input_dir, output_dir)
        end

        slug = runtime_slug(job)
        name = "runtime_$(slug).bc"
        path = joinpath(output_dir, name)

        lib = try
            if ispath(path)
                open(path) do io
                    parse(LLVM.Module, read(io); ctx)
                end
            end
        catch ex
            @warn "Failed to load GPU runtime library at $path" exception=(ex, catch_backtrace())
            nothing
        end

        if lib === nothing
            @debug "Building the GPU runtime library at $path"
            mkpath(output_dir)
            lib = build_runtime(job; ctx)

            # atomic write to disk
            temp_path, io = mktemp(dirname(path); cleanup=false)
            write(io, lib)
            close(io)
            Base.rename(temp_path, path; force=true)
        end

        return lib
    end
end

# remove the existing cache
# NOTE: call this function from global scope, so any change triggers recompilation.
function reset_runtime()
    lock(runtime_lock) do
        rm(cachedir(); recursive=true, force=true)
        # create an empty cache directory. since we only ever load from the first existing cachedir,
        # this effectively invalidates preexisting caches in lower layers of the depot.
        mkpath(cachedir())
    end

    return
end
