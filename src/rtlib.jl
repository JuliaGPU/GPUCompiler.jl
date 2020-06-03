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

const libcache = Dict{String, LLVM.Module}()

# get the path to a directory where we can put cache files (machine-specific, ephemeral)
# NOTE: maybe we should use XDG_CACHE_PATH/%LOCALAPPDATA%, but other Julia cache files
#       are put in .julia anyway so let's just follow suit for now.
function cachedir(depot=DEPOT_PATH[1])
    # this mimicks Base.compilecache. we can't just call the function, or we might actually
    # _generate_ a cache file, e.g., when running with `--compiled-modules=no`.
    if VERSION >= v"1.3.0-alpha.146"
        entrypath, entryfile = Base.cache_file_entry(Base.PkgId(GPUCompiler))
        abspath(depot, entrypath, entryfile)
    else
        cachefile = abspath(depot, Base.cache_file_entry(Base.PkgId(GPUCompiler)))

        # the cachefile consists of `/depot/compiled/vXXX/GPUCompiler/$slug.ji`
        # transform that into `/depot/compiled/vXXX/GPUCompiler/$slug/`
        splitext(cachefile)[1]
    end
end


## higher-level functionality to work with runtime functions

function LLVM.call!(builder, rt::Runtime.RuntimeMethodInstance, args=LLVM.Value[])
    bb = position(builder)
    f = LLVM.parent(bb)
    mod = LLVM.parent(f)

    # get or create a function prototype
    if haskey(functions(mod), rt.llvm_name)
        f = functions(mod)[rt.llvm_name]
        ft = eltype(llvmtype(f))
    else
        ft = LLVM.FunctionType(rt.llvm_return_type, rt.llvm_types)
        f = LLVM.Function(mod, rt.llvm_name, ft)
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

function emit_function!(mod, job::CompilerJob, f, method)
    tt = Base.to_tuple_type(method.types)
    new_mod, entry = codegen(:llvm, similar(job, FunctionSpec(f, tt, #=kernel=# false));
                             optimize=false, libraries=false)
    if return_type(eltype(llvmtype(entry))) != method.llvm_return_type
        error("Invalid return type for runtime function '$(method.name)': expected $(method.llvm_return_type), got $(return_type(eltype(llvmtype(entry))))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    if VERSION >= v"1.5-"
        pm = ModulePassManager()
        strip_dead_prototypes!(pm)
        run!(pm, new_mod)
        dispose(pm)
    end

    temp_name = LLVM.name(entry)
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

function build_runtime(job::CompilerJob)
    mod = LLVM.Module("GPUCompiler run-time library", JuliaContext())

    for method in values(Runtime.methods)
        def = if isa(method.def, Symbol)
            isdefined(runtime_module(job), method.def) || continue
            getfield(runtime_module(job), method.def)
        else
            method.def
        end
        emit_function!(mod, job, def, method)
    end

    optimize!(job, mod)

    mod
end

function load_runtime(job::CompilerJob)
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

    get!(libcache, path) do
        if ispath(path)
            open(path) do io
                parse(LLVM.Module, read(io), JuliaContext())
            end
        else
            @debug "Building the GPU runtime library at $path"
            mkpath(output_dir)
            lib = build_runtime(job)
            open(path, "w") do io
                write(io, lib)
            end
            lib
        end
    end
end

# remove the existing cache
# NOTE: call this function from global scope, so any change triggers recompilation.
function reset_runtime()
    rm(cachedir(); recursive=true, force=true)
    # create an empty cache directory. since we only ever load from the first existing cachedir,
    # this effectively invalidates preexisting caches in lower layers of the depot.
    mkpath(cachedir())

    # wipe the cache so we can use this function at run-time too
    empty!(libcache)

    return
end
