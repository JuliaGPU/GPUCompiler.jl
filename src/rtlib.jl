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


## higher-level functionality to work with runtime functions

function LLVM.call!(builder, rt::Runtime.RuntimeMethodInstance, args=LLVM.Value[])
    bb = position(builder)
    f = LLVM.parent(bb)
    mod = LLVM.parent(f)

    # get or create a function prototype
    if haskey(functions(mod), rt.llvm_name)
        f = functions(mod)[rt.llvm_name]
        ft = function_type(f)
    else
        ft = convert(LLVM.FunctionType, rt)
        f = LLVM.Function(mod, rt.llvm_name, ft)
    end

    # runtime functions are written in Julia, while we're calling from LLVM,
    # this often results in argument type mismatches. try to fix some here.
    args = LLVM.Value[args...]
    for (i,arg) in enumerate(args)
        if value_type(arg) != parameters(ft)[i]
            if (value_type(arg) isa LLVM.PointerType) &&
               (parameters(ft)[i] isa LLVM.IntegerType)
                # Julia pointers are passed as integers
                args[i] = ptrtoint!(builder, args[i], parameters(ft)[i])
            else
                error("Don't know how to convert ", arg, " argument to ", parameters(ft)[i])
            end
        end
    end

    call!(builder, ft, f, args)
end


## functionality to build the runtime library

function emit_function!(mod, config::CompilerConfig, f, method)
    tt = Base.to_tuple_type(method.types)
    source = generic_methodinstance(f, tt)
    new_mod, meta = codegen(:llvm, CompilerJob(source, config);
                            optimize=false, libraries=false, validate=false)
    ft = function_type(meta.entry)
    expected_ft = convert(LLVM.FunctionType, method)
    if return_type(ft) != return_type(expected_ft)
        error("Invalid return type for runtime function '$(method.name)': expected $(return_type(expected_ft)), got $(return_type(ft))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    if use_newpm
        run!(StripDeadPrototypesPass(), new_mod, llvm_machine(config.target))
    else
        @dispose pm=ModulePassManager() begin
            strip_dead_prototypes!(pm)
            run!(pm, new_mod)
        end
    end

    temp_name = LLVM.name(meta.entry)
    link!(mod, new_mod)
    entry = functions(mod)[temp_name]

    # if a declaration already existed, replace it with the function to avoid aliasing
    # (and getting function names like gpu_signal_exception1)
    name = method.llvm_name
    if haskey(functions(mod), name)
        decl = functions(mod)[name]
        @assert value_type(decl) == value_type(entry)
        replace_uses!(decl, entry)
        unsafe_delete!(mod, decl)
    end
    LLVM.name!(entry, name)
end

function build_runtime(@nospecialize(job::CompilerJob))
    mod = LLVM.Module("GPUCompiler run-time library")

    # the compiler job passed into here is identifies the job that requires the runtime.
    # derive a job that represents the runtime itself (notably with kernel=false).
    config = CompilerConfig(job.config; kernel=false)

    for method in values(Runtime.methods)
        def = if isa(method.def, Symbol)
            isdefined(runtime_module(job), method.def) || continue
            getfield(runtime_module(job), method.def)
        else
            method.def
        end
        emit_function!(mod, config, typeof(def), method)
    end

    # we cannot optimize the runtime library, because the code would then be optimized again
    # during main compilation (and optimizing twice isn't safe). for example, optimization
    # removes Julia address spaces, which would then lead to type mismatches when using
    # functions from the runtime library from IR that has not been stripped of AS info.

    mod
end

const runtime_lock = ReentrantLock()

@locked function load_runtime(@nospecialize(job::CompilerJob))
    lock(runtime_lock) do
        slug = runtime_slug(job)
        if !supports_typed_pointers(context())
            slug *= "-opaque"
        end
        name = "runtime_$(slug).bc"
        path = joinpath(compile_cache, name)

        lib = try
            if ispath(path)
                open(path) do io
                    parse(LLVM.Module, read(io))
                end
            end
        catch ex
            @warn "Failed to load GPU runtime library at $path" exception=(ex, catch_backtrace())
            nothing
        end

        if lib === nothing
            @debug "Building the GPU runtime library at $path"
            mkpath(compile_cache)
            lib = build_runtime(job)

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
        rm(compile_cache; recursive=true, force=true)
    end

    return
end
