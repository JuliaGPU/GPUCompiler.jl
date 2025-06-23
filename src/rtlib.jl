# compiler support for working with run-time libraries

link_library!(mod::LLVM.Module, lib::LLVM.Module) = link_library!(mod, [lib])
function link_library!(mod::LLVM.Module, libs::Vector{LLVM.Module})
    # linking is destructive, so copy the libraries
    libs = [copy(lib) for lib in libs]

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
    if !isdeclaration(f) && (rt.name !== :gc_pool_alloc && rt.name !== :report_exception)
        # XXX: uses of the gc_pool_alloc intrinsic can be introduced _after_ the runtime
        #      is linked, as part of the lower_gc_frame! optimization pass.
        # XXX: report_exception can also be used after the runtime is linked during
        #      CUDA/Enzyme nested compilation
        error("Calling an intrinsic function that clashes with an existing definition: ",
               string(ft), " ", rt.name)
    end

    # runtime functions are written in Julia, while we're calling from LLVM,
    # this often results in argument type mismatches. try to fix some here.
    args = LLVM.Value[args...]
    if length(args) != length(parameters(ft))
        error("Incorrect number of arguments for runtime function: ",
              "passing ", length(args), " argument(s) to '", string(ft), " ", rt.name, "'")
    end
    for (i,arg) in enumerate(args)
        if value_type(arg) != parameters(ft)[i]
            args[i] = if (value_type(arg) isa LLVM.PointerType) &&
               (parameters(ft)[i] isa LLVM.IntegerType)
                # pointers are passed as integers on Julia 1.11 and earlier
                ptrtoint!(builder, args[i], parameters(ft)[i])
            elseif value_type(arg) isa LLVM.PointerType &&
                   parameters(ft)[i] isa LLVM.PointerType &&
                   addrspace(value_type(arg)) != addrspace(parameters(ft)[i])
                # runtime functions are always in the default address space,
                # while arguments may come from globals in other address spaces.
                addrspacecast!(builder, args[i], parameters(ft)[i])
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
    new_mod, meta = compile_unhooked(:llvm, CompilerJob(source, config))
    ft = function_type(meta.entry)
    expected_ft = convert(LLVM.FunctionType, method)
    if return_type(ft) != return_type(expected_ft)
        error("Invalid return type for runtime function '$(method.name)': expected $(return_type(expected_ft)), got $(return_type(ft))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    run!(StripDeadPrototypesPass(), new_mod, llvm_machine(config.target))

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
        erase!(decl)
    end
    LLVM.name!(entry, name)
end

function build_runtime(@nospecialize(job::CompilerJob))
    mod = LLVM.Module("GPUCompiler run-time library")

    # the compiler job passed into here is identifies the job that requires the runtime.
    # derive a job that represents the runtime itself (notably with kernel=false).
    config = CompilerConfig(job.config; kernel=false, toplevel=false, only_entry=false, strip=false)

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

const runtime_cache = Dict{String, Vector{UInt8}}()

@locked function load_runtime(@nospecialize(job::CompilerJob))
    global compile_cache
    if compile_cache === nothing    # during precompilation
        return build_runtime(job)
    end

    lock(runtime_lock) do
        slug = runtime_slug(job)
        if !supports_typed_pointers(context())
            slug *= "-opaque"
        end
        name = "runtime_$(slug).bc"
        path = joinpath(compile_cache, name)

        # cache the runtime library on disk and in memory
        lib = if haskey(runtime_cache, slug)
            parse(LLVM.Module, runtime_cache[slug])
        elseif ispath(path)
            runtime_cache[slug] = open(path) do io
                read(io)
            end
            parse(LLVM.Module, runtime_cache[slug])
        end

        if lib === nothing
            @debug "Building the GPU runtime library at $path"
            mkpath(compile_cache)
            lib = build_runtime(job)

            # atomic write to disk
            temp_path, io = mktemp(dirname(path); cleanup=false)
            write(io, lib)
            close(io)
            @static if VERSION >= v"1.12.0-DEV.1023"
                mv(temp_path, path; force=true)
            else
                Base.rename(temp_path, path, force=true)
            end
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
