# compiler support for working with run-time libraries

#
# GPU run-time library
#


## higher-level functionality to work with runtime functions

function LLVM.call!(builder, rt::Runtime.RuntimeMethodInstance, args=LLVM.Value[])
    bb = builder.insert_block
    f = bb.parent
    mod = f.parent

    # get or create a function prototype
    if haskey(mod.functions, rt.llvm_name)
        f = mod.functions[rt.llvm_name]
        ft = f.function_type
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
    if length(args) != length(ft.parameters)
        error("Incorrect number of arguments for runtime function: ",
              "passing ", length(args), " argument(s) to '", string(ft), " ", rt.name, "'")
    end
    for (i,arg) in enumerate(args)
        if arg.value_type != ft.parameters[i]
            args[i] = if (arg.value_type isa LLVM.PointerType) &&
               (ft.parameters[i] isa LLVM.IntegerType)
                # pointers are passed as integers on Julia 1.11 and earlier
                ptrtoint!(builder, args[i], ft.parameters[i])
            elseif arg.value_type isa LLVM.PointerType &&
                   ft.parameters[i] isa LLVM.PointerType &&
                   arg.value_type.addrspace != ft.parameters[i].addrspace
                # runtime functions are always in the default address space,
                # while arguments may come from globals in other address spaces.
                addrspacecast!(builder, args[i], ft.parameters[i])
            else
                error("Don't know how to convert ", arg, " argument to ", ft.parameters[i])
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
    ft = meta.entry.function_type
    expected_ft = convert(LLVM.FunctionType, method)
    if ft.return_type != expected_ft.return_type
        error("Invalid return type for runtime function '$(method.name)': expected $(expected_ft.return_type), got $(ft.return_type)")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    run!(StripDeadPrototypesPass(), new_mod, llvm_machine(config.target))

    temp_name = meta.entry.name
    link!(mod, new_mod)
    entry = mod.functions[temp_name]

    # if a declaration already existed, replace it with the function to avoid aliasing
    # (and getting function names like gpu_signal_exception1)
    name = method.llvm_name
    if haskey(mod.functions, name)
        decl = mod.functions[name]
        @assert decl.value_type == entry.value_type
        replace_uses!(decl, entry)
        erase!(decl)
    end
    entry.name = name
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

@static if VERSION >= v"1.11.0"
    import Core.Compiler: is_asserts
else
    is_asserts() = false
end

@locked function load_runtime(@nospecialize(job::CompilerJob))
    global compile_cache
    if compile_cache === nothing    # during precompilation
        return build_runtime(job)
    end

    slug = runtime_slug(job)
    if !supports_typed_pointers(context())
        slug *= "-opaque"
    end

    # Julia codegen changes metadata in modules when `FORCE_ASSERTIONS=1`
    if is_asserts()
        slug *= "-asserts"
    end

    name = "runtime_$(slug).bc"
    path = joinpath(compile_cache, name)

    # the cache is shared across processes and may disappear at any point
    # (e.g. `reset_runtime()` in another process), so treat it as best-effort
    if ispath(path)
        try
            return parse(LLVM.Module, MemoryBufferFile(path); lazy=true)
        catch err
            @debug "Failed to load cached GPU runtime library; rebuilding" exception=(err, catch_backtrace())
        end
    end

    @debug "Building the GPU runtime library at $path"
    lib = build_runtime(job)

    try
        # atomic write to disk
        mkpath(compile_cache)
        temp_path, io = mktemp(compile_cache; cleanup=false)
        write(io, lib)
        close(io)
        @static if VERSION >= v"1.12.0-DEV.1023"
            mv(temp_path, path; force=true)
        else
            Base.rename(temp_path, path, force=true)
        end
    catch err
        @warn "Failed to cache GPU runtime library" exception=(err, catch_backtrace()) maxlog=1
    end

    return lib
end

# remove the existing cache
# NOTE: call this function from global scope, so any change triggers recompilation.
reset_runtime() = rm(compile_cache; recursive=true, force=true)
