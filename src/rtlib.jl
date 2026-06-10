# compiler support for working with run-time libraries

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

# Per-function compilation results for the GPU runtime library, cached through the
# same `cached_results` mechanism back-ends use for kernels. On 1.11+ the bitcode
# thus lives on the runtime function's `CodeInstance` — possibly alongside a
# back-end's own results struct — and persists through precompilation, so sessions
# loading a back-end that compiled its runtime during precompile skip codegen
# entirely. On 1.10 it is cached for the duration of the session.
mutable struct RuntimeFunctionResults
    bitcode::Union{Nothing,Vector{UInt8}}
    RuntimeFunctionResults() = new(nothing)
end

# Compile a single runtime function and link it into `mod`. The renamed bitcode is
# memoized through `RuntimeFunctionResults`; the session-local `runtime_libs` cache
# below additionally avoids repeating the parse-and-link work within a session.
function emit_function!(mod, config::CompilerConfig, f, method)
    tt = Base.to_tuple_type(method.types)
    source = generic_methodinstance(f, tt)
    name = method.llvm_name
    rt_job = CompilerJob(source, config)

    res = cached_results(RuntimeFunctionResults, rt_job)
    if res.bitcode !== nothing
        link!(mod, parse(LLVM.Module, MemoryBuffer(res.bitcode)))
        return
    end

    new_mod, meta = compile_unhooked(:llvm, rt_job)
    ft = function_type(meta.entry)
    expected_ft = convert(LLVM.FunctionType, method)
    if return_type(ft) != return_type(expected_ft)
        error("Invalid return type for runtime function '$(method.name)': expected $(return_type(expected_ft)), got $(return_type(ft))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    run!(StripDeadPrototypesPass(), new_mod, llvm_machine(config.target))

    # rename to the final `gpu_*` name on the per-function module, so the cached bitcode
    # is immediately link-ready (no per-session rename pass on a cache hit).
    if haskey(functions(new_mod), name) && functions(new_mod)[name] !== meta.entry
        decl = functions(new_mod)[name]
        @assert value_type(decl) == value_type(meta.entry)
        replace_uses!(decl, meta.entry)
        erase!(decl)
    end
    LLVM.name!(meta.entry, name)

    io = IOBuffer()
    write(io, new_mod)
    res.bitcode = take!(io)

    link!(mod, new_mod)
end

function build_runtime(@nospecialize(job::CompilerJob))
    mod = LLVM.Module("GPUCompiler run-time library")

    # the compiler job passed into here identifies the job that requires the runtime.
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

# Session-local cache of assembled runtime libraries, keyed by
# `(cache_owner(job), opaque_pointers)`. Cross-session persistence is the back-end's
# concern: rebuild on first use of each session, then reuse within the session.
const runtime_libs = Dict{Tuple{Any, Bool}, Vector{UInt8}}()
const runtime_libs_lock = ReentrantLock()

@locked function load_runtime(@nospecialize(job::CompilerJob))
    key = (cache_owner(job), !supports_typed_pointers(context()))

    bytes = Base.@lock runtime_libs_lock get!(runtime_libs, key) do
        lib = build_runtime(job)
        io = IOBuffer()
        write(io, lib)
        take!(io)
    end

    return parse(LLVM.Module, MemoryBuffer(bytes); lazy=true)
end

# clear the session-local runtime library cache
reset_runtime() = Base.@lock runtime_libs_lock empty!(runtime_libs)
