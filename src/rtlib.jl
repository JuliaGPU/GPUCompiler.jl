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
function emit_function!(mod, config::CompilerConfig, source::MethodInstance, method,
                        world::UInt)
    name = method.llvm_name
    rt_job = CompilerJob(source, config, world)

    # Don't run a standalone inference walk on a miss: `compile_unhooked` below drives
    # inference itself.
    ci, res = runtime_function_results(rt_job)
    if res !== nothing && res.bitcode !== nothing
        link!(mod, parse(LLVM.Module, MemoryBuffer(res.bitcode)))
        ci === nothing && (ci = runtime_code_instance(rt_job))
        return ci::CodeInstance
    end

    new_mod, meta = compile_unhooked(:llvm, rt_job)
    ft = function_type(meta.entry)
    expected_ft = convert(LLVM.FunctionType, method)
    if return_type(ft) != return_type(expected_ft)
        error("Invalid return type for runtime function '$(method.name)': expected $(return_type(expected_ft)), got $(return_type(ft))")
    end

    # recent Julia versions include prototypes for all runtime functions, even if unused
    run!(StripDeadPrototypesPass(), new_mod, llvm_machine(config.target))

    # runtime functions may reference Julia objects through `julia.constgv` globals (e.g.
    # `box_bool` returning the `jl_true`/`jl_false` singletons). Kernels get theirs
    # relocated when the fully-linked toplevel module is finalized, but the runtime's
    # mappings would be dropped along with the rest of its per-function metadata: resolve
    # the slots into the cached bitcode instead, and keep functions whose resolution baked
    # session-absolute addresses (rather than materialized session-portable constants)
    # out of package images.
    if !isempty(meta.gv_to_value)
        portable = relocate_gvs!(new_mod, meta.gv_to_value)
        portable || mark_session_dependent!(rt_job)
    end

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
    ci === nothing && (ci = runtime_code_instance(rt_job))
    res === nothing && (res = job_results(RuntimeFunctionResults, ci, rt_job.config))
    res.bitcode = take!(io)

    link!(mod, new_mod)
    return ci::CodeInstance
end

function runtime_function_results(@nospecialize(job::CompilerJob))
    ci = job_code_instance(job)
    ci === nothing && return nothing, nothing
    return ci, job_results(RuntimeFunctionResults, ci, job.config)
end

function runtime_method_instance(@nospecialize(job::CompilerJob), method)
    def = if isa(method.def, Symbol)
        isdefined(runtime_module(job), method.def) || return nothing
        getfield(runtime_module(job), method.def)
    else
        method.def
    end
    # Resolve at the requesting job's explicit world, rather than this task's TLS world.
    # Runtime methods may be redefined while a long-lived compilation task is running.
    return generic_methodinstance(
        typeof(def), Base.to_tuple_type(method.types), job.world)
end

function runtime_code_instance(@nospecialize(job::CompilerJob))
    ci = @static if HAS_INTEGRATED_CACHE
        job_code_instance(job)
    else
        cache = WorldView(get_code_cache(job), job.world, job.world)
        CC.get(cache, job.source, nothing)
    end
    ci === nothing && error("Missing CodeInstance after compiling $(job.source)")
    return ci::CodeInstance
end

# the compiler job passed into here identifies the job that requires the runtime.
# derive a config that represents the runtime itself (notably with kernel=false).
# Fields that identify or optimize only the *kernel* job are reset so runtime artifacts are
# keyed identically for all kernels sharing the remaining codegen-relevant settings. Runtime
# functions always use the specfunc ABI and are deliberately left unoptimized until linked
# into the toplevel module, making the kernel's entry ABI and LLVM opt level irrelevant.
function runtime_config(@nospecialize(job::CompilerJob))
    CompilerConfig(job.config; kernel=false, entry_abi=:specfunc, opt_level=0,
                   toplevel=false, only_entry=false, strip=false, name=nothing)
end

function build_runtime(@nospecialize(job::CompilerJob), config::CompilerConfig)
    mod = LLVM.Module("GPUCompiler run-time library")
    sources = MethodInstance[]
    code_instances = CodeInstance[]

    for method in values(Runtime.methods)
        resolved = runtime_method_instance(job, method)
        resolved === nothing && continue
        source = resolved
        push!(sources, source)
        push!(code_instances, emit_function!(mod, config, source, method, job.world))
    end

    # we cannot optimize the runtime library, because the code would then be optimized again
    # during main compilation (and optimizing twice isn't safe). for example, optimization
    # removes Julia address spaces, which would then lead to type mismatches when using
    # functions from the runtime library from IR that has not been stripped of AS info.

    return mod, sources, code_instances
end

# Session-local cache of assembled runtime libraries, keyed by
# `(runtime_config(job), opaque_pointers)`: the derived runtime config covers every
# codegen-relevant setting (e.g. the debug level, which is baked into the runtime IR
# as a constant), while cosmetic kernel-job fields are normalized away. Cross-session
# persistence happens at the per-function level (see `RuntimeFunctionResults`):
# reassemble on first use of each session, then reuse within the session.
#
# Keep both the selected MethodInstances and their contributing CodeInstances with the
# assembled bytes. Re-resolving methods after the world changes detects direct method-table
# changes; clipped CI ranges detect invalidated transitive callees. This makes the assembled
# cache follow Julia's validity model without a manual `reset_runtime` hook.
mutable struct RuntimeLibrary
    bytes::Vector{UInt8}
    sources::Vector{MethodInstance}
    code_instances::Vector{CodeInstance}
    validated_world::UInt
end

function runtime_library_valid(lib::RuntimeLibrary, @nospecialize(job::CompilerJob))
    # Method-table changes and CI invalidations always advance the world counter. A runtime
    # already validated for this exact world therefore needs no per-function scan on the
    # common cache-hit path.
    job.world == lib.validated_world && return true

    i = 0
    for method in values(Runtime.methods)
        resolved = runtime_method_instance(job, method)
        resolved === nothing && continue
        i += 1
        i <= length(lib.sources) || return false
        resolved === lib.sources[i] || return false
    end
    i == length(lib.sources) || return false
    all(ci -> ci.min_world <= job.world <= ci.max_world, lib.code_instances) || return false
    lib.validated_world = job.world
    return true
end

const runtime_libs = Dict{Tuple{CompilerConfig, Bool}, RuntimeLibrary}()
const runtime_libs_lock = ReentrantLock()

@locked function load_runtime(@nospecialize(job::CompilerJob))
    config = runtime_config(job)
    key = (config, !supports_typed_pointers(context()))

    cached = Base.@lock runtime_libs_lock begin
        cached = get(runtime_libs, key, nothing)
        if cached === nothing || !runtime_library_valid(cached, job)
            lib, sources, code_instances = build_runtime(job, config)
            io = IOBuffer()
            write(io, lib)
            cached = RuntimeLibrary(take!(io), sources, code_instances, job.world)
            runtime_libs[key] = cached
        end
        cached
    end

    return parse(LLVM.Module, MemoryBuffer(cached.bytes); lazy=true)
end
