# compiler driver and main interface


## LLVM context handling

export JuliaContext

# transitionary feature to deal versions of Julia that rely on a global context
#
# Julia 1.9 removed the global LLVM context, requiring to pass a context to codegen APIs,
# so the GPUCompiler APIs have been adapted to require passing a Context object as well.
# however, on older versions of Julia we cannot make codegen emit into that context. we
# could use a hack (serialize + deserialize) to move code into the correct context, however
# as it turns out some of our optimization passes erroneously rely on the context being
# global and unique, resulting in segfaults when we use a local context instead.
#
# to work around this mess, and still present a reasonably unified API, we introduce the
# JuliaContext helper below, which returns a local context on Julia 1.9, and the global
# unique context on all other versions. Once we only support Julia 1.9, we'll deprecate
# this helper to a regular `Context()` call.
function JuliaContext(; opaque_pointers=nothing)
    # XXX: remove
    ThreadSafeContext(; opaque_pointers)
end
function JuliaContext(f; kwargs...)
    ThreadSafeContext(; kwargs...) do ts_ctx
        # for now, also activate the underlying context
        # XXX: this is wrong; we can't expose the underlying LLVM context, but should
        #      instead always go through the callback in order to unlock it properly.
        #      rework this once we depend on Julia 1.9 or later.
        ctx = context(ts_ctx)
        context!(ctx) do
            f(ctx)
        end
    end
end


## compiler entrypoint

export compile

"""
    run_compile_hook(job)

Report `job` to the hook installed by [`with_compile_hook`](@ref) or a `@device_code_*`
macro in the current task, if any. Back-ends call this every time they look up a kernel,
whether or not its compilation is cached, so that reflection also sees cached kernels;
there is no need to recompile while a hook is installed. [`compile`](@ref) reports the
jobs it compiles too, so a job may be reported more than once.

Jobs need not be `CompilerJob`s or use LLVM. They must have stable `hash` and `isequal`
semantics for deduplication, support `show`, and implement the requested GPUCompiler
reflection functions: `code_lowered(job)`, `code_typed(job; kwargs...)`, or
`code_warntype`, `code_llvm`, and `code_native` with `(io, job; kwargs...)`. The
all-stage `@device_code` dump still requires a `CompilerJob` and the LLVM pipeline.
"""
function run_compile_hook(@nospecialize(job))
    hook = active_compile_hook[]
    # launches may run in a frozen world, while the hook lives in the latest one
    hook === nothing || Base.invokelatest(hook, job)
    return
end

"""
    with_compile_hook(f, hook)

Call `f()` with `hook` installed, calling `hook(job)` for the jobs that `f` and the tasks
it spawns compile or launch (see [`run_compile_hook`](@ref)). The same job can be
reported several times; pass `nothing` to report to no hook at all. Wait for child tasks
before `f` returns to observe all of them.
"""
with_compile_hook(f, hook) = with(f, active_compile_hook => hook)

"""
    compile(target::Symbol, job::CompilerJob)

Compile a `job` to one of the following formats as specified by the `target` argument:
`:llvm` for LLVM IR, `:asm` for assembly, or `:obj` for object code.

The default [`relocation_lowering`](@ref) strategy resolves Julia-value relocations in the
`:llvm` result. Other strategies retain relocation metadata for their loader.

The IR in the result (the `:llvm` module, or `meta.ir` and `meta.entry`) lives in the
LLVM context of the enclosing [`JuliaContext`](@ref) block, and is freed with it. Use it,
and dispose of it, within that block. The do-block form, `compile(f, target, job)`, takes
care of both.
"""
function compile(target::Symbol, @nospecialize(job::CompilerJob))
    run_compile_hook(job)
    return report_failure(job) do
        compile_unhooked(target, job)
    end
end

"""
    compile(f, target::Symbol, job::CompilerJob)

Compile `job` as `compile(target, job)` does, in an LLVM context of its own, and return
`f(result, meta)`. `result` is the module for `:llvm`, and the code otherwise.

`f` runs with the context of the IR active, so it can work with the module and its values
directly. The IR and its context are freed when `f` returns: return only values that don't
refer to them, e.g., strings, or the code.

```julia
compile(:obj, job) do obj, meta
    (; obj, entry=meta.entry.name)
end
```
"""
function compile(f, target::Symbol, @nospecialize(job::CompilerJob))
    return JuliaContext() do _
        result, meta = compile(target, job)
        ir = target === :llvm ? result : meta.ir
        try
            f(result, meta)
        finally
            dispose(ir)
        end
    end
end

function report_failure(f, @nospecialize(job::CompilerJob))
    hook = compile_failure_hook[]
    hook === nothing && return f()
    try
        f()
    catch err
        err isa InterruptException || Base.invokelatest(hook, job, err)
        rethrow()
    end
end

function compile_unhooked(output::Symbol, @nospecialize(job::CompilerJob);
                          resolve_relocations::Bool=true)
    if context(; throw_error=false) === nothing
        error("No active LLVM context. Use `JuliaContext()` do-block syntax to create one.")
    end

    @tracepoint "Validation" begin
        check_method(job)   # not optional
        job.config.validate && check_invocation(job)
    end

    prepare_job!(job)


    ## LLVM IR

    ir, ir_meta = emit_llvm(job; resolve_relocations)

    if output == :llvm
        if job.config.strip
            @tracepoint "strip debug info" strip_debuginfo!(ir)
        end

        return ir, ir_meta
    end


    ## machine code

    format = if output == :asm
        LLVM.CodeGenFileType.Assembly
    elseif output == :obj
        LLVM.CodeGenFileType.Object
    else
        error("Unknown assembly format $output")
    end
    asm, asm_meta = emit_asm(job, ir, ir_meta.relocations, format)

    if output == :asm || output == :obj
        return asm, (; asm_meta..., ir_meta..., ir)
    end


    error("Unknown compilation output $output")
end

# Deferred compilation: code obtains the address of a function that the compilation it ends
# up in compiles too (see `deferred_codegen` and `deferred_codegen!`), through calls to a
# `deferred_codegen` marker with an id that identifies what to compile.
#
# Jobs registered here directly, with the id of their marker as key, are the legacy way of
# requesting them, which Enzyme.jl and releases of back-ends that predate
# `deferred_codegen!` use. New code should use `deferred_codegen!` instead, which registers
# requests without racing with concurrent compilations.
const deferred_codegen_jobs = Dict{Int, Any}()

# We make this function explicitly callable so that we can drive OrcJIT's
# lazy compilation from, while also enabling recursive compilation.
# see `register_deferred_codegen`
function deferred_codegen(ptr::Ptr{Cvoid})::Ptr{Cvoid}
    ptr
end

"""
    deferred_codegen(::Val{ft}, ::Val{tt}) -> Ptr{Cvoid}

Return a pointer to the function of type `ft` compiled for arguments of the types in the
tuple type `tt`, by the job compiling the caller and in its world, e.g. to launch it as a
kernel from device code.
"""
deferred_codegen(f::Val, tt::Val) = Base.@noinline deferred_codegen_request(f, tt)

# The request is identified by the specialization of this method. `irgen` registers it,
# and gives the legacy marker in its code the request's id (see `lower_deferred_requests!`).
# Nothing is registered while inferring the caller, which happens concurrently on Julia
# 1.12+ and possibly in another process (when the caller's code is cached in a package
# image). The call is kept from being inlined (also when inlining the public wrapper), and
# its opaque result from being folded. The id 0 is never looked up, and keeps this method
# valid on the host (see `register_deferred_codegen`).
@noinline function deferred_codegen_request(::Val{ft}, ::Val{tt}) where {ft,tt}
    # TODO: add an edge to the child's method instance to support method redefinitions
    ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), 0)
end
const deferred_codegen_request_method = which(deferred_codegen_request, Tuple{Val,Val})

# Requests for deferred code generation: function and argument types relative to the
# requesting job (from `deferred_codegen(::Val, ::Val)`), or a job (from back-ends).
#
# The id of a request is its address, which the jobs in `deferred_codegen_jobs` don't use:
# they are registered under small integers or addresses of other objects. (Checking that
# would race with code registering them.) Ids only have a meaning in the process that
# generated the code: addresses change when an image is restored, so a request found by
# its id is only used if that is still its address, and registering it again indexes it
# under its current one. Requests are registered once, so that
# this only grows with the number of distinct ones. That may happen in generated functions,
# so use a lock that doesn't yield, and compare requests by identity, which calls no
# user-defined `hash`.
mutable struct DeferredRequest
    const spec::Any
end
const deferred_requests = IdDict{Any,DeferredRequest}()
const deferred_requests_by_id = Dict{Int,DeferredRequest}()
const deferred_requests_lock = Threads.SpinLock()

request_id(request::DeferredRequest) = reinterpret(Int, pointer_from_objref(request))

function register_deferred_request(@nospecialize(spec))
    Base.@lock deferred_requests_lock begin
        request = get!(deferred_requests, spec) do
            DeferredRequest(spec)
        end
        id = request_id(request)
        deferred_requests_by_id[id] = request
        return id
    end
end

# the request with a marker's id, or the job registered under it with `deferred_codegen_jobs`
function deferred_spec(id::Int)
    request = Base.@lock deferred_requests_lock get(deferred_requests_by_id, id, nothing)
    if request !== nothing && request_id(request) == id
        return request.spec
    end
    global deferred_codegen_jobs
    spec = get(deferred_codegen_jobs, id, nothing)
    spec === nothing && error("No deferred codegen request has id $id")
    return spec
end

# the job that compiles a request as part of `job`
function nest_deferred_job(@nospecialize(dyn_job::CompilerJob), @nospecialize(job::CompilerJob))
    target = nest_target(dyn_job.config.target, job.config.target)
    params = nest_params(dyn_job.config.params, job.config.params)
    config = CompilerConfig(dyn_job.config; toplevel=false, target, params)
    return CompilerJob(dyn_job; config)
end

# the job a request specifies, for the job whose module contains it
function deferred_job(@nospecialize(spec), @nospecialize(job::CompilerJob))
    spec isa CompilerJob && return spec
    ft, tt = spec
    return CompilerJob(methodinstance(ft, tt, job.world), job.config, job.world)
end

# In a module that Julia generated, give the legacy markers in the code of requests the
# id of their request. Unlike the names of functions, that survives linking modules.
function lower_deferred_requests!(mod::LLVM.Module, compiled)
    marker = get(mod.functions, "deferred_codegen", nothing)
    marker === nothing && return
    T_id = only(LLVM.parameters(marker.function_type))
    for (mi, entry) in compiled
        mi.def === deferred_codegen_request_method || continue
        _, ft, tt = Base.unwrap_unionall(mi.specTypes).parameters
        id = register_deferred_request((ft.parameters[1], tt.parameters[1]))
        for name in (entry.specfunc, entry.func)
            name === nothing && continue
            f = get(mod.functions, name, nothing)
            (f === nothing || isdeclaration(f)) && continue
            for bb in f.blocks, inst in bb.instructions
                inst isa LLVM.CallInst && inst.called_operand == marker || continue
                inst.operands[1] = ConstantInt(T_id, id)
            end
            # with the request identified by its id, it can be inlined
            delete!(f.function_attributes, :noinline)
            push!(f.function_attributes, EnumAttribute(:alwaysinline))
        end
    end
    return
end

"""
    deferred_codegen!(builder::IRBuilder, job::CompilerJob) -> LLVM.Value

Emit code at the position of `builder` that evaluates to the address of the function that
`job` compiles to. This is the counterpart of [`deferred_codegen`](@ref) for code
generation, e.g. to call a function from [`finish_module!`](@ref) that is only known to be
needed then. The compilation resolves requests after `finish_module!`, so they can't be made
later (e.g. from `finish_ir!`), and it doesn't resolve them when compiling only the entry
point (`only_entry=true`).

`job` is compiled like the functions that [`deferred_codegen`](@ref) requests: as part of
the requesting compilation, with its target and parameters nested in the requesting job's
ones (see `nest_target` and `nest_params`), but otherwise with its own source, world and
configuration (e.g. `kernel=false` for a function that is only called). Only kernels remain
entry points.

The result is an LLVM pointer. To call the function, cast it to a pointer to the function's
type (with `pointercast!`, which does nothing with opaque pointers). Once the request is
resolved, that call is a direct call of a defined function, which LLVM requires to have a
debug location if the calling function has debug info: set one on `builder` (e.g. with
`debug_location!`) before emitting the call.
"""
function deferred_codegen!(builder::IRBuilder, @nospecialize(job::CompilerJob))
    bb = builder.insert_block
    bb === nothing && throw(ArgumentError("The IR builder is not positioned"))
    mod = LLVM.parent(LLVM.parent(bb))
    context!(LLVM.context(mod)) do
        # the marker, as Julia declares it for `ccall("extern deferred_codegen", llvmcall,
        # Ptr{Cvoid}, (Int,), id)`
        ft = LLVM.FunctionType(convert(LLVMType, Ptr{Cvoid}), [convert(LLVMType, Int)])
        marker = get!(mod.functions, "deferred_codegen") do
            LLVM.Function(mod, "deferred_codegen", ft)
        end
        marker.function_type == ft ||
            error("The deferred_codegen marker in this module has type $(marker.function_type), expected $ft")

        id = register_deferred_request(job)
        ptr = call!(builder, ft, marker, [ConstantInt(only(LLVM.parameters(ft)), id)])
        if LLVM.return_type(ft) isa LLVM.IntegerType
            # Julia before 1.12 represents `Ptr` as an integer
            T_ptr = supports_typed_pointers(context()) ?
                    LLVM.PointerType(LLVM.Int8Type()) : LLVM.PointerType()
            ptr = inttoptr!(builder, ptr, T_ptr)
        end
        return ptr
    end
end

@public deferred_codegen, deferred_codegen!

# Register deferred_codegen as a global function so that it can be called with
# `ccall("extern deferred_codegen", ...)`. Called from __init__.
#
# On 1.11+ this is needed due to a Julia bug that drops the pointer when code-coverage is
# enabled. On 1.14+ (JuliaLang/julia#60988), there is no shared external JITDylib anymore;
# instead, register the symbol in the `JuliaGlobals` JD, which JD links to with
# `MatchExportedSymbolsOnly`.
function register_deferred_codegen()
    @dispose jljit=JuliaOJIT() begin
        jd = @static if VERSION >= v"1.14.0-DEV.2171"
            es = jljit.execution_session
            something(lookup_dylib(es, "JuliaGlobals"))
        else
            jljit.external_dylib
        end

        address = @cfunction(deferred_codegen, Ptr{Cvoid}, (Ptr{Cvoid},))
        name = mangle(jljit, "deferred_codegen")
        mu = absolute_symbols(name => address)
        define!(jd, mu)
        addr = lookup(jljit, jd, "deferred_codegen")
        @assert addr != C_NULL "Failed to register deferred_codegen"
    end
    return nothing
end

# LLVM's target registry isn't thread-safe: registering a target from two threads at once
# can make the list of targets cyclic, after which every target lookup hangs. so register
# them when loading GPUCompiler, which also lets `llvm_machine` work before compiling.
# register them with Julia's library, which GPUCompiler requires as its IR comes from
# Julia's code generator, even when the `libLLVM` preference makes another the default.
function initialize_llvm(lib::LLVM.Library=LLVM.Library(:julia))
    LLVM.library!(lib) do
        LLVM.InitializeAllTargets()
        LLVM.InitializeAllTargetInfos()
        LLVM.InitializeAllAsmPrinters()
        LLVM.InitializeAllAsmParsers()
        LLVM.InitializeAllTargetMCs()
    end
    return
end

# `__init__` doesn't run while generating output for GPUCompiler itself, or for a system
# image, so compilations register the targets then. LLVM's initializers write to the
# registry every time, so only do so once per process, as concurrent compilations may be
# using it. they may run in generated functions, so use a lock that doesn't yield, and
# look up the library (which may) outside of it.
const llvm_init_lock = Threads.SpinLock()
# whether this process registered them, as a pointer, which images reset to `C_NULL`
mutable struct LLVMInitState
    @atomic initialized::Ptr{Cvoid}
end
const llvm_init_state = LLVMInitState(C_NULL)
llvm_initialized() = (@atomic :acquire llvm_init_state.initialized) != C_NULL
mark_llvm_initialized() = @atomic :release llvm_init_state.initialized = Ptr{Cvoid}(1)
function initialize_llvm_for_output()
    llvm_initialized() && return
    lib = LLVM.Library(:julia)
    Base.@lock llvm_init_lock begin
        llvm_initialized() || initialize_llvm(lib)
        mark_llvm_initialized()
    end
    return
end

@locked function emit_llvm(@nospecialize(job::CompilerJob);
                           resolve_relocations::Bool=true)
    ccall(:jl_generating_output, Cint, ()) == 1 && initialize_llvm_for_output()

    @tracepoint "IR generation" begin
        ir, compiled, relocations = irgen(job)
        if job.config.entry_abi === :specfunc
            entry_fn = compiled[job.source].specfunc
        else
            entry_fn = compiled[job.source].func
        end
        entry = ir.functions[entry_fn]
    end

    # finalize the current module. this needs to happen before linking deferred modules,
    # since those modules have been finalized themselves, and we don't want to re-finalize.
    entry = finish_module!(job, ir, entry)

    # deferred code generation
    has_deferred_jobs = job.config.toplevel && !job.config.only_entry &&
                        haskey(ir.functions, "deferred_codegen")
    jobs = Dict{CompilerJob, String}(job => entry_fn)
    nested_jobs = Dict{CompilerJob, String}()
    if has_deferred_jobs
        dyn_marker = ir.functions["deferred_codegen"]

        # iterative compilation (non-recursive)
        changed = true
        while changed
            changed = false

            # find deferred compiler
            # TODO: recover this information earlier, from the Julia IR
            worklist = Dict{CompilerJob, Vector{LLVM.CallInst}}()
            for use in dyn_marker.uses
                # decode the call
                call = use.user::LLVM.CallInst
                id = convert(Int, first(call.operands))

                dyn_job = deferred_job(deferred_spec(id), job)
                push!(get!(worklist, dyn_job, LLVM.CallInst[]), call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation: of the same job (e.g. a kernel requesting itself), or,
                # for functions that aren't kernels, of one that is the same once nested in
                # this compilation (e.g. a function that both a job and its deferred child
                # request). kernels are entry points, which `finish_ir!` processes with the
                # requested job, so they are compiled for each one (which can make kernels
                # requested from different levels of nesting get the same name).
                dyn_entry_fn = get(jobs, dyn_job, nothing)
                if dyn_entry_fn === nothing
                    # We are potentially calling into a sibling compiler (Enzyme)
                    # which was loaded in a newer world.
                    nested_job = @invokelatest nest_deferred_job(dyn_job, job)
                    if !dyn_job.config.kernel
                        dyn_entry_fn = get(nested_jobs, nested_job, nothing)
                        dyn_entry_fn === nothing || (jobs[dyn_job] = dyn_entry_fn)
                    end
                end
                if dyn_entry_fn === nothing
                    dyn_ir, dyn_meta = @invokelatest compile_unhooked(:llvm, nested_job)
                    dyn_entry_fn = dyn_meta.entry.name
                    # the sibling compiler's result isn't inferred
                    dyn_compiled = dyn_meta.compiled
                    if dyn_compiled isa Dict{Any,Any}
                        merge!(compiled, dyn_compiled)
                    else
                        Base.inferencebarrier(merge!)(compiled, dyn_compiled)
                    end
                    @assert dyn_ir.context == ir.context
                    link_relocatable!(ir, relocations, dyn_ir,
                                      dyn_meta.relocations)
                    jobs[dyn_job] = dyn_entry_fn
                    dyn_job.config.kernel || (nested_jobs[nested_job] = dyn_entry_fn)
                    changed = true
                end
                dyn_entry = ir.functions[dyn_entry_fn]

                # insert a pointer to the function everywhere the entry is used
                T_ptr = convert(LLVMType, Ptr{Cvoid})
                for call in worklist[dyn_job]
                    @dispose builder=IRBuilder() begin
                        position!(builder, LLVM.before(call))
                        fptr = if LLVM.version() >= v"17"
                            T_ptr = LLVM.PointerType()
                            bitcast!(builder, dyn_entry, T_ptr)
                        elseif VERSION >= v"1.12.0-DEV.225"
                            T_ptr = LLVM.PointerType(LLVM.Int8Type())
                            bitcast!(builder, dyn_entry, T_ptr)
                        else
                            ptrtoint!(builder, dyn_entry, T_ptr)
                        end
                        replace_uses!(call, fptr)
                    end
                    erase!(call)
                end
            end

            # minimal optimization to convert the inttoptr/call into a direct call
            @dispose pb=PassBuilder() begin
                add!(pb, FunctionPassManager()) do fpm
                    add!(fpm, instcombine_pass(job))
                end
                with_llvm_machine(job.config.target) do tm
                    run!(pb, ir, tm)
                end
            end
        end

        # all deferred compilations should have been resolved
        @compiler_assert isempty(dyn_marker.uses) job
        erase!(dyn_marker)
    end

    # only deferred jobs that are kernels themselves (e.g. child kernels for dynamic
    # parallelism) are entrypoints in their own right. other deferred functions (e.g. the
    # wrappers Enzyme generates) are only called from within this module, so they should
    # be internalized like any other function and dropped once inlined, rather than kept
    # around for back-ends that cannot express their signatures (e.g. SPIR-V).
    entrypoints = filter(((job′, _),) -> job′ === job || job′.config.kernel, jobs)

    if job.config.toplevel && job.config.libraries
        # load the runtime outside of a timing block (because it recurses into the compiler)
        if !uses_julia_runtime(job)
            runtime, runtime_relocs = load_runtime(job)
        end

        @tracepoint "Library linking" begin
            # target-specific libraries
            @tracepoint "target libraries" link_libraries!(job, ir)

            # GPU run-time library
            if !uses_julia_runtime(job)
                @tracepoint "runtime library" begin
                    declare_late_runtime_functions!(job, ir)
                    link_relocatable!(ir, relocations, runtime, runtime_relocs;
                                      only_needed=true)
                    # the linked runtime may itself allocate. import what that requires
                    # before internalization, so that linking again reuses the rest.
                    if declare_late_runtime_functions!(job, ir)
                        runtime, runtime_relocs = load_runtime(job)
                        link_relocatable!(ir, relocations, runtime, runtime_relocs;
                                          only_needed=true)
                    end
                end
            end
        end
    end

    exception_sites = ExceptionSite[]
    @tracepoint "IR post-processing" begin
        # mark the kernel entry-point functions (optimization may need it)
        if job.config.kernel
            mark_kernel!(entry)

            # device code cannot call a kernel, so it cannot recurse (clang does the same
            # for CUDA and HIP kernels). this lets `ReversePostOrderFunctionAttrsPass` mark
            # the functions it calls, which the AMDGPU back-end otherwise assumes may
            # recurse, giving the kernel a dynamic stack. on the CPU, a kernel is an
            # ordinary function that could be called back from code outside the module.
            if !(job.config.target isa NativeCompilerTarget) && isempty(entry.uses)
                push!(entry.function_attributes, EnumAttribute("norecurse", 0))
            end
        end

        if job.config.toplevel
            # mark everything internal except for entrypoints and any exported
            # global variables. this makes sure that the optimizer can, e.g.,
            # rewrite function signatures.
            preserved_gvs = collect(values(entrypoints))
            for gvar in ir.globals
                if gvar.linkage == LLVM.Linkage.External
                    push!(preserved_gvs, gvar.name)
                end
            end
            # until `GPUInternalizeRuntimePass` (see `late_runtime_functions`)
            for rt in late_runtime_functions(job, ir)
                push!(preserved_gvs, rt.llvm_name)
            end
            with_llvm_machine(job.config.target) do tm
                run!(InternalizePass(; preserved_gvs), ir, tm)
            end

            finish_linked_module!(job, ir)
            lower_syncscopes!(job, ir)

            # Resolve early so optimization sees concrete values.
            resolve_early = resolve_relocations && relocation_lowering(job) === :bake
            if resolve_early
                prune_dead_relocations!(ir, relocations)
                bake_relocations!(ir, relocations)
            end

            if job.config.optimize
                @tracepoint "optimization" begin
                    optimize!(job, ir, relocations; job.config.opt_level)

                    # deferred codegen has some special optimization requirements,
                    # which also need to happen _after_ regular optimization.
                    # XXX: make these part of the optimizer pipeline?
                    if has_deferred_jobs
                        @dispose pb=PassBuilder() begin
                            add!(pb, FunctionPassManager()) do fpm
                                add!(fpm, instcombine_pass(job))
                            end
                            add!(pb, AlwaysInlinerPass())
                            add!(pb, FunctionPassManager()) do fpm
                                add!(fpm, SROAPass())
                                add!(fpm, GVNPass())
                            end
                            add!(pb, MergeFunctionsPass())
                            with_llvm_machine(job.config.target) do tm
                                run!(pb, ir, tm)
                            end
                        end
                    end
                end
            end

            if job.config.cleanup
                @tracepoint "clean-up" begin
                    @dispose pb=PassBuilder() begin
                        add!(pb, RecomputeGlobalsAAPass())
                        add!(pb, GlobalOptPass())
                        add!(pb, GlobalDCEPass())
                        add!(pb, StripDeadPrototypesPass())
                        add!(pb, ConstantMergePass())
                        with_llvm_machine(job.config.target) do tm
                            run!(pb, ir, tm)
                        end
                    end
                end
            end

            # Do not expose dead sites to loaders.
            resolve_early || prune_dead_relocations!(ir, relocations)

            # optimization may have replaced functions, so look the entry point up again
            entry = ir.functions[entry_fn]

            # finish the module
            #
            # we want to finish the module after optimization, so we cannot do so
            # during deferred code generation. instead, process the deferred jobs
            # here.
            entry = finish_ir!(job, ir, entry)
            for (job′, fn′) in entrypoints
                job′ === job && continue
                finish_ir!(job′, ir, ir.functions[fn′])
            end

            # back-ends that inline throwing code into kernels did this during `finish_ir!`
            if uses_exception_sites(job)
                finalize_exception_sites!(job, ir)
                merge_exception_sites!(job, ir)
                exception_sites = take_exception_sites!(ir)
            end
        end

        # replace non-entry function definitions with a declaration
        # NOTE: we can't do this before optimization, because the definitions of called
        #       functions may affect optimization.
        if job.config.only_entry
            for f in ir.functions
                f == entry && continue
                isdeclaration(f) && continue
                LLVM.isintrinsic(f) && continue
                empty!(f)
            end
        end
    end

    if job.config.toplevel && job.config.validate
        @tracepoint "validation" begin
            check_ir(job, ir, relocations; compiled)
        end
    end

    # Collect cglobal loads as relocation records now (after validation, which wants to see
    # the original loads), so the `:llvm`-level relocation metadata is complete for loaders
    # and for consumers that apply it themselves. Not under `:bake`, whose contract is a
    # fully-resolved `:llvm` result; its loads are collected and resolved in
    # `prepare_execution!`, which remains an idempotent safety net for the other strategies
    # (and for direct `emit_asm` callers).
    if job.config.toplevel && relocation_lowering(job) !== :bake
        @tracepoint "cglobal relocations" begin
            collect_cglobal_relocations!(job, ir, relocations)
            prune_dead_relocations!(ir, relocations)
        end
    end

    if should_verify()
        @tracepoint "verification" verify(ir)
    end

    # XXX: before Julia 1.13.5, the precompiled keyword wrapper of `emit_llvm` can keep
    #      the return type inferred while `uses_exception_sites` only had its fallback, so
    #      callers drop `exception_sites` after a back-end defines its own method. hiding
    #      the call from inference makes the return type cover both shapes of the metadata.
    #      fixed by JuliaLang/julia#64702.
    @static if VERSION < v"1.13.5"
        with_sites = job.config.toplevel &&
                     Base.inferencebarrier(uses_exception_sites(job))::Bool
    else
        with_sites = job.config.toplevel && uses_exception_sites(job)
    end
    if with_sites
        return ir, (; entry, compiled, relocations, exception_sites)
    end
    return ir, (; entry, compiled, relocations)
end

# Compatibility for back-ends that resolve relocations during `emit_llvm`.
emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module,
         format::LLVM.CodeGenFileType.T) =
    emit_asm(job, ir, Relocations(), format)

@locked function emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module,
                          relocs::Relocations, format::LLVM.CodeGenFileType.T)
    # NOTE: strip after validation to get better errors
    if job.config.strip
        @tracepoint "Debug info removal" strip_debuginfo!(ir)
    end

    @tracepoint "LLVM back-end" begin
        @tracepoint "preparation" prepare_execution!(job, ir, relocs)

        code = @tracepoint "machine-code generation" mcgen(job, ir, format)
    end

    return code, (;)
end
