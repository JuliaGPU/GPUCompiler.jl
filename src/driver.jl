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
    if VERSION >= v"1.9.0-DEV.516"
        # Julia 1.9 knows how to deal with arbitrary contexts,
        # and uses ORC's thread safe versions.
        ctx = ThreadSafeContext(; opaque_pointers)
    elseif VERSION >= v"1.9.0-DEV.115"
        # Julia 1.9 knows how to deal with arbitrary contexts
        ctx = Context(; opaque_pointers)
    else
        # earlier versions of Julia claim so, but actually use a global context
        isboxed_ref = Ref{Bool}()
        typ = LLVMType(ccall(:jl_type_to_llvm, LLVM.API.LLVMTypeRef,
                       (Any, Ptr{Bool}), Any, isboxed_ref))
        ctx = context(typ)
        if opaque_pointers !== nothing && supports_typed_pointers(ctx) !== !opaque_pointers
            error("Cannot use $(opaque_pointers ? "opaque" : "typed") pointers, as the context has already been configured to use $(supports_typed_pointers(ctx) ? "typed" : "opaque") pointers, and this version of Julia does not support changing that.")
        end
    end

    ctx
end
function JuliaContext(f; kwargs...)
    if VERSION >= v"1.9.0-DEV.516"
        ts_ctx = JuliaContext(; kwargs...)
        # for now, also activate the underlying context
        # XXX: this is wrong; we can't expose the underlying LLVM context, but should
        #      instead always go through the callback in order to unlock it properly.
        #      rework this once we depend on Julia 1.9 or later.
        ctx = context(ts_ctx)
        activate(ctx)
        try
            f(ctx)
        finally
            deactivate(ctx)
            dispose(ts_ctx)
        end
    elseif VERSION >= v"1.9.0-DEV.115"
        Context(f)
    else
        ctx = JuliaContext()
        activate(ctx)
        try
            f(ctx)
        finally
            deactivate(ctx)
            # we cannot dispose of the global unique context
        end
    end
end


## compiler entrypoint

export compile

# NOTE: the keyword arguments to compile/codegen control those aspects of compilation that
#       might have to be changed (e.g. set libraries=false when recursing, or set
#       strip=true for reflection). What remains defines the compilation job itself,
#       and those values are contained in the CompilerJob struct.

# (::CompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(target::Symbol, job::CompilerJob;
            libraries=true, optimize=true, strip=false, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `target` argument: `:julia` for Julia IR, `:llvm` for
LLVM IR and `:asm` for machine code.

The following keyword arguments are supported:
- `libraries`: link the GPU runtime and `libdevice` libraries (if required)
- `optimize`: optimize the code (default: true)
- `cleanup`: run cleanup passes on the code (default: true)
- `strip`: strip non-functional metadata and debug information (default: false)
- `validate`: enable optional validation of input and outputs (default: true)
- `only_entry`: only keep the entry function, remove all others (default: false).
  This option is only for internal use, to implement reflection's `dump_module`.

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
function compile(target::Symbol, @nospecialize(job::CompilerJob);
                 libraries::Bool=true, toplevel::Bool=true,
                 optimize::Bool=true, cleanup::Bool=true, strip::Bool=false,
                 validate::Bool=true, only_entry::Bool=false)
    if compile_hook[] !== nothing
        compile_hook[](job)
    end

    return codegen(target, job;
                   libraries, toplevel, optimize, cleanup, strip, validate, only_entry)
end

function codegen(output::Symbol, @nospecialize(job::CompilerJob);
                 libraries::Bool=true, toplevel::Bool=true, optimize::Bool=true,
                 cleanup::Bool=true, strip::Bool=false, validate::Bool=true,
                 only_entry::Bool=false, parent_job::Union{Nothing, CompilerJob}=nothing)
    if context(; throw_error=false) === nothing
        error("No active LLVM context. Use `JuliaContext()` do-block syntax to create one.")
    elseif VERSION < v"1.9.0-DEV.115" && context() != JuliaContext()
        error("""Julia <1.9 does not suppport generating code in an arbitrary LLVM context.
                 Use `JuliaContext()` do-block syntax to get an appropriate one.""")
    end

    @timeit_debug to "Validation" begin
        check_method(job)   # not optional
        validate && check_invocation(job)
    end


    ## LLVM IR

    ir, ir_meta = emit_llvm(job; libraries, toplevel, optimize, cleanup, only_entry, validate)

    if output == :llvm
        if strip
            @timeit_debug to "strip debug info" strip_debuginfo!(ir)
        end

        return ir, ir_meta
    end


    ## machine code

    format = if output == :asm
        LLVM.API.LLVMAssemblyFile
    elseif output == :obj
        LLVM.API.LLVMObjectFile
    else
        error("Unknown assembly format $output")
    end
    asm, asm_meta = emit_asm(job, ir; strip, validate, format)

    if output == :asm || output == :obj
        return asm, (; asm_meta..., ir_meta..., ir)
    end


    error("Unknown compilation output $output")
end

# primitive mechanism for deferred compilation, for implementing CUDA dynamic parallelism.
# this could both be generalized (e.g. supporting actual function calls, instead of
# returning a function pointer), and be integrated with the nonrecursive codegen.
const deferred_codegen_jobs = Dict{Int, Any}()

# We make this function explicitly callable so that we can drive OrcJIT's
# lazy compilation from, while also enabling recursive compilation.
Base.@ccallable Ptr{Cvoid} function deferred_codegen(ptr::Ptr{Cvoid})
    ptr
end

@generated function deferred_codegen(::Val{ft}, ::Val{tt}) where {ft,tt}
    id = length(deferred_codegen_jobs) + 1
    deferred_codegen_jobs[id] = (; ft, tt)
    # don't bother looking up the method instance, as we'll do so again during codegen
    # using the world age of the parent.
    #
    # this also works around an issue on <1.10, where we don't know the world age of
    # generated functions so use the current world counter, which may be too new
    # for the world we're compiling for.

    pseudo_ptr = reinterpret(Ptr{Cvoid}, id)
    quote
        # TODO: add an edge to this method instance to support method redefinitions
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $pseudo_ptr)
    end
end

const __llvm_initialized = Ref(false)

@locked function emit_llvm(@nospecialize(job::CompilerJob);
                           libraries::Bool=true, toplevel::Bool=true, optimize::Bool=true,
                           cleanup::Bool=true, only_entry::Bool=false, validate::Bool=true)
    if !__llvm_initialized[]
        InitializeAllTargets()
        InitializeAllTargetInfos()
        InitializeAllAsmPrinters()
        InitializeAllAsmParsers()
        InitializeAllTargetMCs()
        __llvm_initialized[] = true
    end

    @timeit_debug to "IR generation" begin
        ir, compiled = irgen(job)
        if job.config.entry_abi === :specfunc
            entry_fn = compiled[job.source].specfunc
        else
            entry_fn = compiled[job.source].func
        end
        entry = functions(ir)[entry_fn]
    end

    # finalize the current module. this needs to happen before linking deferred modules,
    # since those modules have been finalized themselves, and we don't want to re-finalize.
    entry = finish_module!(job, ir, entry)

    # deferred code generation
    has_deferred_jobs = !only_entry &&
                        haskey(functions(ir), "deferred_codegen")
    jobs = Dict{CompilerJob, String}(job => entry_fn)
    if has_deferred_jobs
        dyn_marker = functions(ir)["deferred_codegen"]

        # iterative compilation (non-recursive)
        changed = true
        while changed
            changed = false

            # find deferred compiler
            # TODO: recover this information earlier, from the Julia IR
            worklist = Dict{CompilerJob, Vector{LLVM.CallInst}}()
            for use in uses(dyn_marker)
                # decode the call
                call = user(use)::LLVM.CallInst
                id = convert(Int, first(operands(call)))

                global deferred_codegen_jobs
                dyn_val = deferred_codegen_jobs[id]

                # get a job in the appopriate world
                dyn_job = if dyn_val isa CompilerJob
                    # trust that the user knows what they're doing
                    dyn_val
                else
                    ft, tt = dyn_val
                    dyn_src = methodinstance(ft, tt, tls_world_age())
                    CompilerJob(dyn_src, job.config)
                end

                push!(get!(worklist, dyn_job, LLVM.CallInst[]), call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_entry_fn = get!(jobs, dyn_job) do
                    dyn_ir, dyn_meta = codegen(:llvm, dyn_job; validate=false,
                                               optimize=false,
                                               toplevel=false,
                                               parent_job=job)
                    dyn_entry_fn = LLVM.name(dyn_meta.entry)
                    merge!(compiled, dyn_meta.compiled)
                    @assert context(dyn_ir) == context(ir)
                    link!(ir, dyn_ir)
                    changed = true
                    dyn_entry_fn
                end
                dyn_entry = functions(ir)[dyn_entry_fn]

                # insert a pointer to the function everywhere the entry is used
                T_ptr = convert(LLVMType, Ptr{Cvoid})
                for call in worklist[dyn_job]
                    @dispose builder=IRBuilder() begin
                        position!(builder, call)
                        fptr = ptrtoint!(builder, dyn_entry, T_ptr)
                        replace_uses!(call, fptr)
                    end
                    unsafe_delete!(LLVM.parent(call), call)
                end
            end
        end

        # all deferred compilations should have been resolved
        @compiler_assert isempty(uses(dyn_marker)) job
        unsafe_delete!(ir, dyn_marker)
    end

    if toplevel
        # always preload the runtime, and do so early; it cannot be part of any
        # timing block because it recurses into the compiler
        if !uses_julia_runtime(job) && libraries
            runtime = load_runtime(job)
            runtime_fns = LLVM.name.(defs(runtime))
            runtime_intrinsics = ["julia.gc_alloc_obj"]
        end

        @timeit_debug to "Library linking" begin
            if libraries
                # target-specific libraries
                undefined_fns = LLVM.name.(decls(ir))
                @timeit_debug to "target libraries" link_libraries!(job, ir, undefined_fns)

                # GPU run-time library
                if !uses_julia_runtime(job) && any(fn -> fn in runtime_fns ||
                                                         fn in runtime_intrinsics,
                                                   undefined_fns)
                    @timeit_debug to "runtime library" link_library!(ir, runtime)
                end
            end
        end
    end

    @timeit_debug to "IR post-processing" begin
        # mark everything internal except for entrypoints and any exported
        # global variables. this makes sure that the optimizer can, e.g.,
        # rewrite function signatures.
        if toplevel
            # TODO: there's no good API to use internalize with the new pass manager yet
            @dispose pm=ModulePassManager() begin
                exports = collect(values(jobs))
                for gvar in globals(ir)
                    if linkage(gvar) == LLVM.API.LLVMExternalLinkage
                        push!(exports, LLVM.name(gvar))
                    end
                end
                internalize!(pm, exports)
                run!(pm, ir)
            end
        end

        # mark the kernel entry-point functions (optimization may need it)
        if job.config.kernel
            push!(metadata(ir)["julia.kernel"], MDNode([entry]))

            # IDEA: save all jobs, not only kernels, and save other attributes
            #       so that we can reconstruct the CompileJob instead of setting it globally
        end

        if optimize
            @timeit_debug to "optimization" begin
                optimize!(job, ir)

                # deferred codegen has some special optimization requirements,
                # which also need to happen _after_ regular optimization.
                # XXX: make these part of the optimizer pipeline?
                if has_deferred_jobs
                    if use_newpm
                        @dispose pb=PassBuilder() mpm=NewPMModulePassManager(pb) begin
                            add!(mpm, NewPMFunctionPassManager) do fpm
                                add!(fpm, InstCombinePass())
                            end
                            add!(mpm, AlwaysInlinerPass())
                            add!(mpm, NewPMFunctionPassManager) do fpm
                                add!(fpm, SROAPass())
                                add!(fpm, GVNPass())
                            end
                            add!(mpm, MergeFunctionsPass())
                            run!(mpm, ir)
                        end
                    else
                        @dispose pm=ModulePassManager() begin
                            # inline and optimize the call to e deferred code. in particular we want
                            # to remove unnecessary alloca's created by pass-by-ref semantics.
                            instruction_combining!(pm)
                            always_inliner!(pm)
                            scalar_repl_aggregates_ssa!(pm)
                            promote_memory_to_register!(pm)
                            gvn!(pm)

                            # merge duplicate functions, since each compilation invocation emits everything
                            # XXX: ideally we want to avoid emitting these in the first place
                            merge_functions!(pm)

                            run!(pm, ir)
                        end
                    end
                end
            end

            # optimization may have replaced functions, so look the entry point up again
            entry = functions(ir)[entry_fn]
        end

        if cleanup
            @timeit_debug to "clean-up" begin
                if use_newpm
                    @dispose pb=PassBuilder() mpm=NewPMModulePassManager(pb) begin
                        add!(mpm, RecomputeGlobalsAAPass())
                        add!(mpm, GlobalOptPass())
                        add!(mpm, GlobalDCEPass())
                        add!(mpm, StripDeadPrototypesPass())
                        add!(mpm, ConstantMergePass())
                        run!(mpm, ir)
                    end
                else
                    # we can only clean-up now, as optimization may lower or introduce calls to
                    # functions from the GPU runtime (e.g. julia.gc_alloc_obj -> gpu_gc_pool_alloc)
                    @dispose pm=ModulePassManager() begin
                        # eliminate all unused internal functions
                        global_optimizer!(pm)
                        global_dce!(pm)
                        strip_dead_prototypes!(pm)

                        # merge constants (such as exception messages)
                        constant_merge!(pm)

                        run!(pm, ir)
                    end
                end
            end
        end

        # finish the module
        #
        # we want to finish the module after optimization, so we cannot do so
        # during deferred code generation. instead, process the deferred jobs
        # here.
        if toplevel
            entry = finish_ir!(job, ir, entry)

            for (job′, fn′) in jobs
                job′ == job && continue
                finish_ir!(job′, ir, functions(ir)[fn′])
            end
        end

        # replace non-entry function definitions with a declaration
        # NOTE: we can't do this before optimization, because the definitions of called
        #       functions may affect optimization.
        if only_entry
            for f in functions(ir)
                f == entry && continue
                isdeclaration(f) && continue
                LLVM.isintrinsic(f) && continue
                empty!(f)
            end
        end
    end

    if validate
        @timeit_debug to "Validation" begin
            check_ir(job, ir)
        end
    end

    if should_verify()
        @timeit_debug to "verification" verify(ir)
    end

    return ir, (; entry, compiled)
end

@locked function emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module;
                          strip::Bool=false, validate::Bool=true, format::LLVM.API.LLVMCodeGenFileType)
    # NOTE: strip after validation to get better errors
    if strip
        @timeit_debug to "Debug info removal" strip_debuginfo!(ir)
    end

    @timeit_debug to "LLVM back-end" begin
        @timeit_debug to "preparation" prepare_execution!(job, ir)

        code = @timeit_debug to "machine-code generation" mcgen(job, ir, format)
    end

    return code, ()
end
