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
end


## deferred compilation

"""
    var"gpuc.deferred"(meta, f, args...)::Ptr{Cvoid}

As if we were to call `f(args...)` but instead we are
putting down a marker and return a function pointer to later
call.
"""
function var"gpuc.deferred" end

## compiler entrypoint

export compile

# NOTE: the keyword arguments to compile/codegen control those aspects of compilation that
#       might have to be changed (e.g. set libraries=false when recursing, or set
#       strip=true for reflection). What remains defines the compilation job itself,
#       and those values are contained in the CompilerJob struct.

# (::CompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(target::Symbol, job::CompilerJob; kwargs...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `target` argument: `:julia` for Julia IR, `:llvm` for
LLVM IR and `:asm` for machine code.

The following keyword arguments are supported:
- `toplevel`: indicates that this compilation is the outermost invocation of the compiler
  (default: true)
- `libraries`: link the GPU runtime and `libdevice` libraries (default: true, if toplevel)
- `optimize`: optimize the code (default: true, if toplevel)
- `cleanup`: run cleanup passes on the code (default: true, if toplevel)
- `validate`: enable optional validation of input and outputs (default: true, if toplevel)
- `strip`: strip non-functional metadata and debug information (default: false)
- `only_entry`: only keep the entry function, remove all others (default: false).
  This option is only for internal use, to implement reflection's `dump_module`.

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
function compile(target::Symbol, @nospecialize(job::CompilerJob); kwargs...)
    if compile_hook[] !== nothing
        compile_hook[](job)
    end

    return codegen(target, job; kwargs...)
end

function codegen(output::Symbol, @nospecialize(job::CompilerJob); toplevel::Bool=true,
                 libraries::Bool=toplevel, optimize::Bool=toplevel, cleanup::Bool=toplevel,
                 validate::Bool=toplevel, strip::Bool=false, only_entry::Bool=false,
                 parent_job::Union{Nothing, CompilerJob}=nothing)
    if context(; throw_error=false) === nothing
        error("No active LLVM context. Use `JuliaContext()` do-block syntax to create one.")
    end

    @timeit_debug to "Validation" begin
        check_method(job)   # not optional
        validate && check_invocation(job)
    end

    prepare_job!(job)


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

const __llvm_initialized = Ref(false)

@locked function emit_llvm(@nospecialize(job::CompilerJob); toplevel::Bool,
                           libraries::Bool, optimize::Bool, cleanup::Bool,
                           validate::Bool, only_entry::Bool)
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
        edge = Edge(inference_metadata(job), job.source)
        if job.config.entry_abi === :specfunc
            entry_fn = compiled[edge].specfunc
        else
            entry_fn = compiled[edge].func
        end
        entry = functions(ir)[entry_fn]
    end

    # finalize the current module.
    entry = finish_module!(job, ir, entry)

    # rewrite "gpuc.lookup" for deferred code generation
    run_optimization_for_deferred = false
    if haskey(functions(ir), "gpuc.lookup")
        run_optimization_for_deferred = true
        dyn_marker = functions(ir)["gpuc.lookup"]

        # gpuc.deferred is lowered to a gpuc.lookup foreigncall, so we need to extract the
        # target method instance from the LLVM IR
        function find_base_object(val)
            while true
                if val isa ConstantExpr && (opcode(val) == LLVM.API.LLVMIntToPtr ||
                                            opcode(val) == LLVM.API.LLVMBitCast ||
                                            opcode(val) == LLVM.API.LLVMAddrSpaceCast)
                    val = first(operands(val))
                elseif val isa LLVM.IntToPtrInst ||
                       val isa LLVM.BitCastInst ||
                       val isa LLVM.AddrSpaceCastInst
                    val = first(operands(val))
                elseif val isa LLVM.LoadInst
                    # In 1.11+ we no longer embed integer constants directly.
                    gv = first(operands(val))
                    if gv isa LLVM.GlobalValue
                        val = LLVM.initializer(gv)
                        continue
                    end
                    break
                else
                    break
                end
            end
            return val
        end

        worklist = Dict{Edge, Vector{LLVM.CallInst}}()
        for use in uses(dyn_marker)
            # decode the call
            call = user(use)::LLVM.CallInst
            dyn_meta_inst = find_base_object(operands(call)[1])
            @compiler_assert isa(dyn_meta_inst, LLVM.ConstantInt) job
            dyn_mi_inst = find_base_object(operands(call)[2])
            @compiler_assert isa(dyn_mi_inst, LLVM.ConstantInt) job
            dyn_meta = Base.unsafe_pointer_to_objref(
                convert(Ptr{Cvoid}, convert(Int, dyn_meta_inst)))
            dyn_mi = Base.unsafe_pointer_to_objref(
                convert(Ptr{Cvoid}, convert(Int, dyn_mi_inst)))::MethodInstance
            push!(get!(worklist, Edge(dyn_meta, dyn_mi), LLVM.CallInst[]), call)
        end

        for dyn_edge in keys(worklist)
            dyn_fn_name = compiled[dyn_edge].specfunc
            dyn_fn = functions(ir)[dyn_fn_name]

            # insert a pointer to the function everywhere the entry is used
            T_ptr = convert(LLVMType, Ptr{Cvoid})
            for call in worklist[dyn_edge]
                @dispose builder=IRBuilder() begin
                    position!(builder, call)
                    fptr = if LLVM.version() >= v"17"
                        T_ptr = LLVM.PointerType()
                        bitcast!(builder, dyn_fn, T_ptr)
                    elseif VERSION >= v"1.12.0-DEV.225"
                        T_ptr = LLVM.PointerType(LLVM.Int8Type())
                        bitcast!(builder, dyn_fn, T_ptr)
                    else
                        ptrtoint!(builder, dyn_fn, T_ptr)
                    end
                    replace_uses!(call, fptr)
                end
                erase!( call)
            end
        end

        # all deferred compilations should have been resolved
        @compiler_assert isempty(uses(dyn_marker)) job
        erase!(dyn_marker)
    end

    if libraries
        # load the runtime outside of a timing block (because it recurses into the compiler)
        if !uses_julia_runtime(job)
            runtime = load_runtime(job)
            runtime_fns = LLVM.name.(defs(runtime))
            runtime_intrinsics = ["julia.gc_alloc_obj"]
        end

        @timeit_debug to "Library linking" begin
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

    for (name, plugin) in PLUGINS
        if plugin.finalize_module !== nothing
            plugin.finalize_module(job, compiled, ir)
        end
    end

    @timeit_debug to "IR post-processing" begin
        # mark everything internal except for entrypoints and any exported
        # global variables. this makes sure that the optimizer can, e.g.,
        # rewrite function signatures.
        if toplevel
            preserved_gvs = [entry_fn]
            for gvar in globals(ir)
                if linkage(gvar) == LLVM.API.LLVMExternalLinkage
                    push!(preserved_gvs, LLVM.name(gvar))
                end
            end
            if LLVM.version() >= v"17"
                run!(InternalizePass(; preserved_gvs), ir,
                     llvm_machine(job.config.target))
            else
                @dispose pm=ModulePassManager() begin
                    internalize!(pm, preserved_gvs)
                    run!(pm, ir)
                end
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
                optimize!(job, ir; job.config.opt_level)

                # deferred codegen has some special optimization requirements,
                # which also need to happen _after_ regular optimization.
                # XXX: make these part of the optimizer pipeline?
                if run_optimization_for_deferred
                    @dispose pb=NewPMPassBuilder() begin
                        add!(pb, NewPMFunctionPassManager()) do fpm
                            add!(fpm, InstCombinePass())
                        end
                        add!(pb, AlwaysInlinerPass())
                        add!(pb, NewPMFunctionPassManager()) do fpm
                            add!(fpm, SROAPass())
                            add!(fpm, GVNPass())
                        end
                        add!(pb, MergeFunctionsPass())
                        run!(pb, ir, llvm_machine(job.config.target))
                    end
                end
            end

            # optimization may have replaced functions, so look the entry point up again
            entry = functions(ir)[entry_fn]
        end

        if cleanup
            @timeit_debug to "clean-up" begin
                @dispose pb=NewPMPassBuilder() begin
                    add!(pb, RecomputeGlobalsAAPass())
                    add!(pb, GlobalOptPass())
                    add!(pb, GlobalDCEPass())
                    add!(pb, StripDeadPrototypesPass())
                    add!(pb, ConstantMergePass())
                    run!(pb, ir, llvm_machine(job.config.target))
                end
            end
        end

        # finish the module
        #
        # we want to finish the module after optimization, so we cannot do so
        # during deferred code generation. Instead, process the merged module
        # from all the jobs here.
        if toplevel # TODO: We should be able to remove this now
            entry = finish_ir!(job, ir, entry)

            # for (job′, fn′) in jobs
            #     job′ == job && continue
            #     finish_ir!(job′, ir, functions(ir)[fn′])
            # end
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
                          strip::Bool, validate::Bool, format::LLVM.API.LLVMCodeGenFileType)
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
