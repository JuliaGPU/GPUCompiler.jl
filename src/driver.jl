# compiler driver and main interface

# NOTE: the keyword arguments to compile/codegen control those aspects of compilation that
#       might have to be changed (e.g. set libraries=false when recursing, or set
#       strip=true for reflection). What remains defines the compilation job itself,
#       and those values are contained in the CompilerJob struct.

# (::CompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(target::Symbol, job::CompilerJob;
            libraries=true, deferred_codegen=true,
            optimize=true, strip=false, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `target` argument: `:julia` for Julia IR, `:llvm` for
LLVM IR and `:asm` for machine code.

The following keyword arguments are supported:
- `libraries`: link the GPU runtime and `libdevice` libraries (if required)
- `deferred_codegen`: resolve deferred compiler invocations (if required)
- `optimize`: optimize the code (default: true)
- `strip`: strip non-functional metadata and debug information (default: false)
- `validate`: validate the generated IR before emitting machine code (default: true)
- `only_entry`: only keep the entry function, remove all others (default: false).
  This option is only for internal use, to implement reflection's `dump_module`.

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
function compile(target::Symbol, @nospecialize(job::CompilerJob);
                 libraries::Bool=true, deferred_codegen::Bool=true,
                 optimize::Bool=true, strip::Bool=false, validate::Bool=true,
                 only_entry::Bool=false)
    if compile_hook[] !== nothing
        compile_hook[](job)
    end

    return codegen(target, job;
                   libraries, deferred_codegen, optimize, strip, validate, only_entry)
end

function codegen(output::Symbol, @nospecialize(job::CompilerJob);
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false,
                 parent_job::Union{Nothing, CompilerJob} = nothing)
    ## Julia IR

    mi, mi_meta = emit_julia(job)

    if output == :julia
        return mi, mi_meta
    end


    ## LLVM IR

    ir, ir_meta = emit_llvm(job, mi; libraries, deferred_codegen, optimize, only_entry)

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
        return asm, asm_meta
    end


    error("Unknown compilation output $output")
end

@locked function emit_julia(@nospecialize(job::CompilerJob))
    @timeit_debug to "validation" check_method(job)

    @timeit_debug to "Julia front-end" begin

        # get the method instance
        meth = which(job.source.f, job.source.tt)
        sig = Base.signature_type(job.source.f, job.source.tt)::Type
        (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                          (Any, Any), sig, meth.sig)::Core.SimpleVector
        meth = Base.func_for_method_checked(meth, ti, env)
        method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                      (Any, Any, Any, UInt), meth, ti, env, job.source.world)

        for var in env
            if var isa TypeVar
                throw(KernelError(job, "method captures a typevar (you probably use an unbound type variable)"))
            end
        end
    end

    return method_instance, ()
end

# primitive mechanism for deferred compilation, for implementing CUDA dynamic parallelism.
# this could both be generalized (e.g. supporting actual function calls, instead of
# returning a function pointer), and be integrated with the nonrecursive codegen.
const deferred_codegen_jobs = Dict{Int, Union{FunctionSpec, CompilerJob}}()

# We make this function explicitly callable so that we can drive OrcJIT's
# lazy compilation from, while also enabling recursive compilation.
Base.@ccallable Ptr{Cvoid} function deferred_codegen(ptr::Ptr{Cvoid})
    ptr
end

@generated function deferred_codegen(::Val{f}, ::Val{tt}) where {f,tt}
    id = length(deferred_codegen_jobs) + 1
    deferred_codegen_jobs[id] = FunctionSpec(f,tt)

    pseudo_ptr = reinterpret(Ptr{Cvoid}, id)
    quote
        # TODO: add an edge to this method instance to support method redefinitions
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $pseudo_ptr)
    end
end

const __llvm_initialized = Ref(false)

@locked function emit_llvm(@nospecialize(job::CompilerJob), @nospecialize(method_instance);
                           libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                           only_entry::Bool=false)
    if !__llvm_initialized[]
        InitializeAllTargets()
        InitializeAllTargetInfos()
        InitializeAllAsmPrinters()
        InitializeAllAsmParsers()
        InitializeAllTargetMCs()
        __llvm_initialized[] = true
    end

    @timeit_debug to "IR generation" begin
        ir, compiled = irgen(job, method_instance)
        ctx = context(ir)
        entry_fn = compiled[method_instance].specfunc
        entry = functions(ir)[entry_fn]
    end

    # always preload the runtime, and do so early; it cannot be part of any timing block
    # because it recurses into the compiler
    if libraries
        runtime = load_runtime(job; ctx)
        runtime_fns = LLVM.name.(defs(runtime))
        runtime_intrinsics = ["julia.gc_alloc_obj"]
    end

    @timeit_debug to "Library linking" begin
        if libraries
            # target-specific libraries
            undefined_fns = LLVM.name.(decls(ir))
            @timeit_debug to "target libraries" link_libraries!(job, ir, undefined_fns)

            # GPU run-time library
            if any(fn -> fn in runtime_fns || fn in runtime_intrinsics, undefined_fns)
                @timeit_debug to "runtime library" link_library!(ir, runtime)
            end
        end
    end

    # mark everything internal except for the entry and any exported global variables.
    # this makes sure that the optimizer can, e.g., touch function signatures.
    ModulePassManager() do pm
        # NOTE: this needs to happen after linking libraries to remove unused functions,
        #       but before deferred codegen so that all kernels remain available.
        exports = String[entry_fn]
        for gvar in globals(ir)
            if linkage(gvar) == LLVM.API.LLVMExternalLinkage
                push!(exports, LLVM.name(gvar))
            end
        end
        internalize!(pm, exports)
        run!(pm, ir)
    end

    # finalize the current module. this needs to happen before linking deferred modules,
    # since those modules have been finalized themselves, and we don't want to re-finalize.
    entry = finish_module!(job, ir, entry)

    # deferred code generation
    do_deferred_codegen = !only_entry && deferred_codegen &&
                          haskey(functions(ir), "deferred_codegen")
    if do_deferred_codegen
        dyn_marker = functions(ir)["deferred_codegen"]

        cache = Dict{CompilerJob, String}(job => entry_fn)

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
                dyn_job = deferred_codegen_jobs[id]
                if dyn_job isa FunctionSpec
                    dyn_job = similar(job, dyn_job)
                end
                push!(get!(worklist, dyn_job, LLVM.CallInst[]), call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_entry_fn = get!(cache, dyn_job) do
                    dyn_ir, dyn_meta = codegen(:llvm, dyn_job; optimize=false,
                                               deferred_codegen=false, parent_job=job)
                    dyn_entry_fn = LLVM.name(dyn_meta.entry)
                    merge!(compiled, dyn_meta.compiled)
                    @assert context(dyn_ir) == ctx
                    link!(ir, dyn_ir)
                    changed = true
                    dyn_entry_fn
                end
                dyn_entry = functions(ir)[dyn_entry_fn]

                # insert a pointer to the function everywhere the entry is used
                T_ptr = convert(LLVMType, Ptr{Cvoid}; ctx)
                for call in worklist[dyn_job]
                    Builder(ctx) do builder
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

    @timeit_debug to "IR post-processing" begin
        if optimize
            @timeit_debug to "optimization" begin
                optimize!(job, ir)

                # deferred codegen has some special optimization requirements,
                # which also need to happen _after_ regular optimization.
                # XXX: make these part of the optimizer pipeline?
                do_deferred_codegen && ModulePassManager() do pm
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

            # optimization may have replaced functions, so look the entry point up again
            entry = functions(ir)[entry_fn]
        end

        @timeit_debug to "clean-up" begin
            # we can only clean-up now, as optimization may lower or introduce calls to
            # functions from the GPU runtime (e.g. julia.gc_alloc_obj -> gpu_gc_pool_alloc)
            ModulePassManager() do pm
                # eliminate all unused internal functions
                global_optimizer!(pm)
                global_dce!(pm)
                strip_dead_prototypes!(pm)

                # merge constants (such as exception messages)
                constant_merge!(pm)

                run!(pm, ir)
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

        if ccall(:jl_is_debugbuild, Cint, ()) == 1
            @timeit_debug to "verification" verify(ir)
        end
    end

    return ir, (; entry, compiled)
end

@locked function emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module;
                          strip::Bool=false, validate::Bool=true, format::LLVM.API.LLVMCodeGenFileType)
    if validate
        @timeit_debug to "validation" begin
            check_invocation(job)
            check_ir(job, ir)
        end
    end

    # NOTE: strip after validation to get better errors
    if strip
        @timeit_debug to "strip debug info" strip_debuginfo!(ir)
    end

    @timeit_debug to "LLVM back-end" begin
        @timeit_debug to "preparation" prepare_execution!(job, ir)

        code = @timeit_debug to "machine-code generation" mcgen(job, ir, format)
    end

    return code, ()
end
