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

# JuliaLang/julia#34516: keyword functions drop @nospecialize
@locked emit_llvm(@nospecialize(job::CompilerJob), method_instance::Core.MethodInstance;
                  libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                  only_entry::Bool=false) =
    emit_llvm(job, method_instance, libraries, deferred_codegen, optimize, only_entry)
function emit_llvm(@nospecialize(job::CompilerJob), method_instance::Core.MethodInstance,
                   libraries::Bool, deferred_codegen::Bool, optimize::Bool,
                   only_entry::Bool)
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
        if haskey(globals(runtime), "llvm.used")
            # the runtime shouldn't link-in stuff that gets preserved in the output. this is
            # a hack to get rid of the device function slots emitted by the PTX back-end,
            # but it also makes sense.
            gv = globals(runtime)["llvm.used"]
            LLVM.unsafe_delete!(runtime, gv)
        end
        runtime_fns = LLVM.name.(defs(runtime))
    end

    @timeit_debug to "LLVM middle-end" begin
        # target-specific libraries
        if libraries
            undefined_fns = LLVM.name.(decls(ir))
            @timeit_debug to "target libraries" @invokelatest link_libraries!(job, ir, undefined_fns)
        end

        if optimize
            @timeit_debug to "optimization" optimize!(job, ir)

            # optimization may have replaced functions, so look the entry point up again
            entry = functions(ir)[entry_fn]
        end

        if libraries
            undefined_fns = LLVM.name.(decls(ir))
            if any(fn -> fn in runtime_fns, undefined_fns)
                @timeit_debug to "runtime library" link_library!(ir, runtime)
            end
        end

        if ccall(:jl_is_debugbuild, Cint, ()) == 1
            @timeit_debug to "verification" verify(ir)
        end

        if only_entry
            # replace non-entry function definitions with a declaration
            for f in functions(ir)
                f == entry && continue
                isdeclaration(f) && continue
                LLVM.isintrinsic(f) && continue
                # FIXME: expose llvm::Function::deleteBody with a C API
                fn = LLVM.name(f)
                LLVM.name!(f, "")
                f′ = LLVM.Function(ir, fn, eltype(llvmtype(f)))
                # copying attributes is broken due to maleadt/LLVM.jl#186,
                # but that doesn't matter because `only_entry` is only used for reflection,
                # and the emitted code has already been optimized at this point.
                replace_uses!(f, f′)
            end
        end

        # remove everything except for the entry and any exported global variables
        @timeit_debug to "clean-up" begin
            exports = String[entry_fn]
            for gvar in globals(ir)
                push!(exports, LLVM.name(gvar))
            end

            ModulePassManager() do pm
                internalize!(pm, exports)

                # eliminate all unused internal functions
                global_optimizer!(pm)
                global_dce!(pm)
                strip_dead_prototypes!(pm)

                # merge constants (such as exception messages) from the runtime
                constant_merge!(pm)

                run!(pm, ir)
            end
        end
    end

    # deferred code generation
    if !only_entry && deferred_codegen && haskey(functions(ir), "deferred_codegen")
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
                    dyn_ir, dyn_meta = codegen(:llvm, dyn_job; optimize,
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

        # merge constants (such as exception messages) from each entry
        # and on platforms that support it inline and optimize the call to
        # the deferred code, in particular we want to remove unnecessary
        # alloca's that are created by pass-by-ref semantics.
        ModulePassManager() do pm
            instruction_combining!(pm)
            constant_merge!(pm)
            always_inliner!(pm)
            scalar_repl_aggregates_ssa!(pm)
            promote_memory_to_register!(pm)
            gvn!(pm)

            run!(pm, ir)
        end

        # all deferred compilations should have been resolved
        @compiler_assert isempty(uses(dyn_marker)) job
        unsafe_delete!(ir, dyn_marker)
    end

    return ir, (; entry, compiled)
end

# JuliaLang/julia#34516: keyword functions drop @nospecialize
@locked emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module;
                 strip::Bool=false, validate::Bool=true, format::LLVM.API.LLVMCodeGenFileType) =
    emit_asm(job, ir, strip, validate, format)
function emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module,
                                strip::Bool, validate::Bool, format::LLVM.API.LLVMCodeGenFileType)
    @invokelatest finish_module!(job, ir)

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
