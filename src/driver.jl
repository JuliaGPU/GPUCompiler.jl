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
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false)
    ## Julia IR

    method_instance, world = emit_julia(job)

    output == :julia && return method_instance


    ## LLVM IR

    ir, kernel = emit_llvm(job, method_instance, world;
                           libraries, deferred_codegen, optimize, only_entry)

    if output == :llvm
        if strip
            @timeit_debug to "strip debug info" strip_debuginfo!(ir)
        end

        return ir, kernel
    end


    ## machine code

    format = if output == :asm
        LLVM.API.LLVMAssemblyFile
    elseif output == :obj
        LLVM.API.LLVMObjectFile
    else
        error("Unknown assembly format $output")
    end
    code = emit_asm(job, ir, kernel; strip, validate, format)

    undefined_fns = LLVM.name.(decls(ir))
    undefined_gbls = map(x->(name=LLVM.name(x),type=llvmtype(x),external=isextinit(x)), LLVM.globals(ir))

    (output == :asm || output == :obj) && return code, LLVM.name(kernel), undefined_fns, undefined_gbls


    error("Unknown compilation output $output")
end

function emit_julia(@nospecialize(job::CompilerJob))
    @timeit_debug to "validation" check_method(job)

    @timeit_debug to "Julia front-end" begin

        # get the method instance
        world = Base.get_world_counter()
        meth = which(job.source.f, job.source.tt)
        sig = Base.signature_type(job.source.f, job.source.tt)::Type
        (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                          (Any, Any), sig, meth.sig)::Core.SimpleVector
        meth = Base.func_for_method_checked(meth, ti, env)
        method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                      (Any, Any, Any, UInt), meth, ti, env, world)

        for var in env
            if var isa TypeVar
                throw(KernelError(job, "method captures a typevar (you probably use an unbound type variable)"))
            end
        end
    end

    return method_instance, world
end

# primitive mechanism for deferred compilation, for implementing CUDA dynamic parallelism.
# this could both be generalized (e.g. supporting actual function calls, instead of
# returning a function pointer), and be integrated with the nonrecursive codegen.
const deferred_codegen_jobs = Vector{Tuple{Core.Function,Type}}()
@generated function deferred_codegen(::Val{f}, ::Val{tt}) where {f,tt}
    push!(deferred_codegen_jobs, (f,tt))
    id = length(deferred_codegen_jobs)

    quote
        # TODO: add an edge to this method instance to support method redefinitions
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), $id)
    end
end

@locked function emit_llvm(@nospecialize(job::CompilerJob), @nospecialize(method_instance), world;
                           libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                           only_entry::Bool=false)
    @timeit_debug to "IR generation" begin
        ir, kernel = irgen(job, method_instance, world)
        ctx = context(ir)
        kernel_fn = LLVM.name(kernel)
    end

    # always preload the runtime, and do so early; it cannot be part of any timing block
    # because it recurses into the compiler
    if libraries
        runtime = load_runtime(job, ctx)
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
            @timeit_debug to "target libraries" link_libraries!(job, ir, undefined_fns)
        end

        if optimize
            @timeit_debug to "optimization" optimize!(job, ir)

            # optimization may have replaced functions, so look the entry point up again
            kernel = functions(ir)[kernel_fn]
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
                f == kernel && continue
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

        # remove everything except for the kernel and any exported global variables
        @timeit_debug to "clean-up" begin
            exports = String[kernel_fn]
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

        cache = Dict{CompilerJob, String}(job => kernel_fn)

        # iterative compilation (non-recursive)
        changed = true
        while changed
            changed = false

            # find deferred compiler
            # TODO: recover this information earlier, from the Julia IR
            worklist = MultiDict{CompilerJob, LLVM.CallInst}()
            for use in uses(dyn_marker)
                # decode the call
                call = user(use)::LLVM.CallInst
                id = convert(Int, first(operands(call)))

                global deferred_codegen_jobs
                dyn_f, dyn_tt = deferred_codegen_jobs[id]
                dyn_job = similar(job, FunctionSpec(dyn_f, dyn_tt, #=kernel=# true))
                push!(worklist, dyn_job => call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_kernel_fn = get!(cache, dyn_job) do
                    dyn_ir, dyn_kernel = codegen(:llvm, dyn_job; optimize,
                                                 deferred_codegen=false)
                    dyn_kernel_fn = LLVM.name(dyn_kernel)
                    @assert context(dyn_ir) == ctx
                    link!(ir, dyn_ir)
                    changed = true
                    dyn_kernel_fn
                end
                dyn_kernel = functions(ir)[dyn_kernel_fn]

                # insert a pointer to the function everywhere the kernel is used
                T_ptr = convert(LLVMType, Ptr{Cvoid}, ctx)
                for call in worklist[dyn_job]
                    Builder(ctx) do builder
                        position!(builder, call)
                        fptr = ptrtoint!(builder, dyn_kernel, T_ptr)
                        replace_uses!(call, fptr)
                    end
                    unsafe_delete!(LLVM.parent(call), call)
                end
            end
        end

        # merge constants (such as exception messages) from each kernel
        ModulePassManager() do pm
            constant_merge!(pm)

            run!(pm, ir)
        end

        # all deferred compilations should have been resolved
        @compiler_assert isempty(uses(dyn_marker)) job
        unsafe_delete!(ir, dyn_marker)
    end

    return ir, kernel
end

@locked function emit_asm(@nospecialize(job::CompilerJob), ir::LLVM.Module, kernel::LLVM.Function;
                          strip::Bool=false, validate::Bool=true, format::LLVM.API.LLVMCodeGenFileType)
    finish_module!(job, ir)

    if validate
        @timeit_debug to "validation" begin
            check_invocation(job, kernel)
            check_ir(job, ir)
        end
    end

    # NOTE: strip after validation to get better errors
    if strip
        @timeit_debug to "strip debug info" strip_debuginfo!(ir)
    end

    @timeit_debug to "LLVM back-end" begin
        @timeit_debug to "preparation" prepare_execution!(job, ir)

        code = @timeit_debug to "machine-code generation" mcgen(job, ir, kernel, format)
    end

    return code
end
