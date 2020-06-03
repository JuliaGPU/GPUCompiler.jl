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

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
function compile(target::Symbol, job::CompilerJob;
                 libraries::Bool=true, deferred_codegen::Bool=true,
                 optimize::Bool=true, strip::Bool=false, validate::Bool=true)
    if compile_hook[] != nothing
        compile_hook[](job)
    end

    return codegen(target, job;
                   libraries=libraries, deferred_codegen=deferred_codegen,
                   optimize=optimize, strip=strip, validate=validate)
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

function codegen(output::Symbol, job::CompilerJob;
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                 strip::Bool=false, validate::Bool=true)
    ## Julia IR

    @timeit_debug to "validation" check_method(job)

    @timeit_debug to "Julia front-end" begin

        # get the method instance
        world = typemax(UInt)
        meth = which(job.source.f, job.source.tt)
        sig = Base.signature_type(job.source.f, job.source.tt)::Type
        (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                          (Any, Any), sig, meth.sig)::Core.SimpleVector
        if VERSION >= v"1.2.0-DEV.320"
            meth = Base.func_for_method_checked(meth, ti, env)
        else
            meth = Base.func_for_method_checked(meth, ti)
        end
        method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                      (Any, Any, Any, UInt), meth, ti, env, world)

        for var in env
            if var isa TypeVar
                throw(KernelError(job, "method captures a typevar (you probably use an unbound type variable)"))
            end
        end
    end

    output == :julia && return method_instance


    ## LLVM IR

    # always preload the runtime, and do so early; it cannot be part of any timing block
    # because it recurses into the compiler
    if libraries
        runtime = load_runtime(job)
        runtime_fns = LLVM.name.(defs(runtime))
    end

    @timeit_debug to "LLVM middle-end" begin
        ir, kernel = @timeit_debug to "IR generation" irgen(job, method_instance, world)
        kernel_fn = LLVM.name(kernel)

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

        # remove everything except for the kernel
        @timeit_debug to "clean-up" begin
            exports = String[kernel_fn]
            ModulePassManager() do pm
                # internalize all functions that aren't exports
                internalize!(pm, exports)

                # eliminate all unused internal functions
                global_optimizer!(pm)
                global_dce!(pm)
                strip_dead_prototypes!(pm)

                run!(pm, ir)
            end
        end
    end

    # deferred code generation
    if deferred_codegen && haskey(functions(ir), "deferred_codegen")
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
                    dyn_ir, dyn_kernel = codegen(:llvm, dyn_job; optimize=optimize,
                                                 strip=strip, validate=validate,
                                                 deferred_codegen=false)
                    dyn_kernel_fn = LLVM.name(dyn_kernel)
                    link!(ir, dyn_ir)
                    changed = true
                    dyn_kernel_fn
                end
                dyn_kernel = functions(ir)[dyn_kernel_fn]

                # insert a pointer to the function everywhere the kernel is used
                T_ptr = convert(LLVMType, Ptr{Cvoid})
                for call in worklist[dyn_job]
                    Builder(JuliaContext()) do builder
                        position!(builder, call)
                        fptr = ptrtoint!(builder, dyn_kernel, T_ptr)
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

    if output == :llvm
        if strip
            @timeit_debug to "strip debug info" strip_debuginfo!(ir)
        end

        return ir, kernel
    end


    ## machine code

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

        if output == :asm
            code = @timeit_debug to "machine-code generation" mcgen(job, ir, kernel, LLVM.API.LLVMAssemblyFile)
        elseif output == :obj
            code = @timeit_debug to "machine-code generation" mcgen(job, ir, kernel, LLVM.API.LLVMObjectFile)
        end
    end

    undefined_fns = LLVM.name.(decls(ir))
    undefined_gbls = map(x->(name=LLVM.name(x),type=llvmtype(x),external=isextinit(x)), LLVM.globals(ir))

    (output == :asm || output == :obj) && return code, kernel_fn, undefined_fns, undefined_gbls


    error("Unknown compilation output $output")
end
