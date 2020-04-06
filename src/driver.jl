# compiler driver and main interface

# NOTE: the keyword arguments to compile/codegen control those aspects of compilation that
#       might have to be changed (e.g. set libraries=false when recursing, or set
#       strip=true for reflection). What remains defines the compilation job itself,
#       and those values are contained in the AbstractCompilerJob struct.

# (::AbstractCompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(target::Symbol, job::AbstractCompilerJob;
            libraries=true, dynamic_parallelism=true,
            optimize=true, strip=false, strict=true, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `target` argument: `:julia` for Julia IR, `:llvm` for
LLVM IR, `:ptx` for PTX assembly and `:cuda` for CUDA driver objects. If the `kernel` flag
is set, specialized code generation and optimization for kernel functions is enabled.

The following keyword arguments are supported:
- `libraries`: link the GPU runtime and `libdevice` libraries (if required)
- `dynamic_parallelism`: resolve dynamic parallelism (if required)
- `optimize`: optimize the code (default: true)
- `strip`: strip non-functional metadata and debug information (default: false)
- `strict`: perform code validation either as early or as late as possible

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
function compile(target::Symbol, job::AbstractCompilerJob;
                 libraries::Bool=true, dynamic_parallelism::Bool=true,
                 optimize::Bool=true, strip::Bool=false, strict::Bool=true)
    if compile_hook[] != nothing
        compile_hook[](job)
    end

    return codegen(target, job;
                   libraries=libraries, dynamic_parallelism=dynamic_parallelism,
                   optimize=optimize, strip=strip, strict=strict)
end

function codegen(target::Symbol, job::AbstractCompilerJob;
                 libraries::Bool=true, dynamic_parallelism::Bool=true, optimize::Bool=true,
                 strip::Bool=false, strict::Bool=true)
    ## Julia IR

    @timeit_debug to "validation" check_method(job)

    @timeit_debug to "Julia front-end" begin

        # get the method instance
        world = typemax(UInt)
        meth = which(job.f, job.tt)
        sig = Base.signature_type(job.f, job.tt)::Type
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

    target == :julia && return method_instance


    ## LLVM IR

    # always preload the runtime, and do so early; it cannot be part of any timing block
    # because it recurses into the compiler
    if libraries
        runtime = load_runtime(job)
        runtime_fns = LLVM.name.(defs(runtime))
    end

    @timeit_debug to "LLVM middle-end" begin
        ir, kernel = @timeit_debug to "IR generation" irgen(job, method_instance, world)

        if libraries && job.target.link_libraries != nothing
            undefined_fns = LLVM.name.(decls(ir))
            @timeit_debug to "target libraries" job.target.link_libraries(job, ir, undefined_fns)
        end

        if optimize
            kernel = @timeit_debug to "optimization" optimize!(job, ir, kernel)
        end

        if libraries
            undefined_fns = LLVM.name.(decls(ir))
            if any(fn -> fn in runtime_fns, undefined_fns)
                @timeit_debug to "runtime library" link_library!(job, ir, runtime)
            end
        end

        if ccall(:jl_is_debugbuild, Cint, ()) == 1
            @timeit_debug to "verification" verify(ir)
        end

        kernel_fn = LLVM.name(kernel)
    end

    # dynamic parallelism
    if dynamic_parallelism && haskey(functions(ir), "cudanativeCompileKernel")
        dyn_marker = functions(ir)["cudanativeCompileKernel"]

        cache = Dict{AbstractCompilerJob, String}(job => kernel_fn)

        # iterative compilation (non-recursive)
        changed = true
        while changed
            changed = false

            # find dynamic kernel invocations
            # TODO: recover this information earlier, from the Julia IR
            worklist = MultiDict{AbstractCompilerJob, LLVM.CallInst}()
            for use in uses(dyn_marker)
                # decode the call
                call = user(use)::LLVM.CallInst
                id = convert(Int, first(operands(call)))

                global delayed_cufunctions
                dyn_f, dyn_tt = delayed_cufunctions[id]
                dyn_job = AbstractCompilerJob(dyn_f, dyn_tt, job.cap, #=kernel=# true)
                push!(worklist, dyn_job => call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_kernel_fn = get!(cache, dyn_job) do
                    dyn_ir, dyn_kernel = codegen(:llvm, dyn_job;
                                                 optimize=optimize, strip=strip,
                                                 dynamic_parallelism=false, strict=false)
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

        # all dynamic launches should have been resolved
        @compiler_assert isempty(uses(dyn_marker)) job
        unsafe_delete!(ir, dyn_marker)
    end

    if strict
        # NOTE: keep in sync with non-strict check below
        @timeit_debug to "validation" begin
            check_invocation(job, kernel)
            check_ir(job, ir)
        end
    end

    if strip
        @timeit_debug to "strip debug info" strip_debuginfo!(ir)
    end

    target == :llvm && return ir, kernel


    ## PTX machine code

    @timeit_debug to "LLVM back-end" begin
        @timeit_debug to "preparation" prepare_execution!(job, ir)

        asm = @timeit_debug to "machine-code generation" mcgen(job, ir, kernel)
    end

    undefined_fns = LLVM.name.(decls(ir))
    target == :ptx && return asm, kernel_fn, undefined_fns


    error("Unknown compilation target $target")
end
