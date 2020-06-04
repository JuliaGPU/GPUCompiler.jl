# LLVM IR generation


## method compilation tracer

# this functionality is used to detect recursion, and functions that shouldn't be called.
# it is a hack, and should disappear over time. don't add new features to it.

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(job::CompilerJob, call_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in call_stack
        method = method_instance.def
        if method.name === :overdub && isdefined(method, :generator)
            # The inline frames are maintained by the dwarf based backtrace, but here we only have the
            # calls to overdub directly, the backtrace therefore is collapsed and we have to
            # lookup the overdubbed function, but only if we likely are using the generated variant.
            actual_sig = Tuple{method_instance.specTypes.parameters[3:end]...}
            m = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), actual_sig, typemax(UInt))
            method = m.func::Method
        end
        frame = StackTraces.StackFrame(method.name, method.file, method.line)
        pushfirst!(bt, frame)
    end
    bt
end

# NOTE: we use an exception to be able to display a stack trace using the logging framework
struct MethodSubstitutionWarning <: Exception
    original::Method
    substitute::Method
end
Base.showerror(io::IO, err::MethodSubstitutionWarning) =
    print(io, "You called $(err.original), maybe you intended to call $(err.substitute) instead?")
const method_substitution_whitelist = [:hypot, :exp]

mutable struct MethodCompileTracer
    job::CompilerJob
    call_stack::Vector{Core.MethodInstance}
    last_method_instance::Union{Nothing,Core.MethodInstance}

    MethodCompileTracer(job, start) = new(job, Core.MethodInstance[start])
    MethodCompileTracer(job) = new(job, Core.MethodInstance[])
end

function Base.push!(tracer::MethodCompileTracer, method_instance)
    push!(tracer.call_stack, method_instance)

    if VERSION < v"1.5.0-DEV.393"
        # check for recursion
        if method_instance in tracer.call_stack[1:end-1]
            throw(KernelError(tracer.job, "recursion is currently not supported";
                              bt=backtrace(tracer.job, tracer.call_stack)))
        end
    end

    # check for Base functions that exist in the GPU package
    # FIXME: this might be too coarse
    method = method_instance.def
    if Base.moduleroot(method.module) == Base &&
        isdefined(runtime_module(tracer.job), method_instance.def.name) &&
        !in(method_instance.def.name, method_substitution_whitelist)
        substitute_function = getfield(runtime_module(tracer.job), method.name)
        tt = Tuple{method_instance.specTypes.parameters[2:end]...}
        if hasmethod(substitute_function, tt)
            method′ = which(substitute_function, tt)
            if method′.module == runtime_module(tracer.job)
                @warn "calls to Base intrinsics might be GPU incompatible" exception=(MethodSubstitutionWarning(method, method′), backtrace(tracer.job, tracer.call_stack))
            end
        end
    end
end

function Base.pop!(tracer::MethodCompileTracer, method_instance)
    @compiler_assert last(tracer.call_stack) == method_instance tracer.job
    tracer.last_method_instance = pop!(tracer.call_stack)
end

Base.last(tracer::MethodCompileTracer) = tracer.last_method_instance


## Julia compiler integration

if VERSION >= v"1.5.0-DEV.393"

# JuliaLang/julia#25984 significantly restructured the compiler

function compile_method_instance(job::CompilerJob, method_instance::Core.MethodInstance, world)
    # set-up the compiler interface
    tracer = MethodCompileTracer(job, method_instance)
    hook_emit_function(method_instance, code) = push!(tracer, method_instance)
    hook_emitted_function(method_instance, code) = pop!(tracer, method_instance)
    param_kwargs = [:track_allocations  => false,
                    :code_coverage      => false,
                    :static_alloc       => false,
                    :prefer_specsig     => true,
                    :emit_function      => hook_emit_function,
                    :emitted_function   => hook_emitted_function]
    if LLVM.version() >= v"8.0" && VERSION >= v"1.3.0-DEV.547"
        push!(param_kwargs, :gnu_pubnames => false)

        debug_info_kind = if Base.JLOptions().debug_level == 0
            LLVM.API.LLVMDebugEmissionKindNoDebug
        elseif Base.JLOptions().debug_level == 1
            LLVM.API.LLVMDebugEmissionKindLineTablesOnly
        elseif Base.JLOptions().debug_level >= 2
            LLVM.API.LLVMDebugEmissionKindFullDebug
        end

        # LLVM's debug info crashes older CUDA assemblers
        if job.target isa PTXCompilerTarget # && driver_version(job.target) < v"10.2"
            # FIXME: this was supposed to be fixed on 10.2
            @debug "Incompatibility detected between CUDA and LLVM 8.0+; disabling debug info emission" maxlog=1
            debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
        end

        push!(param_kwargs, :debug_info_kind => Cint(debug_info_kind))
    end
    params = Base.CodegenParams(;param_kwargs...)

    # generate IR
    if VERSION >= v"1.5.0-DEV.851"
        native_code = ccall(:jl_create_native, Ptr{Cvoid},
                            (Vector{Core.MethodInstance}, Base.CodegenParams, Cint),
                            [method_instance], params, #=extern policy=# 1)
    else
        native_code = ccall(:jl_create_native, Ptr{Cvoid},
                            (Vector{Core.MethodInstance}, Base.CodegenParams),
                            [method_instance], params)
    end
    @assert native_code != C_NULL
    llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                         (Ptr{Cvoid},), native_code)
    @assert llvm_mod_ref != C_NULL
    llvm_mod = LLVM.Module(llvm_mod_ref)

    # get the top-level code
    code = if VERSION >= v"1.6.0-DEV.12"
        # TODO: use our own interpreter
        interpreter = Core.Compiler.NativeInterpreter(world)
        Core.Compiler.inf_for_methodinstance(interpreter, method_instance, world, world)
    else
        Core.Compiler.inf_for_methodinstance(method_instance, world, world)
    end

    # get the top-level function index
    llvm_func_idx = Ref{Int32}(-1)
    llvm_specfunc_idx = Ref{Int32}(-1)
    ccall(:jl_breakpoint, Nothing, ())
    ccall(:jl_get_function_id, Nothing,
          (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
          native_code, code, llvm_func_idx, llvm_specfunc_idx)
    @assert llvm_func_idx[] != -1
    @assert llvm_specfunc_idx[] != -1

    # get the top-level function)
    llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                     (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
    @assert llvm_func_ref != C_NULL
    llvm_func = LLVM.Function(llvm_func_ref)
    llvm_specfunc_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                         (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
    @assert llvm_specfunc_ref != C_NULL
    llvm_specfunc = LLVM.Function(llvm_specfunc_ref)

    # configure the module
    triple!(llvm_mod, llvm_triple(job.target))
    if llvm_datalayout(job.target) !== nothing
        datalayout!(llvm_mod, llvm_datalayout(job.target))
    end

    return llvm_specfunc, llvm_mod
end

else

function module_setup(job::CompilerJob, mod::LLVM.Module)
    # configure the module
    triple!(mod, llvm_triple(job.target))
    datalayout!(mod, llvm_datalayout(job.target))

    # add debug info metadata
    if LLVM.version() >= v"8.0"
        # Set Dwarf Version to 2, the DI printer will downgrade to v2 automatically,
        # but this is technically correct and the only version supported by NVPTX
        LLVM.flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
            Metadata(ConstantInt(Int32(2), JuliaContext()))
        LLVM.flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorError] =
            Metadata(ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext()))
    else
        push!(metadata(mod), "llvm.module.flags",
             MDNode([ConstantInt(Int32(1), JuliaContext()),    # llvm::Module::Error
                     MDString("Debug Info Version"),
                     ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext())]))
    end
end

function compile_method_instance(job::CompilerJob, method_instance::Core.MethodInstance, world)
    function postprocess(ir)
        # get rid of jfptr wrappers
        for llvmf in functions(ir)
            startswith(LLVM.name(llvmf), "jfptr_") && unsafe_delete!(ir, llvmf)
        end

        return
    end

    # set-up the compiler interface
    tracer = MethodCompileTracer(job)
    hook_emit_function(method_instance, code, world) = push!(tracer, method_instance)
    hook_emitted_function(method_instance, code, world) = pop!(tracer, method_instance)
    dependencies = MultiDict{Core.MethodInstance,LLVM.Function}()
    function hook_module_setup(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        module_setup(job, ir)
    end
    function hook_module_activation(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        postprocess(ir)

        # find the function that this module defines
        llvmfs = filter(llvmf -> !isdeclaration(llvmf) &&
                                 linkage(llvmf) == LLVM.API.LLVMExternalLinkage,
                        collect(functions(ir)))

        llvmf = nothing
        if length(llvmfs) == 1
            llvmf = first(llvmfs)
        elseif length(llvmfs) > 1
            llvmfs = filter!(llvmf -> startswith(LLVM.name(llvmf), "julia_"), llvmfs)
            if length(llvmfs) == 1
                llvmf = first(llvmfs)
            end
        end

        @compiler_assert llvmf !== nothing job

        insert!(dependencies, last(tracer), llvmf)
    end
    param_kwargs = [:cached             => false,
                    :track_allocations  => false,
                    :code_coverage      => false,
                    :static_alloc       => false,
                    :prefer_specsig     => true,
                    :module_setup       => hook_module_setup,
                    :module_activation  => hook_module_activation,
                    :emit_function      => hook_emit_function,
                    :emitted_function   => hook_emitted_function]
    if LLVM.version() >= v"8.0" && VERSION >= v"1.3.0-DEV.547"
        push!(param_kwargs, :gnu_pubnames => false)

        debug_info_kind = if Base.JLOptions().debug_level == 0
            LLVM.API.LLVMDebugEmissionKindNoDebug
        elseif Base.JLOptions().debug_level == 1
            LLVM.API.LLVMDebugEmissionKindLineTablesOnly
        elseif Base.JLOptions().debug_level >= 2
            LLVM.API.LLVMDebugEmissionKindFullDebug
        end

        # LLVM's debug info crashes older CUDA assemblers
        if job.target isa PTXCompilerTarget # && driver_version(job.target) < v"10.2"
            # FIXME: this was supposed to be fixed on 10.2
            @debug "Incompatibility detected between CUDA and LLVM 8.0+; disabling debug info emission" maxlog=1
            debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
        end

        push!(param_kwargs, :debug_info_kind => Cint(debug_info_kind))
    end
    params = Base.CodegenParams(;param_kwargs...)

    # get the code
    ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                (Any, UInt, Bool, Bool, Base.CodegenParams),
                method_instance, world, #=wrapper=#false, #=optimize=#false, params)
    if ref == C_NULL
        throw(InternalCompilerError(job, "the Julia compiler could not generate LLVM IR"))
    end
    llvmf = LLVM.Function(ref)
    ir = LLVM.parent(llvmf)
    postprocess(ir)

    # link in dependent modules
    entry = llvmf
    mod = LLVM.parent(entry)
    @timeit_debug to "linking" begin
        # we disable Julia's compilation cache not to poison it with GPU-specific code.
        # as a result, we might get multiple modules for a single method instance.
        cache = Dict{String,String}()

        for called_method_instance in keys(dependencies)
            llvmfs = dependencies[called_method_instance]

            # link the first module
            llvmf = popfirst!(llvmfs)
            llvmfn = LLVM.name(llvmf)
            link!(mod, LLVM.parent(llvmf))

            # process subsequent duplicate modules
            for dup_llvmf in llvmfs
                if Base.JLOptions().debug_level >= 2
                    # link them too, to ensure accurate backtrace reconstruction
                    link!(mod, LLVM.parent(dup_llvmf))
                else
                    # don't link them, but note the called function name in a cache
                    dup_llvmfn = LLVM.name(dup_llvmf)
                    cache[dup_llvmfn] = llvmfn
                end
            end
        end

        # resolve function declarations with cached entries
        for llvmf in filter(isdeclaration, collect(functions(mod)))
            llvmfn = LLVM.name(llvmf)
            if haskey(cache, llvmfn)
                def_llvmfn = cache[llvmfn]
                replace_uses!(llvmf, functions(mod)[def_llvmfn])

                @compiler_assert isempty(uses(llvmf)) job
                unsafe_delete!(LLVM.parent(llvmf), llvmf)
            end
        end
    end

    return entry, mod
end

end

function irgen(job::CompilerJob, method_instance::Core.MethodInstance, world)
    entry, mod = @timeit_debug to "emission" compile_method_instance(job, method_instance, world)

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            # only occurs in debug builds
            delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, JuliaContext()))

            if VERSION < v"1.5.0-DEV.393"
                # make function names safe for ptxas
                llvmfn = LLVM.name(llvmf)
                if !isdeclaration(llvmf)
                    llvmfn′ = safe_name(llvmfn)
                    if llvmfn != llvmfn′
                        LLVM.name!(llvmf, llvmfn′)
                        llvmfn = llvmfn′
                    end
                end
            end

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    # target-specific processing
    process_module!(job, mod)

    # rename the entry point
    if job.source.name !== nothing
        llvmfn = safe_name(string("julia_", job.source.name))
    else
        # strip the globalUnique counter
        llvmfn = LLVM.name(entry)
    end
    LLVM.name!(entry, llvmfn)

    # promote entry-points to kernels and mangle its name
    if job.source.kernel
        entry = promote_kernel!(job, mod, entry)
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        internalize!(pm, [LLVM.name(entry)])

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)

        # NOTE: if an optimization is missing, try scheduling an entirely new optimization
        # to see which passes need to be added to the target-specific list
        #     LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
        #     ModulePassManager() do pm
        #         add_library_info!(pm, triple(mod))
        #         add_transform_info!(pm, tm)
        #         PassManagerBuilder() do pmb
        #             populate!(pm, pmb)
        #         end
        #         run!(pm, mod)
        #     end
    end

    return mod, entry
end


## name mangling

# we generate function names that look like C++ functions, because many NVIDIA tools
# support them, e.g., grouping different instantiations of the same kernel together.

function mangle_param(t)
    t == Nothing && return "v"

    if isa(t, DataType) || isa(t, Core.Function)
        tn = safe_name(t)
        str = "$(length(tn))$tn"

        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t)
            end
            str *= "E"
        end

        str
    elseif isa(t, Integer)
        "Li$(t)E"
    else
        tn = safe_name(t)
        "$(length(tn))$tn"
    end
end

function mangle_call(f, tt)
    fn = safe_name(f)
    str = "_Z$(length(fn))$fn"

    for t in tt.parameters
        str *= mangle_param(t)
    end

    return str
end

# make names safe for ptxas
safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9_]"=>"_")
safe_name(f::Union{Core.Function,DataType}) = safe_name(String(nameof(f)))
safe_name(f::LLVM.Function) = safe_name(LLVM.name(f))
safe_name(x) = safe_name(repr(x))


## exception handling

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "lower throw" begin

    throw_functions = Dict{String,String}(
        "jl_throw"                      => "exception",
        "jl_error"                      => "error",
        "jl_too_few_args"               => "too few arguments exception",
        "jl_too_many_args"              => "too many arguments exception",
        "jl_type_error"                 => "type error",
        "jl_type_error_rt"              => "type error",
        "jl_undefined_var_error"        => "undefined variable error",
        "jl_bounds_error"               => "bounds error",
        "jl_bounds_error_v"             => "bounds error",
        "jl_bounds_error_int"           => "bounds error",
        "jl_bounds_error_tuple_int"     => "bounds error",
        "jl_bounds_error_unboxed_int"   => "bounds error",
        "jl_bounds_error_ints"          => "bounds error",
        "jl_eof_error"                  => "EOF error"
    )

    for (fn, name) in throw_functions
        if haskey(functions(mod), fn)
            f = functions(mod)[fn]

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                let builder = Builder(JuliaContext())
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                    dispose(builder)
                end

                # remove the call
                call_args = collect(operands(call))[1:end-1] # last arg is function itself
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(operands(cast))
                        isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                        unsafe_delete!(LLVM.parent(arg), arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(uses(f)) job
         end
     end

    end
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` will be emitted, On
# debug level 1, the exception name will be printed, and on debug level 2 the individual
# stack frames (as recovered from the LLVM debug information) will be printed as well.
function emit_exception!(builder, name, inst)
    job = current_job::CompilerJob
    bb = position(builder)
    fun = LLVM.parent(bb)
    mod = LLVM.parent(fun)

    # report the exception
    if Base.JLOptions().debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if Base.JLOptions().debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if Base.JLOptions().debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(rt.llvm_types[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(rt.llvm_types[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception
    call!(builder, Runtime.get(:signal_exception))

    emit_trap!(job, builder, mod, inst)
end

function emit_trap!(job::CompilerJob, builder, mod, inst)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(JuliaContext())))
    end
    call!(builder, trap)
end


## kernel promotion

# promote a function to a kernel
function promote_kernel!(job::CompilerJob, mod::LLVM.Module, kernel::LLVM.Function)
    # pass non-opaque pointer arguments by value (this improves performance,
    # and is mandated by certain back-ends like SPIR-V). only do so for values
    # that aren't a Julia pointer, so we ca still pass those directly.
    kernel_ft = eltype(llvmtype(kernel)::LLVM.PointerType)::LLVM.FunctionType
    kernel_sig = Base.signature_type(job.source.f, job.source.tt)::Type
    kernel_types = filter(dt->!isghosttype(dt) &&
                              (VERSION < v"1.5.0-DEV.581" || !Core.Compiler.isconstType(dt)),
                          [kernel_sig.parameters...])
    @compiler_assert length(kernel_types) == length(parameters(kernel_ft)) job
    for (i, (param_ft,arg_typ)) in enumerate(zip(parameters(kernel_ft), kernel_types))
        if param_ft isa LLVM.PointerType && issized(eltype(param_ft)) &&
           !(arg_typ <: Ptr) && !(VERSION >= v"1.5" && arg_typ <: Core.LLVMPtr)
            push!(parameter_attributes(kernel, i), EnumAttribute("byval"))
        end
    end

    # target-specific processing
    kernel = process_kernel!(job, mod, kernel)

    return kernel
end
