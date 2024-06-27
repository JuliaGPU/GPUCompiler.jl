# LLVM IR generation

function irgen(@nospecialize(job::CompilerJob))
    mod, compiled = @timeit_debug to "emission" compile_method_instance(job)
    if job.config.entry_abi === :specfunc
        entry_fn = compiled[job.source].specfunc
    else
        entry_fn = compiled[job.source].func
    end
    @assert entry_fn !== nothing
    entry = functions(mod)[entry_fn]

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            if VERSION < v"1.9" || Base.isdebugbuild()
                # only occurs in debug builds
                delete!(function_attributes(llvmf),
                        EnumAttribute("sspstrong", 0))
            end

            delete!(function_attributes(llvmf),
                    StringAttribute("probe-stack", "inline-asm"))

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end

            # remove the non-specialized jfptr functions
            # TODO: Do we need to remove these?
            if job.config.entry_abi === :specfunc
                if startswith(LLVM.name(llvmf), "jfptr_")
                    unsafe_delete!(mod, llvmf)
                end
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    deprecation_marker = process_module!(job, mod)
    if deprecation_marker != DeprecationMarker()
        Base.depwarn("GPUCompiler.process_module! is deprecated; implement GPUCompiler.finish_module! instead", :process_module)
    end

    # sanitize global values (Julia doesn't when using the external codegen policy)
    for val in [collect(globals(mod)); collect(functions(mod))]
        isdeclaration(val) && continue
        old_name = LLVM.name(val)
        new_name = safe_name(old_name)
        if old_name != new_name
            LLVM.name!(val, new_name)
        end
    end

    # rename and process the entry point
    if job.config.name !== nothing
        LLVM.name!(entry, safe_name(job.config.name))
    elseif job.config.kernel
        LLVM.name!(entry, mangle_sig(job.source.specTypes))
    end
    deprecation_marker = process_entry!(job, mod, entry)
    if deprecation_marker != DeprecationMarker()
        Base.depwarn("GPUCompiler.process_entry! is deprecated; implement GPUCompiler.finish_module! instead", :process_entry)
        entry = deprecation_marker
    end
    if job.config.entry_abi === :specfunc
        func = compiled[job.source].func
        specfunc = LLVM.name(entry)
    else
        func = LLVM.name(entry)
        specfunc = compiled[job.source].specfunc
    end

    compiled[job.source] =
        (; compiled[job.source].ci, func, specfunc)

    # minimal required optimization
    @timeit_debug to "rewrite" begin
        if job.config.kernel && needs_byval(job)
            # pass all bitstypes by value; by default Julia passes aggregates by reference
            # (this improves performance, and is mandated by certain back-ends like SPIR-V).
            args = classify_arguments(job, function_type(entry))
            for arg in args
                if arg.cc == BITS_REF
                    llvm_typ = convert(LLVMType, arg.typ)
                    attr = TypeAttribute("byval", llvm_typ)
                    push!(parameter_attributes(entry, arg.idx), attr)
                end
            end
        end

        # internalize all functions and, but keep exported global variables.
        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        preserved_gvs = String[LLVM.name(entry)]
        for gvar in globals(mod)
            push!(preserved_gvs, LLVM.name(gvar))
        end
        if use_newpm && LLVM.version() >= v"17"
            @dispose pb=NewPMPassBuilder() begin
                add!(pb, InternalizePass(InternalizePassOptions(; preserved_gvs)))
                add!(pb, AlwaysInlinerPass())
                run!(pb, mod, llvm_machine(job.config.target))
            end
        else
            @dispose pm=ModulePassManager() begin
                internalize!(pm, preserved_gvs)
                always_inliner!(pm)
                run!(pm, mod)
            end
        end

        global current_job
        current_job = job
        can_throw(job) || lower_throw!(mod)
    end

    return mod, compiled
end


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

    throw_functions = [
        # unsupported runtime functions that are used to throw specific exceptions
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
        "jl_eof_error"                  => "EOF error",
    ]

    for f in functions(mod)
        fn = LLVM.name(f)
        for (throw_fn, name) in throw_functions
            occursin(throw_fn, fn) || continue

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                @dispose builder=IRBuilder() begin
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                end

                # remove the call
                call_args = arguments(call)
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                #       this is needed for throwing objects with @nospecialize constructors.
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
            break
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
        ft = convert(LLVM.FunctionType, rt)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(parameters(ft)[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(parameters(ft)[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception
    call!(builder, Runtime.get(:signal_exception))

    emit_trap!(job, builder, mod, inst)
end

function emit_trap!(@nospecialize(job::CompilerJob), builder, mod, inst)
    trap_ft = LLVM.FunctionType(LLVM.VoidType())
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", trap_ft)
    end
    call!(builder, trap_ft, trap)
end


## kernel promotion

@enum ArgumentCC begin
    BITS_VALUE  # bitstype, passed as value
    BITS_REF    # bitstype, passed as pointer
    MUT_REF     # jl_value_t*, or the anonymous equivalent
    GHOST       # not passed
end

# Determine the calling convention of a the arguments of a Julia function, given the
# LLVM function type as generated by the Julia code generator. Returns an vector with one
# element for each Julia-level argument, containing a tuple with the following fields:
# - `cc`: the calling convention of the argument
# - `typ`: the Julia type of the argument
# - `name`: the name of the argument
# - `idx`: the index of the argument in the LLVM function type, or `nothing` if the argument
#          is not passed at the LLVM level.
function classify_arguments(@nospecialize(job::CompilerJob), codegen_ft::LLVM.FunctionType)
    source_sig = job.source.specTypes
    source_types = [source_sig.parameters...]

    source_argnames = Base.method_argnames(job.source.def)
    while length(source_argnames) < length(source_types)
        # this is probably due to a trailing vararg; repeat its name
        push!(source_argnames, source_argnames[end])
    end

    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = 1
    for (source_i, (source_typ, source_name)) in enumerate(zip(source_types, source_argnames))
        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc=GHOST, typ=source_typ, name=source_name, idx=nothing))
            continue
        end

        codegen_typ = codegen_types[codegen_i]

        if codegen_typ isa LLVM.PointerType
            llvm_source_typ = convert(LLVMType, source_typ; allow_boxed=true)
            # pointers are used for multiple kinds of arguments
            # - literal pointer values
            if source_typ <: Ptr || source_typ <: Core.LLVMPtr
                @assert llvm_source_typ == codegen_typ
                push!(args, (cc=BITS_VALUE, typ=source_typ, name=source_name, idx=codegen_i))
            # - boxed values
            #   XXX: use `deserves_retbox` instead?
            elseif llvm_source_typ isa LLVM.PointerType
                @assert llvm_source_typ == codegen_typ
                push!(args, (cc=MUT_REF, typ=source_typ, name=source_name, idx=codegen_i))
            # - references to aggregates
            else
                @assert llvm_source_typ != codegen_typ
                push!(args, (cc=BITS_REF, typ=source_typ, name=source_name, idx=codegen_i))
            end
        else
            push!(args, (cc=BITS_VALUE, typ=source_typ, name=source_name, idx=codegen_i))
        end

        codegen_i += 1
    end

    return args
end

function is_immutable_datatype(T::Type)
    isa(T,DataType) && !Base.ismutabletype(T)
end

function is_inlinealloc(T::Type)
    mayinlinealloc = (T.name.flags >> 2) & 1 == true
    # FIXME: To simple
    if mayinlinealloc
        if !Base.datatype_pointerfree(T)
            t_name(dt::DataType)=dt.name
            if t_name(T).n_uninitialized != 0
                return false
            end
        end
        return true
    end
    return false
end

function is_concrete_immutable(T::Type)
    is_immutable_datatype(T) && T.layout !== C_NULL
end

function is_pointerfree(T::Type)
    if !is_immutable_datatype(T)
        return false
    end
    return Base.datatype_pointerfree(T)
end

function deserves_stack(@nospecialize(T))
    if !is_concrete_immutable(T)
        return false
    end
    return is_inlinealloc(T)
end

deserves_argbox(T) = !deserves_stack(T)
deserves_retbox(T) = deserves_argbox(T)
function deserves_sret(T, llvmT)
    @assert isa(T,DataType)
    sizeof(T) > sizeof(Ptr{Cvoid}) && !isa(llvmT, LLVM.FloatingPointType) && !isa(llvmT, LLVM.VectorType)
end


# byval lowering
#
# some back-ends don't support byval, or support it badly, so lower it eagerly ourselves
# https://reviews.llvm.org/D79744
function lower_byval(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ft = function_type(f)
    @timeit_debug to "lower byval" begin

    # classify the arguments
    args = classify_arguments(job, ft)
    filter!(args) do arg
        arg.cc != GHOST
    end

    # find the byval parameters
    byval = BitVector(undef, length(parameters(ft)))
    for i in 1:length(byval)
        attrs = collect(parameter_attributes(f, i))
        byval[i] = any(attrs) do attr
            kind(attr) == kind(TypeAttribute("byval", LLVM.VoidType()))
        end
    end

    # fixup metadata
    #
    # Julia emits invariant.load and const TBAA metadata on loads from pointer args,
    # which is invalid now that we have materialized the byval.
    for (i, param) in enumerate(parameters(f))
        if byval[i]
            # collect all uses of the argument
            worklist = Vector{LLVM.Instruction}(user.(collect(uses(param))))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = metadata(value)
                if haskey(md, LLVM.MD_invariant_load)
                    delete!(md, LLVM.MD_invariant_load)
                end
                if haskey(md, LLVM.MD_tbaa)
                    delete!(md, LLVM.MD_tbaa)
                end

                # recurse on the output of some instructions
                if isa(value, LLVM.BitCastInst) ||
                   isa(value, LLVM.GetElementPtrInst) ||
                   isa(value, LLVM.AddrSpaceCastInst)
                    append!(worklist, user.(collect(uses(value))))
                end
            end
        end
    end

    # generate the new function type & definition
    new_types = LLVM.LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        if byval[i]
            llvm_typ = convert(LLVMType, args[i].typ)
            push!(new_types, llvm_typ)
        else
            push!(new_types, param)
        end
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(ft))
            if byval[i]
                # copy the argument value to a stack slot, and reference it.
                llvm_typ = convert(LLVMType, args[i].typ)
                ptr = alloca!(builder, llvm_typ)
                if LLVM.addrspace(param) != 0
                    ptr = addrspacecast!(builder, ptr, param)
                end
                store!(builder, parameters(new_f)[i], ptr)
                push!(new_args, ptr)
            else
                push!(new_args, parameters(new_f)[i])
                for attr in collect(parameter_attributes(f, i))
                    push!(parameter_attributes(new_f, i), attr)
                end
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # fall through
        br!(builder, blocks(new_f)[2])
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    return new_f

    end
end


# kernel state arguments
#
# to facilitate passing stateful information to kernels without having to recompile, e.g.,
# the storage location for exception flags, or the location of a I/O buffer, we enable the
# back-end to specify a Julia object that will be passed to the kernel by-value, and to
# every called function by-reference. Access to this object is done using the
# `julia.gpu.state_getter` intrinsic. after optimization, these intrinsics will be lowered
# to refer to the state argument.
#
# note that we deviate from the typical Julia calling convention, by always passing the
# state objects by value instead of by reference, this to ensure that the state object
# is not copied to the stack (because LLVM doesn't see that all uses are read-only).
# in principle, `readonly byval` should be equivalent, but LLVM doesn't realize that.
# also see https://github.com/JuliaGPU/CUDA.jl/pull/1167 and the comments in that PR.
# once LLVM supports this pattern, consider going back to passing the state by reference,
# so that the julia.gpu.state_getter` can be simplified to return an opaque pointer.

# add a state argument to every function in the module, starting from the kernel entry point
function add_kernel_state!(mod::LLVM.Module)
    job = current_job::CompilerJob

    # check if we even need a kernel state argument
    state = kernel_state_type(job)
    @assert job.config.kernel
    if state === Nothing
        return false
    end
    T_state = convert(LLVMType, state)

    # intrinsic returning an opaque pointer to the kernel state.
    # this is both for extern uses, and to make this transformation a two-step process.
    state_intr = kernel_state_intr(mod, T_state)
    state_intr_ft = LLVM.FunctionType(T_state)

    kernels = []
    kernels_md = metadata(mod)["julia.kernel"]
    for kernel_md in operands(kernels_md)
        push!(kernels, Value(operands(kernel_md)[1]))
    end

    # determine which functions need a kernel state argument
    #
    # previously, we add the argument to every function and relied on unused arg elim to
    # clean-up the IR. however, some libraries do Funny Stuff, e.g., libdevice bitcasting
    # function pointers. such IR is hard to rewrite, so instead be more conservative.
    worklist = Set{LLVM.Function}([state_intr, kernels...])
    worklist_length = 0
    while worklist_length != length(worklist)
        # iteratively discover functions that use the intrinsic or any function calling it
        worklist_length = length(worklist)
        additions = LLVM.Function[]
        function check_user(val)
            if val isa Instruction
                bb = LLVM.parent(val)
                new_f = LLVM.parent(bb)
                in(new_f, worklist) || push!(additions, new_f)
            elseif val isa ConstantExpr
                # constant expressions don't have a parent; we need to look up their uses
                for use in uses(val)
                    check_user(user(use))
                end
            else
                error("Don't know how to check uses of $val. Please file an issue.")
            end
        end
        for f in worklist, use in uses(f)
            check_user(user(use))
        end
        for f in additions
            push!(worklist, f)
        end
    end
    delete!(worklist, state_intr)

    # add a state argument
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = LLVM.name(f)
        ft = function_type(f)
        LLVM.name!(f, fn * ".stateless")

        # create a new function
        new_param_types = [T_state, parameters(ft)...]
        new_ft = LLVM.FunctionType(return_type(ft), new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        LLVM.name!(parameters(new_f)[1], "state")
        linkage!(new_f, linkage(f))
        for (arg, new_arg) in zip(parameters(f), parameters(new_f)[2:end])
            LLVM.name!(new_arg, LLVM.name(arg))
        end

        workmap[f] = new_f
    end

    # clone and rewrite the function bodies, replacing uses of the old stateless function
    # with the newly created definition that includes the state argument.
    #
    # most uses are rewritten by LLVM by putting the functions in the value map.
    # a separate value materializer is used to recreate constant expressions.
    #
    # note that this only _replaces_ the uses of these functions, we'll still need to
    # _correct_ the uses (i.e. actually add the state argument) afterwards.
    function materializer(val)
        if val isa ConstantExpr
            if opcode(val) == LLVM.API.LLVMBitCast
                target = operands(val)[1]
                if target isa LLVM.Function && haskey(workmap, target)
                    # the function is being bitcasted to a different function type.
                    # we need to mutate that function type to include the state argument,
                    # or we'd be invoking the original function in an invalid way.
                    #
                    # XXX: ptrtoint/inttoptr pairs can also lose the state argument...
                    #      is all this even sound?
                    typ = value_type(val)::LLVM.PointerType
                    ft = eltype(typ)::LLVM.FunctionType
                    new_ft = LLVM.FunctionType(return_type(ft), [T_state, parameters(ft)...])
                    return const_bitcast(workmap[target], LLVM.PointerType(new_ft, addrspace(typ)))
                end
            elseif opcode(val) == LLVM.API.LLVMPtrToInt
                target = operands(val)[1]
                if target isa LLVM.Function && haskey(workmap, target)
                    return const_ptrtoint(workmap[target], value_type(val))
                end
            end
        end
        return nothing # do not claim responsibility
    end
    for (f, new_f) in workmap
        # use a value mapper for rewriting function arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(parameters(f), parameters(new_f)[2:end])
            LLVM.name!(new_param, LLVM.name(param))
            value_map[param] = new_param
        end

        # rewrite references to the old function
        merge!(value_map, workmap)

        clone_into!(new_f, f; value_map, materializer,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # remove the function IR so that we won't have any uses left after this pass.
        empty!(f)
    end

    # ensure the old (stateless) functions don't have uses anymore, and remove them
    for f in keys(workmap)
        for use in uses(f)
            val = user(use)
            if val isa ConstantExpr
                # XXX: shouldn't clone_into! remove unused CEs?
                isempty(uses(val)) || error("old function still has uses (via a constant expr)")
                LLVM.unsafe_destroy!(val)
            else
                error("old function still has uses")
            end
        end
        replace_metadata_uses!(f, workmap[f])
        unsafe_delete!(mod, f)
    end

    # update uses of the new function, modifying call sites to include the kernel state
    function rewrite_uses!(f, ft)
        # update uses
        @dispose builder=IRBuilder() begin
            for use in uses(f)
                val = user(use)
                if val isa LLVM.CallBase && called_operand(val) == f
                    # NOTE: we don't rewrite calls using Julia's jlcall calling convention,
                    #       as those have a fixed argument list, passing actual arguments
                    #       in an array of objects. that doesn't matter, for now, since
                    #       GPU back-ends don't support such calls anyhow. but if we ever
                    #       want to support kernel state passing on more capable back-ends,
                    #       we'll need to update the argument array instead.
                    if callconv(val) == 37 || callconv(val) == 38
                        # TODO: update for LLVM 15 when JuliaLang/julia#45088 is merged.
                        continue
                    end

                    # forward the state argument
                    position!(builder, val)
                    state = call!(builder, state_intr_ft, state_intr, Value[], "state")
                    new_val = if val isa LLVM.CallInst
                        call!(builder, ft, f, [state, arguments(val)...], operand_bundles(val))
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.CallBase
                    # the function is being passed as an argument. to avoid having to
                    # rewrite the target function, instead case the rewritten function to
                    # the old stateless type.
                    # XXX: we won't have to do this with opaque pointers.
                    position!(builder, val)
                    target_ft = called_type(val)
                    new_args = map(zip(parameters(target_ft),
                                       arguments(val))) do (param_typ, arg)
                        if value_type(arg) != param_typ
                            const_bitcast(arg, param_typ)
                        else
                            arg
                        end
                    end
                    new_val = call!(builder, called_type(val), called_operand(val), new_args,
                                    operand_bundles(val))
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.StoreInst
                    # the function is being stored, which again we'll permit like before.
                elseif val isa ConstantExpr
                    rewrite_uses!(val, ft)
                else
                    error("Cannot rewrite $(typeof(val)) use of function: $val")
                end
            end
        end
    end
    for f in values(workmap)
        ft = function_type(f)
        rewrite_uses!(f, ft)
    end

    return true
end
if LLVM.has_newpm()
    AddKernelStatePass() = NewPMModulePass("AddKernelStatePass", add_kernel_state!)
else
    add_kernel_state!(pm::PassManager) =
        add!(pm, ModulePass("AddKernelStatePass", add_kernel_state!))
end

# lower calls to the state getter intrinsic. this is a two-step process, so that the state
# argument can be added before optimization, and that optimization can introduce new uses
# before the intrinsic getting lowered late during optimization.
function lower_kernel_state!(fun::LLVM.Function)
    job = current_job::CompilerJob
    mod = LLVM.parent(fun)
    changed = false

    # check if we even need a kernel state argument
    state = kernel_state_type(job)
    if state === Nothing
        return false
    end

    # fixup all uses of the state getter to use the newly introduced function state argument
    if haskey(functions(mod), "julia.gpu.state_getter")
        state_intr = functions(mod)["julia.gpu.state_getter"]
        state_arg = nothing # only look-up when needed

        @dispose builder=IRBuilder() begin
            for use in uses(state_intr)
                inst = user(use)
                @assert inst isa LLVM.CallInst
                bb = LLVM.parent(inst)
                LLVM.parent(bb) == fun || continue

                position!(builder, inst)
                bb = LLVM.parent(inst)
                f = LLVM.parent(bb)

                if state_arg === nothing
                    # find the kernel state argument. this should be the first argument of
                    # the function, but only when this function needs the state!
                    state_arg = parameters(fun)[1]
                    T_state = convert(LLVMType, state)
                    @assert value_type(state_arg) == T_state
                end

                replace_uses!(inst, state_arg)

                @assert isempty(uses(inst))
                unsafe_delete!(LLVM.parent(inst), inst)

                changed = true
            end
        end
    end

    return changed
end
if LLVM.has_newpm()
    LowerKernelStatePass() = NewPMFunctionPass("LowerKernelStatePass", lower_kernel_state!)
else
    lower_kernel_state!(pm::PassManager) =
        add!(pm, FunctionPass("LowerKernelStatePass", lower_kernel_state!))
end

function cleanup_kernel_state!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false

    # remove the getter intrinsic
    if haskey(functions(mod), "julia.gpu.state_getter")
        intr = functions(mod)["julia.gpu.state_getter"]
        if isempty(uses(intr))
            # if we're not emitting a kernel, we can't resolve the intrinsic to an argument.
            unsafe_delete!(mod, intr)
            changed = true
        end
    end

    return changed
end
if LLVM.has_newpm()
    CleanupKernelStatePass() = NewPMModulePass("CleanupKernelStatePass", cleanup_kernel_state!)
else
    cleanup_kernel_state!(pm::PassManager) =
        add!(pm, ModulePass("CleanupKernelStatePass", cleanup_kernel_state!))
end

function kernel_state_intr(mod::LLVM.Module, T_state)
    state_intr = if haskey(functions(mod), "julia.gpu.state_getter")
        functions(mod)["julia.gpu.state_getter"]
    else
        LLVM.Function(mod, "julia.gpu.state_getter", LLVM.FunctionType(T_state))
    end
    push!(function_attributes(state_intr), EnumAttribute("readnone", 0))

    return state_intr
end

# run-time equivalent
function kernel_state_value(state)
    @dispose ctx=Context() begin
        T_state = convert(LLVMType, state)

        # create function
        llvm_f, _ = create_function(T_state)
        mod = LLVM.parent(llvm_f)

        # get intrinsic
        state_intr = kernel_state_intr(mod, T_state)
        state_intr_ft = function_type(state_intr)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            val = call!(builder, state_intr_ft, state_intr, Value[], "state")

            ret!(builder, val)
        end

        call_function(llvm_f, state)
    end
end
