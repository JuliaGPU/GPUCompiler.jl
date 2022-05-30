# LLVM IR generation

function irgen(@nospecialize(job::CompilerJob), method_instance::Core.MethodInstance;
               ctx::JuliaContextType)
    mod, compiled = @timeit_debug to "emission" compile_method_instance(job, method_instance; ctx)
    if job.entry_abi === :specfunc
        entry_fn = compiled[method_instance].specfunc
    else
        entry_fn = compiled[method_instance].func
    end

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            if VERSION < v"1.9.0-DEV.516"
                # only occurs in debug builds
                delete!(function_attributes(llvmf),
                        EnumAttribute("sspstrong", 0; ctx=unwrap_context(ctx)))
            end

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end

            # remove the non-specialized jfptr functions
            # TODO: Do we need to remove these?
            if job.entry_abi === :specfunc
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

    # target-specific processing
    process_module!(job, mod)
    entry = functions(mod)[entry_fn]

    # sanitize function names
    # FIXME: Julia should do this, but apparently fails (see maleadt/LLVM.jl#201)
    for f in functions(mod)
        LLVM.isintrinsic(f) && continue
        llvmfn = LLVM.name(f)
        # XXX: simplify this by only renaming definitions, not declarations?
        startswith(llvmfn, "julia.") && continue # Julia intrinsics
        startswith(llvmfn, "llvm.") && continue # unofficial LLVM intrinsics
        startswith(llvmfn, "air.") && continue # Metal AIR intrinsics
        llvmfn′ = safe_name(llvmfn)
        if llvmfn != llvmfn′
            @assert !haskey(functions(mod), llvmfn′)
            LLVM.name!(f, llvmfn′)
        end
    end

    # rename and process the entry point
    if job.source.name !== nothing
        LLVM.name!(entry, safe_name(string("julia_", job.source.name)))
    end
    if job.source.kernel
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end
    entry = process_entry!(job, mod, entry)
    if job.entry_abi === :specfunc
        func = compiled[method_instance].func
        specfunc = LLVM.name(entry)
    else
        func = LLVM.name(entry)
        specfunc = compiled[method_instance].specfunc
    end

    compiled[method_instance] =
        (; compiled[method_instance].ci, func, specfunc)

    # minimal required optimization
    @timeit_debug to "rewrite" @dispose pm=ModulePassManager() begin
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)

        # internalize all functions, but keep exported global variables
        exports = String[LLVM.name(entry)]
        for gvar in globals(mod)
            push!(exports, LLVM.name(gvar))
        end
        internalize!(pm, exports)

        # inline llvmcall bodies
        always_inliner!(pm)

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)
    end

    return mod, compiled
end


## name mangling

# we generate function names that look like C++ functions, because many NVIDIA tools
# support them, e.g., grouping different instantiations of the same kernel together.

function mangle_param(t, substitutions)
    t == Nothing && return "v"

    if isa(t, DataType) && t <: Ptr
        tn = mangle_param(eltype(t), substitutions)
        "P$tn"
    elseif isa(t, DataType) || isa(t, Core.Function)
        tn = safe_name(t)

        # handle substitutions
        sub = findfirst(isequal(tn), substitutions)
        if sub === nothing
            str = "$(length(tn))$tn"
            push!(substitutions, tn)
        elseif sub == 1
            str = "S_"
        else
            str = "S$(sub-2)_"
        end

        # encode typevars as template parameters
        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t, substitutions)
            end
            str *= "E"
        end

        str
    elseif isa(t, Integer)
        t > 0 ? "Li$(t)E" : "Lin$(abs(t))E"
    else
        tn = safe_name(t)   # TODO: actually does support digits...
        if startswith(tn, r"\d")
            # C++ classes cannot start with a digit, so mangling doesn't support it
            tn = "_$(tn)"
        end
        "$(length(tn))$tn"
    end
end

function mangle_call(f, tt)
    fn = safe_name(f)
    str = "_Z$(length(fn))$fn"

    substitutions = String[]
    for t in tt.parameters
        str *= mangle_param(t, substitutions)
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
    ctx = context(mod)
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
                @dispose builder=Builder(ctx) begin
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
    ctx = context(mod)

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
        ft = convert(LLVM.FunctionType, rt; ctx)
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
    ctx = context(mod)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    call!(builder, trap)
end


## kernel promotion

@enum ArgumentCC begin
    BITS_VALUE  # bitstype, passed as value
    BITS_REF    # bitstype, passed as pointer
    MUT_REF     # jl_value_t*, or the anonymous equivalent
    GHOST       # not passed
end

function method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

function classify_arguments(@nospecialize(job::CompilerJob), codegen_ft::LLVM.FunctionType)
    source_sig = typed_signature(job)

    source_types = [source_sig.parameters...]
    source_method = only(method_matches(typed_signature(job); job.source.world)).method
    source_arguments = method_argnames(source_method)

    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = 1
    for (source_i, source_typ) in enumerate(source_types)
        source_name = source_arguments[min(source_i, length(source_arguments))]
        # NOTE: in case of varargs, we have fewer arguments than parameters

        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc=GHOST, typ=source_typ, name=source_name))
            continue
        end

        codegen_typ = codegen_types[codegen_i]
        if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            push!(args, (cc=MUT_REF, typ=source_typ, name=source_name,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            push!(args, (cc=BITS_REF, typ=source_typ, name=source_name,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        else
            push!(args, (cc=BITS_VALUE, typ=source_typ, name=source_name,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        end
        codegen_i += 1
    end

    return args
end

function classify_fields(julia::DataType, llvm::LLVMType)
    nfields = fieldcount(julia)
    fieldoffsets = [fieldoffset(julia, i) for i in 1:nfields]
    fieldsizes = similar(fieldoffsets)
    for i in 1:nfields
        field_start = fieldoffsets[i]
        field_end = i == nfields ? sizeof(julia) : fieldoffsets[i+1]
        fieldsizes[i] = field_end - field_start
    end
    fieldsizes

    args = []
    codegen_i = 1
    for source_i in 1:nfields
        source_name = fieldname(julia, source_i)
        source_typ = fieldtype(julia, source_i)
        if fieldsizes[source_i] == 0
            push!(args, (; typ=source_typ, name=source_name))
            continue
        end

        # NOTE: a cc doesn't make sense here, so the lack of codegen field should be checked

        codegen_typ = elements(llvm)[codegen_i]
        push!(args, (typ=source_typ, name=source_name, codegen=(typ=codegen_typ, i=codegen_i)))
        codegen_i += 1
    end

    return args
end

if VERSION >= v"1.7.0-DEV.204"
    function is_immutable_datatype(T::Type)
        isa(T,DataType) && !Base.ismutabletype(T)
    end
else
    function is_immutable_datatype(T::Type)
        isa(T,DataType) && !T.mutable
    end
end

if VERSION >= v"1.7.0-DEV.204"
    function is_inlinealloc(T::Type)
        mayinlinealloc = (T.name.flags >> 2) & 1 == true
        # FIXME: To simple
        return mayinlinealloc
    end
else
    function is_inlinealloc(T::Type)
        return T.isinlinealloc
    end
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
    ctx = context(mod)
    ft = eltype(llvmtype(f))
    @compiler_assert LLVM.return_type(ft) == LLVM.VoidType(ctx) job
    @timeit_debug to "lower byval" begin

    # find the byval parameters
    byval = BitVector(undef, length(parameters(ft)))
    if LLVM.version() >= v"12"
        for i in 1:length(byval)
            attrs = collect(parameter_attributes(f, i))
            byval[i] = any(attrs) do attr
                kind(attr) == kind(EnumAttribute("byval", 0; ctx))
            end
        end
    else
        # XXX: byval is not round-trippable on LLVM < 12 (see maleadt/LLVM.jl#186)
        #      so we need to re-classify the Julia arguments.
        #      remove this once we only support 1.7.
        args = classify_arguments(job, ft)
        filter!(args) do arg
            arg.cc != GHOST
        end
        for arg in args
            if arg.cc == BITS_REF
                # NOTE: +1 since this pass runs after introducing the kernel state
                byval[arg.codegen.i] = true
            end
        end
    end

    # fixup metadata
    #
    # Julia emits invariant.load and const TBAA metadta on loads from pointer args,
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
            push!(new_types, eltype(param::LLVM.PointerType))
        else
            push!(new_types, param)
        end
    end
    new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=Builder(ctx) begin
        entry = BasicBlock(new_f, "entry"; ctx)
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(ft))
            if byval[i]
                # copy the argument value to a stack slot, and reference it.
                ptr = alloca!(builder, eltype(param))
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
    ctx = context(mod)

    # check if we even need a kernel state argument
    state = kernel_state_type(job)
    @assert job.source.kernel
    if state === Nothing
        return false
    end
    T_state = convert(LLVMType, state; ctx)

    # intrinsic returning an opaque pointer to the kernel state.
    # this is both for extern uses, and to make this transformation a two-step process.
    state_intr = kernel_state_intr(mod, T_state)

    kernels = []
    kernels_md = metadata(mod)["julia.kernel"]
    for kernel_md in operands(kernels_md)
        push!(kernels, Value(operands(kernel_md)[1]; ctx))
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
        ft = eltype(llvmtype(f))
        LLVM.name!(f, fn * ".stateless")

        # create a new function
        new_param_types = [T_state, parameters(ft)...]
        new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_param_types)
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
                    typ = llvmtype(val)::LLVM.PointerType
                    ft = eltype(typ)::LLVM.FunctionType
                    new_ft = LLVM.FunctionType(LLVM.return_type(ft), [T_state, parameters(ft)...])
                    return const_bitcast(workmap[target], LLVM.PointerType(new_ft, addrspace(typ)))
                end
            elseif opcode(val) == LLVM.API.LLVMPtrToInt
                target = operands(val)[1]
                if target isa LLVM.Function && haskey(workmap, target)
                    return const_ptrtoint(workmap[target], llvmtype(val))
                end
            end
        end
        return val
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
    function rewrite_uses!(f)
        # update uses
        @dispose builder=Builder(ctx) begin
            for use in uses(f)
                val = user(use)
                if val isa LLVM.CallBase && called_value(val) == f
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
                    state = call!(builder, state_intr, Value[], "state")
                    new_val = if val isa LLVM.CallInst
                        call!(builder, f, [state, arguments(val)...], operand_bundles(val))
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.CallBase
                    # the function is being passed as an argument, which we'll just permit,
                    # because we expect to have rewritten the call down the line separately.
                elseif val isa LLVM.StoreInst
                    # the function is being stored, which again we'll permit like before.
                elseif val isa ConstantExpr
                    rewrite_uses!(val)
                else
                    error("Cannot rewrite $(typeof(val)) use of function: $val")
                end
            end
        end
    end
    for f in values(workmap)
        rewrite_uses!(f)
    end

    return true
end

# lower calls to the state getter intrinsic. this is a two-step process, so that the state
# argument can be added before optimization, and that optimization can introduce new uses
# before the intrinsic getting lowered late during optimization.
function lower_kernel_state!(fun::LLVM.Function)
    job = current_job::CompilerJob
    mod = LLVM.parent(fun)
    ctx = context(fun)
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

        @dispose builder=Builder(ctx) begin
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
                    T_state = convert(LLVMType, state; ctx)
                    @assert llvmtype(state_arg) == T_state
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

function cleanup_kernel_state!(mod::LLVM.Module)
    job = current_job::CompilerJob
    ctx = context(mod)
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

function kernel_state_intr(mod::LLVM.Module, T_state)
    ctx = context(mod)

    state_intr = if haskey(functions(mod), "julia.gpu.state_getter")
        functions(mod)["julia.gpu.state_getter"]
    else
        LLVM.Function(mod, "julia.gpu.state_getter", LLVM.FunctionType(T_state))
    end
    push!(function_attributes(state_intr), EnumAttribute("readnone", 0; ctx))

    return state_intr
end

# run-time equivalent
function kernel_state_value(state)
    @dispose ctx=Context() begin
        T_state = convert(LLVMType, state; ctx)

        # create function
        llvm_f, _ = create_function(T_state)
        mod = LLVM.parent(llvm_f)

        # get intrinsic
        state_intr = kernel_state_intr(mod, T_state)

        # generate IR
        @dispose builder=Builder(ctx) begin
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            val = call!(builder, state_intr, Value[], "state")

            ret!(builder, val)
        end

        call_function(llvm_f, state)
    end
end
