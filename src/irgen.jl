# LLVM IR generation

function irgen(@nospecialize(job::CompilerJob))
    mod, compiled, gv_to_value = @tracepoint "emission" compile_method_instance(job)
    if job.config.entry_abi === :specfunc
        entry_fn = compiled[job.source].specfunc
    else
        entry_fn = compiled[job.source].func
    end
    @assert entry_fn !== nothing
    entry = mod.functions[entry_fn]

    # clean up incompatibilities
    @tracepoint "clean-up" begin
        for llvmf in mod.functions
            if Base.isdebugbuild()
                # only occurs in debug builds
                delete!(llvmf.function_attributes,
                        EnumAttribute("sspstrong", 0))
            end

            delete!(llvmf.function_attributes,
                    StringAttribute("probe-stack", "inline-asm"))

            if Sys.iswindows()
                llvmf.personality = nothing
            end

            # remove the non-specialized jfptr functions
            # TODO: Do we need to remove these?
            if job.config.entry_abi === :specfunc
                if startswith(llvmf.name, "jfptr_")
                    erase!(llvmf)
                end
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && haskey(mod.functions, "__julia_personality")
            llvmf = mod.functions["__julia_personality"]
            @compiler_assert isempty(llvmf.uses) job
            erase!(llvmf)
        end
    end

    deprecation_marker = process_module!(job, mod)
    if deprecation_marker != DeprecationMarker()
        safe_depwarn("GPUCompiler.process_module! is deprecated; implement GPUCompiler.finish_module! instead", :process_module)
    end

    # sanitize global values (Julia doesn't when using the external codegen policy)
    for val in [collect(mod.globals); collect(mod.functions)]
        isdeclaration(val) && continue
        old_name = val.name
        new_name = safe_name(old_name)
        if old_name != new_name
            val.name = new_name
            val = get(gv_to_value, old_name, nothing)
            if val !== nothing
                delete!(gv_to_value, old_name)
                gv_to_value[new_name] = val
            end
        end
    end

    # rename and process the entry point
    if job.config.name !== nothing
        entry.name = safe_name(job.config.name)
    elseif job.config.kernel
        entry.name = mangle_sig(job.source.specTypes)
    end
    deprecation_marker = process_entry!(job, mod, entry)
    if deprecation_marker != DeprecationMarker()
        safe_depwarn("GPUCompiler.process_entry! is deprecated; implement GPUCompiler.finish_module! instead", :process_entry)
        entry = deprecation_marker
    end
    if job.config.entry_abi === :specfunc
        func = compiled[job.source].func
        specfunc = entry.name
    else
        func = entry.name
        specfunc = compiled[job.source].specfunc
    end

    compiled[job.source] =
        (; compiled[job.source].ci, func, specfunc)

    # minimal required optimization
    @tracepoint "rewrite" begin
        if job.config.kernel && pass_by_value(job)
            # pass all bitstypes by value; by default Julia passes aggregates by reference
            # (this improves performance, and is mandated by certain back-ends like SPIR-V).
            args = classify_arguments(job, entry.function_type)
            for arg in args
                if arg.cc == BITS_REF
                    llvm_typ = convert(LLVMType, arg.typ)
                    if pass_by_ref(job)
                        attr = TypeAttribute("byref", llvm_typ)
                    else
                        attr = TypeAttribute("byval", llvm_typ)
                    end
                    push!(entry.parameter_attributes[arg.idx], attr)
                end
            end
        end

        # back-end-provided runtime stubs (`Runtime.compile(:sym, ...)`) carry
        # a *weak* `gpu_<name>` body so CPU-AOT pipelines (juliac, sysimage,
        # PrecompileTools) can satisfy CPU symbol resolution. On the GPU path
        # we want the back-end's strong definition from the runtime library to
        # take over, but `InternalizePass` below would convert the weak body
        # to `internal` and `link!(...; only_needed=true)` would then refuse
        # to import the strong override (`Linker::linkIfNeeded` requires the
        # destination to be a true declaration, not merely weak). Erase the
        # stub bodies so each becomes a plain external declaration; the
        # runtime library's strong def is then linked in normally.
        for method in values(Runtime.methods)
            method.def isa Symbol || continue
            haskey(mod.functions, method.llvm_name) || continue
            f = mod.functions[method.llvm_name]
            isdeclaration(f) && continue
            empty!(f)
            f.linkage = LLVM.API.LLVMExternalLinkage
        end

        # internalize all functions and, but keep exported global variables.
        entry.linkage = LLVM.API.LLVMExternalLinkage
        preserved_gvs = String[entry.name]
        for gvar in mod.globals
            push!(preserved_gvs, gvar.name)
        end
        @dispose pb=PassBuilder() begin
            add!(pb, InternalizePass(; preserved_gvs))
            add!(pb, AlwaysInlinerPass())
            run!(pb, mod, llvm_machine(job.config.target))
        end

        can_throw(job) || lower_throw!(job, mod)

        # resolve the `julia.gpu.debug_level` intrinsic (see `kernel_debug_level_value`) to
        # the job's configured level, so device code can branch on it as a compile-time
        # constant that is part of the cache key (unlike reading the `-g` global directly).
        lower_debug_level!(job, mod)

        # materialize `GPUCompiler.alloca` intrinsics as real entry-block allocas, before the
        # optimizer runs so the slots can be promoted (see `lower_alloca!`).
        lower_alloca!(job, mod)
    end

    return mod, compiled, gv_to_value
end


## exception handling

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false
    @tracepoint "lower throw" begin

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

    # Julia's codegen replaces an `llvmcall` of an intrinsic it doesn't know (e.g. one
    # removed from LLVM) by a run-time `jl_error`. report those at compile time instead.
    errors = IRError[]

    for f in mod.functions
        fn = f.name
        for (throw_fn, name) in throw_functions
            occursin(throw_fn, fn) || continue

            for use in collect(f.uses)
                call = use.user::LLVM.CallInst
                if is_unknown_intrinsic_error(call)
                    push!(errors, (UNKNOWN_INTRINSIC, backtrace(call), nothing))
                end

                # replace the throw with a PTX-compatible exception
                @dispose builder=IRBuilder() begin
                    position!(builder, LLVM.before(call))
                    emit_exception!(job, builder, name, call)
                end

                # remove the call (collecting its arguments first, as the view is live)
                call_args = collect(call.arguments)
                erase!(call)

                # HACK: kill the exceptions' unused arguments
                #       this is needed for throwing objects with @nospecialize constructors.
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(cast.operands)
                        isempty(cast.uses) && erase!(cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(arg.uses)
                        erase!(arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(f.uses) job
            break
         end
     end

    end
    isempty(errors) || throw(InvalidIRError(job, errors))
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` is emitted. on debug
# level 1, the exception name is printed, and on debug level 2 the individual stack frames (as
# recovered from the LLVM debug information) are printed as well.
#
# the `trap` here is *not* the final lowering of the exception: some targets cannot tolerate a
# hardware trap (on Apple M1 compute a `trap` wedges the whole GPU, JuliaGPU/Metal.jl#433; and
# SPIR-V/PoCL have no abort), so those backends strip it post-optimization in
# `lower_unreachable_control_flow!` and let the lane exit via a clean `ret`. the trap must
# nonetheless survive through `optimize!`: it is `noreturn`, and that is what stops InstCombine's
# `removeInstructionsBeforeUnreachable` (which erases instructions preceding an `unreachable`
# while `!mayThrow() && willReturn()`) from deleting the `signal_exception` call below and
# folding away the guarding bounds-check branch. so the trap is the optimizer-correctness guard;
# do not move its removal earlier than post-`optimize!`.
function emit_exception!(@nospecialize(job::CompilerJob), builder, name, inst)
    bb = builder.insert_block
    fun = bb.parent
    mod = fun.parent

    # report the exception
    if job.config.debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if job.config.debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if job.config.debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        ft = convert(LLVM.FunctionType, rt)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(ft.parameters[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(ft.parameters[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception to the host (backend-specific: writes a `KernelState` mailbox).
    # the host reads this mailbox after synchronizing.
    call!(builder, Runtime.get(:signal_exception))

    emit_trap!(job, builder, mod, inst)
end

function emit_trap!(@nospecialize(job::CompilerJob), builder, mod, inst)
    trap_ft = LLVM.FunctionType(LLVM.VoidType())
    trap = if haskey(mod.functions, "llvm.trap")
        mod.functions["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", trap_ft)
    end
    call!(builder, trap_ft, trap)
end


## unreachable control flow handling

# check if a function contains unreachable control flow
# (`unreachable` terminator or `trap` call)
function has_unreachable_control_flow(f::LLVM.Function)
    for bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.UnreachableInst)
            return true
        end
        if isa(inst, LLVM.CallInst)
            callee = inst.called_operand
            if isa(callee, LLVM.Function) && callee.name == "llvm.trap"
                return true
            end
        end
    end
    return false
end

# force-inline every function with unreachable control flow into kernels, so that
# `lower_unreachable_control_flow!` can rewrite it into a `ret` soundly.
#
# this is a fixpoint iteration based on `has_unreachable_control_flow`: each round marks the
# functions that currently contain unreachable control flow and inlines them, which exposes it
# in their callers, until it has all been hoisted up into the kernels. this naturally handles
# the `kernel → A → B` case where only `B` traps: `A` is marked once `B` is inlined into it,
# without us having to reason about call-graph paths.
function inline_unreachable_control_flow!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false
    alwaysinline_attr = EnumAttribute("alwaysinline", 0)
    noinline_attr = EnumAttribute("noinline", 0)
    kernel_fns = kernels(mod)

    @tracepoint "inline unreachable control flow" begin
    while true
        marked = false
        for f in mod.functions
            isdeclaration(f) && continue
            # never inline a kernel, and don't bother marking a function with no call sites
            # (the inliner can't inline it anyway).
            (f in kernel_fns || isempty(f.uses)) && continue
            attrs = f.function_attributes
            alwaysinline_attr in collect(attrs) && continue
            has_unreachable_control_flow(f) || continue

            delete!(attrs, noinline_attr)
            push!(attrs, alwaysinline_attr)
            marked = true
        end
        marked || break

        @dispose pb=PassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            run!(pb, mod, llvm_machine(job.config.target))
        end
        changed = true
    end
    end

    return changed
end

# demote unordered atomic loads and stores to plain ones
#
# Julia marks accesses to heap references `unordered` so that a read racing with the GC, or
# with another thread's write, cannot observe a torn pointer. There is no device GC and no
# such race for GPUCompiler to protect against, so the ordering carries no meaning here, but
# not every back-end can express it: SPIR-V's OpAtomicLoad/OpAtomicStore only take scalar
# integer or floating-point operands, so the Khronos translator turns an `unordered` load of a
# pointer into an invalid pointer-typed atomic that consumers reject (Intel's compiler fails
# with an undefined `__spirv_AtomicLoad(long**, int, int)`), and AIR has no atomic load or
# store instructions at all. Run after optimization, where dropping the ordering cannot
# enable new transformations; stronger orderings are left intact.
function demote_unordered_atomics!(mod::LLVM.Module)
    changed = false
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        (inst isa LLVM.LoadInst || inst isa LLVM.StoreInst) || continue
        isatomic(inst) && inst.ordering == LLVM.API.LLVMAtomicOrderingUnordered || continue
        inst.ordering = LLVM.API.LLVMAtomicOrderingNotAtomic
        changed = true
    end
    return changed
end

# lower `trap` to a clean return to get rid of `unreachable` and `noreturn`
#
# this is for compatibility with back-ends that don't support (SPIR-V) or have
# problems with `trap` (Metal on Apple M1 and M2). note that the rewrite is not
# entirely correct: barriers may deadlock if a participating lane has exited.
# however, it's generally not possible to do better without hardware support.
function lower_unreachable_control_flow!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false
    @tracepoint "lower unreachable control flow" begin

    # the rewrite below only makes sense in a kernel: a function whose `ret` exits to the host rather
    # than to a caller. kernels are the only thing we emit, and their top-level `ret` is what we rely
    # on here; everything else is a callee that the inlining below folds into its kernel(s).

    # hoist every throwing function up into its kernel(s) first, so that each `unreachable` we
    # rewrite below belongs to a kernel whose `ret` is a genuine exit (see the comment above).
    changed |= inline_unreachable_control_flow!(job, mod)

    # defensively drop any dead leftovers before the back-end sees them. `AlwaysInlinerPass` already
    # erases the throwing helpers it fully inlines (they are `internal`, hence discardable), so in
    # practice this is a no-op; it is here only to catch dead remnants of partial inlining, since the
    # regular `cleanup` DCE ran before `finish_ir!` and won't see anything produced above.
    @dispose pb=PassBuilder() begin
        add!(pb, GlobalDCEPass())
        run!(pb, mod, llvm_machine(job.config.target))
    end

    # lower the unreachable control flow, but *only* in the kernels: there, turning an `unreachable`
    # into a `ret` is a genuine exit. we deliberately do not touch any other function: one that still
    # contains `unreachable`/`trap` after the inlining above is one we couldn't hoist into a kernel
    # (recursive or address-taken throwing code), and rewriting its `unreachable` into a `ret` would
    # silently resume execution in the caller instead of exiting. we leave it as-is — keeping its
    # `trap`/`unreachable`, which the back-end may reject, but that honestly surfaces an unsupported
    # construct instead of quietly miscompiling it — and warn.
    kernel_fns = kernels(mod)
    for f in mod.functions
        isdeclaration(f) && continue
        if f in kernel_fns
            changed |= lower_unreachable_control_flow!(f)
        elseif has_unreachable_control_flow(f) && !isempty(f.uses)
            @safe_warn "Cannot lower unreachable control flow in '$(f.name)': it has callers but could not be inlined into a kernel (it is likely recursive or address-taken). Leaving its trap/unreachable in place; this may not be supported by the back-end."
        end
    end

    # scrub every `noreturn` attribute (functions *and* call sites), module-wide. after the rewrite
    # above the entry points no longer trap or run off into `unreachable`, but `noreturn` is a
    # cached fact that outlives the instructions it was derived from — and a stale `noreturn` lets
    # a trusting back-end (Metal's AIR optimizer, the SPIR-V translator) re-derive an
    # `unreachable`/`OpUnreachable`/trap right after the call and undo our work. we do this here,
    # not per-function, to also reach functions the rewrite skipped: `noreturn` declarations the
    # kernel calls, and genuinely-`noreturn` functions (e.g. infinite loops) we left out-of-line.
    # dropping it is always safe — it only relaxes an optimization hint; the back-end may re-infer
    # it on a function that really never returns, but with no trap to reconstruct that is harmless.
    noreturn_attr = EnumAttribute("noreturn", 0)
    for f in mod.functions
        delete!(f.function_attributes, noreturn_attr)
        for bb in f.blocks, inst in bb.instructions
            isa(inst, LLVM.CallInst) && delete!(inst.function_attributes, noreturn_attr)
        end
    end

    # erase the now-unused `llvm.trap` declaration. guarded by `isempty(uses(...))` so we only
    # ever drop it when the calls above are gone (other backends create their own `llvm.trap`
    # and never invoke this pass, so theirs is untouched).
    if haskey(mod.functions, "llvm.trap")
        trap = mod.functions["llvm.trap"]
        if isempty(trap.uses)
            erase!(trap)
            changed = true
        end
    end

    end
    return changed
end

function lower_unreachable_control_flow!(f::LLVM.Function)
    changed = false

    # Pass 1: strip every `llvm.trap` call, regardless of shape.
    for bb in f.blocks, inst in collect(bb.instructions)
        if isa(inst, LLVM.CallInst)
            callee = inst.called_operand
            if isa(callee, LLVM.Function) && callee.name == "llvm.trap"
                erase!(inst)
                changed = true
            end
        end
    end

    # Pass 2: lower every `unreachable` terminator to a branch to a return
    # block. this also covers `unreachable` not preceded by a trap.
    unreachables = Instruction[]
    exit_blocks = BasicBlock[]
    for bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.UnreachableInst)
            push!(unreachables, inst)
        end
        if isa(inst, LLVM.RetInst)
            push!(exit_blocks, bb)
        end
    end
    isempty(unreachables) && return changed

    @dispose builder=IRBuilder() begin
        local return_block
        if isempty(exit_blocks)
            # the function has no normal return (e.g. a kernel whose only path is a `throw`).
            # synthesize a return block so we can turn the `unreachable` into a clean return.
            return_block = BasicBlock(f, "ret")
            position!(builder, LLVM.at_end(return_block))
            rt = f.function_type.return_type
            if rt == LLVM.VoidType()
                ret!(builder)
            else
                ret!(builder, UndefValue(rt))
            end
        else
            # if we have multiple exit blocks, take the last one, which is hopefully the least
            # divergent (assuming divergent control flow is the root of the problem here).
            exit_block = last(exit_blocks)
            ret = exit_block.terminator

            # create a return block with only the return instruction, so that we only have to
            # care about any values returned, and not about any other SSA value in the block.
            if first(exit_block.instructions) == ret
                # we can reuse the exit block if it only contains the return
                return_block = exit_block
            else
                # split the exit block right before the ret
                return_block = BasicBlock(f, "ret")
                move!(return_block, LLVM.after(exit_block))

                # emit a branch
                position!(builder, LLVM.before(ret))
                br!(builder, return_block)

                # move the return
                remove!(ret)
                position!(builder, LLVM.at_end(return_block))
                move!(ret, builder.position)
            end

            # when returning a value, add a phi node to the return block, so that we can later
            # add incoming undef values when branching from `unreachable` blocks
            if !isempty(ret.operands)
                position!(builder, LLVM.before(ret))
                # XXX: support aggregate returns?
                val = only(ret.operands)
                phi = phi!(builder, val.value_type)
                for pred in return_block.predecessors
                    push!(phi.incoming, (val, pred))
                end
                ret.operands[1] = phi
            end
        end

        # replace the unreachable with a branch to the return block
        for unreachable in unreachables
            bb = unreachable.parent

            position!(builder, LLVM.before(unreachable))
            br!(builder, return_block)
            erase!(unreachable)

            # patch up any phi nodes in the return block
            for inst in return_block.instructions
                if isa(inst, LLVM.PHIInst)
                    undef = UndefValue(inst.value_type)
                    vals = inst.incoming
                    push!(vals, (undef, bb))
                end
            end
        end
    end

    return true
end


## kernel promotion

@enum ArgumentCC begin
    BITS_VALUE      # bitstype, passed as value
    BITS_REF        # bitstype, passed as pointer
    MUT_REF         # jl_value_t*, or the anonymous equivalent
    GHOST           # not passed
    KERNEL_STATE    # the kernel state argument
end

# Determine the calling convention of a the arguments of a Julia function, given the
# LLVM function type as generated by the Julia code generator. Returns an vector with one
# element for each Julia-level argument, containing a tuple with the following fields:
# - `cc`: the calling convention of the argument
# - `typ`: the Julia type of the argument
# - `name`: the name of the argument
# - `idx`: the index of the argument in the LLVM function type, or `nothing` if the argument
#          is not passed at the LLVM level.
function classify_arguments(@nospecialize(job::CompilerJob), codegen_ft::LLVM.FunctionType;
                            post_optimization::Bool=false)
    source_sig = job.source.specTypes
    source_types = [source_sig.parameters...]

    source_argnames = Base.method_argnames(job.source.def)
    while length(source_argnames) < length(source_types)
        # this is probably due to a trailing vararg; repeat its name
        push!(source_argnames, source_argnames[end])
    end

    codegen_types = codegen_ft.parameters

    if post_optimization && kernel_state_type(job) !== Nothing
        args = []
        push!(args, (cc=KERNEL_STATE, typ=kernel_state_type(job), name=:kernel_state, idx=1))
        codegen_i = 2
    else
        args = []
        codegen_i = 1
    end
    for (source_typ, source_name) in zip(source_types, source_argnames)
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
    ft = f.function_type
    @tracepoint "lower byval" begin

    # find the byval parameters
    byval = BitVector(undef, length(ft.parameters))
    types = Vector{LLVMType}(undef, length(ft.parameters))
    for i in 1:length(byval)
        byval[i] = false
        for attr in collect(f.parameter_attributes[i])
            if attr.kind == :byval
                byval[i] = true
                types[i] = attr.value
            end
        end
    end

    # fixup metadata
    #
    # Julia tags loads from by-pointer arguments with const-region metadata (`!tbaa jtbaa_const`,
    # `!invariant.load`, and the `jnoalias_const`/`jnoalias_stack` `!alias.scope`/`!noalias`
    # package). Materializing the byval as a caller stack slot below makes that metadata false, so
    # we strip it (JuliaLang/julia#44285).
    #
    # We cannot defer this to Julia the way the inliner does. `CleanupIR` runs back in `optimize!`,
    # while this is still a real by-pointer const argument whose metadata is true; the
    # materialization that breaks it happens here, after the last Julia pass, so nothing downstream
    # repairs it. `CleanupIR` also never touches `!alias.scope`/`!noalias`. The inliner copes with
    # that because it clones alias scopes per instance, but `clone_into!` below copies the metadata
    # verbatim, so we strip the whole package ourselves.
    for (i, param) in enumerate(f.parameters)
        byval[i] && strip_julia_const_region_metadata_from_derived_uses!(param)
    end

    # generate the new function type & definition
    new_types = LLVM.LLVMType[]
    for (i, param) in enumerate(ft.parameters)
        if byval[i]
            llvm_typ = convert(LLVMType, types[i])
            push!(new_types, llvm_typ)
        else
            push!(new_types, param)
        end
    end
    new_ft = LLVM.FunctionType(ft.return_type, new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    for (arg, new_arg) in zip(f.parameters, new_f.parameters)
        new_arg.name = arg.name
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, LLVM.at_end(entry))

        # perform argument conversions
        for (i, param) in enumerate(ft.parameters)
            if byval[i]
                # copy the argument value to a stack slot, and reference it.
                llvm_typ = convert(LLVMType, types[i])
                ptr = alloca!(builder, llvm_typ)
                if param.addrspace != 0
                    ptr = addrspacecast!(builder, ptr, param)
                end
                store!(builder, new_f.parameters[i], ptr)
                push!(new_args, ptr)
            else
                push!(new_args, new_f.parameters[i])
                for attr in collect(f.parameter_attributes[i])
                    push!(new_f.parameter_attributes[i], attr)
                end
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(f.parameters)
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # fall through
        br!(builder, new_f.blocks[2])
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = f.name
    @assert isempty(f.uses)
    replace_metadata_uses!(f, new_f)
    erase!(f)
    new_f.name = fn

    return new_f

    end
end

const JuliaConstRegionMetadataKinds =
    (LLVM.MD_invariant_load, LLVM.MD_tbaa, LLVM.MD_tbaa_struct,
     LLVM.MD_alias_scope, LLVM.MD_noalias)

function strip_julia_const_region_metadata!(inst::LLVM.Instruction)
    changed = false
    md = inst.metadata
    for kind in JuliaConstRegionMetadataKinds
        if haskey(md, kind)
            delete!(md, kind)
            changed = true
        end
    end
    return changed
end

function is_pointer_derivation_inst(v)
    return v isa LLVM.BitCastInst ||
           v isa LLVM.GetElementPtrInst ||
           v isa LLVM.AddrSpaceCastInst
end

function strip_julia_const_region_metadata_from_derived_uses!(root)
    changed = false
    seen = Base.IdSet{LLVM.Value}()  # `IdSet` is not visible unqualified on Julia 1.10
    worklist = Vector{LLVM.Instruction}(collect(root.users))
    while !isempty(worklist)
        inst = popfirst!(worklist)
        inst in seen && continue
        push!(seen, inst)

        changed |= strip_julia_const_region_metadata!(inst)

        is_pointer_derivation_inst(inst) || continue
        append!(worklist, collect(inst.users))
    end
    return changed
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
struct AddKernelState
    job::CompilerJob
end
function (self::AddKernelState)(mod::LLVM.Module)
    # check if we even need a kernel state argument
    self.job.config.kernel || return false
    state = kernel_state_type(self.job)
    if state === Nothing
        return false
    end
    T_state = convert(LLVMType, state)

    # intrinsic returning an opaque pointer to the kernel state.
    # this is both for extern uses, and to make this transformation a two-step process.
    state_intr = kernel_state_intr(mod, T_state)
    state_intr_ft = LLVM.FunctionType(T_state)

    # determine which functions need a kernel state argument
    #
    # previously, we add the argument to every function and relied on unused arg elim to
    # clean-up the IR. however, some libraries do Funny Stuff, e.g., libdevice bitcasting
    # function pointers. such IR is hard to rewrite, so instead be more conservative.
    worklist = Set{LLVM.Function}([state_intr, kernels(mod)...])
    worklist_length = 0
    while worklist_length != length(worklist)
        # iteratively discover functions that use the intrinsic or any function calling it
        worklist_length = length(worklist)
        additions = LLVM.Function[]
        function check_user(val)
            if val isa Instruction
                bb = val.parent
                new_f = bb.parent
                in(new_f, worklist) || push!(additions, new_f)
            elseif val isa ConstantExpr
                # constant expressions don't have a parent; we need to look up their uses
                for use in val.uses
                    check_user(use.user)
                end
            else
                error("Don't know how to check uses of $val. Please file an issue.")
            end
        end
        for f in worklist, use in f.uses
            check_user(use.user)
        end
        for f in additions
            push!(worklist, f)
        end
    end
    delete!(worklist, state_intr)

    # add a state argument
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = f.name
        ft = f.function_type
        f.name = fn * ".stateless"

        # create a new function
        new_param_types = [T_state, ft.parameters...]
        new_ft = LLVM.FunctionType(ft.return_type, new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        new_f.parameters[1].name = "state"
        new_f.linkage = f.linkage
        for (arg, new_arg) in zip(f.parameters, new_f.parameters[2:end])
            new_arg.name = arg.name
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
            if val.opcode == LLVM.API.LLVMBitCast
                target = val.operands[1]
                if target isa LLVM.Function && haskey(workmap, target)
                    # the function is being bitcasted to a different function type.
                    # we need to mutate that function type to include the state argument,
                    # or we'd be invoking the original function in an invalid way.
                    #
                    # XXX: ptrtoint/inttoptr pairs can also lose the state argument...
                    #      is all this even sound?
                    typ = val.value_type::LLVM.PointerType
                    ft = typ.element_type::LLVM.FunctionType
                    new_ft = LLVM.FunctionType(ft.return_type, [T_state, ft.parameters...])
                    return const_bitcast(workmap[target], LLVM.PointerType(new_ft, typ.addrspace))
                end
            elseif val.opcode == LLVM.API.LLVMPtrToInt
                target = val.operands[1]
                if target isa LLVM.Function && haskey(workmap, target)
                    return const_ptrtoint(workmap[target], val.value_type)
                end
            end
        end
        return nothing # do not claim responsibility
    end
    for (f, new_f) in workmap
        # use a value mapper for rewriting function arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(f.parameters, new_f.parameters[2:end])
            new_param.name = param.name
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
        prune_constexpr_uses!(f)
        @assert isempty(f.uses)
        replace_metadata_uses!(f, workmap[f])
        erase!(f)
    end

    # update uses of the new function, modifying call sites to include the kernel state
    function rewrite_uses!(f, ft)
        # update uses
        @dispose builder=IRBuilder() begin
            for use in collect(f.uses)
                val = use.user
                if val isa LLVM.CallBase && val.called_operand == f
                    # NOTE: we don't rewrite calls using Julia's jlcall calling convention,
                    #       as those have a fixed argument list, passing actual arguments
                    #       in an array of objects. that doesn't matter, for now, since
                    #       GPU back-ends don't support such calls anyhow. but if we ever
                    #       want to support kernel state passing on more capable back-ends,
                    #       we'll need to update the argument array instead.
                    if val.callconv == 37 || val.callconv == 38
                        # TODO: update for LLVM 15 when JuliaLang/julia#45088 is merged.
                        continue
                    end

                    # forward the state argument
                    position!(builder, LLVM.before(val))
                    state = call!(builder, state_intr_ft, state_intr, Value[], "state")
                    new_val = if val isa LLVM.CallInst
                        call!(builder, ft, f, [state, val.arguments...], val.operand_bundles)
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    new_val.callconv = val.callconv

                    replace_uses!(val, new_val)
                    @assert isempty(val.uses)
                    erase!(val)
                elseif val isa LLVM.CallBase
                    # the function is being passed as an argument. to avoid having to
                    # rewrite the target function, instead case the rewritten function to
                    # the old stateless type.
                    # XXX: we won't have to do this with opaque pointers.
                    position!(builder, LLVM.before(val))
                    target_ft = val.called_type
                    new_args = map(zip(target_ft.parameters,
                                       val.arguments)) do (param_typ, arg)
                        if arg.value_type != param_typ
                            const_bitcast(arg, param_typ)
                        else
                            arg
                        end
                    end
                    new_val = call!(builder, val.called_type, val.called_operand, new_args,
                                    val.operand_bundles)
                    new_val.callconv = val.callconv

                    replace_uses!(val, new_val)
                    @assert isempty(val.uses)
                    erase!(val)
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
        ft = f.function_type
        rewrite_uses!(f, ft)
    end

    return true
end
AddKernelStatePass(job) = ModulePass("AddKernelStatePass", AddKernelState(job))

# lower calls to the state getter intrinsic. this is a two-step process, so that the state
# argument can be added before optimization, and that optimization can introduce new uses
# before the intrinsic getting lowered late during optimization.
struct LowerKernelState
    job::CompilerJob
end
function (self::LowerKernelState)(fun::LLVM.Function)
    mod = fun.parent
    changed = false

    # check if we even need a kernel state argument
    state = kernel_state_type(self.job)
    if state === Nothing
        return false
    end

    # fixup all uses of the state getter to use the newly introduced function state argument
    if haskey(mod.functions, "julia.gpu.state_getter")
        state_intr = mod.functions["julia.gpu.state_getter"]
        state_arg = nothing # only look-up when needed

        @dispose builder=IRBuilder() begin
            for use in collect(state_intr.uses)
                inst = use.user
                @assert inst isa LLVM.CallInst
                bb = inst.parent
                bb.parent == fun || continue

                position!(builder, LLVM.before(inst))
                bb = inst.parent
                f = bb.parent

                if state_arg === nothing
                    # find the kernel state argument. this should be the first argument of
                    # the function, but only when this function needs the state!
                    params = fun.parameters
                    if isempty(params)
                        # `add_kernel_state!` should have given every function that uses the
                        # state intrinsic a state argument. if it didn't, fail with a clear
                        # message (naming the offending function) instead of an opaque
                        # `BoundsError`, so the bug is diagnosable from the error alone.
                        error("""kernel state lowering: function `$(fun.name)` uses the \
                                 kernel state intrinsic but was not given a state argument. \
                                 This is a GPUCompiler bug; please file an issue.""")
                    end
                    state_arg = params[1]
                    T_state = convert(LLVMType, state)
                    @assert state_arg.value_type == T_state
                end

                replace_uses!(inst, state_arg)

                @assert isempty(inst.uses)
                erase!(inst)

                changed = true
            end
        end
    end

    return changed
end
LowerKernelStatePass(job) = FunctionPass("LowerKernelStatePass", LowerKernelState(job))

struct CleanupKernelState
    job::CompilerJob
end
function (self::CleanupKernelState)(mod::LLVM.Module)
    changed = false

    # remove the getter intrinsic
    if haskey(mod.functions, "julia.gpu.state_getter")
        intr = mod.functions["julia.gpu.state_getter"]
        if isempty(intr.uses)
            # if we're not emitting a kernel, we can't resolve the intrinsic to an argument.
            erase!(intr)
            changed = true
        end
    end

    return changed
end
CleanupKernelStatePass(job) = ModulePass("CleanupKernelStatePass", CleanupKernelState(job))

function kernel_state_intr(mod::LLVM.Module, T_state)
    state_intr = if haskey(mod.functions, "julia.gpu.state_getter")
        mod.functions["julia.gpu.state_getter"]
    else
        LLVM.Function(mod, "julia.gpu.state_getter", LLVM.FunctionType(T_state))
    end
    push!(state_intr.function_attributes, EnumAttribute("readnone", 0))

    return state_intr
end

# run-time equivalent
kernel_state_value(state) = generate_llvmcall(state, Tuple{}) do builder
    T_state = convert(LLVMType, state)
    state_intr = kernel_state_intr(current_module(builder), T_state)
    call!(builder, state_intr.function_type, state_intr, Value[], "state")
end


## debug level

# device code can query the job's configured debug level as a compile-time constant via
# `kernel_debug_level()`, which emits the `julia.gpu.debug_level` intrinsic; `lower_debug_level!`
# (run from `irgen`, with `job` in scope) replaces it with the constant. this keeps
# the level part of the cache key (it lives in `CompilerConfig`), unlike reading the `-g`
# global at parse time (which would bake the wrong level under pkgimage reuse across `-g`).

function debug_level_intr(mod::LLVM.Module)
    intr = if haskey(mod.functions, "julia.gpu.debug_level")
        mod.functions["julia.gpu.debug_level"]
    else
        LLVM.Function(mod, "julia.gpu.debug_level", LLVM.FunctionType(LLVM.Int32Type()))
    end
    push!(intr.function_attributes, EnumAttribute("readnone", 0))

    return intr
end

# run-time equivalent: emits a call to the debug-level intrinsic, returning the job's
# configured `debug_level` as an `Int32` (lowered to a constant by `lower_debug_level!`).
kernel_debug_level_value() = generate_llvmcall(Int32, Tuple{}) do builder
    intr = debug_level_intr(current_module(builder))
    call!(builder, intr.function_type, intr, Value[], "debug_level")
end

# device-facing accessor: the compiling job's debug level as an `Int32` compile-time constant.
# exported for back-end device runtimes (e.g. to gate exception reporting); it takes no
# back-end-specific argument, so unlike `kernel_state` there's no need for a per-back-end
# definition. not intended for user code.
@inline @generated kernel_debug_level() = kernel_debug_level_value()
export kernel_debug_level

# replace every `julia.gpu.debug_level` call with the job's configured level
function lower_debug_level!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    haskey(mod.functions, "julia.gpu.debug_level") || return false

    intr = mod.functions["julia.gpu.debug_level"]
    level = ConstantInt(LLVM.Int32Type(), job.config.debug_level)
    for use in collect(intr.uses)
        inst = use.user
        @assert inst isa LLVM.CallInst
        replace_uses!(inst, level)
        erase!(inst)
    end
    @assert isempty(intr.uses)
    erase!(intr)

    return true
end


## stack allocation

# device code can request a fixed-size, per-workitem stack scratch buffer via
# `alloca(T, Val(N), Val(AS))`, returning an `LLVMPtr{T,AS}` to uninitialized storage for `N`
# elements of `T` in address space `AS`. this emits a call to the `julia.gpu.alloca` intrinsic
# with the size and alignment as constant operands, which `lower_alloca!` (run from `irgen`,
# before the optimizer) materializes as a real entry-block `alloca`.
#
# this exists because emitting an `alloca` directly through `llvmcall` is unsound/ineffective:
# the `Ptr` round-trip through `ptrtoint`/`inttoptr` blocks SROA/mem2reg promotion, the target
# stack address space (e.g. AS 5 on NVPTX/AMDGPU) isn't known at the front-end, and the
# LangRef lifetime of an `alloca` is tied to the (inlined) `llvmcall` wrapper. lowering it
# ourselves lets us place the slot in the kernel entry block, in the datalayout's alloca
# address space, early enough for the optimizer to promote it.

function alloca_intr(mod::LLVM.Module, T_ptr::LLVMType)
    name = "julia.gpu.alloca"
    intr = if haskey(mod.functions, name)
        mod.functions[name]
    else
        # takes the size in bytes and the alignment as constant operands, and returns a
        # pointer in the requested address space; intentionally *not* readnone/speculatable,
        # as each call must yield a distinct slot and must not be hoisted or CSE'd. a module
        # only ever targets a single address space, so one declaration suffices.
        T_i64 = LLVM.Int64Type()
        LLVM.Function(mod, name, LLVM.FunctionType(T_ptr, [T_i64, T_i64]))
    end
    return intr
end

# run-time equivalent: emits a call to the alloca intrinsic, returning an `LLVMPtr{T,AS}` to
# scratch storage for `N` elements of `T` in address space `AS` (materialized by
# `lower_alloca!`).
function alloca_value(@nospecialize(T), N::Int, AS::Int)
    isbitstype(T) ||
        error("GPUCompiler.alloca only supports `isbits` element types, got $T")
    N >= 0 || throw(ArgumentError("GPUCompiler.alloca count must be non-negative, got $N"))

    bytes = sizeof(T) * N
    align = Base.datatype_alignment(T)

    # a zero-byte allocation has no storage to point at; hand back a null pointer rather than
    # emitting a degenerate 0-element alloca.
    if bytes == 0
        return :(reinterpret(Core.LLVMPtr{$T,$AS}, C_NULL))
    end

    generate_llvmcall(Core.LLVMPtr{T,AS}, Tuple{}) do builder
        # `LLVMPtr{T,AS}` lowers to an (i8/opaque) pointer in address space `AS`; match that
        # as the intrinsic's return type so the `llvmcall` boundary type-checks.
        T_ptr = convert(LLVMType, Core.LLVMPtr{T,AS})
        intr = alloca_intr(current_module(builder), T_ptr)
        args = Value[ConstantInt(LLVM.Int64Type(), bytes),
                     ConstantInt(LLVM.Int64Type(), align)]
        call!(builder, intr.function_type, intr, args, "alloca")
    end
end

# device-facing accessor: an `LLVMPtr{T,AS}` to per-workitem stack scratch for `N` elements of
# `T` in address space `AS`. the storage is uninitialized and only valid within the calling
# kernel. `T` must be `isbits` (an `alloca` of GC-tracked references would be unrooted).
# intended as a building block for higher-level scratch abstractions (e.g. KernelAbstractions'
# `@private`).
@inline @generated alloca(::Type{T}, ::Val{N}, ::Val{AS}) where {T,N,AS} = alloca_value(T, N, AS)

# pick the element type for a `bytes`-sized, `align`-aligned stack slot. rather than a flat
# `[bytes x i8]`, emit aligned integer chunks: SROA takes a hint from the element type and
# will happily shred an i8 array into unaligned scalars (terrible for vectorization), whereas
# an element size equal to the alignment makes it split into aligned pieces instead. the
# element size is capped at 64 bits since not all back-ends support wider integers. mirrors
# Julia's `emit_static_alloca` (src/codegen.cpp).
function alloca_slot_type(bytes::Integer, align::Integer)
    elsize = min(align, 8)
    padded = cld(bytes, elsize) * elsize
    eltyp = LLVM.IntType(elsize * 8)
    # a single element covers the whole slot; don't bother wrapping it in a length-1 array.
    return padded == elsize ? eltyp : LLVM.ArrayType(eltyp, padded ÷ elsize)
end

# replace every `julia.gpu.alloca` call with an entry-block alloca in the containing function
function lower_alloca!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    haskey(mod.functions, "julia.gpu.alloca") || return false
    intr = mod.functions["julia.gpu.alloca"]

    @dispose builder=IRBuilder() begin
        for use in collect(intr.uses)
            call = use.user
            @assert call isa LLVM.CallInst
            bytes, align = convert.(Int, call.operands[1:2])
            f = call.parent.parent

            # materialize the slot at the top of the entry block so that it is a static
            # alloca (promotable, and allocated once rather than per loop iteration).
            position!(builder, LLVM.before(first(first(f.blocks).instructions)))
            slot = alloca!(builder, alloca_slot_type(bytes, align), "alloca")
            slot.alignment = align

            # `alloca!` placed the slot in the datalayout's alloca address space; cast it to
            # the intrinsic's return type, i.e. the address space requested by the caller
            # (emitting an addrspacecast when it differs from the alloca address space).
            ptr = pointercast!(builder, slot, call.value_type)

            replace_uses!(call, ptr)
            erase!(call)
        end
    end

    @assert isempty(intr.uses)
    erase!(intr)

    return true
end

# convert kernel state argument from pass-by-value to pass-by-reference
#
# the kernel state argument is always passed by value to avoid codegen issues with byval.
# some back-ends however do not support passing kernel arguments by value, so this pass
# serves to convert that argument (and is conceptually the inverse of `lower_byval`).
function kernel_state_to_reference!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                    f::LLVM.Function)
    ft = f.function_type

    # check if we even need a kernel state argument
    state = kernel_state_type(job)
    if state === Nothing
        return f
    end

    T_state = convert(LLVMType, state)

    # find the kernel state parameter (should be the first argument)
    if isempty(ft.parameters) || f.parameters[1].value_type != T_state
        return f
    end

    @tracepoint "kernel state to reference" begin
        # turn the leading kernel-state value parameter into a pointer the body loads from
        new_types = Union{Nothing,LLVM.LLVMType}[
            i == 1 ? LLVM.PointerType(T_state) : nothing for i in 1:length(ft.parameters)]
        new_f = clone_with_converted_args!(mod, f, new_types,
            (builder, param, i) -> load!(builder, T_state, param, "state"))
        new_f.parameters[1].name = "state_ptr"

        # set the attributes for the state pointer parameter
        attrs = new_f.parameter_attributes[1]
        # the pointer itself cannot be captured since we immediately load from it.
        # `nocapture` was replaced by `captures(none)` (an integer-valued IntAttr,
        # value 0 == CaptureInfo::none()) in LLVM 21.
        push!(attrs, LLVM.version() >= v"21" ? EnumAttribute("captures", 0)
                                             : EnumAttribute("nocapture", 0))
        # each kernel state is separate
        push!(attrs, EnumAttribute("noalias", 0))
        # the state is read-only
        push!(attrs, EnumAttribute("readonly", 0))

        # remove the old function
        replace_function!(f, new_f)

        # minimal optimization
        @dispose pb=PassBuilder() begin
            add!(pb, SimplifyCFGPass())
            run!(pb, new_f, llvm_machine(job.config.target))
        end

        return new_f
    end
end

function add_input_arguments!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                              entry::LLVM.Function, kernel_intrinsics::Dict)
    entry_fn = entry.name

    # figure out which intrinsics are used and need to be added as arguments
    used_intrinsics = filter(keys(kernel_intrinsics)) do intr_fn
        haskey(mod.functions, intr_fn)
    end |> collect
    nargs = length(used_intrinsics)

    # determine which functions need these arguments
    worklist = Set{LLVM.Function}([entry])
    for intr_fn in used_intrinsics
        push!(worklist, mod.functions[intr_fn])
    end
    worklist_length = 0
    while worklist_length != length(worklist)
        # iteratively discover functions that use an intrinsic or any function calling it
        worklist_length = length(worklist)
        additions = Set{LLVM.Function}()
        function scan_uses(val)
            for use in val.uses
                candidate = use.user
                if isa(candidate, Instruction)
                    bb = candidate.parent
                    new_f = bb.parent
                    in(new_f, worklist) || push!(additions, new_f)
                elseif isa(candidate, ConstantExpr)
                    scan_uses(candidate)
                else
                    error("Don't know how to check uses of $candidate. Please file an issue.")
                end
            end
        end
        for f in worklist
            scan_uses(f)
        end
        for f in additions
            push!(worklist, f)
        end
    end
    for intr_fn in used_intrinsics
        delete!(worklist, mod.functions[intr_fn])
    end

    # add the arguments
    # NOTE: we don't need to be fine-grained here, as unused args will be removed during opt
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = f.name
        ft = f.function_type
        f.name = fn * ".orig"
        # create a new function
        new_param_types = LLVMType[ft.parameters...]

        for intr_fn in used_intrinsics
            llvm_typ = convert(LLVMType, kernel_intrinsics[intr_fn].typ)
            push!(new_param_types, llvm_typ)
        end
        new_ft = LLVM.FunctionType(ft.return_type, new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        new_f.linkage = f.linkage
        for (arg, new_arg) in zip(f.parameters, new_f.parameters)
            new_arg.name = arg.name
        end
        for (intr_fn, new_arg) in zip(used_intrinsics, new_f.parameters[end-nargs+1:end])
            new_arg.name = kernel_intrinsics[intr_fn].name
        end

        workmap[f] = new_f
    end

    # clone and rewrite the function bodies.
    # we don't need to rewrite much as the arguments are added last.
    for (f, new_f) in workmap
        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(f.parameters, new_f.parameters)
            new_param.name = param.name
            value_map[param] = new_param
        end

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)

        # we can't remove this function yet, as we might still need to rewrite any called,
        # but remove the IR already
        empty!(f)
    end

    # drop unused constants that may be referring to the old functions
    # XXX: can we do this differently?
    for f in worklist
        prune_constexpr_uses!(f)
    end

    # update other uses of the old function, modifying call sites to pass the arguments
    function rewrite_uses!(f, new_f)
        # update uses
        @dispose builder=IRBuilder() begin
            for use in collect(f.uses)
                val = use.user
                if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                    callee_f = val.parent.parent
                    # forward the arguments
                    position!(builder, LLVM.before(val))
                    new_val = if val isa LLVM.CallInst
                        call!(builder, new_f.function_type, new_f,
                              [val.arguments..., callee_f.parameters[end-nargs+1:end]...],
                              val.operand_bundles)
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    new_val.callconv = val.callconv

                    replace_uses!(val, new_val)
                    @assert isempty(val.uses)
                    erase!(val)
                elseif val isa LLVM.ConstantExpr && val.opcode == LLVM.API.LLVMBitCast
                    # XXX: why isn't this caught by the value materializer above?
                    target = val.operands[1]
                    @assert target == f
                    new_val = LLVM.const_bitcast(new_f, val.value_type)
                    rewrite_uses!(val, new_val)
                    # we can't simply replace this constant expression, as it may be used
                    # as a call, taking arguments (so we need to rewrite it to pass the input arguments)

                    # drop the old constant if it is unused
                    # XXX: can we do this differently?
                    if isempty(val.uses)
                        LLVM.unsafe_destroy!(val)
                    end
                else
                    error("Cannot rewrite unknown use of function: $val")
                end
            end
        end
    end
    for (f, new_f) in workmap
        rewrite_uses!(f, new_f)
        @assert isempty(f.uses)
        replace_metadata_uses!(f, new_f)
        erase!(f)
    end

    # replace uses of the intrinsics with references to the input arguments
    for (i, intr_fn) in enumerate(used_intrinsics)
        intr = mod.functions[intr_fn]
        for use in collect(intr.uses)
            val = use.user
            callee_f = val.parent.parent
            if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                replace_uses!(val, callee_f.parameters[end-nargs+i])
            else
                error("Cannot rewrite unknown use of function: $val")
            end

            @assert isempty(val.uses)
            erase!(val)
        end
        @assert isempty(intr.uses)
        erase!(intr)
    end

    return mod.functions[entry_fn]
end
