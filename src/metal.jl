# implementation of the GPUCompiler interfaces for generating Metal code


## target info

# Metal has no target machine, so provide our own TTI
struct MetalTTI <: LLVM.AbstractTargetTransformInfo end

# teache LLVM about Metal's address-space hierarchy:
#   0: Generic    1: Device       2: Constant
#   3: ThreadGroup 4: Thread      5: ThreadGroup_ImgBlock  6: Ray
# AS 0 is the flat/generic space; only casts involving it are legal, and the
# specific spaces are mutually disjoint.
LLVM.flat_address_space(::MetalTTI) = UInt(0)
LLVM.is_noop_addr_space_cast(::MetalTTI, from::Unsigned, to::Unsigned) =
    from == 0 || to == 0
LLVM.is_valid_addr_space_cast(::MetalTTI, from::Unsigned, to::Unsigned) =
    from == to || from == 0 || to == 0

# distinct specific address spaces are disjoint; only the generic AS overlaps.
LLVM.addrspaces_may_alias(::MetalTTI, a::Unsigned, b::Unsigned) =
    a == b || a == 0 || b == 0

# used as a coarse "this is a GPU target" switch by several IR passes (e.g.
# JumpThreading and non-trivial SimpleLoopUnswitch become no-ops), not just
# UniformityAnalysis — which we don't have consumers for anyway.
LLVM.has_branch_divergence(::MetalTTI) = true

# deliberately not overriding `is_single_threaded`: a kernel is multi-lane, and
# returning `true` would let LICM sink stores onto paths that didn't store,
# producing races across lanes.

# only the spaces backed by static storage admit non-undef initializers; thread,
# threadgroup and ray-payload spaces are populated at dispatch/invocation time.
LLVM.can_have_non_undef_global_initializer_in_address_space(::MetalTTI, as::Unsigned) =
    as == 0 || as == 1 || as == 2


## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    # version numbers
    macos::VersionNumber
    air::VersionNumber
    metal::VersionNumber
end

# for backwards compatibility
MetalCompilerTarget(macos::VersionNumber) =
    MetalCompilerTarget(; macos, air=v"2.4", metal=v"2.4")

function Base.hash(target::MetalCompilerTarget, h::UInt)
    h = hash(target.macos, h)
    h = hash(target.air, h)
    h = hash(target.metal, h)
end

source_code(target::MetalCompilerTarget) = "text"

# Metal is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::MetalCompilerTarget) = nothing

llvm_triple(target::MetalCompilerTarget) = "air64-apple-macosx$(target.macos)"

llvm_datalayout(target::MetalCompilerTarget) =
    "e-p:64:64:64"*
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
    "-f32:32:32-f64:64:64"*
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"*
    "-n8:16:32"

llvm_targetinfo(::MetalCompilerTarget) = MetalTTI()

pass_by_value(job::CompilerJob{MetalCompilerTarget}) = false


## job

# the debug-info level is part of the slug (as on PTX): it changes the debug metadata emitted
# into the runtime library, and the device-exception reporters branch on it, so a library
# built at one level must not be reused at another.
runtime_slug(job::CompilerJob{MetalCompilerTarget}) =
    "metal-macos$(job.config.target.macos)-debuginfo=$(Int(llvm_debug_info(job)))"

isintrinsic(@nospecialize(job::CompilerJob{MetalCompilerTarget}), fn::String) =
    return startswith(fn, "air.")

function finish_linked_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module)
    for f in kernels(mod)
        # update calling conventions
        f = pass_by_reference!(job, mod, f)
        f = add_input_arguments!(job, mod, f, kernel_intrinsics)
    end

    # emit the AIR and Metal version numbers as constants in the module. this makes it
    # possible to 'query' these in device code, relying on LLVM to optimize the checks away
    # and generate static code. note that we only do so if there's actual uses of these
    # variables; unconditionally creating a gvar would result in duplicate declarations.
    for (name, value) in ["air_major"   => job.config.target.air.major,
                          "air_minor"   => job.config.target.air.minor,
                          "metal_major" => job.config.target.metal.major,
                          "metal_minor" => job.config.target.metal.minor]
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            initializer!(gv, ConstantInt(LLVM.Int32Type(), value))
            # change the linkage so that we can inline the value
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        end
    end

    # add metadata to AIR intrinsics LLVM doesn't know about
    annotate_air_intrinsics!(job, mod)

    # we emit properties (of the air and metal version) as private global constants,
    # so run the optimizer so that they are inlined before the rest of the optimizer runs.
    @dispose pb=NewPMPassBuilder() begin
        LLVM.target_transform_info!(pb, MetalTTI())
        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        run!(pb, mod)
    end

    return
end

function validate_ir(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module)
    errors = IRError[]

    # Metal does not support double precision, except for logging
    function is_illegal_double(val)
        T_bad = LLVM.DoubleType()
        if value_type(val) != T_bad
            return false
        end

        function used_for_logging(use::LLVM.Use)
            usr = user(use)
            if usr isa LLVM.CallInst
                callee = called_operand(usr)
                if callee isa LLVM.Function && startswith(name(callee), "metal_os_log")
                    return true
                end
            end
            return false
        end
        if all(used_for_logging, uses(val))
            return false
        end

        return true
    end
    append!(errors, check_ir_values(mod, is_illegal_double, "use of double value"))

    # Metal never supports 128-bit integers
    append!(errors, check_ir_values(mod, LLVM.IntType(128)))

    errors
end

function finish_ir!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module,
                                  entry::LLVM.Function)
    entry_fn = LLVM.name(entry)

    # convert the kernel state argument to a reference
    if job.config.kernel && kernel_state_type(job) !== Nothing
        entry = kernel_state_to_reference!(job, mod, entry)
    end

    # add kernel metadata
    if job.config.kernel
        entry = add_parameter_address_spaces!(job, mod, entry)
        entry = add_global_address_spaces!(job, mod, entry)

        # narrow generic pointer parameters whose callers all pass a specific-AS pointer, so
        # the constant globals read by out-of-line runtime functions (e.g. the exception
        # reporters) load from the constant space rather than crashing Metal's validator.
        propagate_argument_address_spaces!(mod)

        # propagate specific address spaces through addrspacecast chains introduced
        # by the rewrites above, so that loads/stores happen in the right address
        # space (e.g. constant globals in addrspace 2 rather than via a cast to 0,
        # which Metal's backend cannot handle correctly for dynamic indices).
        @dispose pb=NewPMPassBuilder() begin
            LLVM.target_transform_info!(pb, MetalTTI())
            add!(pb, NewPMFunctionPassManager()) do fpm
                add!(fpm, InferAddressSpacesPass())
                add!(fpm, SROAPass())
                add!(fpm, instcombine_pass(job))
                add!(fpm, EarlyCSEPass())
                add!(fpm, SimplifyCFGPass())
            end
            run!(pb, mod)
        end

        add_argument_metadata!(job, mod, entry)

        add_module_metadata!(job, mod)
    end

    # strip device-side `trap`s and rewrite `unreachable` into clean returns (#433, #370). this
    # runs post-`optimize!`, after the trap has finished serving as the optimizer guard; the pass
    # force-inlines throwing functions into the kernel first so the rewrite is sound, then scrubs
    # every `noreturn` attribute.
    #
    # this also subsumes the old `hide_noreturn!` workaround for #113 (kernel hangs from divergent
    # `noreturn` control flow on older macOS). that bug reduced to a `noinline` helper of the shape
    # `trap; unreachable` called divergently, and `hide_noreturn!` worked by force-inlining it;
    # this pass inlines the same helper (keying on the `trap`/`unreachable` it contains, not the
    # attribute), rewrites its `unreachable` into a clean branch-to-`ret`, and drops the `noreturn`,
    # leaving nothing divergent for the back-end to choke on. (the only `noreturn` shape it doesn't
    # inline is a genuine infinite loop — but inlining can't make that return either, so
    # `hide_noreturn!` never fixed that case to begin with.)
    lower_unreachable_control_flow!(job, mod)

    # lower LLVM intrinsics that AIR doesn't support
    changed = false
    for f in functions(mod)
        changed |= lower_llvm_intrinsics!(job, f)
    end
    if changed
        # lowering may have introduced additional functions marked `alwaysinline`
        @dispose pb=NewPMPassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            add!(pb, NewPMFunctionPassManager()) do fpm
                add!(fpm, SimplifyCFGPass())
                add!(fpm, instcombine_pass(job))
            end
            run!(pb, mod)
        end
    end

    # perform codegen passes that would normally run during machine code emission
    if LLVM.has_oldpm()
        # XXX: codegen passes don't seem available in the new pass manager yet
        @dispose pm=ModulePassManager() begin
            expand_reductions!(pm)
            run!(pm, mod)
        end
    end

    return functions(mod)[entry_fn]
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
    # our LLVM version does not support emitting Metal libraries
    return nothing
end


# generic pointer removal
#
# every pointer argument (i.e. byref objs) to a kernel needs an address space attached.
# this pass rewrites pointers to reference arguments to be located in address space 1.
#
# NOTE: this pass only rewrites byref objs, not plain pointers being passed; the user is
# responsible for making sure these pointers have an address space attached (using LLVMPtr).
#
# NOTE: this pass also only rewrites pointers _without_ address spaces, which requires it to
# be executed after optimization (where Julia's address spaces are stripped). If we ever
# want to execute it earlier, adapt remapType to rewrite all pointer types.
function add_parameter_address_spaces!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                       f::LLVM.Function)
    ft = function_type(f)

    # find the byref parameters
    byref = BitVector(undef, length(parameters(ft)))
    args = classify_arguments(job, ft; post_optimization=job.config.optimize)
    filter!(args) do arg
        arg.cc != GHOST
    end
    for arg in args
        byref[arg.idx] = (arg.cc == BITS_REF || arg.cc == KERNEL_STATE)
    end

    function remapType(src)
        # TODO: shouldn't we recurse into structs here, making sure the parent object's
        #       address space matches the contained one? doesn't matter right now as we
        #       only use LLVMPtr (i.e. no rewriting of contained pointers needed) in the
        #       device addrss space (i.e. no mismatch between parent and field possible)
        dst = if src isa LLVM.PointerType && addrspace(src) == 0
            if supports_typed_pointers(context())
                LLVM.PointerType(remapType(eltype(src)), #=device=# 1)
            else
                LLVM.PointerType(#=device=# 1)
            end
        else
            src
        end
        return dst
    end

    # generate the new function type & definition
    new_types = LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        if byref[i]
            push!(new_types, remapType(param::LLVM.PointerType))
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

    # we cannot simply remap the function arguments, because that will not propagate the
    # address space changes across, e.g, bitcasts (the dest would still be in AS 0).
    # using a type remapper on the other hand changes too much, including unrelated insts.
    # so instead, we load the arguments in stack slots and dereference them so that we can
    # keep on using the original IR that assumed pointers without address spaces
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(ft))
            if byref[i]
                # load the argument in a stack slot
                llvm_typ = convert(LLVMType, args[i].typ)
                val = load!(builder, llvm_typ, parameters(new_f)[i])
                ptr = alloca!(builder, llvm_typ)
                store!(builder, val, ptr)
                push!(new_args, ptr)
            else
                push!(new_args, parameters(new_f)[i])
            end
            for attr in collect(parameter_attributes(f, i))
                push!(parameter_attributes(new_f, i), attr)
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
    fn = LLVM.name(f)
    prune_constexpr_uses!(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)

    # clean-up after this pass (which runs after optimization)
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, SimplifyCFGPass())
        add!(pb, SROAPass())
        add!(pb, EarlyCSEPass())
        add!(pb, instcombine_pass(job))

        run!(pb, mod)
    end

    return new_f
end

# update address spaces of constant global objects
#
# global constant objects need to reside in address space 2, so we clone each function
# that uses global objects and rewrite the globals used by it
function add_global_address_spaces!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                    entry::LLVM.Function)
    # determine global variables we need to update
    global_map = Dict{LLVM.Value, LLVM.Value}()
    for gv in globals(mod)
        isconstant(gv) || continue
        addrspace(value_type(gv)) == 0 || continue

        gv_ty = global_value_type(gv)
        gv_name = LLVM.name(gv)

        LLVM.name!(gv, gv_name * ".old")
        new_gv = GlobalVariable(mod, gv_ty, gv_name, 2)

        alignment!(new_gv, alignment(gv))
        unnamed_addr!(new_gv, unnamed_addr(gv))
        initializer!(new_gv, initializer(gv))
        constant!(new_gv, true)
        linkage!(new_gv, linkage(gv))
        visibility!(new_gv, visibility(gv))

        # we can't map the global variable directly, as the type change won't be applied
        # recursively. so instead map a constant expression converting the value of the
        # global into one with the old address space, avoiding a type change.
        ptr = const_addrspacecast(new_gv, value_type(gv))

        global_map[gv] = ptr
    end
    isempty(global_map) && return entry

    # determine which functions we need to update
    function_worklist = Set{LLVM.Function}()
    function check_user(val)
        if val isa LLVM.Instruction
            bb = LLVM.parent(val)
            f = LLVM.parent(bb)

            push!(function_worklist, f)
        elseif val isa LLVM.ConstantExpr
            for use in uses(val)
                check_user(user(use))
            end
        end
    end
    for gv in keys(global_map), use in uses(gv)
        check_user(user(use))
    end

    # update functions that use the global
    if !isempty(function_worklist)
        entry_fn = LLVM.name(entry)
        for fun in function_worklist
            fn = LLVM.name(fun)

            new_fun = clone(fun; value_map=global_map)
            replace_uses!(fun, new_fun)
            replace_metadata_uses!(fun, new_fun)
            erase!(fun)

            LLVM.name!(new_fun, fn)
        end
        entry = LLVM.functions(mod)[entry_fn]
    end

    # delete old globals
    for (old, new) in global_map
        prune_constexpr_uses!(old)
        @assert isempty(uses(old))
        replace_metadata_uses!(old, new)
        erase!(old)
    end

    return entry
end


# interprocedural address-space narrowing
#
# `InferAddressSpaces` rewrites a generic (flat) load/store into a concrete address space
# when it can trace the pointer back to an `addrspacecast` from that space, but only within
# one function. A pointer crossing a call boundary as a generic parameter loses that
# provenance: a constant global passed to an out-of-line runtime function (the exception
# reporters take `Ptr` arguments) arrives generic and is read with a generic-space load,
# which crashes Metal's shader validator.
#
# This pass is the interprocedural complement. When every caller passes the same kind of
# value for a generic pointer parameter, `addrspacecast(<ptr in a specific space> ->
# generic)`, it retargets the parameter to that space, drops the casts at the call sites,
# and casts back to generic on entry so the body is unchanged. That only relocates a
# side-effect-free cast across the boundary, so it is trivially correct; the following
# `InferAddressSpaces` run folds the entry cast away. The source need not be a constant
# global; any pointer with a known address space qualifies, so any back-end can run it.
#
# With typed pointers (older LLVM) a `Ptr` argument is lowered to an integer rather than a
# pointer, so the same boundary crossing arrives as `ptrtoint(addrspacecast(<specific> ->
# generic))` at the call site and `inttoptr` in the body. Such an integer parameter is
# retargeted the same way: it becomes a pointer in the agreed space, the call sites pass the
# bare source pointer, and entry rebuilds the original integer as `ptrtoint(addrspacecast(
# param -> generic))`. The cloned body's `inttoptr` then composes with that and the same
# `InferAddressSpaces` run folds the whole chain to a specific-space load. (Without this the
# leftover `ptrtoint(addrspacecast(...))` constant feeds a generic-space load that, e.g., the
# LLVM-16 Metal bitcode downgrade miscompiles into an invalid metallib — JuliaGPU/Metal.jl
# device exceptions on Julia 1.11.)
#
# Narrowing one function makes its body forward an `addrspacecast`-from-specific to the
# functions it calls, exposing them in turn. We therefore iterate to a fixed point so a
# constant reaches an arbitrarily deep callee (e.g. an exception reporter that delegates to
# another) regardless of the order functions are visited in. This terminates: each sweep
# that changes anything strictly reduces the number of generic pointer parameters in the
# module, and narrowing never introduces a new one.

# If `v` is an `addrspacecast` (instruction or constant expression) of a pointer from a
# specific (non-generic) address space to the generic one, return that source pointer;
# otherwise `nothing`.
function addrspacecast_to_generic_source(@nospecialize(v))
    (v isa LLVM.Instruction || v isa LLVM.ConstantExpr) || return nothing
    opcode(v) == LLVM.API.LLVMAddrSpaceCast || return nothing
    addrspace(value_type(v)) == 0 || return nothing
    src = operands(v)[1]
    (value_type(src) isa LLVM.PointerType && addrspace(value_type(src)) != 0) ||
        return nothing
    return src
end

# The typed-pointer counterpart of `addrspacecast_to_generic_source`. With typed pointers a
# `Ptr` argument is lowered to an integer, so a specific-space pointer crossing a call boundary
# arrives as `ptrtoint(addrspacecast(<ptr in a specific space> -> generic))` rather than the
# bare cast. If `v` is that shape, return the specific-space source pointer; otherwise `nothing`.
function ptrtoint_of_generic_source(@nospecialize(v))
    (v isa LLVM.Instruction || v isa LLVM.ConstantExpr) || return nothing
    opcode(v) == LLVM.API.LLVMPtrToInt || return nothing
    return addrspacecast_to_generic_source(operands(v)[1])
end

# If integer parameter `arg` is consumed only by `inttoptr` (all uses agreeing on the result
# pointer type), return that type; otherwise `nothing`. Such a parameter is the integer image
# of a pointer that crossed the call boundary, and can be retargeted to a pointer the same way
# a generic pointer parameter is (see `propagate_argument_address_spaces!`, the integer case).
function integer_param_pointer_type(arg::LLVM.Argument)
    ptrty = nothing
    for use in uses(arg)
        u = user(use)
        (u isa LLVM.Instruction && opcode(u) == LLVM.API.LLVMIntToPtr) || return nothing
        t = value_type(u)
        ptrty === nothing ? (ptrty = t) : (ptrty == t || return nothing)
    end
    return ptrty
end

function propagate_argument_address_spaces!(mod::LLVM.Module)
    changed = false
    while propagate_argument_address_spaces_once!(mod)
        changed = true
    end
    return changed
end

# a single narrowing sweep over the module; returns whether anything changed.
function propagate_argument_address_spaces_once!(mod::LLVM.Module)
    changed = false
    for f in collect(functions(mod))
        isempty(blocks(f)) && continue          # only functions we can rewrite (have a body)

        # rewriting a signature is only sound with no callers outside the module, so require
        # local (internal/private) linkage. by `finish_ir!` the pipeline has internalized
        # everything but the kernel entrypoints, so the runtime helpers we target qualify.
        linkage(f) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage) || continue

        param_types = parameters(function_type(f))

        # collect call sites; bail unless every use is a direct call we can update
        callsites = LLVM.CallInst[]
        only_calls = true
        for use in uses(f)
            v = user(use)
            if v isa LLVM.CallInst && called_operand(v) == f
                push!(callsites, v)
            else
                only_calls = false
                break
            end
        end
        (only_calls && !isempty(callsites)) || continue

        # for each narrowable parameter, find the address space its callers agree on. a
        # generic pointer parameter is passed as `addrspacecast(<specific> -> generic)`; with
        # typed pointers a `Ptr` argument is instead an integer passed as `ptrtoint` of that
        # cast (`int_ptr_types[i]` records the pointer it is reconstructed to, marking Case B).
        new_addrspaces = fill(-1, length(param_types))
        int_ptr_types = Vector{Any}(nothing, length(param_types))
        for (i, pty) in enumerate(param_types)
            extract = if pty isa LLVM.PointerType && addrspace(pty) == 0
                addrspacecast_to_generic_source
            elseif pty isa LLVM.IntegerType &&
                   integer_param_pointer_type(parameters(f)[i]) !== nothing
                ptrtoint_of_generic_source
            else
                continue
            end
            as = -1
            for cs in callsites
                src = extract(arguments(cs)[i])
                if src === nothing
                    as = -1; break
                end
                src_as = addrspace(value_type(src))
                as == -1 ? (as = src_as) : (as == src_as || (as = -1; break))
            end
            if as > 0
                new_addrspaces[i] = as
                pty isa LLVM.IntegerType &&
                    (int_ptr_types[i] = integer_param_pointer_type(parameters(f)[i]))
            end
        end
        any(>=(0), new_addrspaces) || continue

        narrow_pointer_parameters!(mod, f, new_addrspaces, int_ptr_types, callsites)
        changed = true
    end
    return changed
end

# copy the call-site attributes (function/return/per-argument) from `src` onto `dst`. the
# narrowing keeps argument positions unchanged, so they map across one-to-one.
function copy_callsite_attributes!(dst::LLVM.CallInst, src::LLVM.CallInst)
    for attr in collect(function_attributes(src))
        push!(function_attributes(dst), attr)
    end
    for attr in collect(return_attributes(src))
        push!(return_attributes(dst), attr)
    end
    for i in 1:length(arguments(src))
        for attr in collect(argument_attributes(src, i))
            push!(argument_attributes(dst, i), attr)
        end
    end
    return dst
end

# rewrite a single call so it targets `new_f`/`new_ft`, passing the un-casted source value
# for each retargeted argument (and the original argument otherwise). Preserves calling
# convention, operand bundles and attributes; replaces and erases the old call.
function rewrite_narrowed_call!(builder::IRBuilder, cs::LLVM.CallInst,
                                new_f::LLVM.Function, new_ft::LLVM.FunctionType,
                                new_addrspaces::Vector{Int}, int_ptr_types::Vector{Any})
    position!(builder, cs)
    new_param_types = parameters(new_ft)
    new_args = LLVM.Value[]
    for (i, arg) in enumerate(arguments(cs))
        if new_addrspaces[i] < 0
            push!(new_args, arg)
        elseif int_ptr_types[i] !== nothing
            # Case B: strip the `ptrtoint` then the cast, and bitcast the bare specific-space
            # pointer to the retargeted parameter's pointer type
            src = addrspacecast_to_generic_source(operands(arg)[1])
            push!(new_args, bitcast!(builder, src, new_param_types[i]))
        else
            push!(new_args, addrspacecast_to_generic_source(arg))
        end
    end
    new_call = call!(builder, new_ft, new_f, new_args, operand_bundles(cs))
    callconv!(new_call, callconv(cs))
    copy_callsite_attributes!(new_call, cs)
    replace_uses!(cs, new_call)
    erase!(cs)
    return new_call
end

# Clone `f` with the pointer parameters listed in `new_addrspaces` (index => address space,
# `-1` to leave alone) retargeted to those address spaces, casting each retargeted parameter
# back to generic on entry so the cloned body is unchanged. Rewrite `callsites` to pass the
# un-casted source value for each retargeted argument; recursive self-calls are handled too.
function narrow_pointer_parameters!(mod::LLVM.Module, f::LLVM.Function,
                                    new_addrspaces::Vector{Int}, int_ptr_types::Vector{Any},
                                    callsites)
    ft = function_type(f)
    # retarget a pointer to address space `as`, taking its pointee from `srcptr` (only needed
    # for typed pointers; `eltype` is invalid on opaque ones, so keep it lazy)
    retarget(as::Integer, srcptr::LLVM.PointerType) =
        supports_typed_pointers(context()) ? LLVM.PointerType(eltype(srcptr), as) :
                                             LLVM.PointerType(as)
    # the retargeted parameter type: for a pointer parameter (Case A) keep its pointee; for an
    # integer parameter (Case B) use the pointee of the pointer its body reconstructs.
    new_param_type(i, param_typ) =
        new_addrspaces[i] < 0 ? param_typ :
        int_ptr_types[i] !== nothing ?
            retarget(new_addrspaces[i], int_ptr_types[i]::LLVM.PointerType) :
            retarget(new_addrspaces[i], param_typ::LLVM.PointerType)
    new_types = LLVM.LLVMType[new_param_type(i, param_typ)
                              for (i, param_typ) in enumerate(parameters(ft))]
    new_ft = LLVM.FunctionType(return_type(ft), new_types)

    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    callconv!(new_f, callconv(f))
    for (old_arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(old_arg))
    end

    # cast each retargeted parameter back to generic so the cloned body keeps using it
    # unchanged (InferAddressSpaces folds the cast away afterwards)
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)
        new_args = LLVM.Value[]
        for (i, param_typ) in enumerate(parameters(ft))
            if new_addrspaces[i] < 0
                push!(new_args, parameters(new_f)[i])
            elseif int_ptr_types[i] !== nothing
                # Case B: rebuild the original integer as `ptrtoint(addrspacecast(param ->
                # generic))`; the cloned body's `inttoptr` composes with it, and the following
                # InferAddressSpaces run folds the whole chain to a specific-space load.
                gen = addrspacecast!(builder, parameters(new_f)[i],
                                     int_ptr_types[i]::LLVM.PointerType)
                push!(new_args, ptrtoint!(builder, gen, param_typ))
            else
                push!(new_args, addrspacecast!(builder, parameters(new_f)[i], param_typ))
            end
        end

        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(parameters(f)))
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        br!(builder, blocks(new_f)[2])  # fall through to the cloned entry block
    end

    # `clone_into!` copies a parameter's attributes only when it maps to a new argument; the
    # retargeted ones map to the entry cast instead, so theirs are dropped. Reattach them for
    # Case A (still valid on the narrowed pointer). Skip Case B: those were integer attributes
    # (e.g. `zeroext`) that are invalid on the now-pointer parameter.
    for i in 1:length(new_addrspaces)
        (new_addrspaces[i] >= 0 && int_ptr_types[i] === nothing) || continue
        for attr in collect(parameter_attributes(f, i))
            push!(parameter_attributes(new_f, i), attr)
        end
    end

    # a (directly) recursive `f` has self-calls that cloning retargeted to `new_f` but left
    # with the old signature; collect them from the clone for rewriting. collect first, since
    # the rewritten calls also target `new_f` and must not be revisited.
    self_calls = LLVM.CallInst[]
    for bb in blocks(new_f), inst in instructions(bb)
        inst isa LLVM.CallInst && called_operand(inst) == new_f && push!(self_calls, inst)
    end

    # rewrite call sites to pass the un-casted source value for each retargeted argument
    @dispose builder=IRBuilder() begin
        for cs in callsites
            rewrite_narrowed_call!(builder, cs, new_f, new_ft, new_addrspaces, int_ptr_types)
        end
        for cs in self_calls
            rewrite_narrowed_call!(builder, cs, new_f, new_ft, new_addrspaces, int_ptr_types)
        end
    end

    fn = LLVM.name(f)
    @assert isempty(uses(f))   # every use was a call site we just rewrote
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)
    return new_f
end


# value-to-reference conversion
#
# Metal doesn't support passing values, so we need to convert those to references instead
function pass_by_reference!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ft = function_type(f)

    # generate the new function type & definition
    args = classify_arguments(job, ft)
    new_types = LLVM.LLVMType[]
    bits_as_reference = BitVector(undef, length(parameters(ft)))
    for arg in args
        if arg.cc == BITS_VALUE && !(arg.typ <: Ptr || arg.typ <: Core.LLVMPtr)
            # pass the value as a reference instead
            push!(new_types, LLVM.PointerType(parameters(ft)[arg.idx], #=Constant=# 1))
            bits_as_reference[arg.idx] = true
        elseif arg.cc != GHOST
            push!(new_types, parameters(ft)[arg.idx])
            bits_as_reference[arg.idx] = false
        end
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (i, (arg, new_arg)) in enumerate(zip(parameters(f), parameters(new_f)))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "entry")
        position!(builder, entry)

        # perform argument conversions
        for arg in args
            if arg.cc != GHOST
                if bits_as_reference[arg.idx]
                    # load the reference to get a value back
                    val = load!(builder, parameters(ft)[arg.idx], parameters(new_f)[arg.idx])
                    push!(new_args, val)
                else
                    push!(new_args, parameters(new_f)[arg.idx])
                end
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)

        # fall through
        br!(builder, blocks(new_f)[2])
    end

    # set the attributes (needs to happen _after_ cloning)
    # TODO: verify that clone copies other attributes,
    #       and that other uses of clone don't set parameters before cloning
    for i in 1:length(parameters(new_f))
        if bits_as_reference[i]
            # add appropriate attributes
            # TODO: other attributes (nonnull, readonly, align, dereferenceable)?
            ## we've just emitted a load, so the pointer itself cannot be captured.
            ## `nocapture` was replaced by `captures(none)` in LLVM 21 (an
            ## integer-valued IntAttr, value 0 == CaptureInfo::none()).
            push!(parameter_attributes(new_f, i),
                  LLVM.version() >= v"21" ? EnumAttribute("captures", 0)
                                          : EnumAttribute("nocapture", 0))
            ## Metal.jl emits separate buffers for each scalar argument
            push!(parameter_attributes(new_f, i), EnumAttribute("noalias", 0))
        end
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)

    return new_f
end


# kernel input arguments
#
# hardware index counters (thread id, group id, etc) aren't accessed via intrinsics,
# but using special arguments to the kernel function.

const kernel_intrinsics = Dict()
for intr in [
        "dispatch_quadgroups_per_threadgroup", "dispatch_simdgroups_per_threadgroup",
        "quadgroup_index_in_threadgroup", "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup", "simdgroups_per_threadgroup",
        "thread_index_in_quadgroup", "thread_index_in_simdgroup",
        "thread_index_in_threadgroup", "thread_execution_width", "threads_per_simdgroup"],
    (llvm_typ, julia_typ) in [
        ("i32",  UInt32),
        ("i16",  UInt16),
    ]
    push!(kernel_intrinsics, "julia.air.$intr.$llvm_typ" =>  (name=intr, typ=julia_typ))
end
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"],
    (llvm_typ, julia_typ) in [
        ("i32",   UInt32),
        ("v2i32", NTuple{2, VecElement{UInt32}}),
        ("v3i32", NTuple{3, VecElement{UInt32}}),
        ("i16",   UInt16),
        ("v2i16", NTuple{2, VecElement{UInt16}}),
        ("v3i16", NTuple{3, VecElement{UInt16}}),
    ]
    push!(kernel_intrinsics, "julia.air.$intr.$llvm_typ" => (name=intr, typ=julia_typ))
end

function argument_type_name(typ)
    if typ isa LLVM.IntegerType && width(typ) == 16
        "ushort"
    elseif typ isa LLVM.IntegerType && width(typ) == 32
        "uint"
    elseif typ isa LLVM.VectorType
         argument_type_name(eltype(typ)) * string(Int(length(typ)))
    else
        error("Cannot encode unknown type `$typ`")
    end
end

# argument metadata generation
#
# module metadata is used to identify buffers that are passed as kernel arguments.

function add_argument_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                entry::LLVM.Function)
    entry_ft = function_type(entry)

    ## argument info
    arg_infos = Metadata[]

    # Iterate through arguments and create metadata for them
    args = classify_arguments(job, entry_ft; post_optimization=job.config.optimize)
    i = 1
    for arg in args
        arg.idx ===  nothing && continue
        if job.config.optimize
            @assert parameters(entry_ft)[arg.idx] isa LLVM.PointerType
        else
            parameters(entry_ft)[arg.idx] isa LLVM.PointerType || continue
        end

        # NOTE: we emit the bare minimum of argument metadata to support
        #       bindless argument encoding. Actually using the argument encoder
        #       APIs (deprecated in Metal 3) turned out too difficult, given the
        #       undocumented nature of the argument metadata, and the complex
        #       arguments we encounter with typical Julia kernels.

        md = Metadata[]

        # argument index
        @assert arg.idx == i
        push!(md, Metadata(ConstantInt(Int32(i-1))))

        push!(md, MDString("air.buffer"))

        push!(md, MDString("air.location_index"))
        push!(md, Metadata(ConstantInt(Int32(i-1))))

        # XXX: unknown
        push!(md, Metadata(ConstantInt(Int32(1))))

        push!(md, MDString("air.read_write")) # TODO: Check for const array

        push!(md, MDString("air.address_space"))
        push!(md, Metadata(ConstantInt(Int32(addrspace(parameters(entry_ft)[arg.idx])))))

        arg_type = if arg.typ <: Core.LLVMPtr
            arg.typ.parameters[1]
        else
            arg.typ
        end

        push!(md, MDString("air.arg_type_size"))
        push!(md, Metadata(ConstantInt(Int32(sizeof(arg_type)))))

        push!(md, MDString("air.arg_type_align_size"))
        push!(md, Metadata(ConstantInt(Int32(Base.datatype_alignment(arg_type)))))

        push!(md, MDString("air.arg_type_name"))
        push!(md, MDString(repr(arg.typ)))

        push!(md, MDString("air.arg_name"))
        push!(md, MDString(String(arg.name)))

        push!(arg_infos, MDNode(md))

        i += 1
    end

    # Create metadata for argument intrinsics last
    for intr_arg in parameters(entry)[i:end]
        intr_fn = LLVM.name(intr_arg)

        arg_info = Metadata[]

        push!(arg_info, Metadata(ConstantInt(Int32(i-1))))
        push!(arg_info, MDString("air.$intr_fn" ))

        push!(arg_info, MDString("air.arg_type_name" ))
        push!(arg_info, MDString(argument_type_name(value_type(intr_arg))))

        arg_info = MDNode(arg_info)
        push!(arg_infos, arg_info)

        i += 1
    end
    arg_infos = MDNode(arg_infos)


    ## stage info
    stage_infos = Metadata[]
    stage_infos = MDNode(stage_infos)

    kernel_md = MDNode([entry, stage_infos, arg_infos])
    push!(metadata(mod)["air.kernel"], kernel_md)

    return
end


# module-level metadata

# TODO: determine limits being set dynamically
function add_module_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    # register max device buffer count
    max_buff = Metadata[]
    push!(max_buff, Metadata(ConstantInt(Int32(7))))
    push!(max_buff, MDString("air.max_device_buffers"))
    push!(max_buff, Metadata(ConstantInt(Int32(31))))
    max_buff = MDNode(max_buff)
    push!(metadata(mod)["llvm.module.flags"], max_buff)

    # register max constant buffer count
    max_const_buff_md = Metadata[]
    push!(max_const_buff_md, Metadata(ConstantInt(Int32(7))))
    push!(max_const_buff_md, MDString("air.max_constant_buffers"))
    push!(max_const_buff_md, Metadata(ConstantInt(Int32(31))))
    max_const_buff_md = MDNode(max_const_buff_md)
    push!(metadata(mod)["llvm.module.flags"], max_const_buff_md)

    # register max threadgroup buffer count
    max_threadgroup_buff_md = Metadata[]
    push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(7))))
    push!(max_threadgroup_buff_md, MDString("air.max_threadgroup_buffers"))
    push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(31))))
    max_threadgroup_buff_md = MDNode(max_threadgroup_buff_md)
    push!(metadata(mod)["llvm.module.flags"], max_threadgroup_buff_md)

    # register max texture buffer count
    max_textures_md = Metadata[]
    push!(max_textures_md, Metadata(ConstantInt(Int32(7))))
    push!(max_textures_md, MDString("air.max_textures"))
    push!(max_textures_md, Metadata(ConstantInt(Int32(128))))
    max_textures_md = MDNode(max_textures_md)
    push!(metadata(mod)["llvm.module.flags"], max_textures_md)

    # register max write texture buffer count
    max_rw_textures_md = Metadata[]
    push!(max_rw_textures_md, Metadata(ConstantInt(Int32(7))))
    push!(max_rw_textures_md, MDString("air.max_read_write_textures"))
    push!(max_rw_textures_md, Metadata(ConstantInt(Int32(8))))
    max_rw_textures_md = MDNode(max_rw_textures_md)
    push!(metadata(mod)["llvm.module.flags"], max_rw_textures_md)

    # register max sampler count
    max_samplers_md = Metadata[]
    push!(max_samplers_md, Metadata(ConstantInt(Int32(7))))
    push!(max_samplers_md, MDString("air.max_samplers"))
    push!(max_samplers_md, Metadata(ConstantInt(Int32(16))))
    max_samplers_md = MDNode(max_samplers_md)
    push!(metadata(mod)["llvm.module.flags"], max_samplers_md)

    # add compiler identification
    llvm_ident_md = Metadata[]
    push!(llvm_ident_md, MDString("Julia $(VERSION) with Metal.jl"))
    llvm_ident_md = MDNode(llvm_ident_md)
    push!(metadata(mod)["llvm.ident"], llvm_ident_md)

    # add AIR version
    air_md = Metadata[]
    push!(air_md, Metadata(ConstantInt(Int32(job.config.target.air.major))))
    push!(air_md, Metadata(ConstantInt(Int32(job.config.target.air.minor))))
    push!(air_md, Metadata(ConstantInt(Int32(job.config.target.air.patch))))
    air_md = MDNode(air_md)
    push!(metadata(mod)["air.version"], air_md)

    # add Metal language version
    air_lang_md = Metadata[]
    push!(air_lang_md, MDString("Metal"))
    push!(air_lang_md, Metadata(ConstantInt(Int32(job.config.target.metal.major))))
    push!(air_lang_md, Metadata(ConstantInt(Int32(job.config.target.metal.minor))))
    push!(air_lang_md, Metadata(ConstantInt(Int32(job.config.target.metal.patch))))
    air_lang_md = MDNode(air_lang_md)
    push!(metadata(mod)["air.language_version"], air_lang_md)

    # set sdk version
    sdk_version!(mod, job.config.target.macos)

    return
end


# intrinsics handling
#
# we don't have a proper back-end, so we're missing out on intrinsics-related functionality.

# AIR has no vector floating-point min/max intrinsic; only the scalar `air.fmin`/`air.fmax`
# exist. Julia's NaN-propagating `min`/`max` lower to `llvm.minimum`/`llvm.maximum` (and the
# non-propagating `llvm.minnum`/`llvm.maxnum`), which LLVM's vectorizers can widen to vector
# intrinsics. Lowering those directly would emit a nonexistent `air.fmin.v4f32`-style call, or
# hit the "Unsupported maximum/minimum type" error in the minimum/maximum handler below. So we
# scalarize each vector min/max into element-wise scalar intrinsic calls first and let the
# scalar lowering handle them — the same lowering LLVM itself uses on targets lacking a vector
# form, and semantically exact.
function scalarize_vector_minmax!(fun::LLVM.Function)
    minmax = LLVM.Intrinsic.(["llvm.minnum", "llvm.maxnum", "llvm.minimum", "llvm.maximum"])

    worklist = LLVM.CallBase[]
    for bb in blocks(fun), inst in instructions(bb)
        inst isa LLVM.CallBase || continue
        callee = called_operand(inst)
        (callee isa LLVM.Function && LLVM.isintrinsic(callee)) || continue
        LLVM.Intrinsic(callee) in minmax || continue
        value_type(inst) isa LLVM.VectorType || continue
        push!(worklist, inst)
    end
    isempty(worklist) && return false

    mod = LLVM.parent(fun)
    for call in worklist
        vecty = value_type(call)::LLVM.VectorType
        elty = eltype(vecty)
        # the scalar overload of the same intrinsic, e.g. llvm.minimum.v4f32 -> llvm.minimum.f32
        intr = LLVM.Intrinsic(called_operand(call))
        scalar_f = LLVM.Function(mod, intr, LLVMType[elty])
        scalar_ft = function_type(scalar_f)
        arg0, arg1 = arguments(call)
        @dispose builder=IRBuilder() begin
            position!(builder, call)
            debuglocation!(builder, call)
            res = PoisonValue(vecty)
            for i in 0:Int(length(vecty))-1
                idx = ConstantInt(LLVM.Int32Type(), i)
                a = extract_element!(builder, arg0, idx)
                b = extract_element!(builder, arg1, idx)
                s = call!(builder, scalar_ft, scalar_f, LLVM.Value[a, b])
                res = insert_element!(builder, res, s, idx)
            end
            replace_uses!(call, res)
            erase!(call)
        end
    end
    return true
end

# replace LLVM intrinsics with AIR equivalents
function lower_llvm_intrinsics!(@nospecialize(job::CompilerJob), fun::LLVM.Function)
    isdeclaration(fun) && return false

    # TODO: fastmath

    mod = LLVM.parent(fun)
    changed = false

    # AIR lacks vector min/max intrinsics; scalarize so the per-call lowering below applies.
    changed |= scalarize_vector_minmax!(fun)

    # determine worklist
    worklist = LLVM.CallBase[]
    for bb in blocks(fun), inst in instructions(bb)
        isa(inst, LLVM.CallBase) || continue

        call_fun = called_operand(inst)
        isa(call_fun, LLVM.Function) || continue
        LLVM.isintrinsic(call_fun) || continue

        push!(worklist, inst)
    end

    # lower intrinsics
    for call in worklist
        bb = LLVM.parent(call)
        call_fun = called_operand(call)
        call_ft = function_type(call_fun)
        intr = LLVM.Intrinsic(call_fun)

        # unsupported, but safe to remove
        unsupported_intrinsics = LLVM.Intrinsic.([
            "llvm.experimental.noalias.scope.decl",
            "llvm.lifetime.start",
            "llvm.lifetime.end",
            "llvm.assume"
        ])
        if intr in unsupported_intrinsics
            erase!(call)
            changed = true
        end

        # intrinsics that map straight to AIR
        mappable_intrinsics = Dict(
            # one argument
            LLVM.Intrinsic("llvm.abs")      => ("air.abs", true),
            LLVM.Intrinsic("llvm.fabs")     => ("air.fabs", missing),
            # two arguments
            LLVM.Intrinsic("llvm.umin")     => ("air.min", false),
            LLVM.Intrinsic("llvm.smin")     => ("air.min", true),
            LLVM.Intrinsic("llvm.umax")     => ("air.max", false),
            LLVM.Intrinsic("llvm.smax")     => ("air.max", true),
            LLVM.Intrinsic("llvm.minnum")   => ("air.fmin", missing),
            LLVM.Intrinsic("llvm.maxnum")   => ("air.fmax", missing),

        )
        if haskey(mappable_intrinsics, intr)
            fn, signed = mappable_intrinsics[intr]

            # determine type of the intrinsic
            typ = value_type(call)
            function type_suffix(typ)
                # XXX: can't we use LLVM to do this kind of mangling?
                if typ isa LLVM.IntegerType
                    "i$(width(typ))"
                elseif typ == LLVM.HalfType()
                    "f16"
                elseif typ == LLVM.FloatType()
                    "f32"
                elseif typ == LLVM.DoubleType()
                    "f64"
                elseif typ isa LLVM.VectorType
                    "v$(length(typ))$(type_suffix(eltype(typ)))"
                else
                    error("Unsupported intrinsic type: $typ")
                end
            end

            if typ isa LLVM.IntegerType || (typ isa LLVM.VectorType && eltype(typ) isa LLVM.IntegerType)
                fn *= "." * (signed::Bool ? "s" : "u") * "." * type_suffix(typ)
            else
                fn *= "." * type_suffix(typ)
            end

            new_intr = if haskey(functions(mod), fn)
                functions(mod)[fn]
            else
                LLVM.Function(mod, fn, call_ft)
            end
            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)

                new_value = call!(builder, call_ft, new_intr, arguments(call))
                replace_uses!(call, new_value)
                erase!(call)
                changed = true
            end
        end

        # copysign
        if intr == LLVM.Intrinsic("llvm.copysign")
            arg0, arg1 = operands(call)
            @assert value_type(arg0) == value_type(arg1)
            typ = value_type(call)

            # XXX: LLVM C API doesn't have getPrimitiveSizeInBits
            jltyp = if typ == LLVM.HalfType()
                Float16
            elseif typ == LLVM.FloatType()
                Float32
            elseif typ == LLVM.DoubleType()
                Float64
            else
                error("Unsupported copysign type: $typ")
            end

            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)

                # get bits
                typ′ = LLVM.IntType(8*sizeof(jltyp))
                arg0′ = bitcast!(builder, arg0, typ′)
                arg1′ = bitcast!(builder, arg1, typ′)

                # twiddle bits
                sign = and!(builder, arg1′, LLVM.ConstantInt(typ′, Base.sign_mask(jltyp)))
                mantissa = and!(builder, arg0′, LLVM.ConstantInt(typ′, ~Base.sign_mask(jltyp)))
                new_value = or!(builder, sign, mantissa)

                new_value = bitcast!(builder, new_value, typ)
                replace_uses!(call, new_value)
                erase!(call)
                changed = true
            end
        end

        # IEEE 754-2018 compliant maximum/minimum, propagating NaNs and treating -0 as less than +0
        if intr == LLVM.Intrinsic("llvm.minimum") || intr == LLVM.Intrinsic("llvm.maximum")
            typ = value_type(call)
            is_minimum = intr == LLVM.Intrinsic("llvm.minimum")

            # XXX: LLVM C API doesn't have getPrimitiveSizeInBits
            jltyp = if typ == LLVM.HalfType()
                Float16
            elseif typ == LLVM.FloatType()
                Float32
            elseif typ == LLVM.DoubleType()
                Float64
            else
                error("Unsupported maximum/minimum type: $typ")
            end

            # create a function that performs the IEEE-compliant operation.
            # normally we'd do this inline, but LLVM.jl doesn't have BB split functionality.
            new_intr_fn = if is_minimum
                "air.minimum.f$(8*sizeof(jltyp))"
            else
                "air.maximum.f$(8*sizeof(jltyp))"
            end

            if haskey(functions(mod), new_intr_fn)
                new_intr = functions(mod)[new_intr_fn]
            else
                new_intr = LLVM.Function(mod, new_intr_fn, call_ft)
                push!(function_attributes(new_intr), EnumAttribute("alwaysinline"))

                arg0, arg1 = parameters(new_intr)
                @assert value_type(arg0) == value_type(arg1)

                bb_check_arg0 = BasicBlock(new_intr, "check_arg0")
                bb_nan_arg0 = BasicBlock(new_intr, "nan_arg0")
                bb_check_arg1 = BasicBlock(new_intr, "check_arg1")
                bb_nan_arg1 = BasicBlock(new_intr, "nan_arg1")
                bb_check_zero = BasicBlock(new_intr, "check_zero")
                bb_compare_zero = BasicBlock(new_intr, "compare_zero")
                bb_fallback = BasicBlock(new_intr, "fallback")

                @dispose builder=IRBuilder() begin
                    # first, check if either argument is NaN, and return it if so

                    position!(builder, bb_check_arg0)
                    arg0_nan = fcmp!(builder, LLVM.API.LLVMRealUNO, arg0, arg0)
                    br!(builder, arg0_nan, bb_nan_arg0, bb_check_arg1)

                    position!(builder, bb_nan_arg0)
                    ret!(builder, arg0)

                    position!(builder, bb_check_arg1)
                    arg1_nan = fcmp!(builder, LLVM.API.LLVMRealUNO, arg1, arg1)
                    br!(builder, arg1_nan, bb_nan_arg1, bb_check_zero)

                    position!(builder, bb_nan_arg1)
                    ret!(builder, arg1)

                    # then, check if both arguments are zero and have a mismatching sign.
                    # if so, return in accordance to the intrinsic (minimum or maximum)

                    position!(builder, bb_check_zero)

                    typ′ = LLVM.IntType(8*sizeof(jltyp))
                    arg0′ = bitcast!(builder, arg0, typ′)
                    arg1′ = bitcast!(builder, arg1, typ′)

                    arg0_zero = fcmp!(builder, LLVM.API.LLVMRealUEQ, arg0,
                                      LLVM.ConstantFP(typ, zero(jltyp)))
                    arg1_zero = fcmp!(builder, LLVM.API.LLVMRealUEQ, arg1,
                                      LLVM.ConstantFP(typ, zero(jltyp)))
                    args_zero = and!(builder, arg0_zero, arg1_zero)
                    arg0_sign = and!(builder, arg0′, LLVM.ConstantInt(typ′, Base.sign_mask(jltyp)))
                    arg1_sign = and!(builder, arg1′, LLVM.ConstantInt(typ′, Base.sign_mask(jltyp)))
                    sign_mismatch = icmp!(builder, LLVM.API.LLVMIntNE, arg0_sign, arg1_sign)
                    relevant_zero = and!(builder, args_zero, sign_mismatch)
                    br!(builder, relevant_zero, bb_compare_zero, bb_fallback)

                    position!(builder, bb_compare_zero)
                    arg0_negative = icmp!(builder, LLVM.API.LLVMIntNE, arg0_sign,
                                          LLVM.ConstantInt(typ′, 0))
                    val = if is_minimum
                        select!(builder, arg0_negative, arg0, arg1)
                    else
                        select!(builder, arg0_negative, arg1, arg0)
                    end
                    ret!(builder, val)

                    # finally, it's safe to use the existing minnum/maxnum intrinsics

                    position!(builder, bb_fallback)
                    fallback_intr_fn = if is_minimum
                        "air.fmin.f$(8*sizeof(jltyp))"
                    else
                        "air.fmax.f$(8*sizeof(jltyp))"
                    end
                    fallback_intr = if haskey(functions(mod), fallback_intr_fn)
                        functions(mod)[fallback_intr_fn]
                    else
                        LLVM.Function(mod, fallback_intr_fn, call_ft)
                    end
                    val = call!(builder, call_ft, fallback_intr, collect(parameters(new_intr)))
                    ret!(builder, val)
                end
            end

            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)

                new_value = call!(builder, call_ft, new_intr, arguments(call))
                replace_uses!(call, new_value)
                erase!(call)
                changed = true
            end
        end
    end

    return changed
end

# annotate AIR intrinsics with optimization-related metadata
function annotate_air_intrinsics!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false

    for f in functions(mod)
        isdeclaration(f) || continue
        fn = LLVM.name(f)

        attrs = function_attributes(f)
        function add_attributes(names...)
            for name in names
                if LLVM.version() >= v"16" && name in ["argmemonly", "inaccessiblememonly",
                                                       "inaccessiblemem_or_argmemonly",
                                                       "readnone", "readonly", "writeonly"]
                    # XXX: workaround for changes from https://reviews.llvm.org/D135780
                    continue
                end
                push!(attrs, EnumAttribute(name, 0))
            end
            changed = true
        end

        # synchronization
        if fn == "air.wg.barrier" || fn == "air.simdgroup.barrier"
            add_attributes("nounwind", "mustprogress", "convergent", "willreturn")

        # atomics
        elseif match(r"air.atomic.(local|global).load", fn) !== nothing
            # TODO: "memory(argmem: read)" on LLVM 16+
            add_attributes("argmemonly", "readonly", "nounwind")
        elseif match(r"air.atomic.(local|global).store", fn) !== nothing
            # TODO: "memory(argmem: write)" on LLVM 16+
            add_attributes("argmemonly", "writeonly", "nounwind")
        elseif match(r"air.atomic.(local|global).(xchg|cmpxchg)", fn) !== nothing
            # TODO: "memory(argmem: readwrite)" on LLVM 16+
            add_attributes("argmemonly", "nounwind")
        elseif match(r"^air.atomic.(local|global).(add|sub|min|max|and|or|xor)", fn) !== nothing
            # TODO: "memory(argmem: readwrite)" on LLVM 16+
            add_attributes("argmemonly", "nounwind")

        # simdgroup
        elseif match(r"air.simdgroup_matrix_8x8_multiply_accumulate", fn) !== nothing
            add_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        elseif match(r"air.simdgroup_matrix_8x8_load", fn) !== nothing
            add_attributes("convergent", "mustprogress", "nofree", "nounwind", "readonly", "willreturn")
        elseif match(r"air.simdgroup_matrix_8x8_store", fn) !== nothing
            add_attributes("convergent", "mustprogress", "nounwind", "willreturn", "writeonly")

        # simd permute
        elseif match(r"air.simd_(ballot|all|vote_all|any|vote_any|shuffle|shuffle_xor|shuffle_down|\
            shuffle_up|shuffle_and_fill_down|shuffle_and_fill_up)", fn) !== nothing
            add_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        end
    end

    return changed
end
