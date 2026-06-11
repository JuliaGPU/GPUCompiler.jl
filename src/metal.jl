# implementation of the GPUCompiler interfaces for generating Metal code

const LLVMDowngrader_jll =
    LazyModule("LLVMDowngrader_jll",
               UUID("f52de702-fb25-5922-94ba-81dd59b07444"))


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

    # whether to use fast math; defaults to the process-wide `--math-mode=fast`. mirrors the
    # PTX target: when set, `apply_fastmath!` flags every floating-point op `afn`, which the
    # intrinsic lowering reads to pick the relaxed `air.fast_*` device functions over the
    # precise `air.*` ones (e.g. `air.fast_sqrt` instead of `air.sqrt`).
    fastmath::Bool = Base.JLOptions().fast_math == 1
end

# for backwards compatibility
MetalCompilerTarget(macos::VersionNumber) =
    MetalCompilerTarget(; macos, air=v"2.4", metal=v"2.4")

function Base.hash(target::MetalCompilerTarget, h::UInt)
    h = hash(target.macos, h)
    h = hash(target.air, h)
    h = hash(target.metal, h)
    h = hash(target.fastmath, h)
end

# the canonical text representation is AIR assembly, i.e. LLVM 14 era textual IR
source_code(target::MetalCompilerTarget) = "llvm"

# Metal is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::MetalCompilerTarget) = nothing

# Apple's toolchain encodes the AIR version in the architecture component of the triple, as
# `air64_v<major><minor>` (e.g. `air64_v26` for AIR 2.6). Tools like metal-opt derive the
# expected AIR version from the triple and complain when the `air.version` module metadata
# disagrees, so match the metadata's version here. Older Xcode toolchains used the plain,
# unversioned `air64`, so fall back to that for pre-2.6 targets.
function llvm_triple(target::MetalCompilerTarget)
    arch = if target.air >= v"2.6"
        "air64_v$(target.air.major)$(target.air.minor)"
    else
        "air64"
    end
    return "$arch-apple-macosx$(target.macos)"
end

llvm_datalayout(target::MetalCompilerTarget) =
    "e-p:64:64:64"*
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
    "-f32:32:32-f64:64:64"*
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"*
    "-n8:16:32"

llvm_targetinfo(::MetalCompilerTarget) = MetalTTI()

pass_by_value(job::CompilerJob{MetalCompilerTarget}) = false

# Apple GPUs have fused multiply-add, so `fma` should use the hardware instruction (lowered
# from `llvm.fma` to `air.fma`) rather than Julia's Float64-based `fma_emulated` fallback.
have_fma(@nospecialize(target::MetalCompilerTarget), T::Type) = true


## job

# the debug-info level is part of the slug (as on PTX): it changes the debug metadata emitted
# into the runtime library, and the device-exception reporters branch on it, so a library
# built at one level must not be reused at another.
runtime_slug(job::CompilerJob{MetalCompilerTarget}) =
    "metal-macos$(job.config.target.macos)-debuginfo=$(Int(llvm_debug_info(job)))"

isintrinsic(@nospecialize(job::CompilerJob{MetalCompilerTarget}), fn::String) =
    return startswith(fn, "air.")

# Re-type bfloat AIR intrinsic calls that arrive with `i16` operands to native `bfloat`.
#
# On Julia < 1.13 `BFloat16` has no native LLVM `bfloat` type and lowers to `i16`, so a
# `@typed_ccall` to a bfloat AIR intrinsic (e.g. `air.simdgroup_matrix_8x8_load.v64bf16.p1bf16`)
# arrives with `i16`/`<N x i16>` operands and, under typed pointers, an `i16*` operand -- a call
# whose operand types contradict the `bf16` its mangled name declares, which Apple's AIR back-end
# rejects. The `i16` already holds the exact bfloat bit pattern (BFloat16s stores the raw bits),
# so we repair these calls: swap the declaration for a native-`bfloat` one under the same AIR name
# and `bitcast` the i16 lanes (and, under typed pointers, the pointer) across the call boundary.
# On Julia 1.13+ the operands are already `bfloat`, so the signature is unchanged and this is a
# no-op. Keyed on the `bf16` type token in the AIR intrinsic name, so it covers every bfloat AIR
# intrinsic rather than an enumerated list. Runs pre-`annotate_air_intrinsics!` so the metadata
# annotation, which keys off the (unchanged) intrinsic name, lands on the re-typed declaration.
function promote_bf16_intrinsics!(mod::LLVM.Module)
    changed = false
    bf  = LLVM.BFloatType()
    i16 = LLVM.Int16Type()
    typed_ptrs = supports_typed_pointers(context())

    # the `bfloat` counterpart of an `i16`-flavored type; every other type is left untouched
    bfify(@nospecialize T) =
        if T == i16
            bf
        elseif T isa LLVM.VectorType && eltype(T) == i16
            LLVM.VectorType(bf, Int(length(T)))
        elseif typed_ptrs && T isa LLVM.PointerType && !is_opaque(T) && eltype(T) == i16
            LLVM.PointerType(bf, addrspace(T))
        else
            T
        end

    for old in collect(functions(mod))
        isdeclaration(old) || continue
        fn = LLVM.name(old)
        (startswith(fn, "air.") && occursin("bf16", fn)) || continue

        old_ft = function_type(old)
        old_params = collect(parameters(old_ft))
        new_params = LLVMType[bfify(T) for T in old_params]
        new_ret = bfify(return_type(old_ft))
        # already native `bfloat` (Julia 1.13+): nothing to repair
        (new_ret == return_type(old_ft) && new_params == old_params) && continue
        new_ft = LLVM.FunctionType(new_ret, new_params)

        # gather call sites before mutating the module
        worklist = LLVM.CallBase[]
        for use in uses(old)
            u = user(use)
            (u isa LLVM.CallBase && called_operand(u) === old) && push!(worklist, u)
        end

        # swap the `i16` declaration for a `bfloat`-typed one under the same AIR name
        LLVM.name!(old, fn * ".i16")
        new = LLVM.Function(mod, fn, new_ft)
        linkage!(new, linkage(old))

        for call in worklist
            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)
                args = LLVM.Value[let a = arg
                        value_type(a) == T ? a : bitcast!(builder, a, T)
                    end for (arg, T) in zip(arguments(call), new_params)]
                new_call = call!(builder, new_ft, new, args)
                if new_ret != LLVM.VoidType()
                    res = value_type(call) == new_ret ? new_call :
                          bitcast!(builder, new_call, value_type(call))
                    replace_uses!(call, res)
                end
                erase!(call)
            end
        end
        erase!(old)
        changed = true
    end
    return changed
end

function finish_linked_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module)
    # propagate `target.fastmath` as `@fastmath`-everywhere semantics, so the math-intrinsic
    # lowering in `finish_ir!` picks the relaxed `air.fast_*` functions. done here (post-link,
    # pre-optimize) so bodies pulled in from the runtime library get the flags too, mirroring
    # the PTX target.
    if job.config.target.fastmath
        apply_fastmath!(mod)
    end

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

    # re-type bfloat AIR intrinsic calls that lowered to `i16` (Julia < 1.13) to native `bfloat`,
    # before annotation so the name-keyed metadata lands on the re-typed declaration
    promote_bf16_intrinsics!(mod)

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

# aggregate load splitting (JuliaGPU/Metal.jl#792)
#
# Julia can emit a single by-value `load` of a large, deeply-nested aggregate (e.g. an
# Oceananigans `RectilinearGrid` passed by reference) that feeds several `extractvalue`s.
# Apple's AGX back-end crashes during native-code generation when lowering such a wide
# aggregate load. We rewrite each `extractvalue (load p), idxs` into a narrow field load
# `load (inbounds_gep p, 0, idxs)` and delete the now-dead wide load, so only per-field
# loads reach the back-end.
#
# This is exactly LLVM's own `extractvalue (load)` -> `load (gep)` fold in InstCombine
# (visitExtractValueInst), with one difference: InstCombine guards it on the load having a
# single use, declining multiply-used loads as "a struct with padding [where] we don't want
# to do the transformation as it loses padding knowledge". That guard is a codegen
# heuristic (one wide load can be cheaper than N field loads), not a correctness condition,
# so dropping it is sound — and necessary here, since the crashing pattern is precisely a
# multiply-used aggregate load that InstCombine therefore leaves intact.
#
# Restricted to simple (non-volatile, non-atomic) loads all of whose users are
# `extractvalue` — the by-value-aggregate-argument pattern — so the wide load can be fully
# eliminated. Like LLVM's fold, the field loads take their type's natural (ABI) alignment,
# valid because the aggregate base load is at least that aligned, and AA metadata is copied
# from the wide load (sound for the narrower field loads it subsumes).
function split_aggregate_loads!(mod::LLVM.Module)
    aa_kinds = (LLVM.MD_tbaa, LLVM.MD_tbaa_struct, LLVM.MD_alias_scope, LLVM.MD_noalias)
    changed = false
    for f in functions(mod)
        isdeclaration(f) && continue
        worklist = LLVM.LoadInst[]
        for bb in blocks(f), inst in instructions(bb)
            inst isa LLVM.LoadInst || continue
            T = value_type(inst)
            (T isa LLVM.StructType || T isa LLVM.ArrayType) || continue
            iszero(LLVM.API.LLVMGetVolatile(inst)) || continue
            LLVM.API.LLVMGetOrdering(inst) == LLVM.API.LLVMAtomicOrderingNotAtomic || continue
            uselist = collect(uses(inst))
            isempty(uselist) && continue
            all(u -> user(u) isa LLVM.ExtractValueInst, uselist) || continue
            push!(worklist, inst)
        end
        for ld in worklist
            ptr = operands(ld)[1]
            aggty = value_type(ld)
            md = metadata(ld)
            i32 = LLVM.Int32Type()
            @dispose builder=IRBuilder() begin
                # build the field loads at the wide load's location, not the extractvalue's
                position!(builder, ld)
                for u in collect(uses(ld))
                    ev = user(u)::LLVM.ExtractValueInst
                    n = LLVM.API.LLVMGetNumIndices(ev)
                    idxptr = LLVM.API.LLVMGetIndices(ev)
                    # extractvalue has integer indices; getelementptr takes Values, prefixed
                    # with an i32 0 to step through the pointer to the aggregate's first element.
                    gepidx = LLVM.Value[ConstantInt(i32, 0)]
                    for k in 1:n
                        push!(gepidx, ConstantInt(i32, unsafe_load(idxptr, k)))
                    end
                    gep = inbounds_gep!(builder, aggty, ptr, gepidx)
                    fieldload = load!(builder, value_type(ev), gep)
                    for kind in aa_kinds
                        haskey(md, kind) && (metadata(fieldload)[kind] = md[kind])
                    end
                    replace_uses!(ev, fieldload)
                    erase!(ev)
                end
            end
            erase!(ld)
            changed = true
        end
    end
    return changed
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

        # split multiply-used by-value aggregate loads into narrow per-field loads; the AGX
        # back-end crashes during native codegen on wide aggregate loads (#792).
        split_aggregate_loads!(mod)

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

        add_globals_metadata!(job, mod)

        add_module_metadata!(job, mod)
    end

    return functions(mod)[entry_fn]
end

# lowering of LLVM IR to AIR-compatible IR
#
# Metal does not have an LLVM back-end, so the lowering of target-independent LLVM IR into
# target-specific constructs -- something that normally happens during instruction
# selection -- is implemented here as IR-to-IR rewrites, run at the start of `mcgen`.
# this keeps the `:llvm` output (e.g. `code_llvm`) close to what Julia generated, using
# generic LLVM intrinsics, while the `:asm`/`:obj` outputs contain AIR intrinsics.
function lower_air!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module)
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
        # lowering may have introduced additional functions marked `alwaysinline`,
        # and left dead declarations of the replaced LLVM intrinsics behind
        @dispose pb=NewPMPassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            add!(pb, NewPMFunctionPassManager()) do fpm
                add!(fpm, SimplifyCFGPass())
                add!(fpm, instcombine_pass(job))
            end
            add!(pb, StripDeadPrototypesPass())
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

    return
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
    # lower LLVM constructs that the AIR back-end does not support; this takes the place
    # of instruction selection, as our LLVM does not have a Metal target machine.
    lower_air!(job, mod)

    if !isavailable(LLVMDowngrader_jll)
        error("Metal machine-code generation requires the LLVMDowngrader_jll package, which should be installed and loaded first.")
    end

    # downgrade to AIR. Metal's metallib loader is a backward-compatible reader that accepts
    # real LLVM <= 15 bitcode; target LLVM 14 (typed pointers, but with native `bfloat` —
    # unlike the 5.0/7.0 targets) so BFloat16 kernels compile on Julia 1.13+, where Julia
    # emits the native `bfloat` IR type (JuliaGPU/Metal.jl#817). `llvm-downgrade` reads
    # bitcode, so hand it the module's bitcode rather than its textual form.
    bitcode = let io = IOBuffer()
        write(io, mod)
        take!(io)
    end
    air = run_tool(`$(LLVMDowngrader_jll.llvm_downgrade()) --bitcode-version=14.0 -o - -`, bitcode)

    if format == LLVM.API.LLVMAssemblyFile
        # disassemble the bitcode again to AIR assembly, i.e. LLVM 14 era textual IR
        String(run_tool(`$(LLVMDowngrader_jll.llvm_dis_14()) -o - -`, air))
    else
        air
    end
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
# This pass is the interprocedural complement (Phase 2, `propagate_argument_address_spaces_once!`).
# When every caller passes the same kind of value for a generic pointer parameter,
# `addrspacecast(<ptr in a specific space> -> generic)`, it retargets the parameter to that space,
# drops the casts at the call sites, and casts back to generic on entry so the body is unchanged.
# That only relocates a side-effect-free cast across the boundary, so it is trivially correct; the
# following `InferAddressSpaces` run folds the entry cast away. The source need not be a constant
# global; any pointer with a known address space qualifies, so any back-end can run it.
#
# Narrowing one function makes its body forward an `addrspacecast`-from-specific to the functions
# it calls, exposing them in turn, so we iterate to a fixed point; a constant thus reaches an
# arbitrarily deep callee (e.g. an exception reporter that delegates to another) regardless of the
# order functions are visited in. This terminates: each sweep that changes anything strictly
# reduces the number of narrowable generic-pointer parameters, and narrowing never introduces one.
#
# TYPED-POINTER SHIM (Julia <= 1.11) -- delete `convert_intptr_args!` and its call in
# `propagate_argument_address_spaces!`, along with everything else tagged "typed-pointer shim",
# once 1.12 is the minimum. Before JuliaLang/julia#53687 (`v"1.12.0-DEV.225"`) a `Ptr` argument is
# lowered to an integer rather than a pointer -- a separate switch from LLVM's typed/opaque
# pointers (so the gate keys off the version, not `supports_typed_pointers`; see the gate below).
# The boundary crossing then arrives as `ptrtoint(addrspacecast(<specific> ->
# generic))` and the parameter is an `iN` the body either `inttoptr`s (a leaf reporter that
# dereferences it) or forwards on (a delegator, e.g. `report_exception_name` -> `report_exception`).
# Rather than teach the narrowing above about integers, a separate first phase (Phase 1,
# `convert_intptr_args!`) canonicalizes these back to generic pointers: it rewrites each such
# integer parameter to a generic pointer and strips the `ptrtoint` at the call sites (entry
# rebuilds the original integer as `ptrtoint(param)`; the cloned `inttoptr` then composes with it
# and folds away). It is iterated to a fixed point too, so a forwarder is de-integerized before the
# leaf it feeds re-exposes the `ptrtoint(<generic pointer>)` shape. After Phase 1 every `Ptr`
# parameter is an ordinary generic pointer -- the same shape as under opaque pointers -- so Phase 2
# is identical for both. (Without this the leftover generic-space load is miscompiled by the
# LLVM-16 Metal bitcode downgrade into an invalid metallib -- JuliaGPU/Metal.jl device exceptions
# on Julia 1.11.)

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

# Typed-pointer shim (Julia <= 1.11) -- delete with the rest of the shim once 1.12 is the minimum.
# If `v` is `ptrtoint` of a generic (address space 0) pointer, return that pointer; otherwise
# `nothing`. This is the integer image of a `Ptr` argument at a call site: `ptrtoint(addrspacecast(
# <specific> -> generic))` from a direct caller, or `ptrtoint(<the caller's own retargeted generic
# pointer>)` once a forwarder upstream has been de-integerized.
function generic_ptr_behind_ptrtoint(@nospecialize(v))
    (v isa LLVM.Instruction || v isa LLVM.ConstantExpr) || return nothing
    opcode(v) == LLVM.API.LLVMPtrToInt || return nothing
    p = operands(v)[1]
    (value_type(p) isa LLVM.PointerType && addrspace(value_type(p)) == 0) || return nothing
    return p
end

# Typed-pointer shim (Julia <= 1.11) -- remove once 1.12 is the minimum. Classify integer
# parameter `arg` as the integer image of a pointer that crossed a call boundary, returning the
# generic pointer type to reconstruct it to (so it can be retargeted like a generic pointer
# parameter; see `propagate_argument_address_spaces!`), or `nothing` if it is not safely a
# pointer image. It qualifies when every use is either
#   * an `inttoptr` to the generic space -- the leaf shape, where the body dereferences it (all
#     such uses must agree on the result type, which pins the reconstructed pointee); or
#   * a call argument -- the delegation shape, where the body forwards it on unchanged.
# A purely-forwarding parameter has no `inttoptr` to pin the pointee, so a canonical generic
# pointer is used: every boundary is a `bitcast`/`ptrtoint`, so the choice only affects the
# bridging casts, not the reconstructed value. Any other use (arithmetic, comparison, storing
# the integer, ...) means it is genuinely an integer, so it is left alone -- narrowing it would
# be value-preserving but pointless, and we have no pointee to reconstruct to.
function integer_param_pointer_image_type(arg::LLVM.Argument)
    ptrty = nothing
    forwarded = false
    for use in uses(arg)
        u = user(use)
        if u isa LLVM.Instruction && opcode(u) == LLVM.API.LLVMIntToPtr
            t = value_type(u)
            (t isa LLVM.PointerType && addrspace(t) == 0) || return nothing
            ptrty === nothing ? (ptrty = t) : (ptrty == t || return nothing)
        elseif u isa LLVM.CallInst
            forwarded = true
        else
            return nothing
        end
    end
    ptrty !== nothing && return ptrty
    # a pure forwarder has no `inttoptr` to pin the pointee, so use a canonical generic pointer
    # (opaque `ptr`, or `i8*` under typed pointers)
    forwarded && return supports_typed_pointers(context()) ? LLVM.PointerType(LLVM.Int8Type()) :
                                                             LLVM.PointerType()
    return nothing
end

# the direct call sites of `f`, or `nothing` if any use is not a direct call we can rewrite.
# rewriting a signature is only sound with no callers outside the module; by `finish_ir!` the
# pipeline has internalized everything but the kernel entrypoints, so the runtime helpers qualify.
function direct_callsites(f::LLVM.Function)
    callsites = LLVM.CallInst[]
    for use in uses(f)
        v = user(use)
        (v isa LLVM.CallInst && called_operand(v) == f) || return nothing
        push!(callsites, v)
    end
    return isempty(callsites) ? nothing : callsites
end

# a function whose signature we may rewrite: it has a body and local (internal/private) linkage.
retargetable(f::LLVM.Function) =
    !isempty(blocks(f)) &&
    linkage(f) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage)

# retarget a pointer type to address space `as`, taking its pointee from `srcptr` (only needed for
# typed pointers; `eltype` is invalid on opaque ones, so keep it lazy)
retarget_pointer(as::Integer, srcptr::LLVM.PointerType) =
    supports_typed_pointers(context()) ? LLVM.PointerType(eltype(srcptr), as) :
                                         LLVM.PointerType(as)

# the single source address space every call site's argument `i` is reached from via `extract`,
# or `-1` if they disagree or any does not have the expected shape.
function agreed_source_addrspace(callsites, i, extract)
    as = -1
    for cs in callsites
        src = extract(arguments(cs)[i])
        src === nothing && return -1
        src_as = addrspace(value_type(src))
        as == -1 ? (as = src_as) : (as == src_as || return -1)
    end
    return as
end

# typed-pointer shim (Julia <= 1.11): bridge a pointee-type mismatch when unwrapping the integer
bitcast_if_needed(builder, v, t) = value_type(v) == t ? v : bitcast!(builder, v, t)

# Phase 1 (typed-pointer shim, Julia <= 1.11): a single de-integerization sweep. With typed
# pointers a `Ptr` argument is lowered to an integer; rewrite every internal parameter that is the
# integer image of a pointer (used only via `inttoptr` or forwarded on) back to a generic pointer,
# stripping the `ptrtoint` at the call sites. Iterated to a fixed point in `convert_intptr_args!`
# so a forwarder is de-integerized before the leaf it feeds. Returns whether anything changed.
function convert_intptr_args_once!(mod::LLVM.Module)
    changed = false
    for f in collect(functions(mod))
        retargetable(f) || continue
        callsites = direct_callsites(f)
        callsites === nothing && continue
        param_types = parameters(function_type(f))
        new_types = Vector{Any}(nothing, length(param_types))
        for (i, pty) in enumerate(param_types)
            pty isa LLVM.IntegerType || continue
            ptrty = integer_param_pointer_image_type(parameters(f)[i])
            ptrty === nothing && continue
            # only when every caller already passes `ptrtoint(<generic pointer>)` we can unwrap
            all(cs -> generic_ptr_behind_ptrtoint(arguments(cs)[i]) !== nothing, callsites) || continue
            new_types[i] = ptrty
        end
        any(!isnothing, new_types) || continue
        rewrite_parameters!(mod, f, callsites; new_types,
            rebuild_entry = (b, p, i) -> ptrtoint!(b, p, param_types[i]),
            rewrite_arg   = (b, a, i) -> bitcast_if_needed(b, generic_ptr_behind_ptrtoint(a), new_types[i]),
            keep_attrs    = false)
        changed = true
    end
    return changed
end

# typed-pointer shim (Julia <= 1.11) -- delete with the rest of the shim once 1.12 is the minimum.
function convert_intptr_args!(mod::LLVM.Module)
    changed = false
    while convert_intptr_args_once!(mod)
        changed = true
    end
    return changed
end

# Phase 2: a single address-space narrowing sweep. Retarget every generic pointer parameter that
# all callers feed `addrspacecast(<specific> -> generic)` from the same space, to that space.
# Returns whether anything changed.
function propagate_argument_address_spaces_once!(mod::LLVM.Module)
    changed = false
    for f in collect(functions(mod))
        retargetable(f) || continue
        callsites = direct_callsites(f)
        callsites === nothing && continue
        param_types = parameters(function_type(f))
        new_types = Vector{Any}(nothing, length(param_types))
        for (i, pty) in enumerate(param_types)
            (pty isa LLVM.PointerType && addrspace(pty) == 0) || continue
            as = agreed_source_addrspace(callsites, i, addrspacecast_to_generic_source)
            as > 0 && (new_types[i] = retarget_pointer(as, pty))
        end
        any(!isnothing, new_types) || continue
        rewrite_parameters!(mod, f, callsites; new_types,
            rebuild_entry = (b, p, i) -> addrspacecast!(b, p, param_types[i]),
            rewrite_arg   = (b, a, i) -> addrspacecast_to_generic_source(a),
            keep_attrs    = true)
        changed = true
    end
    return changed
end

# interprocedural address-space narrowing (see the comment above). Under typed pointers, first
# canonicalize integer-image `Ptr` parameters to generic pointers (Phase 1) so the narrowing
# (Phase 2) needs no integer handling; both run to a fixed point. Returns whether anything changed.
function propagate_argument_address_spaces!(mod::LLVM.Module)
    changed = false
    # the shim is needed exactly when Julia lowers `Ptr` arguments to integers: before
    # JuliaLang/julia#53687 (`v"1.12.0-DEV.225"`). that is a separate switch from LLVM's
    # typed/opaque pointers, so gate on the version, not `supports_typed_pointers` -- they agree
    # on releases but can diverge (opaque pointers could be enabled while `Ptr` is still an integer).
    if VERSION < v"1.12.0-DEV.225"
        changed |= convert_intptr_args!(mod)
    end
    while propagate_argument_address_spaces_once!(mod)
        changed = true
    end
    return changed
end

# Clone `f`, retargeting each parameter `i` for which `new_types[i] !== nothing` to that type. A
# retargeted parameter's cloned body must keep seeing a value of the original type, so a fresh
# entry block rebuilds it via `rebuild_entry(builder, new_param, i)` -- only a value-preserving
# cast or round-trip relocated across the boundary -- and the body is cloned to use that (a later
# `InferAddressSpaces`/instcombine folds it away). Each call site's argument `i` is replaced by
# `rewrite_arg(builder, old_arg, i)`. Parameter and call-site argument attributes survive only
# where `keep_attrs`; dropping them lets the typed-pointer shim shed integer attributes (e.g.
# `zeroext`) that are invalid once the parameter is a pointer. Direct and self-recursive calls are
# rewritten; `f` is replaced by the clone and erased.
function rewrite_parameters!(mod::LLVM.Module, f::LLVM.Function, callsites;
                             new_types::Vector, rebuild_entry, rewrite_arg, keep_attrs::Bool)
    new_f = clone_with_converted_args!(mod, f, new_types, rebuild_entry)

    # `clone_into!` copies a parameter's attributes only when it maps to a new argument; the
    # retargeted ones map to the entry rebuild instead, so theirs are dropped. Reattach them where
    # the caller keeps them (still valid on a narrowed pointer); drop them otherwise.
    for i in 1:length(new_types)
        (new_types[i] !== nothing && keep_attrs) || continue
        for attr in collect(parameter_attributes(f, i))
            push!(parameter_attributes(new_f, i), attr)
        end
    end

    # a (directly) recursive `f` has self-calls that cloning retargeted to `new_f` but left with
    # the old signature; collect them from the clone first, since the rewritten calls also target
    # `new_f` and must not be revisited.
    self_calls = LLVM.CallInst[]
    for bb in blocks(new_f), inst in instructions(bb)
        inst isa LLVM.CallInst && called_operand(inst) == new_f && push!(self_calls, inst)
    end

    new_ft = function_type(new_f)
    @dispose builder=IRBuilder() begin
        for cs in Iterators.flatten((callsites, self_calls))
            rewrite_retargeted_call!(builder, cs, new_f, new_ft, new_types, rewrite_arg, keep_attrs)
        end
    end

    return replace_function!(f, new_f)
end

# rewrite a single call to target `new_f`/`new_ft`: each retargeted argument `i` becomes
# `rewrite_arg(builder, old_arg, i)`, others pass through. Preserves calling convention, operand
# bundles and attributes, except retargeted arguments drop their attributes where `!keep_attrs`.
function rewrite_retargeted_call!(builder::IRBuilder, cs::LLVM.CallInst, new_f::LLVM.Function,
                                  new_ft::LLVM.FunctionType, new_types::Vector, rewrite_arg,
                                  keep_attrs::Bool)
    position!(builder, cs)
    new_args = LLVM.Value[new_types[i] === nothing ? arg : rewrite_arg(builder, arg, i)
                          for (i, arg) in enumerate(arguments(cs))]
    new_call = call!(builder, new_ft, new_f, new_args, operand_bundles(cs))
    callconv!(new_call, callconv(cs))
    for attr in collect(function_attributes(cs))
        push!(function_attributes(new_call), attr)
    end
    for attr in collect(return_attributes(cs))
        push!(return_attributes(new_call), attr)
    end
    for i in 1:length(arguments(cs))
        (new_types[i] === nothing || keep_attrs) || continue
        for attr in collect(argument_attributes(cs, i))
            push!(argument_attributes(new_call, i), attr)
        end
    end
    replace_uses!(cs, new_call)
    erase!(cs)
    return new_call
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

# global metadata generation
#
# module metadata is used to identify global buffers that are used as kernel arguments.
function add_globals_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    # Iterate through arguments and create metadata for them
    globs = globals(mod)
    dl = datalayout(mod)

    i = 1
    for gv in globs
        gv_typ = global_value_type(gv)
        (isconstant(gv) && gv_typ isa LLVM.PointerType && addrspace(gv_typ) == 3) || continue

        global_infos = Metadata[]

        push!(global_infos, MDString("air.global_binding"))
        push!(global_infos, Metadata(gv))

        md = Metadata[]

        # argument index
        push!(md, Metadata(ConstantInt(Int32(-1))))

        push!(md, MDString("air.buffer"))

        push!(md, MDString("air.location_index"))
        push!(md, Metadata(ConstantInt(Int32(i-1))))

        # XXX: unknown
        push!(md, Metadata(ConstantInt(Int32(1))))

        push!(md, MDString("air.read_write")) # TODO: Check for const array

        push!(md, MDString("air.address_space"))
        push!(md, Metadata(ConstantInt(Int32(addrspace(global_value_type(gv))))))

        arg_type_name, arg_type_size = if !is_opaque(gv_typ)
            string(eltype(gv_typ)), Int(sizeof(dl, eltype(gv_typ)))
        else
            string(gv_typ), Int(sizeof(dl, gv_typ))
        end

        push!(md, MDString("air.arg_type_size"))
        push!(md, Metadata(ConstantInt(Int32(arg_type_size))))

        push!(md, MDString("air.arg_type_align_size"))
        push!(md, Metadata(ConstantInt(Int32(alignment(gv)))))

        push!(md, MDString("air.arg_type_name"))
        push!(md, MDString(arg_type_name))

        push!(md, MDString("air.arg_name"))
        push!(md, MDString(String(LLVM.name(gv))))

        push!(global_infos, MDNode(md))

        push!(metadata(mod)["air.global_bindings"], MDNode(global_infos))

        i += 1
    end

    return
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
        arg.idx === nothing && continue
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

    # record the compile options Apple's frontend emits. each option is a single-string node
    # under the `air.compile_options` named metadata. denorms and framebuffer fetch match
    # Apple's defaults; fast math tracks `target.fastmath`, which also drives whether the math
    # intrinsics lower to the relaxed `air.fast_*` device functions.
    for option in ["air.compile.denorms_disable",
                   job.config.target.fastmath ? "air.compile.fast_math_enable" :
                                                "air.compile.fast_math_disable",
                   "air.compile.framebuffer_fetch_enable"]
        push!(metadata(mod)["air.compile_options"], MDNode([MDString(option)]))
    end

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

# floating-point math intrinsics that Julia emits as plain `llvm.*` and that Metal exposes as
# AIR device functions. Each has a precise `air.<op>` for f16/f32; some additionally have a
# relaxed, f32-only `air.fast_<op>` that we select when the call is `afn`-flagged — set per-op
# by `@fastmath` or module-wide by `apply_fastmath!` when `target.fastmath` is on.
#
# This is the back-end half of the "front-end emits LLVM, back-end lowers" design (cf. the PTX
# target's fast-math passes): it lets Metal.jl drop its hand-written `air.*`/`air.fast_*`
# overrides for these ops and rely on the LLVM intrinsics Julia already generates. `round` is
# covered too — Julia lowers it to `llvm.rint` (round-to-even).
function lower_math_intrinsics!(fun::LLVM.Function)
    # llvm intrinsic => (precise air op, relaxed f32 air op or `nothing`)
    # Verified against Apple's frontend (`xcrun metal -S -emit-llvm`, precise vs -ffast-math):
    # every op has an `air.<op>.f16` and `air.<op>.f32`; all but `fma` also have an f32-only
    # `air.fast_<op>` that Apple selects under fast math. Half always stays precise, and `fma`
    # is exact so even fast math keeps `air.fma.{f16,f32}`.
    math_intrinsics = Dict(
        LLVM.Intrinsic("llvm.sqrt")  => ("air.sqrt",  "air.fast_sqrt"),
        LLVM.Intrinsic("llvm.fma")   => ("air.fma",   nothing),
        LLVM.Intrinsic("llvm.floor") => ("air.floor", "air.fast_floor"),
        LLVM.Intrinsic("llvm.ceil")  => ("air.ceil",  "air.fast_ceil"),
        LLVM.Intrinsic("llvm.trunc") => ("air.trunc", "air.fast_trunc"),
        LLVM.Intrinsic("llvm.rint")  => ("air.rint",  "air.fast_rint"),
    )

    worklist = Tuple{LLVM.CallBase, String, Union{String,Nothing}}[]
    for bb in blocks(fun), inst in instructions(bb)
        inst isa LLVM.CallBase || continue
        callee = called_operand(inst)
        (callee isa LLVM.Function && LLVM.isintrinsic(callee)) || continue
        mapping = get(math_intrinsics, LLVM.Intrinsic(callee), nothing)
        mapping === nothing && continue
        # Metal floats are f16/f32 only; skip f64 (rejected by validate_ir) and vector types
        # (these ops have no `air.<op>.v4f32`) rather than synthesize a nonexistent intrinsic.
        typ = value_type(inst)
        (typ == LLVM.HalfType() || typ == LLVM.FloatType()) || continue
        push!(worklist, (inst, mapping[1], mapping[2]))
    end
    isempty(worklist) && return false

    mod = LLVM.parent(fun)
    fns = functions(mod)
    for (call, precise, fast) in worklist
        typ = value_type(call)
        # the relaxed variant exists for f32 only; f16 always uses the precise op
        use_fast = fast !== nothing && typ == LLVM.FloatType() && LLVM.fast_math(call).afn
        suffix = typ == LLVM.HalfType() ? "f16" : "f32"
        fn = "$(use_fast ? fast : precise).$suffix"
        ft = function_type(called_operand(call))
        air = haskey(fns, fn) ? fns[fn] : LLVM.Function(mod, fn, ft)
        @dispose builder=IRBuilder() begin
            position!(builder, call)
            debuglocation!(builder, call)
            new_value = call!(builder, ft, air, arguments(call))
            replace_uses!(call, new_value)
            erase!(call)
        end
    end
    return true
end

# Fuse chained integer min/max into AIR's native 3-way builtins: a 2-way
# `air.{min,max}.{s,u}.iN` whose operand is a single-use call to the same builtin becomes
# `air.{min,max}3.{s,u}.iN(a, b, c)`. AGX has 3-way min/max, but neither Julia (which reduces
# `min(a,b,c)` to nested 2-arg calls) nor Apple's own frontend emits it. Done in the back-end so
# every chained min/max benefits, not just literal 3-argument calls. Integer only: float min/max
# go through the NaN-propagating wrapper, which `air.f{min,max}3` would not preserve.
function fuse_minmax3!(fun::LLVM.Function)
    mod = LLVM.parent(fun)
    pat = r"^air\.(min|max)\.(s|u)\.i(8|16|32|64)$"
    changed = false

    # fold one pair then rescan: each fold removes a call, so this terminates, and rescanning
    # avoids mutating the instruction stream while iterating it.
    while true
        fold = nothing  # (outer, inner, other_operand)
        for bb in blocks(fun), outer in instructions(bb)
            outer isa LLVM.CallInst || continue
            oc = called_operand(outer)
            (oc isa LLVM.Function && match(pat, LLVM.name(oc)) !== nothing) || continue
            oargs = arguments(outer)
            length(oargs) == 2 || continue
            for i in 1:2
                inner = oargs[i]
                inner isa LLVM.CallInst || continue
                ic = called_operand(inner)
                # same builtin (name pins down min/max, signedness and width)
                (ic isa LLVM.Function && LLVM.name(ic) == LLVM.name(oc)) || continue
                # inner must feed only this outer, else folding it would drop a live value
                length(collect(uses(inner))) == 1 || continue
                fold = (outer, inner, oargs[i == 1 ? 2 : 1])
                break
            end
            fold === nothing || break
        end
        fold === nothing && break

        outer, inner, other = fold
        m = match(pat, LLVM.name(called_operand(outer)))
        fn3 = "air.$(m[1])3.$(m[2]).i$(m[3])"
        T = value_type(outer)
        ft3 = LLVM.FunctionType(T, LLVMType[T, T, T])
        f3 = haskey(functions(mod), fn3) ? functions(mod)[fn3] : LLVM.Function(mod, fn3, ft3)
        iargs = arguments(inner)
        @dispose builder=IRBuilder() begin
            position!(builder, outer)
            debuglocation!(builder, outer)
            v = call!(builder, ft3, f3, LLVM.Value[iargs[1], iargs[2], other])
            replace_uses!(outer, v)
            erase!(outer)
            erase!(inner)   # now dead (its only use was `outer`)
        end
        changed = true
    end
    return changed
end

# replace LLVM intrinsics with AIR equivalents
function lower_llvm_intrinsics!(@nospecialize(job::CompilerJob), fun::LLVM.Function)
    isdeclaration(fun) && return false

    mod = LLVM.parent(fun)
    changed = false

    # AIR lacks vector min/max intrinsics; scalarize so the per-call lowering below applies.
    changed |= scalarize_vector_minmax!(fun)

    # lower the floating-point math intrinsics Julia emits (sqrt, fma, floor, ...) to their
    # AIR device functions, picking the relaxed `air.fast_*` variant for `afn`-flagged calls.
    changed |= lower_math_intrinsics!(fun)

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

            # AIR has no native bfloat fabs/fmin/fmax (MSL promotes bfloat to float for
            # them), so do the same: call the float intrinsic on fpext'd operands and
            # fptrunc the result back. `optyp` is the type the AIR call actually uses.
            bf = LLVM.BFloatType()
            promote_bf = typ == bf || (typ isa LLVM.VectorType && eltype(typ) == bf)
            optyp = if typ == bf
                LLVM.FloatType()
            elseif promote_bf
                LLVM.VectorType(LLVM.FloatType(), Int(length(typ)))
            else
                typ
            end
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

            if optyp isa LLVM.IntegerType || (optyp isa LLVM.VectorType && eltype(optyp) isa LLVM.IntegerType)
                fn *= "." * (signed::Bool ? "s" : "u") * "." * type_suffix(optyp)
            else
                fn *= "." * type_suffix(optyp)
            end

            # AIR's value intrinsics take only the value operands. `llvm.abs` carries an extra
            # `i1 is_int_min_poison` flag that `air.abs` does not, so drop any operand whose
            # type isn't the result type before building the call. (For the others every
            # operand is the result type, so this is a no-op.)
            air_args = LLVM.Value[a for a in arguments(call) if value_type(a) == typ]
            air_ft = LLVM.FunctionType(optyp, LLVMType[optyp for _ in air_args])
            new_intr = if haskey(functions(mod), fn)
                functions(mod)[fn]
            else
                LLVM.Function(mod, fn, air_ft)
            end
            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)

                call_args = promote_bf ?
                    LLVM.Value[fpext!(builder, a, optyp) for a in air_args] : air_args
                new_value = call!(builder, air_ft, new_intr, call_args)
                if promote_bf
                    new_value = fptrunc!(builder, new_value, typ)
                end
                replace_uses!(call, new_value)
                erase!(call)
                changed = true
            end
        end

        # integer bit intrinsics: pure renames to AIR's builtin names (same signature, including
        # the `i1` on clz/ctz). Apple's frontend emits these `air.*` rather than the `llvm.*`
        # forms, so we rename rather than rely on the metallib loader accepting `llvm.*`.
        bit_renames = Dict(
            LLVM.Intrinsic("llvm.ctlz")       => ("llvm.ctlz",       "air.clz"),
            LLVM.Intrinsic("llvm.cttz")       => ("llvm.cttz",       "air.ctz"),
            LLVM.Intrinsic("llvm.ctpop")      => ("llvm.ctpop",      "air.popcount"),
            LLVM.Intrinsic("llvm.bitreverse") => ("llvm.bitreverse", "air.reverse_bits"),
        )
        if haskey(bit_renames, intr)
            llvm_base, air_base = bit_renames[intr]
            # keep the mangled type suffix, e.g. llvm.ctlz.i32 -> air.clz.i32
            fn = air_base * LLVM.name(call_fun)[length(llvm_base)+1:end]
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

            # AIR has no bfloat min/max, so promote to float as MSL does: build the wrapper
            # in float and fpext/fptrunc around it. `optyp` is the type it operates on.
            promote_bf = typ == LLVM.BFloatType()
            optyp = promote_bf ? LLVM.FloatType() : typ
            op_ft = LLVM.FunctionType(optyp, LLVMType[optyp, optyp])

            # XXX: LLVM C API doesn't have getPrimitiveSizeInBits
            jltyp = if optyp == LLVM.HalfType()
                Float16
            elseif optyp == LLVM.FloatType()
                Float32
            elseif optyp == LLVM.DoubleType()
                Float64
            else
                error("Unsupported maximum/minimum type: $typ")
            end

            # @fastmath / fastmath=true set `nnan` (assume no NaNs), so we can skip the
            # NaN-propagating wrapper and call the relaxed AIR builtin directly, matching Apple's
            # -ffast-math: f32 has air.fast_f{min,max}; f16 has no fast form, so use air.f{min,max}.
            fast_fn = if !LLVM.fast_math(call).nnan
                nothing
            elseif optyp == LLVM.FloatType()
                is_minimum ? "air.fast_fmin.f32" : "air.fast_fmax.f32"
            else
                is_minimum ? "air.fmin.f$(8*sizeof(jltyp))" : "air.fmax.f$(8*sizeof(jltyp))"
            end

            # otherwise create a function that performs the IEEE-compliant operation. normally
            # we'd do this inline, but LLVM.jl doesn't have BB split functionality.
            new_intr_fn = something(fast_fn, is_minimum ? "air.minimum.f$(8*sizeof(jltyp))" :
                                                          "air.maximum.f$(8*sizeof(jltyp))")

            if haskey(functions(mod), new_intr_fn)
                new_intr = functions(mod)[new_intr_fn]
            elseif fast_fn !== nothing
                # relaxed builtin: just declare it, no wrapper needed
                new_intr = LLVM.Function(mod, new_intr_fn, op_ft)
            else
                new_intr = LLVM.Function(mod, new_intr_fn, op_ft)
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
                                      LLVM.ConstantFP(optyp, zero(jltyp)))
                    arg1_zero = fcmp!(builder, LLVM.API.LLVMRealUEQ, arg1,
                                      LLVM.ConstantFP(optyp, zero(jltyp)))
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
                        LLVM.Function(mod, fallback_intr_fn, op_ft)
                    end
                    val = call!(builder, op_ft, fallback_intr, collect(parameters(new_intr)))
                    ret!(builder, val)
                end
            end

            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)

                call_args = promote_bf ?
                    LLVM.Value[fpext!(builder, a, optyp) for a in arguments(call)] :
                    collect(arguments(call))
                new_value = call!(builder, op_ft, new_intr, call_args)
                if promote_bf
                    new_value = fptrunc!(builder, new_value, typ)
                end
                replace_uses!(call, new_value)
                erase!(call)
                changed = true
            end
        end
    end

    # fuse chained integer min/max (now lowered to air.min/max) into AIR's 3-way builtins
    changed |= fuse_minmax3!(fun)

    return changed
end

# annotate AIR intrinsics with optimization-related metadata
function annotate_air_intrinsics!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false

    for f in functions(mod)
        isdeclaration(f) || continue
        fn = LLVM.name(f)

        fn_attrs = function_attributes(f)
        function add_fn_attributes(names...)
            for name in names
                if LLVM.version() >= v"16" && name in ["argmemonly", "inaccessiblememonly",
                                                       "inaccessiblemem_or_argmemonly",
                                                       "readnone", "readonly", "writeonly"]
                    # XXX: workaround for changes from https://reviews.llvm.org/D135780
                    continue
                end
                push!(fn_attrs, EnumAttribute(name, 0))
            end
            changed = true
        end

        function add_param_attributes(idx, names...)
            param_attrs = parameter_attributes(f, idx)
            for name in names
                if name == "nocapture" && LLVM.version() >= v"21"
                    # `nocapture` was replaced by `captures(none)` in LLVM 21 (an
                    # integer-valued IntAttr, value 0 == CaptureInfo::none()).
                    push!(param_attrs, EnumAttribute("captures", 0))
                else
                    push!(param_attrs, EnumAttribute(name, 0))
                end
            end
            changed = true
        end

        # synchronization
        if fn == "air.wg.barrier" || fn == "air.simdgroup.barrier"
            add_fn_attributes("nounwind", "mustprogress", "convergent", "willreturn")

        # sincos
        elseif match(r"^air.(fast_)?sincos", fn) !== nothing
            add_param_attributes(2, "nocapture", "writeonly")

        # atomics
        elseif match(r"air.atomic.(local|global).load", fn) !== nothing
            # TODO: "memory(argmem: read)" on LLVM 16+
            add_fn_attributes("argmemonly", "readonly", "nounwind")
        elseif match(r"air.atomic.(local|global).store", fn) !== nothing
            # TODO: "memory(argmem: write)" on LLVM 16+
            add_fn_attributes("argmemonly", "writeonly", "nounwind")
        elseif match(r"air.atomic.(local|global).(xchg|cmpxchg)", fn) !== nothing
            # TODO: "memory(argmem: readwrite)" on LLVM 16+
            add_fn_attributes("argmemonly", "nounwind")
        elseif match(r"^air.atomic.(local|global).(add|sub|min|max|and|or|xor)", fn) !== nothing
            # TODO: "memory(argmem: readwrite)" on LLVM 16+
            add_fn_attributes("argmemonly", "nounwind")

        # simdgroup
        elseif match(r"air.simdgroup_matrix_8x8_init_filled", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        elseif match(r"air.simdgroup_matrix_8x8_multiply_accumulate", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        elseif match(r"air.simdgroup_matrix_8x8_load", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nofree", "nounwind", "readonly", "willreturn")
        elseif match(r"air.simdgroup_matrix_8x8_store", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nounwind", "willreturn", "writeonly")

        # simd permute
        elseif match(r"air.simd_(ballot|all|vote_all|any|vote_any|shuffle|shuffle_xor|shuffle_down|\
            shuffle_up|shuffle_and_fill_down|shuffle_and_fill_up)", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        end
    end

    return changed
end
