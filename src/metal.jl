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

# Does `gv` look like a boxed-constant replica created by `materialize_box!`? Its `_box` name
# and layout distinguish it from other private constants: `{ i64 header, [payload bytes] }`,
# optionally with leading alignment padding. Relocatable boxes are external and mutable, so
# `isconstant` excludes them (they are delivered through the relocation table instead).
function is_boxed_constant(@nospecialize(gv::LLVM.GlobalVariable))
    isdeclaration(gv) && return false
    endswith(LLVM.name(gv), "_box") || return false
    linkage(gv) == LLVM.API.LLVMPrivateLinkage || return false
    (isconstant(gv) && unnamed_addr(gv)) || return false
    addrspace(value_type(gv)) == 0 || return false
    T = global_value_type(gv)
    T isa LLVM.StructType || return false
    els = elements(T)
    is_hdr(t) = t isa LLVM.IntegerType && width(t) == 8sizeof(UInt)
    is_pad(t) = t isa LLVM.ArrayType && eltype(t) == LLVM.Int8Type()
    return (length(els) >= 1 && is_hdr(els[1])) ||
           (length(els) >= 2 && is_pad(els[1]) && is_hdr(els[2]))
end

# Are all (transitive constant) users of `gv` instructions, i.e. does the box escape only
# through function bodies? A box reachable from another global's *initializer* is the
# `jl_true`/`jl_false` slot→box indirection, which lives fine in the constant space
# (`add_global_address_spaces!`); demoting it would strand the slot's constant pointer. Only
# the isbits-union interior boxes — whose payload address flows through a body `phi`/`select`
# — need demotion, and those have instruction users exclusively.
function box_used_only_by_instructions(@nospecialize(gv::LLVM.GlobalVariable))
    ok = true
    function walk(@nospecialize(v))
        for use in uses(v)
            u = user(use)
            if u isa LLVM.Instruction
                # a body use: fine
            elseif u isa LLVM.GlobalVariable
                ok = false   # initializer reference (slot→box)
            elseif u isa LLVM.Constant
                walk(u)
            else
                ok = false
            end
        end
    end
    walk(gv)
    return ok
end

# Demote materialized boxed-constant globals to per-function stack allocas.
#
# Motivation: an isbits `Union` return is lowered as `{ptr, i8}` whose payload pointer
# `phi`/`select`s the box global against the caller's `sret` alloca. AIR has no generic
# address space — AS 0 *is* thread memory — so once `add_global_address_spaces!` sinks the
# box into AS 2 (constant), the `addrspacecast` back to AS 0 at the use only works where the
# back-end can statically fold it away; across the union's call return or aggregate phi it
# cannot, and the load silently reads thread memory instead of the constant. Copying the box
# to a thread `alloca` is the sanctioned lowering (MSL rejects mixing `thread` and `constant`
# pointers outright); it is sound because box addresses carry no identity (isbits egal is by
# content) and Metal fully inlines device functions.
#
# Runs from `finish_ir!`: post-opt (the box only escapes its `materialize_box!` pointer slot
# once GlobalOpt folds it — still-slotted boxes like `jl_true`/`jl_false` are skipped and
# stay in constant space) and pre-`add_global_address_spaces!`. Relocatable boxes (external
# + `extinit`) are demoted by the `:table` lowering instead, which also fills their header.
function demote_boxed_constants!(mod::LLVM.Module)
    changed = false
    for gv in collect(globals(mod))
        (is_boxed_constant(gv) && box_used_only_by_instructions(gv)) || continue
        if LLVM.version() < v"17"
            # `replace_global_with_local!` needs LLVM.jl's `convert_users_to_instructions!`
            # (LLVM 17+). Without demotion the kernel would compile but read thread memory
            # instead of the boxed constant, so refuse loudly rather than miscompile.
            error("Metal kernels embedding boxed union constants require Julia 1.12 or later")
        end
        boxty = global_value_type(gv)
        init = initializer(gv)
        # keep the box's alignment, but at least Julia's heap alignment (16 B), which is what
        # `materialize_box!` gives these boxes and what the payload's `isbits` layout assumes
        align = max(alignment(gv), 16)
        slots = Dict{LLVM.Function, LLVM.Value}()
        function slot(f::LLVM.Function)
            get!(slots, f) do
                @dispose builder=IRBuilder() begin
                    position!(builder, first(instructions(first(blocks(f)))))
                    ptr = alloca!(builder, boxty)
                    alignment!(ptr, align)
                    store!(builder, init, ptr)
                    ptr
                end
            end
        end
        replace_global_with_local!(gv, slot)
        changed = true
    end
    return changed
end


## relocations as a kernel-state word table
#
# Metal gives a loader no access at all to loaded code: there is no post-load symbol patching
# (no ORC `absoluteSymbols`, no writable program-scope globals to `:patch`). So relocation words
# are delivered as ordinary *run-time data*: the loader resolves them in its session, writes
# `resolved_relocation_table` into a small buffer, and passes that buffer's device address in the
# kernel state at every dispatch. GPUCompiler's `:table` lowering rewrites each record into an
# indexed load off the base this hook returns.
#
# Nothing session-local ends up in the metallib, so it is byte-stable even for
# relocation-carrying kernels — which is what makes them persistable across sessions and
# content-keyable by `MTLBinaryArchive`.
#
# The contract with the back-end is one kernel-state field:
#
#     reloc_table::Core.LLVMPtr{UInt64, AS.Device}
#
# always present (null for relocation-free kernels, which never read it) so that the state
# layout does not depend on what a kernel happens to reference.
const RELOCATION_TABLE_FIELD = :reloc_table

function relocation_table_pointer(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                  builder::IRBuilder, fun::LLVM.Function)
    state = kernel_state_type(job)
    field = state === Nothing ? nothing :
            findfirst(isequal(RELOCATION_TABLE_FIELD), fieldnames(state))
    field === nothing &&
        error("""Metal delivers relocations through the kernel state, so its type must have a
                 `$(RELOCATION_TABLE_FIELD)::Core.LLVMPtr{UInt64, 1}` field; got $state.""")

    # The state arrives as the leading by-reference argument (`kernel_state_to_reference!`,
    # then `add_parameter_address_spaces!`), so it is available on entry and dominates every
    # use. Only a kernel has one, and Metal fully inlines device code, so only the kernel can
    # be holding a relocation by now; insist on that rather than mistaking some other
    # function's first pointer argument for the state.
    (job.config.kernel && fun in kernels(LLVM.parent(fun))) ||
        error("""Metal delivers relocations through the kernel state, which only a kernel has.
                 Function `$(LLVM.name(fun))` is not one, so it cannot carry relocations;
                 compile it as a kernel, or select the `:bake` lowering.""")
    state_ptr = parameters(fun)[1]
    value_type(state_ptr) isa LLVM.PointerType ||
        error("Expected a kernel-state pointer argument, got $(value_type(state_ptr))")

    # Reach the field by its Julia byte offset rather than by a struct element index: Julia
    # renders a struct as an LLVM array when all its fields share a type, and inserts explicit
    # padding elements when they don't, so no element numbering matches the Julia one.
    T_byte = LLVM.Int8Type()
    T_table = convert(LLVMType, fieldtype(state, field))
    as = addrspace(value_type(state_ptr))
    typed = supports_typed_pointers(context())

    base = typed ? bitcast!(builder, state_ptr, LLVM.PointerType(T_byte, as)) : state_ptr
    field_ptr = inbounds_gep!(builder, T_byte, base,
                              [ConstantInt(LLVM.Int32Type(), fieldoffset(state, field))])
    typed && (field_ptr = bitcast!(builder, field_ptr, LLVM.PointerType(T_table, as)))

    table = load!(builder, T_table, field_ptr, "reloc_table")
    alignment!(table, sizeof(UInt))
    return table
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

function finish_runtime_intrinsics!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                    mod::LLVM.Module)
    # AIR input arguments need to be threaded after GC lowering and runtime linking:
    # `gpu_gc_pool_alloc` can call runtime helpers that use thread-position intrinsics.
    #
    # Thread the arguments here, but inline in `optimize_module!`, not now. Julia's const-region
    # metadata on by-pointer aggregate loads is not IPO-safe until `LateLowerGCPass` clears it
    # (JuliaLang/julia#44285), so inlining at this point would let GVN miscompile those loads.
    changed = false
    for f in kernels(mod)
        add_input_arguments!(job, mod, f, kernel_intrinsics)
        changed = true
    end
    return changed
end

# Inline the device code, including the late-linked Metal allocation runtime whose shape
# `lower_air!` later needs to rewrite generic null tests.
#
# This runs at the end of `optimize!`, after `LateLowerGCPass`, and that order matters. Julia's
# by-pointer aggregate loads carry const-region metadata (`!tbaa jtbaa_const`, `!invariant.load`,
# `!alias.scope`/`!noalias` against `jnoalias_stack`) that holds only until the LLVM inliner runs
# (JuliaLang/julia#44285). Inlining such a load onto a caller stack slot makes the metadata false,
# letting GVN fold the load to `undef` (for example turning a bounds check into an unconditional
# throw). `LateLowerGCPass`'s `CleanupIR` defuses this before we inline: it strips `!invariant.load`
# and downgrades `jtbaa_const` to mutable TBAA. The scoped `!noalias` survives, but the inliner
# clones alias scopes per instance, so the result stays correct.
function optimize_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                          mod::LLVM.Module)
    @dispose pb=NewPMPassBuilder() begin
        LLVM.target_transform_info!(pb, MetalTTI())
        add!(pb, ModuleInlinerWrapperPass())
        add!(pb, NewPMFunctionPassManager()) do fpm
            add!(fpm, instcombine_pass(job))
            add!(fpm, SimplifyCFGPass())
        end
        run!(pb, mod, llvm_machine(job.config.target))
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

    # atomics that `lower_atomics!` cannot lower
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        reason = if is_atomic_memop(inst)
            action = metal_atomic_action(job, inst)
            action isa String ? action : nothing
        elseif inst isa LLVM.FenceInst && metal_thread_scope(inst) === nothing
            "fence with synchronization scope $(syncscope_name(inst))"
        else
            nothing
        end
        reason === nothing || push!(errors, (reason, backtrace(inst), string(inst)))
    end

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

# Flatten chained single-index byte `getelementptr`s into one: `gep i8, (gep i8, p, A), B`
# -> `gep i8, p, (A + B)`. The AGX back-end miscompiles a 1-byte load/store made through a
# chained GEP (a byte GEP whose base is another byte GEP) when the grid has exactly two
# threadgroups -- the first threadgroup's access is silently dropped. LLVM deliberately keeps
# such chains split: `InstCombine`'s `visitGEPOfGEP` only merges when the combined index folds
# to an existing value (it bails on variable-plus-constant to avoid materializing an extra
# `add`), and the Metal driver never coalesces them either. On a normal back-end the split form
# is free (the constant GEP folds into the addressing mode), so this is purely an AGX defect; we
# work around it by force-merging here, which only costs a cheap `add` and makes the back-end
# emit correct code. Runs on the optimized, opaque-pointer IR, before AIR lowering.
function merge_byte_gep_chains!(mod::LLVM.Module)
    changed = false
    i8 = LLVM.Int8Type()
    is_byte_gep(v) =
        v isa LLVM.GetElementPtrInst && length(operands(v)) == 2 &&
        LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(v)) == i8

    # the first `gep i8, (gep i8, p, A), B` in `f`, or `nothing`
    function next_chain(f)
        for bb in blocks(f), inst in instructions(bb)
            is_byte_gep(inst) && is_byte_gep(operands(inst)[1]) && return inst
        end
        return nothing
    end

    for f in functions(mod)
        isdeclaration(f) && continue
        # Rescan after each merge instead of keeping a worklist: a merge can erase a *different*
        # chained GEP (a now-dead inner GEP whose only use was the one we just folded), and a
        # block layout that doesn't follow dominance can place that inner GEP after the outer one,
        # so stale worklist entries would be use-after-free. Rescanning only ever inspects live
        # instructions; each merge replaces a chained GEP with a shallower one (and shortens its
        # users), so the total chain depth strictly decreases and this terminates.
        while (gep = next_chain(f)) !== nothing
            src  = operands(gep)[1]
            base = operands(src)[1]
            inbounds = LLVM.API.LLVMIsInBounds(gep) != 0 && LLVM.API.LLVMIsInBounds(src) != 0
            @dispose builder=IRBuilder() begin
                position!(builder, gep)
                sum = add!(builder, operands(src)[2], operands(gep)[2])
                newgep = inbounds ? inbounds_gep!(builder, i8, base, [sum]) :
                                    gep!(builder, i8, base, [sum])
                replace_uses!(gep, newgep)
            end
            erase!(gep)
            isempty(uses(src)) && erase!(src)
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
        # demote escaped boxed-constant globals to stack allocas before the address-space
        # passes move constants into AS 2 (which would produce downgrade-incompatible IR for
        # the ones whose address escapes an isbits union return; see above)
        demote_boxed_constants!(mod)

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
    # Avoid generic-space null tests of pointers selected from concrete address spaces.
    # AIR can miscompile the generic comparison, while comparing in the source address
    # space and casting only the surviving pointer is the shape Julia emits for direct
    # Metal.malloc uses.
    rewrite_generic_null_selects!(mod)

    # lower LLVM atomics to AIR atomic intrinsics (including the fences that ordered atomics
    # get bracketed with on targets without ordered atomics, so this goes first)
    changed = lower_atomics!(job, mod)

    # the macOS 27 back-end rejects bare LLVM fences (Metal.jl#968)
    lower_fences!(job, mod)

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
    for f in functions(mod)
        changed |= lower_llvm_intrinsics!(job, f)
    end
    if changed
        # lowering may have introduced additional functions marked `alwaysinline` (including
        # the atomic expansions), and left dead declarations of replaced LLVM intrinsics behind
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

    # flatten chained byte GEPs that the AGX back-end miscompiles for 1-byte accesses on a
    # 2-threadgroup grid (see `merge_byte_gep_chains!`). run last, after the intrinsic-lowering
    # cleanup above, so the merged form is what reaches the AIR downgrader / back-end.
    merge_byte_gep_chains!(mod)

    return
end

# lowering of LLVM atomics
#
# Metal has no LLVM back-end, so the work that `AtomicExpand` and instruction selection do for
# LLVM atomics on other targets happens here. Front-ends (Metal.jl's atomic functions,
# UnsafeAtomics and Atomix, Julia's atomic intrinsics) emit plain LLVM atomics, with an
# ordering and a synchronization scope, and `lower_atomics!` turns them into the
# `air.atomic.*` intrinsics MSL uses:
#
# - atomics on the thread's own memory become plain accesses (`demote_private_atomic!`);
# - on targets without ordered atomics (MSL < 4.1), an ordered operation becomes a relaxed one
#   bracketed by fences, as `AtomicExpand` does for targets that `shouldInsertFencesForAtomic`;
# - operations that AIR cannot express are rewritten in terms of ones it can
#   (see `metal_atomic_action`): floating-point loads, stores and exchanges are cast to integers,
#   8- and 16-bit operations become masked operations on the containing 32-bit word, and
#   read-modify-write operations without an AIR equivalent become compare-exchange loops;
# - the remaining operations are selected to `air.atomic.*` calls (`select_atomic!`), in the
#   form the target's AIR and MSL versions use.
#
# Front-ends can also call `air.atomic.*` intrinsics directly, e.g., to pass memory flags that
# LLVM cannot express. They do so in the MSL 4.1 (AIR 2.9) form, which `legalize_atomic_abi!`
# rewrites for older targets. `validate_ir` rejects the atomics that cannot be lowered (see
# `metal_atomic_action`), so the lowering can assume every atomic it sees is supported.

# MSL memory_order values: relaxed=0, acquire=2, release=3, acq_rel=4, seq_cst=5
metal_memory_order(order::AtomicOrdering) =
    order == LLVM.API.LLVMAtomicOrderingAcquire ? 2 :
    order == LLVM.API.LLVMAtomicOrderingRelease ? 3 :
    order == LLVM.API.LLVMAtomicOrderingAcquireRelease ? 4 :
    order == LLVM.API.LLVMAtomicOrderingSequentiallyConsistent ? 5 : 0

# MSL mem_flags naming the memory an ordered operation orders. LLVM orders all memory, so
# cover device and threadgroup memory, the writable address spaces LLVM code can access.
const METAL_MEM_FLAGS = 1 | 2   # mem_device | mem_threadgroup

# The MSL thread_scope for the synchronization scope of an atomic operation or fence: thread=0,
# simdgroup=4, threadgroup=1, device=2. Scopes are spelled like the LLVM SPIR-V back-end
# does: `singlethread`, `subgroup`, `workgroup`, `device`, and the system scope (LLVM's
# default). Metal code can only synchronize with other threads on the same device (MSL has no
# scope that includes the host or other devices), so the system scope is the device scope.
# Threadgroup memory is only shared within a threadgroup, and MSL never uses a wider scope
# for it. Returns `nothing` for other scopes, which `validate_ir` rejects (like the NVPTX and
# AMDGPU back-ends do) rather than guessing what they mean.
function metal_thread_scope(inst::LLVM.Instruction, as::Union{Nothing,Int}=nothing)
    ss = syncscope(inst)
    scope = if ss == SyncScope("singlethread")
        0
    elseif ss == SyncScope("subgroup")
        4
    elseif ss == SyncScope("workgroup")
        1
    elseif ss == SyncScope("device") || ss == SyncScope("system")
        2
    else
        return nothing
    end
    return as == 3 && scope == 2 ? 1 : scope
end

# read-modify-write operations AIR has 32-bit intrinsics for, and the intrinsic names
const AIR_ATOMICRMW_OPS = Dict(
    :xchg => "xchg", :add => "add.s", :sub => "sub.s", :and => "and.s", :or => "or.s",
    :xor => "xor.s", :max => "max.s", :min => "min.s", :umax => "max.u", :umin => "min.u",
    :fadd => "add", :fsub => "sub")

# read-modify-write operations we can expand to compare-exchange loops
const EXPANDABLE_ATOMICRMW_OPS = (:nand, :fmax, :fmin, :fmaximum, :fminimum, :uinc_wrap,
                                  :udec_wrap, :usub_cond, :usub_sat)

function atomic_bits(T::LLVMType)
    T isa LLVM.IntegerType && return Int(width(T))
    T isa LLVM.LLVMHalf && return 16
    T isa LLVM.LLVMBFloat && return 16
    T isa LLVM.LLVMFloat && return 32
    T isa LLVM.LLVMDouble && return 64
    T isa LLVM.PointerType && return 64
    return nothing
end

# Does `ptr` point to the thread's own stack, i.e., is every object it can be derived from an
# `alloca`? That is the case for atomics on objects that Julia's `AllocOpt` moved to the stack,
# e.g., a non-escaping mutable struct with `@atomic` fields (GPUCompiler.jl#934). Metal cannot
# express those (MSL only has atomics on device and threadgroup memory), but they don't need
# to be atomic: no other thread can access a thread's stack, even when it has a pointer to it
# (thread memory is private to every thread), so plain accesses behave the same. Anything this
# cannot trace back to an `alloca`, e.g., a function argument or a loaded pointer, is not
# known to be private.
function is_thread_private(ptr::LLVM.Value)
    seen = Set{LLVM.Value}()
    worklist = LLVM.Value[ptr]
    while !isempty(worklist)
        val = pop!(worklist)
        val in seen && continue
        push!(seen, val)
        if val isa LLVM.AllocaInst
            continue
        elseif val isa LLVM.GetElementPtrInst || val isa LLVM.BitCastInst
            push!(worklist, operands(val)[1])
        elseif val isa LLVM.PHIInst
            append!(worklist, first.(LLVM.incoming(val)))
        elseif val isa LLVM.SelectInst
            push!(worklist, operands(val)[2], operands(val)[3])
        else
            return false
        end
    end
    return true
end

# How to lower `inst`, an atomic memory operation, for the job's target. Like the rule tables
# of LLVM's legalizers, the rules are tried in order and the first that applies decides. Returns
# the action, or the reason why the operation cannot be lowered (which `validate_ir` reports):
#
# - `:demote`: an atomic on the thread's own memory, which becomes plain accesses;
# - `:cast`: a floating-point load, store or exchange, which becomes an integer one (like the
#   default `TargetLowering::shouldCast*InIR`), so that AIR's integer intrinsics can be used;
# - `:partword`: an 8- or 16-bit operation, which becomes a masked operation on the containing
#   32-bit word (`AtomicExpand`'s `expandPartwordAtomicRMW` and `expandPartwordCmpXchg`);
# - `:cmpxchg_loop`: a read-modify-write operation without an AIR equivalent, which becomes a
#   compare-exchange loop (`AtomicExpand`'s `insertRMWCmpXchgLoop`);
# - `:select`: an operation AIR can express, which becomes an `air.atomic.*` call.
function metal_atomic_action(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                             inst::LLVM.Instruction)
    target = job.config.target
    op = inst isa LLVM.AtomicRMWInst ? atomicrmw_op(inst) : nothing
    if op !== nothing && !haskey(AIR_ATOMICRMW_OPS, op) && !(op in EXPANDABLE_ATOMICRMW_OPS)
        return "atomicrmw $op operation"
    end

    is_thread_private(atomic_pointer(inst)) && return :demote

    as = addrspace(value_type(atomic_pointer(inst)))
    if as != 1 && as != 3
        return "atomic operation in address space $as (Metal only supports atomics on device and threadgroup memory)"
    end

    T = atomic_value_type(inst)
    bits = atomic_bits(T)
    if bits === nothing || !(bits in (8, 16, 32, 64))
        return "atomic operation on a $(string(T)) value"
    end

    # AIR's only 64-bit atomics are umin/umax on device memory, which don't return the old
    # value (and need an Apple8 GPU). There is no 64-bit compare-exchange to emulate others.
    if bits == 64 && !(op in (:umin, :umax) && T isa LLVM.IntegerType && as == 1 &&
                       isempty(uses(inst)))
        return "64-bit atomic operation (Metal only supports atomic 64-bit umin and umax on device memory, without using the result)"
    end

    if alignment(inst) < bits ÷ 8
        return "misaligned atomic operation"
    end

    if metal_thread_scope(inst) === nothing
        return "atomic operation with synchronization scope $(syncscope_name(inst))"
    end

    # without ordered atomics, orderings are implemented with fences (from MSL 3.2)
    if is_ordered(atomic_ordering(inst)) && target.metal < v"3.2"
        return "ordered atomic operation (Metal $(target.metal) only supports relaxed atomics)"
    end

    if T isa LLVM.FloatingPointType && (op === nothing || op == :xchg) &&
       !(inst isa LLVM.AtomicCmpXchgInst)
        return :cast
    end
    bits < 32 && return :partword
    # (threadgroup floating-point add and subtract need MSL 4.1)
    if op in EXPANDABLE_ATOMICRMW_OPS ||
       (op in (:fadd, :fsub) && as == 3 && target.metal < v"4.1")
        return :cmpxchg_loop
    end
    return :select
end

# Build an expansion that needs control flow as an internal, always-inlined function, and
# replace `inst` with a call to it: LLVM.jl cannot split basic blocks. `body(builder, f,
# params...)` emits the function's code and returns its result, and the inliner that runs at
# the end of `lower_air!` puts the code in place.
function outline_atomic!(body, mod::LLVM.Module, inst::LLVM.Instruction,
                         args::Vector{<:LLVM.Value})
    T_ret = value_type(inst)
    ft = LLVM.FunctionType(T_ret, map(value_type, args))
    f = LLVM.Function(mod, "julia.air.atomic_expansion", ft)
    linkage!(f, LLVM.API.LLVMInternalLinkage)
    push!(function_attributes(f), EnumAttribute("alwaysinline"))
    @dispose builder=IRBuilder() begin
        position!(builder, BasicBlock(f, "entry"))
        result = body(builder, f, parameters(f)...)
        T_ret == LLVM.VoidType() ? ret!(builder) : ret!(builder, result)

        position!(builder, inst)
        debuglocation!(builder, inst)
        call = call!(builder, ft, f, args)
        T_ret == LLVM.VoidType() || replace_uses!(inst, call)
    end
    erase!(inst)
    return
end

function set_atomic!(inst::LLVM.Instruction, order::AtomicOrdering, scope::SyncScope,
                     volatile::Bool=false)
    ordering!(inst, order)
    syncscope!(inst, scope)
    volatile && LLVM.API.LLVMSetVolatile(inst, true)
    return inst
end

# Replace an atomic operation on the thread's own memory (see `is_thread_private`) by plain
# accesses. Its ordering and scope don't matter either: no other thread can observe the memory
# it accesses, so it cannot synchronize with any.
function demote_private_atomic!(inst::LLVM.Instruction)
    ptr = atomic_pointer(inst)
    T = atomic_value_type(inst)
    volatile = is_volatile(inst)
    @dispose builder=IRBuilder() begin
        position!(builder, inst)
        debuglocation!(builder, inst)
        function plain_load()
            ld = load!(builder, T, ptr)
            alignment!(ld, alignment(inst))
            volatile && LLVM.API.LLVMSetVolatile(ld, true)
            ld
        end
        function plain_store(val)
            st = store!(builder, val, ptr)
            alignment!(st, alignment(inst))
            volatile && LLVM.API.LLVMSetVolatile(st, true)
            st
        end
        if inst isa LLVM.LoadInst
            replace_uses!(inst, plain_load())
        elseif inst isa LLVM.StoreInst
            plain_store(operands(inst)[1])
        elseif inst isa LLVM.AtomicRMWInst
            old = plain_load()
            plain_store(atomicrmw_value!(builder, atomicrmw_op(inst), old, operands(inst)[2]))
            replace_uses!(inst, old)
        else
            # compare-exchange: store the new value if the old one matches, else the old one
            cmp, new = operands(inst)[2:3]
            old = plain_load()
            success = icmp!(builder, LLVM.API.LLVMIntEQ, old, cmp)
            plain_store(select!(builder, success, new, old))
            result = insert_value!(builder, UndefValue(value_type(inst)), old, 0)
            replace_uses!(inst, insert_value!(builder, result, success, 1))
        end
    end
    erase!(inst)
    return
end

# Emit a loop that atomically replaces the 32-bit word at `ptr` by `update(builder, word)`
# using compare-exchange, returning the word the successful exchange replaced
# (`AtomicExpand`'s `insertRMWCmpXchgLoop`).
function emit_cmpxchg_loop!(update, builder::IRBuilder, f::LLVM.Function, ptr::LLVM.Value,
                            order::AtomicOrdering, scope::SyncScope, volatile::Bool)
    T_word = LLVM.Int32Type()
    entry = position(builder)
    init = load!(builder, T_word, ptr)
    alignment!(init, 4)
    set_atomic!(init, LLVM.API.LLVMAtomicOrderingMonotonic, scope, volatile)
    loop = BasicBlock(f, "atomicrmw.start")
    done = BasicBlock(f, "atomicrmw.end")
    br!(builder, loop)

    position!(builder, loop)
    loaded = phi!(builder, T_word, "loaded")
    pair = atomic_cmpxchg!(builder, ptr, loaded, update(builder, loaded), order,
                           failure_ordering_for(order), scope)
    volatile && LLVM.API.LLVMSetVolatile(pair, true)
    word = extract_value!(builder, pair, 0)
    br!(builder, extract_value!(builder, pair, 1), done, loop)
    push!(LLVM.incoming(loaded), (init, entry))
    push!(LLVM.incoming(loaded), (word, loop))

    position!(builder, done)
    return word
end

# The 32-bit word containing an 8- or 16-bit value at `ptr`, and the position of the value
# in it (`AtomicExpand`'s `createMaskInstrs`; Metal is little-endian). Like `AtomicExpand`,
# this assumes that whole word can be accessed: true for device buffers, which Metal allocates
# in pages, and for threadgroup arrays, which Metal.jl aligns and pads to 4 bytes.
function partword_layout!(builder::IRBuilder, ptr::LLVM.Value, bits::Int)
    as = addrspace(value_type(ptr))
    T_i8, T_i32, T_i64 = LLVM.Int8Type(), LLVM.Int32Type(), LLVM.Int64Type()
    bytes = bitcast!(builder, ptr, LLVM.PointerType(T_i8, as))
    offset = and!(builder, ptrtoint!(builder, bytes, T_i64), ConstantInt(T_i64, 3))
    word = gep!(builder, T_i8, bytes, [neg!(builder, offset)])
    word = bitcast!(builder, word, LLVM.PointerType(T_i32, as))
    shift = shl!(builder, trunc!(builder, offset, T_i32), ConstantInt(T_i32, 3))
    mask = shl!(builder, ConstantInt(T_i32, (1 << bits) - 1), shift)
    return (; word, shift, mask, inv_mask=not!(builder, mask))
end
partword_extract!(builder, layout, word, T) =
    trunc!(builder, lshr!(builder, word, layout.shift), T)
partword_insert!(builder, layout, word, val) =
    or!(builder, and!(builder, word, layout.inv_mask),
        shl!(builder, zext!(builder, val, LLVM.Int32Type()), layout.shift))

# The operands of an atomic operation that its expansions need, with the ordering it is lowered
# with (see `lowered_ordering`).
function atomic_operands(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                         inst::LLVM.Instruction)
    T = atomic_value_type(inst)
    return (; ptr=atomic_pointer(inst), T, bits=atomic_bits(T),
              op=inst isa LLVM.AtomicRMWInst ? atomicrmw_op(inst) : nothing,
              order=lowered_ordering(job, atomic_ordering(inst)), scope=syncscope(inst),
              volatile=is_volatile(inst))
end

# Cast a floating-point load, store or exchange to an integer one, returning the new operation.
function cast_atomic_to_int!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                             inst::LLVM.Instruction)
    (; ptr, T, bits, order, scope, volatile) = atomic_operands(job, inst)
    T_int = LLVM.IntType(bits)
    @dispose builder=IRBuilder() begin
        position!(builder, inst)
        debuglocation!(builder, inst)
        int_ptr = bitcast!(builder, ptr, LLVM.PointerType(T_int, addrspace(value_type(ptr))))
        new = if inst isa LLVM.LoadInst
            ld = load!(builder, T_int, int_ptr)
            alignment!(ld, alignment(inst))
            set_atomic!(ld, order, scope, volatile)
            replace_uses!(inst, bitcast!(builder, ld, T))
            ld
        elseif inst isa LLVM.StoreInst
            st = store!(builder, bitcast!(builder, operands(inst)[1], T_int), int_ptr)
            alignment!(st, alignment(inst))
            set_atomic!(st, order, scope, volatile)
        else
            rmw = atomic_rmw!(builder, LLVM.API.LLVMAtomicRMWBinOpXchg, int_ptr,
                              bitcast!(builder, operands(inst)[2], T_int), order, scope)
            alignment!(rmw, alignment(inst))
            volatile && LLVM.API.LLVMSetVolatile(rmw, true)
            replace_uses!(inst, bitcast!(builder, rmw, T))
            rmw
        end
        erase!(inst)
        return new
    end
end

# Expand a 32-bit read-modify-write operation to a compare-exchange loop on the word.
function expand_to_cmpxchg_loop!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                 mod::LLVM.Module, inst::LLVM.Instruction)
    (; ptr, T, op, order, scope, volatile) = atomic_operands(job, inst)
    T_i32 = LLVM.Int32Type()
    outline_atomic!(mod, inst, [ptr, operands(inst)[2]]) do builder, f, ptr, val
        word_ptr = bitcast!(builder, ptr, LLVM.PointerType(T_i32, addrspace(value_type(ptr))))
        word = emit_cmpxchg_loop!(builder, f, word_ptr, order, scope, volatile) do builder, loaded
            old = bitcast!(builder, loaded, T)
            bitcast!(builder, atomicrmw_value!(builder, op, old, val), T_i32)
        end
        bitcast!(builder, word, T)
    end
    return
end

# 8- and 16-bit atomics as masked operations on the containing 32-bit word
# (`AtomicExpand`'s `expandPartwordAtomicRMW` and `expandPartwordCmpXchg`)
function expand_partword_atomic!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                 mod::LLVM.Module, inst::LLVM.Instruction)
    (; ptr, T, bits, op, order, scope, volatile) = atomic_operands(job, inst)
    T_i32 = LLVM.Int32Type()

    # operations that don't need a loop: loads, and bitwise operations that leave the rest
    # of the word unchanged
    if inst isa LLVM.LoadInst || op in (:and, :or, :xor)
        @dispose builder=IRBuilder() begin
            position!(builder, inst)
            debuglocation!(builder, inst)
            layout = partword_layout!(builder, ptr, bits)
            word = if inst isa LLVM.LoadInst
                ld = load!(builder, T_i32, layout.word)
                alignment!(ld, 4)
                set_atomic!(ld, order, scope, volatile)
            else
                val = shl!(builder, zext!(builder, operands(inst)[2], T_i32), layout.shift)
                op == :and && (val = or!(builder, val, layout.inv_mask))
                binop = op == :and ? LLVM.API.LLVMAtomicRMWBinOpAnd :
                        op == :or ? LLVM.API.LLVMAtomicRMWBinOpOr :
                                    LLVM.API.LLVMAtomicRMWBinOpXor
                rmw = atomic_rmw!(builder, binop, layout.word, val, order, scope)
                alignment!(rmw, 4)
                volatile && LLVM.API.LLVMSetVolatile(rmw, true)
                rmw
            end
            replace_uses!(inst, partword_extract!(builder, layout, word, T))
        end
        erase!(inst)
        return
    end

    if inst isa LLVM.AtomicCmpXchgInst
        cmp, new = operands(inst)[2:3]
        success = lowered_ordering(job, success_ordering(inst))
        failure = lowered_ordering(job, failure_ordering(inst))
        T_result = value_type(inst)
        outline_atomic!(mod, inst, [ptr, cmp, new]) do builder, f, ptr, cmp, new
            layout = partword_layout!(builder, ptr, bits)
            new_shifted = shl!(builder, zext!(builder, new, T_i32), layout.shift)
            cmp_shifted = shl!(builder, zext!(builder, cmp, T_i32), layout.shift)
            init = load!(builder, T_i32, layout.word)
            alignment!(init, 4)
            set_atomic!(init, LLVM.API.LLVMAtomicOrderingMonotonic, scope, volatile)
            init_rest = and!(builder, init, layout.inv_mask)
            entry = position(builder)
            loop = BasicBlock(f, "partword.cmpxchg.loop")
            failed = BasicBlock(f, "partword.cmpxchg.failure")
            done = BasicBlock(f, "partword.cmpxchg.end")
            br!(builder, loop)

            # retry as long as the exchange only failed because of the rest of the word
            position!(builder, loop)
            rest = phi!(builder, T_i32, "rest")
            pair = atomic_cmpxchg!(builder, layout.word, or!(builder, rest, cmp_shifted),
                                   or!(builder, rest, new_shifted), success, failure, scope)
            volatile && LLVM.API.LLVMSetVolatile(pair, true)
            word = extract_value!(builder, pair, 0)
            ok = extract_value!(builder, pair, 1)
            br!(builder, ok, done, failed)

            position!(builder, failed)
            new_rest = and!(builder, word, layout.inv_mask)
            br!(builder, icmp!(builder, LLVM.API.LLVMIntNE, rest, new_rest), loop, done)
            push!(LLVM.incoming(rest), (init_rest, entry))
            push!(LLVM.incoming(rest), (new_rest, failed))

            position!(builder, done)
            result = insert_value!(builder, UndefValue(T_result),
                                   partword_extract!(builder, layout, word, T), 0)
            insert_value!(builder, result, ok, 1)
        end
        return
    end

    # everything else becomes a compare-exchange loop on the word; the value may be a
    # floating-point one, for the arithmetic read-modify-write operations
    val = inst isa LLVM.StoreInst ? operands(inst)[1] : operands(inst)[2]
    op = something(op, :xchg)   # a store is an exchange with an ignored result
    # compare-exchange needs at least monotonic (like `AtomicExpand`'s `expandAtomicStoreToXChg`)
    order == LLVM.API.LLVMAtomicOrderingUnordered && (order = LLVM.API.LLVMAtomicOrderingMonotonic)
    is_store = inst isa LLVM.StoreInst
    T_int = LLVM.IntType(bits)
    outline_atomic!(mod, inst, [ptr, val]) do builder, f, ptr, val
        layout = partword_layout!(builder, ptr, bits)
        word = emit_cmpxchg_loop!(builder, f, layout.word, order, scope,
                                  volatile) do builder, loaded
            old = bitcast!(builder, partword_extract!(builder, layout, loaded, T_int), T)
            new = bitcast!(builder, atomicrmw_value!(builder, op, old, val), T_int)
            partword_insert!(builder, layout, loaded, new)
        end
        is_store ? nothing :
            bitcast!(builder, partword_extract!(builder, layout, word, T_int), T)
    end
    return
end

# On targets without ordered atomics (MSL < 4.1), bracket the operation with fences
# (`AtomicExpand`'s `bracketInstWithFences` with the default `emitLeadingFence` and
# `emitTrailingFence`), which `lower_fences!` turns into `air.atomic.fence` calls; the
# operation itself is then lowered as a relaxed one (see `lowered_ordering`). The ordering is
# not reset here, as the C API cannot do so for `atomicrmw` before LLVM 18.
function insert_atomic_fences!(inst::LLVM.Instruction)
    order = atomic_ordering(inst)
    is_ordered(order) || return false
    scope = syncscope(inst)
    @dispose builder=IRBuilder() begin
        if is_release(order) && !(inst isa LLVM.LoadInst)
            position!(builder, inst)
            debuglocation!(builder, inst)
            fence!(builder, order, scope)
        end
        if is_acquire(order)
            position!(builder, nextinst(inst))
            debuglocation!(builder, inst)
            fence!(builder, order, scope)
        end
    end
    return true
end

# LLVM requires that threads repeatedly loading an address monotonically eventually see the
# stores of other threads, but Apple's compiler emits relaxed loads of device memory as cached
# loads: on an M1, a spin loop that relaxed-loads a flag another threadgroup sets never sees
# the store (not in 20M iterations; within a threadgroup it does, as it shares the cache).
# Acquire loads invalidate the cache after loading, so lower device-scope monotonic loads of
# device memory as acquire ones (which on targets without ordered atomics adds a fence; before
# MSL 3.2, there are no fences to lower that to).
function strengthen_relaxed_load!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                  inst::LLVM.Instruction)
    job.config.target.metal >= v"3.2" && inst isa LLVM.LoadInst &&
        atomic_ordering(inst) == LLVM.API.LLVMAtomicOrderingMonotonic &&
        addrspace(value_type(atomic_pointer(inst))) == 1 &&
        metal_thread_scope(inst, 1) == 2 || return false
    ordering!(inst, LLVM.API.LLVMAtomicOrderingAcquire)
    return true
end

# With ordered atomics (MSL ≥ 4.1), follow a sequentially-consistent store by a
# sequentially-consistent fence (AtomicExpand's `shouldInsertTrailingSeqCstFenceForAtomicStore`,
# which AArch64 uses for MSVC). Apple compiles such a store as a release store that doesn't
# wait for the write, so a later sequentially-consistent load can be performed first:
# `store x; load y` and `store y; load x` in two threads both return the old values (store
# buffering, ~0.5% of the time on an M1), which sequential consistency forbids.
# Read-modify-writes and compare-exchanges wait for their result and don't need this, and
# fences before loads instead would cost more (loads are more common than stores).
function insert_trailing_seq_cst_fence!(inst::LLVM.Instruction)
    inst isa LLVM.StoreInst &&
        atomic_ordering(inst) == LLVM.API.LLVMAtomicOrderingSequentiallyConsistent ||
        return false
    @dispose builder=IRBuilder() begin
        position!(builder, nextinst(inst))
        debuglocation!(builder, inst)
        fence!(builder, LLVM.API.LLVMAtomicOrderingSequentiallyConsistent, syncscope(inst))
    end
    return true
end

# The ordering to lower an atomic operation with: without ordered atomics (MSL < 4.1), the
# fences `insert_atomic_fences!` added provide the ordering, and the operation is relaxed.
lowered_ordering(@nospecialize(job::CompilerJob{MetalCompilerTarget}), order::AtomicOrdering) =
    job.config.target.metal < v"4.1" ? LLVM.API.LLVMAtomicOrderingMonotonic : order

function air_atomic_function(mod::LLVM.Module, name::String, ft::LLVM.FunctionType)
    if haskey(functions(mod), name)
        f = functions(mod)[name]
        function_type(f) == ft ||
            error("Conflicting declarations of $name: $(function_type(f)) and $ft")
        return f
    end
    f = LLVM.Function(mod, name, ft)
    # as Apple declares them (not `argmemonly` or `readonly`: they order other memory)
    for attr in ("mustprogress", "nounwind", "willreturn")
        push!(function_attributes(f), EnumAttribute(attr, 0))
    end
    return f
end

# Replace an atomic operation by the equivalent `air.atomic.*` call, in the form the
# target's AIR and MSL versions use.
function select_atomic!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                        mod::LLVM.Module, inst::LLVM.Instruction)
    target = job.config.target
    ptr = atomic_pointer(inst)
    T_ptr = value_type(ptr)
    as = addrspace(T_ptr)
    T = atomic_value_type(inst)
    mem = as == 1 ? "global" : "local"
    suffix = T isa LLVM.LLVMFloat ? "f32" : "i$(width(T))"
    T_i32, T_i1 = LLVM.Int32Type(), LLVM.Int1Type()

    # the operands after the memory order(s): scope, flags (from AIR 2.9), volatile
    order = lowered_ordering(job, atomic_ordering(inst))
    # (the scope of a relaxed load of device memory only matters to whether we strengthen it,
    # so use the device scope, like MSL does: it doesn't change the code, but it is recorded)
    relaxed_device_load = inst isa LLVM.LoadInst && as == 1 &&
                          atomic_ordering(inst) == LLVM.API.LLVMAtomicOrderingMonotonic
    scope = ConstantInt(T_i32, relaxed_device_load ? 2 : metal_thread_scope(inst, as))
    flags = ConstantInt(T_i32, is_ordered(order) ? METAL_MEM_FLAGS : 0)
    # MSL sets the volatile bit on every atomic before 4.1, but since then only on atomics
    # of `volatile` objects. Without it, the back-end treats a load like a plain one (e.g.,
    # a relaxed load of memory the kernel doesn't write is hoisted into the uniform preamble,
    # out of any spin loop), and a read-modify-write that doesn't change memory (e.g., adding
    # 0) becomes such a load. LLVM atomics allow neither, so always set it on loads and
    # read-modify-writes; stores and compare-exchanges are compiled the same either way.
    volatile = ConstantInt(T_i1, target.metal < v"4.1" || is_volatile(inst) ||
                                 inst isa LLVM.LoadInst || inst isa LLVM.AtomicRMWInst)
    trailing_types = target.air >= v"2.9" ? [T_i32, T_i32, T_i1] : [T_i32, T_i1]
    trailing = target.air >= v"2.9" ? [scope, flags, volatile] : [scope, volatile]
    memory_order(order) = ConstantInt(T_i32, metal_memory_order(lowered_ordering(job, order)))

    @dispose builder=IRBuilder() begin
        position!(builder, inst)
        debuglocation!(builder, inst)
        if inst isa LLVM.LoadInst
            ft = LLVM.FunctionType(T, [T_ptr, T_i32, trailing_types...])
            f = air_atomic_function(mod, "air.atomic.$mem.load.$suffix", ft)
            new = call!(builder, ft, f, [ptr, memory_order(order), trailing...])
        elseif inst isa LLVM.StoreInst
            ft = LLVM.FunctionType(LLVM.VoidType(), [T_ptr, T, T_i32, trailing_types...])
            f = air_atomic_function(mod, "air.atomic.$mem.store.$suffix", ft)
            new = call!(builder, ft, f, [ptr, operands(inst)[1], memory_order(order),
                                         trailing...])
        elseif inst isa LLVM.AtomicRMWInst
            op = AIR_ATOMICRMW_OPS[atomicrmw_op(inst)]
            # AIR's 64-bit min/max don't return the old value
            T_ret = suffix == "i64" ? LLVM.VoidType() : T
            ft = LLVM.FunctionType(T_ret, [T_ptr, T, T_i32, trailing_types...])
            f = air_atomic_function(mod, "air.atomic.$mem.$op.$suffix", ft)
            new = call!(builder, ft, f, [ptr, operands(inst)[2], memory_order(order),
                                         trailing...])
        else
            # AIR only has a weak compare-exchange, which takes the expected value by
            # reference. Like MSL, derive the success flag from the returned old value.
            cmp, desired = operands(inst)[2:3]
            fn = LLVM.parent(LLVM.parent(inst))
            expected = @dispose entry_builder=IRBuilder() begin
                position!(entry_builder, first(instructions(first(blocks(fn)))))
                alloca!(entry_builder, T)
            end
            store!(builder, cmp, expected)
            ft = LLVM.FunctionType(T, [T_ptr, value_type(expected), T, T_i32, T_i32,
                                       trailing_types...])
            f = air_atomic_function(mod, "air.atomic.$mem.cmpxchg.weak.$suffix", ft)
            old = call!(builder, ft, f,
                        [ptr, expected, desired, memory_order(success_ordering(inst)),
                         memory_order(failure_ordering(inst)), trailing...])
            success = icmp!(builder, LLVM.API.LLVMIntEQ, old, cmp)
            new = insert_value!(builder, UndefValue(value_type(inst)), old, 0)
            new = insert_value!(builder, new, success, 1)
        end
        isempty(uses(inst)) || replace_uses!(inst, new)
    end
    erase!(inst)
    return
end

# the number of operands of an `air.atomic.*` intrinsic in its MSL 4.1 (AIR 2.9) form
function air_atomic_arity(name::String)
    occursin(".cmpxchg.", name) && return 8
    occursin(".load.", name) && return 5
    return 6
end

# Rewrite the `air.atomic.*` calls front-ends emit (in the MSL 4.1 form) into the form the
# target uses: AIR 2.9 introduced the memory flags operand, and before MSL 4.1 the volatile bit
# is always set, only relaxed orderings are supported and the flags must be zero. Also tell the
# downgrader the element types of the pointer operands, which LLVM cannot infer.
function legalize_atomic_abi!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                              mod::LLVM.Module)
    target = job.config.target
    changed = false
    for f in collect(functions(mod))
        fn = LLVM.name(f)
        m = match(r"^air\.atomic\.(global|local)\..+\.(i32|f32|i64)$", fn)
        m === nothing && continue

        # pointer element types for the typed-pointer downgrader
        T = m.captures[2] == "f32" ? LLVM.FloatType() : LLVM.IntType(parse(Int, m.captures[2][2:end]))
        mds = []
        for (i, param) in enumerate(parameters(function_type(f)))
            i <= 2 && param isa LLVM.PointerType || continue
            push!(mds, ConstantInt(Int32(i - 1)))
            push!(mds, null(T))
        end
        metadata(f)["arg_eltypes"] = MDNode(mds)

        target.metal >= v"4.1" && continue
        nparams = length(parameters(function_type(f)))
        # older front-ends rewrote their calls for the target themselves
        nparams == air_atomic_arity(fn) || continue

        is_cmpxchg = occursin(".cmpxchg.", fn)
        calls = [user(u)::LLVM.CallInst for u in uses(f)]
        for call in calls
            args = collect(arguments(call))
            orders = is_cmpxchg ? args[end-4:end-3] : args[end-3:end-3]
            if !all(o -> o isa ConstantInt && convert(Int, o) == 0, [orders..., args[end-1]])
                error("$fn with an ordering or memory flags requires MSL 4.1; " *
                      "the target is MSL $(target.metal)")
            end
        end
        changed |= !isempty(calls)

        if target.air >= v"2.9"
            # the flags operand exists, but the volatile bit must be set
            for call in calls
                # LLVM models the callee as the final operand, after all arguments
                operands(call)[end-1] = ConstantInt(true)
            end
            continue
        end

        # before AIR 2.9, there is no flags operand
        ft = function_type(f)
        params = parameters(ft)
        legacy_ft = LLVM.FunctionType(LLVM.return_type(ft), [params[1:end-2]..., params[end]])
        LLVM.name!(f, fn * ".msl41")
        legacy_f = LLVM.Function(mod, fn, legacy_ft)
        for attr in collect(function_attributes(f))
            push!(function_attributes(legacy_f), attr)
        end
        metadata(legacy_f)["arg_eltypes"] = metadata(f)["arg_eltypes"]
        for call in calls
            args = collect(arguments(call))
            @dispose builder=IRBuilder() begin
                position!(builder, call)
                debuglocation!(builder, call)
                new = call!(builder, legacy_ft, legacy_f,
                            [args[1:end-2]..., ConstantInt(true)])
                replace_uses!(call, new)
            end
            erase!(call)
        end
        erase!(f)
    end
    return changed
end

function lower_atomics!(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                        mod::LLVM.Module)
    # first rewrite the intrinsic calls front-ends emitted, so that the ones we select below
    # agree with them
    changed = legalize_atomic_abi!(job, mod)

    atomics = [inst for f in functions(mod) for bb in blocks(f) for inst in instructions(bb)
               if is_atomic_memop(inst)]
    isempty(atomics) && return changed

    for inst in atomics
        action = metal_atomic_action(job, inst)
        action isa String && continue   # unsupported, reported below
        if action === :demote
            demote_private_atomic!(inst)
            continue
        end
        strengthen_relaxed_load!(job, inst)
        if job.config.target.metal < v"4.1"
            insert_atomic_fences!(inst)
        else
            insert_trailing_seq_cst_fence!(inst)
        end
        if action === :cast
            inst = cast_atomic_to_int!(job, inst)
            action = metal_atomic_action(job, inst)   # e.g. a half-precision load is partword
        end
        if action === :partword
            expand_partword_atomic!(job, mod, inst)
        elseif action === :cmpxchg_loop
            expand_to_cmpxchg_loop!(job, mod, inst)
        end
    end

    # select the atomics that are left, including the ones the expansions introduced
    unsupported = IRError[]
    for f in functions(mod), bb in blocks(f), inst in collect(instructions(bb))
        is_atomic_memop(inst) || continue
        action = metal_atomic_action(job, inst)
        if action === :select
            select_atomic!(job, mod, inst)
        elseif action isa String
            push!(unsupported, (action, backtrace(inst), string(inst)))
        else
            error("Atomic operation was not legalized ($action): $inst")
        end
    end

    # `validate_ir` rejects these, but validation can be disabled (e.g., for reflection).
    # Rather than selecting AIR intrinsics that don't exist, fail here too.
    isempty(unsupported) || throw(InvalidIRError(job, unsupported))

    # attach element type metadata to the declarations we introduced
    legalize_atomic_abi!(job, mod)
    return true
end

# Before LLVM 18, `ordering(inst)` calls `LLVMGetOrdering`, which incorrectly casts fences
# to AtomicRMWInst. Use the stable textual form on all versions to keep this workaround tested.
function fence_ordering(inst::LLVM.FenceInst)
    # Scope names escape embedded quotes as \22; metadata follows the ordering.
    m = match(r"^\s*fence(?:\s+syncscope\(\"[^\"]*\"\))?\s+(acquire|release|acq_rel|seq_cst)\b",
              string(inst))
    m === nothing && error("Unexpected fence instruction: $inst")
    return m.captures[1] == "acquire" ? LLVM.API.LLVMAtomicOrderingAcquire :
           m.captures[1] == "release" ? LLVM.API.LLVMAtomicOrderingRelease :
           m.captures[1] == "acq_rel" ? LLVM.API.LLVMAtomicOrderingAcquireRelease :
                                        LLVM.API.LLVMAtomicOrderingSequentiallyConsistent
end

# Lower LLVM fences to air.atomic.fence(flags, order, scope), as MSL's atomic_thread_fence
# does. Bare fences from Julia's atomic_fence crash the macOS 27 AGX back-end (Metal.jl#968).
#
# Metal 3.2-4.0 only supports relaxed/seq_cst fences, so strengthen other orderings to
# seq_cst. Before Metal 3.2 the intrinsic is unavailable; retain the bare fence. Such targets
# still require a back-end that accepts bare fences.
#
# Cover device and threadgroup memory (`METAL_MEM_FLAGS`), the shared writable LLVM address
# spaces, and map the scope like for atomics (`metal_thread_scope`).
function lower_fences!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module)
    metal = job.config.target.metal
    metal >= v"3.2" || return false

    worklist = LLVM.FenceInst[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        inst isa LLVM.FenceInst && push!(worklist, inst)
    end
    isempty(worklist) && return false

    T_int32 = LLVM.Int32Type()
    fence_ft = LLVM.FunctionType(LLVM.VoidType(), [T_int32, T_int32, T_int32])
    fence_fn = if haskey(functions(mod), "air.atomic.fence")
        functions(mod)["air.atomic.fence"]
    else
        LLVM.Function(mod, "air.atomic.fence", fence_ft)
    end

    for inst in worklist
        order = metal < v"4.1" ? 5 : metal_memory_order(fence_ordering(inst))
        # (`validate_ir` rejects unknown scopes; without validation, use the widest one)
        scope = something(metal_thread_scope(inst), 2)

        @dispose builder=IRBuilder() begin
            position!(builder, inst)
            debuglocation!(builder, inst)
            call!(builder, fence_ft, fence_fn,
                  [ConstantInt(T_int32, METAL_MEM_FLAGS), ConstantInt(T_int32, order),
                   ConstantInt(T_int32, scope)])
        end
        erase!(inst)
    end
    return true
end

# Julia names each generated LLVM function `julia_<name>_<counter>`, where the counter is
# drawn from a process-global codegen sequence and so differs from one session to the next.
# Left anywhere in the emitted bitcode it makes the AIR (and the metallib wrapping it)
# non-reproducible across sessions, defeating byte-stable caching and content-keyed binary
# archives. And a symbol name is not the only place it appears: inlining a Julia function
# leaves its name behind in the block labels the inliner synthesizes, in the (now orphaned)
# `DISubprogram` its debug locations still point at, and in the alias-scope strings Julia's
# codegen derived from it.
#
# So rewrite every occurrence, wherever it appears: map each distinct codegen name to a
# deterministic module-local form (its rank in a fixed traversal), then substitute that map
# into symbol names, value names, and metadata strings. A subprogram belonging to a function
# instead adopts that function's current name, so the entry — already renamed to a stable
# mangled symbol in `irgen.jl` — keeps a `linkageName` matching the symbol it describes.
# Runs on the final (post-`lower_air!`) module, just before the bitcode goes to the downgrader.
function normalize_julia_symbol_names!(mod::LLVM.Module)
    # The counter is the trailing digit group, so match the name part lazily. The word
    # boundaries keep the pattern from biting into a longer token: `myjulia_foo_1` is a user
    # symbol that merely ends this way, and `julia_foo_12bar` is one that merely contains it.
    # The name part must cover Julia's full method-name alphabet, not just `\w`: mutating
    # functions carry `!` (`julia_record_exception!_18521`) and closures carry `#`
    # (`julia_#kernel#123_456`), and a missed name leaks the per-session counter into the
    # bitcode — exactly the byte-instability this pass exists to prevent.
    codegen_name = r"(?<!\w)julia_[\w!#]*?_[0-9]+(?!\w)"
    renames = Dict{String,String}()
    deterministic(match::AbstractString) = get!(renames, match) do
        replace(match, r"_[0-9]+$" => "") * "_$(length(renames) + 1)"
    end
    normalize(str::AbstractString) = replace(str, codegen_name => deterministic)
    rename!(value) = let str = LLVM.name(value)
        occursin(codegen_name, str) && LLVM.name!(value, normalize(str))
    end

    # Metadata forms a graph (a debug location points at its subprogram, an alias scope at its
    # domain), so walk it, replacing every string that carries a name. Only nodes reachable
    # from a function or an instruction are visited, which is where inlining leaves its traces.
    visited = Set{LLVM.API.LLVMMetadataRef}()
    function normalize_metadata!(@nospecialize(md), replacement=nothing)
        md isa LLVM.MDNode || return
        md.ref in visited && return
        push!(visited, md.ref)
        for (i, op) in enumerate(operands(md))
            if op isa LLVM.MDString
                str = convert(String, op)
                occursin(codegen_name, str) || continue
                new = replacement === nothing ? normalize(str) : replacement
                new == str || LLVM.replace_operand(md, i, LLVM.MDString(new))
            elseif op isa LLVM.MDNode
                normalize_metadata!(op)
            end
        end
    end

    # the instruction metadata that can name a function: a debug location points at the
    # subprogram it came from (orphaned once that function is inlined away), and Julia's alias
    # scopes are labelled with the function they were derived for
    md_kinds = (LLVM.MD_dbg, LLVM.MD_alias_scope, LLVM.MD_noalias, LLVM.MD_tbaa,
                LLVM.MD_tbaa_struct, LLVM.MD_loop)

    # Symbols first, so that a subprogram can adopt its function's final name...
    for f in functions(mod)
        isdeclaration(f) || rename!(f)
    end
    # ...and every surviving function adopts before any instruction walk runs, since a walk
    # can reach another function's subprogram through an inlined debug location and would
    # otherwise give it a rank name instead of that function's symbol.
    for f in functions(mod)
        isdeclaration(f) && continue
        sp = LLVM.subprogram(f)
        sp === nothing || normalize_metadata!(sp, LLVM.name(f))
    end
    for f in functions(mod)
        isdeclaration(f) && continue
        for bb in blocks(f)
            rename!(bb)
            for inst in instructions(bb)
                rename!(inst)
                md = metadata(inst)
                for kind in md_kinds
                    haskey(md, kind) && normalize_metadata!(md[kind])
                end
            end
        end
    end

    # LLVM's function-local value symbol table preserves insertion history, which is not
    # visible in textual IR but does affect bitcode string-table order. Transformations above
    # can create the same named values through pointer-keyed worklists in different orders.
    # Reinsert every local name in IR order so equivalent modules serialize identically.
    for f in functions(mod)
        named_values = Pair{LLVM.Value,String}[]
        for value in parameters(f)
            name = LLVM.name(value)
            isempty(name) || push!(named_values, value => name)
        end
        for bb in blocks(f)
            name = LLVM.name(bb)
            isempty(name) || push!(named_values, bb => name)
            for inst in instructions(bb)
                name = LLVM.name(inst)
                isempty(name) || push!(named_values, inst => name)
            end
        end
        for (value, _) in named_values
            LLVM.name!(value, "")
        end
        for (value, name) in named_values
            LLVM.name!(value, name)
        end
    end
    return
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
    # lower LLVM constructs that the AIR back-end does not support; this takes the place
    # of instruction selection, as our LLVM does not have a Metal target machine.
    lower_air!(job, mod)

    # scrub process-global Julia codegen counters from symbol / debug names, so the emitted
    # bitcode is reproducible across sessions
    normalize_julia_symbol_names!(mod)

    if !isavailable(LLVMDowngrader_jll)
        error("Metal machine-code generation requires the LLVMDowngrader_jll package, which should be installed and loaded first.")
    end

    # downgrade to AIR. Metal's metallib loader is a backward-compatible reader that accepts
    # real LLVM <= 15 bitcode; target LLVM 14 (typed pointers, but with native `bfloat` —
    # unlike the 5.0/7.0 targets) so BFloat16 kernels compile on Julia 1.13+, where Julia
    # emits the native `bfloat` IR type (JuliaGPU/Metal.jl#817).
    air = downgrade(bitcode(mod), v"14.0")

    if format == LLVM.API.LLVMAssemblyFile
        # disassemble the AIR again. the downgrader no longer ships the legacy `llvm-dis`,
        # so parse the bitcode with the in-process LLVM instead; it auto-upgrades on load,
        # so this is textual IR in the in-process LLVM's dialect rather than LLVM 14's.
        Context() do ctx
            @dispose air_mod = parse(LLVM.Module, air) begin
                string(air_mod)
            end
        end
    else
        air
    end
end

# downgrade bitcode to the format of an older LLVM through libllvm_downgrade
function downgrade(input::Vector{UInt8}, version::VersionNumber)
    backend = ExternalBackend(LLVMDowngrader_jll.libllvm_downgrade, "LLVMDG")
    buffer = Ref{Ptr{Cvoid}}(C_NULL)
    message = Ref{Cstring}(C_NULL)
    status = @ccall $(api(backend, "Downgrade"))(input::Ptr{UInt8}, length(input)::Csize_t,
                                                 version.major::Cuint, version.minor::Cuint,
                                                 buffer::Ptr{Ptr{Cvoid}},
                                                 message::Ptr{Cstring})::Cint
    external_result(backend, status, "Failed to downgrade bitcode to LLVM $(version)",
                    message[], String[], input)
    return take_buffer(backend, buffer[])
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
    byref = falses(length(parameters(ft)))
    # ... and among those, the boxed ones: an argument that survived `check_invocation`
    # despite not being a bitstype has no fields, so it can only be used by identity (an
    # interned `Symbol` being the typical case). Rather than a buffer, the host passes its
    # address as a bare word, which this pass turns back into the pointer the body expects.
    identity_word = falses(length(parameters(ft)))
    args = classify_arguments(job, ft; post_optimization=job.config.optimize)
    filter!(args) do arg
        arg.cc != GHOST
    end
    for arg in args
        param = parameters(ft)[arg.idx]
        identity_word[arg.idx] = arg.cc == MUT_REF && param isa LLVM.PointerType &&
                                 addrspace(param) == 0
        byref[arg.idx] = arg.cc == BITS_REF || arg.cc == KERNEL_STATE ||
                         identity_word[arg.idx]
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
            if identity_word[i]
                # recover the boxed argument's address from the word the host passed
                T_word = convert(LLVMType, UInt)
                slot = parameters(new_f)[i]
                if supports_typed_pointers(context())
                    slot = bitcast!(builder, slot,
                                    LLVM.PointerType(T_word, addrspace(value_type(slot))))
                end
                push!(new_args, inttoptr!(builder, load!(builder, T_word, slot), param))
            elseif byref[i]
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
# packages can override this for target-specific globals that must stay in AS0.
metal_global_constant_addrspace(@nospecialize(job::CompilerJob{MetalCompilerTarget}),
                                @nospecialize(gv::LLVM.GlobalVariable)) = 2

function add_global_address_spaces!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                    entry::LLVM.Function)
    # determine global variables we need to update
    global_map = Dict{LLVM.Value, LLVM.Value}()
    for gv in globals(mod)
        isconstant(gv) || continue
        addrspace(value_type(gv)) == 0 || continue

        new_addrspace = metal_global_constant_addrspace(job, gv)
        new_addrspace == addrspace(value_type(gv)) && continue

        gv_ty = global_value_type(gv)
        gv_name = LLVM.name(gv)

        LLVM.name!(gv, gv_name * ".old")
        new_gv = GlobalVariable(mod, gv_ty, gv_name, new_addrspace)

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
    # Clones are appended in processing order, so don't iterate the pointer-hashed worklist.
    if !isempty(function_worklist)
        entry_fn = LLVM.name(entry)
        for fun in [f for f in functions(mod) if f in function_worklist]
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
        # Rewrite constant-expression uses left after cloning.
        replace_uses!(old, new)
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

function select_source_and_null(sel::LLVM.SelectInst)
    ops = collect(operands(sel))
    cond, lhs, rhs = ops
    src_lhs = addrspacecast_to_generic_source(lhs)
    src_rhs = addrspacecast_to_generic_source(rhs)
    if src_lhs !== nothing && isnull(rhs)
        return cond, src_lhs, true
    elseif isnull(lhs) && src_rhs !== nothing
        return cond, src_rhs, false
    end
    return nothing
end

function rewrite_generic_null_selects!(mod::LLVM.Module)
    changed = false
    worklist = LLVM.ICmpInst[]
    for f in functions(mod)
        isdeclaration(f) && continue
        for bb in blocks(f), inst in instructions(bb)
            inst isa LLVM.ICmpInst || continue
            pred = predicate(inst)
            pred in (LLVM.API.LLVMIntEQ, LLVM.API.LLVMIntNE) || continue
            ops = collect(operands(inst))
            sel_idx = ops[1] isa LLVM.SelectInst && isnull(ops[2]) ? 1 :
                      ops[2] isa LLVM.SelectInst && isnull(ops[1]) ? 2 : 0
            sel_idx == 0 && continue
            sel = ops[sel_idx]::LLVM.SelectInst
            value_type(sel) isa LLVM.PointerType || continue
            addrspace(value_type(sel)) == 0 || continue
            select_info = select_source_and_null(sel)
            select_info === nothing && continue
            push!(worklist, inst)
        end
    end

    for inst in worklist
        ops = collect(operands(inst))
        sel = (ops[1] isa LLVM.SelectInst ? ops[1] : ops[2])::LLVM.SelectInst
        cond, src, cast_is_true_value = select_source_and_null(sel)
        @dispose builder=IRBuilder() begin
            position!(builder, inst)
            src_is_null = icmp!(builder, LLVM.API.LLVMIntEQ, src, null(value_type(src)))
            replacement = if predicate(inst) == LLVM.API.LLVMIntEQ
                select!(builder, cond,
                        cast_is_true_value ? src_is_null : ConstantInt(LLVM.Int1Type(), 1),
                        cast_is_true_value ? ConstantInt(LLVM.Int1Type(), 1) : src_is_null)
            else
                src_is_not_null = icmp!(builder, LLVM.API.LLVMIntNE, src, null(value_type(src)))
                select!(builder, cond,
                        cast_is_true_value ? src_is_not_null : ConstantInt(LLVM.Int1Type(), 0),
                        cast_is_true_value ? ConstantInt(LLVM.Int1Type(), 0) : src_is_not_null)
            end
            replace_uses!(inst, replacement)
            erase!(inst)
        end
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

        # only pointer-to-data arguments are written through; by-reference values (kernel
        # state, bitstype objects) are read into a stack slot and never written back.
        if arg.cc == BITS_VALUE && (arg.typ <: Ptr || arg.typ <: Core.LLVMPtr)
            push!(md, MDString("air.read_write"))
        else
            push!(md, MDString("air.read"))
        end

        push!(md, MDString("air.address_space"))
        push!(md, Metadata(ConstantInt(Int32(addrspace(parameters(entry_ft)[arg.idx])))))

        # A fieldless boxed argument (e.g. an interned `Symbol`) is passed as its bare address
        # word, so describe that word: its Julia type has no size to report.
        arg_type = if arg.cc == MUT_REF
            UInt
        elseif arg.typ <: Core.LLVMPtr
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

# The function `name` of `mod`, declaring it with type `ft` if it doesn't exist yet. An
# existing function must have that type.
function declare!(mod::LLVM.Module, name::String, ft::LLVM.FunctionType)
    fns = functions(mod)
    haskey(fns, name) || return LLVM.Function(mod, name, ft)
    f = fns[name]
    function_type(f) == ft ||
        error("Conflicting declarations of $name: $(function_type(f)) and $ft")
    return f
end

# Call the function `name`, declaring it for the types of `args` and return type `T_ret`.
function call_declared!(builder::IRBuilder, name::String, T_ret::LLVMType,
                        args::Vector{<:LLVM.Value})
    mod = LLVM.parent(LLVM.parent(position(builder)))
    ft = LLVM.FunctionType(T_ret, LLVMType[value_type(arg) for arg in args])
    return call!(builder, ft, declare!(mod, name, ft), args)
end

# the suffix LLVM and AIR use to mangle overloaded intrinsics on `typ`, e.g. `v4f32`
function type_suffix(@nospecialize(typ::LLVMType))
    typ isa LLVM.IntegerType && return "i$(width(typ))"
    typ == LLVM.HalfType() && return "f16"
    typ == LLVM.BFloatType() && return "bf16"
    typ == LLVM.FloatType() && return "f32"
    typ == LLVM.DoubleType() && return "f64"
    typ isa LLVM.VectorType && return "v$(length(typ))$(type_suffix(eltype(typ)))"
    error("Unsupported intrinsic type: $typ")
end

# the Julia floating-point type of an LLVM one (the C API lacks getPrimitiveSizeInBits)
function julia_float_type(typ::LLVMType)
    typ == LLVM.HalfType() && return Float16
    typ == LLVM.FloatType() && return Float32
    typ == LLVM.DoubleType() && return Float64
    error("Unsupported floating-point type: $typ")
end

# the intrinsic that `inst` calls, if any
# The tables below map intrinsics by name, but calls are matched on the intrinsic's ID:
# `LLVM.name` cannot name an overloaded intrinsic (LLVM asserts that it isn't), and IDs are
# only known at run time.
intrinsic_table(table) = Dict(LLVM.Intrinsic(name) => val for (name, val) in table)

function called_intrinsic(inst::LLVM.Instruction)
    inst isa LLVM.CallBase || return nothing
    callee = called_operand(inst)
    (callee isa LLVM.Function && LLVM.isintrinsic(callee)) || return nothing
    return LLVM.Intrinsic(callee)
end

# Lower the calls to intrinsics in `fun`: `lower(builder, call, intrinsic)` is called for each,
# with `builder` positioned at the call, and returns the value that replaces the call,
# `:erase` to remove it, or `nothing` to keep it. The calls are collected first, so `lower`
# can emit code, but it must not erase other instructions.
function lower_intrinsic_calls!(lower, fun::LLVM.Function)
    calls = LLVM.CallBase[]
    for bb in blocks(fun), inst in instructions(bb)
        inst isa LLVM.CallBase || continue
        called_intrinsic(inst) === nothing || push!(calls, inst)
    end
    isempty(calls) && return false
    changed = false
    @dispose builder=IRBuilder() begin
        for call in calls
            position!(builder, call)
            debuglocation!(builder, call)
            new = lower(builder, call, called_intrinsic(call))
            new === nothing && continue
            new === :erase || replace_uses!(call, new)
            erase!(call)
            changed = true
        end
    end
    return changed
end

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
    mod = LLVM.parent(fun)
    return lower_intrinsic_calls!(fun) do builder, call, intr
        vecty = value_type(call)
        (intr in minmax && vecty isa LLVM.VectorType) || return nothing
        # the scalar overload of the same intrinsic, e.g. llvm.minimum.v4f32 -> llvm.minimum.f32
        scalar_f = LLVM.Function(mod, intr, LLVMType[eltype(vecty)])
        scalar_ft = function_type(scalar_f)
        arg0, arg1 = arguments(call)
        res = PoisonValue(vecty)
        for i in 0:Int(length(vecty))-1
            idx = ConstantInt(LLVM.Int32Type(), i)
            a = extract_element!(builder, arg0, idx)
            b = extract_element!(builder, arg1, idx)
            s = call!(builder, scalar_ft, scalar_f, LLVM.Value[a, b])
            res = insert_element!(builder, res, s, idx)
        end
        res
    end
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
#
# llvm intrinsic => (precise air op, relaxed f32 air op or `nothing`)
# Verified against Apple's frontend (`xcrun metal -S -emit-llvm`, precise vs -ffast-math):
# every op has an `air.<op>.f16` and `air.<op>.f32`; all but `fma` also have an f32-only
# `air.fast_<op>` that Apple selects under fast math. Half always stays precise, and `fma`
# is exact so even fast math keeps `air.fma.{f16,f32}`.
const AIR_MATH_INTRINSICS = Dict(
    "llvm.sqrt"  => ("air.sqrt",  "air.fast_sqrt"),
    "llvm.fma"   => ("air.fma",   nothing),
    "llvm.floor" => ("air.floor", "air.fast_floor"),
    "llvm.ceil"  => ("air.ceil",  "air.fast_ceil"),
    "llvm.trunc" => ("air.trunc", "air.fast_trunc"),
    "llvm.rint"  => ("air.rint",  "air.fast_rint"),
)
function lower_math_intrinsics!(fun::LLVM.Function)
    math_intrinsics = intrinsic_table(AIR_MATH_INTRINSICS)
    return lower_intrinsic_calls!(fun) do builder, call, intr
        mapping = get(math_intrinsics, intr, nothing)
        mapping === nothing && return nothing
        # Metal floats are f16/f32 only; skip f64 (rejected by validate_ir) and vector types
        # (these ops have no `air.<op>.v4f32`) rather than synthesize a nonexistent intrinsic.
        typ = value_type(call)
        (typ == LLVM.HalfType() || typ == LLVM.FloatType()) || return nothing
        precise, fast = mapping
        # the relaxed variant exists for f32 only; f16 always uses the precise op
        use_fast = fast !== nothing && typ == LLVM.FloatType() && LLVM.fast_math(call).afn
        call_declared!(builder, "$(use_fast ? fast : precise).$(type_suffix(typ))", typ,
                       collect(LLVM.Value, arguments(call)))
    end
end

# Fuse chained integer min/max into AIR's native 3-way builtins: a 2-way
# `air.{min,max}.{s,u}.iN` whose operand is a single-use call to the same builtin becomes
# `air.{min,max}3.{s,u}.iN(a, b, c)`. AGX has 3-way min/max, but neither Julia (which reduces
# `min(a,b,c)` to nested 2-arg calls) nor Apple's own frontend emits it. Done in the back-end so
# every chained min/max benefits, not just literal 3-argument calls. Integer only: float min/max
# go through the NaN-propagating wrapper, which `air.f{min,max}3` would not preserve.
function fuse_minmax3!(fun::LLVM.Function)
    pat = r"^air\.(min|max)\.(s|u)\.i(8|16|32|64)$"
    function minmax_callee(inst)
        inst isa LLVM.CallInst || return nothing
        callee = called_operand(inst)
        (callee isa LLVM.Function && occursin(pat, LLVM.name(callee))) || return nothing
        return LLVM.name(callee)
    end

    # the next pair to fold: an outer call with an operand that is a call to the same builtin
    # (the name pins down min/max, signedness and width) feeding only the outer one, else
    # folding it would drop a live value
    function next_fold()
        for bb in blocks(fun), outer in instructions(bb)
            outer isa LLVM.CallInst || continue
            fn = minmax_callee(outer)
            fn === nothing && continue
            args = arguments(outer)
            length(args) == 2 || continue
            for (inner, other) in ((args[1], args[2]), (args[2], args[1]))
                minmax_callee(inner) == fn && length(collect(uses(inner))) == 1 &&
                    return outer, inner, other
            end
        end
        return nothing
    end

    # fold one pair then rescan: each fold removes a call, so this terminates, and rescanning
    # avoids mutating the instruction stream while iterating it.
    changed = false
    while (fold = next_fold()) !== nothing
        outer, inner, other = fold
        fn3 = replace(LLVM.name(called_operand(outer)), r"^air\.(min|max)\." => s"air.\g<1>3.")
        @dispose builder=IRBuilder() begin
            position!(builder, outer)
            debuglocation!(builder, outer)
            a, b = arguments(inner)
            replace_uses!(outer, call_declared!(builder, fn3, value_type(outer),
                                                LLVM.Value[a, b, other]))
            erase!(outer)
            erase!(inner)   # now dead (its only use was `outer`)
        end
        changed = true
    end
    return changed
end

# unsupported intrinsics that are safe to remove
const REMOVABLE_INTRINSICS = ("llvm.experimental.noalias.scope.decl", "llvm.lifetime.start",
                              "llvm.lifetime.end", "llvm.assume")

# intrinsics that map straight to AIR functions on the same values, suffixed by the type and,
# for integers, the signedness: llvm intrinsic => (air function, signed)
const AIR_VALUE_INTRINSICS = Dict(
    # one argument
    "llvm.abs"      => ("air.abs", true),
    "llvm.fabs"     => ("air.fabs", missing),
    # two arguments
    "llvm.umin"     => ("air.min", false),
    "llvm.smin"     => ("air.min", true),
    "llvm.umax"     => ("air.max", false),
    "llvm.smax"     => ("air.max", true),
    "llvm.minnum"   => ("air.fmin", missing),
    "llvm.maxnum"   => ("air.fmax", missing),
)

# integer bit intrinsics: pure renames to AIR's builtin names (same signature, including
# the `i1` on clz/ctz). Apple's frontend emits these `air.*` rather than the `llvm.*`
# forms, so we rename rather than rely on the metallib loader accepting `llvm.*`.
const AIR_BIT_INTRINSICS = Dict(
    "llvm.ctlz"       => "air.clz",
    "llvm.cttz"       => "air.ctz",
    "llvm.ctpop"      => "air.popcount",
    "llvm.bitreverse" => "air.reverse_bits",
)

function lower_value_intrinsic!(builder::IRBuilder, call::LLVM.CallBase, fn::String, signed)
    typ = value_type(call)
    elty = typ isa LLVM.VectorType ? eltype(typ) : typ

    # AIR has no native bfloat fabs/fmin/fmax (MSL promotes bfloat to float for them), so do
    # the same: call the float function on fpext'd operands and fptrunc the result back.
    # `optyp` is the type the AIR call actually uses.
    promote_bf = elty == LLVM.BFloatType()
    optyp = if !promote_bf
        typ
    elseif typ isa LLVM.VectorType
        LLVM.VectorType(LLVM.FloatType(), Int(length(typ)))
    else
        LLVM.FloatType()
    end
    fn *= elty isa LLVM.IntegerType ? ".$(signed::Bool ? "s" : "u").$(type_suffix(optyp))" :
                                      ".$(type_suffix(optyp))"

    # AIR's value intrinsics take only the value operands. `llvm.abs` carries an extra
    # `i1 is_int_min_poison` flag that `air.abs` does not, so drop any operand whose type
    # isn't the result type. (For the others every operand is the result type.)
    args = LLVM.Value[arg for arg in arguments(call) if value_type(arg) == typ]
    promote_bf && (args = LLVM.Value[fpext!(builder, arg, optyp) for arg in args])
    new = call_declared!(builder, fn, optyp, args)
    return promote_bf ? fptrunc!(builder, new, typ) : new
end

# floating-point class tests, which LLVM forms out of combined comparisons (e.g., of
# `isfinite` and `iszero`) but AIR does not support: test the value's bits instead
function lower_is_fpclass!(builder::IRBuilder, call::LLVM.CallBase)
    x, test = arguments(call)
    jltyp = julia_float_type(value_type(x))
    mask = convert(Int, test)

    ityp = LLVM.IntType(8*sizeof(jltyp))
    bits = bitcast!(builder, x, ityp)
    magnitude = and!(builder, bits, LLVM.ConstantInt(ityp, ~Base.sign_mask(jltyp)))

    # tests for the classes of the magnitude, emitted when needed
    inf = Base.exponent_mask(jltyp)
    qnan = inf | (Base.significand_mask(jltyp) + one(inf)) >> 1
    normal = reinterpret(Unsigned, floatmin(jltyp))
    compare(pred, lhs, rhs) = icmp!(builder, pred, lhs, LLVM.ConstantInt(ityp, rhs))
    test_nan() = compare(LLVM.API.LLVMIntUGT, magnitude, inf)
    test_qnan() = compare(LLVM.API.LLVMIntUGE, magnitude, qnan)
    test_inf() = compare(LLVM.API.LLVMIntEQ, magnitude, inf)
    test_normal() = compare(LLVM.API.LLVMIntULT,
                            sub!(builder, magnitude, LLVM.ConstantInt(ityp, normal)),
                            inf - normal)
    test_subnormal() = compare(LLVM.API.LLVMIntULT,
                               sub!(builder, magnitude, LLVM.ConstantInt(ityp, 1)),
                               normal - 1)
    test_zero() = compare(LLVM.API.LLVMIntEQ, magnitude, 0)
    negative = nothing
    function with_sign(test, neg)
        if negative === nothing
            negative = compare(LLVM.API.LLVMIntSLT, bits, 0)
        end
        and!(builder, test, neg ? negative : not!(builder, negative))
    end

    # combine the tested classes, as encoded by the `FPClassTest` mask bits
    terms = LLVM.Value[]
    tested(bit) = mask & (1 << bit) != 0
    if tested(0) && tested(1)
        push!(terms, test_nan())
    elseif tested(0)
        push!(terms, and!(builder, test_nan(), not!(builder, test_qnan())))
    elseif tested(1)
        push!(terms, test_qnan())
    end
    for (test, negbit, posbit) in ((test_inf, 2, 9), (test_normal, 3, 8),
                                   (test_subnormal, 4, 7), (test_zero, 5, 6))
        if tested(negbit) && tested(posbit)
            push!(terms, test())
        elseif tested(negbit) || tested(posbit)
            push!(terms, with_sign(test(), tested(negbit)))
        end
    end
    return isempty(terms) ? LLVM.ConstantInt(LLVM.Int1Type(), 0) :
                            foldl((a, b) -> or!(builder, a, b), terms)
end

# copysign, by twiddling the sign bit
function lower_copysign!(builder::IRBuilder, call::LLVM.CallBase)
    arg0, arg1 = arguments(call)
    typ = value_type(call)
    jltyp = julia_float_type(typ)
    ityp = LLVM.IntType(8*sizeof(jltyp))
    arg0′ = bitcast!(builder, arg0, ityp)
    arg1′ = bitcast!(builder, arg1, ityp)
    sign = and!(builder, arg1′, LLVM.ConstantInt(ityp, Base.sign_mask(jltyp)))
    mantissa = and!(builder, arg0′, LLVM.ConstantInt(ityp, ~Base.sign_mask(jltyp)))
    return bitcast!(builder, or!(builder, sign, mantissa), typ)
end

# IEEE 754-2018 compliant maximum/minimum, propagating NaNs and treating -0 as less than +0
function lower_minimum_maximum!(builder::IRBuilder, call::LLVM.CallBase, minmax::String)
    mod = LLVM.parent(LLVM.parent(position(builder)))
    typ = value_type(call)

    # AIR has no bfloat min/max, so promote to float as MSL does: build the wrapper
    # in float and fpext/fptrunc around it. `optyp` is the type it operates on.
    promote_bf = typ == LLVM.BFloatType()
    optyp = promote_bf ? LLVM.FloatType() : typ
    op_ft = LLVM.FunctionType(optyp, LLVMType[optyp, optyp])
    jltyp = julia_float_type(optyp)
    bits = 8*sizeof(jltyp)

    # @fastmath / fastmath=true set `nnan` (assume no NaNs), so we can skip the
    # NaN-propagating wrapper and call the relaxed AIR builtin directly, matching Apple's
    # -ffast-math: f32 has air.fast_f{min,max}; f16 has no fast form, so use air.f{min,max}.
    # otherwise create a function that performs the IEEE-compliant operation. normally
    # we'd do this inline, but LLVM.jl doesn't have BB split functionality.
    nnan = LLVM.fast_math(call).nnan
    fn = if !nnan
        "air.$(minmax)imum.f$bits"
    elseif optyp == LLVM.FloatType()
        "air.fast_f$minmax.f32"
    else
        "air.f$minmax.f$bits"
    end
    f = if nnan || haskey(functions(mod), fn)
        declare!(mod, fn, op_ft)
    else
        build_minimum_maximum!(mod, fn, op_ft, jltyp, minmax)
    end

    args = collect(LLVM.Value, arguments(call))
    promote_bf && (args = LLVM.Value[fpext!(builder, arg, optyp) for arg in args])
    new = call!(builder, op_ft, f, args)
    return promote_bf ? fptrunc!(builder, new, typ) : new
end

# integer power, which AIR lacks (MSL has no `pown`, and `pow` is undefined for negative
# bases), by exponentiation by squaring as LLVM's back-ends expand it: a constant exponent
# is unrolled into multiplies like SelectionDAG's `ExpandPowI`, any other calls a loop over
# the exponent's bits like compiler-rt's `__powisf2`. A negative exponent takes the
# reciprocal, and `x^0` is 1 (even for NaN). The exponent of a vector `powi` is a scalar.
function lower_powi!(builder::IRBuilder, call::LLVM.CallBase)
    x, n = arguments(call)
    if n isa LLVM.ConstantInt && width(value_type(n)) <= 64
        return expand_powi!(builder, x, convert(Int, n))
    end

    # the loop halves the exponent, which doesn't work for an `i1` (where 2 wraps to 0)
    width(value_type(n)) < 32 && (n = sext!(builder, n, LLVM.Int32Type()))

    mod = LLVM.parent(LLVM.parent(position(builder)))
    typ, ntyp = value_type(x), value_type(n)
    fn ="air.powi.$(type_suffix(typ)).$(type_suffix(ntyp))"
    f = haskey(functions(mod), fn) ? functions(mod)[fn] : build_powi!(mod, fn, typ, ntyp)
    return call!(builder, function_type(f), f, LLVM.Value[x, n])
end

# 1.0 of a floating-point type, splat across the lanes of a vector type
fp_one(typ::LLVMType) = LLVM.Value(LLVM.API.LLVMConstReal(typ, 1.0))

function expand_powi!(builder::IRBuilder, x::LLVM.Value, n::Int)
    m = unsigned(abs(n))    # also for `typemin(n)`, which `abs` returns as is
    res = nothing           # 1.0, until the first set bit
    sq = x                  # x^(2^i) for the current bit i
    while m != 0
        isodd(m) && (res = res === nothing ? sq : fmul!(builder, res, sq))
        m >>= 1
        m != 0 && (sq = fmul!(builder, sq, sq))
    end
    res = something(res, fp_one(value_type(x)))
    return n < 0 ? fdiv!(builder, fp_one(value_type(x)), res) : res
end

function build_powi!(mod::LLVM.Module, fn::String, typ::LLVMType, ntyp::LLVMType)
    f = LLVM.Function(mod, fn, LLVM.FunctionType(typ, LLVMType[typ, ntyp]))
    linkage!(f, LLVM.API.LLVMInternalLinkage)
    push!(function_attributes(f), EnumAttribute("alwaysinline"))
    x, n = parameters(f)
    one = fp_one(typ)
    zero = LLVM.ConstantInt(ntyp, 0)

    bb_entry = BasicBlock(f, "entry")
    bb_loop = BasicBlock(f, "loop")
    bb_done = BasicBlock(f, "done")
    @dispose builder=IRBuilder() begin
        position!(builder, bb_entry)
        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, n, zero), bb_done, bb_loop)

        # multiply the squares selected by the bits of `n`, least significant first. like
        # `__powisf2`, shift the signed `n` by halving it (rounding towards zero), so as not
        # to take `abs(n)`, which InstCombine would turn into an `llvm.abs` after that has
        # already been lowered.
        position!(builder, bb_loop)
        acc = phi!(builder, typ, "acc")
        sq = phi!(builder, typ, "sq")
        rest = phi!(builder, ntyp, "rest")
        bit = trunc!(builder, rest, LLVM.Int1Type())
        acc′ = select!(builder, bit, fmul!(builder, acc, sq), acc)
        sq′ = fmul!(builder, sq, sq)
        rest′ = sdiv!(builder, rest, LLVM.ConstantInt(ntyp, 2))
        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, rest′, zero), bb_done, bb_loop)
        append!(incoming(acc), [(one, bb_entry), (acc′, bb_loop)])
        append!(incoming(sq), [(x, bb_entry), (sq′, bb_loop)])
        append!(incoming(rest), [(n, bb_entry), (rest′, bb_loop)])

        position!(builder, bb_done)
        pow = phi!(builder, typ, "pow")
        append!(incoming(pow), [(one, bb_entry), (acc′, bb_loop)])
        negative = icmp!(builder, LLVM.API.LLVMIntSLT, n, zero)
        ret!(builder, select!(builder, negative, fdiv!(builder, one, pow), pow))
    end
    return f
end

function build_minimum_maximum!(mod::LLVM.Module, fn::String, op_ft::LLVM.FunctionType,
                                jltyp::Type, minmax::String)
    optyp = return_type(op_ft)
    f = LLVM.Function(mod, fn, op_ft)
    push!(function_attributes(f), EnumAttribute("alwaysinline"))
    arg0, arg1 = parameters(f)

    bb_check_arg0 = BasicBlock(f, "check_arg0")
    bb_nan_arg0 = BasicBlock(f, "nan_arg0")
    bb_check_arg1 = BasicBlock(f, "check_arg1")
    bb_nan_arg1 = BasicBlock(f, "nan_arg1")
    bb_check_zero = BasicBlock(f, "check_zero")
    bb_compare_zero = BasicBlock(f, "compare_zero")
    bb_fallback = BasicBlock(f, "fallback")

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
        val = if minmax == "min"
            select!(builder, arg0_negative, arg0, arg1)
        else
            select!(builder, arg0_negative, arg1, arg0)
        end
        ret!(builder, val)

        # finally, it's safe to use the existing minnum/maxnum intrinsics

        position!(builder, bb_fallback)
        fallback = declare!(mod, "air.f$minmax.f$(8*sizeof(jltyp))", op_ft)
        ret!(builder, call!(builder, op_ft, fallback, collect(parameters(f))))
    end
    return f
end

# replace LLVM intrinsics with AIR equivalents
function lower_llvm_intrinsics!(@nospecialize(job::CompilerJob), fun::LLVM.Function)
    isdeclaration(fun) && return false

    # AIR lacks vector min/max intrinsics; scalarize so the per-call lowering below applies.
    changed = scalarize_vector_minmax!(fun)

    # lower the floating-point math intrinsics Julia emits (sqrt, fma, floor, ...) to their
    # AIR device functions, picking the relaxed `air.fast_*` variant for `afn`-flagged calls.
    changed |= lower_math_intrinsics!(fun)

    removable = Set(LLVM.Intrinsic.(REMOVABLE_INTRINSICS))
    value_intrinsics = intrinsic_table(AIR_VALUE_INTRINSICS)
    bit_intrinsics = intrinsic_table(AIR_BIT_INTRINSICS)
    is_fpclass, copysign, minimum, maximum, powi =
        LLVM.Intrinsic.(("llvm.is.fpclass", "llvm.copysign", "llvm.minimum", "llvm.maximum",
                         "llvm.powi"))
    changed |= lower_intrinsic_calls!(fun) do builder, call, intr
        if intr in removable
            :erase
        elseif haskey(value_intrinsics, intr)
            lower_value_intrinsic!(builder, call, value_intrinsics[intr]...)
        elseif haskey(bit_intrinsics, intr)
            # keep the mangled type suffix, e.g. llvm.ctlz.i32 -> air.clz.i32
            typ = value_type(call)
            call_declared!(builder, "$(bit_intrinsics[intr]).$(type_suffix(typ))", typ,
                           collect(LLVM.Value, arguments(call)))
        elseif intr == is_fpclass
            lower_is_fpclass!(builder, call)
        elseif intr == copysign
            lower_copysign!(builder, call)
        elseif intr == minimum || intr == maximum
            lower_minimum_maximum!(builder, call, intr == minimum ? "min" : "max")
        elseif intr == powi
            lower_powi!(builder, call)
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

        # atomics: as Apple declares them. not `argmemonly` or `readonly`, which would let
        # LLVM hoist an atomic load out of a spin loop, or move other memory accesses across
        # an ordered atomic.
        elseif match(r"^air.atomic.(local|global)\.", fn) !== nothing
            add_fn_attributes("mustprogress", "nounwind", "willreturn")

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
        elseif match(r"air.(simd|quad)_(ballot|all|vote_all|any|vote_any|shuffle|shuffle_xor|shuffle_down|\
            shuffle_up|shuffle_and_fill_down|shuffle_and_fill_up)", fn) !== nothing
            add_fn_attributes("convergent", "mustprogress", "nounwind", "willreturn")
        end
    end

    return changed
end
