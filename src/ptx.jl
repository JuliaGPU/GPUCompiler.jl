# implementation of the GPUCompiler interfaces for generating PTX code

## target

export PTXCompilerTarget

# Wire-format encoding of the feature set, stamped into the `sm_features` LLVM
# global by `finish_module!` and read back by host-side runtime intrinsics (e.g.
# CUDA.jl's `target_feature_set()`).
@enum TargetFeatureSet::UInt32 begin
    BaselineFeatures = 0
    FamilyFeatures   = 1
    ArchFeatures     = 2
end

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    cap::VersionNumber
    ptx::VersionNumber = v"6.0" # for compatibility with older versions of CUDA.jl

    # subtarget feature set, selecting the suffix on the LLVM CPU name (and `.target`):
    #   :baseline (no suffix)   - forward-compatible (sm_X for any sm_Y >= X)
    #   :family   ('f' suffix)  - same-major-family-portable; gates 'f'-tier intrinsics
    #   :arch     ('a' suffix)  - locked to one exact CC; unlocks all arch-accel intrinsics
    feature_set::Symbol = :baseline

    # codegen quirks
    ## can we emit debug info in the PTX assembly?
    debuginfo::Bool = false

    # optional properties
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing

    fastmath::Bool = Base.JLOptions().fast_math == 1

    # deprecated; remove with next major version
    exitable::Union{Nothing,Bool} = nothing
    unreachable::Union{Nothing,Bool} = nothing
end

function Base.hash(target::PTXCompilerTarget, h::UInt)
    h = hash(target.cap, h)
    h = hash(target.ptx, h)
    h = hash(target.feature_set, h)

    h = hash(target.debuginfo, h)

    h = hash(target.minthreads, h)
    h = hash(target.maxthreads, h)
    h = hash(target.blocks_per_sm, h)
    h = hash(target.maxregs, h)
    h = hash(target.fastmath, h)

    h
end

# format the LLVM CPU / PTX `.target` name for this target
function cpu_name(target::PTXCompilerTarget)
    suffix = target.feature_set === :arch    ? "a" :
             target.feature_set === :family  ? "f" :
             target.feature_set === :baseline ? "" :
             error("PTXCompilerTarget.feature_set must be one of :baseline, :family, :arch; got $(repr(target.feature_set))")
    return "sm_$(target.cap.major)$(target.cap.minor)$suffix"
end

source_code(target::PTXCompilerTarget) = "ptx"

llvm_triple(target::PTXCompilerTarget) = Int===Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda"

function llvm_machine(target::PTXCompilerTarget)
    @static if :NVPTX ∉ LLVM.backends()
        return nothing
    end
    triple = llvm_triple(target)
    t = Target(triple=triple)

    tm = TargetMachine(t, triple, cpu_name(target),
                       "+ptx$(target.ptx.major)$(target.ptx.minor)")
    asm_verbosity!(tm, true)

    return tm
end

# the default datalayout does not match the one in the NVPTX user guide
llvm_datalayout(target::PTXCompilerTarget) =
    # little endian
    "e-" *
    # on 32-bit systems, use 32-bit pointers.
    # on 64-bit systems, use 64-bit pointers.
    (Int === Int64 ? "p:64:64:64-" :  "p:32:32:32-") *
    # alignment of integer types
    "i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-" *
    # alignment of floating point types
    "f32:32:32-f64:64:64-" *
    # alignment of vector types
    "v16:16:16-v32:32:32-v64:64:64-v128:128:128-" *
    # native integer widths
    "n16:32:64"

have_fma(@nospecialize(target::PTXCompilerTarget), T::Type) = true

dwarf_version(target::PTXCompilerTarget) = Int32(2) # Cuda only supports dwarfv2

can_vectorize(job::CompilerJob{PTXCompilerTarget}) = true

## job

function Base.show(io::IO, @nospecialize(job::CompilerJob{PTXCompilerTarget}))
    print(io, "PTX CompilerJob of ", job.source)
    print(io, " for ", cpu_name(job.config.target))

    job.config.target.minthreads !== nothing && print(io, ", minthreads=$(job.config.target.minthreads)")
    job.config.target.maxthreads !== nothing && print(io, ", maxthreads=$(job.config.target.maxthreads)")
    job.config.target.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.config.target.blocks_per_sm)")
    job.config.target.maxregs !== nothing && print(io, ", maxregs=$(job.config.target.maxregs)")
    job.config.target.fastmath && print(io, ", fast math enabled")
end

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(@nospecialize(job::CompilerJob{PTXCompilerTarget}), fn::String) =
    in(fn, ptx_intrinsics)

# XXX: the debuginfo part should be handled by GPUCompiler as it applies to all back-ends.
runtime_slug(@nospecialize(job::CompilerJob{PTXCompilerTarget})) =
    "ptx$(job.config.target.ptx.major)$(job.config.target.ptx.minor)" *
    "-$(cpu_name(job.config.target))" *
    "-debuginfo=$(Int(llvm_debug_info(job)))"

function finish_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    # tell NVVMReflect whether to flush denormals; this mirrors what Clang does
    # for `-fcuda-flush-denormals-to-zero` and is the only `__nvvm_reflect` key
    # LLVM's NVVMReflectPass honors besides `__CUDA_ARCH`. only emit it on the
    # toplevel module that runs through `optimize!`, as sub-modules (the cached
    # runtime, deferred jobs) don't need it, and the cached runtime in
    # particular would otherwise conflict on link if it was built with a
    # different `fastmath` setting (which isn't part of `runtime_slug`).
    if job.config.toplevel
        flags(mod)["nvvm-reflect-ftz", LLVM.API.LLVMModuleFlagBehaviorOverride] =
            Metadata(ConstantInt(Int32(job.config.target.fastmath ? 1 : 0)))
    end

    # emit the device capability and ptx isa version as constants in the module. this makes
    # it possible to 'query' these in device code, relying on LLVM to optimize the checks
    # away and generate static code. note that we only do so if there's actual uses of these
    # variables; unconditionally creating a gvar would result in duplicate declarations.
    sm_features = job.config.target.feature_set === :arch    ? ArchFeatures :
                  job.config.target.feature_set === :family  ? FamilyFeatures :
                                                               BaselineFeatures
    for (name, value) in ["sm_major"    => job.config.target.cap.major,
                          "sm_minor"    => job.config.target.cap.minor,
                          "sm_features" => UInt32(sm_features),
                          "ptx_major"   => job.config.target.ptx.major,
                          "ptx_minor"   => job.config.target.ptx.minor]
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            initializer!(gv, ConstantInt(LLVM.Int32Type(), value))
            # change the linkage so that we can inline the value
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        end
    end

    # update calling convention
    if LLVM.version() >= v"8"
        for f in functions(mod)
            # JuliaGPU/GPUCompiler.jl#97
            #callconv!(f, LLVM.API.LLVMPTXDeviceCallConv)
        end
    end
    if job.config.kernel && LLVM.version() >= v"8"
        callconv!(entry, LLVM.API.LLVMPTXKernelCallConv)
    end

    if job.config.kernel
        # work around bad byval codegen (JuliaGPU/GPUCompiler.jl#92)
        entry = lower_byval(job, mod, entry)

        # emit kernel property annotations into the module. these have to be in
        # place before optimization runs: LLVM's NVPTX target machine registers a
        # PipelineStart EP callback that schedules NVVMIntrRangePass, which calls
        # `getMaxNTID` on every function. That populates a module-keyed
        # `AnnotationCache` entry (empty, because `nvvm.annotations` isn't there
        # yet), and subsequent lookups by the asm printer hit the stale empty
        # entry instead of re-reading the metadata.
        annotations = Metadata[entry]

        # kernel metadata
        #
        # on LLVM >= 20 the `ptx_kernel` calling convention already marks the
        # entry; the redundant "kernel" nvvm.annotation causes miscompilations.
        if LLVM.version() < v"20"
            append!(annotations, [MDString("kernel"),
                                  ConstantInt(Int32(1))])
        end

        # expected CTA sizes
        if job.config.target.minthreads !== nothing
            bounds = ntuple(i -> i <= length(job.config.target.minthreads) ?
                                 job.config.target.minthreads[i] : 1, 3)
            for (bound, name) in zip(bounds, (:x, :y, :z))
                append!(annotations, [MDString("reqntid$name"),
                                      ConstantInt(Int32(bound))])
            end
            if LLVM.version() >= v"21"
                push!(function_attributes(entry),
                      StringAttribute("nvvm.reqntid", join(bounds, ",")))
            end
        end
        if job.config.target.maxthreads !== nothing
            bounds = ntuple(i -> i <= length(job.config.target.maxthreads) ?
                                 job.config.target.maxthreads[i] : 1, 3)
            for (bound, name) in zip(bounds, (:x, :y, :z))
                append!(annotations, [MDString("maxntid$name"),
                                      ConstantInt(Int32(bound))])
            end
            if LLVM.version() >= v"21"
                push!(function_attributes(entry),
                      StringAttribute("nvvm.maxntid", join(bounds, ",")))
            end
        end

        if job.config.target.blocks_per_sm !== nothing
            append!(annotations, [MDString("minctasm"),
                                  ConstantInt(Int32(job.config.target.blocks_per_sm))])
            if LLVM.version() >= v"21"
                push!(function_attributes(entry),
                      StringAttribute("nvvm.minctasm", string(job.config.target.blocks_per_sm)))
            end
        end

        if job.config.target.maxregs !== nothing
            append!(annotations, [MDString("maxnreg"),
                                  ConstantInt(Int32(job.config.target.maxregs))])
            if LLVM.version() >= v"21"
                push!(function_attributes(entry),
                      StringAttribute("nvvm.maxnreg", string(job.config.target.maxregs)))
            end
        end

        if length(annotations) > 1
            push!(metadata(mod)["nvvm.annotations"], MDNode(annotations))
        end
    end

    # we emit properties (of the device and ptx isa) as private global constants,
    # so run the optimizer so that they are inlined before the rest of the optimizer runs.
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        run!(pb, mod, llvm_machine(job.config.target))
    end

    return entry
end

function finish_linked_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                               mod::LLVM.Module)
    # propagate `target.fastmath` as `@fastmath`-everywhere semantics
    # (mirrors nvcc's `--use_fast_math`). post-link so that bodies pulled in
    # from libdevice and the runtime also get the flags.
    if job.config.target.fastmath
        apply_fastmath!(mod)
        # additionally request FTZ on f32: NVPTX' `useF32FTZ` reads
        # `denormal-fp-math-f32` to pick the FTZ variants for
        # fdiv/fsqrt/etc.
        for f in functions(mod)
            isdeclaration(f) && continue
            push!(function_attributes(f),
                  StringAttribute("denormal-fp-math-f32",
                                  "preserve-sign,preserve-sign"))
        end
    end
    return
end

function optimize_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                          mod::LLVM.Module)
    tm = llvm_machine(job.config.target)
    # TODO: Use the registered target passes (JuliaGPU/GPUCompiler.jl#450)
    @dispose pb=NewPMPassBuilder() begin
        register!(pb, PTXRSqrtFastPass())
        register!(pb, PTXFDivFastPass())
        register!(pb, PTXFSqrtFastPass())
        if LLVM.version() < v"17"
            # Pre-17 LLVM has no way to invoke EP callbacks from the string
            # API, so fall back to our own nvvm_reflect! implementation.
            # LLVM 17+ picks up NVPTX's built-in NVVMReflectPass through the
            # PipelineStart EP invocations woven into `buildNewPMPipeline!`.
            register!(pb, NVVMReflectPass())
            add!(pb, NVVMReflectPass())
        end
        if get(optimization_options(job), :ptxfastmath, true)
            add!(pb, PTXRSqrtFastPass())
            add!(pb, PTXFDivFastPass())
            add!(pb, PTXFSqrtFastPass())
        end

        add!(pb, NewPMFunctionPassManager()) do fpm
            # needed by GemmKernels.jl-like code
            add!(fpm, SpeculativeExecutionPass())

            # NVPTX's target machine info enables runtime unrolling,
            # but Julia's pass sequence only invokes the simple unroller.
            add!(fpm, LoopUnrollPass(; job.config.opt_level))
            add!(fpm, instcombine_pass(job))        # clean-up redundancy
            add!(fpm, NewPMLoopPassManager(; use_memory_ssa=true)) do lpm
                add!(lpm, LICMPass())           # the inner runtime check might be
                                                # outer loop invariant
            end

            # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
            # that code still needs to be optimized (arguably, multiple unroll passes should be
            # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
            if job.config.opt_level == 2
                add!(fpm, GVNPass())
            elseif job.config.opt_level == 1
                add!(fpm, EarlyCSEPass())
            end
            add!(fpm, DSEPass())

            add!(fpm, SimplifyCFGPass())
        end

        # get rid of the internalized functions; now possible unused
        add!(pb, GlobalDCEPass())

        run!(pb, mod, tm)
    end
end

function finish_ir!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                    mod::LLVM.Module, entry::LLVM.Function)
    if LLVM.version() < v"17"
        for f in functions(mod)
            lower_unreachable!(f)
        end
    end

    return entry
end

function llvm_debug_info(@nospecialize(job::CompilerJob{PTXCompilerTarget}))
    # allow overriding the debug info from CUDA.jl
    if job.config.target.debuginfo
        invoke(llvm_debug_info, Tuple{CompilerJob}, job)
    else
        LLVM.API.LLVMDebugEmissionKindNoDebug
    end
end


## LLVM passes

# lower `unreachable` to `exit` so that the emitted PTX has correct control flow
#
# During back-end compilation, `ptxas` inserts instructions to manage the harware's
# reconvergence stack (SSY and SYNC). In order to do so, it needs to identify
# divergent regions:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     ...
#     bra.uni cont;
#   cont:
#     // end of divergent region
#     bar.sync 0;
#
# Meanwhile, LLVM's branch-folder and block-placement MIR passes will try to optimize
# the block layout, e.g., by placing unlikely blocks at the end of the function:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     @%p1 bra unlikely;
#     bra.uni cont;
#   cont:
#     // end of divergent region
#     bar.sync 0;
#   unlikely:
#     bra.uni cont;
#
# That is not a problem as long as the unlikely block continunes back into the
# divergent region. Crucially, this is not the case with unreachable control flow:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     @%p1 bra throw;
#     bra.uni cont;
#   cont:
#     bar.sync 0;
#   throw:
#     call throw_and_trap();
#     // unreachable
#   exit:
#     // end of divergent region
#     ret;
#
# Dynamically, this is fine, because the called function does not return.
# However, `ptxas` does not know that and adds a successor edge to the `exit`
# block, widening the divergence range. In this example, that's not allowed, as
# `bar.sync` cannot be executed divergently on Pascal hardware or earlier.
#
# To avoid these fall-through successors that change the control flow,
# we replace `unreachable` instructions with a call to `trap` and `exit`. This
# informs `ptxas` that the thread exits, and allows it to correctly construct a
# CFG, and consequently correctly determine the divergence regions as intended.
# Note that we first emit a call to `trap`, so that the behaviour is the same
# as before.
function lower_unreachable!(f::LLVM.Function)
    mod = LLVM.parent(f)

    # TODO:
    # - if unreachable blocks have been merged, we still may be jumping from different
    #   divergent regions, potentially causing the same problem as above:
    #     entry:
    #       // start of divergent region 1
    #       @%p0 bra cont1;
    #       @%p1 bra throw;
    #       bra.uni cont1;
    #     cont1:
    #       // end of divergent region 1
    #       bar.sync 0;   // is this executed divergently?
    #       // start of divergent region 2
    #       @%p2 bra cont2;
    #       @%p3 bra throw;
    #       bra.uni cont2;
    #     cont2:
    #       // end of divergent region 2
    #       ...
    #     throw:
    #       trap;
    #       br throw;
    #   if this is a problem, we probably need to clone blocks with multiple
    #   predecessors so that there's a unique path from each region of
    #   divergence to every `unreachable` terminator

    # remove `noreturn` attributes, to avoid the (minimal) optimization that
    # happens during `prepare_execution!` undoing our work here.
    # this shouldn't be needed when we upstream the pass.
    attrs = function_attributes(f)
    delete!(attrs, EnumAttribute("noreturn", 0))

    # find unreachable blocks
    unreachable_blocks = BasicBlock[]
    for block in blocks(f)
        if terminator(block) isa LLVM.UnreachableInst
            push!(unreachable_blocks, block)
        end
    end
    isempty(unreachable_blocks) && return false

    # inline assembly to exit a thread
    exit_ft = LLVM.FunctionType(LLVM.VoidType())
    exit = InlineAsm(exit_ft, "exit;", "", true)
    trap_ft = LLVM.FunctionType(LLVM.VoidType())
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", trap_ft)
    end

    # rewrite the unreachable terminators
    @dispose builder=IRBuilder() begin
        entry_block = first(blocks(f))
        for block in unreachable_blocks
            inst = terminator(block)
            @assert inst isa LLVM.UnreachableInst

            position!(builder, inst)
            call!(builder, trap_ft, trap)
            call!(builder, exit_ft, exit)
        end
    end

    return true
end

# Replace occurrences of __nvvm_reflect("foo") and llvm.nvvm.reflect with an integer.
#
# This is a back-port of LLVM's NVVMReflectPass for LLVM < 17, where the
# built-in pass cannot be invoked via the string-API PipelineStart EP callback.
# Semantics match LLVM's: `__CUDA_ARCH` is derived from the target capability,
# `__CUDA_FTZ` is read from the `nvvm-reflect-ftz` module flag, and every other
# key folds to 0. Knobs like denormal flushing or FMAD contraction must be
# configured through module flags or LLVM fast-math flags, not here.
const NVVM_REFLECT_FUNCTION = "__nvvm_reflect"
function nvvm_reflect!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @tracepoint "nvvmreflect" begin

    # find and sanity check the nnvm-reflect function
    # TODO: also handle the llvm.nvvm.reflect intrinsic
    haskey(LLVM.functions(mod), NVVM_REFLECT_FUNCTION) || return false
    reflect_function = functions(mod)[NVVM_REFLECT_FUNCTION]
    isdeclaration(reflect_function) || error("_reflect function should not have a body")
    reflect_typ = return_type(function_type(reflect_function))
    isa(reflect_typ, LLVM.IntegerType) || error("_reflect's return type should be integer")

    # pull __CUDA_FTZ from the nvvm-reflect-ftz module flag (same source LLVM uses)
    ftz_val = 0
    if haskey(flags(mod), "nvvm-reflect-ftz")
        flag = flags(mod)["nvvm-reflect-ftz"]
        if flag isa LLVM.ConstantAsMetadata
            c = LLVM.Value(flag)
            if c isa ConstantInt
                ftz_val = Int(convert(Int64, c))
            end
        end
    end

    to_remove = []
    for use in uses(reflect_function)
        call = user(use)
        isa(call, LLVM.CallInst) || continue
        if length(operands(call)) != 2
            @error """Unrecognized format of __nvvm_reflect call:
                      $(string(call))
                      Wrong number of operands: expected 2, got $(length(operands(call)))."""
            continue
        end

        # decode the string argument
        if LLVM.version() >= v"17"
            sym = operands(call)[1]
        else
            str = operands(call)[1]
            if !isa(str, LLVM.ConstantExpr) || opcode(str) != LLVM.API.LLVMGetElementPtr
                @safe_error """Unrecognized format of __nvvm_reflect call:
                               $(string(call))
                               Operand should be a GEP instruction, got a $(typeof(str)). Please file an issue."""
                continue
            end
            sym = operands(str)[1]
            if isa(sym, LLVM.ConstantExpr) && opcode(sym) == LLVM.API.LLVMGetElementPtr
                # CUDA 11.0 or below
                sym = operands(sym)[1]
            end
        end
        if !isa(sym, LLVM.GlobalVariable)
            @safe_error """Unrecognized format of __nvvm_reflect call:
                           $(string(call))
                           Operand should be a global variable, got a $(typeof(sym)). Please file an issue."""
            continue
        end
        sym_op = operands(sym)[1]
        if !isa(sym_op, LLVM.ConstantArray) && !isa(sym_op, LLVM.ConstantDataArray)
            @safe_error """Unrecognized format of __nvvm_reflect call:
                           $(string(call))
                           Operand should be a constant array, got a $(typeof(sym_op)). Please file an issue."""
        end
        chars = convert.(Ref(UInt8), collect(sym_op))
        reflect_arg = String(chars[1:end-1])

        # match LLVM's NVVMReflectPass: unknown keys fold to 0.
        reflect_val = if reflect_arg == "__CUDA_ARCH"
            ConstantInt(reflect_typ,
                        job.config.target.cap.major*100 + job.config.target.cap.minor*10)
        elseif reflect_arg == "__CUDA_FTZ"
            ConstantInt(reflect_typ, ftz_val)
        else
            ConstantInt(reflect_typ, 0)
        end

        replace_uses!(call, reflect_val)
        push!(to_remove, call)
    end

    # remove the calls to the function
    for val in to_remove
        @assert isempty(uses(val))
        erase!(val)
    end

    # maybe also delete the function
    if isempty(uses(reflect_function))
        erase!(reflect_function)
    end

    end
    return changed
end
NVVMReflectPass() = NewPMModulePass("custom-nvvm-reflect", nvvm_reflect!)

# Same source NVPTX' `useF32FTZ` reads — `apply_fastmath!` sets it when
# `target.fastmath=true`. Used here to pick FTZ variants for the f32 rewrite.
function f32_ftz(f::LLVM.Function)
    for attr in collect(LLVM.function_attributes(f))
        attr isa LLVM.StringAttribute || continue
        LLVM.kind(attr) == "denormal-fp-math-f32" || continue
        return startswith(LLVM.value(attr), "preserve-sign")
    end
    return false
end

# All three passes below rewrite `afn`-flagged ops to NVPTX' fast lowerings.
# `apply_fastmath!` propagates job-wide `target.fastmath=true` as per-
# instruction `afn`, so a single flag check covers both per-call `@fastmath`
# and the job toggle. We emit NVPTX intrinsics by name (rather than libdevice
# `__nv_*`) so this doesn't depend on which libdevice symbols got linked in.
#
# `PTXRSqrtFastPass` runs first: the rsqrt pattern (`fdiv afn 1.0, sqrt afn x`)
# spans an fdiv and a sqrt, so it has to claim both operands before the per-op
# passes below eat them. NVPTX has native `rsqrt.approx.{f,d}` for both f32 and
# f64, so this is a single-instruction lowering for both types.
#
# `PTXFDivFastPass` / `PTXFSqrtFastPass` are temporary backports for LLVM 18:
# - `PTXFSqrtFastPass`: on LLVM 21+ `usePrecSqrtF32` honors per-instruction
#   `afn` + the `unsafe-fp-math` attribute, so `DAGCombiner::visitFSQRT` →
#   `NVPTXTargetLowering::getSqrtEstimate` emits the f32 `sqrt.approx{,.ftz}`
#   and f64 `rcp(rsqrt(x))` sequences itself. LLVM 18's `usePrecSqrtF32` only
#   consults `TargetMachine.Options.UnsafeFPMath`, which is unreachable
#   through LLVM.jl.
# - `PTXFDivFastPass`'s f32 path is similarly fixed on LLVM 21+;
#   `getDivF32Level` there honors `afn` + the function attribute. The f64
#   path stays needed until NVPTX gains a `getRecipEstimate` hook (filed
#   upstream).
# On LLVM 21+ both passes (and the f32 path of `PTXRSqrtFastPass`) can be
# dropped together — they have to leave the pipeline as a set, because as
# long as `PTXFDivFastPass` runs and rewrites `fdiv afn 1.0, sqrt(x)` into
# `nvvm.div.approx.f(1.0, ...)`, the rsqrt tablegen pattern can't match.

# Rewrite `fdiv afn 1.0, sqrt afn(x)` to `nvvm.rsqrt.approx.{f,d}(x)`. Must run
# before `PTXFDivFastPass` (which would rewrite the fdiv to `nvvm.div.approx.f`,
# defeating ISel pattern-matching) and `PTXFSqrtFastPass` (which for f64
# expands sqrt into `rcp(rsqrt(...))`, hiding the pattern entirely).
#
# Why we can't rely on LLVM upstream:
# - f32: NVPTX has tablegen patterns (`NVPTXIntrinsics.td`, `doRsqrtOpt`) that
#   match `fdiv 1.0, sqrt_approx(x)` → `rsqrt.approx.f32` — but they landed in
#   LLVM 19 (so absent on our LLVM 18 floor), and even on LLVM 21+ they only
#   fire if the fdiv is still a generic `fdiv`. PTXFDivFastPass kills that.
# - f64: no upstream fold exists at all. NVPTX doesn't override
#   `getRecipEstimateSqrtEnabled`, so the DAGCombiner's generic rsqrt path is
#   disabled, and there's no f64 equivalent of the f32 tablegen patterns.
#   `rsqrt.approx.f64` is a real instruction; it just isn't selected for
#   `1/sqrt(x)` upstream.
function ptx_rsqrt_fast!(mod::LLVM.Module)
    changed = false
    @tracepoint "ptx-rsqrt-fast" begin

    f32 = LLVM.FloatType()
    f64 = LLVM.DoubleType()

    # collect first to avoid mutation-during-iteration
    to_replace = Tuple{LLVM.FDivInst, LLVM.CallInst, Bool}[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        inst isa LLVM.FDivInst || continue
        is_f32 = LLVM.value_type(inst) == f32
        is_f64 = LLVM.value_type(inst) == f64
        (is_f32 || is_f64) || continue
        LLVM.fast_math(inst).afn || continue

        # numerator must be the constant 1.0
        lhs = operands(inst)[1]
        lhs isa LLVM.ConstantFP || continue
        convert(Float64, lhs) == 1.0 || continue

        # denominator must be an `afn`-flagged `llvm.sqrt.f{32,64}` call
        rhs = operands(inst)[2]
        rhs isa LLVM.CallInst || continue
        callee = LLVM.called_operand(rhs)
        callee isa LLVM.Function || continue
        name = LLVM.name(callee)
        expected = is_f32 ? "llvm.sqrt.f32" : "llvm.sqrt.f64"
        name == expected || continue
        LLVM.fast_math(rhs).afn || continue

        push!(to_replace, (inst, rhs, is_f32))
    end
    isempty(to_replace) && return false

    fns = functions(mod)
    declare(name, ft) = haskey(fns, name) ? fns[name] : LLVM.Function(mod, name, ft)
    f32_ft = LLVM.FunctionType(f32, [f32])
    rsqrt_f32     = declare("llvm.nvvm.rsqrt.approx.f",     f32_ft)
    rsqrt_f32_ftz = declare("llvm.nvvm.rsqrt.approx.ftz.f", f32_ft)
    f64_ft = LLVM.FunctionType(f64, [f64])
    rsqrt_f64 = declare("llvm.nvvm.rsqrt.approx.d", f64_ft)

    @dispose builder=IRBuilder() begin
        for (fdiv, sqrt_call, is_f32) in to_replace
            x = operands(sqrt_call)[1]
            position!(builder, fdiv)

            replacement = if is_f32
                f = LLVM.parent(LLVM.parent(fdiv))
                call!(builder, f32_ft, f32_ftz(f) ? rsqrt_f32_ftz : rsqrt_f32, [x])
            else
                call!(builder, f64_ft, rsqrt_f64, [x])
            end

            replace_uses!(fdiv, replacement)
            erase!(fdiv)
            # sqrt may still be used elsewhere; only clean it up if dead now.
            if isempty(uses(sqrt_call))
                erase!(sqrt_call)
            end
            changed = true
        end
    end

    end # @tracepoint
    return changed
end
PTXRSqrtFastPass() = NewPMModulePass("ptx-rsqrt-fast", ptx_rsqrt_fast!)

# Rewrite `afn`-flagged `fdiv`:
# - f32 → `llvm.nvvm.div.approx{,.ftz}.f`.
# - f64 → `rcp.approx.ftz.d` + one Newton step (no native fast f64 fdiv).
function ptx_fdiv_fast!(mod::LLVM.Module)
    changed = false
    @tracepoint "ptx-fdiv-fast" begin

    f32 = LLVM.FloatType()
    f64 = LLVM.DoubleType()

    # collect first to avoid mutation-during-iteration
    to_replace = Tuple{LLVM.FDivInst, Bool}[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        inst isa LLVM.FDivInst || continue
        is_f32 = LLVM.value_type(inst) == f32
        is_f64 = LLVM.value_type(inst) == f64
        (is_f32 || is_f64) || continue
        LLVM.fast_math(inst).afn || continue
        push!(to_replace, (inst, is_f32))
    end
    isempty(to_replace) && return false

    # declare intrinsics by name so LLVM keeps the exact non-overloaded names;
    # LLVM.Intrinsic + type params would mangle to *.f64, unrecognized by NVPTX.
    fns = functions(mod)
    declare(name, ft) = haskey(fns, name) ? fns[name] : LLVM.Function(mod, name, ft)
    f32_ft = LLVM.FunctionType(f32, [f32, f32])
    div_f32     = declare("llvm.nvvm.div.approx.f",     f32_ft)
    div_f32_ftz = declare("llvm.nvvm.div.approx.ftz.f", f32_ft)
    f64_ft1 = LLVM.FunctionType(f64, [f64])
    rcp_f64 = declare("llvm.nvvm.rcp.approx.ftz.d", f64_ft1)
    fma_ft  = LLVM.FunctionType(f64, [f64, f64, f64])
    fma_f64 = declare("llvm.fma.f64", fma_ft)
    one_f64 = ConstantFP(f64, 1.0)

    @dispose builder=IRBuilder() begin
        for (inst, is_f32) in to_replace
            lhs, rhs = operands(inst)[1], operands(inst)[2]
            position!(builder, inst)

            replacement = if is_f32
                f = LLVM.parent(LLVM.parent(inst))
                call!(builder, f32_ft, f32_ftz(f) ? div_f32_ftz : div_f32, [lhs, rhs])
            else
                inv_y   = call!(builder, f64_ft1, rcp_f64, [rhs])
                neg_rhs = fneg!(builder, rhs)
                # Newton refinement, matching CUDA.jl's `FastMath.inv_fast(::Float64)`
                e       = call!(builder, fma_ft, fma_f64, [inv_y, neg_rhs, one_f64])
                e       = call!(builder, fma_ft, fma_f64, [e, e, e])
                inv_ref = call!(builder, fma_ft, fma_f64, [e, inv_y, inv_y])
                fmul!(builder, lhs, inv_ref)
            end

            replace_uses!(inst, replacement)
            erase!(inst)
            changed = true
        end
    end

    end # @tracepoint
    return changed
end
PTXFDivFastPass() = NewPMModulePass("ptx-fdiv-fast", ptx_fdiv_fast!)

# Rewrite `afn`-flagged `llvm.sqrt.f{32,64}`:
# - f32 → `llvm.nvvm.sqrt.approx{,.ftz}.f`.
# - f64 → `rcp.approx.ftz.d(rsqrt.approx.d(x))` (no native fast f64 sqrt).
function ptx_fsqrt_fast!(mod::LLVM.Module)
    changed = false
    @tracepoint "ptx-fsqrt-fast" begin

    f32 = LLVM.FloatType()
    f64 = LLVM.DoubleType()

    to_replace = Tuple{LLVM.CallInst, Bool}[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        callee isa LLVM.Function || continue
        name = LLVM.name(callee)
        is_f32 = name == "llvm.sqrt.f32"
        is_f64 = name == "llvm.sqrt.f64"
        (is_f32 || is_f64) || continue
        LLVM.fast_math(inst).afn || continue
        push!(to_replace, (inst, is_f32))
    end
    isempty(to_replace) && return false

    fns = functions(mod)
    declare(name, ft) = haskey(fns, name) ? fns[name] : LLVM.Function(mod, name, ft)
    f32_ft = LLVM.FunctionType(f32, [f32])
    sqrt_f32     = declare("llvm.nvvm.sqrt.approx.f",     f32_ft)
    sqrt_f32_ftz = declare("llvm.nvvm.sqrt.approx.ftz.f", f32_ft)
    f64_ft = LLVM.FunctionType(f64, [f64])
    rcp_f64   = declare("llvm.nvvm.rcp.approx.ftz.d", f64_ft)
    rsqrt_f64 = declare("llvm.nvvm.rsqrt.approx.d",   f64_ft)

    @dispose builder=IRBuilder() begin
        for (inst, is_f32) in to_replace
            x = operands(inst)[1]
            position!(builder, inst)

            replacement = if is_f32
                f = LLVM.parent(LLVM.parent(inst))
                call!(builder, f32_ft, f32_ftz(f) ? sqrt_f32_ftz : sqrt_f32, [x])
            else
                # No native fast f64 sqrt; emit the same `rcp(rsqrt(x))`
                # sequence NVPTX' `getSqrtEstimate` would have used.
                rsqrt = call!(builder, f64_ft, rsqrt_f64, [x])
                call!(builder, f64_ft, rcp_f64, [rsqrt])
            end

            replace_uses!(inst, replacement)
            erase!(inst)
            changed = true
        end
    end

    end # @tracepoint
    return changed
end
PTXFSqrtFastPass() = NewPMModulePass("ptx-fsqrt-fast", ptx_fsqrt_fast!)
