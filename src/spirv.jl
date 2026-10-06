# implementation of the GPUCompiler interfaces for generating SPIR-V code

# https://github.com/llvm/llvm-project/blob/master/clang/lib/Basic/Targets/SPIR.h
# https://github.com/KhronosGroup/LLVM-SPIRV-Backend/blob/master/llvm/docs/SPIR-V-Backend.rst
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

const SPIRV_LLVM_Backend_jll =
    LazyModule("SPIRV_LLVM_Backend_jll",
               UUID("4376b9bf-cff8-51b6-bb48-39421dff0d0c"))
const SPIRV_LLVM_Translator_jll =
    LazyModule("SPIRV_LLVM_Translator_jll",
               UUID("4a5d46fc-d8cf-5151-a261-86b458210efb"))
const SPIRV_Tools_jll =
    LazyModule("SPIRV_Tools_jll",
               UUID("6ac6d60f-d740-5983-97d7-a4482c0689f4"))


## target

export SPIRVCompilerTarget, SPIRVAtomics

# The atomic operations that the device, its driver and the SPIR-V toolchain run correctly
# (not: that are native in hardware), for which `lower_atomics!` selects SPIR-V instructions.
# Floating-point operations without such an instruction become compare-exchange loops, which
# need integer atomics of the same width.
#
# There is nothing to enable native floating-point min/max
# (SPV_EXT_shader_atomic_float_min_max): those may return either zero when comparing -0.0 and
# +0.0, while LLVM's `atomicrmw fmin` and `fmax` order -0.0 before +0.0, and `atomicrmw` has
# no fast-math flags to relax that. So these are compare-exchange loops too, which rules them
# out for half-precision numbers, and needs `int64` for double-precision ones.
#
# Which synchronization scopes and memory orderings the device supports is not conveyed
# either: front-ends must only use those their device supports (e.g., OpenCL 3.0 makes the
# device scope optional, and the system scope needs fine-grained SVM atomics).
Base.@kwdef struct SPIRVAtomics
    # 64-bit integer atomics (cl_khr_int64_base_atomics and cl_khr_int64_extended_atomics).
    # enabled by default, as GPUCompiler emitted them without checking before.
    int64::Bool = true

    # atomic addition and subtraction of half-, single- and double-precision floating-point
    # numbers in global and local memory (SPV_EXT_shader_atomic_float_add and
    # SPV_EXT_shader_atomic_float16_add, e.g. from cl_ext_float_atomics)
    fadd_f16_global::Bool = false
    fadd_f32_global::Bool = false
    fadd_f64_global::Bool = false
    fadd_f16_local::Bool = false
    fadd_f32_local::Bool = false
    fadd_f64_local::Bool = false
end

Base.@kwdef struct SPIRVCompilerTarget <: AbstractCompilerTarget
    version::Union{Nothing,VersionNumber} = nothing
    # SPIR-V extensions, as the comma-separated specifier string passed verbatim to the
    # translator/back-end via `--spirv-ext`, e.g. "+SPV_EXT_shader_atomic_float_add,+SPV_KHR_expect_assume"
    # (LLVM feature-string style, cf. `GCNCompilerTarget.features`). Kept as a plain
    # `String` -- not a `Vector` -- so `jl_egal`-based owner/config lookups can match
    # structurally equivalent targets after package-image deserialization.
    extensions::String = ""
    # the atomics the device supports; the extensions they need are added to `extensions`
    atomics::SPIRVAtomics = SPIRVAtomics()
    supports_fp16::Bool = true
    supports_fp64::Bool = true
    supports_bfloat16::Bool = false

    backend::Symbol = isavailable(SPIRV_LLVM_Backend_jll) ? :llvm : :khronos

    # the driver that will consume the SPIR-V, used to work around its bugs. `:generic` if
    # unknown, `:intel` for Intel's GPU driver (NEO, with IGC), `:pocl`, `:nvidia`, ...
    driver::Symbol = :generic

    # XXX: these don't really belong in the _target_ struct
    validate::Bool = false
    optimize::Bool = false
end

function llvm_triple(target::SPIRVCompilerTarget)
    if target.backend == :llvm
        architecture = Int===Int64 ? "spirv64" : "spirv32"  # could also be "spirv" for logical addressing
        subarchitecture = target.version === nothing ? "" : "v$(target.version.major).$(target.version.minor)"
        vendor = "unknown"  # could also be AMD
        os = "unknown"
        environment = "unknown"
        return "$architecture$subarchitecture-$vendor-$os-$environment"
    elseif target.backend == :khronos
        return Int===Int64 ? "spir64-unknown-unknown" : "spirv-unknown-unknown"
    end
end

# SPIRV is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::SPIRVCompilerTarget) = nothing

function runtime_cstring_type(job::CompilerJob{SPIRVCompilerTarget})
    LLVM.DataLayout(llvm_datalayout(job.config.target)) do dl
        Core.LLVMPtr{Cchar, dl.globals_addrspace}
    end
end

# OpenCL requires `fma` to be supported, and correctly rounded, so `fma` should use it rather
# than Julia's Float64-based `fma_emulated` fallback (which fails without Float64 support).
have_fma(@nospecialize(target::SPIRVCompilerTarget), T::Type) = true

# the SPIR-V extensions that the atomic instructions the target supports need
function spirv_atomic_extensions(atomics::SPIRVAtomics)
    extensions = String[]
    f16 = atomics.fadd_f16_global || atomics.fadd_f16_local
    if f16 || atomics.fadd_f32_global || atomics.fadd_f32_local ||
       atomics.fadd_f64_global || atomics.fadd_f64_local
        push!(extensions, "SPV_EXT_shader_atomic_float_add")
    end
    # (the half-precision extension extends the first one)
    f16 && push!(extensions, "SPV_EXT_shader_atomic_float16_add")
    return extensions
end

# The `--spirv-ext` specifier to compile with: the target's, with the extensions its atomics
# need enabled. The back-ends only declare the extensions that the module uses.
function spirv_extensions(target::SPIRVCompilerTarget)
    spec = target.extensions
    entries = filter(!isempty, strip.(split(spec, ',')))
    for ext in spirv_atomic_extensions(target.atomics)
        # the last entry that mentions the extension decides
        enabled = nothing
        for entry in entries
            if entry[1] in ('+', '-') && (entry[2:end] == ext || entry[2:end] == "all")
                enabled = entry[1] == '+'
            end
        end
        if enabled === false
            error("The SPIR-V extension $ext, which the atomics of the target need, is disabled by its extensions $(repr(spec))")
        end
        enabled === true && continue
        spec = isempty(spec) ? "+$ext" : "$spec,+$ext"
    end
    return spec
end

llvm_datalayout(::SPIRVCompilerTarget) = Int===Int64 ?
    "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1" :
    "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"


## job

function finish_module!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                        entry::LLVM.Function)
    # update calling convention
    for f in mod.functions
        # JuliaGPU/GPUCompiler.jl#97
        #f.callconv = LLVM.CallConv.SPIRFUNC
    end
    if job.config.kernel
        entry.callconv = LLVM.CallConv.SPIRKERNEL
    end

    # (fail early when the extensions conflict with the atomics)
    spirv_extensions(job.config.target)

    return entry
end

function validate_ir(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module)
    errors = IRError[]

    # support for half and double depends on the target
    if !job.config.target.supports_fp16
        append!(errors, check_ir_values(mod, LLVM.HalfType()))
    end
    if !job.config.target.supports_fp64
        append!(errors, check_ir_values(mod, LLVM.DoubleType()))
    end
    if !job.config.target.supports_bfloat16 && isdefined(LLVM, :BFloatType)
        append!(errors, check_ir_values(mod, LLVM.BFloatType()))
    end

    # atomics that `lower_atomics!` cannot lower
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        reason = if is_atomic_memop(inst)
            action = spirv_atomic_action(job, inst)
            action isa String ? action : nothing
        elseif inst isa LLVM.FenceInst && spirv_syncscope(inst.syncscope.name) === nothing
            "fence with synchronization scope $(repr(inst.syncscope.name))"
        else
            nothing
        end
        reason === nothing || push!(errors, (reason, backtrace(inst), string(inst)))
    end

    return errors
end

function finish_ir!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                    entry::LLVM.Function)
    # SPIR-V has no `trap` and no mechanism to abort a compute kernel (OpKill is fragment-only),
    # so strip the device-side `trap` and lower `unreachable` to a clean `ret`. running this here
    # (post-`optimize!`) is the correct spot: the trap has finished serving as the optimizer
    # guard (see `emit_exception!`), and turning `unreachable` into `ret` also avoids emitting
    # OpUnreachable (UB if reached), which PoCL and friends handle poorly.
    lower_unreachable_control_flow!(job, mod)

    # SPIR-V cannot express atomic loads and stores of pointers, which is what Julia's
    # `unordered` heap-reference accesses are (see `demote_unordered_atomics!`)
    demote_unordered_atomics!(mod)

    # the SPIR-V back-ends lower the floating-point minimum and maximum intrinsics to OpenCL's
    # `fmin`/`fmax`, which ignore NaNs and don't order signed zeros
    lower_minimum_maximum!(mod)

    # IGC drops fields when legalizing aggregates built by nested `insertvalue`s
    if job.config.target.driver === :intel
        flatten_nested_insertvalue!(mod)
    end

    # convert the kernel state argument to a byval reference
    if job.config.kernel
        state = kernel_state_type(job)
        if state !== Nothing
            new_entry = kernel_state_to_reference!(job, mod, entry)

            # only if there was a kernel state parameter to convert, which optimization adds
            # (so not when emitting unoptimized IR)
            if new_entry != entry
                entry = new_entry
                T_state = convert(LLVMType, state)
                push!(entry.parameter_attributes[1], TypeAttribute(:byval, T_state))
            end
        end
    end

    # HACK: Intel's compute runtime doesn't properly support SPIR-V's byval attribute.
    #       they do support struct byval, for OpenCL, so wrap byval parameters in a struct.
    if job.config.kernel
        entry = wrap_byval(job, mod, entry)
    end

    # SPIR-V does not support i128, convert alloca arrays to vector types
    convert_i128_allocas!(mod)

    # add module metadata
    ## OpenCL 2.0
    push!(get!(mod.metadata, "opencl.ocl.version").operands,
          MDNode([ConstantInt(Int32(2)),
                  ConstantInt(Int32(0))]))
    ## SPIR-V 1.5
    push!(get!(mod.metadata, "opencl.spirv.version").operands,
          MDNode([ConstantInt(Int32(1)),
                  ConstantInt(Int32(5))]))

    return entry
end

# mirrors `SPIRVCompileOptions` from libspirv.h
struct SPIRVCompileOptions
    is_64bit::Cint
    version_major::Cuint
    version_minor::Cuint
    extensions::Cstring
    opt_level::Cint
end

# mirrors `LLVMSPIRVTranslateOptions` from libllvm_spirv.h
struct LLVMSPIRVTranslateOptions
    max_version_major::Cuint
    max_version_minor::Cuint
    extensions::Cstring
    debug_info_version::Cint    # LLVMSPIRVDebugInfoVersion
end
const LLVMSPIRVDebugInfoOpenCL100 = Cint(1)

# translate bitcode to SPIR-V through libllvm_spirv. unlike the back-ends' `Compile`, the
# translator's entry point takes no diagnostic handler, so this doesn't use
# `external_compile`.
function translate(input::Vector{UInt8}, options::Ref{LLVMSPIRVTranslateOptions})
    backend = ExternalBackend(SPIRV_LLVM_Translator_jll.libllvm_spirv, "LLVMSPIRV")
    buffer = Ref{Ptr{Cvoid}}(C_NULL)
    message = Ref{Cstring}(C_NULL)
    status = @ccall $(api(backend, "Translate"))(input::Ptr{UInt8}, length(input)::Csize_t,
                                                 options::Ptr{Cvoid},
                                                 buffer::Ptr{Ptr{Cvoid}},
                                                 message::Ptr{Cstring})::Cint
    external_result(backend, status, "Failed to translate LLVM code to SPIR-V",
                    message[], String[], input)
    return take_buffer(backend, buffer[])
end

@unlocked function mcgen(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.CodeGenFileType.Assembly)
    target = job.config.target

    # The SPIRV Tools don't handle Julia's debug info, rejecting DW_LANG_Julia...
    strip_debuginfo!(mod)

    # the LLVM to SPIR-V translator does not support the freeze instruction
    # (SPIRV-LLVM-Translator#1140)
    rm_freeze!(job, mod)

    # translate to SPIR-V
    input = bitcode(mod)
    dump_input() = let path = tempname(cleanup=false) * ".bc"
        write(path, input)
        path
    end
    spirv = if target.backend === :llvm
        # compile in-process through libspirv. the back-end derives the triple from the
        # options; unlike `llc`, it rejects unknown extensions instead of ignoring them.
        backend = ExternalBackend(SPIRV_LLVM_Backend_jll.libspirv, "SPIRV")
        version = something(target.version, v"0.0")
        extensions = spirv_extensions(target)
        GC.@preserve extensions begin
            options = Ref(SPIRVCompileOptions(Int === Int64, version.major, version.minor,
                                              Base.unsafe_convert(Cstring, extensions),
                                              #=opt_level=# 2))
            external_compile(backend, input, options,
                             "Failed to compile to SPIR-V with the SPIR-V back-end")
        end
    elseif target.backend === :khronos
        # translate in-process through libllvm_spirv. like `llvm-spirv`, the translator
        # takes the triple from the module and rejects unknown extensions.
        version = something(target.version, v"0.0")
        extensions = spirv_extensions(target)
        GC.@preserve extensions begin
            options = Ref(LLVMSPIRVTranslateOptions(version.major, version.minor,
                                                    Base.unsafe_convert(Cstring, extensions),
                                                    LLVMSPIRVDebugInfoOpenCL100))
            translate(input, options)
        end
    else
        error("Unsupported SPIR-V back-end $(repr(target.backend)); expected :llvm or :khronos.")
    end

    # the SPIR-V tools work on files
    translated = tempname(cleanup=false) * ".spv"
    write(translated, spirv)

    # validate
    if target.validate
        try
            run(`$(SPIRV_Tools_jll.spirv_val()) $translated`)
        catch e
            error("""Failed to validate generated SPIR-V.
                     If you think this is a bug, please file an issue and attach $(dump_input()) and $(translated).""")
        end
    end

    # optimize
    if target.optimize
        optimized = tempname(cleanup=false) * ".spv"
        try
            run(```$(SPIRV_Tools_jll.spirv_opt()) -O --skip-validation
                                                  $translated -o $optimized```)
        catch
            error("""Failed to optimize generated SPIR-V.
                     If you think this is a bug, please file an issue and attach $(dump_input()) and $(translated).""")
        end
        spirv = read(optimized)
        rm(optimized)
    end

    output = if format == LLVM.CodeGenFileType.Object
        spirv
    else
        # disassemble
        write(translated, spirv)
        read(`$(SPIRV_Tools_jll.spirv_dis()) $translated`, String)
    end
    rm(translated)

    return output
end

source_code(target::SPIRVCompilerTarget) = "spirv"

# reimplementation that uses `spirv-dis`, giving much more pleasant output
function code_native(io::IO, job::CompilerJob{SPIRVCompilerTarget}; raw::Bool=false, dump_module::Bool=false)
    config = CompilerConfig(job.config; strip=!raw, only_entry=!dump_module, validate=false)
    obj = JuliaContext() do ctx
        obj, meta = compile(:obj, CompilerJob(job; config))
        dispose(meta.ir)
        obj
    end
    mktemp() do input_path, input_io
        write(input_io, obj)
        flush(input_io)

        disassembler = SPIRV_Tools_jll.spirv_dis()
        mktemp() do output_path, output_io
            run(`$disassembler $input_path -o $output_path`)
            asm = read(output_io, String)
            highlight(io, asm, source_code(job.config.target))
        end
    end
end


## atomics

# SPIR-V Scope operands
const SPIRV_SCOPE_CROSS_DEVICE = 0
const SPIRV_SCOPE_DEVICE = 1
const SPIRV_SCOPE_WORKGROUP = 2
const SPIRV_SCOPE_SUBGROUP = 3
const SPIRV_SCOPE_INVOCATION = 4

# MemorySemantics bits naming the memory an operation orders
const SPIRV_SUBGROUP_MEMORY = 0x80
const SPIRV_WORKGROUP_MEMORY = 0x100
const SPIRV_CROSS_WORKGROUP_MEMORY = 0x200
const SPIRV_IMAGE_MEMORY = 0x800

# The SPIR-V Scope of a synchronization scope, spelled like for `llvm_syncscope`, and the
# MemorySemantics bits for the memory it names (see `split_syncscope`; `nothing` for all
# memory). The system scope, LLVM's default, is the cross-device one. Returns `nothing` for
# other scopes, and for imageblock memory, which SPIR-V doesn't have, which `validate_ir`
# rejects rather than guessing what they mean.
function spirv_syncscope(name::String)
    parts = split_syncscope(name)
    parts === nothing && return nothing
    base, memory = parts
    scope = if base == "singlethread"
        SPIRV_SCOPE_INVOCATION
    elseif base == "subgroup"
        SPIRV_SCOPE_SUBGROUP
    elseif base == "workgroup"
        SPIRV_SCOPE_WORKGROUP
    elseif base == "device"
        SPIRV_SCOPE_DEVICE
    elseif base == "system" || base == ""
        SPIRV_SCOPE_CROSS_DEVICE
    else
        return nothing
    end
    if memory !== nothing
        memory & 0b1000 == 0 || return nothing
        memory = (memory & 0b001 == 0 ? 0 : SPIRV_CROSS_WORKGROUP_MEMORY) |
                 (memory & 0b010 == 0 ? 0 : SPIRV_WORKGROUP_MEMORY) |
                 (memory & 0b100 == 0 ? 0 : SPIRV_IMAGE_MEMORY)
    end
    return (; scope, memory)
end

# read-modify-write operations SPIR-V has instructions for, and the names of their builtins.
# floating-point addition needs an extension the device may not support (see `SPIRVAtomics`),
# and subtraction is the addition of the negated value.
const SPIRV_ATOMICRMW_OPS = let Op = LLVM.AtomicRMWBinOp
    Dict(Op.Xchg => "Exchange", Op.Add => "IAdd", Op.Sub => "ISub", Op.And => "And",
         Op.Or => "Or", Op.Xor => "Xor", Op.Max => "SMax", Op.Min => "SMin",
         Op.UMax => "UMax", Op.UMin => "UMin", Op.FAdd => "FAddEXT", Op.FSub => "FAddEXT")
end

# read-modify-write operations we expand to compare-exchange loops, which `expand_to_cmpxchg!`
# computes like LLVM does (operations the LLVM in use doesn't support can't occur in the IR).
# that includes floating-point min/max, see `SPIRVAtomics`.
const SPIRV_EXPANDABLE_ATOMICRMW_OPS = let Op = LLVM.AtomicRMWBinOp
    (Op.Nand, Op.FMax, Op.FMin, Op.UIncWrap, Op.UDecWrap, Op.USubCond, Op.USubSat,
     Op.FMaximum, Op.FMinimum, Op.FMaximumNum, Op.FMinimumNum)
end

# can the target atomically add floating-point numbers of type `T` in address space `as`?
# (generic pointers can point to global and local memory)
function spirv_fadd_supported(atomics::SPIRVAtomics, T::LLVMType, as::Integer)
    global_, local_ = if T isa LLVM.HalfType
        atomics.fadd_f16_global, atomics.fadd_f16_local
    elseif T isa LLVM.FloatType
        atomics.fadd_f32_global, atomics.fadd_f32_local
    elseif T isa LLVM.DoubleType
        atomics.fadd_f64_global, atomics.fadd_f64_local
    else
        false, false
    end
    return as == 1 ? global_ : as == 3 ? local_ : global_ && local_
end

# How to lower `inst`, an atomic memory operation, for the job's target. Like the rule tables
# of LLVM's legalizers, the rules are tried in order and the first that applies decides. Returns
# the action, or the reason why the operation cannot be lowered (which `validate_ir` reports):
#
# - `:demote`: an atomic on the thread's own memory, which becomes plain accesses;
# - `:cast`: a floating-point or pointer load, store or exchange, or a pointer
#   compare-exchange, which becomes an integer one (like `AtomicExpand` does for most
#   targets), so that only the integer forms of these instructions are used;
# - `:cmpxchg_loop`: a read-modify-write operation without a SPIR-V instruction the target
#   supports, which becomes a compare-exchange loop (`AtomicExpand`'s `insertRMWCmpXchgLoop`);
# - `:select`: an operation SPIR-V can express, which becomes a `__spirv_Atomic*` call.
#
# Casts and loops need integer atomics of the same size, which rules out half-precision
# numbers other than for a native addition, and needs 64-bit integer atomics for 64-bit values.
function spirv_atomic_action(@nospecialize(job::CompilerJob{SPIRVCompilerTarget}),
                             inst::LLVM.Instruction)
    atomics = job.config.target.atomics
    op = inst isa LLVM.AtomicRMWInst ? inst.binop : nothing
    if op !== nothing && !haskey(SPIRV_ATOMICRMW_OPS, op) &&
       !(op in SPIRV_EXPANDABLE_ATOMICRMW_OPS)
        return "atomicrmw $(LLVM.irname(op)) operation"
    end

    is_thread_private(inst.pointer_operand) && return :demote

    # (SPIR-V only has volatile atomics with the Vulkan memory model)
    inst.volatile && return "volatile atomic operation"

    as = inst.pointer_operand.value_type.addrspace
    if !(as in (1, 3, 4))
        return "atomic operation in address space $as (SPIR-V only supports atomics on global, local and generic memory)"
    end

    if spirv_syncscope(inst.syncscope.name) === nothing
        return "atomic operation with synchronization scope $(repr(inst.syncscope.name))"
    end

    T = atomic_value_type(inst)
    bits = atomic_bits(inst)
    if bits in (8, 16) && !(T isa LLVM.HalfType)
        return "$bits-bit atomic operation (SPIR-V only supports 32- and 64-bit atomics)"
    elseif bits === nothing || !(bits in (16, 32, 64))
        return "atomic operation on a $(string(T)) value"
    end

    if inst.alignment < bits ÷ 8
        return "misaligned atomic operation"
    end

    Op = LLVM.AtomicRMWBinOp
    action = if inst isa LLVM.AtomicCmpXchgInst
        T isa LLVM.PointerType ? :cast : :select
    elseif op === nothing || op == Op.Xchg
        T isa LLVM.IntegerType ? :select : :cast
    elseif op == Op.FAdd || op == Op.FSub
        spirv_fadd_supported(atomics, T, as) ? :select : :cmpxchg_loop
    elseif haskey(SPIRV_ATOMICRMW_OPS, op)
        :select
    else
        :cmpxchg_loop
    end

    if action !== :select || !(T isa LLVM.FloatingPointType)
        if bits == 16
            return "half-precision atomic operation (SPIR-V only supports half-precision atomic addition, if the target does)"
        elseif bits == 64 && !atomics.int64
            return "64-bit atomic operation (the target does not support 64-bit integer atomics)"
        end
    end

    return action
end


## LLVM passes

# remove freeze and replace uses by the original value
# (KhronosGroup/SPIRV-LLVM-Translator#1140)
function rm_freeze!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false
    @tracepoint "remove freeze" begin

    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        if inst isa LLVM.FreezeInst
            orig = first(inst.operands)
            replace_uses!(inst, orig)
            @compiler_assert isempty(inst.uses) job
            erase!(inst)
            changed = true
        end
    end

    end
    return changed
end

# flatten `insertvalue`s with multiple indices into single-index ones, extracting and
# re-inserting the intermediate aggregates: `insertvalue %agg, %val, 1, 0` becomes
#   %sub = extractvalue %agg, 1
#   %new = insertvalue %sub, %val, 0
#          insertvalue %agg, %new, 1
#
# this works around a bug in Intel's graphics compiler, whose `TypesLegalizationPass` splits
# aggregate stores (and phis, which it lowers to stores) into per-field stores by looking up
# each field in the `insertvalue` chain. that lookup gives up on an `insertvalue` with more
# indices than the field it is looking for, silently dropping the store of that field.
# e.g., storing `(flag::Bool, (a, b))` built by inserting `flag`, `a` and `b`, loses `flag`
# (intel/intel-graphics-compiler#378, JuliaGPU/OpenCL.jl#502, JuliaGPU/oneAPI.jl#259).
# this needs to run after optimization, as InstCombine folds these sequences back together.
function flatten_nested_insertvalue!(mod::LLVM.Module)
    changed = false
    @tracepoint "flatten nested insertvalue" begin

    for f in mod.functions, bb in f.blocks
        worklist = filter(collect(bb.instructions)) do inst
            inst isa LLVM.InsertValueInst && length(inst.indices) > 1
        end
        isempty(worklist) && continue

        @dispose builder=IRBuilder() begin
            for inst in worklist
                agg, val = inst.operands
                indices = collect(inst.indices)

                position!(builder, LLVM.before(inst))
                new = flatten_insertvalue!(builder, agg, val, indices)
                replace_uses!(inst, new)
                erase!(inst)
                changed = true
            end
        end
    end

    end
    return changed
end

function flatten_insertvalue!(builder::IRBuilder, agg::LLVM.Value, val::LLVM.Value,
                              indices::AbstractVector)
    idx = first(indices)
    if length(indices) > 1
        sub = extract_value!(builder, agg, idx)
        val = flatten_insertvalue!(builder, sub, val, @view indices[2:end])
    end
    return insert_value!(builder, agg, val, idx)
end

# expand the floating-point minimum and maximum intrinsics, which both SPIR-V back-ends
# translate to OpenCL's `fmin` and `fmax`. those return the other operand when one is a
# (quiet) NaN, like `llvm.minnum` and `llvm.maxnum`, but may return either zero when comparing
# -0.0 and +0.0, so lower every family to `llvm.minnum` and `llvm.maxnum` and fix up what
# differs, unless the call's fast-math flags say it doesn't occur:
#
# - `llvm.minimum`/`llvm.maximum` (Julia's `min` and `max` of floating-point numbers) return
#   NaN when either operand is NaN, and order -0.0 before +0.0;
# - `llvm.minnum`/`llvm.maxnum` order -0.0 before +0.0 (as LLVM specifies them since 22);
# - `llvm.minimumnum`/`llvm.maximumnum` return the other operand for any NaN, including a
#   signaling one (for which `fmin` may return NaN), and order -0.0 before +0.0.
#
# the `llvm.minnum`/`llvm.maxnum` calls this introduces are marked `nsz` when the signs of
# zeros are fixed up separately, so running this again doesn't change them.
function lower_minimum_maximum!(mod::LLVM.Module)
    changed = false
    @tracepoint "lower minimum/maximum" begin

    calls = Tuple{LLVM.CallInst,Bool,Symbol}[]
    for f in mod.functions
        isdeclaration(f) || continue
        m = match(r"^llvm\.(min|max)(imum|num|imumnum)\.", f.name)
        m === nothing && continue
        is_min = m[1] == "min"
        nans = m[2] == "imum" ? :propagate : m[2] == "num" ? :quiet : :ignore
        for use in f.uses
            call = use.user
            call isa LLVM.CallInst && push!(calls, (call, is_min, nans))
        end
    end

    for (call, is_min, nans) in calls
        typ = call.value_type
        eltyp = typ isa LLVM.VectorType ? typ.element_type : typ
        bits = if eltyp isa LLVM.HalfType
            16
        elseif eltyp isa LLVM.FloatType
            32
        elseif eltyp isa LLVM.DoubleType
            64
        else
            continue
        end
        ityp = LLVM.IntType(bits)
        if typ isa LLVM.VectorType
            ityp = LLVM.VectorType(ityp, typ.length)
        end

        x, y = call.arguments
        flags = NamedTuple(call.fast_math)
        fix_zeros = !flags.nsz
        fix_nans = nans !== :quiet && !flags.nnan
        nans === :quiet && !fix_zeros && continue

        num = LLVM.Function(mod, LLVM.Intrinsic(is_min ? "llvm.minnum" : "llvm.maxnum"),
                            LLVMType[typ])
        @dispose builder=IRBuilder() begin
            position!(builder, LLVM.before(call))
            builder.debug_location = call.debug_location

            res = call!(builder, num.function_type, num, LLVM.Value[x, y])
            res.fast_math = flags
            fix_zeros && (res.fast_math.nsz = true)

            # if both operands are zero, combine their sign bits
            if fix_zeros
                zero = LLVM.null(typ)
                both_zero = and!(builder, fcmp!(builder, LLVM.RealPredicate.OEQ, x, zero),
                                          fcmp!(builder, LLVM.RealPredicate.OEQ, y, zero))
                xi = bitcast!(builder, x, ityp)
                yi = bitcast!(builder, y, ityp)
                zi = is_min ? or!(builder, xi, yi) : and!(builder, xi, yi)
                res = select!(builder, both_zero, bitcast!(builder, zi, typ), res)
            end

            if fix_nans && nans === :propagate
                # if either operand is NaN, return a NaN
                either_nan = fcmp!(builder, LLVM.RealPredicate.UNO, x, y)
                res = select!(builder, either_nan, fadd!(builder, x, y), res)
            elseif fix_nans
                # if one operand is NaN, return the other one
                res = select!(builder, fcmp!(builder, LLVM.RealPredicate.UNO, y, y), x, res)
                res = select!(builder, fcmp!(builder, LLVM.RealPredicate.UNO, x, x), y, res)
            end

            replace_uses!(call, res)
            erase!(call)
        end
        changed = true
    end

    for f in collect(mod.functions)
        isdeclaration(f) && isempty(f.uses) || continue
        occursin(r"^llvm\.(min|max)imum(num)?\.", f.name) && erase!(f)
    end

    end
    return changed
end

# convert alloca [N x i128] to alloca [N x <2 x i64>]
# SPIR-V doesn't support i128 types, but we can represent them as vectors
function convert_i128_allocas!(mod::LLVM.Module)
    changed = false
    @tracepoint "convert i128 allocas" begin

    for f in mod.functions, bb in f.blocks
        for inst in bb.instructions
            if inst isa LLVM.AllocaInst
                alloca_type = inst.allocated_type

                # Check if this is an i128 or an array of i128
                if alloca_type isa LLVM.ArrayType
                    T = alloca_type.element_type
                else
                    T = alloca_type
                end
                if T isa LLVM.IntegerType && T.width == 128
                    # replace i128 with <2 x i64>
                    vec_type = LLVM.VectorType(LLVM.Int64Type(), 2)

                    if alloca_type isa LLVM.ArrayType
                        array_size = alloca_type.length
                        new_alloca_type = LLVM.ArrayType(vec_type, array_size)
                    else
                        new_alloca_type = vec_type
                    end
                    align_val = inst.alignment

                    # Create new alloca with vector type
                    @dispose builder=IRBuilder() begin
                        position!(builder, LLVM.before(inst))
                        new_alloca = alloca!(builder, new_alloca_type; align=align_val)

                        # Bitcast the new alloca back to the original pointer type
                        # XXX: The issue only seems to manifest itself on LLVM >= 18
                        #      where we use opaque pointers anyways, so not sure this
                        #      is needed
                        old_ptr_type = inst.value_type
                        bitcast_ptr = bitcast!(builder, new_alloca, old_ptr_type)

                        replace_uses!(inst, bitcast_ptr)
                        erase!(inst)
                        changed = true
                    end
                end
            end
        end
    end

    end
    return changed
end

# wrap byval pointers in a single-value struct
function wrap_byval(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ft = f.function_type::LLVM.FunctionType

    # find the byval parameters
    byval = BitVector(undef, length(ft.parameters))
    types = Vector{LLVMType}(undef, length(ft.parameters))
    for i in 1:length(byval)
        attr = get(f.parameter_attributes[i], :byval, nothing)
        byval[i] = attr !== nothing
        byval[i] && (types[i] = attr.value)
    end

    # generate the wrapper function type & definition: byval params become pointers to a struct
    # wrapping the value, and the body GEPs into that struct to recover the original pointer.
    wrapper(i) = LLVM.StructType([convert(LLVMType, types[i])])
    new_types = Union{Nothing,LLVM.LLVMType}[
        byval[i] ? LLVM.PointerType(wrapper(i), ft.parameters[i].addrspace) : nothing
        for i in 1:length(ft.parameters)]
    new_f = clone_with_converted_args!(mod, f, new_types,
        (builder, param, i) -> struct_gep!(builder, wrapper(i), param, 0))

    # apply byval attributes again (`clone_into!` didn't due to the type mismatch)
    for i in 1:length(byval)
        byval[i] && push!(new_f.parameter_attributes[i], TypeAttribute(:byval, wrapper(i)))
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    replace_function!(f, new_f)

    # XXX: work around KhronosGroup/SPIRV-LLVM-Translator#3389
    if job.config.target.backend === :khronos
        @dispose pb=PassBuilder() begin
            add!(pb, SimplifyCFGPass())
            with_llvm_machine(job.config.target) do tm
                run!(pb, new_f, tm)
            end
        end
    end
    return new_f
end
