# implementation of the GPUCompiler interfaces for generating GCN code

const AMDGPU_LLVM_Backend_jll =
    LazyModule("AMDGPU_LLVM_Backend_jll",
               UUID("cc5c0156-bd05-5a77-8a68-bb0aafb29019"))

## target

export GCNCompilerTarget

Base.@kwdef struct GCNCompilerTarget <: AbstractCompilerTarget
    dev_isa::String
    features::String=""

    backend::Symbol = isavailable(AMDGPU_LLVM_Backend_jll) ? :external : :inprocess

    # optional launch bounds (scalar or per-dimension; flattened to their product)
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
end

function Base.hash(target::GCNCompilerTarget, h::UInt)
    h = hash(target.dev_isa, h)
    h = hash(target.features, h)
    h = hash(target.backend, h)
    h = hash(target.minthreads, h)
    h = hash(target.maxthreads, h)
    h
end
GCNCompilerTarget(dev_isa; kwargs...) = GCNCompilerTarget(; dev_isa, kwargs...)

llvm_triple(::GCNCompilerTarget) = "amdgcn-amd-amdhsa"

source_code(target::GCNCompilerTarget) = "gcn"

function llvm_machine(target::GCNCompilerTarget)
    @static if :AMDGPU ∉ LLVM.backends()
        return nothing
    end
    triple = llvm_triple(target)
    t = LLVM.Target(triple=triple)

    cpu = target.dev_isa
    feat = target.features
    reloc = LLVM.RelocMode.PIC
    tm = LLVM.TargetMachine(t, triple; cpu, features=feat, reloc)
    LLVM.asm_verbosity!(tm, true)

    return tm
end


# AMDGPU calls the device scope `agent`, and the subgroup one `wavefront`
llvm_syncscope(::GCNCompilerTarget, name::String) =
    name == "device" ? "agent" : name == "subgroup" ? "wavefront" : name

## job

function isintrinsic(@nospecialize(job::CompilerJob{GCNCompilerTarget}), fn::String)
    startswith(fn, "llvm.amdgcn.") && return true
    # The ROCm device libraries use `llvm.frexp` and `llvm.ldexp`, which were only added in
    # LLVM 17, so they are not recognized as intrinsics by older versions of LLVM.
    # Final code generation knows how to lower them, so accept them here.
    return startswith(fn, "llvm.frexp.") || startswith(fn, "llvm.ldexp.")
end

pass_by_ref(@nospecialize(job::CompilerJob{GCNCompilerTarget})) = true

# AMD GPUs have fused multiply-add, so `fma` should use the hardware instruction rather than
# Julia's Float64-based `fma_emulated` fallback.
have_fma(@nospecialize(target::GCNCompilerTarget), T::Type) = true

function finish_module!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    lower_throw_extra!(job, mod)

    if job.config.kernel
        # calling convention
        entry.callconv = LLVM.CallConv.AMDGPUKERNEL

        # workgroup size bounds; the backend sizes its register budget for the
        # worst case (1,1024) when unset
        if job.config.target.minthreads !== nothing ||
           job.config.target.maxthreads !== nothing
            lo = prod(something(job.config.target.minthreads, 1))
            hi = prod(something(job.config.target.maxthreads, 1024))
            push!(entry.function_attributes,
                  StringAttribute("amdgpu-flat-work-group-size", "$lo,$hi"))
        end
    end

    return entry
end

function finish_ir!(
        @nospecialize(job::CompilerJob{GCNCompilerTarget}), mod::LLVM.Module,
        entry::LLVM.Function
    )
    if job.config.kernel
        entry = add_kernarg_address_spaces!(job, mod, entry)

        # optimize after address space rewriting: propagate addrspace(4) through
        # the addrspacecast chains, then clean up newly-exposed opportunities
        with_llvm_machine(job.config.target) do tm
            @dispose pb=PassBuilder() begin
                add!(pb, FunctionPassManager()) do fpm
                    add!(fpm, InferAddressSpacesPass())
                    add!(fpm, SROAPass())
                    add!(fpm, instcombine_pass(job))
                    add!(fpm, EarlyCSEPass())
                    add!(fpm, SimplifyCFGPass())
                end
                run!(pb, mod, tm)
            end
        end
    end

    # after optimization, which can change address spaces, and also without validation
    expand_atomics!(job, mod)

    return entry
end

# Rewrite byref kernel parameters from flat (addrspace 0) to constant (addrspace 4).
#
# On AMDGPU, kernel arguments reside in the constant address space (addrspace 4),
# which is scalar-loadable via s_load. Julia initially emits byref parameters as
# pointers in addrspace(11) (tracked/derived), but RemoveJuliaAddrspacesPass strips
# all non-integral address spaces to flat (addrspace 0) during optimization. This pass
# restores addrspace(4) on byref parameters so that the backend can emit s_load
# instead of flat_load for struct field accesses.
#
# NOTE: must run after optimization, where RemoveJuliaAddrspacesPass has already
# converted Julia's addrspace(11) to flat (addrspace 0) on these parameters.
function add_kernarg_address_spaces!(
        @nospecialize(job::CompilerJob), mod::LLVM.Module,
        f::LLVM.Function
    )
    ft = f.function_type

    # find the byref parameters by checking for the byref attribute directly,
    # rather than re-classifying arguments (which can fail on typed-pointer LLVM
    # due to element type mismatches in classify_arguments assertions).
    byref_mask = BitVector(undef, length(ft.parameters))
    for i in 1:length(ft.parameters)
        byref_mask[i] = haskey(f.parameter_attributes[i], :byref)
    end

    # check if any flat pointer byref params need rewriting
    needs_rewrite = false
    for (i, param) in enumerate(ft.parameters)
        if byref_mask[i] && param isa LLVM.PointerType && param.addrspace == 0
            needs_rewrite = true
            break
        end
    end
    needs_rewrite || return f

    # generate the new function type with constant address space on byref flat-pointer params
    param_types = ft.parameters
    flat_byref(i) = byref_mask[i] && param_types[i] isa LLVM.PointerType && param_types[i].addrspace == 0
    new_types = Union{Nothing,LLVMType}[
        flat_byref(i) ? (supports_typed_pointers(context()) ?
                            LLVM.PointerType(param_types[i].element_type, #=constant=# 4) :
                            LLVM.PointerType(#=constant=# 4)) :
                        nothing
        for i in 1:length(param_types)]

    # insert addrspacecasts from kernarg (4) back to flat (0) so that the cloned IR (which expects
    # flat pointers) continues to work; the AMDGPU backend's AMDGPULowerKernelArguments traces these
    # casts and produces s_load.
    new_f = clone_with_converted_args!(mod, f, new_types,
        (builder, param, i) -> addrspacecast!(builder, param, param_types[i]))

    # copy parameter attributes AFTER clone_into!, because CloneFunctionInto overwrites all
    # attributes via setAttributes. For byref params, the VMap maps old args to addrspacecast
    # instructions (not Arguments), so LLVM's attribute remapping silently drops them.
    for i in 1:length(param_types)
        append!(new_f.parameter_attributes[i], f.parameter_attributes[i])
    end

    replace_function!(f, new_f)

    # clean up the extra conversion block
    @dispose pb=PassBuilder() begin
        add!(pb, FunctionPassManager()) do fpm
            add!(fpm, SimplifyCFGPass())
        end
        run!(pb, mod)
    end

    return new_f
end

# mirrors `AMDGPUCompileOptions` and `AMDGPUFileType` from libamdgpu.h
struct AMDGPUCompileOptions
    cpu::Cstring
    features::Cstring
    opt_level::Cint
    filetype::Cint
end
const AMDGPUAssemblyFile = Cint(0)
const AMDGPUObjectFile = Cint(1)

@unlocked function mcgen(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                         mod::LLVM.Module, format=LLVM.CodeGenFileType.Assembly)
    target = job.config.target

    if target.backend === :inprocess
        if :AMDGPU ∉ LLVM.backends()
            error("The in-process LLVM lacks the AMDGPU target; cannot compile to GCN. " *
                  "Load AMDGPU_LLVM_Backend_jll and use `backend=:external` instead.")
        end
        return invoke(mcgen, Tuple{CompilerJob, LLVM.Module, typeof(format)},
                      job, mod, format)
    elseif target.backend !== :external
        error("Unsupported GCN back-end $(repr(target.backend)); " *
              "expected :external or :inprocess.")
    end

    if !isavailable(AMDGPU_LLVM_Backend_jll) || !AMDGPU_LLVM_Backend_jll.is_available()
        error("The :external GCN back-end requires AMDGPU_LLVM_Backend_jll, which " *
              "should be installed and loaded first.")
    end
    backend = ExternalBackend(AMDGPU_LLVM_Backend_jll.libamdgpu, "AMDGPU")

    filetype = if format == LLVM.CodeGenFileType.Assembly
        AMDGPUAssemblyFile
    elseif format == LLVM.CodeGenFileType.Object
        AMDGPUObjectFile
    else
        error("Unsupported GCN output format $format")
    end

    # the back-end targets amdgcn-amd-amdhsa with the PIC relocation model; unlike `llc`,
    # it rejects unknown processors and features instead of silently falling back.
    cpu = target.dev_isa
    features = target.features
    code = GC.@preserve cpu features begin
        options = Ref(AMDGPUCompileOptions(Base.unsafe_convert(Cstring, cpu),
                                           Base.unsafe_convert(Cstring, features),
                                           #=opt_level=# 2, filetype))
        external_compile(backend, bitcode(mod), options,
                         "Failed to compile to GCN with the AMDGPU back-end";
                         warn=msg->@safe_warn(msg))
    end
    return String(code)
end

# the version of the LLVM that generates code for the target, which can differ from the one
# GPUCompiler runs in. only the release matters, so JLL rebuilds are ignored.
function backend_llvm_version(target::GCNCompilerTarget)
    target.backend === :external || return LLVM.version()
    jll = get(Base.loaded_modules, getfield(AMDGPU_LLVM_Backend_jll, :pkg), nothing)
    # (without the back-end, `mcgen` reports a clearer error)
    jll === nothing && return LLVM.version()
    version = pkgversion(jll)
    return VersionNumber(version.major, version.minor, version.patch)
end


## validation

# Atomic operations and fences come from many front-ends (AMDGPU.jl's atomic functions,
# UnsafeAtomics and Atomix, Enzyme, Julia's atomic intrinsics), so they are validated here,
# on the IR, rather than in any of them. The AMDGPU back-end does not reject everything the
# target cannot run: Julia's LLVM emits calls to libatomic for oversized or misaligned
# atomics, which only fail when loading the code object, some atomics abort instruction
# selection, and the errors it does report don't point at the Julia code.
# Validation runs after `lower_syncscopes!`, so the scopes are the ones AMDGPU knows.
function validate_ir(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module)
    errors = IRError[]
    dl = mod.datalayout
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        reason = if is_atomic_memop(inst)
            gcn_atomic_error(job, dl, inst)
        elseif inst isa LLVM.FenceInst
            gcn_syncscope_error(job.config.target, inst, "fence")
        else
            nothing
        end
        reason === nothing || push!(errors, (reason, backtrace(inst), string(inst)))
    end
    return errors
end

# the synchronization scopes of AMDGPU, as `lower_syncscopes!` leaves them, and their
# variants that only order the address space of the operation (`one-as`)
const GCN_SYNCSCOPES = ("singlethread", "wavefront", "workgroup", "agent", "system",
                        "singlethread-one-as", "wavefront-one-as", "workgroup-one-as",
                        "agent-one-as", "one-as")

function gcn_syncscope_error(target::GCNCompilerTarget, inst::LLVM.Instruction,
                             what="atomic operation")
    name = inst.syncscope.name
    name in GCN_SYNCSCOPES && return nothing
    # (a cluster is the agent on targets without workgroup clusters)
    name in ("cluster", "cluster-one-as") && backend_llvm_version(target) >= v"22" &&
        return nothing
    return "$what with synchronization scope $(repr(name))"
end

# Why the target cannot run the atomic memory operation `inst`, or `nothing` if it can.
# Operations without an instruction (e.g. 8- and 16-bit ones, or ones on private memory)
# are fine: the back-end expands them.
function gcn_atomic_error(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                          dl::LLVM.DataLayout, inst::LLVM.Instruction)
    reason = gcn_syncscope_error(job.config.target, inst)
    reason === nothing || return reason

    # (constant memory isn't writable, and the region (GDS) and buffer resource address
    # spaces abort instruction selection)
    as = inst.pointer_operand.value_type.addrspace
    if !(as in (0, 1, 3, 5, 7))
        return "atomic operation in address space $as (GCN only supports atomics on flat, global, local, private and buffer memory)"
    end

    # (this includes packed floating-point values, which some targets support natively)
    bits = Int(LLVM.bit_size(dl, atomic_value_type(inst)))
    if bits > 64
        return "$bits-bit atomic operation (GCN supports atomics of at most 64 bits)"
    end
    if inst.alignment < bits ÷ 8
        return "atomic operation with alignment $(inst.alignment) (requires at least $(bits ÷ 8)-byte alignment)"
    end

    return nothing
end


## LLVM passes

# Expand atomic read-modify-writes that the back-end cannot compile to compare-exchange
# loops. These work around bugs in the back-end, and should be revisited when updating it.
function expand_atomics!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                         mod::LLVM.Module)
    target = job.config.target
    version = backend_llvm_version(target)
    # (expansion changes the control flow, so collect the instructions first)
    insts = LLVM.AtomicRMWInst[]
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        if inst isa LLVM.AtomicRMWInst && needs_cmpxchg_expansion(target, version, inst)
            push!(insts, inst)
        end
    end
    foreach(expand_to_cmpxchg!, insts)
    return !isempty(insts)
end

# the operations AMDGPUAtomicOptimizer combines across a wavefront
const OPTIMIZED_ATOMICRMW_OPS = let Op = LLVM.AtomicRMWBinOp
    (Op.Add, Op.Sub, Op.And, Op.Or, Op.Xor, Op.Min, Op.Max, Op.UMin, Op.UMax)
end

function needs_cmpxchg_expansion(target::GCNCompilerTarget, version::VersionNumber,
                                 inst::LLVM.AtomicRMWInst)
    as = inst.pointer_operand.value_type.addrspace
    T = inst.value_operand.value_type

    # LLVM 22 selects 32-bit `usub_sat` on gfx10.3 and later, but lacks the pattern for
    # flat memory, and for local memory before gfx12, failing with "Cannot select"
    # (llvm/llvm-project#229442, unfixed as of LLVM 23). Expand it on flat memory on every
    # target, where LLVM otherwise expands it depending on the scope and metadata.
    if version >= v"22" && inst.binop == LLVM.AtomicRMWBinOp.USubSat &&
       T isa LLVM.IntegerType && T.width == 32
        as == 0 && return true
        as == 3 && occursin(r"^gfx(103\d|11\d\d|10-3-generic|11(-\d+)?-generic)$",
                            target.dev_isa) && return true
    end

    # Before LLVM 21, the atomic optimizer broadcasts the result of 8- and 16-bit operations
    # on uniform addresses with an illegal `readfirstlane`, crashing the back-end
    # (llvm/llvm-project#128388). Expand the operations it optimizes when their result is
    # used, without trying to predict whether the address is uniform.
    if version < v"21" && T isa LLVM.IntegerType && T.width in (8, 16) && as in (0, 1, 3) &&
       inst.binop in OPTIMIZED_ATOMICRMW_OPS && !isempty(inst.uses)
        return true
    end

    return false
end

function lower_throw_extra!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    changed = false
    @tracepoint "lower throw (extra)" begin

    throw_functions = [
        r"julia_bounds_error.*",
        r"julia_throw_boundserror.*",
        r"julia_error_if_canonical_getindex.*",
        r"julia_error_if_canonical_setindex.*",
        r"julia___subarray_throw_boundserror.*",
    ]

    for f in mod.functions
        f_name = f.name
        for fn in throw_functions
            if occursin(fn, f_name)
                for use in collect(f.uses)
                    call = use.user::LLVM.CallInst

                    # replace the throw with a trap
                    @dispose builder=IRBuilder() begin
                        position!(builder, LLVM.before(call))
                        emit_exception!(job, builder, f_name, call)
                    end

                    # remove the call (collecting its arguments first, as the view is live)
                    nargs = length(f.parameters)
                    call_args = collect(call.arguments)
                    erase!(call)

                    # HACK: kill the exceptions' unused arguments
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
            end
        end
    end

    end
    return changed
end

can_vectorize(job::CompilerJob{GCNCompilerTarget}) = true
