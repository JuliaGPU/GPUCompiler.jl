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
    t = Target(triple=triple)

    cpu = target.dev_isa
    feat = target.features
    reloc = LLVM.API.LLVMRelocPIC
    tm = TargetMachine(t, triple, cpu, feat; reloc)
    asm_verbosity!(tm, true)

    return tm
end


## job

function isintrinsic(@nospecialize(job::CompilerJob{GCNCompilerTarget}), fn::String)
    startswith(fn, "llvm.amdgcn.") && return true
    # The ROCm device libraries use `llvm.frexp` and `llvm.ldexp`, which were only added in
    # LLVM 17, so they are not recognized as intrinsics by older versions of LLVM.
    # Final code generation knows how to lower them, so accept them here.
    return startswith(fn, "llvm.frexp.") || startswith(fn, "llvm.ldexp.")
end

pass_by_ref(@nospecialize(job::CompilerJob{GCNCompilerTarget})) = true

function finish_module!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    lower_throw_extra!(job, mod)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMAMDGPUKERNELCallConv)

        # workgroup size bounds; the backend sizes its register budget for the
        # worst case (1,1024) when unset
        if job.config.target.minthreads !== nothing ||
           job.config.target.maxthreads !== nothing
            lo = prod(something(job.config.target.minthreads, 1))
            hi = prod(something(job.config.target.maxthreads, 1024))
            push!(function_attributes(entry),
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
        tm = llvm_machine(job.config.target)
        @dispose pb=NewPMPassBuilder() begin
            add!(pb, NewPMFunctionPassManager()) do fpm
                add!(fpm, InferAddressSpacesPass())
                add!(fpm, SROAPass())
                add!(fpm, instcombine_pass(job))
                add!(fpm, EarlyCSEPass())
                add!(fpm, SimplifyCFGPass())
            end
            run!(pb, mod, tm)
        end
    end
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
    ft = function_type(f)

    # find the byref parameters by checking for the byref attribute directly,
    # rather than re-classifying arguments (which can fail on typed-pointer LLVM
    # due to element type mismatches in classify_arguments assertions).
    byref_kind = LLVM.API.LLVMGetEnumAttributeKindForName("byref", 5)
    byref_mask = BitVector(undef, length(parameters(ft)))
    for i in 1:length(parameters(ft))
        attrs = collect(parameter_attributes(f, i))
        byref_mask[i] = any(a -> a isa TypeAttribute && kind(a) == byref_kind, attrs)
    end

    # check if any flat pointer byref params need rewriting
    needs_rewrite = false
    for (i, param) in enumerate(parameters(ft))
        if byref_mask[i] && param isa LLVM.PointerType && addrspace(param) == 0
            needs_rewrite = true
            break
        end
    end
    needs_rewrite || return f

    # generate the new function type with constant address space on byref flat-pointer params
    param_types = parameters(ft)
    flat_byref(i) = byref_mask[i] && param_types[i] isa LLVM.PointerType && addrspace(param_types[i]) == 0
    new_types = Union{Nothing,LLVMType}[
        flat_byref(i) ? (supports_typed_pointers(context()) ?
                            LLVM.PointerType(eltype(param_types[i]), #=constant=# 4) :
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
        for attr in collect(parameter_attributes(f, i))
            push!(parameter_attributes(new_f, i), attr)
        end
    end

    replace_function!(f, new_f)

    # clean up the extra conversion block
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, NewPMFunctionPassManager()) do fpm
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
                         mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
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

    filetype = if format == LLVM.API.LLVMAssemblyFile
        AMDGPUAssemblyFile
    elseif format == LLVM.API.LLVMObjectFile
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


## LLVM passes

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

    for f in functions(mod)
        f_name = LLVM.name(f)
        for fn in throw_functions
            if occursin(fn, f_name)
                for use in uses(f)
                    call = user(use)::LLVM.CallInst

                    # replace the throw with a trap
                    @dispose builder=IRBuilder() begin
                        position!(builder, call)
                        emit_exception!(job, builder, f_name, call)
                    end

                    # remove the call
                    nargs = length(parameters(f))
                    call_args = arguments(call)
                    erase!(call)

                    # HACK: kill the exceptions' unused arguments
                    for arg in call_args
                        # peek through casts
                        if isa(arg, LLVM.AddrSpaceCastInst)
                            cast = arg
                            arg = first(operands(cast))
                            isempty(uses(cast)) && erase!(cast)
                        end

                        if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                            erase!(arg)
                        end
                    end

                    changed = true
                end

                @compiler_assert isempty(uses(f)) job
            end
        end
    end

    end
    return changed
end

can_vectorize(job::CompilerJob{GCNCompilerTarget}) = true
