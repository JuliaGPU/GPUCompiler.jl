# implementation of the GPUCompiler interfaces for generating GCN code

const AMDGPU_LLVM_Backend_jll =
    LazyModule("AMDGPU_LLVM_Backend_jll",
               UUID("cc5c0156-bd05-5a77-8a68-bb0aafb29019"))


## target

export GCNCompilerTarget

Base.@kwdef struct GCNCompilerTarget <: AbstractCompilerTarget
    dev_isa::String
    features::String=""
end
GCNCompilerTarget(dev_isa; features="") = GCNCompilerTarget(dev_isa, features)

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

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{GCNCompilerTarget}) = "gcn-$(job.config.target.dev_isa)$(job.config.target.features)"

const gcn_intrinsics = () # TODO: ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(::CompilerJob{GCNCompilerTarget}, fn::String) = in(fn, gcn_intrinsics)

pass_by_ref(@nospecialize(job::CompilerJob{GCNCompilerTarget})) = true

function finish_module!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    lower_throw_extra!(job, mod)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMAMDGPUKERNELCallConv)
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

@unlocked function mcgen(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                         mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    if !isavailable(AMDGPU_LLVM_Backend_jll) || !AMDGPU_LLVM_Backend_jll.is_available()
        # fall back to the in-process LLVM back-end, which is deprecated and will be
        # removed in the next breaking release in favor of AMDGPU_LLVM_Backend_jll.
        safe_depwarn(
            "Generating GCN machine code with the in-process LLVM is deprecated; " *
            "load AMDGPU_LLVM_Backend_jll to use the external back-end instead.",
            :mcgen)
        if :AMDGPU ∉ LLVM.backends()
            error("AMDGPU LLVM back-end not loaded and the in-process LLVM lacks the " *
                  "AMDGPU target; cannot compile to GCN.")
        end
        return invoke(mcgen, Tuple{CompilerJob, LLVM.Module, typeof(format)},
                      job, mod, format)
    end

    target = job.config.target
    filetype = if format == LLVM.API.LLVMAssemblyFile
        "asm"
    elseif format == LLVM.API.LLVMObjectFile
        "obj"
    else
        error("Unsupported GCN output format $format")
    end

    input  = tempname(cleanup=false) * ".bc"
    output = tempname(cleanup=false) * (filetype == "asm" ? ".s" : ".o")
    write(input, mod)

    cmd = `$(AMDGPU_LLVM_Backend_jll.llc()) $input
              -mtriple=$(llvm_triple(target))
              -mcpu=$(target.dev_isa)
              -mattr=$(target.features)
              --relocation-model=pic
              -filetype=$filetype
              -o $output`
    out = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out); wait=false)
    close(out.in)
    log = strip(read(out, String))
    wait(proc)
    if !success(proc)
        # keep the input around for debugging
        msg = "Failed to compile to GCN with external llc"
        isempty(log) || (msg *= ":\n" * log)
        msg *= "\nIf you think this is a bug, please file an issue and attach $(input)."
        isfile(output) && rm(output)
        error(msg)
    elseif !isempty(log)
        # llc only diagnoses on stderr; even successful compilation may e.g. have
        # ignored an unrecognized CPU or feature, so make sure this surfaces.
        @warn "External llc reported:\n$log"
    end

    code = filetype == "asm" ? read(output, String) : String(read(output))
    rm(input)
    rm(output)
    return code
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
