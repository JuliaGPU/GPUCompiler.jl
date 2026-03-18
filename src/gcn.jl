# implementation of the GPUCompiler interfaces for generating GCN code

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
    lower_throw_extra!(mod)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMAMDGPUKERNELCallConv)
    end

    return entry
end

function finish_ir!(@nospecialize(job::CompilerJob{GCNCompilerTarget}), mod::LLVM.Module,
                    entry::LLVM.Function)
    if job.config.kernel
        entry = add_kernarg_address_spaces!(job, mod, entry)
    end
    return entry
end

# Rewrite byref kernel parameters from flat (addrspace 0) to kernarg (addrspace 4).
#
# On AMDGPU, the kernarg segment is in address space 4 and is scalar-loadable via s_load.
# Clang emits byref parameters as `ptr addrspace(4)` from the frontend, but Julia's
# RemoveJuliaAddrspacesPass strips all address spaces to flat. This pass restores the
# correct address space so that struct field loads from byref arguments become s_load
# instead of flat_load.
#
# NOTE: must run after optimization, where RemoveJuliaAddrspacesPass has already
# converted Julia's addrspace(11) to flat (addrspace 0) on these parameters.
function add_kernarg_address_spaces!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                     f::LLVM.Function)
    ft = function_type(f)

    # find the byref parameters
    byref_mask = BitVector(undef, length(parameters(ft)))
    args = classify_arguments(job, ft; post_optimization=job.config.optimize)
    filter!(args) do arg
        arg.cc != GHOST
    end
    for arg in args
        byref_mask[arg.idx] = (arg.cc == BITS_REF || arg.cc == KERNEL_STATE)
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

    # generate the new function type with kernarg address space on byref params
    new_types = LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        if byref_mask[i] && param isa LLVM.PointerType && addrspace(param) == 0
            push!(new_types, LLVM.PointerType(#=kernarg=# 4))
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

    # insert addrspacecasts from kernarg (4) back to flat (0) so that the cloned IR
    # (which expects flat pointers) continues to work. InferAddressSpaces will then
    # propagate addrspace(4) through GEPs and loads, eliminating the casts.
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry_bb = BasicBlock(new_f, "conversion")
        position!(builder, entry_bb)

        for (i, param) in enumerate(parameters(ft))
            if byref_mask[i] && param isa LLVM.PointerType && addrspace(param) == 0
                cast = addrspacecast!(builder, parameters(new_f)[i], LLVM.PointerType(0))
                push!(new_args, cast)
            else
                push!(new_args, parameters(new_f)[i])
            end
            for attr in collect(parameter_attributes(f, i))
                push!(parameter_attributes(new_f, i), attr)
            end
        end

        # clone the original function body
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(parameters(f))
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # fall through from conversion block to cloned entry
        br!(builder, blocks(new_f)[2])
    end

    # replace the old function
    fn = LLVM.name(f)
    prune_constexpr_uses!(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)

    # propagate addrspace(4) through GEPs and loads, then clean up
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, NewPMFunctionPassManager()) do fpm
            add!(fpm, InferAddressSpacesPass())
        end
        add!(pb, NewPMFunctionPassManager()) do fpm
            add!(fpm, SimplifyCFGPass())
            add!(fpm, SROAPass())
            add!(fpm, EarlyCSEPass())
            add!(fpm, InstCombinePass())
        end
        run!(pb, mod)
    end

    return functions(mod)[fn]
end


## LLVM passes

function lower_throw_extra!(mod::LLVM.Module)
    job = current_job::CompilerJob
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
                        emit_exception!(builder, f_name, call)
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

function emit_trap!(job::CompilerJob{GCNCompilerTarget}, builder, mod, inst)
    trap_ft = LLVM.FunctionType(LLVM.VoidType())
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", trap_ft)
    end
    call!(builder, trap_ft, trap)
end

can_vectorize(job::CompilerJob{GCNCompilerTarget}) = true
