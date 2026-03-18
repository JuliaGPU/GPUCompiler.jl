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

# AMDGPU constant/kernarg address space
const GCN_ADDRSPACE_CONSTANT = 4

# Rewrite byref pointer parameters from addrspace 0 to addrspace 4 (kernarg),
# loading the data into local allocas so the function body can use generic pointers.
# SROA will decompose the allocas during the optimization pipeline that follows.
function rewrite_byref_addrspaces!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                                   mod::LLVM.Module, f::LLVM.Function)
    ft = function_type(f)

    # find byref parameters and their types
    args = classify_arguments(job, ft)
    filter!(args) do arg
        arg.cc != GHOST
    end

    byref = BitVector(undef, length(parameters(ft)))
    byref_types = Vector{Any}(undef, length(parameters(ft)))
    for i in 1:length(byref)
        byref[i] = false
        for attr in collect(parameter_attributes(f, i))
            if kind(attr) == kind(TypeAttribute("byref", LLVM.VoidType()))
                byref[i] = true
            end
        end
    end
    for arg in args
        if arg.idx !== nothing && byref[arg.idx]
            byref_types[arg.idx] = arg.typ
        end
    end
    any(byref) || return f

    # build new function type with addrspace(4) pointers for byref params
    new_types = LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        if byref[i]
            push!(new_types, LLVM.PointerType(GCN_ADDRSPACE_CONSTANT))
        else
            push!(new_types, param)
        end
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    callconv!(new_f, callconv(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # copy parameter attributes
    for (i, _) in enumerate(parameters(ft))
        for attr in collect(parameter_attributes(f, i))
            push!(parameter_attributes(new_f, i), attr)
        end
    end

    # load byref arguments from addrspace(4) into local allocas
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)

        for (i, param) in enumerate(parameters(ft))
            if byref[i]
                # load the value from the kernarg pointer and store into a stack slot,
                # so the function body can keep using addrspace(0) pointers.
                # SROA will decompose this during optimization.
                llvm_typ = convert(LLVMType, byref_types[i])
                val = load!(builder, llvm_typ, parameters(new_f)[i])
                ptr = alloca!(builder, llvm_typ)
                store!(builder, val, ptr)
                push!(new_args, ptr)
            else
                push!(new_args, parameters(new_f)[i])
            end
        end

        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(parameters(f))
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        br!(builder, blocks(new_f)[2])
    end

    # replace old function
    fn = LLVM.name(f)
    prune_constexpr_uses!(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)

    return new_f
end

function finish_module!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    lower_throw_extra!(mod)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMAMDGPUKERNELCallConv)

        # rewrite byref parameters to use the kernarg address space
        entry = rewrite_byref_addrspaces!(job, mod, entry)
    end

    return entry
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
