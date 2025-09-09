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

function finish_module!(@nospecialize(job::CompilerJob{GCNCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    lower_throw_extra!(mod)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMAMDGPUKERNELCallConv)

        # work around bad byval codegen (JuliaGPU/GPUCompiler.jl#92)
        entry = lower_byval(job, mod, entry)
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
