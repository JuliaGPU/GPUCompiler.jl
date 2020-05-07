# implementation of the GPUCompiler interfaces for generating GCN code

## target

export GCNCompilerTarget

Base.@kwdef struct GCNCompilerTarget <: AbstractCompilerTarget
    dev_isa::String
end

llvm_triple(::GCNCompilerTarget) = "amdgcn-amd-amdhsa"

function llvm_machine(target::GCNCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple)

    cpu = target.dev_isa
    feat = ""
    optlevel = LLVM.API.LLVMCodeGenLevelDefault
    reloc = LLVM.API.LLVMRelocPIC
    tm = TargetMachine(t, triple, cpu, feat, optlevel, reloc)
    asm_verbosity!(tm, true)

    return tm
end


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{GCNCompilerTarget}) = "gcn-$(job.target.dev_isa)"

const gcn_intrinsics = () # TODO: ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(::CompilerJob{GCNCompilerTarget}, fn::String) = in(fn, gcn_intrinsics)

function process_kernel!(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module, kernel::LLVM.Function)
    # AMDGPU kernel calling convention
    callconv!(kernel, LLVM.API.LLVMCallConv(91))
end

function add_lowering_passes!(job::CompilerJob{GCNCompilerTarget}, pm::LLVM.PassManager)
    add!(pm, ModulePass("LowerThrowExtra", lower_throw_extra!))
end

function lower_throw_extra!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "lower throw (extra)" begin

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

                    # replace the throw with a return
                    new_insn = nothing
                    let builder = Builder(JuliaContext())
                        position!(builder, call)
                        new_insn = ret!(builder)
                        dispose(builder)
                    end

                    # HACK: kill instructions in block at and after the call
                    bb = LLVM.parent(call)
                    call_args = collect(operands(call))[1:end-1] # last arg is function itself
                    unsafe_delete!(LLVM.parent(call), call)
                    kill = false
                    for insn in instructions(bb)
                        if insn == new_insn
                            kill = true
                        elseif kill
                            if insn isa LLVM.UnreachableInst
                                break
                            elseif insn isa LLVM.BrInst
                                @warn "Killing branch, pass may fail: $insn"
                            end
                            unsafe_delete!(bb, insn)
                        end
                    end

                    # remove the call

                    # HACK: kill the exceptions' unused arguments
                    for arg in call_args
                        # peek through casts
                        if isa(arg, LLVM.AddrSpaceCastInst)
                            cast = arg
                            arg = first(operands(cast))
                            isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                        end

                        if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                            unsafe_delete!(LLVM.parent(arg), arg)
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
