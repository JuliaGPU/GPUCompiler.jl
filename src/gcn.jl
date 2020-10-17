# implementation of the GPUCompiler interfaces for generating GCN code

## target

export GCNCompilerTarget

Base.@kwdef struct GCNCompilerTarget <: AbstractCompilerTarget
    dev_isa::String
end

llvm_triple(::GCNCompilerTarget) = "amdgcn-amd-amdhsa"

function llvm_machine(target::GCNCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple=triple)

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

# we have to fake our target early in the pipeline because Julia's optimization passes
# weren't designed for a non-0 stack addrspace, and the AMDGPU target is very strict
# about which addrspaces are permitted for various code patterns
function process_module!(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module)
    triple!(mod, llvm_triple(NativeCompilerTarget()))
    datalayout!(mod, llvm_datalayout(NativeCompilerTarget()))
end

function process_kernel!(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module, kernel::LLVM.Function)
    kernel = lower_byval(job, mod, kernel)

    # calling convention
    callconv!(kernel, LLVM.API.LLVMAMDGPUKERNELCallConv)

    kernel
end

function add_lowering_passes!(job::CompilerJob{GCNCompilerTarget}, pm::LLVM.PassManager)
    add!(pm, ModulePass("LowerThrowExtra", lower_throw_extra!))
end
# We need to do alloca rewriting (from 0 to 5) after Julia's optimization
# passes because of two reasons:
# 1. Debug builds call the target verifier first, which would trip if AMDGPU
#    was the target at that time
# 2. We don't want any chance of messing with Julia's optimizations, since they
#    eliminate target-unsafe IR patterns
function optimize_module!(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module)
    # revert back to the AMDGPU target
    triple!(mod, llvm_triple(job.target))
    datalayout!(mod, llvm_datalayout(job.target))

    tm = llvm_machine(job.target)
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        add!(pm, FunctionPass("FixAllocaAddrspace", fix_alloca_addrspace!))

        run!(pm, mod)
    end
end

function lower_throw_extra!(mod::LLVM.Module)
    job = current_job::CompilerJob
    ctx = context(mod)
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

                    # replace the throw with a trap
                    let builder = Builder(ctx)
                        position!(builder, call)
                        emit_exception!(builder, f_name, call)
                        dispose(builder)
                    end

                    # remove the call
                    call_args = collect(operands(call))[1:end-1] # last arg is function itself
                    unsafe_delete!(LLVM.parent(call), call)

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
function fix_alloca_addrspace!(fn::LLVM.Function)
    changed = false
    alloca_as = 5
    ctx = context(fn)

    for bb in blocks(fn)
        for insn in instructions(bb)
            if isa(insn, LLVM.AllocaInst)
                ty = llvmtype(insn)
                ety = eltype(ty)
                addrspace(ty) == alloca_as && continue

                new_insn = nothing
                Builder(ctx) do builder
                    position!(builder, insn)
                    _alloca = alloca!(builder, ety, name(insn))
                    new_insn = addrspacecast!(builder, _alloca, ty)
                end
                replace_uses!(insn, new_insn)
                unsafe_delete!(LLVM.parent(insn), insn)
            end
        end
    end

    return changed
end


function emit_trap!(job::CompilerJob{GCNCompilerTarget}, builder, mod, inst)
    ctx = context(mod)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    if Base.libllvm_version < v"9"
        rl_ft = LLVM.FunctionType(LLVM.Int32Type(ctx),
                                  [LLVM.Int32Type(ctx)])
        rl = if haskey(functions(mod), "llvm.amdgcn.readfirstlane")
            functions(mod)["llvm.amdgcn.readfirstlane"]
        else
            LLVM.Function(mod, "llvm.amdgcn.readfirstlane", rl_ft)
        end
        # FIXME: Early versions of the AMDGPU target fail to skip machine
        # blocks with certain side effects when EXEC==0, except when certain
        # criteria are met within said block. We emit a v_readfirstlane_b32
        # instruction here, as that is sufficient to trigger a skip. Without
        # this, the target will only attempt to do a "masked branch", which
        # only works on vector instructions (trap is a scalar instruction, and
        # therefore it is executed even when EXEC==0).
        rl_val = call!(builder, rl, [ConstantInt(Int32(32), ctx)])
        rl_bc = inttoptr!(builder, rl_val, LLVM.PointerType(LLVM.Int32Type(ctx)))
        store!(builder, rl_val, rl_bc)
    end
    call!(builder, trap)
end
