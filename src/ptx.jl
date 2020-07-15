# implementation of the GPUCompiler interfaces for generating PTX code

## target

export PTXCompilerTarget

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    cap::VersionNumber

    # optional properties
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing
end

source_code(::PTXCompilerTarget) = "ptx"

llvm_triple(::PTXCompilerTarget) = Int===Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda"

function llvm_machine(target::PTXCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple=triple)

    cpu = "sm_$(target.cap.major)$(target.cap.minor)"
    feat = "+ptx60" # we only support CUDA 9.0+ and LLVM 6.0+
    tm = TargetMachine(t, triple, cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end

# the default datalayout does not match the one in the NVPTX user guide
llvm_datalayout(::PTXCompilerTarget) = Int===Int64 ?
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64" :
    "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


## job

function Base.show(io::IO, job::CompilerJob{PTXCompilerTarget})
    print(io, "PTX CompilerJob of ", job.source)
    print(io, " for sm_$(job.target.cap.major)$(job.target.cap.minor)")

    job.target.minthreads !== nothing && print(io, ", minthreads=$(job.target.minthreads)")
    job.target.maxthreads !== nothing && print(io, ", maxthreads=$(job.target.maxthreads)")
    job.target.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.target.blocks_per_sm)")
    job.target.maxregs !== nothing && print(io, ", maxregs=$(job.target.maxregs)")
end

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(::CompilerJob{PTXCompilerTarget}, fn::String) = in(fn, ptx_intrinsics)

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{PTXCompilerTarget}) = "ptx-sm_$(job.target.cap.major)$(job.target.cap.minor)"

function process_kernel!(job::CompilerJob{PTXCompilerTarget}, mod::LLVM.Module, kernel::LLVM.Function)
    # property annotations
    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1), JuliaContext())])

    ## expected CTA sizes
    if job.target.minthreads != nothing
        for (dim, name) in enumerate([:x, :y, :z])
            bound = dim <= length(job.target.minthreads) ? job.target.minthreads[dim] : 1
            append!(annotations, [MDString("reqntid$name"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end
    if job.target.maxthreads != nothing
        for (dim, name) in enumerate([:x, :y, :z])
            bound = dim <= length(job.target.maxthreads) ? job.target.maxthreads[dim] : 1
            append!(annotations, [MDString("maxntid$name"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end

    if job.target.blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"),
                              ConstantInt(Int32(job.target.blocks_per_sm), JuliaContext())])
    end

    if job.target.maxregs != nothing
        append!(annotations, [MDString("maxnreg"),
                              ConstantInt(Int32(job.target.maxregs), JuliaContext())])
    end

    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    if LLVM.version() >= v"8"
        # calling convention
        for fun in functions(mod)
            callconv!(kernel, LLVM.API.LLVMPTXDeviceCallConv)
        end
        callconv!(kernel, LLVM.API.LLVMPTXKernelCallConv)
    end

    return kernel
end

function add_lowering_passes!(job::CompilerJob{PTXCompilerTarget}, pm::LLVM.PassManager)
    add!(pm, FunctionPass("HideUnreachable", hide_unreachable!))
    add!(pm, ModulePass("HideTrap", hide_trap!))
end

function add_optimization_passes!(job::CompilerJob{PTXCompilerTarget}, pm::LLVM.PassManager)
    # NVPTX's target machine info enables runtime unrolling,
    # but Julia's pass sequence only invokes the simple unroller.
    loop_unroll!(pm)
    instruction_combining!(pm)  # clean-up redundancy
    licm!(pm)                   # the inner runtime check might be outer loop invariant

    # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
    # that code still needs to be optimized (arguably, multiple unroll passes should be
    # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
    early_csemem_ssa!(pm) # TODO: gvn instead? see NVPTXTargetMachine.cpp::addEarlyCSEOrGVNPass
    dead_store_elimination!(pm)

    constant_merge!(pm)

    cfgsimplification!(pm)

    # get rid of the internalized functions; now possible unused
    global_dce!(pm)
end


## LLVM passes

# HACK: this pass removes `unreachable` information from LLVM
#
# `ptxas` is buggy and cannot deal with thread-divergent control flow in the presence of
# shared memory (see JuliaGPU/CUDAnative.jl#4). avoid that by rewriting control flow to fall
# through any other block. this is semantically invalid, but the code is unreachable anyhow
# (and we expect it to be preceded by eg. a noreturn function, or a trap).
#
# TODO: can LLVM do this with structured CFGs? It seems to have some support, but seemingly
#       only to prevent introducing non-structureness during optimization (ie. the front-end
#       is still responsible for generating structured control flow).
function hide_unreachable!(fun::LLVM.Function)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "hide unreachable" begin

    # remove `noreturn` attributes
    #
    # when calling a `noreturn` function, LLVM places an `unreachable` after the call.
    # this leads to an early `ret` from the function.
    attrs = function_attributes(fun)
    delete!(attrs, EnumAttribute("noreturn", 0, JuliaContext()))

    # build a map of basic block predecessors
    predecessors = Dict(bb => Set{LLVM.BasicBlock}() for bb in blocks(fun))
    @timeit_debug to "predecessors" for bb in blocks(fun)
        insts = instructions(bb)
        if !isempty(insts)
            inst = last(insts)
            if isterminator(inst)
                for bb′ in successors(inst)
                    push!(predecessors[bb′], bb)
                end
            end
        end
    end

    # scan for unreachable terminators and alternative successors
    worklist = Pair{LLVM.BasicBlock, Union{Nothing,LLVM.BasicBlock}}[]
    @timeit_debug to "find" for bb in blocks(fun)
        unreachable = terminator(bb)
        if isa(unreachable, LLVM.UnreachableInst)
            unsafe_delete!(bb, unreachable)
            changed = true

            try
                terminator(bb)
                # the basic-block is still terminated properly, nothing to do
                # (this can happen with `ret; unreachable`)
                # TODO: `unreachable; unreachable`
            catch ex
                isa(ex, UndefRefError) || rethrow(ex)
                let builder = Builder(JuliaContext())
                    position!(builder, bb)

                    # find the strict predecessors to this block
                    preds = collect(predecessors[bb])

                    # find a fallthrough block: recursively look at predecessors
                    # and find a successor that branches to any other block
                    fallthrough = nothing
                    while !isempty(preds)
                        # find an alternative successor
                        for pred in preds, succ in successors(terminator(pred))
                            if succ != bb
                                fallthrough = succ
                                break
                            end
                        end
                        fallthrough === nothing || break

                        # recurse upwards
                        old_preds = copy(preds)
                        empty!(preds)
                        for pred in old_preds
                            append!(preds, predecessors[pred])
                        end
                    end
                    push!(worklist, bb => fallthrough)

                    dispose(builder)
                end
            end
        end
    end

    # apply the pending terminator rewrites
    @timeit_debug to "replace" if !isempty(worklist)
        let builder = Builder(JuliaContext())
            for (bb, fallthrough) in worklist
                position!(builder, bb)
                if fallthrough !== nothing
                    br!(builder, fallthrough)
                else
                    # couldn't find any other successor. this happens with functions
                    # that only contain a single block, or when the block is dead.
                    ft = eltype(llvmtype(fun))
                    if return_type(ft) == LLVM.VoidType(JuliaContext())
                        # even though returning can lead to invalid control flow,
                        # it mostly happens with functions that just throw,
                        # and leaving the unreachable there would make the optimizer
                        # place another after the call.
                        ret!(builder)
                    else
                        unreachable!(builder)
                    end
                end
            end
        end
    end

    end
    return changed
end

# HACK: this pass removes calls to `trap` and replaces them with inline assembly
#
# if LLVM knows we're trapping, code is marked `unreachable` (see `hide_unreachable!`).
function hide_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "hide trap" begin

    # inline assembly to exit a thread, hiding control flow from LLVM
    exit_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()))
    exit = if job.target.cap < v"7"
        # ptxas for old compute capabilities has a bug where it messes up the
        # synchronization stack in the presence of shared memory and thread-divergend exit.
        InlineAsm(exit_ft, "trap;", "", true)
    else
        InlineAsm(exit_ft, "exit;", "", true)
    end

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                let builder = Builder(JuliaContext())
                    position!(builder, val)
                    call!(builder, exit)
                    dispose(builder)
                end
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end
    end

    end
    return changed
end
