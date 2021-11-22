# implementation of the GPUCompiler interfaces for generating PTX code

## target

export PTXCompilerTarget

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    cap::VersionNumber
    ptx::VersionNumber = v"6.0" # for compatibility with older versions of CUDA.jl

    # codegen quirks
    ## can we emit debug info in the PTX assembly?
    debuginfo::Bool = false
    ## do we permit unrachable statements, which often result in divergent control flow?
    unreachable::Bool = false
    ## can exceptions use `exit` (which doesn't kill the GPU), or should they use `trap`?
    exitable::Bool = false

    # optional properties
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing
end

function Base.hash(target::PTXCompilerTarget, h::UInt)
    h = hash(target.cap, h)
    h = hash(target.ptx, h)

    h = hash(target.debuginfo, h)
    h = hash(target.unreachable, h)
    h = hash(target.exitable, h)

    h = hash(target.minthreads, h)
    h = hash(target.maxthreads, h)
    h = hash(target.blocks_per_sm, h)
    h = hash(target.maxregs, h)

    h
end

source_code(target::PTXCompilerTarget) = "ptx"

llvm_triple(target::PTXCompilerTarget) = Int===Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda"

function llvm_machine(target::PTXCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple=triple)

    tm = TargetMachine(t, triple, "sm_$(target.cap.major)$(target.cap.minor)",
                       "+ptx$(target.ptx.major)$(target.ptx.minor)")
    asm_verbosity!(tm, true)

    return tm
end

# the default datalayout does not match the one in the NVPTX user guide
llvm_datalayout(target::PTXCompilerTarget) = Int===Int64 ?
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64" :
    "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
     "-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


## job

function Base.show(io::IO, @nospecialize(job::CompilerJob{PTXCompilerTarget}))
    print(io, "PTX CompilerJob of ", job.source)
    print(io, " for sm_$(job.target.cap.major)$(job.target.cap.minor)")

    job.target.minthreads !== nothing && print(io, ", minthreads=$(job.target.minthreads)")
    job.target.maxthreads !== nothing && print(io, ", maxthreads=$(job.target.maxthreads)")
    job.target.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.target.blocks_per_sm)")
    job.target.maxregs !== nothing && print(io, ", maxregs=$(job.target.maxregs)")
end

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(@nospecialize(job::CompilerJob{PTXCompilerTarget}), fn::String) =
    in(fn, ptx_intrinsics)

# XXX: the debuginfo part should be handled by GPUCompiler as it applies to all back-ends.
runtime_slug(@nospecialize(job::CompilerJob{PTXCompilerTarget})) =
    "ptx-sm_$(job.target.cap.major)$(job.target.cap.minor)" *
       "-debuginfo=$(Int(llvm_debug_info(job)))" *
       "-exitable=$(job.target.exitable)"

function process_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}), mod::LLVM.Module)
    ctx = context(mod)

    # calling convention
    if LLVM.version() >= v"8"
        for f in functions(mod)
            # JuliaGPU/GPUCompiler.jl#97
            #callconv!(f, LLVM.API.LLVMPTXDeviceCallConv)
        end
    end

    # emit the device capability and ptx isa version as constants in the module. this makes
    # it possible to 'query' these in device code, relying on LLVM to optimize the checks
    # away and generate static code. note that we only do so if there's actual uses of these
    # variables; unconditionally creating a gvar would result in duplicate declarations.
    for (name, value) in ["sm_major"  => job.target.cap.major,
                          "sm_minor"  => job.target.cap.minor,
                          "ptx_major" => job.target.ptx.major,
                          "ptx_minor" => job.target.ptx.minor]
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            initializer!(gv, ConstantInt(LLVM.Int32Type(ctx), value))
            # change the linkage so that we can inline the value
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        end
    end
end

function process_entry!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        if LLVM.version() >= v"8"
            # calling convention
            callconv!(entry, LLVM.API.LLVMPTXKernelCallConv)
        end
    end

    return entry
end

function add_lowering_passes!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                              pm::LLVM.PassManager)
    # hide `unreachable` from LLVM so that it doesn't introduce divergent control flow
    if !job.target.unreachable
        add!(pm, FunctionPass("HideUnreachable", hide_unreachable!))
    end

    # even if we support `unreachable`, we still prefer `exit` to `trap`
    add!(pm, ModulePass("HideTrap", hide_trap!))

    # we emit properties (of the device and ptx isa) as private global constants,
    # so run the optimizer so that they are inlined before the rest of the optimizer runs.
    global_optimizer!(pm)
end

function optimize_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                          mod::LLVM.Module)
    tm = llvm_machine(job.target)
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        # needed by GemmKernels.jl-like code
        speculative_execution_if_has_branch_divergence!(pm)

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

        cfgsimplification!(pm)

        # get rid of the internalized functions; now possible unused
        global_dce!(pm)

        run!(pm, mod)
    end
end

function finish_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # work around bad byval codegen (JuliaGPU/GPUCompiler.jl#92)
        entry = lower_byval(job, mod, entry)
        # TODO: optimization passes to clean-up byval

        # add metadata annotations for the assembler to the module
        # NOTE: we need to do this as late as possible, because otherwise the metadata (which
        #       refers to a specific function) can get lost when cloning functions. normally
        #       RAUW updates those references, but we can't RAUW with a changed function type.

        # property annotations
        annotations = Metadata[entry]

        ## kernel metadata
        append!(annotations, [MDString("kernel"; ctx),
                              ConstantInt(Int32(1); ctx)])

        ## expected CTA sizes
        if job.target.minthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.target.minthreads) ? job.target.minthreads[dim] : 1
                append!(annotations, [MDString("reqntid$name"; ctx),
                                      ConstantInt(Int32(bound); ctx)])
            end
        end
        if job.target.maxthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.target.maxthreads) ? job.target.maxthreads[dim] : 1
                append!(annotations, [MDString("maxntid$name"; ctx),
                                      ConstantInt(Int32(bound); ctx)])
            end
        end

        if job.target.blocks_per_sm !== nothing
            append!(annotations, [MDString("minctasm"; ctx),
                                  ConstantInt(Int32(job.target.blocks_per_sm); ctx)])
        end

        if job.target.maxregs !== nothing
            append!(annotations, [MDString("maxnreg"; ctx),
                                  ConstantInt(Int32(job.target.maxregs); ctx)])
        end

        push!(metadata(mod)["nvvm.annotations"], MDNode(annotations; ctx))
    end

    return entry
end

function llvm_debug_info(@nospecialize(job::CompilerJob{PTXCompilerTarget}))
    # allow overriding the debug info from CUDA.jl
    if job.target.debuginfo
        invoke(llvm_debug_info, Tuple{CompilerJob}, job)
    else
        LLVM.API.LLVMDebugEmissionKindNoDebug
    end
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
    ctx = context(fun)
    changed = false
    @timeit_debug to "hide unreachable" begin

    # remove `noreturn` attributes
    #
    # when calling a `noreturn` function, LLVM places an `unreachable` after the call.
    # this leads to an early `ret` from the function.
    attrs = function_attributes(fun)
    delete!(attrs, EnumAttribute("noreturn", 0; ctx))

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
                let builder = Builder(ctx)
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
        let builder = Builder(ctx)
            for (bb, fallthrough) in worklist
                position!(builder, bb)
                if fallthrough !== nothing
                    br!(builder, fallthrough)
                else
                    # couldn't find any other successor. this happens with functions
                    # that only contain a single block, or when the block is dead.
                    ft = eltype(llvmtype(fun))
                    if return_type(ft) == LLVM.VoidType(ctx)
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
    ctx = context(mod)
    changed = false
    @timeit_debug to "hide trap" begin

    # inline assembly to exit a thread, hiding control flow from LLVM
    exit_ft = LLVM.FunctionType(LLVM.VoidType(ctx))
    exit = if job.target.exitable
        InlineAsm(exit_ft, "exit;", "", true)
    else
        InlineAsm(exit_ft, "trap;", "", true)
    end

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                let builder = Builder(ctx)
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
