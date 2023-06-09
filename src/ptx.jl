# implementation of the GPUCompiler interfaces for generating PTX code

## target

export PTXCompilerTarget

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    cap::VersionNumber
    ptx::VersionNumber = v"6.0" # for compatibility with older versions of CUDA.jl

    # codegen quirks
    ## can we emit debug info in the PTX assembly?
    debuginfo::Bool = false

    # optional properties
    minthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    maxthreads::Union{Nothing,Int,NTuple{<:Any,Int}} = nothing
    blocks_per_sm::Union{Nothing,Int} = nothing
    maxregs::Union{Nothing,Int} = nothing

    # deprecated; remove with next major version
    exitable::Union{Nothing,Bool} = nothing
    unreachable::Union{Nothing,Bool} = nothing
end

function Base.hash(target::PTXCompilerTarget, h::UInt)
    h = hash(target.cap, h)
    h = hash(target.ptx, h)

    h = hash(target.debuginfo, h)

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
llvm_datalayout(target::PTXCompilerTarget) =
    # little endian
    "e-" *
    # on 32-bit systems, use 32-bit pointers.
    # on 64-bit systems, use 64-bit pointers.
    (Int === Int64 ? "p:64:64:64-" :  "p:32:32:32-") *
    # alignment of integer types
    "i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-" *
    # alignment of floating point types
    "f32:32:32-f64:64:64-" *
    # alignment of vector types
    "v16:16:16-v32:32:32-v64:64:64-v128:128:128-" *
    # native integer widths
    "n16:32:64"

have_fma(@nospecialize(target::PTXCompilerTarget), T::Type) = true


## job

function Base.show(io::IO, @nospecialize(job::CompilerJob{PTXCompilerTarget}))
    print(io, "PTX CompilerJob of ", job.source)
    print(io, " for sm_$(job.config.target.cap.major)$(job.config.target.cap.minor)")

    job.config.target.minthreads !== nothing && print(io, ", minthreads=$(job.config.target.minthreads)")
    job.config.target.maxthreads !== nothing && print(io, ", maxthreads=$(job.config.target.maxthreads)")
    job.config.target.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.config.target.blocks_per_sm)")
    job.config.target.maxregs !== nothing && print(io, ", maxregs=$(job.config.target.maxregs)")
end

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(@nospecialize(job::CompilerJob{PTXCompilerTarget}), fn::String) =
    in(fn, ptx_intrinsics)

# XXX: the debuginfo part should be handled by GPUCompiler as it applies to all back-ends.
runtime_slug(@nospecialize(job::CompilerJob{PTXCompilerTarget})) =
    "ptx-sm_$(job.config.target.cap.major)$(job.config.target.cap.minor)" *
       "-debuginfo=$(Int(llvm_debug_info(job)))"

function finish_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)

    # emit the device capability and ptx isa version as constants in the module. this makes
    # it possible to 'query' these in device code, relying on LLVM to optimize the checks
    # away and generate static code. note that we only do so if there's actual uses of these
    # variables; unconditionally creating a gvar would result in duplicate declarations.
    for (name, value) in ["sm_major"  => job.config.target.cap.major,
                          "sm_minor"  => job.config.target.cap.minor,
                          "ptx_major" => job.config.target.ptx.major,
                          "ptx_minor" => job.config.target.ptx.minor]
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            initializer!(gv, ConstantInt(LLVM.Int32Type(ctx), value))
            # change the linkage so that we can inline the value
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        end
    end

    # update calling convention
    if LLVM.version() >= v"8"
        for f in functions(mod)
            # JuliaGPU/GPUCompiler.jl#97
            #callconv!(f, LLVM.API.LLVMPTXDeviceCallConv)
        end
    end
    if job.config.kernel && LLVM.version() >= v"8"
        callconv!(entry, LLVM.API.LLVMPTXKernelCallConv)
    end

    if job.config.kernel
        # work around bad byval codegen (JuliaGPU/GPUCompiler.jl#92)
        entry = lower_byval(job, mod, entry)
    end

    @dispose pm=ModulePassManager() begin
        # we emit properties (of the device and ptx isa) as private global constants,
        # so run the optimizer so that they are inlined before the rest of the optimizer runs.
        global_optimizer!(pm)

        run!(pm, mod)
    end

    return entry
end

function optimize_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                          mod::LLVM.Module)
    tm = llvm_machine(job.config.target)
    @dispose pm=ModulePassManager() begin
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        # TODO: need to run this earlier; optimize_module! is called after addOptimizationPasses!
        add!(pm, FunctionPass("NVVMReflect", nvvm_reflect!))

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

function finish_ir!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)

    @dispose pm=ModulePassManager() begin
        # structurize unreachable control flow
        add!(pm, FunctionPass("StructurizeUnreachable", structurize_unreachable!))

        # we prefer `exit` over `trap`
        add!(pm, ModulePass("LowerTrap", lower_trap!))

        run!(pm, mod)
    end

    if job.config.kernel
        # add metadata annotations for the assembler to the module

        # property annotations
        annotations = Metadata[entry]

        ## kernel metadata
        append!(annotations, [MDString("kernel"; ctx),
                              ConstantInt(Int32(1); ctx)])

        ## expected CTA sizes
        if job.config.target.minthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.config.target.minthreads) ? job.config.target.minthreads[dim] : 1
                append!(annotations, [MDString("reqntid$name"; ctx),
                                      ConstantInt(Int32(bound); ctx)])
            end
        end
        if job.config.target.maxthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.config.target.maxthreads) ? job.config.target.maxthreads[dim] : 1
                append!(annotations, [MDString("maxntid$name"; ctx),
                                      ConstantInt(Int32(bound); ctx)])
            end
        end

        if job.config.target.blocks_per_sm !== nothing
            append!(annotations, [MDString("minctasm"; ctx),
                                  ConstantInt(Int32(job.config.target.blocks_per_sm); ctx)])
        end

        if job.config.target.maxregs !== nothing
            append!(annotations, [MDString("maxnreg"; ctx),
                                  ConstantInt(Int32(job.config.target.maxregs); ctx)])
        end

        push!(metadata(mod)["nvvm.annotations"], MDNode(annotations; ctx))
    end

    return entry
end

function llvm_debug_info(@nospecialize(job::CompilerJob{PTXCompilerTarget}))
    # allow overriding the debug info from CUDA.jl
    if job.config.target.debuginfo
        invoke(llvm_debug_info, Tuple{CompilerJob}, job)
    else
        LLVM.API.LLVMDebugEmissionKindNoDebug
    end
end


## LLVM passes

# structurize unreachable control flow
#
# The CUDA back-end compiler ptxas needs to insert instructions that manage the harware's
# reconvergence stack. In order to do so, it needs to identify divergent regions and
# emit SSY and SYNC instructions:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     ...
#     bra.uni cont;
#   cont:
#     // end of divergent region
#     bar.sync 0;
#
# Meanwhile, LLVM's branch-folder and block-placement MIR passes will try to optimize the
# block layout, e.g., by placing unlikely blocks at the end of the function:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     @%p1 bra unlikely;
#     bra.uni cont;
#   cont:
#     // end of divergent region
#     bar.sync 0;
#   unlikely:
#     bra.uni cont;
#
# That is not a problem on the condition that the unlikely block continunes back into the
# divergent region. However, this is not the case with unreachable control flow:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     @%p1 bra throw;
#     bra.uni cont;
#   cont:
#     // end of divergent region
#     bar.sync 0;
#   throw:
#     trap;
#   exit:
#     ret;
#
# Here, the `throw` block does not have a continuation back into the divergent region.
# This is fine by itself, because the `trap` instruction will halt execution. However,
# it confuses `ptxas` in that the `throw` block now gets a fall-through edge to `exit`,
# which is outside of the intended divergent region. This results in a much larger
# divergent region, causing `bar.sync` to be executed divergently, which is not allowed.
#
# Note that the above may also happen with function calls that do not return, and is not
# limited to `trap` instructions. Also note that the problem manifests predominantly on
# Pascal hardware and earlier, as newer hardware is much more flexible in terms of
# unstructured control flow.
#
# To avoid the above, we replace `unreachable` instructions with an unconditional branch
# back into the divergent region. We identify this target by scanning for successors that
# have a multi-way branch with one of the targets being the `unreachable` block.
function structurize_unreachable!(f::LLVM.Function)
    ctx = context(f)

    # TODO:
    # - consider using branch-weight metadata to give the back-end information similar to
    #   what the `unreachable` used to encode (although NVIDIA mentioned that block layout
    #   shouldn't happen as `ptxas` does this all over again)
    # - try cloning paths with multiple predecessors, as those may originate from
    #   differently-divergent regions

    # find unreachable blocks
    unreachable_blocks = Set{BasicBlock}()
    for block in blocks(f)
        if terminator(block) isa LLVM.UnreachableInst
            push!(unreachable_blocks, block)
        end
    end
    isempty(unreachable_blocks) && return false

    changed = false
    @dispose builder=IRBuilder(ctx) begin
        # helper functions
        function replace_unreachable(block, target=nothing)
            # replace an unreachable block with a branch or return
            position!(builder, block)

            inst = terminator(block)
            isa(inst, LLVM.UnreachableInst) ||
                error("expected unreachable instruction, got $(typeof(inst)): $inst")
            unsafe_delete!(block, inst)

            if target === nothing
                # return
                f = LLVM.parent(block)
                ft = function_type(f)
                rettyp = return_type(ft)
                if rettyp == LLVM.VoidType(ctx)
                    ret!(builder)
                else
                    ret!(builder, LLVM.UndefValue(rettyp))
                end
            else
                # branch
                br!(builder, target)

                # we need to add a value to the successor's phi nodes
                for inst in instructions(target)
                    isa(inst, LLVM.PHIInst) || break
                    phi_edges = LLVM.incoming(inst)
                    push!(phi_edges, (LLVM.UndefValue(value_type(inst)), block))
                end
            end

            changed = true
        end

        function find_alternative_successor(path)
            function analyze_predecessor(path_predecessor)
                candidate_successors = filter(!isequal(path[1]), successors(path_predecessor))
                if !isempty(candidate_successors)
                    # nice, we've got a way out
                    # XXX: is there a difference which successor we pick?
                    return [first(candidate_successors)]
                else
                    # too bad, let's try to ascend
                    if in(path_predecessor, path)
                        return BasicBlock[]
                    else
                        return find_alternative_successor([path_predecessor; path])
                    end
                end
            end

            # look at predecessors to start of this path
            path_predecessors = collect(predecessors(path[1]))
            if isempty(path_predecessors)
                # we are at the entry block, so there's nothing to do
                return BasicBlock[]
            elseif length(path_predecessors) == 1
                path_predecessor = only(path_predecessors)
                return analyze_predecessor(path_predecessor)
            else
                alternative_successors = Set{BasicBlock}()
                for path_predecessor in path_predecessors
                    union!(alternative_successors, analyze_predecessor(path_predecessor))
                end
                return collect(alternative_successors)
            end
        end

        # TODO: branch weights, for optimizability similar to unreachable
        for block in unreachable_blocks
            alternative_successors = find_alternative_successor([block])
            if isempty(alternative_successors)
                replace_unreachable(block)
            else
                # multiple alternative successors indicates that there's multiple paths
                # into this unreachable block. we could try and deduplicate those paths,
                # since it's possible that they originate from differently divergent blocks
                # (e.g. if a throw block was deduplicated). however, that is tricky, and
                # easily introduces loops. furthermore, it is expected that ptxas _will_
                # support unreachable blocks in the future, so it's not worth the effort.
                alternative_successor = first(alternative_successors)
                replace_unreachable(block, alternative_successor)
            end
        end
    end

    return changed
end

# replace calls to `trap` with inline assembly calling `exit`, which isn't fatal
function lower_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    ctx = context(mod)
    changed = false
    @timeit_debug to "lower trap" begin

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        # inline assembly to exit a thread
        exit_ft = LLVM.FunctionType(LLVM.VoidType(ctx))
        exit = InlineAsm(exit_ft, "exit;", "", true)

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                @dispose builder=IRBuilder(ctx) begin
                    position!(builder, val)
                    call!(builder, exit_ft, exit)
                end
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end
    end

    end
    return changed
end

# Replace occurrences of __nvvm_reflect("foo") and llvm.nvvm.reflect with an integer.
#
# NOTE: this is the same as LLVM's NVVMReflect pass, which we cannot use because it is
#       not exported. It is meant to be added to a pass pipeline automatically, by
#       calling adjustPassManager, but we don't use a PassManagerBuilder so cannot do so.
const NVVM_REFLECT_FUNCTION = "__nvvm_reflect"
function nvvm_reflect!(fun::LLVM.Function)
    job = current_job::CompilerJob
    mod = LLVM.parent(fun)
    ctx = context(fun)
    changed = false
    @timeit_debug to "nvvmreflect" begin

    # find and sanity check the nnvm-reflect function
    # TODO: also handle the llvm.nvvm.reflect intrinsic
    haskey(LLVM.functions(mod), NVVM_REFLECT_FUNCTION) || return false
    reflect_function = functions(mod)[NVVM_REFLECT_FUNCTION]
    isdeclaration(reflect_function) || error("_reflect function should not have a body")
    reflect_typ = return_type(function_type(reflect_function))
    isa(reflect_typ, LLVM.IntegerType) || error("_reflect's return type should be integer")

    to_remove = []
    for use in uses(reflect_function)
        call = user(use)
        isa(call, LLVM.CallInst) || continue
        length(operands(call)) == 2 || error("Wrong number of operands to __nvvm_reflect function")

        # decode the string argument
        str = operands(call)[1]
        isa(str, LLVM.ConstantExpr) || error("Format of __nvvm__reflect function not recognized")
        sym = operands(str)[1]
        if isa(sym, LLVM.ConstantExpr) && opcode(sym) == LLVM.API.LLVMGetElementPtr
            # CUDA 11.0 or below
            sym = operands(sym)[1]
        end
        isa(sym, LLVM.GlobalVariable) || error("Format of __nvvm__reflect function not recognized")
        sym_op = operands(sym)[1]
        isa(sym_op, LLVM.ConstantArray) || isa(sym_op, LLVM.ConstantDataArray) ||
            error("Format of __nvvm__reflect function not recognized")
        chars = convert.(Ref(UInt8), collect(sym_op))
        reflect_arg = String(chars[1:end-1])

        # handle possible cases
        # XXX: put some of these property in the compiler job?
        #      and/or first set the "nvvm-reflect-*" module flag like Clang does?
        fast_math = Base.JLOptions().fast_math == 1
        # NOTE: we follow nvcc's --use_fast_math
        reflect_val = if reflect_arg == "__CUDA_FTZ"
            # single-precision denormals support
            ConstantInt(reflect_typ, fast_math ? 1 : 0)
        elseif reflect_arg == "__CUDA_PREC_DIV"
            # single-precision floating-point division and reciprocals.
            ConstantInt(reflect_typ, fast_math ? 0 : 1)
        elseif reflect_arg == "__CUDA_PREC_SQRT"
            # single-precision denormals support
            ConstantInt(reflect_typ, fast_math ? 0 : 1)
        elseif reflect_arg == "__CUDA_FMAD"
            # contraction of floating-point multiplies and adds/subtracts into
            # floating-point multiply-add operations (FMAD, FFMA, or DFMA)
            ConstantInt(reflect_typ, fast_math ? 1 : 0)
        elseif reflect_arg == "__CUDA_ARCH"
            ConstantInt(reflect_typ, job.config.target.cap.major*100 + job.config.target.cap.minor*10)
        else
            @warn "Unknown __nvvm_reflect argument: $reflect_arg. Please file an issue."
        end

        replace_uses!(call, reflect_val)
        push!(to_remove, call)
    end

    # remove the calls to the function
    for val in to_remove
        @assert isempty(uses(val))
        unsafe_delete!(LLVM.parent(val), val)
    end

    # maybe also delete the function
    if isempty(uses(reflect_function))
        unsafe_delete!(mod, reflect_function)
    end

    end
    return changed
end
