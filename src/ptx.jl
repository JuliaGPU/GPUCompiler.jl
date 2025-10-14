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

    fastmath::Bool = Base.JLOptions().fast_math == 1

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
    h = hash(target.fastmath, h)

    h
end

source_code(target::PTXCompilerTarget) = "ptx"

llvm_triple(target::PTXCompilerTarget) = Int===Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda"

function llvm_machine(target::PTXCompilerTarget)
    @static if :NVPTX ∉ LLVM.backends()
        return nothing
    end
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

dwarf_version(target::PTXCompilerTarget) = Int32(2) # Cuda only supports dwarfv2

can_vectorize(job::CompilerJob{PTXCompilerTarget}) = true

## job

function Base.show(io::IO, @nospecialize(job::CompilerJob{PTXCompilerTarget}))
    print(io, "PTX CompilerJob of ", job.source)
    print(io, " for sm_$(job.config.target.cap.major)$(job.config.target.cap.minor)")

    job.config.target.minthreads !== nothing && print(io, ", minthreads=$(job.config.target.minthreads)")
    job.config.target.maxthreads !== nothing && print(io, ", maxthreads=$(job.config.target.maxthreads)")
    job.config.target.blocks_per_sm !== nothing && print(io, ", blocks_per_sm=$(job.config.target.blocks_per_sm)")
    job.config.target.maxregs !== nothing && print(io, ", maxregs=$(job.config.target.maxregs)")
    job.config.target.fastmath && print(io, ", fast math enabled")
end

const ptx_intrinsics = ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(@nospecialize(job::CompilerJob{PTXCompilerTarget}), fn::String) =
    in(fn, ptx_intrinsics)

# XXX: the debuginfo part should be handled by GPUCompiler as it applies to all back-ends.
runtime_slug(@nospecialize(job::CompilerJob{PTXCompilerTarget})) =
    "ptx$(job.config.target.ptx.major)$(job.config.target.ptx.minor)" *
    "-sm_$(job.config.target.cap.major)$(job.config.target.cap.minor)" *
    "-debuginfo=$(Int(llvm_debug_info(job)))"

function finish_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                        mod::LLVM.Module, entry::LLVM.Function)
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
            initializer!(gv, ConstantInt(LLVM.Int32Type(), value))
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

    # we emit properties (of the device and ptx isa) as private global constants,
    # so run the optimizer so that they are inlined before the rest of the optimizer runs.
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        run!(pb, mod, llvm_machine(job.config.target))
    end

    return entry
end

function optimize_module!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                          mod::LLVM.Module)
    tm = llvm_machine(job.config.target)
    # TODO: Use the registered target passes (JuliaGPU/GPUCompiler.jl#450)
    @dispose pb=NewPMPassBuilder() begin
        register!(pb, NVVMReflectPass())

        add!(pb, NewPMFunctionPassManager()) do fpm
            # TODO: need to run this earlier; optimize_module! is called after addOptimizationPasses!
            add!(fpm, NVVMReflectPass())

            # needed by GemmKernels.jl-like code
            add!(fpm, SpeculativeExecutionPass())

            # NVPTX's target machine info enables runtime unrolling,
            # but Julia's pass sequence only invokes the simple unroller.
            add!(fpm, LoopUnrollPass(; job.config.opt_level))
            add!(fpm, InstCombinePass())        # clean-up redundancy
            add!(fpm, NewPMLoopPassManager(; use_memory_ssa=true)) do lpm
                add!(lpm, LICMPass())           # the inner runtime check might be
                                                # outer loop invariant
            end

            # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
            # that code still needs to be optimized (arguably, multiple unroll passes should be
            # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
            if job.config.opt_level == 2
                add!(fpm, GVNPass())
            elseif job.config.opt_level == 1
                add!(fpm, EarlyCSEPass())
            end
            add!(fpm, DSEPass())

            add!(fpm, SimplifyCFGPass())
        end

        # get rid of the internalized functions; now possible unused
        add!(pb, GlobalDCEPass())

        run!(pb, mod, tm)
    end
end

function finish_ir!(@nospecialize(job::CompilerJob{PTXCompilerTarget}),
                    mod::LLVM.Module, entry::LLVM.Function)
    if LLVM.version() < v"17"
        for f in functions(mod)
            lower_unreachable!(f)
        end
    end

    if job.config.kernel
        # add metadata annotations for the assembler to the module

        # property annotations
        annotations = Metadata[entry]

        ## kernel metadata
        append!(annotations, [MDString("kernel"),
                              ConstantInt(Int32(1))])

        ## expected CTA sizes
        if job.config.target.minthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.config.target.minthreads) ? job.config.target.minthreads[dim] : 1
                append!(annotations, [MDString("reqntid$name"),
                                      ConstantInt(Int32(bound))])
            end
        end
        if job.config.target.maxthreads !== nothing
            for (dim, name) in enumerate([:x, :y, :z])
                bound = dim <= length(job.config.target.maxthreads) ? job.config.target.maxthreads[dim] : 1
                append!(annotations, [MDString("maxntid$name"),
                                      ConstantInt(Int32(bound))])
            end
        end

        if job.config.target.blocks_per_sm !== nothing
            append!(annotations, [MDString("minctasm"),
                                  ConstantInt(Int32(job.config.target.blocks_per_sm))])
        end

        if job.config.target.maxregs !== nothing
            append!(annotations, [MDString("maxnreg"),
                                  ConstantInt(Int32(job.config.target.maxregs))])
        end

        push!(metadata(mod)["nvvm.annotations"], MDNode(annotations))
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

# lower `unreachable` to `exit` so that the emitted PTX has correct control flow
#
# During back-end compilation, `ptxas` inserts instructions to manage the harware's
# reconvergence stack (SSY and SYNC). In order to do so, it needs to identify
# divergent regions:
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
# Meanwhile, LLVM's branch-folder and block-placement MIR passes will try to optimize
# the block layout, e.g., by placing unlikely blocks at the end of the function:
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
# That is not a problem as long as the unlikely block continunes back into the
# divergent region. Crucially, this is not the case with unreachable control flow:
#
#   entry:
#     // start of divergent region
#     @%p0 bra cont;
#     @%p1 bra throw;
#     bra.uni cont;
#   cont:
#     bar.sync 0;
#   throw:
#     call throw_and_trap();
#     // unreachable
#   exit:
#     // end of divergent region
#     ret;
#
# Dynamically, this is fine, because the called function does not return.
# However, `ptxas` does not know that and adds a successor edge to the `exit`
# block, widening the divergence range. In this example, that's not allowed, as
# `bar.sync` cannot be executed divergently on Pascal hardware or earlier.
#
# To avoid these fall-through successors that change the control flow,
# we replace `unreachable` instructions with a call to `trap` and `exit`. This
# informs `ptxas` that the thread exits, and allows it to correctly construct a
# CFG, and consequently correctly determine the divergence regions as intended.
# Note that we first emit a call to `trap`, so that the behaviour is the same
# as before.
function lower_unreachable!(f::LLVM.Function)
    mod = LLVM.parent(f)

    # TODO:
    # - if unreachable blocks have been merged, we still may be jumping from different
    #   divergent regions, potentially causing the same problem as above:
    #     entry:
    #       // start of divergent region 1
    #       @%p0 bra cont1;
    #       @%p1 bra throw;
    #       bra.uni cont1;
    #     cont1:
    #       // end of divergent region 1
    #       bar.sync 0;   // is this executed divergently?
    #       // start of divergent region 2
    #       @%p2 bra cont2;
    #       @%p3 bra throw;
    #       bra.uni cont2;
    #     cont2:
    #       // end of divergent region 2
    #       ...
    #     throw:
    #       trap;
    #       br throw;
    #   if this is a problem, we probably need to clone blocks with multiple
    #   predecessors so that there's a unique path from each region of
    #   divergence to every `unreachable` terminator

    # remove `noreturn` attributes, to avoid the (minimal) optimization that
    # happens during `prepare_execution!` undoing our work here.
    # this shouldn't be needed when we upstream the pass.
    attrs = function_attributes(f)
    delete!(attrs, EnumAttribute("noreturn", 0))

    # find unreachable blocks
    unreachable_blocks = BasicBlock[]
    for block in blocks(f)
        if terminator(block) isa LLVM.UnreachableInst
            push!(unreachable_blocks, block)
        end
    end
    isempty(unreachable_blocks) && return false

    # inline assembly to exit a thread
    exit_ft = LLVM.FunctionType(LLVM.VoidType())
    exit = InlineAsm(exit_ft, "exit;", "", true)
    trap_ft = LLVM.FunctionType(LLVM.VoidType())
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", trap_ft)
    end

    # rewrite the unreachable terminators
    @dispose builder=IRBuilder() begin
        entry_block = first(blocks(f))
        for block in unreachable_blocks
            inst = terminator(block)
            @assert inst isa LLVM.UnreachableInst

            position!(builder, inst)
            call!(builder, trap_ft, trap)
            call!(builder, exit_ft, exit)
        end
    end

    return true
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
    changed = false
    @tracepoint "nvvmreflect" begin

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
        if length(operands(call)) != 2
            @error """Unrecognized format of __nvvm_reflect call:
                      $(string(call))
                      Wrong number of operands: expected 2, got $(length(operands(call)))."""
            continue
        end

        # decode the string argument
        if LLVM.version() >= v"17"
            sym = operands(call)[1]
        else
            str = operands(call)[1]
            if !isa(str, LLVM.ConstantExpr) || opcode(str) != LLVM.API.LLVMGetElementPtr
                @safe_error """Unrecognized format of __nvvm_reflect call:
                               $(string(call))
                               Operand should be a GEP instruction, got a $(typeof(str)). Please file an issue."""
                continue
            end
            sym = operands(str)[1]
            if isa(sym, LLVM.ConstantExpr) && opcode(sym) == LLVM.API.LLVMGetElementPtr
                # CUDA 11.0 or below
                sym = operands(sym)[1]
            end
        end
        if !isa(sym, LLVM.GlobalVariable)
            @safe_error """Unrecognized format of __nvvm_reflect call:
                           $(string(call))
                           Operand should be a global variable, got a $(typeof(sym)). Please file an issue."""
            continue
        end
        sym_op = operands(sym)[1]
        if !isa(sym_op, LLVM.ConstantArray) && !isa(sym_op, LLVM.ConstantDataArray)
            @safe_error """Unrecognized format of __nvvm_reflect call:
                           $(string(call))
                           Operand should be a constant array, got a $(typeof(sym_op)). Please file an issue."""
        end
        chars = convert.(Ref(UInt8), collect(sym_op))
        reflect_arg = String(chars[1:end-1])

        # handle possible cases
        # XXX: put some of these property in the compiler job?
        #      and/or first set the "nvvm-reflect-*" module flag like Clang does?
        fast_math = current_job.config.target.fastmath
        # NOTE: we follow nvcc's --use_fast_math
        reflect_val = if reflect_arg == "__CUDA_FTZ"
            # single-precision denormals support
            ConstantInt(reflect_typ, fast_math ? 1 : 0)
        elseif reflect_arg == "__CUDA_PREC_DIV"
            # single-precision floating-point division and reciprocals.
            ConstantInt(reflect_typ, fast_math ? 0 : 1)
        elseif reflect_arg == "__CUDA_PREC_SQRT"
            # single-precision floating point square roots.
            ConstantInt(reflect_typ, fast_math ? 0 : 1)
        elseif reflect_arg == "__CUDA_FMAD"
            # contraction of floating-point multiplies and adds/subtracts into
            # floating-point multiply-add operations (FMAD, FFMA, or DFMA)
            ConstantInt(reflect_typ, fast_math ? 1 : 0)
        elseif reflect_arg == "__CUDA_ARCH"
            ConstantInt(reflect_typ, job.config.target.cap.major*100 + job.config.target.cap.minor*10)
        else
            @safe_error """Unrecognized format of __nvvm_reflect call:
                           $(string(call))
                           Unknown argument $reflect_arg. Please file an issue."""
            continue
        end

        replace_uses!(call, reflect_val)
        push!(to_remove, call)
    end

    # remove the calls to the function
    for val in to_remove
        @assert isempty(uses(val))
        erase!(val)
    end

    # maybe also delete the function
    if isempty(uses(reflect_function))
        erase!(reflect_function)
    end

    end
    return changed
end
NVVMReflectPass() = NewPMFunctionPass("custom-nvvm-reflect", nvvm_reflect!)
