@testsetup module Plugin

using Test
using ReTestItems
import LLVM
import GPUCompiler

function mark(x)
    ccall("extern gpucompiler.mark", llvmcall, Nothing, (Int,), x)
end

function remove_mark!(@nospecialize(job), intrinsic, mod::LLVM.Module)
    changed = false

    for use in LLVM.uses(intrinsic)
        val = LLVM.user(use)
        if isempty(LLVM.uses(val))
            LLVM.erase!(val)
            changed = true
        else
            # the validator will detect this
        end
    end

    return changed
end

GPUCompiler.register_plugin!("gpucompiler.mark", false, 
                             pipeline_callback=remove_mark!)

current_inlinestate() = nothing

abstract type InlineStateMeta end
struct AlwaysInlineMeta <: InlineStateMeta end
struct NeverInlineMeta <: InlineStateMeta end

import GPUCompiler: abstract_call_known, GPUInterpreter
import Core.Compiler: CallMeta, Effects, NoCallInfo, ArgInfo,
                      StmtInfo, AbsIntState, EFFECTS_TOTAL,
                      MethodResultPure, CallInfo, IRCode

function abstract_call_known(meta::InlineStateMeta, interp::GPUInterpreter, @nospecialize(f),
                             arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    (; fargs, argtypes) = arginfo

    if f === current_inlinestate
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CallMeta(Union{}, Effects(), NoCallInfo())
            else
                return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
            end
        end
        @static if VERSION < v"1.11.0-"
            return CallMeta(Core.Const(meta), EFFECTS_TOTAL, MethodResultPure())
        else
            return CallMeta(Core.Const(meta), Union{}, EFFECTS_TOTAL, MethodResultPure())
        end
    end
    return nothing
end

import GPUCompiler: inlining_handler, NoInlineCallInfo, AlwaysInlineCallInfo
function inlining_handler(meta::InlineStateMeta, interp::GPUInterpreter, @nospecialize(atype), callinfo)
    if meta isa NeverInlineMeta
        return NoInlineCallInfo(callinfo, atype, :default)
    elseif meta isa AlwaysInlineMeta
        return AlwaysInlineCallInfo(callinfo, atype)
    end
    return nothing
end

struct MockEnzymeMeta end

# Having to define this function is annoying
# introduce `abstract type InferenceMeta`
function inlining_handler(meta::MockEnzymeMeta, interp::GPUInterpreter, @nospecialize(atype), callinfo)
    return nothing
end

function autodiff end

import GPUCompiler: DeferredCallInfo
struct AutodiffCallInfo <: CallInfo
    rt
    info::DeferredCallInfo
end

function abstract_call_known(meta::Nothing, interp::GPUInterpreter, f::typeof(autodiff),
                             arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    (; fargs, argtypes) = arginfo

    @assert f === autodiff
    if length(argtypes) <= 1
        @static if VERSION < v"1.11.0-"
            return CallMeta(Union{}, Effects(), NoCallInfo())
        else
            return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
        end
    end

    other_fargs = fargs === nothing ? nothing : fargs[2:end]
    other_arginfo = ArgInfo(other_fargs, argtypes[2:end])
    # TODO: Ought we not change absint to use MockEnzymeMeta(), otherwise we fill the cache for nothing.
    call = Core.Compiler.abstract_call(interp, other_arginfo, si, sv, max_methods)
    callinfo = DeferredCallInfo(MockEnzymeMeta(), call.rt, call.info)

    # Real Enzyme must compute `rt` and `exct` according to enzyme semantics
    # and likely perform a unwrapping of fargs...
    rt = call.rt

    # TODO: Edges? Effects?
    @static if VERSION < v"1.11.0-"
        # Can't use call.effects since otherwise this call might be just replaced with rt
        return CallMeta(rt, Effects(), AutodiffCallInfo(rt, callinfo))
    else
        return CallMeta(rt, call.exct, Effects(), AutodiffCallInfo(rt, callinfo))
    end
end

function abstract_call_known(meta::MockEnzymeMeta, interp::GPUInterpreter, @nospecialize(f),
                             arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    return nothing
end

import Core.Compiler: insert_node!, NewInstruction, ReturnNode, Instruction, InliningState, Signature

# We really need a Compiler stdlib
Base.getindex(ir::IRCode, i) = Core.Compiler.getindex(ir, i)
Base.setindex!(inst::Instruction, val, i) = Core.Compiler.setindex!(inst, val, i)

const FlagType = VERSION >= v"1.11.0-" ? UInt32 : UInt8
function Core.Compiler.handle_call!(todo::Vector{Pair{Int,Any}}, ir::IRCode, stmt_idx::Int,
                         stmt::Expr, info::AutodiffCallInfo, flag::FlagType,
                         sig::Signature, state::InliningState)

    # Goal:
    # The IR we want to inline here is:
    # unpack the args ..
    # ptr = gpuc.deferred(MockEnzymeMeta(), f, primal_args...) 
    # ret = ccall("extern __autodiff", llvmcall, RT, Tuple{Ptr{Cvoid, args...}}, ptr, adjoint_args...)

    # 0. Obtain primal mi from DeferredCallInfo
    # TODO: remove this code duplication
    deferred_info = info.info
    minfo = deferred_info.info
    results = minfo.results
    if length(results.matches) != 1
        return nothing
    end
    match = only(results.matches)

    # lookup the target mi with correct edge tracking
    # TODO: Effects?
    case = Core.Compiler.compileable_specialization(
        match, Core.Compiler.Effects(), Core.Compiler.InliningEdgeTracker(state), info)
    @assert case isa Core.Compiler.InvokeCase
    @assert stmt.head === :call

    # Now create the IR we want to inline
    ir = Core.Compiler.IRCode() # contains a placeholder
    args = [Core.Compiler.Argument(i) for i in 2:length(stmt.args)] # f, args...
    idx = 0

    # 0. Enzyme proper: Desugar args 
    primal_args = args
    primal_argtypes = match.spec_types.parameters[2:end]

    adjoint_rt = info.rt
    adjoint_args = args # TODO
    adjoint_argtypes = primal_argtypes
    
    # 1: Since Julia's inliner goes bottom up we need to pretend that we inlined the deferred call
    expr = Expr(:foreigncall, 
        "extern gpuc.lookup",
        Ptr{Cvoid},
        Core.svec(#=meta=# Any, #=mi=# Any, #=f=# Any, primal_argtypes...), # Must use Any for MethodInstance or ftype
        0,
        QuoteNode(:llvmcall),
        deferred_info.meta,
        case.invoke,
        primal_args...
    )
    ptr = insert_node!(ir, (idx += 1), NewInstruction(expr, Ptr{Cvoid}))

    # 2. Call to magic `__autodiff`
    expr = Expr(:foreigncall,
        "extern __autodiff",
        adjoint_rt,
        Core.svec(Ptr{Cvoid}, Any, adjoint_argtypes...),
        0,
        QuoteNode(:llvmcall),
        ptr,
        adjoint_args...
    )
    ret = insert_node!(ir, idx, NewInstruction(expr, adjoint_rt))
    
    # Finally replace placeholder return
    ir[Core.SSAValue(1)][:inst] = Core.ReturnNode(ret)
    ir[Core.SSAValue(1)][:type] = Ptr{Cvoid}

    ir = Core.Compiler.compact!(ir)

    # which mi to use here?
    # push inlining todos
    # TODO: Effects
    # aviatesk mentioned using inlining_policy instead...
    itodo = Core.Compiler.InliningTodo(case.invoke, ir, Core.Compiler.Effects())
    @assert itodo.linear_inline_eligible
    push!(todo, (stmt_idx=>itodo))

    return nothing
end

function mock_enzyme!(@nospecialize(job), intrinsic, mod::LLVM.Module)
    changed = false

    for use in LLVM.uses(intrinsic)
        call = LLVM.user(use)
        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            ops = LLVM.operands(call)
            target = ops[1]
            if target isa LLVM.ConstantExpr && (LLVM.opcode(target) == LLVM.API.LLVMPtrToInt ||
                                                LLVM.opcode(target) == LLVM.API.LLVMBitCast)
                target = first(LLVM.operands(target))
            end
            funcT = LLVM.called_type(call)
            funcT = LLVM.FunctionType(LLVM.return_type(funcT), LLVM.parameters(funcT)[3:end])
            direct_call =  LLVM.call!(builder, funcT, target, ops[3:end - 1]) # why is the -1 necessary

            LLVM.replace_uses!(call, direct_call)
        end
        if isempty(LLVM.uses(call))
            LLVM.erase!(call)
            changed = true
        else
            # the validator will detect this
        end
    end

    return changed
end

GPUCompiler.register_plugin!("__autodiff", mock_enzyme!)

end #module