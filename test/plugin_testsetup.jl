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
                      MethodResultPure

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


end