# This file is a part of Julia. License is MIT: https://julialang.org/license

# This is a forward port from https://github.com/JuliaLang/julia/pull/52964
module CCMixin

import Core.Compiler as CC
import .CC: NativeInterpreter, AbstractInterpreter, ArgInfo, StmtInfo, AbsIntState, CallMeta, Effects,
            get_max_methods, Const, method_table, MethodResultPure, CallInfo, singleton_type

@static if VERSION >= v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end

abstract type AbstractCompiler end
const CompilerInstance = Union{AbstractCompiler, Nothing}
const NativeCompiler = Nothing

# current_compiler() = ccall(:jl_get_current_task, Ref{Task}, ()).compiler::CompilerInstance

"""
    abstract_interpreter(::CompilerInstance, world::UInt)

Construct an appropriate abstract interpreter for the given compiler instance.
"""
function abstract_interpreter end

abstract_interpreter(::Nothing, world::UInt) = NativeInterpreter(world)

"""
    compiler_world(::CompilerInstance)

The compiler world to execute this compiler instance in.
"""
compiler_world(::Nothing) = unsafe_load(cglobal(:jl_typeinf_world, UInt))
compiler_world(::AbstractCompiler) = get_world_counter() # equivalent to invokelatest

struct WithinCallInfo <: CallInfo
    compiler::CompilerInstance
    info::CallInfo
end

function _call_within end


"""
    invoke_within(compiler, f, args...; kwargs...)

Call `f(args...; kwargs...)` within the compiler context provided by `compiler`.
"""
function invoke_within(compiler::CompilerInstance, @nospecialize(f), @nospecialize args...; kwargs...)
    kwargs = Base.merge(NamedTuple(), kwargs)
    if isempty(kwargs)
        return _call_within(compiler, f, args...)
    end
    return _call_within(compiler, Core.kwcall, kwargs, f, args...)
end

function abstract_call_within(interp::AbstractInterpreter, (; fargs, argtypes)::ArgInfo, si::StmtInfo,
                              sv::AbsIntState, max_methods::Int=get_max_methods(interp, sv))
    if length(argtypes) < 2
        @static if VERSION < v"1.11.0-"
            return CallMeta(Union{}, Effects(), NoCallInfo())
        else
            return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
        end
    end
    CT = argtypes[2]
    other_compiler = singleton_type(CT)
    if other_compiler === nothing
        if CT isa Const
            other_compiler = CT.val
        else
            # Compiler is not a singleton type result may depend on runtime configuration
            add_remark!(interp, sv, "Skipped call_within since compiler plugin not constant")
            return CallMeta(Any, Any, Effects(), NoCallInfo())
        end
    end
    # Change world to one where our methods exist.
    cworld = invokelatest(compiler_world, other_compiler)::UInt
    other_interp = Core._call_in_world(cworld, abstract_interpreter, other_compiler, get_inference_world(interp))
    other_fargs = fargs === nothing ? nothing : fargs[3:end]
    other_arginfo = ArgInfo(other_fargs, argtypes[3:end])
    call = Core._call_in_world(cworld, CC.abstract_call, other_interp, other_arginfo, si, sv, max_methods)
    # TODO: Edges? Effects?
    @static if VERSION < v"1.11.0-"
        return CallMeta(call.rt, call.effects, WithinCallInfo(other_compiler, call.info))
    else
        return CallMeta(call.rt, call.exct, call.effects, WithinCallInfo(other_compiler, call.info))
    end
end

Base.getindex(ir::CC.IRCode, idx::Core.SSAValue) = CC.getindex(ir, idx)
Base.setindex!(inst::CC.Instruction, val::UInt8, idx::Symbol) = CC.setindex!(inst, val, idx)

# allow inling of WithinCallInfo, why not
const FlagType = VERSION >= v"1.11.0-" ? UInt32 : UInt8
function CC.handle_call!(todo::Vector{Pair{Int,Any}}, ir::CC.IRCode, idx::CC.Int,
                         stmt::Expr, info::WithinCallInfo, flag::FlagType,
                         sig::CC.Signature, state::CC.InliningState)
    # I failed at inlining the call, codegen currently can't handle call_within so we have to
    # handle it ourselves.
    minfo = info.info
    if !(minfo isa CC.MethodMatchInfo)
        return nothing
    end
    results = minfo.results
    if length(results.matches) != 1
        return nothing
    end
    match = only(results.matches)

    # lookup the target mi with correct edge tracking
    # do we need to do this within the other compiler?
    case = CC.compileable_specialization(match, CC.Effects(), CC.InliningEdgeTracker(state), info)

    @assert case isa CC.InvokeCase
    @assert stmt.head === :call

    args = Any[
        "extern gpuc.call_within",
        ir[CC.SSAValue(idx)][:type],
        Core.svec(Any, Any, Any, match.spec_types.parameters[2:end]...), # Must use Any for MethodInstance or ftype
        0,
        QuoteNode(:llvmcall),
        info.compiler, # we could also use the compiler as passed in stmt.args[2]
        case.invoke,
        stmt.args[3:end]...
    ]
    stmt.head = :foreigncall
    stmt.args = args
    ir[CC.SSAValue(idx)][:flag] |= CC.flags_for_effects(case.effects)
    return nothing

    # info = info.info
    # @assert info.in isa CC.MethodMatchInfo
    # results = info.results
    # match = only(results.matches)
    # @show match
    # new_argtypes = sig.argtypes[3:end]
    # item = CC.analyze_method!(match, new_argtypes, info, flag, state; allow_typevars=false)
    # @assert item isa CC.InvokeCase
    # # handle_single_case inlined
    # stmt.head = :invoke
    # stmt.args = stmt.args[3:end]
    # pushfirst!(stmt.args, item.invoke)
    # ir[CC.SSAValue(idx)][:flag] |= CC.flags_for_effects(item.effects)
    # return nothing
    # @show todo
    # @show res
    # return res
    # @show match
    # error("")
    # ft = sig.argtypes[3]
    # f = singleton_type(ft)
    # if f === nothing
    #     if ft isa Const
    #         f = ft.val
    #     else
    #         error("")
    #         # # Compiler is not a singleton type result may depend on runtime configuration
    #         # add_remark!(interp, sv, "Skipped call_within since compiler plugin not constant")
    #         # return CallMeta(Any, Any, Effects(), NoCallInfo())
    #     end
    # end
    # new_sig = CC.Signature(f, CC.widenconst(ft), sig.argtypes[3:end])
    # stmt.args = stmt.args[3:end]
    # @show new_sig = CC.call_sig(ir, stmt)
    # # @show info.info
    # res = CC.handle_call!(todo, ir, idx, stmt, info.info, flag, new_sig, state)
    # @show res
    # @show todo
    # return res
    # # new_stmt = Expr(stmt.head, stmt.args[3:end])
    # @show stmt.head
    # if stmt.head === :invoke
    #     @show new_stmt
    #     res = CC.handle_invoke_expr!(todo, ir, idx, new_stmt, info.info, flag, new_sig, state)
    # else
    #     res = CC.handle_call!(todo, ir, idx, new_stmt, info.info, flag, new_sig, state)
    # end
    # @show res
    # return res
end

struct Edges
    edges::Vector{Tuple{CompilerInstance, CC.MethodInstance}}
end

function find_edges(ir::CC.IRCode)
    edges = Tuple{CompilerInstance, CC.MethodInstance}[]
    # XXX: can we add this instead in handle_call?
    for stmt in ir.stmts
        inst = stmt[:inst]
        inst isa Expr || continue
        expr = inst::Expr
        if expr.head === :foreigncall &&
            expr.args[1] == "extern gpuc.call_within"
            @show expr
            mi = expr.args[7]
            compiler = expr.args[6]
            push!(edges, (compiler, mi))
        end
    end
    unique!(edges)
    return edges
end

if VERSION >= v"1.11.0-"
function CC.ipo_dataflow_analysis!(interp::AbstractGPUInterpreter, ir::CC.IRCode,
                                    caller::CC.InferenceResult)
    edges = find_edges(ir)
    if !isempty(edges)
        CC.stack_analysis_result!(caller, Edges(edges))
    end
    @invoke CC.ipo_dataflow_analysis!(interp::CC.AbstractInterpreter, ir::CC.IRCode,
                                        caller::CC.InferenceResult)
end
else # v1.10
# 1.10 doesn't have stack_analysis_result or ipo_dataflow_analysis
function CC.finish(interp::AbstractGPUInterpreter, opt::CC.OptimizationState, ir::CC.IRCode,
                    caller::CC.InferenceResult)
    edges = find_edges(ir)
    if !isempty(edges)
        # HACK: we store the deferred edges in the argescapes field, which is invalid,
        #       but nobody should be running EA on our results.
        caller.argescapes = Edges(edges)
    end
    @invoke CC.finish(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
                        ir::CC.IRCode, caller::CC.InferenceResult)
end
end

function current_method_table end

function abstract_call_current_method_table(interp::AbstractInterpreter, (; fargs, argtypes)::ArgInfo, si::StmtInfo,
                                            sv::AbsIntState, max_methods::Int=get_max_methods(interp, sv))
    if length(argtypes) != 1
        @static if VERSION < v"1.11.0-"
            return CallMeta(Union{}, Effects(), NoCallInfo())
        else
            return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
        end
    end
    mt = Const(method_table(interp))
    @static if VERSION < v"1.11.0-"
        return CallMeta(mt, CC.EFFECTS_TOTAL, MethodResultPure())
    else
        return CallMeta(mt, Union{}, CC.EFFECTS_TOTAL, MethodResultPure())
    end
end

    

abstract type AbstractGPUInterpreter <: AbstractInterpreter end

function CC.abstract_call_known(interp::AbstractGPUInterpreter, @nospecialize(f),
            arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState,
            max_methods::Int = CC.get_max_methods(interp, f, sv))
    (; fargs, argtypes) = arginfo
    if f === _call_within
        return abstract_call_within(interp, arginfo, si, sv, max_methods)
    elseif f === current_method_table
        return abstract_call_current_method_table(interp, arginfo, si, sv, max_methods)
    end
    return @invoke CC.abstract_call_known(interp::AbstractInterpreter, f,
                                          arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState,
                                          max_methods::Int)
    end
end

