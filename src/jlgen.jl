# Julia compiler integration

## world age lookups

# `tls_world_age` should be used to look up the current world age. in most cases, this is
# what you should use to invoke the compiler with.

if isdefined(Base, :tls_world_age)
    import Base: tls_world_age
else
    tls_world_age() = ccall(:jl_get_tls_world_age, UInt, ())
end


## looking up method instances

export methodinstance, generic_methodinstance

@inline function signature_type_by_tt(ft::Type, tt::Type)
    u = Base.unwrap_unionall(tt)::DataType
    return Base.rewrap_unionall(Tuple{ft, u.parameters...}, tt)
end

# create a MethodError from a function type
# TODO: fix upstream
function unsafe_function_from_type(ft::Type)
    if isdefined(ft, :instance)
        ft.instance
    else
        # HACK: dealing with a closure or something... let's do somthing really invalid,
        #       which works because MethodError doesn't actually use the function
        Ref{ft}()[]
    end
end
function MethodError(ft::Type{<:Function}, tt::Type, world::Integer=typemax(UInt))
    Base.MethodError(unsafe_function_from_type(ft), tt, world)
end
MethodError(ft, tt, world=typemax(UInt)) = Base.MethodError(ft, tt, world)

# generate a LineInfoNode for the current source code location
macro LineInfoNode(method)
    Core.LineInfoNode(__module__, method, __source__.file, Int32(__source__.line), Int32(0))
end

"""
    methodinstance(ft::Type, tt::Type, [world::UInt])

Look up the method instance that corresponds to invoking the function with type `ft` with
argument typed `tt`. If the `world` argument is specified, the look-up is static and will
always return the same result. If the `world` argument is not specified, the look-up is
dynamic and the returned method instance will depende on the current world age. If no method
is found, a `MethodError` is thrown.

This function is highly optimized, and results do not need to be cached additionally.

Only use this function with concrete signatures, i.e., using the types of values you would
pass at run time. For non-concrete signatures, use `generic_methodinstance` instead.

"""
methodinstance

function generic_methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                                world::Integer=tls_world_age())
    sig = signature_type_by_tt(ft, tt)

    match, _ = CC._findsup(sig, nothing, world)
    match === nothing && throw(MethodError(ft, tt, world))

    mi = CC.specialize_method(match)

    return mi::MethodInstance
end

# on 1.11 (JuliaLang/julia#52572, merged as part of JuliaLang/julia#52233) we can use
# Julia's cached method lookup to simply look up method instances at run time.
if VERSION >= v"1.11.0-DEV.1552"

# XXX: version of Base.method_instance that uses a function type
@inline function methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                                world::Integer=tls_world_age())
    sig = signature_type_by_tt(ft, tt)
    @assert Base.isdispatchtuple(sig)   # JuliaLang/julia#52233

    mi = ccall(:jl_method_lookup_by_tt, Any,
               (Any, Csize_t, Any),
               sig, world, #=method_table=# nothing)
    mi === nothing && throw(MethodError(ft, tt, world))
    mi = mi::MethodInstance

    # `jl_method_lookup_by_tt` and `jl_method_lookup` can return a unspecialized mi
    if !Base.isdispatchtuple(mi.specTypes)
        mi = CC.specialize_method(mi.def, sig, mi.sparam_vals)::MethodInstance
    end

    return mi
end

# on older versions of Julia, we always need to use the generic lookup
else

const methodinstance = generic_methodinstance

function methodinstance_generator(world::UInt, source, self, ft::Type, tt::Type)
    @nospecialize
    @assert CC.isType(ft) && CC.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :ft, :tt), Core.svec())

    # look up the method match
    method_error = :(throw(MethodError(ft, tt, $world)))
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    match = ccall(:jl_gf_invoke_lookup_worlds, Any,
                  (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                  sig, #=mt=# nothing, world, min_world, max_world)
    match === nothing && return stub(world, source, method_error)

    # look up the method and code instance
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
    ci = CC.retrieve_code_info(mi, world)

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    empty!(new_ci.linetable)
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0

    # propagate edge metadata
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = MethodInstance[mi]

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]

    # return the method instance
    push!(new_ci.code, CC.ReturnNode(mi))
    push!(new_ci.ssaflags, 0x00)
    push!(new_ci.linetable, @LineInfoNode(methodinstance))
    push!(new_ci.codelocs, 1)
    new_ci.ssavaluetypes += 1

    return new_ci
end

@eval function methodinstance(ft, tt)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, methodinstance_generator))
end

end


## code instance cache

const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

if !HAS_INTEGRATED_CACHE
struct CodeCache
    dict::IdDict{MethodInstance,Vector{CodeInstance}}

    CodeCache() = new(IdDict{MethodInstance,Vector{CodeInstance}}())
end

function Base.show(io::IO, ::MIME"text/plain", cc::CodeCache)
    print(io, "CodeCache with $(mapreduce(length, +, values(cc.dict); init=0)) entries")
    if !isempty(cc.dict)
        print(io, ": ")
        for (mi, cis) in cc.dict
            println(io)
            print(io, "  ")
            show(io, mi)

            function worldstr(min_world, max_world)
                if min_world == typemax(UInt)
                    "empty world range"
                elseif max_world == typemax(UInt)
                    "worlds $(Int(min_world))+"
                else
                    "worlds $(Int(min_world)) to $(Int(max_world))"
                end
            end

            for (i,ci) in enumerate(cis)
                println(io)
                print(io, "    CodeInstance for ", worldstr(ci.min_world, ci.max_world))
            end
        end
    end
end

Base.empty!(cc::CodeCache) = empty!(cc.dict)

const GLOBAL_CI_CACHES = Dict{CompilerConfig, CodeCache}()
const GLOBAL_CI_CACHES_LOCK = ReentrantLock()


## method invalidations

function CC.setindex!(cache::CodeCache, ci::CodeInstance, mi::MethodInstance)
    # make sure the invalidation callback is attached to the method instance
    add_codecache_callback!(cache, mi)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

# invalidation (like invalidate_method_instance, but for our cache)
struct CodeCacheCallback
    cache::CodeCache
end

@static if VERSION ≥ v"1.11.0-DEV.798"

function add_codecache_callback!(cache::CodeCache, mi::MethodInstance)
    callback = CodeCacheCallback(cache)
    CC.add_invalidation_callback!(callback, mi)
end
function (callback::CodeCacheCallback)(replaced::MethodInstance, max_world::UInt32)
    cis = get(callback.cache.dict, replaced, nothing)
    if cis === nothing
        return
    end
    for ci in cis
        if ci.max_world == ~0 % Csize_t
            @assert ci.min_world - 1 <= max_world "attempting to set illogical constraints"
@static if VERSION >= v"1.11.0-DEV.1390"
            @atomic ci.max_world = max_world
else
            ci.max_world = max_world
end
        end
        @assert ci.max_world <= max_world
    end
end

else

function add_codecache_callback!(cache::CodeCache, mi::MethodInstance)
    callback = CodeCacheCallback(cache)
    if !isdefined(mi, :callbacks)
        mi.callbacks = Any[callback]
    elseif !in(callback, mi.callbacks)
        push!(mi.callbacks, callback)
    end
end
function (callback::CodeCacheCallback)(replaced::MethodInstance, max_world::UInt32,
                                       seen::Set{MethodInstance}=Set{MethodInstance}())
    push!(seen, replaced)

    cis = get(callback.cache.dict, replaced, nothing)
    if cis === nothing
        return
    end
    for ci in cis
        if ci.max_world == ~0 % Csize_t
            @assert ci.min_world - 1 <= max_world "attempting to set illogical constraints"
            ci.max_world = max_world
        end
        @assert ci.max_world <= max_world
    end

    # recurse to all backedges to update their valid range also
    if isdefined(replaced, :backedges)
        backedges = filter(replaced.backedges) do @nospecialize(mi)
            if mi isa MethodInstance
                mi ∉ seen
            elseif mi isa Type
                # an `invoke` call, which is a `(sig, MethodInstance)` pair.
                # let's ignore the `sig` and process the `MethodInstance` next.
                false
            else
                error("invalid backedge")
            end
        end

        # Don't touch/empty backedges `invalidate_method_instance` in C will do that later
        # replaced.backedges = Any[]

        for mi in backedges
            callback(mi::MethodInstance, max_world, seen)
        end
    end
end

end
end # !HAS_INTEGRATED_CACHE


## method overrides

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)


## interpreter

@static if VERSION >= v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end

using Core.Compiler: OverlayMethodTable
const MTType = Core.MethodTable
if isdefined(Core.Compiler, :CachedMethodTable)
    using Core.Compiler: CachedMethodTable
    const GPUMethodTableView = CachedMethodTable{OverlayMethodTable}
    get_method_table_view(world::UInt, mt::MTType) =
        CachedMethodTable(OverlayMethodTable(world, mt))
else
    const GPUMethodTableView = OverlayMethodTable
    get_method_table_view(world::UInt, mt::MTType) = OverlayMethodTable(world, mt)
end

struct GPUInterpreter <: CC.AbstractInterpreter
    meta::Any
    world::UInt
    method_table::GPUMethodTableView

@static if HAS_INTEGRATED_CACHE
    token::Any
else
    code_cache::CodeCache
end
    inf_cache::Vector{CC.InferenceResult}

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

@static if HAS_INTEGRATED_CACHE
function GPUInterpreter(world::UInt=Base.get_world_counter();
                        meta = nothing,
                        method_table::MTType,
                        token::Any,
                        inf_params::CC.InferenceParams,
                        opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    method_table = get_method_table_view(world, method_table)
    inf_cache = Vector{CC.InferenceResult}()

    return GPUInterpreter(meta, world, method_table,
                          token, inf_cache,
                          inf_params, opt_params)
end

function GPUInterpreter(interp::GPUInterpreter;
                        meta=interp.meta,
                        world::UInt=interp.world,
                        method_table::GPUMethodTableView=interp.method_table,
                        token::Any=interp.token,
                        inf_cache::Vector{CC.InferenceResult}=interp.inf_cache,
                        inf_params::CC.InferenceParams=interp.inf_params,
                        opt_params::CC.OptimizationParams=interp.opt_params)
    return GPUInterpreter(meta, world, method_table,
                          token, inf_cache,
                          inf_params, opt_params)
end

else

function GPUInterpreter(world::UInt=Base.get_world_counter();
                        meta=nothing,
                        method_table::MTType,
                        code_cache::CodeCache,
                        inf_params::CC.InferenceParams,
                        opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    method_table = get_method_table_view(world, method_table)
    inf_cache = Vector{CC.InferenceResult}()

    return GPUInterpreter(meta, world, method_table,
                          code_cache, inf_cache,
                          inf_params, opt_params)
end

function GPUInterpreter(interp::GPUInterpreter;
                        meta=interp.meta,
                        world::UInt=interp.world,
                        method_table::GPUMethodTableView=interp.method_table,
                        code_cache::CodeCache=interp.code_cache,
                        inf_cache::Vector{CC.InferenceResult}=interp.inf_cache,
                        inf_params::CC.InferenceParams=interp.inf_params,
                        opt_params::CC.OptimizationParams=interp.opt_params)
    return GPUInterpreter(meta, world, method_table,
                          code_cache, inf_cache,
                          inf_params, opt_params)
end
end # HAS_INTEGRATED_CACHE

CC.InferenceParams(interp::GPUInterpreter) = interp.inf_params
CC.OptimizationParams(interp::GPUInterpreter) = interp.opt_params
#=CC.=#get_inference_world(interp::GPUInterpreter) = interp.world
CC.get_inference_cache(interp::GPUInterpreter) = interp.inf_cache
if HAS_INTEGRATED_CACHE
    CC.cache_owner(interp::GPUInterpreter) = interp.token
else
    CC.code_cache(interp::GPUInterpreter) = WorldView(interp.code_cache, interp.world)
end

# No need to do any locking since we're not putting our results into the runtime cache
CC.lock_mi_inference(interp::GPUInterpreter, mi::MethodInstance) = nothing
CC.unlock_mi_inference(interp::GPUInterpreter, mi::MethodInstance) = nothing

function CC.add_remark!(interp::GPUInterpreter, sv::CC.InferenceState, msg)
    @safe_debug "Inference remark during GPU compilation of $(sv.linfo): $msg"
end

CC.may_optimize(interp::GPUInterpreter) = true
CC.may_compress(interp::GPUInterpreter) = true
CC.may_discard_trees(interp::GPUInterpreter) = true
CC.verbose_stmt_info(interp::GPUInterpreter) = false
CC.method_table(interp::GPUInterpreter) = interp.method_table

# semi-concrete interepretation is broken with overlays (JuliaLang/julia#47349)
function CC.concrete_eval_eligible(interp::GPUInterpreter,
    @nospecialize(f), result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    # NOTE it's fine to skip overloading with `sv::IRInterpretationState` since we disables
    #      semi-concrete interpretation anyway.
    ret = @invoke CC.concrete_eval_eligible(interp::CC.AbstractInterpreter,
        f::Any, result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    if ret === :semi_concrete_eval
        return :none
    end
    return ret
end
function CC.concrete_eval_eligible(interp::GPUInterpreter,
    @nospecialize(f), result::CC.MethodCallResult, arginfo::CC.ArgInfo)
    ret = @invoke CC.concrete_eval_eligible(interp::CC.AbstractInterpreter,
        f::Any, result::CC.MethodCallResult, arginfo::CC.ArgInfo)
    ret === false && return nothing
    return ret
end


within_gpucompiler() = false

## deferred compilation

struct DeferredCallInfo <: CC.CallInfo
    meta::Any
    rt::DataType
    info::CC.CallInfo
end

# recognize calls to gpuc.deferred and save DeferredCallInfo metadata
# default implementation, extensible through meta argument. 
# XXX: (or should we dispatch on `f`)?
function abstract_call_known(meta::Nothing, interp::GPUInterpreter, @nospecialize(f),
                             arginfo::CC.ArgInfo, si::CC.StmtInfo, sv::CC.AbsIntState,
                             max_methods::Int = CC.get_max_methods(interp, f, sv))
    (; fargs, argtypes) = arginfo
    if f === var"gpuc.deferred"
        argvec = argtypes[3:end]
        call = CC.abstract_call(interp, CC.ArgInfo(nothing, argvec), si, sv, max_methods)
        metaT = argtypes[2]
        meta = CC.singleton_type(metaT)
        if meta === nothing
            if metaT isa Core.Const
                meta = metaT.val
            else
                # meta is not a singleton type result may depend on runtime configuration
                add_remark!(interp, sv, "Skipped gpuc.deferred since meta not constant")
                @static if VERSION < v"1.11.0-"
                    return CC.CallMeta(Union{}, CC.Effects(), CC.NoCallInfo())
                else
                    return CC.CallMeta(Union{}, Union{}, CC.Effects(), CC.NoCallInfo())
                end
            end
        end

        callinfo = DeferredCallInfo(meta, call.rt, call.info)
        @static if VERSION < v"1.11.0-"
            return CC.CallMeta(Ptr{Cvoid}, CC.Effects(), callinfo)
        else
            return CC.CallMeta(Ptr{Cvoid}, Union{}, CC.Effects(), callinfo)
        end
    elseif f === within_gpucompiler
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CC.CallMeta(Union{}, CC.Effects(), CC.NoCallInfo())
            else
                return CC.CallMeta(Union{}, Union{}, CC.Effects(), CC.NoCallInfo())
            end
        end
        @static if VERSION < v"1.11.0-"
            return CC.CallMeta(Core.Const(true), CC.EFFECTS_TOTAL, CC.MethodResultPure())
        else
            return CC.CallMeta(Core.Const(true), Union{}, CC.EFFECTS_TOTAL, CC.MethodResultPure(),)
        end
    end
    return nothing
end

function CC.abstract_call_known(interp::GPUInterpreter, @nospecialize(f),
                                arginfo::CC.ArgInfo, si::CC.StmtInfo, sv::CC.AbsIntState,
                                max_methods::Int = CC.get_max_methods(interp, f, sv))
    candidate = abstract_call_known(interp.meta, interp, f, arginfo, si, sv, max_methods)
    if candidate === nothing && interp.meta !== nothing
        candidate = abstract_call_known(interp.meta, interp, f, arginfo, si, sv, max_methods)
    end
    if candidate !== nothing
        return candidate
    end
    
    return @invoke CC.abstract_call_known(interp::CC.AbstractInterpreter, f,
        arginfo::CC.ArgInfo, si::CC.StmtInfo, sv::CC.AbsIntState,
        max_methods::Int)
end

# during inlining, refine deferred calls to gpuc.lookup foreigncalls
const FlagType = VERSION >= v"1.11.0-" ? UInt32 : UInt8
function CC.handle_call!(todo::Vector{Pair{Int,Any}}, ir::CC.IRCode, idx::CC.Int,
                         stmt::Expr, info::DeferredCallInfo, flag::FlagType,
                         sig::CC.Signature, state::CC.InliningState)
    minfo = info.info
    results = minfo.results
    if length(results.matches) != 1
        return nothing
    end
    match = only(results.matches)

    # lookup the target mi with correct edge tracking
    case = CC.compileable_specialization(match, CC.Effects(), CC.InliningEdgeTracker(state),
                                         info)
    @assert case isa CC.InvokeCase
    @assert stmt.head === :call

    args = Any[
        "extern gpuc.lookup",
        Ptr{Cvoid},
        Core.svec(Any, Any, Any, match.spec_types.parameters[2:end]...), # Must use Any for MethodInstance or ftype
        0,
        QuoteNode(:llvmcall),
        info.meta,
        case.invoke,
        stmt.args[3:end]...
    ]
    stmt.head = :foreigncall
    stmt.args = args
    return nothing
end

struct Edge
    meta::Any
    mi::MethodInstance
end

struct DeferredEdges
    edges::Vector{Edge}
end

function find_deferred_edges(ir::CC.IRCode)
    edges = Edge[]
    # XXX: can we add this instead in handle_call?
    for stmt in ir.stmts
        inst = stmt[:inst]
        inst isa Expr || continue
        expr = inst::Expr
        if expr.head === :foreigncall &&
            expr.args[1] == "extern gpuc.lookup"
            deferred_meta = expr.args[6]
            deferred_mi = expr.args[7]
            push!(edges, Edge(deferred_meta, deferred_mi))
        end
    end
    unique!(edges)
    return edges
end

if VERSION >= v"1.11.0-"
function CC.ipo_dataflow_analysis!(interp::GPUInterpreter, ir::CC.IRCode,
                                   caller::CC.InferenceResult)
    edges = find_deferred_edges(ir)
    if !isempty(edges)
        CC.stack_analysis_result!(caller, DeferredEdges(edges))
    end
    @invoke CC.ipo_dataflow_analysis!(interp::CC.AbstractInterpreter, ir::CC.IRCode,
                                      caller::CC.InferenceResult)
end
else # v1.10
# 1.10 doesn't have stack_analysis_result or ipo_dataflow_analysis
function CC.finish(interp::GPUInterpreter, opt::CC.OptimizationState, ir::CC.IRCode,
                   caller::CC.InferenceResult)
    edges = find_deferred_edges(ir)
    if !isempty(edges)
        # HACK: we store the deferred edges in the argescapes field, which is invalid,
        #       but nobody should be running EA on our results.
        caller.argescapes = DeferredEdges(edges)
    end
    @invoke CC.finish(interp::CC.AbstractInterpreter, opt::CC.OptimizationState,
                      ir::CC.IRCode, caller::CC.InferenceResult)
end
end

import .CC: CallInfo
struct NoInlineCallInfo <: CallInfo
    info::CallInfo # wrapped call
    tt::Any # ::Type
    kind::Symbol
    NoInlineCallInfo(@nospecialize(info::CallInfo), @nospecialize(tt), kind::Symbol) =
        new(info, tt, kind)
end

CC.nsplit_impl(info::NoInlineCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::NoInlineCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::NoInlineCallInfo, idx::Int) = CC.getresult(info.info, idx)
struct AlwaysInlineCallInfo <: CallInfo
    info::CallInfo # wrapped call
    tt::Any # ::Type
    AlwaysInlineCallInfo(@nospecialize(info::CallInfo), @nospecialize(tt)) = new(info, tt)
end

CC.nsplit_impl(info::AlwaysInlineCallInfo) = Core.Compiler.nsplit(info.info)
CC.getsplit_impl(info::AlwaysInlineCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::AlwaysInlineCallInfo, idx::Int) = CC.getresult(info.info, idx)


function inlining_handler(meta::Nothing, interp::GPUInterpreter, @nospecialize(atype), callinfo)
    return nothing
end

using Core.Compiler: ArgInfo, StmtInfo, AbsIntState
function CC.abstract_call_gf_by_type(interp::GPUInterpreter, @nospecialize(f), arginfo::ArgInfo,
                                     si::StmtInfo, @nospecialize(atype), sv::AbsIntState, max_methods::Int)
    ret = @invoke CC.abstract_call_gf_by_type(interp::CC.AbstractInterpreter, f::Any, arginfo::ArgInfo,
                                              si::StmtInfo, atype::Any, sv::AbsIntState, max_methods::Int)

    callinfo = nothing
    if interp.meta !== nothing
        callinfo = inlining_handler(interp.meta, interp, atype, ret.info)
    end
    if callinfo === nothing
        callinfo = inlining_handler(nothing, interp, atype, ret.info)
    end
    if callinfo === nothing
        callinfo = ret.info
    end
    
    @static if VERSION ≥ v"1.11-"
        return CC.CallMeta(ret.rt, ret.exct, ret.effects, callinfo)
    else
        return CC.CallMeta(ret.rt, ret.effects, callinfo)
    end
end

@static if VERSION < v"1.12.0-DEV.45" 
let # overload `inlining_policy`
    @static if VERSION ≥ v"1.11.0-DEV.879"
        sigs_ex = :(
            interp::GPUInterpreter,
            @nospecialize(src),
            @nospecialize(info::CC.CallInfo),
            stmt_flag::UInt32,
        )
        args_ex = :(
            interp::CC.AbstractInterpreter,
            src::Any,
            info::CC.CallInfo,
            stmt_flag::UInt32,
        )
    else
        sigs_ex = :(
            interp::GPUInterpreter,
            @nospecialize(src),
            @nospecialize(info::CC.CallInfo),
            stmt_flag::UInt8,
            mi::MethodInstance,
            argtypes::Vector{Any},
        )
        args_ex = :(
            interp::CC.AbstractInterpreter,
            src::Any,
            info::CC.CallInfo,
            stmt_flag::UInt8,
            mi::MethodInstance,
            argtypes::Vector{Any},
        )
    end
    @eval function CC.inlining_policy($(sigs_ex.args...))
        if info isa NoInlineCallInfo
            @safe_debug "Blocking inlining" info.tt info.kind
            return nothing
        elseif info isa AlwaysInlineCallInfo
            @safe_debug "Forcing inlining for" info.tt
            return src
        end
        return @invoke CC.inlining_policy($(args_ex.args...))
    end
end
else
function CC.src_inlining_policy(interp::GPUInterpreter,
                                @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)
                                
    if info isa NoInlineCallInfo
        @safe_debug "Blocking inlining" info.tt info.kind
        return false
    elseif info isa AlwaysInlineCallInfo
        @safe_debug "Forcing inlining for" info.tt
        return true
    end
    return @invoke CC.src_inlining_policy(interp::CC.AbstractInterpreter, src, info::CC.CallInfo, stmt_flag::UInt32)
end
end


## world view of the cache
using Core.Compiler: WorldView

if !HAS_INTEGRATED_CACHE

function CC.haskey(wvc::WorldView{CodeCache}, mi::MethodInstance)
    CC.get(wvc, mi, nothing) !== nothing
end

function CC.get(wvc::WorldView{CodeCache}, mi::MethodInstance, default)
    # check the cache
    for ci in get!(wvc.cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            # TODO: if (code && (code == jl_nothing || jl_ir_flag_inferred((jl_array_t*)code)))
            src = if ci.inferred isa Vector{UInt8}
                ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                       mi.def, C_NULL, ci.inferred)
            else
                ci.inferred
            end
            return ci
        end
    end

    return default
end

function CC.getindex(wvc::WorldView{CodeCache}, mi::MethodInstance)
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{CodeCache}, ci::CodeInstance, mi::MethodInstance)
    CC.setindex!(wvc.cache, ci, mi)
end

end # HAS_INTEGRATED_CACHE

## codegen/inference integration

function ci_cache_populate(interp, cache, mi, min_world, max_world)
    if VERSION >= v"1.12.0-DEV.15"
        inferred_ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE) # or SOURCE_MODE_FORCE_SOURCE_UNCACHED?
        @assert inferred_ci !== nothing "Inference of $mi failed"

        # inference should have populated our cache
        wvc = WorldView(cache, min_world, max_world)
        @assert CC.haskey(wvc, mi)
        ci = CC.getindex(wvc, mi)

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            CC.setindex!(wvc, inferred_ci, mi)
            ci = CC.getindex(wvc, mi)
        end
    else
        src = CC.typeinf_ext_toplevel(interp, mi)

        # inference should have populated our cache
        wvc = WorldView(cache, min_world, max_world)
        @assert CC.haskey(wvc, mi)
        ci = CC.getindex(wvc, mi)

        # if ci is rettype_const, the inference result won't have been cached
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end
    end

    return ci::CodeInstance
end

function ci_cache_lookup(cache, mi, min_world, max_world)
    wvc = WorldView(cache, min_world, max_world)
    ci = CC.get(wvc, mi, nothing)
    if ci !== nothing && ci.inferred === nothing
        # if for some reason we did end up with a codeinfo without inferred source, e.g.,
        # because of calling `Base.return_types` which only sets rettyp, pretend we didn't
        # run inference so that we re-infer now and not during codegen (which is disallowed)
        return nothing
    end
    return ci
end


## interface

# for platforms without @cfunction-with-closure support
const _method_instances = Ref{Any}()
const _cache = Ref{Any}()
function _lookup_fun(mi, min_world, max_world)
    push!(_method_instances[], mi)
    ci_cache_lookup(_cache[], mi, min_world, max_world)
end

@enum CompilationPolicy::Cint begin
    CompilationPolicyDefault = 0
    CompilationPolicyExtern = 1
end

# HACK: in older versions of Julia, `jl_create_native` doesn't take a world argument
#       but instead always generates code for the current world. note that this doesn't
#       actually change the world age, but just spoofs the counter `jl_create_native` reads.
# XXX: Base.get_world_counter is supposed to be monotonically increasing and is runtime global.
macro in_world(world, ex)
    quote
        actual_world = Base.get_world_counter()
        world_counter = cglobal(:jl_world_counter, Csize_t)
        unsafe_store!(world_counter, $(esc(world)))
        try
            $(esc(ex))
        finally
            unsafe_store!(world_counter, actual_world)
        end
    end
end

"""
    precompile(job::CompilerJob)

Compile the GPUCompiler job. In particular this will run inference using the foreign
abstract interpreter.
"""
function Base.precompile(@nospecialize(job::CompilerJob))
    if job.source.def.primary_world > job.world || job.world > job.source.def.deleted_world
        error("Cannot compile $(job.source) for world $(job.world); method is only valid in worlds $(job.source.def.primary_world) to $(job.source.def.deleted_world)")
    end

    # populate the cache
    interp = get_interpreter(job)
    cache = CC.code_cache(interp)
    if ci_cache_lookup(cache, job.source, job.world, job.world) === nothing
        ci_cache_populate(interp, cache, job.source, job.world, job.world)
        return ci_cache_lookup(cache, job.source, job.world, job.world) !== nothing
    end
    return true
end

function compile_method_instance(@nospecialize(job::CompilerJob))
    if job.source.def.primary_world > job.world || job.world > job.source.def.deleted_world
        error("Cannot compile $(job.source) for world $(job.world); method is only valid in worlds $(job.source.def.primary_world) to $(job.source.def.deleted_world)")
    end

    # A poor man's worklist implementation.
    # `compiled` contains a mapping from `mi->ci, func, specfunc`
    # FIXME: Since we are disabling Julia internal caching we might
    #        generate for the same mi multiple LLVM functions. 
    # `outstanding` are the missing edges that were not compiled by `compile_method_instance`
    # Currently these edges are generated through deferred codegen.
    compiled = IdDict{Edge, Any}()
    llvm_mod, outstanding = compile_method_instance(job, compiled)
    worklist = outstanding
    while !isempty(worklist)
        edge = pop!(worklist)
        haskey(compiled, edge) && continue # We have fulfilled the request already
        source = edge.mi
        meta = edge.meta
        # Create a new compiler job for this edge, reusing the config settings from the inital one
        job2 = CompilerJob(source, CompilerConfig(job.config; meta))
        llvm_mod2, outstanding = compile_method_instance(job2, compiled)
        append!(worklist, outstanding) # merge worklist with new outstanding edges
        @assert context(llvm_mod) == context(llvm_mod2)
        link!(llvm_mod, llvm_mod2)
    end

    return llvm_mod, compiled
end

function compile_method_instance(@nospecialize(job::CompilerJob), compiled::IdDict{Edge, Any})
    # populate the cache
    interp = get_interpreter(job)
    cache = CC.code_cache(interp)
    if ci_cache_lookup(cache, job.source, job.world, job.world) === nothing
        ci_cache_populate(interp, cache, job.source, job.world, job.world)
        @assert ci_cache_lookup(cache, job.source, job.world, job.world) !== nothing
    end

    # create a callback to look-up function in our cache,
    # and keep track of the method instances we needed.
    method_instances = Any[]
    if Sys.ARCH == :x86 || Sys.ARCH == :x86_64
        function lookup_fun(mi, min_world, max_world)
            push!(method_instances, mi)
            ci_cache_lookup(cache, mi, min_world, max_world)
        end
        lookup_cb = @cfunction($lookup_fun, Any, (Any, UInt, UInt))
    else
        _cache[] = cache
        _method_instances[] = method_instances
        lookup_cb = @cfunction(_lookup_fun, Any, (Any, UInt, UInt))
    end

    # set-up the compiler interface
    debug_info_kind = llvm_debug_info(job)
    cgparams = (;
        track_allocations  = false,
        code_coverage      = false,
        prefer_specsig     = true,
        gnu_pubnames       = false,
        debug_info_kind    = Cint(debug_info_kind),
        lookup             = Base.unsafe_convert(Ptr{Nothing}, lookup_cb),
        safepoint_on_entry = can_safepoint(job),
        gcstack_arg        = false)
    params = Base.CodegenParams(; cgparams...)

    # generate IR
    GC.@preserve lookup_cb begin
        # create and configure the module
        ts_mod = ThreadSafeModule("start")
        ts_mod() do mod
            triple!(mod, llvm_triple(job.config.target))
            if julia_datalayout(job.config.target) !== nothing
                datalayout!(mod, julia_datalayout(job.config.target))
            end
            flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                Metadata(ConstantInt(dwarf_version(job.config.target)))
            flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                Metadata(ConstantInt(DEBUG_METADATA_VERSION()))
        end

        native_code = ccall(:jl_create_native, Ptr{Cvoid},
                (Vector{MethodInstance}, LLVM.API.LLVMOrcThreadSafeModuleRef, Ptr{Base.CodegenParams}, Cint, Cint, Cint, Csize_t),
                [job.source], ts_mod, Ref(params), CompilationPolicyExtern, #=imaging mode=# 0, #=external linkage=# 0, job.world)
        @assert native_code != C_NULL

        llvm_mod_ref =
            ccall(:jl_get_llvm_module, LLVM.API.LLVMOrcThreadSafeModuleRef,
                  (Ptr{Cvoid},), native_code)
        @assert llvm_mod_ref != C_NULL

        # XXX: this is wrong; we can't expose the underlying LLVM module, but should
        #      instead always go through the callback in order to unlock it properly.
        #      rework this once we depend on Julia 1.9 or later.
        llvm_ts_mod = LLVM.ThreadSafeModule(llvm_mod_ref)
        llvm_mod = nothing
        llvm_ts_mod() do mod
            llvm_mod = mod
        end
    end
    if !(Sys.ARCH == :x86 || Sys.ARCH == :x86_64)
        cache_gbl = nothing
    end

    # process all compiled method instances
    meta = inference_metadata(job)
    for mi in method_instances
        ci = ci_cache_lookup(cache, mi, job.world, job.world)
        ci === nothing && continue

        # get the function index
        llvm_func_idx = Ref{Int32}(-1)
        llvm_specfunc_idx = Ref{Int32}(-1)
        ccall(:jl_get_function_id, Nothing,
              (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
              native_code, ci, llvm_func_idx, llvm_specfunc_idx)
        @assert llvm_func_idx[] != -1 || llvm_specfunc_idx[] != -1 "Static compilation failed"

        # get the function
        llvm_func = if llvm_func_idx[] >= 1
            llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                                  (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
            @assert llvm_func_ref != C_NULL
            LLVM.name(LLVM.Function(llvm_func_ref))
        else
            nothing
        end

        llvm_specfunc = if llvm_specfunc_idx[] >= 1
            llvm_specfunc_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                                      (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
            @assert llvm_specfunc_ref != C_NULL
            LLVM.name(LLVM.Function(llvm_specfunc_ref))
        else
            nothing
        end

        # NOTE: it's not safe to store raw LLVM functions here, since those may get
        #       removed or renamed during optimization, so we store their name instead.
        # FIXME: Enable this assert when we have a fully featured worklist
        # @assert !haskey(compiled, mi)
        compiled[Edge(meta, mi)] = (; ci, func=llvm_func, specfunc=llvm_specfunc)
    end

    # Collect the deferred edges
    outstanding = Edge[]
    for mi in method_instances
        edge = Edge(meta, mi)
        !haskey(compiled, edge) && continue # Equivalent to ci_cache_lookup == nothing
        ci = compiled[edge].ci
        @static if VERSION >= v"1.11.0-"
            edges = CC.traverse_analysis_results(ci) do @nospecialize result
                return result isa DeferredEdges ? result : return
            end
        else
            edges = ci.argescapes
            if !(edges isa Union{Nothing, DeferredEdges})
                edges = nothing
            end
        end
        if edges !== nothing
            for other in (edges::DeferredEdges).edges
                if !haskey(compiled, other)
                    push!(outstanding, other)
                end
            end
        end
    end

    # ensure that the requested method instance was compiled
    @assert haskey(compiled, Edge(meta, job.source))

    return llvm_mod, outstanding
end

# partially revert JuliaLangjulia#49391
@static if v"1.11.0-DEV.1603" <= VERSION < v"1.12.0-DEV.347" && # reverted on master
           !(v"1.11-beta2" <= VERSION < v"1.12")                # reverted on 1.11-beta2
function CC.typeinf(interp::GPUInterpreter, frame::CC.InferenceState)
    if CC.__measure_typeinf__[]
        CC.Timings.enter_new_timer(frame)
        v = CC._typeinf(interp, frame)
        CC.Timings.exit_current_timer(frame)
        return v
    else
        return CC._typeinf(interp, frame)
    end
end
end
