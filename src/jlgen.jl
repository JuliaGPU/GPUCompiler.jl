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
global MethodError
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
@static if VERSION >= v"1.11.0-DEV.1552"

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
                        method_table::MTType,
                        token::Any,
                        inf_params::CC.InferenceParams,
                        opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    method_table = get_method_table_view(world, method_table)
    inf_cache = Vector{CC.InferenceResult}()

    return GPUInterpreter(world, method_table,
                          token, inf_cache,
                          inf_params, opt_params)
end

function GPUInterpreter(interp::GPUInterpreter;
                        world::UInt=interp.world,
                        method_table::GPUMethodTableView=interp.method_table,
                        token::Any=interp.token,
                        inf_cache::Vector{CC.InferenceResult}=interp.inf_cache,
                        inf_params::CC.InferenceParams=interp.inf_params,
                        opt_params::CC.OptimizationParams=interp.opt_params)
    return GPUInterpreter(world, method_table,
                          token, inf_cache,
                          inf_params, opt_params)
end

else

function GPUInterpreter(world::UInt=Base.get_world_counter();
                        method_table::MTType,
                        code_cache::CodeCache,
                        inf_params::CC.InferenceParams,
                        opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    method_table = get_method_table_view(world, method_table)
    inf_cache = Vector{CC.InferenceResult}()

    return GPUInterpreter(world, method_table,
                          code_cache, inf_cache,
                          inf_params, opt_params)
end

function GPUInterpreter(interp::GPUInterpreter;
                        world::UInt=interp.world,
                        method_table::GPUMethodTableView=interp.method_table,
                        code_cache::CodeCache=interp.code_cache,
                        inf_cache::Vector{CC.InferenceResult}=interp.inf_cache,
                        inf_params::CC.InferenceParams=interp.inf_params,
                        opt_params::CC.OptimizationParams=interp.opt_params)
    return GPUInterpreter(world, method_table,
                          code_cache, inf_cache,
                          inf_params, opt_params)
end
end # HAS_INTEGRATED_CACHE

CC.InferenceParams(interp::GPUInterpreter) = interp.inf_params
CC.OptimizationParams(interp::GPUInterpreter) = interp.opt_params
#=CC.=#get_inference_world(interp::GPUInterpreter) = interp.world
CC.get_inference_cache(interp::GPUInterpreter) = interp.inf_cache
@static if HAS_INTEGRATED_CACHE
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
@static if VERSION <= v"1.12.0-DEV.1531"
CC.verbose_stmt_info(interp::GPUInterpreter) = false
end
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
    codeinfos = Pair{CodeInstance, CodeInfo}[]
    @static if VERSION >= v"1.12.0-DEV.1434"
        # see typeinfer.jl: typeinf_ext_toplevel
        ci = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_NOT_REQUIRED)
        inspected = IdSet{CodeInstance}()
        tocompile = CodeInstance[ci]
        while !isempty(tocompile)
            callee = pop!(tocompile)
            callee in inspected && continue
            push!(inspected, callee)
            # now make sure everything has source code, if desired
            mi = CC.get_ci_mi(callee)
            def = mi.def
            if CC.use_const_api(callee)
                src = CC.codeinfo_for_const(interp, mi, ci.rettype_const)
            else
                # TODO: typeinf_code could return something with different edges/ages/owner/abi (needing an update to callee), which we don't handle here
                src = CC.typeinf_code(interp, mi, true)
            end
            if src isa CodeInfo
                CC.collectinvokes!(tocompile, src)
                push!(codeinfos, callee => src)
            end
        end
    elseif VERSION >= v"1.12.0-DEV.15"
        inferred_ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_FORCE_SOURCE)
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

    return codeinfos
end

function ci_cache_lookup(cache, mi, min_world, max_world)
    wvc = WorldView(cache, min_world, max_world)
    ci = CC.get(wvc, mi, nothing)
    if VERSION < v"1.12.0-DEV.1434" && ci !== nothing && ci.inferred === nothing
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
    ci_cache_populate(interp, cache, job.source, job.world, job.world)
    return true
end

function compile_method_instance(@nospecialize(job::CompilerJob))
    if job.source.def.primary_world > job.world || job.world > job.source.def.deleted_world
        error("Cannot compile $(job.source) for world $(job.world); method is only valid in worlds $(job.source.def.primary_world) to $(job.source.def.deleted_world)")
    end

    # populate the cache
    interp = get_interpreter(job)
    cache = CC.code_cache(interp)
    populated = ci_cache_populate(interp, cache, job.source, job.world, job.world)

    # create a callback to look-up function in our cache,
    # and keep track of the method instances we needed.
    method_instances = []
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
        safepoint_on_entry = can_safepoint(job),
        gcstack_arg        = false)
    if VERSION < v"1.12.0-DEV.1667"
        cgparams = (; lookup = Base.unsafe_convert(Ptr{Nothing}, lookup_cb), cgparams... )
    end
    if VERSION >= v"1.12.0-DEV.2126"
        cgparams = (; force_emit_all = true , cgparams...)
    end
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

        native_code = if VERSION >= v"1.12.0-DEV.1823"
            codeinfos = Any[]
            for (ci, src) in populated
                # each item in the list should be a CodeInstance followed by a CodeInfo
                # indicating something to compile
                push!(codeinfos, ci::CodeInstance)
                push!(codeinfos, src::CodeInfo)
            end
            @ccall jl_emit_native(codeinfos::Vector{Any}, ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef, Ref(params)::Ptr{Base.CodegenParams}, #=extern linkage=# false::Cint)::Ptr{Cvoid}
        elseif VERSION >= v"1.12.0-DEV.1667"
            ccall(:jl_create_native, Ptr{Cvoid},
                (Vector{MethodInstance}, LLVM.API.LLVMOrcThreadSafeModuleRef, Ptr{Base.CodegenParams}, Cint, Cint, Cint, Csize_t, Ptr{Cvoid}),
                [job.source], ts_mod, Ref(params), CompilationPolicyExtern, #=imaging mode=# 0, #=external linkage=# 0, job.world, Base.unsafe_convert(Ptr{Nothing}, lookup_cb))
        else
            ccall(:jl_create_native, Ptr{Cvoid},
                (Vector{MethodInstance}, LLVM.API.LLVMOrcThreadSafeModuleRef, Ptr{Base.CodegenParams}, Cint, Cint, Cint, Csize_t),
                [job.source], ts_mod, Ref(params), CompilationPolicyExtern, #=imaging mode=# 0, #=external linkage=# 0, job.world)
        end
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

    if VERSION >= v"1.12.0-DEV.1703"
        # on sufficiently recent versions of Julia, we can query the MIs compiled.
        # this is required after the move to `invokce(::CodeInstance)`, because our
        # lookup function (used to populate method_instances) isn't always called then.

        num_mis = Ref{Csize_t}(0)
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               C_NULL::Ptr{Cvoid})::Nothing
        resize!(method_instances, num_mis[])
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               method_instances::Ptr{Cvoid})::Nothing
    end

    # process all compiled method instances
    compiled = Dict()
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
        compiled[mi] = (; ci, func=llvm_func, specfunc=llvm_specfunc)
    end

    # ensure that the requested method instance was compiled
    @assert haskey(compiled, job.source)

    return llvm_mod, compiled
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
