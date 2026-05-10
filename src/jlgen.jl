# Julia compiler integration

import CompilerCaching
using CompilerCaching: CacheView, @setup_caching


## world age lookups

import Base: tls_world_age


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
function methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                        world::Integer=tls_world_age())
    sig = signature_type_by_tt(ft, tt)
    @assert Base.isdispatchtuple(sig)

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

function generic_methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                                world::Integer=tls_world_age())
    sig = signature_type_by_tt(ft, tt)

    match, _ = CC._findsup(sig, nothing, world)
    match === nothing && throw(MethodError(ft, tt, world))

    mi = CC.specialize_method(match)

    return mi::MethodInstance
end


## method overrides

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)

# Implements a priority lookup for method tables, where the first match in the stack get's returned.
struct StackedMethodTable{MTV<:CC.MethodTableView} <: CC.MethodTableView
    world::UInt
    mt::Core.MethodTable
    parent::MTV
end
StackedMethodTable(world::UInt, mt::Core.MethodTable) = StackedMethodTable(world, mt, CC.InternalMethodTable(world))
StackedMethodTable(world::UInt, mt::Core.MethodTable, parent::Core.MethodTable) = StackedMethodTable(world, mt, StackedMethodTable(world, parent))

CC.isoverlayed(::StackedMethodTable) = true

function CC.findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = CC._findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing # to many matches
    nr = CC.length(result)
    if nr ≥ 1 && CC.getindex(result, nr).fully_covers
        # no need to fall back to the parent method view
        return result
    end

    parent_result = CC.findall(sig, table.parent; limit)::Union{Nothing, CC.MethodLookupResult}
    parent_result === nothing && return nothing #too many matches

    # merge the parent match results with the internal method table
    return CC.MethodLookupResult(
        CC.vcat(result.matches, parent_result.matches),
        CC.WorldRange(
            CC.max(result.valid_worlds.min_world, parent_result.valid_worlds.min_world),
            CC.min(result.valid_worlds.max_world, parent_result.valid_worlds.max_world)),
        result.ambig | parent_result.ambig)
end

function CC.findsup(@nospecialize(sig::Type), table::StackedMethodTable)
    match, valid_worlds = CC._findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds
    parent_match, parent_valid_worlds = CC.findsup(sig, table.parent)
    return (
        parent_match,
        CC.WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds.min_world),
            min(valid_worlds.max_world, parent_valid_worlds.max_world))
        )
end


## interpreter

import Core.Compiler: get_inference_world
using Base: get_world_counter

if isdefined(Core.Compiler, :CachedMethodTable)
    using Core.Compiler: CachedMethodTable
    maybe_cached(mtv::CC.MethodTableView) = CachedMethodTable(mtv)
else
    maybe_cached(mtv::CC.MethodTableView) = mtv
end

get_method_table_view(world::UInt, mt::CC.MethodTable) = CC.OverlayMethodTable(world, mt)

# VERSION >= v"1.14.0-DEV.1691"
const INFERENCE_CACHE_TYPE = isdefined(CC, :InferenceCache) ? CC.InferenceCache : Vector{CC.InferenceResult}

"""
    GPUInterpreter{MTV, K, V}

Foreign abstract interpreter that drives Julia inference for GPU compilation. Parametric
on the method-table view (`MTV`), the cache owner type (`K`), and the consumer's results
struct type (`V`). The cache is a `CompilerCaching.CacheView{K, V}` — `@setup_caching`
wires `cache_owner` and `finish!` so each `CodeInstance` carries a `V()` results struct
on its `analysis_results` chain, partitioned in the global CI cache by the owner token.
"""
struct GPUInterpreter{MTV<:CC.MethodTableView, K, V} <: CC.AbstractInterpreter
    world::UInt
    method_table_view::MTV
    cache::CacheView{K, V}
    inf_cache::INFERENCE_CACHE_TYPE
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function GPUInterpreter(world::UInt; method_table_view::CC.MethodTableView,
                        cache::CacheView,
                        inf_params::CC.InferenceParams,
                        opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()
    GPUInterpreter(world, method_table_view, cache, INFERENCE_CACHE_TYPE(),
                   inf_params, opt_params)
end

function GPUInterpreter(interp::GPUInterpreter;
                        world::UInt=interp.world,
                        method_table_view::CC.MethodTableView=interp.method_table_view,
                        cache::CacheView=interp.cache,
                        inf_cache::INFERENCE_CACHE_TYPE=interp.inf_cache,
                        inf_params::CC.InferenceParams=interp.inf_params,
                        opt_params::CC.OptimizationParams=interp.opt_params)
    GPUInterpreter(world, method_table_view, cache, inf_cache, inf_params, opt_params)
end

CC.InferenceParams(interp::GPUInterpreter) = interp.inf_params
CC.OptimizationParams(interp::GPUInterpreter) = interp.opt_params
get_inference_world(interp::GPUInterpreter) = interp.world
CC.get_inference_cache(interp::GPUInterpreter) = interp.inf_cache

@setup_caching GPUInterpreter.cache

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
CC.method_table(interp::GPUInterpreter) = interp.method_table_view

# semi-concrete interepretation is broken with overlays (JuliaLang/julia#47349)
function CC.concrete_eval_eligible(interp::GPUInterpreter,
    @nospecialize(f), result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    ret = @invoke CC.concrete_eval_eligible(interp::CC.AbstractInterpreter,
        f::Any, result::CC.MethodCallResult, arginfo::CC.ArgInfo, sv::CC.InferenceState)
    if ret === :semi_concrete_eval
        return :none
    end
    return ret
end


## codegen/inference integration

const HAS_LLVM_GET_CIS = (
    VERSION >= v"1.13.0-DEV.1120" || (
        Libdl.dlsym(
            unsafe_load(cglobal(:jl_libjulia_handle, Ptr{Cvoid})), :jl_get_llvm_cis, throw_error = false
        ) !== nothing
    )
)

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
    if job.source.def.primary_world > job.world
        error("Cannot compile $(job.source) for world $(job.world); method is only valid from world $(job.source.def.primary_world) onwards")
    end
    interp = get_interpreter(job)
    CompilerCaching.typeinf!(interp.cache, interp, job.source)
    return true
end


## code coverage

# When the host process runs with `--code-coverage`, lowered code contains
# `:code_coverage_effect` expressions, which the compiler only inserts for code that is
# actually being tracked. Device code cannot count executions, as it cannot call into the
# Julia runtime, so instead we mark those lines as visited when compiling them: Coverage of
# device code thus means "this code was compiled", with counts reflecting the number of
# compilations rather than executions.
function coverage_visit_line(@nospecialize(file), line::Integer)
    file isa Symbol || return
    filename = String(file)
    (isempty(filename) || line <= 0) && return
    ccall(:jl_coverage_visit_line, Cvoid, (Cstring, Csize_t, Cint),
          filename, ncodeunits(filename), line)
    return
end

function record_coverage(mi::MethodInstance, src::CodeInfo)
    tracked = false
    for (pc, stmt) in enumerate(src.code)
        (stmt isa Expr && stmt.head === :code_coverage_effect) || continue
        tracked = true
        @static if VERSION >= v"1.12-"
            scopes = Base.IRShow.buildLineInfoNode(src.debuginfo, nothing, pc)
            isempty(scopes) && continue
            loc = scopes[end]   # innermost scope, i.e., the inlined callee
            file, line = loc.file, loc.line
        else
            lineidx = src.codelocs[pc]
            lineidx == 0 && continue
            loc = src.linetable[lineidx]::Core.LineInfoNode
            file, line = loc.file, loc.line
        end
        coverage_visit_line(file, line)
    end

    # the definition (signature) line isn't a `:code_coverage_effect`; Julia's codegen
    # visits it separately at the prologue. Mirror that, gated on the body being tracked,
    # or the signature reads as missed while the body is hit.
    if tracked
        def = mi.def
        if def isa Method
            coverage_visit_line(def.file, def.line)
        end
    end
    return
end

# for platforms without @cfunction-with-closure support
const _method_instances = Ref{Any}()
const _cache = Ref{Any}()
function _lookup_fun(mi, min_world, max_world)
    push!(_method_instances[], mi)
    cache = _cache[]::CacheView
    get(cache, mi, nothing)
end

function compile_method_instance(@nospecialize(job::CompilerJob))
    if job.source.def.primary_world > job.world
        error("Cannot compile $(job.source) for world $(job.world); method is only valid from world $(job.source.def.primary_world) onwards")
    end

    # populate the cache (inference)
    interp = get_interpreter(job)
    cache = interp.cache
    CompilerCaching.typeinf!(cache, interp, job.source)
    ci = get(cache, job.source, nothing)::CodeInstance

    # collect (CI, CodeInfo) pairs for jl_emit_native
    @static if VERSION >= v"1.12.0-DEV.1823"
        codeinfo_pairs = CompilerCaching.get_codeinfos(ci)
    else
        codeinfo_pairs = nothing
    end

    # record line coverage of all compiled code (on older versions of Julia, where
    # inference does not return sources, this happens after codegen instead)
    if Base.JLOptions().code_coverage != 0 && codeinfo_pairs !== nothing
        for (ci′, src) in codeinfo_pairs
            record_coverage(ci′.def::MethodInstance, src::CodeInfo)
        end
    end

    # create a callback to look-up function in our cache,
    # and keep track of the method instances we needed.
    method_instances = []
    if Sys.ARCH == :x86 || Sys.ARCH == :x86_64
        function lookup_fun(mi, min_world, max_world)
            push!(method_instances, mi)
            get(cache, mi, nothing)
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
    if v"1.12.0-DEV.2126" <= VERSION < v"1.13-" || VERSION >= v"1.13.0-DEV.285"
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
            for (ci′, src) in codeinfo_pairs
                push!(codeinfos, ci′::CodeInstance)
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
        llvm_ts_mod = LLVM.ThreadSafeModule(llvm_mod_ref)
        llvm_mod = nothing
        llvm_ts_mod() do mod
            llvm_mod = mod
        end
    end

    # Since Julia 1.13, the caller is responsible for initializing global variables that
    # point to global values or bindings with their address in memory.
    gvs = nothing
    inits = nothing
    @static if VERSION >= v"1.13.0-DEV.623"
        num_gvars = Ref{Csize_t}(0)
        @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
            C_NULL::Ptr{Cvoid}
        )::Nothing
        gvs = Vector{Ptr{LLVM.API.LLVMOpaqueValue}}(undef, num_gvars[])
        @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
            gvs::Ptr{LLVM.API.LLVMOpaqueValue}
        )::Nothing

        inits = Vector{Ptr{Cvoid}}(undef, num_gvars[])
        @ccall jl_get_llvm_gv_inits(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
                                    inits::Ptr{Cvoid})::Nothing
    elseif HAS_LLVM_GVS_GLOBALS
        if VERSION >= v"1.12.0-DEV.1703"
            num_gvars = Ref{Csize_t}(0)
            @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
                C_NULL::Ptr{Cvoid}
            )::Nothing
            gvs = Vector{Ptr{LLVM.API.LLVMOpaqueValue}}(undef, num_gvars[])
            @ccall jl_get_llvm_gvs_globals(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
                gvs::Ptr{LLVM.API.LLVMOpaqueValue}
            )::Nothing
            inits = Vector{Ptr{Cvoid}}(undef, num_gvars[])
            @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num_gvars::Ptr{Csize_t},
                inits::Ptr{Cvoid}
            )::Nothing
        else
            gvs = get_llvm_global_vars(native_code)
            inits = get_llvm_global_inits(native_code)
        end
    end

    # Maintain a map from global variables to their initialized Julia values.
    # The objects pointed to are perma-rooted, during codegen.
    gv_to_value = Dict{String, Ptr{Cvoid}}()

    if gvs === nothing
        for gv in globals(llvm_mod)
            if !haskey(metadata(gv), "julia.constgv")
                continue
            end
            gv_to_value[LLVM.name(gv)] = C_NULL
            val = initializer(gv)
            if val === nothing
                continue
            end
            while isa(val, LLVM.ConstantExpr)
                if in(opcode(val), (LLVM.API.LLVMBitCast, LLVM.API.LLVMPtrToInt, LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMIntToPtr))
                    val = operands(val)[1]
                    continue
                end
                break
            end
            if isa(val, LLVM.ConstantInt)
                gv_to_value[LLVM.name(gv)] = reinterpret(Ptr{Cvoid}, convert(UInt, val))
            end
        end
    else
        @assert inits !== nothing
        for (gv_ref, init) in zip(gvs, inits)
            gv = GlobalVariable(gv_ref)
            gv_to_value[LLVM.name(gv)] = init
            if LLVM.isnull(initializer(gv))
                val = const_inttoptr(ConstantInt(Int64(init)), value_type(initializer(gv)))
                initializer!(gv, val)
            end
        end
    end

    code_instances = Core.CodeInstance[]

    if HAS_LLVM_GET_CIS
        # on sufficiently recent versions of Julia, we can query the CIs compiled.
        num_cis = Ref{Csize_t}(0)
        @ccall jl_get_llvm_cis(native_code::Ptr{Cvoid}, num_cis::Ptr{Csize_t},
                               C_NULL::Ptr{Cvoid})::Nothing
        resize!(code_instances, num_cis[])
        @ccall jl_get_llvm_cis(native_code::Ptr{Cvoid}, num_cis::Ptr{Csize_t},
            code_instances::Ptr{Cvoid}
        )::Nothing
    elseif VERSION >= v"1.12.0-DEV.1703"
        # slightly older versions of Julia used MIs directly
        num_mis = Ref{Csize_t}(0)
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               C_NULL::Ptr{Cvoid})::Nothing
        resize!(method_instances, num_mis[])
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               method_instances::Ptr{Cvoid})::Nothing
    end

    if !HAS_LLVM_GET_CIS
        for mi in method_instances
            ci′ = get(cache, mi, nothing)
            ci′ === nothing && continue

            llvm_func_idx = Ref{Int32}(-1)
            llvm_specfunc_idx = Ref{Int32}(-1)
            ccall(
                :jl_get_function_id, Nothing,
                (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
                native_code, ci′, llvm_func_idx, llvm_specfunc_idx
            )
            if llvm_func_idx[] == -1
                continue
            end
            push!(code_instances, ci′)
        end
    else
        # `jl_get_llvm_cis` can report stale CIs that no longer cover the world
        # this job was compiled in. The explicit cache lookup path already filters
        # by world; do the same here before de-duplicating by MI.
        filter!(code_instances) do ci′
            ci′.min_world <= job.world <= ci′.max_world
        end

        # To avoid a clash in the compiled cache containing both an interpreter token and native,
        # prefer the non-native code-instance.
        owned_mis = Set{MethodInstance}()
        for ci′ in code_instances
            if ci′.owner !== nothing
                push!(owned_mis, ci′.def::MethodInstance)
            end
        end
        filter!(code_instances) do ci′
            return ci′.owner !== nothing || ci′.def ∉ owned_mis
        end
    end

    unique!(code_instances)

    # record line coverage of all compiled code (on newer versions of Julia, this happens
    # based on the sources returned by inference instead)
    if Base.JLOptions().code_coverage != 0 && codeinfo_pairs === nothing
        for ci in code_instances
            src = ci.inferred
            if src isa String
                src = Base._uncompressed_ir(ci, src)
            end
            if src isa CodeInfo
                record_coverage(ci.def::MethodInstance, src)
            end
        end
    end

    resize!(method_instances, length(code_instances))
    for (i, ci′) in enumerate(code_instances)
        method_instances[i] = ci′.def::MethodInstance
    end

    # process all compiled method instances
    compiled = Dict()
    for (ci′, mi) in zip(code_instances, method_instances)
        # get the function index
        llvm_func_idx = Ref{Int32}(-1)
        llvm_specfunc_idx = Ref{Int32}(-1)
        ccall(:jl_get_function_id, Nothing,
              (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
              native_code, ci′, llvm_func_idx, llvm_specfunc_idx)
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

        @assert !haskey(compiled, mi)

        # NOTE: it's not safe to store raw LLVM functions here, since those may get
        #       removed or renamed during optimization, so we store their name instead.
        compiled[mi] = (; ci=ci′, func=llvm_func, specfunc=llvm_specfunc)
    end

    # ensure that the requested method instance was compiled
    @assert haskey(compiled, job.source)

    return llvm_mod, compiled, gv_to_value
end
