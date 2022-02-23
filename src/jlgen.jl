# Julia compiler integration

## cache

using Core.Compiler: CodeInstance, MethodInstance

struct CodeCache
    dict::Dict{MethodInstance,Vector{CodeInstance}}

    CodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
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

const GLOBAL_CI_CACHE = CodeCache()


## method invalidations

function Core.Compiler.setindex!(cache::CodeCache, ci::CodeInstance, mi::MethodInstance)
    # make sure the invalidation callback is attached to the method instance
    callback(mi, max_world) = invalidate(cache, mi, max_world)
    if !isdefined(mi, :callbacks)
        mi.callbacks = Any[callback]
    elseif !in(callback, mi.callbacks)
        push!(mi.callbacks, callback)
    end

    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

# invalidation (like invalidate_method_instance, but for our cache)
function invalidate(cache::CodeCache, replaced::MethodInstance, max_world, seen=Set{MethodInstance}())
    push!(seen, replaced)

    cis = get(cache.dict, replaced, nothing)
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
        backedges = replaced.backedges
        # Don't touch/empty backedges `invalidate_method_instance` in C will do that later
        # replaced.backedges = Any[]

        for mi in filter(!in(seen), backedges)
            invalidate(cache, mi, max_world, seen)
        end
    end
end


## method overrides

@static if isdefined(Base.Experimental, Symbol("@overlay"))

# use an overlay method table

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)

else

# use an overlay world -- a special world that contains all method overrides

const GLOBAL_METHOD_TABLE = nothing

const override_world = typemax(Csize_t) - 1

struct WorldOverlayMethodTable <: Core.Compiler.MethodTableView
    world::UInt
end

function Core.Compiler.findall(@nospecialize(sig::Type{<:Tuple}), table::WorldOverlayMethodTable; limit::Int=typemax(Int))
    _min_val = Ref{UInt}(typemin(UInt))
    _max_val = Ref{UInt}(typemax(UInt))
    _ambig = Ref{Int32}(0)
    ms = Base._methods_by_ftype(sig, limit, override_world, false, _min_val, _max_val, _ambig)
    if ms === false
        return Core.Compiler.missing
    elseif isempty(ms)
        # no override, so look in the regular world
        _min_val[] = typemin(UInt)
        _max_val[] = typemax(UInt)
        ms = Base._methods_by_ftype(sig, limit, table.world, false, _min_val, _max_val, _ambig)
    else
        # HACK: inference doesn't like our override world
        _min_val[] = table.world
    end
    if ms === false
        return Core.Compiler.missing
    end
    return Core.Compiler.MethodLookupResult(ms::Vector{Any}, Core.Compiler.WorldRange(_min_val[], _max_val[]), _ambig[] != 0)
end

end

"""
    @override mt def

!!! warning

    On Julia 1.6, evaluation of the expression returned by this macro should be postponed
    until run time (i.e. don't just call this macro or return its returned value, but
    save it in a global expression and `eval` it during `__init__`, additionally guarded
    by a check to `ccall(:jl_generating_output, Cint, ()) != 0`).
"""
macro override(mt, ex)
    if isdefined(Base.Experimental, Symbol("@overlay"))
        esc(quote
            Base.Experimental.@overlay $mt $ex
        end)
    else
        quote
            world_counter = cglobal(:jl_world_counter, Csize_t)
            regular_world = unsafe_load(world_counter)

            $(Expr(:tryfinally, # don't introduce scope
                quote
                    unsafe_store!(world_counter, $(override_world-1))
                    $(esc(ex))
                end,
                quote
                    unsafe_store!(world_counter, regular_world)
                end
            ))
        end
    end
end


## interpreter

using Core.Compiler:
    AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams

struct GPUInterpreter <: AbstractInterpreter
    global_cache::CodeCache
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    function GPUInterpreter(cache::CodeCache, mt::Union{Nothing,Core.MethodTable}, world::UInt)
        @assert world <= Base.get_world_counter()

        return new(
            cache,
            mt,

            # Initially empty cache
            Vector{InferenceResult}(),

            # world age counter
            world,

            # parameters for inference and optimization
            InferenceParams(unoptimize_throw_blocks=false),
            VERSION >= v"1.8.0-DEV.486" ? OptimizationParams() :
                                          OptimizationParams(unoptimize_throw_blocks=false),
        )
    end
end

Core.Compiler.InferenceParams(interp::GPUInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::GPUInterpreter) = interp.opt_params
Core.Compiler.get_world_counter(interp::GPUInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::GPUInterpreter) = interp.local_cache
Core.Compiler.code_cache(interp::GPUInterpreter) = WorldView(interp.global_cache, interp.world)

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(interp::GPUInterpreter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(interp::GPUInterpreter, mi::MethodInstance) = nothing

function Core.Compiler.add_remark!(interp::GPUInterpreter, sv::InferenceState, msg)
    @safe_debug "Inference remark during GPU compilation of $(sv.linfo): $msg"
end

Core.Compiler.may_optimize(interp::GPUInterpreter) = true
Core.Compiler.may_compress(interp::GPUInterpreter) = true
Core.Compiler.may_discard_trees(interp::GPUInterpreter) = true
@static if VERSION >= v"1.7.0-DEV.577"
Core.Compiler.verbose_stmt_info(interp::GPUInterpreter) = false
end

@static if isdefined(Base.Experimental, Symbol("@overlay"))
using Core.Compiler: OverlayMethodTable
Core.Compiler.method_table(interp::GPUInterpreter, sv::InferenceState) =
    OverlayMethodTable(interp.world, interp.method_table)
else
Core.Compiler.method_table(interp::GPUInterpreter, sv::InferenceState) =
    WorldOverlayMethodTable(interp.world)
end


## world view of the cache

using Core.Compiler: WorldView

function Core.Compiler.haskey(wvc::WorldView{CodeCache}, mi::MethodInstance)
    Core.Compiler.get(wvc, mi, nothing) !== nothing
end

function Core.Compiler.get(wvc::WorldView{CodeCache}, mi::MethodInstance, default)
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

function Core.Compiler.getindex(wvc::WorldView{CodeCache}, mi::MethodInstance)
    r = Core.Compiler.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function Core.Compiler.setindex!(wvc::WorldView{CodeCache}, ci::CodeInstance, mi::MethodInstance)
    src = if ci.inferred isa Vector{UInt8}
        ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                mi.def, C_NULL, ci.inferred)
    else
        ci.inferred
    end
    Core.Compiler.setindex!(wvc.cache, ci, mi)
end


## codegen/inference integration

function ci_cache_populate(interp, cache, mt, mi, min_world, max_world)
    src = Core.Compiler.typeinf_ext_toplevel(interp, mi)

    # inference populates the cache, so we don't need to jl_get_method_inferred
    wvc = WorldView(cache, min_world, max_world)
    @assert Core.Compiler.haskey(wvc, mi)

    # if src is rettyp_const, the codeinfo won't cache ci.inferred
    # (because it is normally not supposed to be used ever again).
    # to avoid the need to re-infer, set that field here.
    ci = Core.Compiler.getindex(wvc, mi)
    if ci !== nothing && ci.inferred === nothing
        ci.inferred = src
    end

    return ci
end

function ci_cache_lookup(cache, mi, min_world, max_world)
    wvc = WorldView(cache, min_world, max_world)
    ci = Core.Compiler.get(wvc, mi, nothing)
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

function compile_method_instance(@nospecialize(job::CompilerJob),
                                 method_instance::MethodInstance)
    # populate the cache
    cache = ci_cache(job)
    mt = method_table(job)
    interp = get_interpreter(job)
    if ci_cache_lookup(cache, method_instance, job.source.world, typemax(Cint)) === nothing
        ci_cache_populate(interp, cache, mt, method_instance, job.source.world, typemax(Cint))
    end

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
    params = Base.CodegenParams(;
        track_allocations  = false,
        code_coverage      = false,
        prefer_specsig     = true,
        gnu_pubnames       = false,
        debug_info_kind    = Cint(debug_info_kind),
        lookup             = Base.unsafe_convert(Ptr{Nothing}, lookup_cb))

    # generate IR
    GC.@preserve lookup_cb begin
        native_code = if VERSION >= v"1.8.0-DEV.661"
            ccall(:jl_create_native, Ptr{Cvoid},
                  (Vector{MethodInstance}, Ptr{Base.CodegenParams}, Cint),
                  [method_instance], Ref(params), #=extern policy=# 1)
        else
            ccall(:jl_create_native, Ptr{Cvoid},
                  (Vector{MethodInstance}, Base.CodegenParams, Cint),
                  [method_instance], params, #=extern policy=# 1)
        end
        @assert native_code != C_NULL
        llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                             (Ptr{Cvoid},), native_code)
        @assert llvm_mod_ref != C_NULL
        llvm_mod = LLVM.Module(llvm_mod_ref)
    end
    if !(Sys.ARCH == :x86 || Sys.ARCH == :x86_64)
        cache_gbl = nothing
    end

    # process all compiled method instances
    compiled = Dict()
    for mi in method_instances
        ci = ci_cache_lookup(cache, mi, job.source.world, typemax(Cint))

        if ci !== nothing
            # get the function index
            llvm_func_idx = Ref{Int32}(-1)
            llvm_specfunc_idx = Ref{Int32}(-1)
            ccall(:jl_get_function_id, Nothing,
                (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
                native_code, ci, llvm_func_idx, llvm_specfunc_idx)

            # get the function
            llvm_func = if llvm_func_idx[] != -1
                llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                                      (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
                @assert llvm_func_ref != C_NULL
                LLVM.name(LLVM.Function(llvm_func_ref))
            else
                nothing
            end


            llvm_specfunc = if llvm_specfunc_idx[] != -1
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
    end

    # configure the module
    triple!(llvm_mod, llvm_triple(job.target))
    if julia_datalayout(job.target) !== nothing
        datalayout!(llvm_mod, julia_datalayout(job.target))
    end

    return llvm_mod, compiled
end
