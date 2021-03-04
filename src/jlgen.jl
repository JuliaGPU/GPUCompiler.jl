# Julia compiler integration

## cache

using Core.Compiler: CodeInstance, MethodInstance

if VERSION >= v"1.7-"

struct CodeCache
    dict::Dict{MethodInstance,Vector{CodeInstance}}

    CodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
end

else

struct CodeCache
    dict::Dict{MethodInstance,Vector{CodeInstance}}

    override_table::Dict{Type,Function}
    override_aliases::Dict{Method,Type}

    CodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}(),
                      Dict{Type,Function}(), Dict{Method,Type}())
end

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
    else
        if all(cb -> cb !== callback, mi.callbacks)
            push!(mi.callbacks, callback)
        end
    end

    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

# invalidation (like invalidate_method_instance, but for our cache)
function invalidate(cache::CodeCache, replaced::MethodInstance, max_world, depth=0)
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

        for mi in backedges
            invalidate(cache, mi, max_world, depth + 1)
        end
    end
end


## method overrides

@static if VERSION >= v"1.7-"

# use an overlay method table

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)

else

# spoof the codeinstance cache

const GLOBAL_METHOD_TABLE = nothing

# conceptually, for each overridden function `f` the cache has a method table used to find
# the replacement method using familiar dispatch semantics.
#
# practically, the API to construct and query a method table doesn't appear complete
# (`jl_methtable_lookup` only returns exact matches; `jl_typemap_assoc_by_type` isn't
# exported), so instead of creating a method table we generate an new anonymous function,
# stored in `cache.override_table`, and instead of directly looking up the replacement
# function we look up a dummy method and match it to the replacement one using the
# `cache.override_aliases` dictionary.

# `argdata` is `Core.svec(Core.svec(types...), Core.svec(typevars...), LineNumberNode)`
jl_method_def(argdata::Core.SimpleVector, ci::Core.CodeInfo, mod::Module) =
    ccall(:jl_method_def, Nothing, (Core.SimpleVector, Any, Any), argdata, ci, mod)

argdata(sig, source) =
    Core.svec(Base.unwrap_unionall(sig).parameters::Core.SimpleVector,
              Core.svec(typevars(sig)...),
              source)

typevars(T::UnionAll) = (T.var, typevars(T.body)...)
typevars(T::DataType) = ()

getmodule(F::Type{<:Function}) = F.name.mt.module
getmodule(f::Function) = getmodule(typeof(f))

function add_override!(cache::CodeCache, @nospecialize(f::Function),
                       @nospecialize(f′::Function), @nospecialize(tt::Type),
                       source::Core.LineNumberNode, mod::Module)
    # create an "overrides function" for this source function,
    # whose method table we'll abuse to attach overrides to
    mt = get!(cache.override_table, typeof(f)) do
        mod.eval(quote
            function $(gensym())
            end
        end)
    end

    # create an "overrides method" corresponding to this override
    ci = mod.eval(Expr(:lambda, [Symbol("#self#"); fill(Symbol("#unused#"), length(tt.parameters))],
                                Expr(:return, nothing)))
    sig = Base.signature_type(mt, tt)
    jl_method_def(argdata(sig, source), ci, mod)
    # NOTE: we use jl_method_def instead of Expr(:method) as we have the signature already

    meth = which(mt, tt)
    @assert meth.sig === sig
    cache.override_aliases[meth] = typeof(f′)

    return
end

# get the replacement function type for a signature, or nothing
function get_override(cache::CodeCache, @nospecialize(sig); world=typemax(UInt))
    ft, t... = [sig.parameters...]

    # do we have overrides for this function?
    haskey(cache.override_table, ft) || return nothing
    mt = cache.override_table[ft]

    # do we have an override for these argument types?
    hasmethod(mt, t) || return nothing
    match = which(mt, t)

    return cache.override_aliases[match]
end

end

"""
    @override cache mt source_function(::ArgTyp...) replacement_function
"""
macro override(cache, mt, ex)
    if VERSION >= v"1.7-"
        esc(quote
            Base.Experimental.@overlay $mt $ex
        end)
    else
        def = splitdef(ex)
        f = def[:name]

        # recombine into a replacement function
        new_f = gensym()
        def[:name] = new_f
        new_ex = combinedef(def)

        quote
            $(esc(new_ex))

            # use the newly-defined method to find the tuple type (without parsing `ex`)
            new_method = first(methods($(esc(new_f))))
            tt = Tuple{new_method.sig.parameters[2:end]...}

            GPUCompiler.add_override!($(esc(cache)), $(esc(f)), $(esc(new_f)), tt,
                                      LineNumberNode($(__source__.line),
                                                     $(QuoteNode(__source__.file))),
                                      $__module__)
        end
    end
end


## interpreter

using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams

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
if VERSION >= v"1.7.0-DEV.577"
Core.Compiler.verbose_stmt_info(interp::GPUInterpreter) = false
end

if VERSION >= v"1.7-"
Core.Compiler.method_table(interp::GPUInterpreter, sv::InferenceState) =
    Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)
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
            return ci
        end
    end

    if VERSION < v"1.7-"
        sig = Base.unwrap_unionall(mi.specTypes)

        # check if we have any overrides for this method instance's function type
        actual_mi = mi
        ft′ = get_override(wvc.cache, sig; world=wvc.worlds.min_world)
        if ft′ !== nothing
            sig′ = Tuple{ft′, sig.parameters[2:end]...}
            meth = which(sig′)

            (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                                (Any, Any), sig′, meth.sig)::Core.SimpleVector
            meth = Base.func_for_method_checked(meth, ti, env)
            actual_mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                (Any, Any, Any, UInt), meth, ti, env, wvc.worlds.min_world)
        end

        # if we want to override a method instance, eagerly put its replacement in the cache.
        # this is necessary, because we generally don't populate the cache, inference does,
        # and it won't put the replacement method instance in the cache by itself.
        if mi !== actual_mi
            # XXX: is this OK to do? shouldn't we _inform_ the compiler about the replacement
            # method instead of just spoofing the code instance? I tried to do so using a
            # MethodTableView, but the fact that the resulting MethodMatch referred the
            # replacement function, while there was still a GlobalRef in the IR pointing to
            # the original function, resulted in optimizer confusion.
            ci = ci_cache_populate(wvc.cache, nothing, actual_mi, wvc.worlds.min_world, wvc.worlds.max_world)

            # make sure to recompress any IR in the CodeInstance we'll be spoofing,
            # as the process uses both the CodeInstance and its parent Method
            # (values are encoded as indices into the method->roots array).
            src = if ci.inferred isa Vector{UInt8}
                temp = ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                            actual_mi.def, C_NULL, ci.inferred)
                ccall(:jl_compress_ir, Any, (Any, Any), mi.def, temp)
            else
                copy(ci.inferred)
            end

            # copy the CodeInstance we'll be spoofing to ensure no cross-method effects
            # (such as IR compression) end up in the cache of the original method.
            if isdefined(ci, :rettype_const)
                const_flags = 0x2
                inferred_const = ci.rettype_const
            else
                const_flags = 0x0
                inferred_const = nothing
            end
            ci′ = Core.CodeInstance(mi, ci.rettype, inferred_const, src,
                                Int32(const_flags), ci.min_world, ci.max_world)

            Core.Compiler.setindex!(wvc.cache, ci′, mi)
            return ci′
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
    Core.Compiler.setindex!(wvc.cache, ci, mi)
end


## codegen/inference integration

function ci_cache_populate(cache, mt, mi, min_world, max_world)
    interp = GPUInterpreter(cache, mt, min_world)
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
    return Core.Compiler.get(wvc, mi, nothing)
end


## interface

function compile_method_instance(@nospecialize(job::CompilerJob),
                                 method_instance::MethodInstance, world)
    # populate the cache
    cache = ci_cache(job)
    mt = method_table(job)
    if ci_cache_lookup(cache, method_instance, world, world) === nothing
        ci_cache_populate(cache, mt, method_instance, world, world)
    end

    # set-up the compiler interface
    debug_info_kind = if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
    if job.target isa PTXCompilerTarget && !job.target.debuginfo
        debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
    end
    lookup_fun = (mi, min_world, max_world) -> ci_cache_lookup(cache, mi, min_world, max_world)
    lookup_cb = @cfunction($lookup_fun, Any, (Any, UInt, UInt))
    params = Base.CodegenParams(;
        track_allocations  = false,
        code_coverage      = false,
        prefer_specsig     = true,
        gnu_pubnames       = false,
        debug_info_kind    = Cint(debug_info_kind),
        lookup             = Base.unsafe_convert(Ptr{Nothing}, lookup_cb))

    # generate IR
    GC.@preserve lookup_cb begin
        native_code = ccall(:jl_create_native, Ptr{Cvoid},
                            (Vector{MethodInstance}, Base.CodegenParams, Cint),
                            [method_instance], params, #=extern policy=# 1)
        @assert native_code != C_NULL
        llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                            (Ptr{Cvoid},), native_code)
        @assert llvm_mod_ref != C_NULL
        llvm_mod = LLVM.Module(llvm_mod_ref)
    end

    # get the top-level code
    code = ci_cache_lookup(cache, method_instance, world, world)

    # get the top-level function index
    llvm_func_idx = Ref{Int32}(-1)
    llvm_specfunc_idx = Ref{Int32}(-1)
    ccall(:jl_get_function_id, Nothing,
          (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
          native_code, code, llvm_func_idx, llvm_specfunc_idx)
    @assert llvm_func_idx[] != -1
    @assert llvm_specfunc_idx[] != -1

    # get the top-level function)
    llvm_func_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                          (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
    @assert llvm_func_ref != C_NULL
    llvm_func = LLVM.Function(llvm_func_ref)
    llvm_specfunc_ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                              (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
    @assert llvm_specfunc_ref != C_NULL
    llvm_specfunc = LLVM.Function(llvm_specfunc_ref)

    # configure the module
    triple!(llvm_mod, llvm_triple(job.target))
    if julia_datalayout(job.target) !== nothing
        datalayout!(llvm_mod, julia_datalayout(job.target))
    end

    return llvm_specfunc, llvm_mod
end
