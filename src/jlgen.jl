# Julia compiler integration

## cache

using Core.Compiler: CodeInstance, MethodInstance

struct GPUCodeCache
    dict::Dict{MethodInstance,Vector{CodeInstance}}
    GPUCodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
end

function Core.Compiler.setindex!(cache::GPUCodeCache, ci::CodeInstance, mi::MethodInstance)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end

const GPU_CI_CACHE = GPUCodeCache()


## interpreter

using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams

struct GPUInterpreter <: AbstractInterpreter
    # Cache of inference results for this particular interpreter
    cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    function GPUInterpreter(world::UInt)
        @assert world <= Base.get_world_counter()

        return new(
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

# Quickly and easily satisfy the AbstractInterpreter API contract
Core.Compiler.get_world_counter(ni::GPUInterpreter) = ni.world
Core.Compiler.get_inference_cache(ni::GPUInterpreter) = ni.cache
Core.Compiler.InferenceParams(ni::GPUInterpreter) = ni.inf_params
Core.Compiler.OptimizationParams(ni::GPUInterpreter) = ni.opt_params
Core.Compiler.may_optimize(ni::GPUInterpreter) = true
Core.Compiler.may_compress(ni::GPUInterpreter) = true
Core.Compiler.may_discard_trees(ni::GPUInterpreter) = true
Core.Compiler.add_remark!(ni::GPUInterpreter, sv::InferenceState, msg) = nothing # TODO


## world view of the cache

using Core.Compiler: WorldView

function Core.Compiler.haskey(wvc::WorldView{GPUCodeCache}, mi::MethodInstance)
    Core.Compiler.get(wvc, mi, nothing) !== nothing
end

function Core.Compiler.get(wvc::WorldView{GPUCodeCache}, mi::MethodInstance, default)
    cache = wvc.cache
    for ci in get!(cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            # TODO: if (code && (code == jl_nothing || jl_ir_flag_inferred((jl_array_t*)code)))
            return ci
        end
    end

    return default
end

function Core.Compiler.getindex(wvc::WorldView{GPUCodeCache}, mi::MethodInstance)
    r = Core.Compiler.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

Core.Compiler.setindex!(wvc::WorldView{GPUCodeCache}, ci::CodeInstance, mi::MethodInstance) =
    Core.Compiler.setindex!(wvc.cache, ci, mi)


## codegen/inference integration

Core.Compiler.code_cache(ni::GPUInterpreter) = WorldView(GPU_CI_CACHE, ni.world)

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(ni::GPUInterpreter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(ni::GPUInterpreter, mi::MethodInstance) = nothing

function gpu_ci_cache_lookup(mi, min_world, max_world)
    wvc = WorldView(GPU_CI_CACHE, min_world, max_world)
    if !Core.Compiler.haskey(wvc, mi)
        interp = GPUInterpreter(min_world)
        src = Core.Compiler.typeinf_ext_toplevel(interp, mi)
        # inference populates the cache, so we don't need to jl_get_method_inferred
        @assert Core.Compiler.haskey(wvc, mi)

        # if src is rettyp_const, the codeinfo won't cache ci.inferred
        # (because it is normally not supposed to be used ever again).
        # to avoid the need to re-infer, set that field here.
        ci = Core.Compiler.getindex(wvc, mi)
        if ci !== nothing && ci.inferred === nothing
            ci.inferred = src
        end
    end
    return Core.Compiler.getindex(wvc, mi)
end


## interface

function compile_method_instance(@nospecialize(job::CompilerJob), method_instance::MethodInstance, world)
    # set-up the compiler interface
    debug_info_kind = if Base.JLOptions().debug_level == 0
        LLVM.API.LLVMDebugEmissionKindNoDebug
    elseif Base.JLOptions().debug_level == 1
        LLVM.API.LLVMDebugEmissionKindLineTablesOnly
    elseif Base.JLOptions().debug_level >= 2
        LLVM.API.LLVMDebugEmissionKindFullDebug
    end
    if job.target isa PTXCompilerTarget # && driver_version(job.target) < v"10.2"
        # LLVM's debug info crashes older CUDA assemblers
        # FIXME: this was supposed to be fixed on 10.2
        @debug "Incompatibility detected between CUDA and LLVM 8.0+; disabling debug info emission" maxlog=1
        debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
    end
    params = Base.CodegenParams(;
                    track_allocations  = false,
                    code_coverage      = false,
                    prefer_specsig     = true,
                    gnu_pubnames       = false,
                    debug_info_kind    = Cint(debug_info_kind),
                    lookup             = @cfunction(gpu_ci_cache_lookup, Any, (Any, UInt, UInt)))

    # generate IR
    native_code = ccall(:jl_create_native, Ptr{Cvoid},
                        (Vector{MethodInstance}, Base.CodegenParams, Cint),
                        [method_instance], params, #=extern policy=# 1)
    @assert native_code != C_NULL
    llvm_mod_ref = ccall(:jl_get_llvm_module, LLVM.API.LLVMModuleRef,
                         (Ptr{Cvoid},), native_code)
    @assert llvm_mod_ref != C_NULL
    llvm_mod = LLVM.Module(llvm_mod_ref)

    # get the top-level code
    code = gpu_ci_cache_lookup(method_instance, world, world)

    # get the top-level function index
    llvm_func_idx = Ref{Int32}(-1)
    llvm_specfunc_idx = Ref{Int32}(-1)
    ccall(:jl_breakpoint, Nothing, ())
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
    if llvm_datalayout(job.target) !== nothing
        datalayout!(llvm_mod, llvm_datalayout(job.target))
    end

    return llvm_specfunc, llvm_mod
end
