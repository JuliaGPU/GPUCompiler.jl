@testsetup module CompilerPlugins

using Test
using ReTestItems

Base.Experimental.@MethodTable(FMAMT)

for (jlf, f) in zip((:+, :*, :-), (:add, :mul, :sub))
    for (T, llvmT) in ((:Float32, "float"), (:Float64, "double"))
        ir = """
            %x = f$f contract nsz $llvmT %0, %1
            ret $llvmT %x
        """
        @eval begin
            # the @pure is necessary so that we can constant propagate.
            Base.Experimental.@overlay FMAMT @inline Base.@pure function $jlf(a::$T, b::$T)
                Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
            end
        end
    end
end

# Define Compiler plugin that will replace methods with their contract version

import Core.Compiler as CC
import GPUCompiler.CCMixin
import GPUCompiler

struct FMACompiler <: CCMixin.AbstractCompiler
    parent_mt::CC.MethodTableView
end
FMACompiler() = FMACompiler(CCMixin.current_method_table())
# CCMixin.compiler_world(::FMACompiler) = COMPILER_WORLD[]
CCMixin.abstract_interpreter(compiler::FMACompiler, world::UInt) =
    FMAInterp(compiler; world)

struct FMAInterp <: CCMixin.AbstractGPUInterpreter
    compiler::FMACompiler
@static if !GPUCompiler.HAS_INTEGRATED_CACHE
    code_cache::GPUCompiler.CodeCache
end
    world::UInt
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
    inf_cache::Vector{CC.InferenceResult}
    function FMAInterp(compiler::FMACompiler;
                world::UInt = Base.get_world_counter(),
                inf_params::CC.InferenceParams = CC.InferenceParams(),
                opt_params::CC.OptimizationParams = CC.OptimizationParams(),
                inf_cache::Vector{CC.InferenceResult} = CC.InferenceResult[])
        @static if !GPUCompiler.HAS_INTEGRATED_CACHE
            # TODO get a cache here properly...
            return new(compiler, GPUCompiler.CodeCache(), world, inf_params, opt_params, inf_cache)
        end
        return new(compiler, world, inf_params, opt_params, inf_cache)
    end
end

@static if VERSION >= v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end

CC.InferenceParams(interp::FMAInterp) = interp.inf_params
CC.OptimizationParams(interp::FMAInterp) = interp.opt_params
get_inference_world(interp::FMAInterp) = interp.world
CC.get_inference_cache(interp::FMAInterp) = interp.inf_cache
if GPUCompiler.HAS_INTEGRATED_CACHE
    CC.cache_owner(interp::FMAInterp) = interp.compiler
else
    CC.code_cache(interp::FMAInterp) = CC.WorldView(interp.code_cache, interp.world)
end
CC.method_table(interp::FMAInterp) = StackedMethodTable(get_inference_world(interp),  FMAMT, interp.compiler.parent_mt)




# vchuravy/Shenanigans.jl

# In a stack MT the lower one takes priority

import Core: MethodTable
import Core.Compiler: MethodTableView, InternalMethodTable,
                        MethodMatchResult, MethodLookupResult, WorldRange
struct StackedMethodTable{MTV<:MethodTableView} <: MethodTableView
    world::UInt
    mt::MethodTable
    parent::MTV
end
StackedMethodTable(world::UInt, mt::MethodTable) = StackedMethodTable(world, mt, InternalMethodTable(world))
StackedMethodTable(world::UInt, mt::MethodTable, parent::MethodTable) = StackedMethodTable(world, mt, StackedMethodTable(world, parent))

import Core.Compiler: findall, _findall, length, vcat, isempty, max, min, getindex
function findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = _findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing # to many matches
    nr = length(result)
    if nr ≥ 1 && getindex(result, nr).fully_covers
        # no need to fall back to the parent method view
        return MethodMatchResult(result, true)
    end

    parent_result = findall(sig, table.parent; limit)::Union{Nothing, MethodMatchResult}
    parent_result === nothing && return nothing #too many matches

    overlayed = parent_result.overlayed | !isempty(result)
    parent_result = parent_result.matches::MethodLookupResult
    
    # merge the parent match results with the internal method table
    return MethodMatchResult(
    MethodLookupResult(
        vcat(result.matches, parent_result.matches),
        WorldRange(
            max(result.valid_worlds.min_world, parent_result.valid_worlds.min_world),
            min(result.valid_worlds.max_world, parent_result.valid_worlds.max_world)),
        result.ambig | parent_result.ambig),
    overlayed)
end

import Core.Compiler: isoverlayed
isoverlayed(::StackedMethodTable) = true

import Core.Compiler: findsup, _findsup
function findsup(@nospecialize(sig::Type), table::StackedMethodTable)
    match, valid_worlds = _findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds, true
    # look up in parent
    parent_match, parent_valid_worlds, overlayed = findsup(sig, table.parent)
    return (
        parent_match,
        WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds.min_world),
            min(valid_worlds.max_world, parent_valid_worlds.max_world)),
        overlayed)
end
