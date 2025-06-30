module Enzyme

using ..GPUCompiler

struct EnzymeTarget{Target<:AbstractCompilerTarget} <: AbstractCompilerTarget
    target::Target
end

function EnzymeTarget(;kwargs...)
    EnzymeTarget(GPUCompiler.NativeCompilerTarget(; jlruntime = true, kwargs...))
end

GPUCompiler.llvm_triple(target::EnzymeTarget) = GPUCompiler.llvm_triple(target.target)
GPUCompiler.llvm_datalayout(target::EnzymeTarget) = GPUCompiler.llvm_datalayout(target.target)
GPUCompiler.llvm_machine(target::EnzymeTarget) = GPUCompiler.llvm_machine(target.target)
GPUCompiler.nest_target(::EnzymeTarget, other::AbstractCompilerTarget) = EnzymeTarget(other)
GPUCompiler.have_fma(target::EnzymeTarget, T::Type) = GPUCompiler.have_fma(target.target, T)
GPUCompiler.dwarf_version(target::EnzymeTarget) = GPUCompiler.dwarf_version(target.target)

abstract type AbstractEnzymeCompilerParams <: AbstractCompilerParams end
struct EnzymeCompilerParams{Params<:AbstractCompilerParams} <: AbstractEnzymeCompilerParams
    params::Params
end
struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
end

EnzymeCompilerParams() = EnzymeCompilerParams(PrimalCompilerParams())

GPUCompiler.nest_params(::EnzymeCompilerParams, other::AbstractCompilerParams) = EnzymeCompilerParams(other)

function GPUCompiler.compile_unhooked(output::Symbol, job::CompilerJob{<:EnzymeTarget})
    config = job.config
    primal_target = (job.config.target::EnzymeTarget).target
    primal_params = (job.config.params::EnzymeCompilerParams).params

    primal_config = CompilerConfig(
        primal_target,
        primal_params;
        toplevel = config.toplevel,
        always_inline = config.always_inline,
        kernel = false,
        libraries = true,
        optimize = false,
        cleanup = false,
        only_entry = false,
        validate = false,
        # ??? entry_abi
    )
    primal_job = CompilerJob(job.source, primal_config, job.world)
    return GPUCompiler.compile_unhooked(output, primal_job)

    # Normally, Enzyme would run here and transform the output of the primal job.
end

import GPUCompiler: deferred_codegen_jobs
import Core.Compiler as CC

function deferred_codegen_id_generator(world::UInt, source, self, ft::Type, tt::Type)
    @nospecialize
    @assert CC.isType(ft) && CC.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:deferred_codegen_id, :ft, :tt), Core.svec())

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
    mi = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
    ci = CC.retrieve_code_info(mi, world)

    # prepare a new code info
    # TODO: Can we create a new CI instead of copying a "wrong" one?
    new_ci = copy(ci)
    empty!(new_ci.code)
    @static if isdefined(Core, :DebugInfo)
      new_ci.debuginfo = Core.DebugInfo(:none)
    else
      empty!(new_ci.codelocs)
      resize!(new_ci.linetable, 1)                # see note below
    end
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0

    # propagate edge metadata
    # new_ci.min_world = min_world[]
    new_ci.min_world = world
    new_ci.max_world = max_world[]
    new_ci.edges = Core.MethodInstance[mi]

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]
    @static if isdefined(Core, :DebugInfo)
        new_ci.nargs = 3
    end

    # We don't know the caller's target so EnzymeTarget uses the default NativeCompilerTarget.
    target = EnzymeTarget()
    params = EnzymeCompilerParams()
    config = CompilerConfig(target, params; kernel=false)
    job = CompilerJob(mi, config, world)

    id = length(deferred_codegen_jobs) + 1
    deferred_codegen_jobs[id] = job

    # return the deferred_codegen_id
    push!(new_ci.code, CC.ReturnNode(id))
    push!(new_ci.ssaflags, 0x00)
        @static if isdefined(Core, :DebugInfo)
    else
      push!(new_ci.codelocs, 1)   # see note below
    end
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

@eval function deferred_codegen_id(ft, tt)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, deferred_codegen_id_generator))
end

@inline function deferred_codegen(f::Type, tt::Type)
    id = deferred_codegen_id(f, tt)
    ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), id)
end

end