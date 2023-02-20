# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

function function_instance(ft)
    if isdefined(ft, :instance)
        return ft.instance
    else
        # dealing with a closure, for which we cannot construct an instance.
        # however, we only use this in the context of method errors, where
        # we really only care about the type of the function, so do something invalid:
        Ref{ft}()[]
    end
end

# generated function that returns an increasing id for each specialization of a function.
# this can be used to index a compilation cache, and only recompile kernels when necessary.

if VERSION >= v"1.10.0-DEV.649"  # JuliaLang/julia#48611

function specialization_id_ex(world, source, ex)
    stub = Core.GeneratedFunctionStub(identity, Core.svec(:specialization_id, :job), Core.svec())
    stub(world, source, ex)
end

function specialization_id_generator(world::UInt, source, self,
                                     job::Type{<:CompilerJob{<:Any,<:Any,FunctionSpec{f,tt}}}) where {f,tt}
    @nospecialize

    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    mthds = _methods_by_ftype(sig, -1, world)
    method_error = :(throw(MethodError($(function_instance(f)), job.source.tt, $world)))
    mthds === nothing && return specialization_id_ex(world, source, method_error)
    Base.isdispatchtuple(tt) || return specialization_id_ex(world, source, :(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return specialization_id_ex(world, source, method_error)
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi, world)::CodeInfo

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    resize!(new_ci.linetable, 1)                # see note below
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :job]
    new_ci.slotflags = UInt8[0x00 for i = 1:2]

    # call the compiler
    push!(new_ci.code, ReturnNode(world))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    push!(new_ci.codelocs, 1)   # see note below
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

@eval function specialization_id(job)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, specialization_id_generator))
end

else

const specialization_counter = Ref{UInt}(0)
function specialization_id_generator(self, job::Type{<:CompilerJob{<:Any,<:Any,FunctionSpec{f,tt}}}) where {f,tt}
    @nospecialize

    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    # XXX: instead of typemax(UInt) we should use the world-age of the fspec
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    method_error = :(throw(MethodError($(function_instance(f)), job.source.tt)))
    mthds === nothing && return method_error
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return method_error
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)::CodeInfo

    # generate a unique id to represent this specialization
    # TODO: just use the lower world age bound in which this code info is valid.
    #       (the method instance doesn't change when called functions are changed).
    #       but how to get that? the ci here always has min/max world 1/-1.
    # XXX: don't use `objectid(ci)` here, apparently it can alias (or the CI doesn't change?)
    id = (specialization_counter[] += 1)

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    resize!(new_ci.linetable, 1)                # see note below
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :job]
    new_ci.slotflags = UInt8[0x00 for i = 1:2]

    # call the compiler
    push!(new_ci.code, ReturnNode(id))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    push!(new_ci.codelocs, 1)   # see note below
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

@eval function specialization_id(job)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta,
            :generated,
            Expr(:new,
                Core.GeneratedFunctionStub,
                :specialization_id_generator,
                Any[:specialization_id, :job],
                Any[],
                @__LINE__,
                QuoteNode(Symbol(@__FILE__)),
                true)))
end

end

const cache_lock = ReentrantLock()
function cached_compilation(cache::AbstractDict,
                            @nospecialize(job::CompilerJob),
                            compiler::Function, linker::Function)
    # XXX: CompilerJob contains a world age, so can't be respecialized.
    #      have specialization_id take a f/tt and return a world to construct a CompilerJob?
    key = hash(job, specialization_id(job))
    force_compilation = compile_hook[] !== nothing

    # XXX: by taking the hash, we index the compilation cache directly with the world age.
    #      that's wrong; we should perform an intersection with the entry its bounds.

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(cache_lock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing || force_compilation
            asm = nothing

            # compile
            if asm === nothing
                if compile_hook[] !== nothing
                    compile_hook[](job)
                end

                asm = compiler(job)
            end

            # link (but not if we got here because of forced compilation)
            if obj === nothing
                obj = linker(job, asm)
                cache[key] = obj
            end
        end
        obj
    finally
        unlock(cache_lock)
    end
end
