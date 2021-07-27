# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

# generated function that crafts a custom code info to call the actual compiler.
# this gives us the flexibility to insert manual back edges for automatic recompilation.
#
# we also increment a global specialization counter and pass it along to index the cache.

const specialization_counter = Ref{UInt}()
@generated function specialization_id(f::F, tt::Type{TT}) where {F, TT}
    # get a hold of the method and code info of the kernel function
    sig = Tuple{F, TT.parameters...}
    # XXX: instead of typemax(UInt) we should use the world-age of the fspec
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    Base.isdispatchtuple(TT) || return(:(error("$TT is not a dispatch tuple")))
    length(mthds) == 1 || return (:(throw(MethodError(f,tt))))
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
    new_ci.slotnames = Symbol[Symbol("#self#"), :cache, :job, :compiler, :linker]
    new_ci.slotflags = UInt8[0x00 for i = 1:5]
    cache = SlotNumber(2)
    job = SlotNumber(3)
    compiler = SlotNumber(4)
    linker = SlotNumber(5)

    # call the compiler
    append!(new_ci.code, [ReturnNode(id)])
    append!(new_ci.codelocs, [1])   # see note below
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

# we need compile-time access to the job's source function and tuple type, but that isn't
# part of the CompilerJob's type parameters (because we generally don't want to specialize
# the compiler on the function being compiled). instead of using additional arguments,
# just look it up from an inlined function (assuming the parent will have specialized
# on the function and tuple type used to construct the CompilerJob).
@inline cached_compilation(cache::AbstractDict, job::CompilerJob,
                           compiler::Function, linker::Function) =
    _cached_compilation(cache, job, compiler, linker, job.source.f, job.source.tt)

# NOTE: questionable use of `@inline` to avoid allocations in the kernel launch path
const cache_lock = ReentrantLock()
@inline function _cached_compilation(cache::AbstractDict, job::CompilerJob,
                                     compiler::Function, linker::Function,
                                     f::F, tt::Type{TT}) where {F, TT}
    # XXX: CompilerJob contains a world age, so can't be respecialized.
    #      have specialization_id take a f/tt and return a world to construct a CompilerJob?
    key = hash(job, specialization_id(f, tt))
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
