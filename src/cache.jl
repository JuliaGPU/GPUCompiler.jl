# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

# generated function that returns the world age of a compilation job. this can be used to
# drive compilation, e.g. by using it as a key for a cache, as the age will change when a
# function or any called function is redefined.


"""
    get_world(ft, tt)

A special function that returns the world age in which the current definition of function
type `ft`, invoked with argument types `tt`, is defined. This can be used to cache
compilation results:

    compilation_cache = Dict()
    function cache_compilation(ft, tt)
        world = get_world(ft, tt)
        get!(compilation_cache, (ft, tt, world)) do
            # compile
        end
    end

What makes this function special is that it is a generated function, returning a constant,
whose result is automatically invalidated when the function `ft` (or any called function) is
redefined. This makes this query ideally suited for hot code, where you want to avoid a
costly look-up of the current world age on every invocation.

Normally, you shouldn't have to use this function, as it's used by `FunctionSpec`.

!!! warning

    Due to a bug in Julia, JuliaLang/julia#34962, this function's results are only
    guaranteed to be correctly invalidated when the target function `ft` is executed or
    processed by codegen (e.g., by calling `code_llvm`).
"""
get_world

if VERSION >= v"1.10.0-DEV.649"

# on 1.10 (JuliaLang/julia#48611) the generated function knows which world it was invoked in

function _generated_ex(world, source, ex)
    stub = Core.GeneratedFunctionStub(identity, Core.svec(:get_world, :ft, :tt), Core.svec())
    stub(world, source, ex)
end

function get_world_generator(world::UInt, source, self, ::Type{Type{ft}}, ::Type{Type{tt}}) where {ft, tt}
    @nospecialize

    # look up the method
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    mthds = if VERSION >= v"1.7.0-DEV.1297"
        Base._methods_by_ftype(sig, #=mt=# nothing, #=lim=# -1,
                               world, #=ambig=# false,
                               min_world, max_world, has_ambig)
        # XXX: use the correct method table to support overlaying kernels
    else
        Base._methods_by_ftype(sig, #=lim=# -1,
                               world, #=ambig=# false,
                               min_world, max_world, has_ambig)
    end

    # check the validity of the method matches
    method_error = :(throw(MethodError(ft, tt, $world)))
    mthds === nothing && return _generated_ex(world, source, method_error)
    Base.isdispatchtuple(tt) || return _generated_ex(world, source, :(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return _generated_ex(world, source, method_error)

    # look up the method and code instance
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
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]

    # return the world
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

@eval function get_world(ft, tt)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, get_world_generator))
end

else

# on older versions of Julia we fall back to looking up the current world. this may be wrong
# when the generator is invoked in a different world (TODO: when does this happen?)

function get_world_generator(self, ::Type{Type{ft}}, ::Type{Type{tt}}) where {ft, tt}
    @nospecialize

    # look up the method
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    mthds = if VERSION >= v"1.7.0-DEV.1297"
        Base._methods_by_ftype(sig, #=mt=# nothing, #=lim=# -1,
                               #=world=# typemax(UInt), #=ambig=# false,
                               min_world, max_world, has_ambig)
        # XXX: use the correct method table to support overlaying kernels
    else
        Base._methods_by_ftype(sig, #=lim=# -1,
                               #=world=# typemax(UInt), #=ambig=# false,
                               min_world, max_world, has_ambig)
    end
    # XXX: using world=-1 is wrong, but the current world isn't exposed to this generator

    # check the validity of the method matches
    method_error = :(throw(MethodError(ft, tt)))
    mthds === nothing && return method_error
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return method_error

    # look up the method and code instance
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)::CodeInfo

    # XXX: we don't know the world age that this generator was requested to run in, so use
    # the current world (we cannot use the mi's world because that doesn't update when
    # called functions are changed). this isn't correct, but should be close.
    world = Base.get_world_counter()

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    resize!(new_ci.linetable, 1)                # see note below
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]

    # return the world
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

@eval function get_world(ft, tt)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta,
           :generated,
           Expr(:new,
                Core.GeneratedFunctionStub,
                :get_world_generator,
                Any[:get_world, :ft, :tt],
                Any[],
                @__LINE__,
                QuoteNode(Symbol(@__FILE__)),
                true)))
end

end

const cache_lock = ReentrantLock()

"""
    cached_compilation(cache::Dict, job::CompilerJob, compiler, linker)

Compile `job` using `compiler` and `linker`, and store the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using a `UInt` and store
whatever the `linker` function returns. The `compiler` function should take a `CompilerJob`
and return data that can be cached across sessions (e.g., LLVM IR). This data is then
forwarded, along with the `CompilerJob`, to the `linker` function which is allowed to create
session-dependent objects (e.g., a `CuModule`).
"""
function cached_compilation(cache::AbstractDict,
                            @nospecialize(job::CompilerJob),
                            compiler::Function, linker::Function)
    # NOTE: it is OK to index the compilation cache directly with the compilation job, i.e.,
    #       using a world age instead of intersecting world age ranges, because we expect
    #       that the world age is aquired through calling `get_world` and thus will only
    #       ever change when the kernel function is redefined.
    #
    #       if we ever want to be able to index the cache using a compilation job that
    #       contains a more recent world age, yet still return an older cached object that
    #       would still be valid, we'd need the cache to store world ranges instead and
    #       use an invalidation callback to add upper bounds to entries.
    key = hash(job)

    force_compilation = compile_hook[] !== nothing

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
