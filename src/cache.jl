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

if VERSION >= v"1.10.0-DEV.873"

# on 1.10 (JuliaLang/julia#48611) the generated function knows which world it was invoked in

function _generated_ex(world, source, ex)
    stub = Core.GeneratedFunctionStub(identity, Core.svec(:get_world, :ft, :tt), Core.svec())
    stub(world, source, ex)
end

function get_world_generator(world::UInt, source, self, ft::Type, tt::Type)
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # look up the method
    method_error = :(throw(MethodError(ft, tt, $world)))
    Base.isdispatchtuple(tt) || return _generated_ex(world, source, :(error("$tt is not a dispatch tuple")))
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
    mthds === nothing && return _generated_ex(world, source, method_error)
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

function get_world_generator(self, ft::Type, tt::Type)
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # look up the method
    method_error = :(throw(MethodError(ft, tt)))
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
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
    mthds === nothing && return method_error
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
    cached_compilation(cache::Dict{UInt}, job::CompilerJob, compiler, linker)

Compile `job` using `compiler` and `linker`, and store the result in `cache`.

The `cache` argument should be a dictionary that can be indexed using a `UInt` and store
whatever the `linker` function returns. The `compiler` function should take a `CompilerJob`
and return data that can be cached across sessions (e.g., LLVM IR). This data is then
forwarded, along with the `CompilerJob`, to the `linker` function which is allowed to create
session-dependent objects (e.g., a `CuModule`).
"""
function cached_compilation(cache::AbstractDict{UInt,V},
                            cfg::CompilerConfig,
                            ft::Type, tt::Type,
                            compiler::Function, linker::Function) where {V}
    # NOTE: it is OK to index the compilation cache directly with the world age, instead of
    #       intersecting world age ranges, because we the world age is aquired by calling
    #       `get_world` and thus will only change when the kernel function is redefined.
    world = get_world(ft, tt)
    key = hash(ft)
    key = hash(tt, key)
    key = hash(world, key)
    key = hash(cfg, key)

    # NOTE: no use of lock(::Function)/@lock/get! to avoid try/catch and closure overhead
    lock(cache_lock)
    obj = get(cache, key, nothing)
    unlock(cache_lock)

    LLVM.Interop.assume(isassigned(compile_hook))
    if obj === nothing || compile_hook[] !== nothing
        obj = actual_compilation(cache, key, cfg, ft, tt, world, compiler, linker)::V
    end
    return obj::V
end

@noinline function actual_compilation(cache::AbstractDict, key::UInt,
                                      cfg::CompilerConfig,
                                      ft::Type, tt::Type, world,
                                      compiler::Function, linker::Function)
    src = FunctionSpec(ft, tt, world)
    job = CompilerJob(src, cfg)

    asm = nothing
    # TODO: consider loading the assembly from an on-disk cache here

    # compile
    if asm === nothing
        if compile_hook[] !== nothing
            compile_hook[](job)
        end

        asm = compiler(job)
    end

    # link (but not if we got here because of forced compilation,
    # in which case the cache will already be populated)
    lock(cache_lock) do
        haskey(cache, key) && return cache[key]

        obj = linker(job, asm)
        cache[key] = obj
        obj
    end
end
