# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber
using Base: _methods_by_ftype

using Serialization, Scratch

const compilelock = ReentrantLock()

@inline function get_interactive(cache, compiler, linker, spec, id; kwargs...)
    # generate a key for indexing the compilation cache
    key = hash(kwargs, id)
    key = hash(spec.name, key)      # fields f and tt are already covered by the id
    key = hash(spec.kernel, key)    # as `cached_compilation` specializes on them.
    for nf in 1:nfields(spec.f)
        # mix in the values of any captured variable
        key = hash(getfield(spec.f, nf), key)
    end

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(compilelock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing || compile_hook[] != nothing
            asm = compiler(spec; kwargs...)
            obj = linker(spec, asm)
            cache[key] = obj
        end
        obj
    finally
        unlock(compilelock)
    end
end

@inline function get_frozen(cache, compiler, linker, spec; kwargs...)
    # generate a key for indexing the compilation cache
    key = hash(kwargs)
    key = hash(spec, key)

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(compilelock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing
            path = joinpath(@get_scratch!("kernels"), "$key.jls")
            if isfile(path)
                @debug "Loading compiled kernel for $spec from $path"
                asm = deserialize(path)
            else
                asm = compiler(spec; kwargs...)
                serialize(path, asm)
            end
            obj = linker(spec, asm)
            cache[key] = obj
        end
        obj
    finally
        unlock(compilelock)
    end
end

# generated function that crafts a custom code info to call the actual cufunction impl.
# this gives us the flexibility to insert manual back edges for automatic recompilation.
#
# we also increment a global specialization counter and pass it along to index the cache.

specialization_counter = 0

const freeze_kernels = Ref(false)

@generated function cached_compilation(cache::Dict, compiler::Function, linker::Function,
                                       spec::FunctionSpec{f,tt}; kwargs...) where {f,tt}
    freeze_kernels[] && return quote
        get_frozen(cache, compiler, linker, spec; kwargs...)
    end

    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return (:(throw(MethodError(spec.f,spec.tt))))
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)
    @assert isa(ci, CodeInfo)

    # generate a unique id to represent this specialization
    global specialization_counter
    id = UInt(specialization_counter += 1)
    # TODO: save the mi/ci here (or embed it in the AST to pass to cufunction)
    #       and use that to drive compilation

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
    new_ci.slotnames = Symbol[:kwfunc, :kwargs, Symbol("#self#"),
                              :cache, :compiler, :linker, :spec]
    new_ci.slotflags = UInt8[0x00 for i = 1:7]
    kwargs = SlotNumber(2)
    cache = SlotNumber(4)
    compiler = SlotNumber(5)
    linker = SlotNumber(6)
    spec = SlotNumber(7)

    # call the compiler
    append!(new_ci.code, [Expr(:call, Core.kwfunc, get_interactive),
                          Expr(:call, merge, NamedTuple(), kwargs),
                          Expr(:call, SSAValue(1), SSAValue(2), get_interactive,
                                      cache, compiler, linker, spec, id),
                          Expr(:return, SSAValue(3))])
    append!(new_ci.codelocs, [1, 1, 1, 1])   # see note below
    new_ci.ssavaluetypes += 4

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end
