# cached compilation

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber
using Base: _methods_by_ftype

const compilecache = Dict{UInt, Any}()
const compilelock = ReentrantLock()

@inline function check_cache(driver, f, tt, id; kwargs...)
    # generate a key for indexing the compilation cache
    key = hash(kwargs, id)
    for nf in 1:nfields(f)
        # mix in the values of any captured variable
        key = hash(getfield(f, nf), key)
    end

    Base.@lock compilelock begin
        get!(compilecache, key) do
            driver(f, tt; kwargs...)
        end
    end
end

specialization_counter = 0

# TODO: use FunctionSpec
@generated function cached_compilation(driver::Core.Function, f::Core.Function,
                                       tt::Type=Tuple, env::UInt=zero(UInt); kwargs...)
    # generated function that crafts a custom code info to call the actual cufunction impl.
    # this gives us the flexibility to insert manual back edges for automatic recompilation.
    tt = tt.parameters[1]

    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return (:(throw(MethodError(f,tt))))
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
    empty!(new_ci.linetable)
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which CUDAnative does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[:kwfunc, :kwargs, Symbol("#self#"), :driver, :f, :tt, :id]
    new_ci.slotflags = UInt8[0x00 for i = 1:7]
    kwargs = SlotNumber(2)
    driver = SlotNumber(4)
    f = SlotNumber(5)
    tt = SlotNumber(6)
    env = SlotNumber(7)

    # call the compiler
    append!(new_ci.code, [Expr(:call, Core.kwfunc, check_cache),
                          Expr(:call, merge, NamedTuple(), kwargs),
                          Expr(:call, hash, env, id),
                          Expr(:call, SSAValue(1), SSAValue(2), check_cache, driver, f, tt, SSAValue(3)),
                          Expr(:return, SSAValue(4))])
    append!(new_ci.codelocs, [0, 0, 0, 0, 0])
    new_ci.ssavaluetypes += 5

    return new_ci
end
