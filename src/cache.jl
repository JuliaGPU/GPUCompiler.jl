# compilation cache

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

using Serialization, Scratch

const compilelock = ReentrantLock()

# whether to cache compiled kernels on disk or not
const disk_cache = Ref(false)

@inline function check_cache(cache, job, prekey,
                             @nospecialize(compiler), @nospecialize(linker))
    key = hash(job, prekey)
    force_compilation = compile_hook[] !== nothing

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(compilelock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing || force_compilation
            asm = nothing

            # can we load from the disk cache?
            if disk_cache[] && !force_compilation
                path = joinpath(@get_scratch!("kernels"), "$key.jls")
                if isfile(path)
                    try
                        asm = deserialize(path)
                        @debug "Loading compiled kernel for $spec from $path"
                    catch ex
                        @warn "Failed to load compiled kernel at $path" exception=(ex, catch_backtrace())
                    end
                end
            end

            # compile
            if asm === nothing
                if compile_hook[] !== nothing
                    compile_hook[](job)
                end

                asm = compiler(job)

                if disk_cache[] && !isfile(path)
                    serialize(path, asm)
                end
            end

            # link (but not if we got here because of forced compilation)
            if obj === nothing
                obj = linker(job, asm)
                cache[key] = obj
            end
        end
        obj
    finally
        unlock(compilelock)
    end
end

# generated function that crafts a custom code info to call the actual compiler.
# this gives us the flexibility to insert manual back edges for automatic recompilation.
#
# we also increment a global specialization counter and pass it along to index the cache.

const specialization_counter = Ref{UInt}(0)

@generated function cached_compilation(cache::Dict,
                                       job::CompilerJob{<:Any,<:Any,FunctionSpec{f,tt}},
                                       compiler::Function, linker::Function) where {f,tt}
    # get a hold of the method and code info of the kernel function
    sig = Tuple{f, tt.parameters...}
    mthds = _methods_by_ftype(sig, -1, typemax(UInt))
    Base.isdispatchtuple(tt) || return(:(error("$tt is not a dispatch tuple")))
    length(mthds) == 1 || return (:(throw(MethodError(job.source.f,job.source.tt))))
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)
    @assert isa(ci, CodeInfo)

    # generate a unique id to represent this specialization
    id = (specialization_counter[] += 1)
    # TODO: save the mi/ci here (or embed it in the AST to pass to the compiler)
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
    new_ci.slotnames = Symbol[Symbol("#self#"), :cache, :job, :compiler, :linker]
    new_ci.slotflags = UInt8[0x00 for i = 1:5]
    cache = SlotNumber(2)
    job = SlotNumber(3)
    compiler = SlotNumber(4)
    linker = SlotNumber(5)

    # call the compiler
    append!(new_ci.code, [Expr(:call, check_cache, cache, job, id, compiler, linker),
                          ReturnNode(SSAValue(1))])
    append!(new_ci.codelocs, [1, 1])   # see note below
    new_ci.ssavaluetypes += 2

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end
