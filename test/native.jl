@testset "native" begin

include("definitions/native.jl")

############################################################################################

@testset "reflection" begin
    job, _ = native_job(identity, (Int,))

    @test only(GPUCompiler.code_lowered(job)) isa Core.CodeInfo

    ci, rt = only(GPUCompiler.code_typed(job))
    @test rt === Int

    ir = sprint(io->GPUCompiler.code_warntype(io, job))
    @test contains(ir, "MethodInstance for identity")

    ir = sprint(io->GPUCompiler.code_llvm(io, job))
    @test contains(ir, "julia_identity")

    asm = sprint(io->GPUCompiler.code_native(io, job))
    @test contains(asm, "julia_identity")
end

@testset "compilation" begin
    @testset "callable structs" begin
        struct MyCallable end
        (::MyCallable)(a, b) = a+b

        (ci, rt) = native_code_typed(MyCallable(), (Int, Int), kernel=false)[1]
        @test ci.slottypes[1] == Core.Compiler.Const(MyCallable())
    end

    @testset "compilation database" begin
        @noinline inner(x) = x+1
        function outer(x)
            return inner(x)
        end

        job, _ = native_job(outer, (Int,))
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(outer, (Int,)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            other_mis = filter(mi->mi.def != meth, keys(meta.compiled))
            @test length(other_mis) == 1
            @test only(other_mis).def in methods(inner)
        end
    end

    @testset "advanced database" begin
        @noinline inner(x) = x+1
        foo(x) = sum(inner, fill(x, 10, 10))

        job, _ = native_job(foo, (Float64,))
        JuliaContext() do ctx
            # shouldn't segfault
            ir, meta = GPUCompiler.compile(:llvm, job; validate=false)

            meth = only(methods(foo, (Float64,)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            inner_methods = filter(keys(meta.compiled)) do mi
                mi.def in methods(inner) && mi.specTypes == Tuple{typeof(inner), Float64}
            end
            @test length(inner_methods) == 1
        end
    end

    @testset "cached compilation" begin
        @gensym child kernel unrelated
        @eval @noinline $child(i) = i
        @eval $kernel(i) = $child(i)+1

        target = NativeCompilerTarget()
        params = TestCompilerParams()
        config = CompilerConfig(target, params; kernel=false)

        # smoke test
        job, _ = native_job(eval(kernel), (Int64,))
        ir = sprint(io->GPUCompiler.code_llvm(io, job))
        @test contains(ir, "add i64 %1, 1")

        # basic redefinition
        @eval $kernel(i) = $child(i)+2
        job, _ = native_job(eval(kernel), (Int64,))
        ir = sprint(io->GPUCompiler.code_llvm(io, job))
        @test contains(ir, "add i64 %1, 2")

        # cached_compilation interface
        invocations = Ref(0)
        function compiler(job)
            invocations[] += 1
            ir = sprint(io->GPUCompiler.code_llvm(io, job))
            return ir
        end
        linker(job, compiled) = compiled
        cache = Dict()
        ft = typeof(eval(kernel))
        tt = Tuple{Int64}

        # initial compilation
        source = methodinstance(ft, tt, Base.get_world_counter())
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, "add i64 %1, 2")
        @test invocations[] == 1

        # cached compilation
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, "add i64 %1, 2")
        @test invocations[] == 1

        # redefinition
        @eval $kernel(i) = $child(i)+3
        source = methodinstance(ft, tt, Base.get_world_counter())
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, "add i64 %1, 3")
        @test invocations[] == 2

        # cached compilation
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, "add i64 %1, 3")
        @test invocations[] == 2

        # redefinition of an unrelated function
        @eval $unrelated(i) = 42
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 2

        # redefining child functions
        @eval @noinline $child(i) = i+1
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 3

        # cached compilation
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 3

        # tasks running in the background should keep on using the old version
        c1, c2 = Condition(), Condition()
        function background(job)
            local_source = methodinstance(ft, tt, Base.get_world_counter())
            notify(c1)
            wait(c2)    # wait for redefinition
            GPUCompiler.cached_compilation(cache, local_source, job.config, compiler, linker)
        end
        t = @async Base.invokelatest(background, job)
        wait(c1)        # make sure the task has started
        @eval $kernel(i) = $child(i)+4
        source = methodinstance(ft, tt, Base.get_world_counter())
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, "add i64 %1, 4")
        notify(c2)      # wake up the task
        ir = fetch(t)
        @test contains(ir, "add i64 %1, 3")
    end
end

############################################################################################

@testset "IR" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    ir = sprint(io->native_code_llvm(io, valid_kernel, Tuple{}; optimize=false, dump_module=true))

    # module should contain our function + a generic call wrapper
    @test occursin(r"define\ .* void\ @.*julia_valid_kernel.*\(\)"x, ir)
    @test !occursin("define %jl_value_t* @jlcall_", ir)

    # there should be no debug metadata
    @test !occursin("!dbg", ir)

    @test native_code_llvm(devnull, invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError native_code_llvm(devnull, invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "unbound typevars" begin
    invalid_kernel() where {unbound} = return
    @test_throws KernelError native_code_llvm(devnull, invalid_kernel, Tuple{})
end

@testset "child functions" begin
    # we often test using `@noinline sink` child functions, so test whether these survive
    @noinline child(i) = sink(i)
    parent(i) = child(i)

    ir = sprint(io->native_code_llvm(io, parent, Tuple{Int}))
    @test occursin(r"call .+ @julia.+child.+", ir)
end

@testset "sysimg" begin
    # bug: use a system image function

    function foobar(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
    end

    ir = sprint(io->native_code_llvm(io, foobar, Tuple{Ptr{Int},Int}))
    @test !occursin("jlsys_", ir)
end

@testset "tracked pointers" begin
    function kernel(a)
        a[1] = 1
        return
    end

    # this used to throw an LLVM assertion (#223)
    native_code_llvm(devnull, kernel, Tuple{Vector{Int}}; kernel=true)
end

@testset "CUDAnative.jl#278" begin
    # codegen idempotency
    # NOTE: this isn't fixed, but surfaces here due to bad inference of checked_sub
    # NOTE: with the fix to print_to_string this doesn't error anymore,
    #       but still have a test to make sure it doesn't regress
    native_code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)
    native_code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)

    # breaking recursion in print_to_string makes it possible to compile
    # even in the presence of the above bug
    native_code_llvm(devnull, Base.print_to_string, Tuple{Int,Int}; optimize=false)
end

@testset "LLVM D32593" begin
    @eval struct D32593_struct
        foo::Float32
        bar::Float32
    end

    D32593(ptr) = unsafe_load(ptr).foo

    native_code_llvm(devnull, D32593, Tuple{Ptr{D32593_struct}})
end

@testset "slow abi" begin
    x = 2
    f = () -> x+1
    ir = sprint(io->native_code_llvm(io, f, Tuple{}, entry_abi=:func, dump_module=true))
    @test occursin(r"define nonnull {}\* @jfptr", ir)
    @test occursin(r"define internal fastcc .+ @julia", ir)
    @test occursin(r"call fastcc .+ @julia", ir)
end

@testset "function entry safepoint emission" begin
    ir = sprint(io->native_code_llvm(io, identity, Tuple{Nothing}; entry_safepoint=false, optimize=false, dump_module=true))
    @test !occursin("%safepoint", ir)

    if v"1.9.0-DEV.1660" <= VERSION < v"1.9.0-alpha1.57" || VERSION >= v"1.10-"
        ir = sprint(io->native_code_llvm(io, identity, Tuple{Nothing}; entry_safepoint=true, optimize=false, dump_module=true))
        @test occursin("%safepoint", ir)
    end
end

@testset "always_inline" begin
    @eval f_expensive(x) = $(foldl((e, _) -> :(sink($e) + sink(x)), 1:100; init=:x))
    function g(x)
        f_expensive(x)
        return
    end
    function h(x)
        f_expensive(x)
        return
    end

    ir = sprint(io->native_code_llvm(io, g, Tuple{Int64}; dump_module=true, kernel=true))
    @test occursin(r"^define.*julia_f_expensive"m, ir)

    ir = sprint(io->native_code_llvm(io, g, Tuple{Int64}; dump_module=true, kernel=true,
                                     always_inline=true))
    @test !occursin(r"^define.*julia_f_expensive"m, ir)

    ir = sprint(io->native_code_llvm(io, h, Tuple{Int64}; dump_module=true, kernel=true,
                                     always_inline=true))
    @test !occursin(r"^define.*julia_f_expensive"m, ir)

    ir = sprint(io->native_code_llvm(io, h, Tuple{Int64}; dump_module=true, kernel=true))
    @test occursin(r"^define.*julia_f_expensive"m, ir)
end

@testset "function attributes" begin
    @inline function convergent_barrier()
        Base.llvmcall(("""
            declare void @barrier() #1

            define void @entry() #0 {
                call void @barrier()
                ret void
            }

            attributes #0 = { alwaysinline }
            attributes #1 = { convergent }""", "entry"),
        Nothing, Tuple{})
    end

    ir = sprint(io->native_code_llvm(io, convergent_barrier, Tuple{}; dump_module=true, raw=true))
    @test occursin(r"attributes #. = \{ convergent \}", ir)
end

end

############################################################################################

@testset "assembly" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    @test native_code_native(devnull, valid_kernel, Tuple{}) == nothing
    @test native_code_native(devnull, invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError native_code_native(devnull, invalid_kernel, Tuple{}; kernel=true)
end

@testset "idempotency" begin
    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)

    kernel() = return
    native_code_native(devnull, kernel, Tuple{})
    native_code_native(devnull, kernel, Tuple{})
end

@testset "compile for host after gpu" begin
    # issue #11: re-using host functions after GPU compilation
    @noinline child(i) = sink(i+1)

    function fromhost()
        child(10)
    end

    function fromptx()
        child(10)
        return
    end

    native_code_native(devnull, fromptx, Tuple{})
    @test fromhost() == 11
end

end

############################################################################################

@testset "errors" begin

@eval Main begin
struct CleverType{T}
    x::T
end
Base.unsafe_trunc(::Type{Int}, x::CleverType) = unsafe_trunc(Int, x.x)
end

@testset "non-isbits arguments" begin
    foobar(i) = (sink(unsafe_trunc(Int,i)); return)

    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{BigInt})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("BigInt", msg)
    end

    # test that we get information about fields and reason why something is not isbits
    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{CleverType{BigInt}})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("CleverType", msg) &&
        occursin("BigInt", msg)
    end
end

@testset "invalid LLVM IR" begin
    foobar(i) = println(i)

    @test_throws_message(InvalidIRError,
                         native_code_execution(foobar, Tuple{Int})) do msg
        occursin("invalid LLVM IR", msg) &&
        (occursin(GPUCompiler.RUNTIME_FUNCTION, msg) ||
         occursin(GPUCompiler.UNKNOWN_FUNCTION, msg) ||
         occursin(GPUCompiler.DYNAMIC_CALL, msg)) &&
        occursin("[1] println", msg) &&
        occursin(r"\[2\] .*foobar", msg)
    end
end

@testset "invalid LLVM IR (ccall)" begin
    foobar(p) = (unsafe_store!(p, ccall(:time, Cint, ())); nothing)

    @test_throws_message(InvalidIRError,
                         native_code_execution(foobar, Tuple{Ptr{Int}})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.POINTER_FUNCTION, msg) &&
        occursin(r"\[1\] .*foobar", msg)
    end
end

@testset "delayed bindings" begin
    kernel() = (undefined; return)

    @test_throws_message(InvalidIRError,
                         native_code_execution(kernel, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DELAYED_BINDING, msg) &&
        occursin("use of 'undefined'", msg) &&
        occursin(r"\[1\] .*kernel", msg)
    end
end

@testset "dynamic call (invoke)" begin
    @eval @noinline nospecialize_child(@nospecialize(i)) = i
    kernel(a, b) = (unsafe_store!(b, nospecialize_child(a)); return)

    @test_throws_message(InvalidIRError,
                         native_code_execution(kernel, Tuple{Int,Ptr{Int}})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to nospecialize_child", msg) &&
        occursin(r"\[1\] .+kernel", msg)
    end
end

@testset "dynamic call (apply)" begin
    func() = println(1)

    @test_throws_message(InvalidIRError,
                         native_code_execution(func, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to println", msg) &&
        occursin("[2] func", msg)
    end
end

end

############################################################################################

@testset "LazyCodegen" begin
    import .LazyCodegen: call_delayed

    f(A) = (A[] += 42; nothing)

    global flag = [0]
    function caller()
        call_delayed(f, flag::Vector{Int})
    end
    @test caller() === nothing
    @test flag[] == 42

    ir = sprint(io->native_code_llvm(io, caller, Tuple{}, dump_module=true))
    @test_broken occursin(r"add i64 %\d+, 42", ir)
    # NOTE: can't just look for `jl_f` here, since it may be inlined and optimized away.

    add(x, y) = x+y
    function call_add(x, y)
        call_delayed(add, x, y)
    end

    @test call_add(1, 3) == 4

    incr(r) = r[] += 1
    function call_incr(r)
        call_delayed(incr, r)
    end
    r = Ref{Int}(0)
    @test call_incr(r) == 1
    @test r[] == 1

    function call_real(c)
        call_delayed(real, c)
    end

    @test call_real(1.0+im) == 1.0

    # Test ABI removal
    # XXX: this relies on llvm_always_inline, which it shouldn't
    ir = sprint(io->native_code_llvm(io, call_real, Tuple{ComplexF64}))
    @test_broken !occursin("alloca", ir)

    ghostly_identity(x, y) = y
    @test call_delayed(ghostly_identity, nothing, 1) == 1

    # tests struct return
    if Sys.ARCH != :aarch64
        @test call_delayed(complex, 1.0, 2.0) == 1.0+2.0im
    else
        @test_broken call_delayed(complex, 1.0, 2.0) == 1.0+2.0im
    end

    throws(arr, i) = arr[i]
    @test call_delayed(throws, [1], 1) == 1
    @test_throws BoundsError call_delayed(throws, [1], 0)

    struct Closure
        x::Int64
    end
    (c::Closure)(b) = c.x+b

    @test call_delayed(Closure(3), 5) == 8

    struct Closure2
        x::Integer
    end
    (c::Closure2)(b) = c.x+b

    @test call_delayed(Closure2(3), 5) == 8
end

############################################################################################

@testset "overrides" begin
    # NOTE: method overrides do not support redefinitions, so we use different kernels

    mod = @eval module $(gensym())
        kernel() = child()
        child() = 0
    end

    ir = sprint(io->native_code_llvm(io, mod.kernel, Tuple{}))
    @test occursin("ret i64 0", ir)

    mod = @eval module $(gensym())
        using ..GPUCompiler

        @static if isdefined(Base.Experimental, Symbol("@overlay"))
        Base.Experimental.@MethodTable(method_table)
        else
        const method_table = nothing
        end

        kernel() = child()
        child() = 0

        Base.Experimental.@overlay method_table child() = 1
    end

    ir = sprint(io->native_code_llvm(io, mod.kernel, Tuple{}; mod.method_table))
    @test occursin("ret i64 1", ir)
end

@testset "#366: semi-concrete interpretation + overlay methods = dynamic dispatch" begin
    mod = @eval module $(gensym())
        using ..GPUCompiler
        using StaticArrays

        function kernel(width, height)
            xy = SVector{2, Float32}(0.5f0, 0.5f0)
            res = SVector{2, UInt32}(width, height)
            floor.(UInt32, max.(0f0, xy) .* res)
            return
        end

        Base.Experimental.@MethodTable method_table
        Base.Experimental.@overlay method_table Base.isnan(x::Float32) =
            (ccall("extern __nv_isnanf", llvmcall, Int32, (Cfloat,), x)) != 0
    end

    ir = sprint(io->native_code_llvm(io, mod.kernel, Tuple{Int, Int};
                                     debuginfo=:none, mod.method_table))
    @test !occursin("apply_generic", ir)
    @test occursin("llvm.floor", ir)
end

@testset "JuliaLang/julia#48097: kwcall inference in the presence of overlay method" begin
    mod = @eval module $(gensym())
        using ..GPUCompiler
        child(; kwargs...) = return
        function parent()
            child(; a=1f0, b=1.0)
            return
        end

        Base.Experimental.@MethodTable method_table
        Base.Experimental.@overlay method_table @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
        return
    end

    ir = sprint(io->native_code_llvm(io, mod.parent, Tuple{};
                                     debuginfo=:none, mod.method_table))
    @test !occursin("jl_invoke", ir)
    @test !occursin("apply_iterate", ir)
    @test !occursin("inttoptr", ir)
    @test occursin("ret void", ir)
end

############################################################################################

end
