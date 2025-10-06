@testset "reflection" begin
    job, _ = Native.create_job(identity, (Int,))

    @test only(GPUCompiler.code_lowered(job)) isa Core.CodeInfo

    ci, rt = only(GPUCompiler.code_typed(job))
    @test rt === Int

    @test @filecheck begin
        check"CHECK: MethodInstance for identity"
        GPUCompiler.code_warntype(job)
    end

    @test @filecheck begin
        check"CHECK: @{{(julia|j)_identity_[0-9]+}}"
        GPUCompiler.code_llvm(job)
    end

    @test @filecheck begin
        check"CHECK: @{{(julia|j)_identity_[0-9]+}}"
        GPUCompiler.code_native(job)
    end
end

@testset "compilation" begin
    @testset "callable structs" begin
        mod = @eval module $(gensym())
            struct MyCallable end
            (::MyCallable)(a, b) = a+b
        end

        (ci, rt) = Native.code_typed(mod.MyCallable(), (Int, Int), kernel=false)[1]
        @test ci.slottypes[1] == Core.Compiler.Const(mod.MyCallable())
    end

    @testset "compilation database" begin
        mod = @eval module $(gensym())
            @noinline inner(x) = x+1
            function outer(x)
                return inner(x)
            end
        end

        job, _ = Native.create_job(mod.outer, (Int,))
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(mod.outer, (Int,)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            other_mis = filter(mi->mi.def != meth, keys(meta.compiled))
            @test length(other_mis) == 1
            @test only(other_mis).def in methods(mod.inner)
        end
    end

    @testset "advanced database" begin
        mod = @eval module $(gensym())
            @noinline inner(x) = x+1
            foo(x) = sum(inner, fill(x, 10, 10))
        end

        job, _ = Native.create_job(mod.foo, (Float64,); validate=false)
        JuliaContext() do ctx
            # shouldn't segfault
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(mod.foo, (Float64,)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            inner_methods = filter(keys(meta.compiled)) do mi
                mi.def in methods(mod.inner) &&
                mi.specTypes == Tuple{typeof(mod.inner), Float64}
            end
            @test length(inner_methods) == 1
        end
    end

    @testset "cached compilation" begin
        mod = @eval module $(gensym())
            @noinline child(i) = i
            kernel(i) = child(i)+1
        end

        # smoke test
        job, _ = Native.create_job(mod.kernel, (Int64,))
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 1"
            GPUCompiler.code_llvm(job)
        end

        # basic redefinition
        @eval mod kernel(i) = child(i)+2
        job, _ = Native.create_job(mod.kernel, (Int64,))
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 2"
            GPUCompiler.code_llvm(job)
        end

        # cached_compilation interface
        invocations = Ref(0)
        function compiler(job)
            invocations[] += 1
            JuliaContext() do ctx
                ir, ir_meta = GPUCompiler.compile(:llvm, job)
                string(ir)
            end
        end
        linker(job, compiled) = compiled
        cache = Dict()
        ft = typeof(mod.kernel)
        tt = Tuple{Int64}

        # initial compilation
        source = methodinstance(ft, tt, Base.get_world_counter())
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 2"
            Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        end
        @test invocations[] == 1

        # cached compilation
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 2"
            Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        end
        @test invocations[] == 1

        # redefinition
        @eval mod kernel(i) = child(i)+3
        source = methodinstance(ft, tt, Base.get_world_counter())
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 3"
            Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        end
        @test invocations[] == 2

        # cached compilation
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 3"
            Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        end
        @test invocations[] == 2

        # redefinition of an unrelated function
        @eval mod unrelated(i) = 42
        Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 2

        # redefining child functions
        @eval mod @noinline child(i) = i+1
        Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 3

        # cached compilation
        Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test invocations[] == 3

        # change in configuration
        config = CompilerConfig(job.config; name="foobar")
        @test @filecheck begin
            check"CHECK: define i64 @foobar"
            Base.invokelatest(GPUCompiler.cached_compilation, cache, source, config, compiler, linker)
        end
        @test invocations[] == 4

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
        @eval mod kernel(i) = child(i)+4
        source = methodinstance(ft, tt, Base.get_world_counter())
        ir = Base.invokelatest(GPUCompiler.cached_compilation, cache, source, job.config, compiler, linker)
        @test contains(ir, r"add i64 %\d+, 4")
        notify(c2)      # wake up the task
        @test @filecheck begin
            check"CHECK-LABEL: define i64 @{{(julia|j)_kernel_[0-9]+}}"
            check"CHECK: add i64 %{{[0-9]+}}, 3"
            fetch(t)
        end
    end

    @testset "allowed mutable types" begin
        # when types have no fields, we should always allow them
        mod = @eval module $(gensym())
            struct Empty end
        end

        Native.code_execution(Returns(nothing), (mod.Empty,))

        # this also applies to Symbols
        Native.code_execution(Returns(nothing), (Symbol,))
    end
end

############################################################################################

@testset "IR" begin

@testset "basic reflection" begin
    mod = @eval module $(gensym())
        valid_kernel() = return
        invalid_kernel() = 1
    end

    @test @filecheck begin
        # module should contain our function + a generic call wrapper
        check"CHECK: @{{(julia|j)_valid_kernel_[0-9]+}}"
        Native.code_llvm(mod.valid_kernel, Tuple{}; optimize=false, dump_module=true)
    end

    @test Native.code_llvm(devnull, mod.invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError Native.code_llvm(devnull, mod.invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "unbound typevars" begin
    mod = @eval module $(gensym())
        invalid_kernel() where {unbound} = return
    end
    @test_throws KernelError Native.code_llvm(devnull, mod.invalid_kernel, Tuple{})
end

@testset "child functions" begin
    # we often test using `@noinline sink` child functions, so test whether these survive
    mod = @eval module $(gensym())
        import ..sink
        @noinline child(i) = sink(i)
        parent(i) = child(i)
    end

    @test @filecheck begin
        check"CHECK-LABEL: define i64 @{{(julia|j)_parent_[0-9]+}}"
        check"CHECK: call{{.*}} i64 @{{(julia|j)_child_[0-9]+}}"
        Native.code_llvm(mod.parent, Tuple{Int})
    end
end

@testset "sysimg" begin
    # bug: use a system image function
    mod = @eval module $(gensym())
        function foobar(a,i)
            Base.pointerset(a, 0, mod1(i,10), 8)
        end
    end

    @test @filecheck begin
        check"CHECK-NOT: jlsys_"
        Native.code_llvm(mod.foobar, Tuple{Ptr{Int},Int})
    end
end

@testset "tracked pointers" begin
    mod = @eval module $(gensym())
        function kernel(a)
            a[1] = 1
            return
        end
    end

    # this used to throw an LLVM assertion (#223)
    Native.code_llvm(devnull, mod.kernel, Tuple{Vector{Int}}; kernel=true)
    @test "We did not crash!" != ""
end

@testset "CUDA.jl#278" begin
    # codegen idempotency
    # NOTE: this isn't fixed, but surfaces here due to bad inference of checked_sub
    # NOTE: with the fix to print_to_string this doesn't error anymore,
    #       but still have a test to make sure it doesn't regress
    Native.code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)
    Native.code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)

    # breaking recursion in print_to_string makes it possible to compile
    # even in the presence of the above bug
    Native.code_llvm(devnull, Base.print_to_string, Tuple{Int,Int}; optimize=false)

    @test "We did not crash!" != ""
end

@testset "LLVM D32593" begin
    mod = @eval module $(gensym())
        struct D32593_struct
            foo::Float32
            bar::Float32
        end

        D32593(ptr) = unsafe_load(ptr).foo
    end

    Native.code_llvm(devnull, mod.D32593, Tuple{Ptr{mod.D32593_struct}})
    @test "We did not crash!" != ""
end

@testset "slow abi" begin
    mod = @eval module $(gensym())
        x = 2
        f = () -> x+1
    end
    @test @filecheck begin
        check"CHECK: define {{.+}} @julia"
        check"TYPED: define nonnull {}* @jfptr"
        check"OPAQUE: define nonnull ptr @jfptr"
        check"CHECK: call {{.+}} @julia"
        Native.code_llvm(mod.f, Tuple{}; entry_abi=:func, dump_module=true)
    end
end

@testset "function entry safepoint emission" begin
    @test @filecheck begin
        check"CHECK-LABEL: define void @{{(julia|j)_identity_[0-9]+}}"
        check"CHECK-NOT: %safepoint"
        Native.code_llvm(identity, Tuple{Nothing}; entry_safepoint=false, optimize=false, dump_module=true)
    end

    # XXX: broken by JuliaLang/julia#57010,
    #      see https://github.com/JuliaLang/julia/pull/57010/files#r2079576894
    if VERSION < v"1.13.0-DEV.533"
        @test @filecheck begin
            check"CHECK-LABEL: define void @{{(julia|j)_identity_[0-9]+}}"
            check"CHECK: %safepoint"
            Native.code_llvm(identity, Tuple{Nothing}; entry_safepoint=true, optimize=false, dump_module=true)
        end
    end
end

@testset "always_inline" begin
    # XXX: broken by JuliaLang/julia#51599, see JuliaGPU/GPUCompiler.jl#527.
    #      yet somehow this works on 1.12?
    broken = VERSION >= v"1.13-"

    mod = @eval module $(gensym())
        import ..sink
        expensive(x) = $(foldl((e, _) -> :($sink($e) + $sink(x)), 1:100; init=:x))
        function g(x)
            expensive(x)
            return
        end
        function h(x)
            expensive(x)
            return
        end
    end

    @test @filecheck begin
        check"CHECK: @{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true)
    end

    @test @filecheck(begin
        check"CHECK-NOT: @{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
    end) broken=broken

    @test @filecheck begin
        check"CHECK: @{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true)
    end

    @test @filecheck(begin
        check"CHECK-NOT: @{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
    end) broken=broken
end

@testset "function attributes" begin
    mod = @eval module $(gensym())
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
    end

    @test @filecheck begin
        check"CHECK: attributes #{{.}} = { convergent }"
        Native.code_llvm(mod.convergent_barrier, Tuple{}; dump_module=true, raw=true)
    end
end

end

############################################################################################

@testset "assembly" begin

@testset "basic reflection" begin
    mod = @eval module $(gensym())
        valid_kernel() = return
        invalid_kernel() = 1
    end

    @test Native.code_native(devnull, mod.valid_kernel, Tuple{}) == nothing
    @test Native.code_native(devnull, mod.invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError Native.code_native(devnull, mod.invalid_kernel, Tuple{}; kernel=true)
end

@testset "idempotency" begin
    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)
    mod = @eval module $(gensym())
        kernel() = return
    end
    Native.code_native(devnull, mod.kernel, Tuple{})
    Native.code_native(devnull, mod.kernel, Tuple{})

    @test "We did not crash!" != ""
end

@testset "compile for host after gpu" begin
    # issue #11: re-using host functions after GPU compilation
    mod = @eval module $(gensym())
        import ..sink
        @noinline child(i) = sink(i+1)

        function fromhost()
            child(10)
        end

        function fromptx()
            child(10)
            return
        end
    end

    Native.code_native(devnull, mod.fromptx, Tuple{})
    @test mod.fromhost() == 11
end

end

############################################################################################

@testset "errors" begin


@testset "non-isbits arguments" begin
    mod = @eval module $(gensym())
        import ..sink
        foobar(i) = (sink(unsafe_trunc(Int,i)); return)
    end

    @test_throws_message(KernelError,
                         Native.code_execution(mod.foobar, Tuple{BigInt})) do msg
        occursin("passing non-bitstype argument", msg) &&
        occursin("BigInt", msg)
    end

    # test that we get information about fields and reason why something is not isbits
    mod = @eval module $(gensym())
        struct CleverType{T}
            x::T
        end
        Base.unsafe_trunc(::Type{Int}, x::CleverType) = unsafe_trunc(Int, x.x)
        foobar(i) = (sink(unsafe_trunc(Int,i)); return)
    end
    @test_throws_message(KernelError,
                         Native.code_execution(mod.foobar, Tuple{mod.CleverType{BigInt}})) do msg
        occursin("passing non-bitstype argument", msg) &&
        occursin("CleverType", msg) &&
        occursin("BigInt", msg)
    end
end

@testset "invalid LLVM IR" begin
    mod = @eval module $(gensym())
        foobar(i) = println(i)
    end

    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.foobar, Tuple{Int})) do msg
        occursin("invalid LLVM IR", msg) &&
        (occursin(GPUCompiler.RUNTIME_FUNCTION, msg) ||
         occursin(GPUCompiler.UNKNOWN_FUNCTION, msg) ||
         occursin(GPUCompiler.DYNAMIC_CALL, msg)) &&
        occursin("[1] println", msg) &&
        occursin("[2] foobar", msg)
    end
end

@testset "invalid LLVM IR (ccall)" begin
    mod = @eval module $(gensym())
        function foobar(p)
            unsafe_store!(p, ccall(:time, Cint, ()))
            return
        end
    end

    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.foobar, Tuple{Ptr{Int}})) do msg
        if VERSION >= v"1.11-"
            occursin("invalid LLVM IR", msg) &&
            occursin(GPUCompiler.LAZY_FUNCTION, msg) &&
            occursin("call to time", msg) &&
            occursin("[1] foobar", msg)
        else
            occursin("invalid LLVM IR", msg) &&
            occursin(GPUCompiler.POINTER_FUNCTION, msg) &&
            occursin("[1] foobar", msg)
        end
    end
end

@testset "delayed bindings" begin
    mod = @eval module $(gensym())
        function kernel()
            undefined
            return
        end
    end

    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DELAYED_BINDING, msg) &&
        occursin(r"use of '.*undefined'", msg) &&
        occursin("[1] kernel", msg)
    end
end

@testset "dynamic call (invoke)" begin
    mod = @eval module $(gensym())
        @noinline nospecialize_child(@nospecialize(i)) = i
        kernel(a, b) = (unsafe_store!(b, nospecialize_child(a)); return)
    end

    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{Int,Ptr{Int}})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to nospecialize_child", msg) &&
        occursin("[1] kernel", msg)
    end
end

@testset "dynamic call (apply)" begin
    mod = @eval module $(gensym())
        func() = println(1)
    end

    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.func, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to print", msg) &&
        occursin("[2] func", msg)
    end
end

end

############################################################################################

@testset "overrides" begin
    # NOTE: method overrides do not support redefinitions, so we use different kernels

    mod = @eval module $(gensym())
        kernel() = child()
        @inline child() = 0
    end

    @test @filecheck begin
        check"CHECK-LABEL: @julia_kernel"
        check"CHECK: ret i64 0"
        Native.code_llvm(mod.kernel, Tuple{})
    end

    mod = @eval module $(gensym())
        using ..GPUCompiler

        Base.Experimental.@MethodTable(method_table)

        kernel() = child()
        @inline child() = 0

        Base.Experimental.@overlay method_table child() = 1
    end

    @test @filecheck begin
        check"CHECK-LABEL: @julia_kernel"
        check"CHECK: ret i64 1"
        Native.code_llvm(mod.kernel, Tuple{}; mod.method_table)
    end
end

@testset "semi-concrete interpretation + overlay methods" begin
    # issue 366, caused dynamic deispatch
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

    @test @filecheck begin
        check"CHECK-LABEL: @julia_kernel"
        check"CHECK-NOT: apply_generic"
        check"CHECK: llvm.floor"
        Native.code_llvm(mod.kernel, Tuple{Int, Int}; debuginfo=:none, mod.method_table)
    end
end

@testset "kwcall inference + overlay method" begin
    # originally broken by JuliaLang/julia#48097
    # broken again by JuliaLang/julia#51092, see JuliaGPU/GPUCompiler.jl#506

    mod = @eval module $(gensym())
        child(; kwargs...) = return
        function parent()
            child(; a=1f0, b=1.0)
            return
        end

        Base.Experimental.@MethodTable method_table
        Base.Experimental.@overlay method_table @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} = return
    end

    @test @filecheck begin
        check"CHECK-LABEL: @julia_parent"
        check"CHECK-NOT: jl_invoke"
        check"CHECK-NOT: apply_iterate"
        check"CHECK-NOT: inttoptr"
        check"CHECK-NOT: apply_type"
        check"CHECK: ret void"
        Native.code_llvm(mod.parent, Tuple{}; debuginfo=:none, mod.method_table)
    end
end

@testset "Mock Enzyme" begin
    function kernel(a)
        a[1] = a[1]^2
        return
    end

    function dkernel(a)
        ptr = Enzyme.deferred_codegen(typeof(kernel), Tuple{Vector{Float64}})
        ccall(ptr, Cvoid, (Vector{Float64},), a)
        return
    end

    ir = sprint(io->Native.code_llvm(io, dkernel, Tuple{Vector{Float64}}; debuginfo=:none))
    @test !occursin("deferred_codegen", ir)
    @test occursin("call void @julia_kernel", ir)
end
