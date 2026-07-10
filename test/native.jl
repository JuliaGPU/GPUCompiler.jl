@testset "reflection" begin
    mod = @eval module $(gensym())
        f(x::Int) = x
    end
    job, _ = Native.create_job(mod.f, (Int,))

    @test only(GPUCompiler.code_lowered(job)) isa Core.CodeInfo

    ci, rt = only(GPUCompiler.code_typed(job))
    @test rt === Int

    @test @filecheck begin
        @check "MethodInstance for {{.*}}f"
        GPUCompiler.code_warntype(job)
    end

    @test @filecheck begin
        @check "@{{(julia|j)_f_[0-9]+}}"
        GPUCompiler.code_llvm(job)
    end

    @test @filecheck begin
        @check "@{{(julia|j)_f_[0-9]+}}"
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
            function outer(x, sym)
                if sym == :a
                    return inner(x)
                end
                return x
            end
        end

        job, _ = Native.create_job(mod.outer, (Int, Symbol); validate=false)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(mod.outer, (Int, Symbol)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            other_mis = filter(mi->mi.def != meth, keys(meta.compiled))
            @test length(other_mis) == 1
            @test only(other_mis).def in methods(mod.inner)

            if VERSION >= v"1.12"
                @test length(meta.gv_to_value) == 1
                for (k, v) in meta.gv_to_value
                    @test v != C_NULL
                end
            end
            # TODO: Global values get privatized, so we can't find them by name anymore.
            # %.not = icmp eq ptr %"sym::Symbol", inttoptr (i64 140096668482288 to ptr), !dbg !38
            # for (name, v) in meta.gv_to_value
            #     gv = globals(ir)[name]
            #     @test LLVM.initializer(gv) === v
            # end
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
            @check_label "define i64 @{{(julia|j)_kernel_[0-9]+}}"
            @check "add i64 %{{[0-9]+}}, 1"
            GPUCompiler.code_llvm(job)
        end

        # basic redefinition
        @eval mod kernel(i) = child(i)+2
        job, _ = Native.create_job(mod.kernel, (Int64,))
        @test @filecheck begin
            @check_label "define i64 @{{(julia|j)_kernel_[0-9]+}}"
            @check "add i64 %{{[0-9]+}}, 2"
            GPUCompiler.code_llvm(job)
        end
    end

    @testset "cached results" begin
        mod = @eval module $(gensym())
            Base.Experimental.@MethodTable(other_method_table)

            mutable struct Results
                asm::Union{Nothing,String}
                Results() = new(nothing)
            end
            mutable struct OtherResults
                data::Any
                OtherResults() = new(nothing)
            end

            @noinline child(i) = i
            kernel(i) = child(i)+1
        end

        job, _ = Native.create_job(mod.kernel, (Int64,))

        @static if GPUCompiler.HAS_INTEGRATED_CACHE
            # before any code exists for the job, the lookup comes up empty
            @test GPUCompiler.cached_results(mod.Results, job) === nothing
        end

        # get-or-create: first access after inference yields an empty struct, later
        # accesses return the same one
        precompile(job)
        res = GPUCompiler.cached_results(mod.Results, job)
        @test res isa mod.Results
        @test res.asm === nothing
        res.asm = "compiled"
        @test GPUCompiler.cached_results(mod.Results, job) === res

        # independent consumers get independent structs for the same job
        other = GPUCompiler.cached_results(mod.OtherResults, job)
        @test other isa mod.OtherResults
        @test GPUCompiler.cached_results(mod.Results, job) === res

        # results are keyed by the full config: a job differing only in codegen-level
        # settings (here: the kernel name) must not share artifacts
        named_job, _ = Native.create_job(mod.kernel, (Int64,); name="custom")
        @test named_job.source === job.source
        named_res = GPUCompiler.cached_results(mod.Results, named_job)
        @test named_res !== res
        @test named_res.asm === nothing

        # ... but an equal config constructed from scratch resolves to the same struct
        job2, _ = Native.create_job(mod.kernel, (Int64,))
        @test GPUCompiler.cached_results(mod.Results, job2) === res

        # vararg kernels: on 1.14+, inference caches these under the compilable
        # (vararg-widened) MethodInstance rather than the fully-specialized job.source,
        # which the lookup needs to chase
        vmod = @eval module $(gensym())
            kernel(args...) = nothing
        end
        vjob, _ = Native.create_job(vmod.kernel, (Int64, Int64))
        precompile(vjob)
        @test GPUCompiler.cached_results(mod.Results, vjob) isa mod.Results

        @static if GPUCompiler.HAS_INTEGRATED_CACHE
            # The compiler may report CIs for several foreign owners when the same MI has
            # been inferred through multiple interpreters. Codegen must select this job's
            # owner rather than treating every non-native CI as interchangeable.
            other_owner_job, _ = Native.create_job(
                mod.kernel, (Int64,); method_table=mod.other_method_table)
            precompile(other_owner_job)
            other_owner_res = GPUCompiler.cached_results(mod.Results, other_owner_job)
            @test other_owner_res !== res
            JuliaContext() do ctx
                _, meta = GPUCompiler.compile(:llvm, job)
                @test meta.compiled[job.source].ci.owner === GPUCompiler.cache_owner(job)
            end
        end

        # redefinition invalidates: a job in the new world gets a fresh struct
        @eval mod kernel(i) = child(i)+2
        new_job, _ = Native.create_job(mod.kernel, (Int64,))
        @static if GPUCompiler.HAS_INTEGRATED_CACHE
            # ... after first showing up empty, as the old CodeInstance no longer covers
            # the new world
            @test GPUCompiler.cached_results(mod.Results, new_job) === nothing
        end
        precompile(new_job)
        new_res = GPUCompiler.cached_results(mod.Results, new_job)
        @test new_res !== res
        @test new_res.asm === nothing

        @static if GPUCompiler.HAS_INTEGRATED_CACHE
            # session-dependent results (e.g. artifacts with relocated GVs) are wiped
            # before image serialization; emulate the atexit-driven wipe directly
            new_res.asm = "session-dependent"
            other_job, _ = Native.create_job(mod.kernel, (Int64,); name="other")
            other_res = GPUCompiler.cached_results(mod.Results, other_job)
            push!(GPUCompiler.session_dependent_jobs, new_job)
            GPUCompiler.wipe_session_dependent_results()
            @test isempty(GPUCompiler.session_dependent_jobs)
            wiped_res = GPUCompiler.cached_results(mod.Results, new_job)
            @test wiped_res !== new_res
            @test wiped_res.asm === nothing
            # ... without affecting other configs on the same CI
            @test GPUCompiler.cached_results(mod.Results, other_job) === other_res
        end
    end

    @testset "runtime cache invalidation" begin
        # The assembled runtime cache must follow Julia's CodeInstance invalidation. Runtime
        # functions are ordinary Julia methods and can be redefined during a session.
        @eval Native.Runtime signal_exception() = nothing
        job, _ = Native.create_job(identity, (Nothing,))

        func_job, _ = Native.create_job(identity, (Nothing,); entry_abi=:func, opt_level=3)
        rt_config = GPUCompiler.runtime_config(func_job)
        @test rt_config.entry_abi === :specfunc
        @test rt_config.opt_level == 0

        JuliaContext() do ctx
            empty!(GPUCompiler.runtime_libs)
            GPUCompiler.load_runtime(job)

            key = (GPUCompiler.runtime_config(job), !GPUCompiler.supports_typed_pointers(ctx))
            old = GPUCompiler.runtime_libs[key]
            @test GPUCompiler.runtime_library_valid(old, job)

            @eval Native.Runtime signal_exception() = return
            new_job, _ = Native.create_job(identity, (Nothing,))
            new_job = CompilerJob(new_job.source, new_job.config, Base.get_world_counter())
            @test !GPUCompiler.runtime_library_valid(old, new_job)

            GPUCompiler.load_runtime(new_job)
            new = GPUCompiler.runtime_libs[key]
            @test new !== old
            @test GPUCompiler.runtime_library_valid(new, new_job)
        end
    end

    @testset "runtime constgv relocation" begin
        # runtime functions like `box_bool` may reference Julia singletons through
        # `julia.constgv` globals. Their session-absolute addresses must be baked into
        # the cached runtime bitcode when it is built: only kernel modules go through
        # `relocate_gvs!`, so a slot left null here would stay null on the device.
        job, _ = Native.create_job(identity, (Nothing,))
        JuliaContext() do ctx
            GPUCompiler.load_runtime(job)
            key = (GPUCompiler.runtime_config(job),
                   !GPUCompiler.supports_typed_pointers(ctx))
            lib = Base.@lock GPUCompiler.runtime_libs_lock GPUCompiler.runtime_libs[key]
            # NOTE: parse eagerly; a lazily-parsed module doesn't expose uses
            rt = parse(LLVM.Module, MemoryBuffer(lib.bytes))
            used = 0
            for gv in globals(rt)
                haskey(metadata(gv), "julia.constgv") || continue
                isempty(uses(gv)) && continue
                used += 1
                init = LLVM.initializer(gv)
                @test init !== nothing && !LLVM.isnull(init)
            end
            @static if VERSION >= v"1.12-"
                # on older versions, Julia bakes addresses without tagging globals
                @test used > 0
            end
        end
    end

    @testset "allowed mutable types" begin
        # when types have no fields, we should always allow them
        mod = @eval module $(gensym())
            struct Empty end
            accept_empty(::Empty) = nothing
            accept_symbol(::Symbol) = nothing
        end

        Native.code_execution(mod.accept_empty, (mod.Empty,))

        # this also applies to Symbols
        Native.code_execution(mod.accept_symbol, (Symbol,))
    end

    @testset "code coverage" begin
        mod = @eval module $(gensym())
            @inline inlined_callee(x) = x + one(x)
            @noinline noinline_callee(x) = x * 2
            entry(x) = noinline_callee(inlined_callee(x))

            # a genuinely multi-line function, so its definition (signature) line is
            # distinct from its body lines; compiled as its own entry below.
            function multiline(x)
                y = x + 1
                z = y * 2
                return z
            end
        end

        # whether any line in `lo:hi` of `file` has a nonzero execution count in an
        # lcov tracefile
        function lcov_any_covered(tracefile, file, lo, hi)
            in_block = false
            for l in eachline(tracefile)
                if startswith(l, "SF:")
                    in_block = (l == "SF:" * file)
                elseif l == "end_of_record"
                    in_block = false
                elseif in_block && startswith(l, "DA:")
                    ln, cnt = parse.(Int, split(l[4:end], ","))
                    lo <= ln <= hi && cnt >= 1 && return true
                end
            end
            return false
        end

        # the execution count recorded for an exact line of `file`, or `nothing` if that
        # line was not instrumented
        function lcov_line_count(tracefile, file, line)
            in_block = false
            for l in eachline(tracefile)
                if startswith(l, "SF:")
                    in_block = (l == "SF:" * file)
                elseif l == "end_of_record"
                    in_block = false
                elseif in_block && startswith(l, "DA:")
                    ln, cnt = parse.(Int, split(l[4:end], ","))
                    ln == line && return cnt
                end
            end
            return nothing
        end

        if Base.JLOptions().code_coverage == 0
            @test_skip "requires --code-coverage"
        else
            for entry in (mod.entry, mod.multiline)
                job, _ = Native.create_job(entry, (Int,))
                JuliaContext() do ctx
                    GPUCompiler.compile(:asm, job)
                end
            end

            # flush coverage in-process; device lines show covered despite never running.
            # bare mktempdir (cleaned at exit, after a GC) dodges the EBUSY `rm` race the
            # `do` form hits on Windows. jl_write_coverage_data needs a `.info` path.
            dir = mktempdir()
            tracefile = joinpath(dir, "coverage.info")
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), tracefile)
            for f in (mod.inlined_callee, mod.noinline_callee, mod.entry)
                m = only(methods(f))
                @test lcov_any_covered(tracefile, string(m.file), m.line, m.line + 1)
            end

            # the definition line must be covered too, not just the body (Julia covers
            # it separately at the prologue)
            m = only(methods(mod.multiline))
            @test lcov_line_count(tracefile, string(m.file), m.line) !== nothing
            @test something(lcov_line_count(tracefile, string(m.file), m.line), 0) >= 1
        end
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
        @check "@{{(julia|j)_valid_kernel_[0-9]+}}"
        Native.code_llvm(mod.valid_kernel, Tuple{}; optimize=false, dump_module=true)
    end

    @test Native.code_llvm(devnull, mod.invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError Native.code_llvm(devnull, mod.invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "unbound typevars" begin
    # suppress the warning Julia emits when defining a method with an unbound typevar
    mod = redirect_stderr(devnull) do
        @eval module $(gensym())
            invalid_kernel() where {unbound} = return
        end
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
        @check_label "define i64 @{{(julia|j)_parent_[0-9]+}}"
        @check "call{{.*}} i64 @{{(julia|j)_child_[0-9]+}}"
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
        @check_not "jlsys_"
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
        @check "define {{.+}} @julia"
        @check cond=typed_ptrs "define nonnull {}* @jfptr"
        @check cond=opaque_ptrs "define nonnull ptr @jfptr"
        @check "call {{.+}} @julia"
        Native.code_llvm(mod.f, Tuple{}; entry_abi=:func, dump_module=true)
    end
end

@testset "function entry safepoint emission" begin
    mod = @eval module $(gensym())
        f(::Nothing) = nothing
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_f_[0-9]+}}"
        @check_not "%safepoint"
        Native.code_llvm(mod.f, Tuple{Nothing}; entry_safepoint=false, optimize=false, dump_module=true)
    end

    # XXX: broken by JuliaLang/julia#57010,
    #      see https://github.com/JuliaLang/julia/pull/57010/files#r2079576894
    if VERSION < v"1.13.0-DEV.533"
        @test @filecheck begin
            @check_label "define void @{{(julia|j)_f_[0-9]+}}"
            @check "%safepoint"
            Native.code_llvm(mod.f, Tuple{Nothing}; entry_safepoint=true, optimize=false, dump_module=true)
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
        @check "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true)
    end

    # suppress FileCheck diagnostics when the failure is known and expected
    quiet(f) = broken ? redirect_stderr(f, devnull) : f()

    @test quiet() do
        @filecheck begin
            @check_not "@{{(julia|j)_expensive_[0-9]+}}"
            Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
        end
    end broken=broken

    @test @filecheck begin
        @check "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true)
    end

    @test quiet() do
        @filecheck begin
            @check_not "@{{(julia|j)_expensive_[0-9]+}}"
            Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
        end
    end broken=broken
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
        @check "attributes #{{.}} = { convergent }"
        Native.code_llvm(mod.convergent_barrier, Tuple{}; dump_module=true, raw=true)
    end
end

@testset "CPU reference resolution" begin
    # JIT-private symbols like `jl_get_pgcstack_resolved` (JuliaLang/julia#61527) cannot
    # be looked up using `jl_cglobal`, so we should only resolve bindings that are
    # actually loaded from, leaving called functions alone.
    job, _ = Native.create_job(identity, (Nothing,))
    JuliaContext() do ctx
        mod = parse(LLVM.Module, """
            declare void @jl_get_pgcstack_resolved()

            define void @entry() {
                call void @jl_get_pgcstack_resolved()
                ret void
            }""")
        GPUCompiler.prepare_execution!(job, mod)
        @test haskey(functions(mod), "jl_get_pgcstack_resolved")
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
        @check_label "@julia_kernel"
        @check "ret i64 0"
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
        @check_label "@julia_kernel"
        @check "ret i64 1"
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
        @check_label "@julia_kernel"
        @check_not "apply_generic"
        @check "llvm.floor"
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
        # @consistent_overlay (Julia 1.11+) is needed for the compiler to optimize through the overlay
        @static if VERSION >= v"1.11-"
            Base.Experimental.@consistent_overlay method_table @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} = return
        else
            Base.Experimental.@overlay method_table @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} = return
        end
    end

    @test @filecheck begin
        @check_label "@julia_parent"
        @check_not "jl_invoke"
        @check_not "apply_iterate"
        @check_not "inttoptr"
        @check_not "apply_type"
        @check "ret void"
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

@testset "stack allocation intrinsic" begin
    mod = @eval module $(gensym())
        import ..GPUCompiler

        function scratch(x)
            p = GPUCompiler.alloca(Float32, Val(8), Val(0))
            @inbounds unsafe_store!(p, x, 1)
            @inbounds unsafe_store!(p, x, 8)
            return @inbounds unsafe_load(p, 1) + unsafe_load(p, 8)
        end

        # zero-element scratch yields a (null) pointer without emitting an alloca
        empty_scratch() = GPUCompiler.alloca(Float32, Val(0), Val(0)) === reinterpret(Core.LLVMPtr{Float32,0}, C_NULL)
    end

    # the intrinsic is materialized as a single entry-block alloca whose element type is
    # sized to the alignment (32 bytes of Float32 scratch → `[8 x i32], align 4`), and no
    # `julia.gpu.alloca` call/declaration survives lowering.
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check "alloca [8 x i32], align 4"
        @check_not "julia.gpu.alloca"
        Native.code_llvm(mod.scratch, Tuple{Float32}; optimize=false, dump_module=true)
    end

    # once optimized the slot is promoted away entirely (result is x + x).
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check_not "alloca"
        @check_not "julia.gpu.alloca"
        Native.code_llvm(mod.scratch, Tuple{Float32})
    end

    # a zero-byte allocation lowers to a null pointer rather than a degenerate alloca.
    @test @filecheck begin
        @check_label "define {{.*}}@{{(julia|j)_empty_scratch_[0-9]+}}"
        @check_not "alloca"
        @check_not "julia.gpu.alloca"
        Native.code_llvm(mod.empty_scratch, Tuple{})
    end
end
