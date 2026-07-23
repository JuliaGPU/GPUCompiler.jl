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

@testset "method instances for type-valued callees and arguments" begin
    # JuliaLang/julia#62001: closed type-valued callees and arguments
    # dispatch on Core.TypeEgal keys instead of Type{T}
    mi = GPUCompiler.methodinstance(Base._stable_typeof(Vector{Int}), Tuple{typeof(undef), Int})
    @test mi isa Core.MethodInstance
    @test Base.isdispatchtuple(mi.specTypes)

    mi = GPUCompiler.methodinstance(typeof(identity), Tuple{Type{Int}})
    @test mi isa Core.MethodInstance
    @test Base.isdispatchtuple(mi.specTypes)
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

        # A JIT back-end keeps the Symbol reference symbolic in `:llvm`.
        job, _ = Native.create_job(mod.outer, (Int, Symbol); validate=false, jit=true)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(mod.outer, (Int, Symbol)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            other_mis = filter(mi->mi.def != meth, keys(meta.compiled))
            @test length(other_mis) == 1
            @test only(other_mis).def in methods(mod.inner)

            if GPUCompiler.supports_relocatable_ir()
                @test length(meta.relocations.sites) == 1
                @test only(values(meta.relocations.sites)) isa GPUCompiler.JuliaValueRef
            end
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

        # before any code exists for the job, the lookup comes up empty
        @test GPUCompiler.cached_results(mod.Results, job) === nothing

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

        # an unrelated world-age advance keeps the existing CodeInstance valid
        @eval mod unrelated() = nothing
        later_job, _ = Native.create_job(mod.kernel, (Int64,))
        @test later_job.world > job.world
        @test GPUCompiler.cached_results(mod.Results, later_job) === res

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
        # ... after first showing up empty, as the old CodeInstance no longer covers
        # the new world
        @test GPUCompiler.cached_results(mod.Results, new_job) === nothing
        precompile(new_job)
        new_res = GPUCompiler.cached_results(mod.Results, new_job)
        @test new_res !== res
        @test new_res.asm === nothing

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

    @testset "runtime relocations" begin
        # runtime functions like `box_bool` may reference Julia singletons through
        # `julia.constgv` globals. Keep their Julia identities with the cached bitcode
        # so the final kernel can resolve them in its own session.
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
                @test init === nothing
                site = GPUCompiler.RelocationSite(LLVM.name(gv), 0)
                @test haskey(lib.relocations.sites, site)
            end
            if GPUCompiler.supports_relocatable_ir()
                # otherwise Julia embeds addresses without tagging globals
                @test used > 0
            end
        end
    end

    @testset "boxed constant materialization" begin
        # since JuliaLang/julia#55045, isbits union constants stay fully boxed; the
        # box must be replicated on device instead of referencing a host address
        mod = @eval module $(gensym())
            union_smalltag(cond::Bool, a::Int32) = cond ? a : Int64(0)
            union_float(cond::Bool, a::Float32) = cond ? a : 1.0
            union_ghost(cond::Bool, a::Int32) = cond ? a : nothing
            @noinline produce_true(cond::Bool, a::Int32) = cond ? a : true
            @noinline produce_false(cond::Bool, a::Int32) = cond ? a : false
            function consume_true(cond::Bool, a::Int32)
                x = produce_true(cond, a)
                x isa Bool && x && return Int32(1)
                return Int32(0)
            end
            function consume_false(cond::Bool, a::Int32)
                x = produce_false(cond, a)
                x isa Bool && !x && return Int32(1)
                return Int32(0)
            end
            function kernel(p::Ptr{Int64}, cond::Bool, a::Int32)
                x = cond ? a : Int64(0)
                unsafe_store!(p, Int64(x))
                return
            end
            function egal_kernel(p::Ptr{Bool}, cond::Bool, a::Int32)
                x = cond ? a : Int64(0)
                unsafe_store!(p, x === Int64(0))
                return
            end
        end

        # smalltag constants materialize fully session-portably
        ir = sprint(io->Native.code_llvm(io, mod.union_smalltag, Tuple{Bool, Int32};
                                         dump_module=true, validate=true))
        @static if VERSION >= v"1.14.0-DEV.1348"
            @test occursin("_box", ir)
            @test !occursin("inttoptr", ir)
        end

        # non-smalltag constants carry a host type pointer in the box header,
        # but the payload is still device-resident
        ir = sprint(io->Native.code_llvm(io, mod.union_float, Tuple{Bool, Float32};
                                         dump_module=true, validate=true))
        @static if VERSION >= v"1.14.0-DEV.1348"
            @test occursin("_box", ir)
        end

        # Boxed Bool leaves use canonical device boxes.
        for (f, name) in ((mod.consume_true, "jl_true"),
                          (mod.consume_false, "jl_false"))
            ir = sprint(io->Native.code_llvm(io, f, Tuple{Bool, Int32};
                                             dump_module=true, validate=true))
            @static if VERSION >= v"1.14.0-DEV.1348"
                @test occursin("@$(name)_box = private unnamed_addr constant", ir)
                @test !occursin("@$name = external", ir)
                @test !occursin("inttoptr", ir)
            end
        end

        # zero-sized identity objects remain opaque host tokens
        ir = sprint(io->Native.code_llvm(io, mod.union_ghost, Tuple{Bool, Int32};
                                         dump_module=true, validate=true))
        @test !occursin("_box", ir)

        # kernel compilation, including bits-egal on the materialized leaf
        Native.code_execution(mod.kernel, (Ptr{Int64}, Bool, Int32))
        Native.code_execution(mod.egal_kernel, (Ptr{Bool}, Bool, Int32))

        # Classification records whether the module stayed session-portable; eager
        # lowering then resolves any remaining relocation slots.
        JuliaContext() do ctx
            # Unlike Int128, vector-shaped tuples are 16-byte aligned on all
            # supported architectures and Julia versions.
            aligned = (VecElement(Int64(1)), VecElement(Int64(2)))
            @test Base.datatype_alignment(typeof(aligned)) > sizeof(Int)
            objs = Any[Int64(42), 1.25, :sym, aligned]
            # pointers to the heap boxes rooted in `objs` (passing an element
            # through a specialized function would re-box, possibly on the stack)
            ptrs = [ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x) for x in objs]
            function slot_module(ptr::Ptr{Cvoid})
                llvm_mod = LLVM.Module("test")
                name = "jl_global#0"
                LLVM.GlobalVariable(llvm_mod, LLVM.PointerType(LLVM.Int8Type()), name)
                llvm_mod, Dict(name => ptr)
            end

            # Bool JuliaVariables are absent from `gv_to_value`.
            m = LLVM.Module("bool singletons")
            for name in ("jl_true", "jl_false")
                gv = LLVM.GlobalVariable(m, LLVM.PointerType(LLVM.Int8Type()), name)
                constant!(gv, true)
            end
            relocs = GPUCompiler.collect_julia_value_relocations!(
                m, Dict{String, Ptr{Cvoid}}())
            @test isempty(relocs.sites)
            GPUCompiler.bake_relocations!(m, relocs)
            bool_ir = string(m)
            for name in ("jl_true", "jl_false")
                @test haskey(globals(m), "$(name)_box")
                @test occursin("@$name = private constant", bool_ir)
            end
            @test !occursin("external", bool_ir)
            @test !occursin("inttoptr", bool_ir)
            dispose(m)

            GC.@preserve objs begin
                # smalltag isbits: materialized, portable
                m, map = slot_module(ptrs[1])
                relocs = GPUCompiler.collect_julia_value_relocations!(m, map)
                @test isempty(relocs.sites)
                GPUCompiler.bake_relocations!(m, relocs)
                @test haskey(globals(m), "jl_global_0_box")
                dispose(m)

                # Float64: the non-smalltag header is an interior relocation.
                m, map = slot_module(ptrs[2])
                relocs = GPUCompiler.collect_julia_value_relocations!(m, map)
                @test length(relocs.sites) == 1
                site, ref = only(relocs.sites)
                @test site.offset == 0
                @test ref.value === Float64
                box = globals(m)[site.name]
                @test isextinit(box)
                @test linkage(box) == LLVM.API.LLVMExternalLinkage
                header_idx = Int(element_at(datalayout(m), global_value_type(box),
                                            site.offset)) + 1
                @test convert(UInt, collect(operands(initializer(box)))[header_idx]) == 0
                GPUCompiler.bake_relocations!(m, relocs)
                @test isempty(relocs.sites)
                @test !isextinit(box)
                @test isconstant(box)
                @test linkage(box) == LLVM.API.LLVMPrivateLinkage
                @test convert(UInt, collect(operands(initializer(box)))[header_idx]) ==
                      GPUCompiler.resolve_relocation_target(ref)
                dispose(m)

                # Symbol: resolved address
                m, map = slot_module(ptrs[3])
                relocs = GPUCompiler.collect_julia_value_relocations!(m, map)
                @test only(values(relocs.sites)).value === objs[3]
                GPUCompiler.bake_relocations!(m, relocs)
                @test isempty(relocs.sites)
                @test !haskey(globals(m), "jl_global_0_box")
                @test occursin("inttoptr", string(m))
                dispose(m)

                # 16-byte-aligned payloads get padded past the header word
                m, map = slot_module(ptrs[4])
                relocs = GPUCompiler.collect_julia_value_relocations!(m, map)
                site, _ = only(relocs.sites)
                @test site.offset == 8
                GPUCompiler.bake_relocations!(m, relocs)
                box = globals(m)[site.name]
                @test length(elements(LLVM.global_value_type(box))) == 3
                dispose(m)
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

@testset "relocation target resolution" begin
    @test_throws ArgumentError GPUCompiler.RelocationSite("invalid", -1)

    sym = :relocation_target_probe
    @test GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(sym)) ==
          UInt(pointer_from_objref(sym))

    singleton = nothing
    @test GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(singleton)) ==
          UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), singleton))

    @test_throws ErrorException GPUCompiler.JuliaValueRef(1.5)
end

@testset "postponed relocation" begin
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            f() = UInt(pointer_from_objref(:relocation_probe))
        end
        job, _ = Native.create_job(mod.f, Tuple{}; jit=true)
        JuliaContext() do ctx
            obj, meta = GPUCompiler.compile(:obj, job)
            relocs = meta.relocations
            @test !isempty(relocs.sites)
            @test all(ref -> ref isa GPUCompiler.JuliaValueRef, values(relocs.sites))
            @test all(site -> isdeclaration(globals(meta.ir)[site.name]), keys(relocs.sites))

            bytes = Vector{UInt8}(codeunits(obj))
            entry = LLVM.name(meta.entry)
            probe_site = findfirst(ref -> ref isa GPUCompiler.JuliaValueRef &&
                                          ref.value === :relocation_probe, relocs.sites)
            @test probe_site !== nothing
            expected = GPUCompiler.resolve_relocation_target(relocs.sites[probe_site])
            fptr, keepalive = Native.load(bytes, entry, relocs, meta.ir)
            try
                GC.@preserve keepalive begin
                    actual = ccall(fptr, UInt, ())
                    @test actual == expected
                end
            finally
                dispose(first(keepalive))
            end

            @test_throws LLVMException Native.load(bytes, entry, GPUCompiler.Relocations(),
                                                   meta.ir)

            # Sites in definitions are applied after the object has been loaded.
            ptr(T) = GPUCompiler.supports_typed_pointers(ctx) ? "$T*" : "ptr"
            patch_mod = parse(LLVM.Module, """
                @patched_box = externally_initialized global { i64, i64 } { i64 0, i64 42 }

                define i64 @patched_header() {
                    %value = load i64, $(ptr("i64")) getelementptr ({ i64, i64 }, $(ptr("{ i64, i64 }")) @patched_box, i32 0, i32 0)
                    ret i64 %value
                }""")
            triple!(patch_mod, GPUCompiler.llvm_triple(job.config.target))
            datalayout!(patch_mod, GPUCompiler.llvm_datalayout(job.config.target))
            patch_relocs = GPUCompiler.Relocations()
            site = GPUCompiler.RelocationSite("patched_box", 0)
            patch_relocs.sites[site] = GPUCompiler.JuliaValueRef(Float64)
            patch_obj = GPUCompiler.mcgen(job, patch_mod, LLVM.API.LLVMObjectFile)
            patch_ptr, patch_keepalive = Native.load(
                Vector{UInt8}(codeunits(patch_obj)), "patched_header", patch_relocs,
                patch_mod)
            try
                GC.@preserve patch_keepalive begin
                    @test ccall(patch_ptr, UInt, ()) ==
                          GPUCompiler.resolve_relocation_target(
                              GPUCompiler.JuliaValueRef(Float64))
                end
            finally
                dispose(first(patch_keepalive))
            end
        end
    end
end

@testset "eager relocation resolution" begin
    # Eager resolution in `emit_llvm` leaves nothing for a loader.
    mod = @eval module $(gensym())
        probe() = UInt(pointer_from_objref(:eager_probe))
    end
    job, _ = Native.create_job(mod.probe, Tuple{})
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test isempty(meta.relocations.sites)
        # nothing is left for a loader to patch or import
        @test !any(GPUCompiler.isextinit, globals(ir))

        # This back-end can emit objects without threading relocation metadata.
        code, _ = GPUCompiler.emit_asm(job, ir, LLVM.API.LLVMObjectFile)
        @test !isempty(code)
    end
end

@testset "patchable relocation" begin
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            f() = UInt(pointer_from_objref(:patch_probe))
        end
        job, _ = Native.create_job(mod.f, Tuple{}; patch=true)
        JuliaContext() do ctx
            obj, meta = GPUCompiler.compile(:obj, job)
            relocs = meta.relocations
            @test !isempty(relocs.sites)

            # every declaration slot became a null-init, externally-initialized definition
            # kept alive by `llvm.used`; the loader patches each site after loading.
            @test haskey(globals(meta.ir), "llvm.used")
            for site in keys(relocs.sites)
                gv = globals(meta.ir)[site.name]
                @test !isdeclaration(gv)
                @test isextinit(gv)
                @test !isconstant(gv)
                @test linkage(gv) == LLVM.API.LLVMExternalLinkage
                @test LLVM.isnull(initializer(gv))
            end

            bytes = Vector{UInt8}(codeunits(obj))
            entry = LLVM.name(meta.entry)
            probe_site = findfirst(ref -> ref isa GPUCompiler.JuliaValueRef &&
                                          ref.value === :patch_probe, relocs.sites)
            @test probe_site !== nothing
            expected = GPUCompiler.resolve_relocation_target(relocs.sites[probe_site])
            fptr, keepalive = Native.load(bytes, entry, relocs, meta.ir)
            try
                GC.@preserve keepalive begin
                    @test ccall(fptr, UInt, ()) == expected
                end
            finally
                dispose(first(keepalive))
            end
        end
    end
end

@testset "deferred relocation" begin
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            f() = UInt(pointer_from_objref(:defer_probe))
        end
        job, _ = Native.create_job(mod.f, Tuple{}; defer=true)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)
            relocs = meta.relocations
            @test !isempty(relocs.sites)

            # a deferring consumer caches the bitcode and the relocation metadata...
            bitcode = let io = IOBuffer()
                write(io, ir)
                take!(io)
            end

            # ...and every session applies the metadata to a freshly parsed module
            for _ in 1:2
                session_mod = parse(LLVM.Module, MemoryBuffer(bitcode))
                roots = GPUCompiler.apply_relocations!(session_mod, relocs)
                @test :defer_probe in roots
                @test !isempty(relocs.sites)   # the metadata is not consumed
                for site in keys(relocs.sites)
                    @test !isdeclaration(globals(session_mod)[site.name])
                end
            end

            # emitting an object while relocations are live is a consumer error
            @test_throws "defers relocation lowering" GPUCompiler.compile(:obj, job)
        end
    end
end

@testset "relocation validation errors" begin
    JuliaContext() do ctx
        word() = GPUCompiler.relocation_word_type()
        nop = (_site, _gv, _ref) -> nothing
        ref = GPUCompiler.JuliaValueRef(:probe)
        reloc(name, offset=0) =
            GPUCompiler.Relocations(Dict(GPUCompiler.RelocationSite(name, offset) => ref))

        # a site whose global is absent from the module
        mod = LLVM.Module("errors")
        @test_throws "Missing relocation global" GPUCompiler.foreach_relocation(
            nop, mod, reloc("absent"))

        # a declaration slot may only carry a zero offset
        mod = LLVM.Module("errors")
        GlobalVariable(mod, word(), "slot")
        @test_throws "nonzero offset" GPUCompiler.foreach_relocation(
            nop, mod, reloc("slot", 8))

        # a declaration slot must be word-sized
        mod = LLVM.Module("errors")
        GlobalVariable(mod, LLVM.Int32Type(), "narrow")
        @test_throws "has size" GPUCompiler.foreach_relocation(nop, mod, reloc("narrow"))

        # an interior site must land within its global
        mod = LLVM.Module("errors")
        gv = GlobalVariable(mod, LLVM.StructType([LLVM.Int64Type(), LLVM.Int64Type()]), "box")
        initializer!(gv, ConstantStruct(LLVM.Constant[ConstantInt(0), ConstantInt(0)]))
        @test_throws "outside its" GPUCompiler.foreach_relocation(nop, mod, reloc("box", 16))
    end
end

@testset "prune dead relocations" begin
    JuliaContext() do ctx
        ptr(T) = GPUCompiler.supports_typed_pointers(ctx) ? "$T*" : "ptr"
        mod = parse(LLVM.Module, """
            @live = external global i64
            @dead = internal global { i64, i64 } { i64 0, i64 0 }
            define i64 @use() {
                %v = load i64, $(ptr("i64")) @live
                ret i64 %v
            }""")
        site(name, offset=0) = GPUCompiler.RelocationSite(name, offset)
        relocs = GPUCompiler.Relocations(Dict(
            site("live")   => GPUCompiler.JuliaValueRef(:live),
            site("dead")   => GPUCompiler.JuliaValueRef(:dead),   # unused definition
            site("absent") => GPUCompiler.JuliaValueRef(:absent), # global already gone
        ))
        GPUCompiler.prune_dead_relocations!(mod, relocs)
        @test collect(keys(relocs.sites)) == [site("live")]
        @test haskey(globals(mod), "live")     # a used declaration survives
        @test !haskey(globals(mod), "dead")    # the dead definition is erased
    end
end

@testset "resolve zeroinitializer box" begin
    # An all-zero box (a patchable header over a zero payload) is folded by LLVM to a
    # `zeroinitializer`, a ConstantAggregateZero that reports no operands; resolution must
    # resolve its header word. Regresses JuliaGPU/oneAPI.jl's "#55: invalid integers created
    # by alloc_opt", where `SVector(0f0, 0f0)` boxed a zero payload.
    JuliaContext() do ctx
        mod = parse(LLVM.Module,
                    "@zero_box = private global { i64, [8 x i8] } zeroinitializer")
        gv = globals(mod)["zero_box"]
        @test initializer(gv) isa LLVM.ConstantAggregateZero   # the folded shape
        relocs = GPUCompiler.Relocations(Dict(
            GPUCompiler.RelocationSite("zero_box", 0) => GPUCompiler.JuliaValueRef(Float64)))
        GPUCompiler.bake_relocations!(mod, relocs)
        init = initializer(gv)
        @test !(init isa LLVM.ConstantAggregateZero)   # rebuilt into explicit fields
        header = convert(UInt, LLVM.Constant[operands(init)...][1])
        @test header == GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(Float64))
        @test isconstant(gv)
        @test isempty(relocs.sites)
    end
end

@testset "cglobal relocation" begin
    # JIT-private symbols like `jl_get_pgcstack_resolved` (JuliaLang/julia#61527) cannot
    # be looked up using `jl_cglobal`, so we should only resolve bindings that are
    # actually loaded from, leaving called functions alone.
    job, _ = Native.create_job(identity, (Nothing,))
    JuliaContext() do ctx
        ptr(T) = GPUCompiler.supports_typed_pointers(ctx) ? "$T*" : "ptr"
        word_ptr = ptr("i8")
        word_ptr_ptr = ptr(word_ptr)
        function_word_ptr(name) = GPUCompiler.supports_typed_pointers(ctx) ?
            "i64* bitcast (i64 ()* @$name to i64*)" : "ptr @$name"

        mod = parse(LLVM.Module, """
            declare void @jl_get_pgcstack_resolved()

            define void @entry() {
                call void @jl_get_pgcstack_resolved()
                ret void
            }""")
        GPUCompiler.prepare_execution!(job, mod)
        @test haskey(functions(mod), "jl_get_pgcstack_resolved")

        mod = parse(LLVM.Module, """
            @jl_float32_type = external global $word_ptr

            define $word_ptr @entry() {
                %value = load $word_ptr, $word_ptr_ptr @jl_float32_type
                ret $word_ptr %value
            }""")
        GPUCompiler.prepare_execution!(job, mod)
        @test !occursin("load $word_ptr, $word_ptr_ptr @jl_float32_type", string(mod))
        @test occursin(r"private .*constant", string(mod))

        mod = parse(LLVM.Module, """
            @jl_float32_type = external global $word_ptr

            define $word_ptr @entry() {
                %value = load $word_ptr, $word_ptr_ptr @jl_float32_type
                ret $word_ptr %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(mod, relocs)
        site = only(keys(relocs.sites))
        @test relocs.sites[site] == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test site.offset == 0
        @test occursin("@$(site.name) = external global i64", string(mod))
        GPUCompiler.emit_patchable_relocations!(mod, relocs)
        @test occursin("externally_initialized global i64 0", string(mod))

        mod = parse(LLVM.Module, """
            declare i64 @jl_float32_type()

            define i64 @entry() {
                %value = load i64, $(function_word_ptr("jl_float32_type"))
                ret i64 %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(mod, relocs)
        site = only(keys(relocs.sites))
        @test relocs.sites[site] == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test occursin("@$(site.name) = external global i64", string(mod))
        GPUCompiler.emit_imported_relocations!(mod, relocs)
        @test relocs.sites[site] == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test isdeclaration(globals(mod)[site.name])
        @test occursin("@$(site.name) = external global i64", string(mod))

        mod = parse(LLVM.Module, """
            @jl_float32_type = external global $word_ptr

            define $word_ptr @entry() {
                %value = load $word_ptr, $word_ptr_ptr @jl_float32_type
                ret $word_ptr %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(mod, relocs)
        site = only(keys(relocs.sites))
        GPUCompiler.emit_imported_relocations!(mod, relocs)
        @test relocs.sites[site] == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test isdeclaration(globals(mod)[site.name])
        @test occursin("@$(site.name) = external global i64", string(mod))
    end
end

@testset "relocation linking" begin
    JuliaContext() do ctx
        ptr(T) = GPUCompiler.supports_typed_pointers(ctx) ? "$T*" : "ptr"

        function slot_module(name, entry)
            parse(LLVM.Module, """
                @$name = external global i64

                define i64 @$entry() {
                    %value = load i64, $(ptr("i64")) @$name
                    ret i64 %value
                }""")
        end

        site(name, offset=0) = GPUCompiler.RelocationSite(name, offset)
        slot_relocs(name, value) = GPUCompiler.Relocations(
            Dict(site(name) => GPUCompiler.JuliaValueRef(value)))

        # Equal Julia identities deliberately share a single slot.
        dest = slot_module("slot", "first")
        dest_relocs = slot_relocs("slot", :shared)
        src = slot_module("slot", "second")
        src_relocs = slot_relocs("slot", :shared)
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs)
        @test collect(keys(dest_relocs.sites)) == [site("slot")]
        @test occursin("@slot = external global i64", string(dest))

        # The name is the slot identity, so conflicting metadata is an error.
        dest = slot_module("slot", "first")
        dest_relocs = slot_relocs("slot", :first)
        src = slot_module("slot", "second")
        src_relocs = slot_relocs("slot", :second)
        @test_throws ErrorException GPUCompiler.link_relocatable!(
            dest, dest_relocs, src, src_relocs)

        # `only_needed` must keep metadata for imported slots and discard metadata for
        # source globals that the LLVM linker did not import.
        dest = parse(LLVM.Module, """
            declare i64 @source()

            define i64 @entry() {
                %value = call i64 @source()
                ret i64 %value
            }""")
        src = parse(LLVM.Module, """
            @used = external global i64
            @unused = external global i64

            define i64 @source() {
                %value = load i64, $(ptr("i64")) @used
                ret i64 %value
            }""")
        src_relocs = GPUCompiler.Relocations(Dict(
            site("used") => GPUCompiler.JuliaValueRef(:used),
            site("unused") => GPUCompiler.JuliaValueRef(:unused),
        ))
        dest_relocs = GPUCompiler.Relocations()
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs;
                                       only_needed=true)
        @test collect(keys(dest_relocs.sites)) == [site("used")]
        @test only(values(dest_relocs.sites)).value === :used

        function interior_module(name, entry)
            parse(LLVM.Module, """
                @$name = externally_initialized global { i64, i64 } { i64 0, i64 1 }

                define i64 @$entry() {
                    %value = load i64, $(ptr("i64")) getelementptr ({ i64, i64 }, $(ptr("{ i64, i64 }")) @$name, i32 0, i32 1)
                    ret i64 %value
                }""")
        end
        interior_relocs(name, value) = GPUCompiler.Relocations(
            Dict(site(name) => GPUCompiler.JuliaValueRef(value)))

        # Identical interior relocation identities merge; conflicting metadata does not.
        dest = interior_module("patch", "first_patch")
        dest_relocs = interior_relocs("patch", Float64)
        src = interior_module("patch", "second_patch")
        src_relocs = interior_relocs("patch", Float64)
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs)
        @test collect(keys(dest_relocs.sites)) == [site("patch")]

        dest = interior_module("patch", "first_patch")
        dest_relocs = interior_relocs("patch", Float64)
        src = interior_module("patch", "second_patch")
        src_relocs = interior_relocs("patch", Int64)
        @test_throws ErrorException GPUCompiler.link_relocatable!(
            dest, dest_relocs, src, src_relocs)

        # Metadata for interior globals not imported under `only_needed` is discarded.
        dest = parse(LLVM.Module, """
            declare i64 @source_patch()
            define i64 @entry_patch() {
                %value = call i64 @source_patch()
                ret i64 %value
            }""")
        src = interior_module("used_patch", "source_patch")
        unused = GlobalVariable(src, LLVM.StructType([LLVM.Int64Type(), LLVM.Int64Type()]),
                                "unused_patch")
        initializer!(unused, ConstantStruct(LLVM.Constant[ConstantInt(0), ConstantInt(1)]))
        src_relocs = GPUCompiler.Relocations(Dict(
            site("used_patch") => GPUCompiler.JuliaValueRef(Float64),
            site("unused_patch") => GPUCompiler.JuliaValueRef(Int64)))
        dest_relocs = GPUCompiler.Relocations()
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs;
                                       only_needed=true)
        @test collect(keys(dest_relocs.sites)) == [site("used_patch")]

        # A shared site whose source global is missing is an inconsistency, as is a source
        # definition with no matching destination global to link against.
        dest = slot_module("slot", "first")
        dest_relocs = slot_relocs("slot", :shared)
        src = LLVM.Module("empty")
        src_relocs = slot_relocs("slot", :shared)
        @test_throws "Missing source relocation global" GPUCompiler.link_relocatable!(
            dest, dest_relocs, src, src_relocs)

        dest = LLVM.Module("empty")
        dest_relocs = interior_relocs("patch", Float64)
        src = interior_module("patch", "second_patch")
        src_relocs = interior_relocs("patch", Float64)
        @test_throws "Missing destination relocation global" GPUCompiler.link_relocatable!(
            dest, dest_relocs, src, src_relocs)
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

@testset "static assertions" begin
    mod = @eval module $(gensym())
        using ..GPUCompiler
        kernel() = (@static_assert true "this should disappear"; return)
    end

    llvm = sprint(io -> Native.code_llvm(io, mod.kernel, Tuple{}; dump_module=true))
    @test !occursin(GPUCompiler.STATIC_ASSERT_MARKER, llvm)
    @test Native.code_execution(mod.kernel, Tuple{}) !== nothing

    mod = @eval module $(gensym())
        using ..GPUCompiler
        kernel() = (@static_assert false "the target is too old"; return)
    end
    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{})) do msg
        occursin(GPUCompiler.STATIC_ASSERTION, msg) &&
        occursin("the target is too old", msg) &&
        occursin("kernel", msg)
    end

    mod = @eval module $(gensym())
        using ..GPUCompiler
        function kernel(condition)
            @static_assert condition "condition was not proven"
            return
        end
    end
    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{Bool}; opt_level=0)) do msg
        occursin(GPUCompiler.STATIC_ASSERTION, msg) &&
        occursin("condition was not proven", msg)
    end

    mod = @eval module $(gensym())
        using ..GPUCompiler
        function kernel()
            if false
                @static_assert false "dead assertion"
            end
            return
        end
    end
    @test Native.code_execution(mod.kernel, Tuple{}; opt_level=0) !== nothing

    mod = @eval module $(gensym())
        using ..GPUCompiler
        function kernel(condition)
            @static_assert condition "first failure"
            @static_assert condition "second failure"
            return
        end
    end
    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{Bool})) do msg
        occursin("first failure", msg) && occursin("second failure", msg) &&
        !occursin("unknown function", msg)
    end

    mod = @eval module $(gensym())
        using ..GPUCompiler
        @inline assertion() = @static_assert false "inlined failure"
        kernel() = (assertion(); return)
    end
    @test_throws_message(InvalidIRError,
                         Native.code_execution(mod.kernel, Tuple{})) do msg
        occursin("inlined failure", msg) &&
        occursin("assertion", msg) && occursin("kernel", msg)
    end

    @test_throws ArgumentError macroexpand(mod, :(@static_assert true string("message")))
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

@testset "Mock Enzyme deferred relocations" begin
    # A deferred child that references a Julia value produces its own relocations; those
    # must merge into the parent's metadata when the child module is linked in.
    mod = @eval module $(gensym())
        import ..Enzyme
        child(sym::Symbol) = sym === :deferred_reloc ? 1 : 2
        function parent(sym::Symbol)
            ptr = Enzyme.deferred_codegen(typeof(child), Tuple{Symbol})
            return ccall(ptr, Int, (Symbol,), sym)
        end
    end

    # Keep the merged relocation symbolic so we can inspect it.
    job, _ = Native.create_job(mod.parent, (Symbol,); jit=true, validate=false)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test !occursin("deferred_codegen", string(ir))
        if GPUCompiler.supports_relocatable_ir()
            @test any(values(meta.relocations.sites)) do ref
                ref isa GPUCompiler.JuliaValueRef && ref.value === :deferred_reloc
            end
        end
    end
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
