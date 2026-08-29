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

@testset "compile hook" begin
    mod = @eval module $(gensym())
        f(x::Int) = nothing
        g(x::Int) = nothing
    end

    # the hook sees every job compiled within its scope, and nothing outside it
    seen = []
    with(GPUCompiler.compile_hook => job -> push!(seen, job)) do
        Native.code_execution(mod.f, (Int,))
    end
    @test length(seen) == 1 && only(seen).source.def.name === :f
    @test GPUCompiler.compile_hook[] === nothing
    Native.code_execution(mod.g, (Int,))
    @test length(seen) == 1

    # the reflection macros go through the hook once per job, even when the user
    # code compiles the same job repeatedly
    lowered = GPUCompiler.@device_code_lowered begin
        Native.code_execution(mod.f, (Int,))
        Native.code_execution(mod.f, (Int,))
    end
    @test length(lowered) == 1
    @test_throws "no kernels executed" GPUCompiler.@device_code_native mod.f(1)

    # concurrent compilations within the scope are all observed, exactly once each
    mod2 = @eval module $(gensym())
        $((:($(Symbol(:f, i))(x::Int) = nothing) for i in 1:8)...)
    end
    fs = [getfield(mod2, Symbol(:f, i)) for i in 1:8]
    typed = GPUCompiler.@device_code_typed begin
        @sync for f in fs, _ in 1:2
            Threads.@spawn Native.code_execution(f, (Int,))
        end
    end
    @test length(typed) == 8
    @test Set(job.source.def.name for job in keys(typed)) == Set(Symbol(:f, i) for i in 1:8)

    # simultaneous scopes in unrelated tasks do not observe each other's jobs
    ready = Channel(2)
    proceed = Channel(2)
    tasks = map((mod.f, mod.g)) do f
        Threads.@spawn GPUCompiler.@device_code_typed begin
            put!(ready, nothing)
            take!(proceed)
            Native.code_execution(f, (Int,))
        end
    end
    take!(ready)
    take!(ready)
    put!(proceed, nothing)
    put!(proceed, nothing)
    outputs = fetch.(tasks)
    @test all(length(output) == 1 for output in outputs)
    @test Set(only(keys(output)).source.def.name for output in outputs) == Set((:f, :g))

    # scoped: tasks spawned inside the scope inherit the hook, others don't
    hook = job -> nothing
    inherited = Ref{Any}(nothing)
    with(GPUCompiler.compile_hook => hook) do
        wait(Threads.@spawn inherited[] = GPUCompiler.compile_hook[])
        @test GPUCompiler.compile_hook[] === hook
    end
    @test inherited[] === hook
    @test fetch(Threads.@spawn GPUCompiler.compile_hook[]) === nothing
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

        # A relocatable back-end keeps the Symbol reference symbolic in `:llvm`.
        job, _ = Native.create_job(mod.outer, (Int, Symbol); validate=false,
                                   relocations=:patch)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            meth = only(methods(mod.outer, (Int, Symbol)))

            mis = filter(mi->mi.def == meth, keys(meta.compiled))
            @test length(mis) == 1

            other_mis = filter(mi->mi.def != meth, keys(meta.compiled))
            @test length(other_mis) == 1
            @test only(other_mis).def in methods(mod.inner)

            if GPUCompiler.supports_relocatable_ir()
                @test length(meta.relocations) == 1
                @test only(meta.relocations.records).target isa GPUCompiler.JuliaValueRef
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

            # An unrelated world change must not invalidate an otherwise-current library.
            Core.eval(Native.Runtime, :(runtime_cache_world_bump() = nothing))
            same_job = CompilerJob(job.source, job.config, Base.get_world_counter())
            @test GPUCompiler.runtime_library_valid(old, same_job)

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
                rec = GPUCompiler.find_relocation(lib.relocations, LLVM.name(gv))
                @test rec !== nothing && rec.kind === GPUCompiler.SlotSite
            end
            if GPUCompiler.supports_relocatable_ir()
                # otherwise Julia embeds addresses without tagging globals
                @test used > 0
            end
        end
    end

    @testset "boxed constant materialization" begin
        # Since JuliaLang/julia#55045, isbits union constants stay boxed. Device jobs must
        # embed copies instead of referring to GC-managed host boxes.
        device = (; jlruntime=false)
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
                                         dump_module=true, validate=true, device...))
        @static if VERSION >= v"1.14.0-DEV.1348"
            @test occursin("_box", ir)
            @test !occursin("inttoptr", ir)
        end

        # non-smalltag constants carry a host type pointer in the box header,
        # but the payload is still device-resident
        ir = sprint(io->Native.code_llvm(io, mod.union_float, Tuple{Bool, Float32};
                                         dump_module=true, validate=true, device...))
        @static if VERSION >= v"1.14.0-DEV.1348"
            @test occursin("_box", ir)
        end

        # Boxed Bool leaves use canonical device boxes.
        for (f, name) in ((mod.consume_true, "jl_true"),
                          (mod.consume_false, "jl_false"))
            ir = sprint(io->Native.code_llvm(io, f, Tuple{Bool, Int32};
                                             dump_module=true, validate=true, device...))
            @static if VERSION >= v"1.14.0-DEV.1348"
                @test occursin("@$(name)_box = private unnamed_addr constant", ir)
                @test !occursin("@$name = external", ir)
                @test !occursin("inttoptr", ir)
            end
        end

        # zero-sized identity objects remain opaque host tokens
        ir = sprint(io->Native.code_llvm(io, mod.union_ghost, Tuple{Bool, Int32};
                                         dump_module=true, validate=true, device...))
        @test !occursin("_box", ir)

        # kernel compilation, including bits-egal on the materialized leaf
        Native.code_execution(mod.kernel, (Ptr{Int64}, Bool, Int32); device...)
        Native.code_execution(mod.egal_kernel, (Ptr{Bool}, Bool, Int32); device...)

        # Classification records whether the module stayed session-portable; eager
        # lowering then resolves any remaining relocation slots.
        collect_job, _ = Native.create_job(mod.kernel, (Ptr{Int64}, Bool, Int32); device...)
        namespace = GPUCompiler.relocation_namespace(collect_job)
        collect!(m, map) = GPUCompiler.collect_julia_value_relocations!(collect_job, m, map)
        JuliaContext() do ctx
            # Unlike Int128, vector-shaped tuples are 16-byte aligned on all
            # supported architectures and Julia versions.
            aligned = (VecElement(Int64(1)), VecElement(Int64(2)))
            @test Base.datatype_alignment(typeof(aligned)) > sizeof(Int)
            objs = Any[Int64(42), 1.25, :sym, aligned, Union{}]
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
            relocs = collect!(m, Dict{String, Ptr{Cvoid}}())
            @test isempty(relocs)
            GPUCompiler.bake_relocations!(m, relocs)
            bool_ir = string(m)
            for name in ("jl_true", "jl_false")
                # a fully-materialized box is private, so it needs no per-job namespace
                @test haskey(globals(m), "$(name)_box")
                @test occursin("@$name = private constant", bool_ir)
            end
            @test !occursin("external", bool_ir)
            @test !occursin("inttoptr", bool_ir)
            dispose(m)

            GC.@preserve objs begin
                # smalltag isbits: materialized, portable
                m, map = slot_module(ptrs[1])
                relocs = collect!(m, map)
                @test isempty(relocs)
                GPUCompiler.bake_relocations!(m, relocs)
                @test haskey(globals(m), "jl_global_0_box")
                dispose(m)

                # Float64: the non-smalltag header is an interior relocation.
                m, map = slot_module(ptrs[2])
                relocs = collect!(m, map)
                @test length(relocs) == 1
                rec = only(relocs.records)
                @test rec.kind === GPUCompiler.InteriorSite
                @test rec.offset == 0
                @test rec.target.value === Float64
                # a relocatable box is addressed by name, so its name carries the namespace
                @test startswith(rec.name, namespace)
                box = globals(m)[rec.name]
                @test isextinit(box)
                @test linkage(box) == LLVM.API.LLVMExternalLinkage
                header_idx = Int(element_at(datalayout(m), global_value_type(box),
                                            rec.offset)) + 1
                @test convert(UInt, collect(operands(initializer(box)))[header_idx]) == 0
                GPUCompiler.bake_relocations!(m, relocs)
                @test isempty(relocs)
                @test !isextinit(box)
                @test isconstant(box)
                @test linkage(box) == LLVM.API.LLVMPrivateLinkage
                @test convert(UInt, collect(operands(initializer(box)))[header_idx]) ==
                      GPUCompiler.resolve_relocation_target(rec.target)
                dispose(m)

                # Symbol: resolved address
                m, map = slot_module(ptrs[3])
                relocs = collect!(m, map)
                rec = only(relocs.records)
                @test rec.kind === GPUCompiler.SlotSite
                @test rec.target.value === objs[3]
                @test startswith(rec.name, namespace)
                GPUCompiler.bake_relocations!(m, relocs)
                @test isempty(relocs)
                @test !haskey(globals(m), "jl_global_0_box")
                @test occursin("inttoptr", string(m))
                dispose(m)

                # Empty type objects have a zero-sized singleton representation.
                m, map = slot_module(ptrs[5])
                relocs = collect!(m, map)
                rec = only(relocs.records)
                @test rec.kind === GPUCompiler.SlotSite
                @test rec.target.value === Union{}
                dispose(m)

                # 16-byte-aligned payloads get padded past the header word
                m, map = slot_module(ptrs[4])
                relocs = collect!(m, map)
                rec = only(relocs.records)
                @test rec.offset == 8
                GPUCompiler.bake_relocations!(m, relocs)
                box = globals(m)[rec.name]
                @test length(elements(LLVM.global_value_type(box))) == 3
                dispose(m)

                # Codegen can emit several slots for one value in a module (observed on
                # 1.11, whose backported GV API does not deduplicate); their
                # content-derived names collide, so later slots must alias the first.
                m = LLVM.Module("duplicate slots")
                gvs = [LLVM.GlobalVariable(m, LLVM.PointerType(LLVM.Int8Type()),
                                           "jl_global#$i") for i in 1:2]
                relocs = collect!(m, Dict("jl_global#1" => ptrs[3],
                                          "jl_global#2" => ptrs[3]))
                rec = only(relocs.records)
                @test rec.target.value === objs[3]
                @test count(gv -> startswith(LLVM.name(gv), namespace), globals(m)) == 1
                @test !any(gv -> startswith(LLVM.name(gv), "jl_global#"), globals(m))
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

        @noinline never(x) = sink(x)
        function i(x)
            expensive(x)
            never(x)
            return
        end
    end

    @test @filecheck begin
        @check "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true)
    end

    @test @filecheck begin
        @check_not "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.g, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
    end

    @test @filecheck begin
        @check "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true)
    end

    @test @filecheck begin
        @check_not "@{{(julia|j)_expensive_[0-9]+}}"
        Native.code_llvm(mod.h, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
    end

    @test @filecheck begin
        @check_not "@{{(julia|j)_expensive_[0-9]+}}"
        @check "@{{(julia|j)_never_[0-9]+}}"
        Native.code_llvm(mod.i, Tuple{Int64}; dump_module=true, kernel=true, always_inline=true)
    end
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
    ref = GPUCompiler.JuliaValueRef(:probe)
    @test_throws ArgumentError GPUCompiler.Relocation(GPUCompiler.SlotSite, "invalid", -1, ref)
    # a slot is a whole word, so only an interior record may carry an offset
    @test_throws ArgumentError GPUCompiler.Relocation(GPUCompiler.SlotSite, "slot", 8, ref)

    sym = :relocation_target_probe
    @test GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(sym)) ==
          UInt(pointer_from_objref(sym))

    singleton = nothing
    @test GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(singleton)) ==
          UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), singleton))

    # Isbits values resolve by egal identity to one globally rooted box.
    a, b = parse(Float64, "1.5"), parse(Float64, "1.5")
    word = GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(a))
    @test word == GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(b))
    @test unsafe_pointer_to_objref(Ptr{Cvoid}(word)) === 1.5
end

@testset "runtime-backed boxed constants" begin
    # Values returned through the boxed `:func` ABI must be valid GC-managed objects.
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            returns_bool(x::Float64) = x > 0
            # A `Union` keeps the `isbits` alternative boxed (JuliaLang/julia#55045).
            returns_float(cond::Bool) = cond ? 2.5 : nothing
        end
        valptr(@nospecialize x) = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x)

        for (f, args, expected) in ((mod.returns_bool, (1.0,), true),
                                    (mod.returns_bool, (-1.0,), false),
                                    (mod.returns_float, (true,), 2.5))
            types = map(typeof, args)
            # The other `Union` arm leaves an allocation in the module; only relocation is
            # under test here.
            ir = sprint(io->Native.code_llvm(io, f, Tuple{types...}; entry_abi=:func,
                                             dump_module=true, validate=false))
            @test !occursin(r"^@\S+_box = "m, ir)   # no materialized replica
            if expected isa Bool
                @test occursin("@jl_$expected = external", ir)
            end

            job, _ = Native.create_job(f, types; entry_abi=:func, relocations=:patch,
                                       validate=false)
            JuliaContext() do ctx
                obj, meta = GPUCompiler.compile(:obj, job)
                relocs = meta.relocations
                # Runtime objects use whole-word slots, never materialized box headers.
                @test all(rec -> rec.kind === GPUCompiler.SlotSite, relocs.records)
                if expected isa Bool
                    # `jl_true` and `jl_false` are libjulia pointer globals.
                    ref = GPUCompiler.CGlobalRef(Symbol("jl_$expected"))
                    @test any(rec -> GPUCompiler.same_relocation_target(rec.target, ref),
                              relocs.records)
                else
                    @test any(rec -> rec.target isa GPUCompiler.JuliaValueRef &&
                                     rec.target.value === expected, relocs.records)
                end

                fptr, lljit, _table = Native.load(Vector{UInt8}(codeunits(obj)),
                                                  LLVM.name(meta.entry), relocs)
                try
                    boxed_args = Any[args...]
                    r = GC.@preserve boxed_args ccall(fptr, Any, (Any, Ptr{Any}, Int32),
                                                      f, pointer(boxed_args), length(args))
                    @test r === expected
                    expected_ptr = if expected isa Bool
                        valptr(expected)
                    else
                        Ptr{Cvoid}(GPUCompiler.resolve_relocation_target(
                            GPUCompiler.JuliaValueRef(expected)))
                    end
                    @test valptr(r) == expected_ptr
                    keep = Any[r]
                    GC.gc(true)
                    @test keep == Any[expected]
                finally
                    dispose(lljit)
                end
            end
        end
    end
end

@testset "applied relocation execution" begin
    # A consumer that resolves the metadata into a module of its own instead of letting a
    # loader do it: it caches the `:llvm` result plus the relocation metadata, and in every
    # later session re-parses, `apply_relocations!`s, and emits an object with nothing left
    # symbolic. AllocCheck does exactly this for the module it analyzes.
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            # the boxed `1.0` alternative of the isbits union is an *interior* record (its
            # header word is a `Float64` type tag), while the Symbol is a whole-word slot;
            # `apply_relocations!` must handle both kinds
            @noinline produce(cond::Bool, a::Int32) = cond ? a : 1.0
            function f(cond::Bool)
                x = produce(cond, Int32(7))
                word = x isa Float64 ? reinterpret(UInt64, x) : UInt64(0)
                return word + UInt(pointer_from_objref(:applied_probe))
            end
        end
        job, _ = Native.create_job(mod.f, (Bool,); relocations=:patch, jlruntime=false)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)
            relocs = meta.relocations
            @test all(rec -> rec.target isa GPUCompiler.JuliaValueRef, relocs.records)
            # both record kinds are exercised
            @test any(rec -> rec.kind === GPUCompiler.SlotSite, relocs.records)
            @test any(rec -> rec.kind === GPUCompiler.InteriorSite, relocs.records)

            # the cache artifact: session-portable bitcode + relocation metadata
            bitcode = let io = IOBuffer()
                write(io, ir)
                take!(io)
            end
            entry = LLVM.name(meta.entry)

            # a fresh session resolves the records into its own copy of the module, and only
            # then emits an object
            session_mod = parse(LLVM.Module, MemoryBuffer(bitcode))
            GPUCompiler.apply_relocations!(session_mod, relocs)
            @test !isempty(relocs)   # the metadata is not consumed
            obj, _ = GPUCompiler.emit_asm(job, session_mod, LLVM.API.LLVMObjectFile)

            expected = reinterpret(UInt64, 1.0) +
                       GPUCompiler.resolve_relocation_target(
                           GPUCompiler.JuliaValueRef(:applied_probe))
            fptr, lljit, _table = Native.load(Vector{UInt8}(codeunits(obj)), entry,
                                              GPUCompiler.Relocations())
            try
                # the boxed alternative: its header tag decides the `isa`, so a stranded
                # interior record would show up as a wrong result rather than a crash
                @test ccall(fptr, UInt, (Bool,), false) == expected
                @test ccall(fptr, UInt, (Bool,), false) == mod.f(false)
                # ...and the inline alternative, which only reads the Symbol slot
                @test ccall(fptr, UInt, (Bool,), true) == mod.f(true)
            finally
                dispose(lljit)
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
        @test isempty(meta.relocations)
        # nothing is left for a loader to patch or import
        @test !any(GPUCompiler.isextinit, globals(ir))

        # This back-end can emit objects without threading relocation metadata.
        code, _ = GPUCompiler.emit_asm(job, ir, LLVM.API.LLVMObjectFile)
        @test !isempty(code)
    end
end

@testset "patchable relocation" begin
    # An object-caching consumer: the words are written into the loaded image, so a cached
    # object needs no compiler at all in a later session. CUDA does this with
    # `cuModuleGetGlobal` + `cuMemcpyHtoD`; the ORC JIT below proves the same works under
    # JITLink on macOS aarch64, the strictest W^X environment (only code pages are hardened).
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            f() = UInt(pointer_from_objref(:patch_probe))
        end
        job, _ = Native.create_job(mod.f, Tuple{}; relocations=:patch)
        JuliaContext() do ctx
            obj, meta = GPUCompiler.compile(:obj, job)
            relocs = meta.relocations
            @test !isempty(relocs)

            # every slot became a null-init, externally-initialized definition kept alive by
            # `llvm.used`; the loader patches each record after loading. Definitions are weak
            # so that two objects defining one record coalesce (see "shared patchable record").
            @test haskey(globals(meta.ir), "llvm.used")
            for rec in relocs.records
                gv = globals(meta.ir)[rec.name]
                @test !isdeclaration(gv)
                @test isextinit(gv)
                @test !isconstant(gv)
                @test linkage(gv) == LLVM.API.LLVMWeakODRLinkage
                rec.kind === GPUCompiler.SlotSite && @test LLVM.isnull(initializer(gv))
            end

            bytes = Vector{UInt8}(codeunits(obj))
            entry = LLVM.name(meta.entry)
            probe = only(filter(rec -> rec.target isa GPUCompiler.JuliaValueRef &&
                                       rec.target.value === :patch_probe, relocs.records))
            expected = GPUCompiler.resolve_relocation_target(probe.target)
            fptr, lljit, _table = Native.load(bytes, entry, relocs)
            try
                @test ccall(fptr, UInt, ()) == expected
            finally
                dispose(lljit)
            end

            # The manifest now describes an emitted object, so dropping a record would leave
            # its definition holding a zero and mis-branch silently. Refuse instead.
            @test_throws "already been lowered" GPUCompiler.prune_dead_relocations!(
                meta.ir, relocs)
            @test_throws "already been lowered" GPUCompiler.add_relocation!(
                relocs, GPUCompiler.SlotSite, "late", 0, probe.target)

            # A consumer that also wants a session-resolved copy to analyze (AllocCheck reads
            # type tags out of one) applies the manifest *after* `emit_asm` froze it. That
            # works because resolution goes into a copy — which freezing could easily break.
            GPUCompiler.apply_relocations!(meta.ir, relocs)
            @test !isempty(relocs)
        end
    end
end

@testset "shared patchable record" begin
    # One record can be defined by two objects at once: a relocation-carrying runtime-library
    # function keeps its own job's namespace in every kernel it is linked into. An
    # AllocCheck-shaped loader puts every object in one JITDylib, where two definitions of a
    # symbol is an error unless they are weak. Construct the collision directly so this does
    # not depend on which runtime functions the test kernel happens to import.
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            f() = 0
        end
        job, _ = Native.create_job(mod.f, Tuple{}; relocations=:patch)
        JuliaContext() do ctx
            ptr(T) = GPUCompiler.supports_typed_pointers(ctx) ? "$T*" : "ptr"
            ref = GPUCompiler.JuliaValueRef(:shared_probe)
            function shared_object(entry)
                m = parse(LLVM.Module, """
                    @shared_reloc = external global i64

                    define i64 @$entry() {
                        %value = load i64, $(ptr("i64")) @shared_reloc
                        ret i64 %value
                    }""")
                relocs = GPUCompiler.Relocations(
                    [GPUCompiler.Relocation(GPUCompiler.SlotSite, "shared_reloc", 0, ref)])
                asm, _ = GPUCompiler.emit_asm(job, m, relocs, LLVM.API.LLVMObjectFile)
                @test linkage(globals(m)["shared_reloc"]) == LLVM.API.LLVMWeakODRLinkage
                return Vector{UInt8}(codeunits(asm)), relocs
            end

            obj_a, relocs_a = shared_object("shared_entry_a")
            obj_b, _ = shared_object("shared_entry_b")
            expected = GPUCompiler.resolve_relocation_target(ref)

            lljit = LLJIT(; tm=JITTargetMachine())
            try
                jd = JITDylib(lljit)
                add!(lljit, jd, MemoryBuffer(obj_a))
                add!(lljit, jd, MemoryBuffer(obj_b))   # the duplicate definition

                # patching the one surviving definition serves both objects
                for (rec, word) in GPUCompiler.resolved_relocations(relocs_a)
                    addr = lookup(lljit, rec.name)
                    unsafe_store!(Ptr{UInt}(pointer(addr) + rec.offset), word)
                end
                for entry in ("shared_entry_a", "shared_entry_b")
                    @test ccall(pointer(lookup(lljit, entry)), UInt, ()) == expected
                end
            finally
                dispose(lljit)
            end
        end
    end
end

@testset "tabulated relocation" begin
    # A consumer with no access at all to loaded code: every record is rewritten into an
    # indexed load from a table of words the loader delivers as run-time data. Metal does
    # this through the kernel state; the test back-end reaches the table through a single
    # patchable global, so the strategy is covered off-device.
    if GPUCompiler.supports_relocatable_ir() && LLVM.version() >= v"17"
        mod = @eval module $(gensym())
            # both record kinds: an interior box header (the `Float64` tag) and a slot
            @noinline produce(cond::Bool, a::Int32) = cond ? a : 2.0
            function f(cond::Bool)
                x = produce(cond, Int32(7))
                word = x isa Float64 ? reinterpret(UInt64, x) : UInt64(0)
                return word + UInt(pointer_from_objref(:table_probe))
            end
        end
        job, _ = Native.create_job(mod.f, (Bool,); relocations=:table, jlruntime=false)
        JuliaContext() do ctx
            obj, meta = GPUCompiler.compile(:obj, job)
            relocs = meta.relocations
            @test !isempty(relocs)
            @test any(rec -> rec.kind === GPUCompiler.SlotSite, relocs.records)
            @test any(rec -> rec.kind === GPUCompiler.InteriorSite, relocs.records)

            # every record's global is gone: slots are erased, boxes demoted to allocas
            for rec in relocs.records
                @test !haskey(globals(meta.ir), rec.name)
            end
            # the words are read out of the table, not baked into the module
            @test haskey(globals(meta.ir), Native.RELOC_TABLE_BASE)

            expected = reinterpret(UInt64, 2.0) +
                       GPUCompiler.resolve_relocation_target(
                           GPUCompiler.JuliaValueRef(:table_probe))
            fptr, lljit, table = Native.load(Vector{UInt8}(codeunits(obj)),
                                             LLVM.name(meta.entry), relocs; table=true)
            try
                GC.@preserve table begin
                    @test ccall(fptr, UInt, (Bool,), false) == expected
                    @test ccall(fptr, UInt, (Bool,), false) == mod.f(false)
                    @test ccall(fptr, UInt, (Bool,), true) == mod.f(true)
                end
            finally
                dispose(lljit)
            end

            # A record's index is its rank in the manifest, and that index is baked into the
            # emitted code — so recompiling the same kernel must produce the same manifest in
            # the same order, or a cached object and a freshly-resolved table would disagree.
            # (This is what makes a relocation-carrying kernel's *bytes* stable, which the
            # Metal.jl suite asserts on a real cache key.)
            _, again = GPUCompiler.compile(:obj, job)
            @test [(rec.kind, rec.name, rec.offset) for rec in again.relocations.records] ==
                  [(rec.kind, rec.name, rec.offset) for rec in relocs.records]

            # The delivered words follow the order the lowering fixed, not the record vector,
            # so losing records can no longer renumber the table. Pruning a copy drops *every*
            # record here (the lowering erased all their globals) and the words still stand.
            mutated = copy(relocs)
            GPUCompiler.prune_dead_relocations!(meta.ir, mutated)
            @test isempty(mutated)
            @test GPUCompiler.resolved_relocation_table(mutated) ==
                  GPUCompiler.resolved_relocation_table(relocs)

            # And the manifest itself refuses to be renumbered at all.
            @test_throws "already been lowered" GPUCompiler.prune_dead_relocations!(
                meta.ir, relocs)
        end
    end
end

@testset "unlowered relocation table" begin
    # Emitting a `:table` module through the 3-argument `emit_asm` hands the lowering an
    # empty manifest, leaving the real one unlowered and the module's slots stranded. The
    # loader is the first thing to notice, so it must say so rather than deliver no words.
    if GPUCompiler.supports_relocatable_ir() && LLVM.version() >= v"17"
        mod = @eval module $(gensym())
            f() = UInt(pointer_from_objref(:unlowered_probe))
        end
        job, _ = Native.create_job(mod.f, Tuple{}; relocations=:table)
        JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)
            @test !isempty(meta.relocations)
            GPUCompiler.emit_asm(job, ir, LLVM.API.LLVMObjectFile)   # the 3-arg form
            @test_throws "never rewritten" GPUCompiler.resolved_relocation_table(
                meta.relocations)
        end
    end
end

@testset "relocation-free tabulated module" begin
    # A module without relocations must not gain any table access at all: the strategy is
    # free for the (overwhelmingly common) relocation-free kernel.
    if LLVM.version() >= v"17"
        mod = @eval module $(gensym())
            f(x::Int) = x + 1
        end
        job, _ = Native.create_job(mod.f, (Int,); relocations=:table)
        JuliaContext() do ctx
            _, meta = GPUCompiler.compile(:obj, job)
            @test isempty(meta.relocations)
            @test !haskey(globals(meta.ir), Native.RELOC_TABLE_BASE)
        end
    end
end

@testset "relocation validation errors" begin
    JuliaContext() do ctx
        word() = GPUCompiler.relocation_word_type()
        nop = (_rec, _gv) -> nothing
        ref = GPUCompiler.JuliaValueRef(:probe)
        reloc(kind, name, offset=0) = GPUCompiler.Relocations(
            [GPUCompiler.Relocation(kind, name, offset, ref)])
        slot(name) = reloc(GPUCompiler.SlotSite, name)
        interior(name, offset) = reloc(GPUCompiler.InteriorSite, name, offset)

        # a record whose global is absent from the module
        mod = LLVM.Module("errors")
        @test_throws "Missing relocation global" GPUCompiler.foreach_relocation(
            nop, mod, slot("absent"))

        # a slot must be word-sized
        mod = LLVM.Module("errors")
        GlobalVariable(mod, LLVM.Int32Type(), "narrow")
        @test_throws "has size" GPUCompiler.foreach_relocation(nop, mod, slot("narrow"))

        # an interior record must name a definition...
        mod = LLVM.Module("errors")
        GlobalVariable(mod, word(), "decl")
        @test_throws "is a declaration" GPUCompiler.foreach_relocation(
            nop, mod, interior("decl", 0))

        # ...whose initializer is a struct...
        mod = LLVM.Module("errors")
        gv = GlobalVariable(mod, word(), "flat")
        initializer!(gv, ConstantInt(word(), 0))
        @test_throws "non-struct initializer" GPUCompiler.foreach_relocation(
            nop, mod, interior("flat", 0))

        # ...and it must land within that global
        mod = LLVM.Module("errors")
        gv = GlobalVariable(mod, LLVM.StructType([LLVM.Int64Type(), LLVM.Int64Type()]), "box")
        initializer!(gv, ConstantStruct(LLVM.Constant[ConstantInt(0), ConstantInt(0)]))
        @test_throws "outside its" GPUCompiler.foreach_relocation(
            nop, mod, interior("box", 16))
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
        relocs = GPUCompiler.Relocations()
        for (name, kind) in ("live"   => GPUCompiler.SlotSite,
                             "dead"   => GPUCompiler.InteriorSite,  # unused definition
                             "absent" => GPUCompiler.SlotSite)      # global already gone
            GPUCompiler.add_relocation!(relocs, kind, name, 0,
                                        GPUCompiler.JuliaValueRef(Symbol(name)))
        end
        GPUCompiler.prune_dead_relocations!(mod, relocs)
        @test [rec.name for rec in relocs.records] == ["live"]
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
        relocs = GPUCompiler.Relocations(
            [GPUCompiler.Relocation(GPUCompiler.InteriorSite, "zero_box", 0,
                                    GPUCompiler.JuliaValueRef(Float64))])
        GPUCompiler.bake_relocations!(mod, relocs)
        init = initializer(gv)
        @test !(init isa LLVM.ConstantAggregateZero)   # rebuilt into explicit fields
        header = convert(UInt, LLVM.Constant[operands(init)...][1])
        @test header == GPUCompiler.resolve_relocation_target(GPUCompiler.JuliaValueRef(Float64))
        @test isconstant(gv)
        @test isempty(relocs)
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
        ir = string(mod)
        @test !occursin("load $word_ptr, $word_ptr_ptr @jl_float32_type", ir)
        expected = GPUCompiler.resolve_relocation_target(
            GPUCompiler.CGlobalRef(:jl_float32_type))
        @test occursin("inttoptr (i64 $expected to $word_ptr)", ir)

        mod = parse(LLVM.Module, """
            @jl_float32_type = external global $word_ptr

            define $word_ptr @entry() {
                %value = load $word_ptr, $word_ptr_ptr @jl_float32_type
                ret $word_ptr %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(job, mod, relocs)
        rec = only(relocs.records)
        @test rec.target == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test rec.kind === GPUCompiler.SlotSite
        @test rec.offset == 0
        # the slot is a symbol a loader addresses, so its name carries the job's namespace
        @test startswith(rec.name, GPUCompiler.relocation_namespace(job))
        @test occursin("@$(rec.name) = external global i64", string(mod))
        GPUCompiler.emit_patchable_relocations!(mod, relocs)
        @test occursin("externally_initialized global i64 0", string(mod))

        mod = parse(LLVM.Module, """
            declare i64 @jl_float32_type()

            define i64 @entry() {
                %value = load i64, $(function_word_ptr("jl_float32_type"))
                ret i64 %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(job, mod, relocs)
        rec = only(relocs.records)
        @test rec.target == GPUCompiler.CGlobalRef(:jl_float32_type)
        @test occursin("@$(rec.name) = external global i64", string(mod))
        GPUCompiler.emit_patchable_relocations!(mod, relocs)
        @test occursin("externally_initialized global i64 0", string(mod))

        # Fold aggregate GEPs according to the module data layout.
        mod = parse(LLVM.Module, """
            target datalayout = "e-p:64:64-i64:64"
            @jl_layout = external global { i8, i64, [3 x i32] }

            define i32 @entry() {
                %value = load i32, $(ptr("i32")) getelementptr (
                    { i8, i64, [3 x i32] }, $(ptr("{ i8, i64, [3 x i32] }")) @jl_layout,
                    i64 0, i32 2, i64 1)
                ret i32 %value
            }""")
        load = first(instructions(first(blocks(functions(mod)["entry"]))))
        @test GPUCompiler.constexpr_byte_offset(operands(load)[1], datalayout(mod)) == 20
        @test_throws ArgumentError GPUCompiler.CGlobalRef(:jl_layout; offset=-1)

        # Julia codegen references small-tagged DataTypes through `jl_small_typeof` offsets.
        small_typeof = cglobal(:jl_small_typeof, Ptr{Cvoid})
        bool_index = findfirst(i -> unsafe_load(small_typeof, i) == pointer_from_objref(Bool),
                               1:(64 << 4) ÷ sizeof(Ptr{Cvoid}))
        bool_index === nothing && error("Bool is absent from jl_small_typeof")
        bool_offset = (bool_index - 1) * sizeof(Ptr{Cvoid})
        @test bool_offset > 0
        table_entry(offset) = GPUCompiler.supports_typed_pointers(ctx) ?
            "bitcast (i8* getelementptr (i8, i8* @jl_small_typeof, i64 $offset) to $word_ptr_ptr)" :
            "getelementptr (i8, ptr @jl_small_typeof, i64 $offset)"
        table_ir = """
            @jl_small_typeof = external global i8

            define $word_ptr @entry() {
                %bool = load $word_ptr, $word_ptr_ptr $(table_entry(bool_offset))
                %first = load $word_ptr, $word_ptr_ptr $(table_entry(0))
                %junk = ptrtoint $word_ptr %first to i64
                ret $word_ptr %bool
            }"""

        mod = parse(LLVM.Module, table_ir)
        relocs = GPUCompiler.Relocations()
        @test GPUCompiler.collect_cglobal_relocations!(job, mod, relocs)
        @test !GPUCompiler.has_unresolved_cglobal_loads(mod, relocs)
        @test length(relocs.records) == 2
        @test allunique(rec.name for rec in relocs.records)
        @test all(rec -> endswith(rec.name, "_$(rec.target.offset)"), relocs.records)
        @test Set(rec.target for rec in relocs.records) ==
              Set([GPUCompiler.CGlobalRef(:jl_small_typeof),
                   GPUCompiler.CGlobalRef(:jl_small_typeof; offset=bool_offset)])
        @test all(rec -> rec.kind === GPUCompiler.SlotSite && rec.offset == 0, relocs.records)
        @test GPUCompiler.resolve_relocation_target(
                  GPUCompiler.CGlobalRef(:jl_small_typeof; offset=bool_offset)) ==
              UInt(pointer_from_objref(Bool))

        mod = parse(LLVM.Module, table_ir)
        GPUCompiler.prepare_execution!(job, mod)
        @test occursin("inttoptr (i64 $(UInt(pointer_from_objref(Bool))) to $word_ptr)",
                       string(mod))

        # Never silently relocate a dynamically-indexed load to the base word.
        mod = parse(LLVM.Module, """
            @jl_small_typeof = external global i8
            @jl_float32_type = external global i8

            define $word_ptr @entry() {
                %value = load $word_ptr, $word_ptr_ptr $(GPUCompiler.supports_typed_pointers(ctx) ?
                    "bitcast (i8* getelementptr (i8, i8* @jl_small_typeof, i64 ptrtoint (i8* @jl_float32_type to i64)) to $word_ptr_ptr)" :
                    "getelementptr (i8, ptr @jl_small_typeof, i64 ptrtoint (ptr @jl_float32_type to i64))")
                ret $word_ptr %value
            }""")
        relocs = GPUCompiler.Relocations()
        @test_throws_message(ErrorException,
                             GPUCompiler.collect_cglobal_relocations!(job, mod, relocs)) do msg
            occursin("Unsupported cglobal 'jl_small_typeof' load", msg)
        end
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

        slot_relocs(name, value) = GPUCompiler.Relocations(
            [GPUCompiler.Relocation(GPUCompiler.SlotSite, name, 0,
                                    GPUCompiler.JuliaValueRef(value))])

        # Per-job namespacing means a name never denotes two different words. Metadata for the
        # same word still merges, though — the runtime library is linked into every kernel.
        dest = slot_module("slot", "first")
        dest_relocs = slot_relocs("slot", :shared)
        src = slot_module("slot", "second")
        src_relocs = slot_relocs("slot", :shared)
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs)
        @test [rec.name for rec in dest_relocs.records] == ["slot"]
        @test occursin("@slot = external global i64", string(dest))

        # Conflicting metadata for one word is an inconsistency, not a merge.
        dest = slot_module("slot", "first")
        dest_relocs = slot_relocs("slot", :first)
        src = slot_module("slot", "second")
        src_relocs = slot_relocs("slot", :second)
        @test_throws "conflicting values" GPUCompiler.link_relocatable!(
            dest, dest_relocs, src, src_relocs)

        # ...as is disagreement about what the word even is.
        dest_relocs = slot_relocs("slot", :shared)
        src_relocs = GPUCompiler.Relocations(
            [GPUCompiler.Relocation(GPUCompiler.InteriorSite, "slot", 0,
                                    GPUCompiler.JuliaValueRef(:shared))])
        @test_throws "recorded as both" GPUCompiler.add_relocation!(
            dest_relocs, only(src_relocs.records))

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
        src_relocs = GPUCompiler.Relocations()
        for name in ("used", "unused")
            GPUCompiler.add_relocation!(src_relocs, GPUCompiler.SlotSite, name, 0,
                                        GPUCompiler.JuliaValueRef(Symbol(name)))
        end
        dest_relocs = GPUCompiler.Relocations()
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs;
                                       only_needed=true)
        @test [rec.name for rec in dest_relocs.records] == ["used"]
        @test only(dest_relocs.records).target.value === :used

        # Metadata for interior globals not imported under `only_needed` is discarded too.
        dest = parse(LLVM.Module, """
            declare i64 @source_patch()
            define i64 @entry_patch() {
                %value = call i64 @source_patch()
                ret i64 %value
            }""")
        src = parse(LLVM.Module, """
            @used_patch = externally_initialized global { i64, i64 } { i64 0, i64 1 }

            define i64 @source_patch() {
                %value = load i64, $(ptr("i64")) getelementptr ({ i64, i64 }, $(ptr("{ i64, i64 }")) @used_patch, i32 0, i32 1)
                ret i64 %value
            }""")
        unused = GlobalVariable(src, LLVM.StructType([LLVM.Int64Type(), LLVM.Int64Type()]),
                                "unused_patch")
        initializer!(unused, ConstantStruct(LLVM.Constant[ConstantInt(0), ConstantInt(1)]))
        src_relocs = GPUCompiler.Relocations()
        for (name, T) in ("used_patch" => Float64, "unused_patch" => Int64)
            GPUCompiler.add_relocation!(src_relocs, GPUCompiler.InteriorSite, name, 0,
                                        GPUCompiler.JuliaValueRef(T))
        end
        dest_relocs = GPUCompiler.Relocations()
        GPUCompiler.link_relocatable!(dest, dest_relocs, src, src_relocs;
                                       only_needed=true)
        @test [rec.name for rec in dest_relocs.records] == ["used_patch"]
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

@testset "runtime functions from overlay methods" begin
    # runtime library functions (`signal_exception`, `malloc`, ...) should be resolved
    # through the job's method table, so that back-ends can keep GPU-only code out of
    # the global method table (JuliaGPU/GPUCompiler.jl#611)
    isdefined(Native.Runtime, :signal_exception) ||
        @eval Native.Runtime signal_exception() = nothing

    mod = @eval module $(gensym())
        using ..GPUCompiler
        import ..Native

        Base.Experimental.@MethodTable(method_table)

        Base.Experimental.@overlay method_table Native.Runtime.signal_exception() = nothing

        kernel(x) = x
    end

    method = GPUCompiler.Runtime.methods[:signal_exception]

    job, _ = Native.create_job(mod.kernel, (Int,); method_table=mod.method_table)
    mi = GPUCompiler.runtime_method_instance(job, method)
    @test mi.def.module === mod

    # the global definition is used with the global method table
    job, _ = Native.create_job(mod.kernel, (Int,); method_table=GPUCompiler.GLOBAL_METHOD_TABLE)
    mi = GPUCompiler.runtime_method_instance(job, method)
    @test mi.def.module === Native.Runtime
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
    job, _ = Native.create_job(mod.parent, (Symbol,); relocations=:patch, validate=false)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @test !occursin("deferred_codegen", string(ir))
        if GPUCompiler.supports_relocatable_ir()
            @test any(meta.relocations.records) do rec
                rec.target isa GPUCompiler.JuliaValueRef &&
                    rec.target.value === :deferred_reloc
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
