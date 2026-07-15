@testset "IR" begin

@testset "exceptions" begin
    # plain exceptions should get lowered to a call to the GPU run-time, not a
    # jl_throw referencing a jl_value_t representing the exception
    @test @filecheck PTX.code_llvm(Tuple{}; dump_module=true) do
        @check_not "jl_throw"
        @check "gpu_report_exception"
        throw(DivideError())
    end
end

@testset "kernel state survives a runtime rebuild" begin
    # Clearing the runtime cache forces the library link inside `emit_llvm` to rebuild
    # the runtime (nested compilation); the kernel must still get its state argument
    # afterwards.
    mod = @eval module $(gensym())
        function kernel(x)
            x < 1 && throw(DivideError())
            return
        end
    end
    Base.@lock GPUCompiler.runtime_libs_lock empty!(GPUCompiler.runtime_libs)
    ir = sprint() do io
        PTX.code_llvm(io, mod.kernel, Tuple{Int}; kernel=true, dump_module=true)
    end
    @test occursin("gpu_report_exception", ir)
    @test occursin("[1 x i64] %state", ir)
end

@testset "global variable relocation" begin
    # references to Julia objects (`julia.constgv` globals, e.g. Symbol literals) must
    # survive until `relocate_gvs!` bakes in their addresses at the toplevel link step.
    # they used to be kept alive as internal globals with a null initializer, which the
    # GlobalOpt run in `finish_module!` folded away, constant-folding any comparison
    # against them (JuliaGPU/CUDA.jl#3185: kernels specialized on Symbols misbehaved).
    mod = @eval module $(gensym())
        kernel(name::Symbol) = name === :var ? 1 : 2
    end
    ir = sprint() do io
        PTX.code_llvm(io, mod.kernel, Tuple{Symbol}; dump_module=true)
    end
    # Julia 1.10 may bake the Symbol pointer into the generated IR before exposing it to
    # GPUCompiler. Newer versions provide a symbolic global that we can preserve and relocate.
    @static if VERSION >= v"1.11"
        @test occursin("@jl_sym_var_", ir)
    else
        @test occursin("@jl_sym_var_", ir) || occursin("inttoptr", ir)
    end
    @test !occursin("@\"jl_sym#", ir)
end

@testset "Julia value global names" begin
    # Julia's external codegen can name the declaration for this Symbol with `#`.
    # The final PTX path must see the sanitized Julia value global, not the original
    # declaration name rejected by NVPTX.
    mod = @eval module $(gensym())
        const unusual_symbol = Symbol("value#global")
        kernel(name::Symbol) = (name === unusual_symbol; return)
    end
    @test PTX.code_execution(mod.kernel, Tuple{Symbol}) !== nothing
end

@testset "boxed Bool singleton relocation" begin
    @static if VERSION >= v"1.14.0-DEV.1348"
        mod = @eval module $(gensym())
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
        end

        for (f, name) in ((mod.consume_true, "jl_true"),
                          (mod.consume_false, "jl_false"))
            ir = sprint(io->PTX.code_llvm(io, f, Tuple{Bool, Int32};
                                          dump_module=true))
            @test occursin("@$(name)_box = private unnamed_addr constant", ir)
            @test !occursin("@$name = external", ir)
            @test !occursin("inttoptr", ir)
        end
    end
end

@testset "kernel functions" begin
@testset "kernel argument attributes" begin
    mod = @eval module $(gensym())
        kernel(x) = return

        struct Aggregate
            x::Int
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check_same cond=typed_ptrs "({{({ i64 }|\\[1 x i64\\])}}*"
        @check_same cond=opaque_ptrs "(ptr"
        PTX.code_llvm(mod.kernel, Tuple{mod.Aggregate})
    end

    @test @filecheck begin
        @check_label "define ptx_kernel void @_Z6kernel9Aggregate"
        @check_not cond=typed_ptrs "*"
        @check_not cond=opaque_ptrs "ptr"
        PTX.code_llvm(mod.kernel, Tuple{mod.Aggregate}; kernel=true)
    end
end

LLVM.version() >= v"8" && @testset "calling convention" begin
    @test @filecheck PTX.code_llvm(Tuple{}; dump_module=true) do
        @check_not "ptx_kernel"
        return
    end
    @test @filecheck PTX.code_llvm(Tuple{}; dump_module=true, kernel=true) do
        @check "ptx_kernel"
        return
    end
end

@testset "kernel state" begin
    # state should be passed by value to kernel functions

    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check "@{{(julia|j)_kernel[0-9_]*}}()"
        PTX.code_llvm(mod.kernel, Tuple{})
    end

    @test @filecheck begin
        @check "@_Z6kernel([1 x i64] %state)"
        PTX.code_llvm(mod.kernel, Tuple{}; kernel=true)
    end

    # state should only passed to device functions that use it

    mod = @eval module $(gensym())
        @noinline child1(ptr) = unsafe_load(ptr)
        @noinline function child2()
            data = $PTX.kernel_state().data
            ptr = reinterpret(Ptr{Int}, data)
            unsafe_load(ptr)
        end

        function kernel(ptr)
            unsafe_store!(ptr, child1(ptr) + child2())
            return
        end
    end

    # kernel should take state argument before all else
    @test @filecheck begin
        @check_label "define ptx_kernel void @_Z6kernelP5Int64([1 x i64] %state"
        @check_not "julia.gpu.state_getter"
        PTX.code_llvm(mod.kernel, Tuple{Ptr{Int64}}; kernel=true, dump_module=true)
    end
    # child1 doesn't use the state
    @test @filecheck begin
        @check_label "define{{.*}} i64 @{{(julia|j)_child1_[0-9]+}}"
        PTX.code_llvm(mod.kernel, Tuple{Ptr{Int64}}; kernel=true, dump_module=true)
    end
    # child2 does
    @test @filecheck begin
        @check_label "define{{.*}} i64 @{{(julia|j)_child2_[0-9]+}}"
        PTX.code_llvm(mod.kernel, Tuple{Ptr{Int64}}; kernel=true, dump_module=true)
    end
end
end

@testset "Mock Enzyme" begin
    function kernel(a)
        unsafe_store!(a, unsafe_load(a)^2)
        return
    end
    
    function dkernel(a)
        ptr = Enzyme.deferred_codegen(typeof(kernel), Tuple{Ptr{Float64}})
        ccall(ptr, Cvoid, (Ptr{Float64},), a)
        return
    end

    ir = sprint(io->Native.code_llvm(io, dkernel, Tuple{Ptr{Float64}}; debuginfo=:none))
    @test !occursin("deferred_codegen", ir)
    @test occursin("call void @julia_", ir)
end

end

############################################################################################
if :NVPTX in LLVM.backends()
@testset "assembly" begin

@testset "boxed Bool singleton relocation" begin
    @static if VERSION >= v"1.14.0-DEV.1348"
        mod = @eval module $(gensym())
            @noinline produce(cond::Bool, a::Int32) = cond ? a : true
            function consume(cond::Bool, a::Int32)
                x = produce(cond, a)
                x isa Bool && x && return Int32(1)
                return Int32(0)
            end
        end
        ptx = sprint(io->PTX.code_native(io, mod.consume, Tuple{Bool, Int32};
                                         dump_module=true))
        @test occursin("jl_true_box", ptx)
        @test !occursin(r"(?m)^\.extern .*\bjl_true\b", ptx)
    end
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)

    mod = @eval module $(gensym())
        import ..sink
        @noinline child(i) = sink(i)
        function parent(i)
            child(i)
            return
        end
    end

    # the assembler emits `call.uni` and the callee name on the same line in
    # LLVM 21+, but on separate lines on older releases.
    @test @filecheck begin
        @check_label ".visible .func {{(julia|j)_parent[0-9_]*}}"
        @check "call.uni"
        @check_same "{{(julia|j)_child_}}"
        PTX.code_native(mod.parent, Tuple{Int64})
    end
end

@testset "kernel functions" begin
    mod = @eval module $(gensym())
        import ..sink
        @noinline nonentry(i) = sink(i)
        function entry(i)
            nonentry(i)
            return
        end
    end

    @test @filecheck begin
        @check_not ".visible .func {{(julia|j)_nonentry}}"
        @check_label ".visible .entry _Z5entry5Int64"
        @check "{{(julia|j)_nonentry}}"
        PTX.code_native(mod.entry, Tuple{Int64}; kernel=true, dump_module=true)
    end

@testset "property_annotations" begin
    @test @filecheck begin
        @check_not "maxntid"
        PTX.code_native(mod.entry, Tuple{Int64}; kernel=true)
    end

    @test @filecheck begin
        @check ".maxntid 42, 1, 1"
        PTX.code_native(mod.entry, Tuple{Int64}; kernel=true, maxthreads=42)
    end

    @test @filecheck begin
        @check ".reqntid 42, 1, 1"
        PTX.code_native(mod.entry, Tuple{Int64}; kernel=true, minthreads=42)
    end

    @test @filecheck begin
        @check ".minnctapersm 42"
        PTX.code_native(mod.entry, Tuple{Int64}; kernel=true, blocks_per_sm=42)
    end

    if LLVM.version() >= v"4.0"
        @test @filecheck begin
            @check ".maxnreg 42"
            PTX.code_native(mod.entry, Tuple{Int64}; kernel=true, maxregs=42)
        end
    end
end
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    mod = @eval module $(gensym())
        import ..sink
        @noinline child(i) = sink(i)
        function parent1(i)
            child(i)
            return
        end
        function parent2(i)
            child(i+1)
            return
        end
    end

    @test @filecheck begin
        @check ".func {{(julia|j)_child}}"
        PTX.code_native(mod.parent1, Tuple{Int})
    end

    @test @filecheck begin
        @check ".func {{(julia|j)_child}}"
        PTX.code_native(mod.parent2, Tuple{Int})
    end
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions

    mod = @eval module $(gensym())
        import ..sink
        @noinline child1(i) = sink(i)
        @noinline child2(i) = sink(i+1)
        function parent1(i)
            child1(i) + child2(i)
            return
        end
        function parent2(i)
            child1(i+1) + child2(i+1)
            return
        end
    end

    @test @filecheck begin
        @check_dag ".func {{(julia|j)_child1}}"
        @check_dag ".func {{(julia|j)_child2}}"
        PTX.code_native(mod.parent1, Tuple{Int})
    end

    @test @filecheck begin
        @check_dag ".func {{(julia|j)_child1}}"
        @check_dag ".func {{(julia|j)_child2}}"
        PTX.code_native(mod.parent2, Tuple{Int})
    end
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the PTX code shouldn't contain a throw)

    # NOTE: Int32 to test for #49
    mod = @eval module $(gensym())
        function kernel(out)
            wid, lane = fldmod1(unsafe_load(out), Int32(32))
            unsafe_store!(out, wid)
            return
        end
    end

    @test @filecheck begin
        @check_label ".visible .func {{(julia|j)_kernel[0-9_]*}}"
        @check_not "jl_throw"
        @check_not "jl_invoke"
        PTX.code_native(mod.kernel, Tuple{Ptr{Int32}})
    end
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    mod = @eval module $(gensym())
        function kernel(x)
            unsafe_trunc(Int, x)
            return
        end
    end
    PTX.code_native(devnull, mod.kernel, Tuple{Float64})
    @test "We did not crash!" != ""
end

@testset "exception arguments" begin
    mod = @eval module $(gensym())
        function kernel(a)
            unsafe_store!(a, trunc(Int, unsafe_load(a)))
            return
        end
    end
    PTX.code_native(devnull, mod.kernel, Tuple{Ptr{Float64}})
    @test "We did not crash!" != ""
end

@testset "GC and TLS lowering" begin
    mod = @eval module $(gensym())
        import ..sink

        mutable struct PleaseAllocate
            y::Csize_t
        end

        # common pattern in Julia 0.7: outlined throw to avoid a GC frame in the calling code
        @noinline function inner(x)
            sink(x.y)
            nothing
        end

        function kernel(i)
            inner(PleaseAllocate(Csize_t(42)))
            nothing
        end
    end

    @test @filecheck begin
        @check_label ".visible .func {{(julia|j)_kernel[0-9_]*}}"
        @check_not "julia.push_gc_frame"
        @check_not "julia.pop_gc_frame"
        @check_not "julia.get_gc_frame_slot"
        @check_not "julia.new_gc_frame"
        @check "gpu_gc_pool_alloc"
        PTX.code_native(mod.kernel, Tuple{Int})
    end

    # make sure that we can still ellide allocations
    mod = @eval module $(gensym())
        function ref_kernel(ptr, i)
            data = Ref{Int64}()
            data[] = 0
            if i > 1
                data[] = 1
            else
                data[] = 2
            end
            unsafe_store!(ptr, data[], i)
            return nothing
        end
    end

    @test @filecheck begin
        @check_label ".visible .func {{(julia|j)_ref_kernel[0-9_]*}}"
        @check_not "gpu_gc_pool_alloc"
        PTX.code_native(mod.ref_kernel, Tuple{Ptr{Int64}, Int})
    end
end

@testset "float boxes" begin
    mod = @eval module $(gensym())
        function kernel(a,b)
            # Int32(a) may fail, boxing the Float32 for the @nospecialize ctor
            c = Int32(a)
            unsafe_store!(b, c)
            return
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        # box: a jl_box_float32 call <1.14; 1.14+ inlines it into the devirt'd ctor
        @check cond=(VERSION < v"1.14-")  "jl_box_float32"
        @check cond=(VERSION >= v"1.14-") "gpu_gc_pool_alloc"
        PTX.code_llvm(mod.kernel, Tuple{Float32,Ptr{Float32}}; dump_module=true)
    end
    PTX.code_native(devnull, mod.kernel, Tuple{Float32,Ptr{Float32}})
end

@testset "FMA contraction" begin
    function contract(x)
        for _ in 1:8
            x = 2.001f0 * x - x
        end
        return x
    end

    ptx = sprint(PTX.code_native, contract, Tuple{Float32})
    @test occursin("mul.f32", ptx)
    @test occursin("sub.f32", ptx)
    @test !occursin("mul.rn.f32", ptx)
    @test !occursin("sub.rn.f32", ptx)
end

@testset "fastmath" begin
    # `fastmath=true` on the target calls `apply_fastmath!` from
    # `finish_linked_module!`, stamping `unsafe-fp-math` + fast-math flags on
    # every FP op and setting `denormal-fp-math-f32` so NVPTX picks the FTZ
    # variants. Sweep both axes (IR attributes + PTX selection) across the
    # flag.
    for fastmath in (false, true)
        # IR attributes; the back-end selects approximate instructions from these.
        @test @filecheck PTX.code_llvm(Tuple{Ptr{Float32},Ptr{Float32}};
                                       dump_module=true, fastmath) do x, out
            @check_not cond=!fastmath "unsafe-fp-math"
            @check_not cond=!fastmath "denormal-fp-math-f32"
            @check     cond=!fastmath "call float @llvm.sqrt.f32"
            @check     cond=fastmath  "call fast float @llvm.sqrt.f32"
            @check     cond=fastmath  "\"denormal-fp-math-f32\"=\"preserve-sign,preserve-sign\""
            @check     cond=fastmath  "\"unsafe-fp-math\"=\"true\""
            unsafe_store!(out, sqrt(unsafe_load(x)))
            return
        end
        # PTX-level selection.
        @test @filecheck PTX.code_native(Tuple{Ptr{Float32},Ptr{Float32}}; fastmath) do x, out
            @check     cond=fastmath  "sqrt.approx.ftz.f32"
            @check     cond=!fastmath "sqrt.rn.f32"
            @check_not cond=!fastmath "sqrt.approx"
            unsafe_store!(out, sqrt(unsafe_load(x)))
            return
        end
    end
end

@testset "fastmath fdiv/fsqrt" begin
    # `afn`-flagged f32 fdiv / `llvm.sqrt.f32` are selected as
    # `*.approx{,.ftz}.f32` by the back-end. For f64, `PTXFDivFastPass` and
    # `PTXFSqrtFastPass` rewrite fdiv → `rcp.approx.ftz.d` + Newton refinement
    # and sqrt → `rcp.approx.ftz.d(rsqrt.approx.d(x))` (NVPTX has no native
    # fast f64). Per-call `@fastmath` is what we test here; the job-wide path
    # is covered by the testset above.

    @test @filecheck PTX.code_native(Tuple{Float32, Float32}) do x, y
        @check "div.approx.f32"
        @fastmath x / y
    end
    @test @filecheck PTX.code_native(Tuple{Float32, Float32}) do x, y
        @check_not "div.approx"
        x / y
    end
    @test @filecheck PTX.code_native(Tuple{Float64, Float64}) do x, y
        @check "rcp.approx.ftz.f64"
        @fastmath x / y
    end
    @test @filecheck PTX.code_native(Tuple{Float64, Float64}) do x, y
        @check_not "rcp.approx"
        x / y
    end

    @test @filecheck PTX.code_native(Tuple{Float32}) do x
        @check "sqrt.approx.f32"
        @fastmath sqrt(x)
    end
    @test @filecheck PTX.code_native(Tuple{Float32}) do x
        @check "sqrt.rn.f32"
        @check_not "sqrt.approx"
        sqrt(x)
    end
    @test @filecheck PTX.code_native(Tuple{Float64}) do x
        @check "rsqrt.approx.f64"
        @check "rcp.approx.ftz.f64"
        @fastmath sqrt(x)
    end
    @test @filecheck PTX.code_native(Tuple{Float64}) do x
        @check "sqrt.rn.f64"
        @check_not "rsqrt"
        sqrt(x)
    end
end

@testset "fastmath rsqrt" begin
    # `fdiv afn 1.0, sqrt afn(x)` folds to a single `rsqrt.approx`
    # instruction: for f32 by the back-end's ISel patterns, for f64 by
    # `PTXRSqrtFastPass` (which must claim the pattern before the per-op f64
    # rewrites expand it into `rcp(rsqrt(...))` and a Newton step). Without
    # afn on both operands, the fold doesn't fire — it would change precision.
    mod = @eval module $(gensym())
        rsqrt32_fast(x::Float32) = @fastmath 1f0 / sqrt(x)
        rsqrt64_fast(x::Float64) = @fastmath 1.0 / sqrt(x)
        rsqrt32(x::Float32) = 1f0 / sqrt(x)
        rsqrt64(x::Float64) = 1.0 / sqrt(x)
    end

    @test @filecheck begin
        @check "rsqrt.approx.f32"
        @check_not "sqrt.approx"
        @check_not "div.approx"
        PTX.code_native(mod.rsqrt32_fast, Tuple{Float32})
    end
    @test @filecheck begin
        @check "rsqrt.approx.ftz.f32"
        @check_not "sqrt.approx"
        @check_not "div.approx"
        PTX.code_native(mod.rsqrt32_fast, Tuple{Float32}; fastmath=true)
    end
    @test @filecheck begin
        @check "rsqrt.approx.f64"
        @check_not "rcp.approx"
        PTX.code_native(mod.rsqrt64_fast, Tuple{Float64})
    end
    @test @filecheck begin
        # job-wide fastmath stamps afn on all FP ops, so the pattern still fires
        @check "rsqrt.approx.f64"
        @check_not "rcp.approx"
        PTX.code_native(mod.rsqrt64, Tuple{Float64}; fastmath=true)
    end

    # Without afn, plain `1/sqrt(x)` must NOT fold to rsqrt: it would change
    # precision. The non-fast f64 emits `sqrt.rn.f64 + div.rn.f64`.
    @test @filecheck begin
        @check "sqrt.rn.f64"
        @check_not "rsqrt.approx"
        PTX.code_native(mod.rsqrt64, Tuple{Float64})
    end
end

@testset "feature_set" begin
    # PTXCompilerTarget.feature_set controls the suffix on the LLVM CPU name, which is
    # what the NVPTX backend uses to flip `hasArchAccelFeatures()`. Verify it makes its
    # way into the `.target` directive that LLVM emits and into the hash.

    mod = @eval module $(gensym())
        kernel() = return
    end

    # cpu_name reflects feature_set
    @test GPUCompiler.cpu_name(PTXCompilerTarget(cap=v"9.0")) == "sm_90"
    @test GPUCompiler.cpu_name(PTXCompilerTarget(cap=v"9.0", feature_set=:baseline)) == "sm_90"
    @test GPUCompiler.cpu_name(PTXCompilerTarget(cap=v"9.0", feature_set=:arch)) == "sm_90a"
    @test GPUCompiler.cpu_name(PTXCompilerTarget(cap=v"10.0", feature_set=:family)) == "sm_100f"
    @test_throws ErrorException GPUCompiler.cpu_name(PTXCompilerTarget(cap=v"9.0", feature_set=:bogus))

    # hash must discriminate, otherwise two targets differing only on feature_set
    # could compare equal inside cache-owner keys.
    @test hash(PTXCompilerTarget(cap=v"9.0", feature_set=:baseline)) !=
          hash(PTXCompilerTarget(cap=v"9.0", feature_set=:arch))

    # LLVM picked up `sm_90a` in v18 (NVPTX.td); older releases don't know the suffix.
    if LLVM.version() >= v"18"
        @test @filecheck begin
            @check ".target sm_90a"
            PTX.code_native(mod.kernel, Tuple{}; cap=v"9.0", ptx=v"8.0",
                            feature_set=:arch, kernel=true, dump_module=true)
        end
    end
    # `sm_100f` (and the rest of the family-/arch-specific Blackwell variants) was added in LLVM 20.
    if LLVM.version() >= v"20"
        @test @filecheck begin
            @check ".target sm_100f"
            PTX.code_native(mod.kernel, Tuple{}; cap=v"10.0", ptx=v"8.8",
                            feature_set=:family, kernel=true, dump_module=true)
        end
        @test @filecheck begin
            @check ".target sm_100a"
            PTX.code_native(mod.kernel, Tuple{}; cap=v"10.0", ptx=v"8.6",
                            feature_set=:arch, kernel=true, dump_module=true)
        end
    end
end

end
end # NVPTX in LLVM.backends()
