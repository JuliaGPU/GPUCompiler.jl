if :AMDGPU in LLVM.backends()

# XXX: generic `sink` generates an instruction selection error
sink_gcn(i) = sink(i, Val(5))

@testset "backend selector" begin
    # in the test environment AMDGPU_LLVM_Backend_jll is loaded, so the default is :external
    @test GCNCompilerTarget(dev_isa="gfx900").backend === :external

    # both constructor forms accept an explicit backend, alongside the other options
    @test GCNCompilerTarget(dev_isa="gfx900"; backend=:inprocess).backend === :inprocess
    @test GCNCompilerTarget("gfx900"; backend=:inprocess).backend === :inprocess
    let target = GCNCompilerTarget("gfx900"; features="+wavefrontsize64", backend=:external)
        @test target.dev_isa == "gfx900"
        @test target.features == "+wavefrontsize64"
        @test target.backend === :external
    end

    mod = @eval module $(gensym())
        kernel() = return
    end

    # the backend participates in the runtime slug, so different back-ends don't share a cache
    job_ext, _ = GCN.create_job(mod.kernel, Tuple{}; backend=:external)
    job_inp, _ = GCN.create_job(mod.kernel, Tuple{}; backend=:inprocess)
    @test endswith(GPUCompiler.runtime_slug(job_ext), "-external")
    @test endswith(GPUCompiler.runtime_slug(job_inp), "-inprocess")
    @test GPUCompiler.runtime_slug(job_ext) != GPUCompiler.runtime_slug(job_inp)

    # the explicit :external backend generates machine code through the external llc
    @test (GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:external); true)

    # the :inprocess backend generates machine code through the in-process LLVM back-end
    @test (GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:inprocess); true)

    # an unknown back-end is rejected at machine-code generation
    @test_throws "Unsupported GCN back-end" GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:bogus)
end

@testset "IR" begin

@testset "kernel calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "amdgpu_kernel"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true)
    end

    @test @filecheck begin
        @check "amdgpu_kernel"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
    end
end

@testset "bounds errors" begin
    mod = @eval module $(gensym())
        function kernel()
            Base.throw_boundserror(1, 2)
            return
        end
    end

    @test @filecheck begin
        @check_not "{{julia_throw_boundserror_[0-9]+}}"
        @check "@gpu_report_exception"
        @check "@gpu_signal_exception"
        GCN.code_llvm(mod.kernel, Tuple{})
    end
end

@testset "kernarg address space for byref parameters" begin
    mod = @eval module $(gensym())
        struct MyStruct
            x::Float64
            y::Float64
        end

        function kernel(s::MyStruct)
            s.x + s.y
            return
        end
    end

    # byref struct params should be ptr addrspace(4) in kernel IR
    @test @filecheck begin
        @check cond=typed_ptrs "define amdgpu_kernel void @_Z6kernel8MyStruct({{.*}} addrspace(4)*"
        @check cond=opaque_ptrs "define amdgpu_kernel void @_Z6kernel8MyStruct(ptr addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.MyStruct}; dump_module=true, kernel=true)
    end

    # non-kernel should NOT have addrspace(4)
    @test @filecheck begin
        @check_not "addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.MyStruct}; dump_module=true, kernel=false)
    end
end

@testset "byref attribute preserved on kernarg parameters" begin
    mod = @eval module $(gensym())
        struct LargeStruct
            a::Float64
            b::Float64
            c::Float64
            d::Float64
        end

        function kernel(s::LargeStruct, out::Ptr{Float64})
            unsafe_store!(out, s.a + s.b + s.c + s.d)
            return
        end
    end

    # the byref attribute must survive the addrspace rewrite (clone_into! can drop it)
    @test @filecheck begin
        @check "byref"
        @check "addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.LargeStruct, Ptr{Float64}};
                       dump_module=true, kernel=true)
    end
end

@testset "mixed byref and scalar kernel parameters" begin
    mod = @eval module $(gensym())
        struct Params
            x::Float64
            y::Float64
        end

        function kernel(a::Float64, s::Params, out::Ptr{Float64})
            unsafe_store!(out, a + s.x + s.y)
            return
        end
    end

    # scalar Float64 should NOT be in addrspace(4),
    # only the struct byref param should be.
    # NOTE: Ptr{Float64} is lowered to i64 on Julia ≤1.11 and ptr on Julia 1.12+.
    @test @filecheck begin
        @check "define amdgpu_kernel void"
        @check_same "double"
        @check_same cond=typed_ptrs "{{.*}} addrspace(4)*"
        @check_same cond=opaque_ptrs "ptr addrspace(4)"
        @check_same "{{(i64|ptr)}}"
        GCN.code_llvm(mod.kernel, Tuple{Float64, mod.Params, Ptr{Float64}};
                       dump_module=true, kernel=true)
    end
end

@testset "add_kernarg_address_spaces! rewrites IR correctly" begin
    mod = @eval module $(gensym())
        struct KernelArgs
            x::Float64
            y::Float64
            z::Float64
        end

        function kernel(s::KernelArgs, scale::Float64, out::Ptr{Float64})
            unsafe_store!(out, (s.x + s.y + s.z) * scale)
            return
        end
    end

    job, _ = GCN.create_job(mod.kernel, Tuple{mod.KernelArgs, Float64, Ptr{Float64}};
                             kernel=true)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)

        entry = meta.entry
        ft = function_type(entry)
        params = parameters(ft)

        # the struct byref param should be ptr addrspace(4)
        has_as4 = any(p -> p isa LLVM.PointerType && addrspace(p) == 4, params)
        @test has_as4

        # non-struct params (double, and i64/ptr for Ptr{Float64}) should NOT
        # be in addrspace(4). Ptr{Float64} is i64 on Julia ≤1.11, ptr on 1.12+.
        non_byref = filter(p -> !(p isa LLVM.PointerType && addrspace(p) == 4), params)
        @test !isempty(non_byref)  # double (and i64 or ptr) params

        # byref attribute must be present
        ir_str = string(ir)
        @test occursin("byref", ir_str)

        dispose(ir)
    end
end

@testset "https://github.com/JuliaGPU/AMDGPU.jl/issues/846" begin
    ir, rt = GCN.code_typed((Tuple{Tuple{Val{4}}, Tuple{Float32}},); always_inline=true) do t
        t[1]
    end |> only
    @test rt == Tuple{Val{4}}
end

end

############################################################################################
@testset "assembly" begin

@testset "s_load for kernarg struct access" begin
    mod = @eval module $(gensym())
        struct MyStruct
            x::Float64
            y::Float64
        end

        function kernel(s::MyStruct, out::Ptr{Float64})
            unsafe_store!(out, s.x + s.y)
            return
        end
    end

    # struct field loads from kernarg should use s_load, not flat_load
    @test @filecheck begin
        @check "s_load_dwordx"
        @check_not "flat_load"
        GCN.code_native(mod.kernel, Tuple{mod.MyStruct, Ptr{Float64}}; kernel=true)
    end
end

@testset "no scratch spills for small struct kernarg" begin
    mod = @eval module $(gensym())
        struct SmallStruct
            x::Float64
            y::Float64
        end

        function kernel(s::SmallStruct, out::Ptr{Float64})
            unsafe_store!(out, s.x + s.y)
            return
        end
    end

    # a small struct kernel should not need scratch memory
    @test @filecheck begin
        @check ".private_segment_fixed_size: 0"
        GCN.code_native(mod.kernel, Tuple{mod.SmallStruct, Ptr{Float64}};
                         dump_module=true, kernel=true)
    end
end

@testset "skip scalar trap" begin
    mod = @eval module $(gensym())
        workitem_idx_x() = ccall("llvm.amdgcn.workitem.id.x", llvmcall, Int32, ())
        trap() = ccall("llvm.trap", llvmcall, Nothing, ())

        function kernel()
            if workitem_idx_x() > 1
                trap()
            end
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_kernel_[0-9]+}}:"
        @check "s_cbranch_exec"
        @check "s_trap 2"
        GCN.code_native(mod.kernel, Tuple{})
    end
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child(i) = sink_gcn(i)
        function parent(i)
            child(i)
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_parent_[0-9]+}}:"
        @check "s_add_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
        @check "s_addc_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
        GCN.code_native(mod.parent, Tuple{Int64}; dump_module=true)
    end
end

@testset "kernel functions" begin
    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline nonentry(i) = sink_gcn(i)
        function entry(i)
            nonentry(i)
            return
        end
    end

    @test @filecheck begin
        @check ".type {{(julia|j)_nonentry_[0-9]+}},@function"
        @check ".symbol:{{.*}}_Z5entry5Int64.kd"
        @check_not ".symbol:{{.*}}nonentry"
        GCN.code_native(mod.entry, Tuple{Int64}; dump_module=true, kernel=true)
    end
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child(i) = sink_gcn(i)
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
        @check ".type {{(julia|j)_child_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        @check ".type {{(julia|j)_child_[0-9]+}},@function"
        GCN.code_native(mod.parent2, Tuple{Int}; dump_module=true)
    end
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions

    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child1(i) = sink_gcn(i)
        @noinline child2(i) = sink_gcn(i+1)
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
        @check_dag ".type {{(julia|j)_child1_[0-9]+}},@function"
        @check_dag ".type {{(julia|j)_child2_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        @check_dag ".type {{(julia|j)_child1_[0-9]+}},@function"
        @check_dag ".type {{(julia|j)_child2_[0-9]+}},@function"
        GCN.code_native(mod.parent2, Tuple{Int}; dump_module=true)
    end
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the GCN code shouldn't contain a throw)

    # NOTE: Int32 to test for #49

    mod = @eval module $(gensym())
        function kernel(out)
            wid, lane = fldmod1(unsafe_load(out), Int32(32))
            unsafe_store!(out, wid)
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_kernel_[0-9]+}}:"
        @check_not "jl_throw"
        @check_not "jl_invoke"
        GCN.code_native(mod.kernel, Tuple{Ptr{Int32}})
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
    GCN.code_native(devnull, mod.kernel, Tuple{Float64})
    @test "We did not crash!" != ""
end

# FIXME: _ZNK4llvm14TargetLowering20scalarizeVectorStoreEPNS_11StoreSDNodeERNS_12SelectionDAGE
false && @testset "exception arguments" begin
    mod = @eval module $(gensym())
        function kernel(a)
            unsafe_store!(a, trunc(Int, unsafe_load(a)))
            return
        end
    end

    GCN.code_native(devnull, mod.kernel, Tuple{Ptr{Float64}})
end

# FIXME: in function julia_inner_18528 void (%jl_value_t addrspace(10)*): invalid addrspacecast
false && @testset "GC and TLS lowering" begin
    mod = @eval module $(gensym())
        import ..sink_gcn
        mutable struct PleaseAllocate
            y::Csize_t
        end

        # common pattern in Julia 0.7: outlined throw to avoid a GC frame in the calling code
        @noinline function inner(x)
            sink_gcn(x.y)
            nothing
        end

        function kernel(i)
            inner(PleaseAllocate(Csize_t(42)))
            nothing
        end
    end

    @test @filecheck begin
        @check_not "jl_push_gc_frame"
        @check_not "jl_pop_gc_frame"
        @check_not "jl_get_gc_frame_slot"
        @check_not "jl_new_gc_frame"
        @check "gpu_gc_pool_alloc"
        GCN.code_native(mod.kernel, Tuple{Int})
    end

    # make sure that we can still ellide allocations
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

    @test @filecheck begin
        @check_not "gpu_gc_pool_alloc"
        GCN.code_native(ref_kernel, Tuple{Ptr{Int64}, Int})
    end
end

@testset "float boxes" begin
    mod = @eval module $(gensym())
        function kernel(a,b)
            c = Int32(a)
            # the conversion to Int32 may fail, in which case the input Float32 is boxed in order to
            # pass it to the @nospecialize exception constructor. we should really avoid that (eg.
            # by avoiding @nospecialize, or optimize the unused arguments away), but for now the box
            # should just work.
            unsafe_store!(b, c)
            return
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "jl_box_float32"
        GCN.code_llvm(mod.kernel, Tuple{Float32,Ptr{Float32}})
    end
    GCN.code_native(devnull, mod.kernel, Tuple{Float32,Ptr{Float32}})
end

@testset "stack allocation intrinsic" begin
    mod = @eval module $(gensym())
        import ..GPUCompiler

        function scratch(x)
            p = GPUCompiler.alloca(Float32, Val(8))
            @inbounds unsafe_store!(p, x, 1)
            @inbounds unsafe_store!(p, x, 8)
            return @inbounds unsafe_load(p, 1) + unsafe_load(p, 8)
        end

        # zero-element scratch yields a (null) pointer without emitting an alloca
        empty_scratch() = GPUCompiler.alloca(Float32, Val(0)) === reinterpret(Ptr{Float32}, C_NULL)
    end

    # AMDGPU uses alloca address space 5, so the materialized slot lives in AS 5 and
    # `lower_alloca!` emits an `addrspacecast` back to generic (AS 0).
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check "alloca [8 x i32], align 4, addrspace(5)"
        @check "addrspacecast"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.scratch, Tuple{Float32}; optimize=false, dump_module=true)
    end

    # once optimized the slot is promoted away entirely (result is x + x).
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.scratch, Tuple{Float32})
    end

    # a zero-byte allocation lowers to a null pointer rather than a degenerate alloca.
    @test @filecheck begin
        @check_label "define {{.*}}@{{(julia|j)_empty_scratch_[0-9]+}}"
        @check_not "alloca"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.empty_scratch, Tuple{})
    end
end

end
end # :AMDGPU in LLVM.backends()
