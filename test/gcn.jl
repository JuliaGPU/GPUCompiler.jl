if :AMDGPU in LLVM.backends()

# XXX: generic `sink` generates an instruction selection error
sink_gcn(i) = sink(i, Val(5))

@testset "IR" begin

@testset "kernel calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        check"CHECK-NOT: amdgpu_kernel"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true)
    end

    @test @filecheck begin
        check"CHECK: amdgpu_kernel"
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
        check"CHECK-NOT: {{julia_throw_boundserror_[0-9]+}}"
        check"CHECK: @gpu_report_exception"
        check"CHECK: @gpu_signal_exception"
        GCN.code_llvm(mod.kernel, Tuple{})
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
        check"CHECK-LABEL: {{(julia|j)_kernel_[0-9]+}}:"
        check"CHECK: s_cbranch_exec"
        check"CHECK: s_trap 2"
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
        check"CHECK-LABEL: {{(julia|j)_parent_[0-9]+}}:"
        check"CHECK: s_add_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
        check"CHECK: s_addc_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
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
        check"CHECK-NOT: .amdhsa_kernel {{(julia|j)_nonentry_[0-9]+}}"
        check"CHECK: .type {{(julia|j)_nonentry_[0-9]+}},@function"
        check"CHECK: .amdhsa_kernel _Z5entry5Int64"
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
        check"CHECK: .type {{(julia|j)_child_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        check"CHECK: .type {{(julia|j)_child_[0-9]+}},@function"
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
        check"CHECK-DAG: .type {{(julia|j)_child1_[0-9]+}},@function"
        check"CHECK-DAG: .type {{(julia|j)_child2_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        check"CHECK-DAG: .type {{(julia|j)_child1_[0-9]+}},@function"
        check"CHECK-DAG: .type {{(julia|j)_child2_[0-9]+}},@function"
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
        check"CHECK-LABEL: {{(julia|j)_kernel_[0-9]+}}:"
        check"CHECK-NOT: jl_throw"
        check"CHECK-NOT: jl_invoke"
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
        check"CHECK-NOT: jl_push_gc_frame"
        check"CHECK-NOT: jl_pop_gc_frame"
        check"CHECK-NOT: jl_get_gc_frame_slot"
        check"CHECK-NOT: jl_new_gc_frame"
        check"CHECK: gpu_gc_pool_alloc"
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
        check"CHECK-NOT: gpu_gc_pool_alloc"
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
        check"CHECK-LABEL: define void @{{(julia|j)_kernel_[0-9]+}}"
        check"CHECK: jl_box_float32"
        GCN.code_llvm(mod.kernel, Tuple{Float32,Ptr{Float32}})
    end
    GCN.code_native(devnull, mod.kernel, Tuple{Float32,Ptr{Float32}})
end

end
end # :AMDGPU in LLVM.backends()
