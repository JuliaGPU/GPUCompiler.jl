if :AMDGPU in LLVM.backends()
@testset "IR" begin

@testset "kernel calling convention" begin
    kernel() = return

    ir = sprint(io->GCN.code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("amdgpu_kernel", ir)

    ir = sprint(io->GCN.code_llvm(io, kernel, Tuple{};
                                         dump_module=true, kernel=true))
    @test occursin("amdgpu_kernel", ir)
end

end

############################################################################################
@testset "assembly" begin

@testset "skip scalar trap" begin
    workitem_idx_x() = ccall("llvm.amdgcn.workitem.id.x", llvmcall, Int32, ())
    trap() = ccall("llvm.trap", llvmcall, Nothing, ())
    function kernel()
        if workitem_idx_x() > 1
            trap()
        end
        return
    end

    asm = sprint(io->GCN.code_native(io, kernel, Tuple{}))
    @test occursin("s_trap 2", asm)
    @test_skip occursin("s_cbranch_execz", asm)
    if Base.libllvm_version < v"9"
        @test_broken occursin("v_readfirstlane", asm)
    end
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    @noinline child(i) = sink_gcn(i)
    function parent(i)
        child(i)
        return
    end

    asm = sprint(io->GCN.code_native(io, parent, Tuple{Int64}; dump_module=true))
    @test occursin(r"s_add_u32.*(julia|j)_child_.*@rel32@", asm)
    @test occursin(r"s_addc_u32.*(julia|j)_child_.*@rel32@", asm)
end

@testset "kernel functions" begin
    @noinline nonentry(i) = sink_gcn(i)
    function entry(i)
        nonentry(i)
        return
    end

    asm = sprint(io->GCN.code_native(io, entry, Tuple{Int64}; dump_module=true, kernel=true))
    @test occursin(r"\.amdhsa_kernel \w*entry", asm)
    @test !occursin(r"\.amdhsa_kernel \w*nonentry", asm)
    @test occursin(r"\.type.*\w*nonentry\w*,@function", asm)
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    mod = @eval module $(gensym())
        export child, parent1, parent2

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

    asm = sprint(io->GCN.code_native(io, mod.parent1, Tuple{Int}; dump_module=true))
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child_\d*,@function", asm)

    asm = sprint(io->GCN.code_native(io, mod.parent2, Tuple{Int}; dump_module=true))
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child_\d*,@function", asm)
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions

    mod = @eval module $(gensym())
        export parent1, parent2, child1, child2

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

    asm = sprint(io->GCN.code_native(io, mod.parent1, Tuple{Int}; dump_module=true))
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child1_\d*,@function", asm)
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child2_\d*,@function", asm)

    asm = sprint(io->GCN.code_native(io, mod.parent2, Tuple{Int}; dump_module=true))
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child1_\d*,@function", asm)
    @test occursin(r"\.type.*(julia|j)_[[:alnum:]_.]*child2_\d*,@function", asm)
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the GCN code shouldn't contain a throw)

    # NOTE: Int32 to test for #49

    function kernel(out)
        wid, lane = fldmod1(unsafe_load(out), Int32(32))
        unsafe_store!(out, wid)
        return
    end

    asm = sprint(io->GCN.code_native(io, kernel, Tuple{Ptr{Int32}}))
    @test !occursin("jl_throw", asm)
    @test !occursin("jl_invoke", asm)   # forced recompilation should still not invoke
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    function kernel(x)
        unsafe_trunc(Int, x)
        return
    end
    GCN.code_native(devnull, kernel, Tuple{Float64})
    @test "We did not crash!" != ""
end

# FIXME: _ZNK4llvm14TargetLowering20scalarizeVectorStoreEPNS_11StoreSDNodeERNS_12SelectionDAGE
false && @testset "exception arguments" begin
    function kernel(a)
        unsafe_store!(a, trunc(Int, unsafe_load(a)))
        return
    end

    GCN.code_native(devnull, kernel, Tuple{Ptr{Float64}})
end

# FIXME: in function julia_inner_18528 void (%jl_value_t addrspace(10)*): invalid addrspacecast
false && @testset "GC and TLS lowering" begin
    mod = @eval module $(gensym())
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

    asm = sprint(io->GCN.code_native(io, mod.kernel, Tuple{Int}))
    @test occursin("gpu_gc_pool_alloc", asm)
    @test !occursin("julia.push_gc_frame", asm)
    @test !occursin("julia.pop_gc_frame", asm)
    @test !occursin("julia.get_gc_frame_slot", asm)
    @test !occursin("julia.new_gc_frame", asm)

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

    asm = sprint(io->GCN.code_native(io, ref_kernel, Tuple{Ptr{Int64}, Int}))


    @test !occursin("gpu_gc_pool_alloc", asm)
end

@testset "float boxes" begin
    function kernel(a,b)
        c = Int32(a)
        # the conversion to Int32 may fail, in which case the input Float32 is boxed in order to
        # pass it to the @nospecialize exception constructor. we should really avoid that (eg.
        # by avoiding @nospecialize, or optimize the unused arguments away), but for now the box
        # should just work.
        unsafe_store!(b, c)
        return
    end

    ir = sprint(io->GCN.code_llvm(io, kernel, Tuple{Float32,Ptr{Float32}}))
    @test occursin("jl_box_float32", ir)
    GCN.code_native(devnull, kernel, Tuple{Float32,Ptr{Float32}})
end

end
end # :AMDGPU in LLVM.backends()
