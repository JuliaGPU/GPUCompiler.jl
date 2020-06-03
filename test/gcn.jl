@testset "GCN" begin

include("definitions/gcn.jl")

############################################################################################

@testset "IR" begin

@testset "kernel calling convention" begin
    kernel() = return

    ir = sprint(io->gcn_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("amdgpu_kernel", ir)

    ir = sprint(io->gcn_code_llvm(io, kernel, Tuple{};
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

    asm = sprint(io->gcn_code_native(io, kernel, Tuple{}))
    @test occursin("s_trap 2", asm)
    @test_broken occursin("s_cbranch_execz", asm)
    if Base.libllvm_version < v"9"
        @test_broken occursin("v_readfirstlane", asm)
    end
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    @noinline child(i) = sink(i)
    function parent(i)
        child(i)
        return
    end

    asm = sprint(io->gcn_code_native(io, parent, Tuple{Int64}))
    @test occursin(r"s_add_u32.*julia_child_.*@rel32@lo\+4", asm)
    @test occursin(r"s_addc_u32.*julia_child_.*@rel32@hi\+4", asm)
end

@testset "kernel functions" begin
    @noinline nonentry(i) = sink(i)
    function entry(i)
        nonentry(i)
        return
    end

    asm = sprint(io->gcn_code_native(io, entry, Tuple{Int64}; kernel=true))
    @test occursin(r"\.amdgpu_hsa_kernel .*julia_entry", asm)
    @test !occursin(r"\.amdgpu_hsa_kernel .*julia_nonentry", asm)
    @test occursin(r"\.type.*julia_nonentry_\d*,@function", asm)
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    @noinline child(i) = sink(i)
    function parent1(i)
        child(i)
        return
    end

    asm = sprint(io->gcn_code_native(io, parent1, Tuple{Int}))
    @test occursin(r"\.type.*julia__\d*_child_\d*,@function", asm)

    function parent2(i)
        child(i+1)
        return
    end

    asm = sprint(io->gcn_code_native(io, parent2, Tuple{Int}))
    @test occursin(r"\.type.*julia__\d*_child_\d*,@function", asm)
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions
    @noinline child1(i) = sink(i)
    @noinline child2(i) = sink(i+1)
    function parent1(i)
        child1(i) + child2(i)
        return
    end
    gcn_code_native(devnull, parent1, Tuple{Int})

    function parent2(i)
        child1(i+1) + child2(i+1)
        return
    end
    gcn_code_native(devnull, parent2, Tuple{Int})
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

    asm = sprint(io->gcn_code_native(io, kernel, Tuple{Ptr{Int32}}))
    @test !occursin("jl_throw", asm)
    @test !occursin("jl_invoke", asm)   # forced recompilation should still not invoke
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    function kernel(x)
        unsafe_trunc(Int, x)
        return
    end
    gcn_code_native(devnull, kernel, Tuple{Float64})
end

@test_broken "exception arguments"
#= FIXME: _ZNK4llvm14TargetLowering20scalarizeVectorStoreEPNS_11StoreSDNodeERNS_12SelectionDAGE
@testset "exception arguments" begin
    function kernel(a)
        unsafe_store!(a, trunc(Int, unsafe_load(a)))
        return
    end

    gcn_code_native(devnull, kernel, Tuple{Ptr{Float64}})
end
=#

@test_broken "GC and TLS lowering"
#= FIXME: in function julia_inner_18528 void (%jl_value_t addrspace(10)*): invalid addrspacecast
@testset "GC and TLS lowering" begin
    @eval mutable struct PleaseAllocate
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

    asm = sprint(io->gcn_code_native(io, kernel, Tuple{Int}))
    @test occursin("gpu_gc_pool_alloc", asm)

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

    asm = sprint(io->gcn_code_native(io, ref_kernel, Tuple{Ptr{Int64}, Int}))


    if VERSION < v"1.2.0-DEV.375"
        @test_broken !occursin("gpu_gc_pool_alloc", asm)
    else
        @test !occursin("gpu_gc_pool_alloc", asm)
    end
end
=#

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

    ir = sprint(io->gcn_code_llvm(io, kernel, Tuple{Float32,Ptr{Float32}}))
    @test occursin("jl_box_float32", ir)
    gcn_code_native(devnull, kernel, Tuple{Float32,Ptr{Float32}})
end

end

############################################################################################

end
