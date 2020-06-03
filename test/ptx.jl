@testset "PTX" begin

include("definitions/ptx.jl")

############################################################################################

@testset "IR" begin

@testset "exceptions" begin
    foobar() = throw(DivideError())
    ir = sprint(io->ptx_code_llvm(io, foobar, Tuple{}))

    # plain exceptions should get lowered to a call to the GPU run-time
    @test occursin("gpu_report_exception", ir)
    # not a jl_throw referencing a jl_value_t representing the exception
    @test !occursin("jl_throw", ir)
end

@testset "kernel functions" begin
@testset "kernel argument attributes" begin
    kernel(x) = return

    @eval struct Aggregate
        x::Int
    end

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{Aggregate}))
    if VERSION < v"1.5.0-DEV.802"
        @test occursin(r"@.*julia_kernel.+\(({ i64 }|\[1 x i64\]) addrspace\(\d+\)?\*", ir)
    else
        @test occursin(r"@.*julia_kernel.+\(({ i64 }|\[1 x i64\])\*", ir)
    end

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{Aggregate}; kernel=true))
    if VERSION < v"1.5.0-DEV.802"
        @test occursin(r"@.*julia_kernel.+\(({ i64 }|\[1 x i64\]) addrspace\(\d+\)?\*.+byval", ir)
    else
        @test occursin(r"@.*julia_kernel.+\(({ i64 }|\[1 x i64\])\*.+byval", ir)
    end
end

@testset "property_annotations" begin
    kernel() = return

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("nvvm.annotations", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                         dump_module=true, kernel=true))
    @test occursin("nvvm.annotations", ir)
    @test !occursin("maxntid", ir)
    @test !occursin("reqntid", ir)
    @test !occursin("minctasm", ir)
    @test !occursin("maxnreg", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, maxthreads=42))
    @test occursin("maxntidx\", i32 42", ir)
    @test occursin("maxntidy\", i32 1", ir)
    @test occursin("maxntidz\", i32 1", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, minthreads=42))
    @test occursin("reqntidx\", i32 42", ir)
    @test occursin("reqntidy\", i32 1", ir)
    @test occursin("reqntidz\", i32 1", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, blocks_per_sm=42))
    @test occursin("minctasm\", i32 42", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, maxregs=42))
    @test occursin("maxnreg\", i32 42", ir)
end

LLVM.version() >= v"8" && @testset "calling convention" begin
    kernel() = return

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("ptx_kernel", ir)

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true))
    @test occursin("ptx_kernel", ir)
end
end

end

############################################################################################

@testset "assembly" begin

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    @noinline child(i) = sink(i)
    function parent(i)
        child(i)
        return
    end

    asm = sprint(io->ptx_code_native(io, parent, Tuple{Int64}))
    @test occursin(r"call.uni\s+julia_.*child_"m, asm)
end

@testset "kernel functions" begin
    @noinline nonentry(i) = sink(i)
    function entry(i)
        nonentry(i)
        return
    end

    asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64}; kernel=true))
    @test occursin(r"\.visible \.entry .*julia_entry", asm)
    @test !occursin(r"\.visible \.func .*julia_nonentry", asm)
    @test occursin(r"\.func .*julia_nonentry", asm)

@testset "property_annotations" begin
    asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64}; kernel=true))
    @test !occursin("maxntid", asm)

    asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64};
                                         kernel=true, maxthreads=42))
    @test occursin(".maxntid 42, 1, 1", asm)

    asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64};
                                         kernel=true, minthreads=42))
    @test occursin(".reqntid 42, 1, 1", asm)

    asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64};
                                         kernel=true, blocks_per_sm=42))
    @test occursin(".minnctapersm 42", asm)

    if LLVM.version() >= v"4.0"
        asm = sprint(io->ptx_code_native(io, entry, Tuple{Int64};
                                             kernel=true, maxregs=42))
        @test occursin(".maxnreg 42", asm)
    end
end
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    @noinline child(i) = sink(i)
    function parent1(i)
        child(i)
        return
    end

    asm = sprint(io->ptx_code_native(io, parent1, Tuple{Int}))
    @test occursin(r".func julia_.*child_", asm)

    function parent2(i)
        child(i+1)
        return
    end

    asm = sprint(io->ptx_code_native(io, parent2, Tuple{Int}))
    @test occursin(r".func julia_.*child_", asm)
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
    ptx_code_native(devnull, parent1, Tuple{Int})

    function parent2(i)
        child1(i+1) + child2(i+1)
        return
    end
    ptx_code_native(devnull, parent2, Tuple{Int})
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the PTX code shouldn't contain a throw)

    # NOTE: Int32 to test for #49

    function kernel(out)
        wid, lane = fldmod1(unsafe_load(out), Int32(32))
        unsafe_store!(out, wid)
        return
    end

    asm = sprint(io->ptx_code_native(io, kernel, Tuple{Ptr{Int32}}))
    @test !occursin("jl_throw", asm)
    @test !occursin("jl_invoke", asm)   # forced recompilation should still not invoke
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    function kernel(x)
        unsafe_trunc(Int, x)
        return
    end
    ptx_code_native(devnull, kernel, Tuple{Float64})
end

@testset "exception arguments" begin
    function kernel(a)
        unsafe_store!(a, trunc(Int, unsafe_load(a)))
        return
    end

    ptx_code_native(devnull, kernel, Tuple{Ptr{Float64}})
end

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

    asm = sprint(io->ptx_code_native(io, kernel, Tuple{Int}))
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

    asm = sprint(io->ptx_code_native(io, ref_kernel, Tuple{Ptr{Int64}, Int}))


    if VERSION < v"1.2.0-DEV.375"
        @test_broken !occursin("gpu_gc_pool_alloc", asm)
    else
        @test !occursin("gpu_gc_pool_alloc", asm)
    end
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

    ir = sprint(io->ptx_code_llvm(io, kernel, Tuple{Float32,Ptr{Float32}}))
    @test occursin("jl_box_float32", ir)
    ptx_code_native(devnull, kernel, Tuple{Float32,Ptr{Float32}})
end

end


############################################################################################

end
