@testset "IR" begin

@testset "exceptions" begin
    foobar() = throw(DivideError())
    ir = sprint(io->PTX.code_llvm(io, foobar, Tuple{}))

    # plain exceptions should get lowered to a call to the GPU run-time
    @test occursin("gpu_report_exception", ir)
    # not a jl_throw referencing a jl_value_t representing the exception
    @test !occursin("jl_throw", ir)
end

@testset "kernel functions" begin
@testset "kernel argument attributes" begin
    mod = @eval module $(gensym())
        kernel(x) = return

        struct Aggregate
            x::Int
        end
    end

    ir = sprint(io->PTX.code_llvm(io, mod.kernel, Tuple{mod.Aggregate}))
    @test occursin(r"@(julia|j)_kernel\w*\(({ i64 }|\[1 x i64\])\* ", ir) ||
          occursin(r"@(julia|j)_kernel\w*\(ptr ", ir)

    ir = sprint(io->PTX.code_llvm(io, mod.kernel, Tuple{mod.Aggregate}; kernel=true))
    @test occursin(r"@_Z6kernel9Aggregate\(.*({ i64 }|\[1 x i64\]) ", ir)
end

@testset "property_annotations" begin
    kernel() = return

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("nvvm.annotations", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true))
    @test occursin("nvvm.annotations", ir)
    @test !occursin("maxntid", ir)
    @test !occursin("reqntid", ir)
    @test !occursin("minctasm", ir)
    @test !occursin("maxnreg", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, maxthreads=42))
    @test occursin("maxntidx\", i32 42", ir)
    @test occursin("maxntidy\", i32 1", ir)
    @test occursin("maxntidz\", i32 1", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, minthreads=42))
    @test occursin("reqntidx\", i32 42", ir)
    @test occursin("reqntidy\", i32 1", ir)
    @test occursin("reqntidz\", i32 1", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, blocks_per_sm=42))
    @test occursin("minctasm\", i32 42", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true, maxregs=42))
    @test occursin("maxnreg\", i32 42", ir)
end

LLVM.version() >= v"8" && @testset "calling convention" begin
    kernel() = return

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("ptx_kernel", ir)

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{};
                                  dump_module=true, kernel=true))
    @test occursin("ptx_kernel", ir)
end

@testset "kernel state" begin
    # state should be passed by value to kernel functions

    mod = @eval module $(gensym())
        export kernel
        kernel() = return
    end

    ir = sprint(io->PTX.code_llvm(io, mod.kernel, Tuple{}))
    @test occursin(r"@(julia|j)_kernel\w*\(\)", ir)

    ir = sprint(io->PTX.code_llvm(io, mod.kernel, Tuple{}; kernel=true))
    @test occursin("@_Z6kernel([1 x i64] %state)", ir)

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

    ir = sprint(io->PTX.code_llvm(io, mod.kernel, Tuple{Ptr{Int64}};
                                  kernel=true, dump_module=true))

    # kernel should take state argument before all else
    @test occursin(r"@_Z6kernelP5Int64\(\[1 x i64\] %state", ir)

    # child1 doesn't use the state
    @test occursin(r"@(julia|j)_child1\w*\((i64|i8\*|ptr)", ir)

    # child2 does
    @test occursin(r"@(julia|j)_child2\w*\(\[1 x i64\] %state", ir)

    # can't have the unlowered intrinsic
    @test !occursin("julia.gpu.state_getter", ir)
end
end

end

############################################################################################
@static if !Sys.isapple()
@testset "assembly" begin

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)

    mod = @eval module $(gensym())
        import ..sink
        export child, parent

        @noinline child(i) = sink(i)
        function parent(i)
            child(i)
            return
        end
    end

    asm = sprint(io->PTX.code_native(io, mod.parent, Tuple{Int64}))
    @test occursin(r"call.uni\s+(julia|j)_child_"m, asm)
end

@testset "kernel functions" begin
    mod = @eval module $(gensym())
        import ..sink
        export nonentry, entry

        @noinline nonentry(i) = sink(i)
        function entry(i)
            nonentry(i)
            return
        end
    end

    asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64};
                                     kernel=true, dump_module=true))
    @test occursin(".visible .entry _Z5entry5Int64", asm)
    @test !occursin(r"\.visible \.func (julia|j)_nonentry", asm)
    @test occursin(r"\.func (julia|j)_nonentry", asm)

@testset "property_annotations" begin
    asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64}; kernel=true))
    @test !occursin("maxntid", asm)

    asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64};
                                         kernel=true, maxthreads=42))
    @test occursin(".maxntid 42, 1, 1", asm)

    asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64};
                                         kernel=true, minthreads=42))
    @test occursin(".reqntid 42, 1, 1", asm)

    asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64};
                                         kernel=true, blocks_per_sm=42))
    @test occursin(".minnctapersm 42", asm)

    if LLVM.version() >= v"4.0"
        asm = sprint(io->PTX.code_native(io, mod.entry, Tuple{Int64};
                                             kernel=true, maxregs=42))
        @test occursin(".maxnreg 42", asm)
    end
end
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    mod = @eval module $(gensym())
        import ..sink
        export child, parent1, parent2

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

    asm = sprint(io->PTX.code_native(io, mod.parent1, Tuple{Int}))
    @test occursin(r"\.func (julia|j)_child_", asm)

    asm = sprint(io->PTX.code_native(io, mod.parent2, Tuple{Int}))
    @test occursin(r"\.func (julia|j)_child_", asm)
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions

    mod = @eval module $(gensym())
        import ..sink
        export parent1, parent2, child1, child2

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

    asm = sprint(io->PTX.code_native(io, mod.parent1, Tuple{Int}))
    @test occursin(r"\.func (julia|j)_child1_", asm)
    @test occursin(r"\.func (julia|j)_child2_", asm)

    asm = sprint(io->PTX.code_native(io, mod.parent2, Tuple{Int}))
    @test occursin(r"\.func (julia|j)_child1_", asm)
    @test occursin(r"\.func (julia|j)_child2_", asm)
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

    asm = sprint(io->PTX.code_native(io, kernel, Tuple{Ptr{Int32}}))
    @test !occursin("jl_throw", asm)
    @test !occursin("jl_invoke", asm)   # forced recompilation should still not invoke
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    function kernel(x)
        unsafe_trunc(Int, x)
        return
    end
    PTX.code_native(devnull, kernel, Tuple{Float64})
    @test "We did not crash!" != ""
end

@testset "exception arguments" begin
    function kernel(a)
        unsafe_store!(a, trunc(Int, unsafe_load(a)))
        return
    end
    PTX.code_native(devnull, kernel, Tuple{Ptr{Float64}})
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

    asm = sprint(io->PTX.code_native(io, mod.kernel, Tuple{Int}))
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

    asm = sprint(io->PTX.code_native(io, ref_kernel, Tuple{Ptr{Int64}, Int}))


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

    ir = sprint(io->PTX.code_llvm(io, kernel, Tuple{Float32,Ptr{Float32}}))
    @test occursin("jl_box_float32", ir)
    PTX.code_native(devnull, kernel, Tuple{Float32,Ptr{Float32}})
end

end
