@testset "IR" begin

@testset "exceptions" begin
    mod = @eval module $(gensym())
        foobar() = throw(DivideError())
    end
    @test @filecheck begin
        @check_label "define void @{{(julia|j)_foobar_[0-9]+}}"
        # plain exceptions should get lowered to a call to the GPU run-time
        # not a jl_throw referencing a jl_value_t representing the exception
        @check_not "jl_throw"
        @check "gpu_report_exception"

        PTX.code_llvm(mod.foobar, Tuple{}; dump_module=true)
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

@testset "property_annotations" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "nvvm.annotations"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true)
    end

    @test @filecheck begin
        @check_not "maxntid"
        @check_not "reqntid"
        @check_not "minctasm"
        @check_not "maxnreg"
        @check "nvvm.annotations"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
    end

    @test @filecheck begin
        @check "maxntidx\", i32 42"
        @check "maxntidy\", i32 1"
        @check "maxntidz\", i32 1"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true, maxthreads=42)
    end

    @test @filecheck begin
        @check "reqntidx\", i32 42"
        @check "reqntidy\", i32 1"
        @check "reqntidz\", i32 1"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true, minthreads=42)
    end

    @test @filecheck begin
        @check "minctasm\", i32 42"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true, blocks_per_sm=42)
    end

    @test @filecheck begin
        @check "maxnreg\", i32 42"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true, maxregs=42)
    end
end

LLVM.version() >= v"8" && @testset "calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "ptx_kernel"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true)
    end

    @test @filecheck begin
        @check "ptx_kernel"
        PTX.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
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

    @test @filecheck begin
        @check_label ".visible .func {{(julia|j)_parent[0-9_]*}}"
        @check "call.uni"
        @check_next "{{(julia|j)_child_}}"
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
            c = Int32(a)
            # the conversion to Int32 may fail, in which case the input Float32 is boxed in
            # order to pass it to the @nospecialize exception constructor. we should really
            # avoid that (eg. by avoiding @nospecialize, or optimize the unused arguments
            # away), but for now the box should just work.
            unsafe_store!(b, c)
            return
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "jl_box_float32"
        PTX.code_llvm(mod.kernel, Tuple{Float32,Ptr{Float32}})
    end
    PTX.code_native(devnull, mod.kernel, Tuple{Float32,Ptr{Float32}})
end

end
end # NVPTX in LLVM.backends()
