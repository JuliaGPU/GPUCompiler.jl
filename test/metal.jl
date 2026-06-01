@testset "IR" begin

@testset "kernel functions" begin
@testset "byref aggregates" begin
    mod = @eval module $(gensym())
        kernel(x) = return
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check_same cond=typed_ptrs "({{(\\{ i64 \\}|\\[1 x i64\\])}}*"
        @check_same cond=opaque_ptrs "(ptr"
        Metal.code_llvm(mod.kernel, Tuple{Tuple{Int}})
    end

    # for kernels, every pointer argument needs to take an address space
    @test @filecheck begin
        @check_label "define void @_Z6kernel5TupleI5Int64E"
        @check_same cond=typed_ptrs "({{(\\{ i64 \\}|\\[1 x i64\\])}} addrspace(1)*"
        @check_same cond=opaque_ptrs "(ptr addrspace(1)"
        Metal.code_llvm(mod.kernel, Tuple{Tuple{Int}}; kernel=true)
    end
end

@testset "byref primitives" begin
    mod = @eval module $(gensym())
        kernel(x) = return
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check_same "(i64"
        Metal.code_llvm(mod.kernel, Tuple{Int})
    end

    # for kernels, every pointer argument needs to take an address space
    @test @filecheck begin
        @check_label "define void @_Z6kernel5Int64"
        @check_same cond=typed_ptrs "(i64 addrspace(1)*"
        @check_same cond=opaque_ptrs "(ptr addrspace(1)"
        Metal.code_llvm(mod.kernel, Tuple{Int}; kernel=true)
    end
end

@testset "module metadata" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check "air.version"
        @check "air.language_version"
        @check "air.max_device_buffers"
        Metal.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
    end
end

@testset "argument metadata" begin
    mod = @eval module $(gensym())
        kernel(x) = return
    end

    @test @filecheck begin
        @check "air.buffer"
        Metal.code_llvm(mod.kernel, Tuple{Int}; dump_module=true, kernel=true)
    end

    # XXX: perform more exhaustive testing of argument passing metadata here,
    #      or just defer to execution testing in Metal.jl?
end

@testset "input arguments" begin
    mod = @eval module $(gensym())
        function kernel(ptr)
            idx = ccall("extern julia.air.thread_position_in_threadgroup.i32",
                        llvmcall, UInt32, ()) + 1
            unsafe_store!(ptr, 42, idx)
            return
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check_same cond=typed_ptrs "({{.+}} addrspace(1)* %{{.+}})"
        @check_same cond=opaque_ptrs "(ptr addrspace(1) %{{.+}})"
        @check "call i32 @julia.air.thread_position_in_threadgroup.i32"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Int,1}})
    end

    @test @filecheck begin
        @check_label "define void @_Z6kernel7LLVMPtrI5Int64Li1EE"
        @check_same cond=typed_ptrs "({{.+}} addrspace(1)* %{{.+}}, i32 %thread_position_in_threadgroup)"
        @check_same cond=opaque_ptrs "(ptr addrspace(1) %{{.+}}, i32 %thread_position_in_threadgroup)"
        @check_not "call i32 @julia.air.thread_position_in_threadgroup.i32"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Int,1}}; kernel=true)
    end
end

@testset "vector intrinsics" begin
    mod = @eval module $(gensym())
        foo(x, y) = ccall("llvm.smax.v2i64", llvmcall, NTuple{2, VecElement{Int64}},
                          (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}}), x, y)
    end

    @test @filecheck begin
        @check_label "define <2 x i64> @{{(julia|j)_foo_[0-9]+}}"
        @check "air.max.s.v2i64"
        Metal.code_llvm(mod.foo, (NTuple{2, VecElement{Int64}}, NTuple{2, VecElement{Int64}}))
    end
end

@testset "unsupported type detection" begin
    mod = @eval module $(gensym())
        function kernel(ptr)
            buf = reinterpret(Ptr{Float32}, ptr)
            val = unsafe_load(buf)
            dval = Cdouble(val)
            # ccall("extern metal_os_log", llvmcall, Nothing, (Float64,), dval)
            Base.llvmcall(("""
                declare void @llvm.va_start(i8*)
                declare void @llvm.va_end(i8*)
                declare void @air.os_log(i8*, i64)

                define void @metal_os_log(...) {
                    %1 = alloca i8*
                    %2 = bitcast i8** %1 to i8*
                    call void @llvm.va_start(i8* %2)
                    %3 = load i8*, i8** %1
                    call void @air.os_log(i8* %3, i64 8)
                    call void @llvm.va_end(i8* %2)
                    ret void
                }

                define void @entry(double %val) #0 {
                    call void (...) @metal_os_log(double %val)
                    ret void
                }

                attributes #0 = { alwaysinline }""", "entry"),
                Nothing, Tuple{Float64}, dval)
            return
        end
    end

    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check "@metal_os_log"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Float32,1}}; validate=true)
    end

    function kernel2(ptr)
        val = unsafe_load(ptr)
        res = val * val
        unsafe_store!(ptr, res)
        return
    end

    @test_throws_message(InvalidIRError,
                         Metal.code_execution(kernel2,
                                              Tuple{Core.LLVMPtr{Float64,1}})) do msg
        occursin("unsupported use of double value", msg)
    end
end

@testset "constant globals" begin
    mod = @eval module $(gensym())
        const xs = (1.0f0, 2f0)

        function kernel(ptr, i)
            unsafe_store!(ptr, xs[i])

            return
        end
    end

    @test @filecheck begin
        @check "@{{.+}} ={{.*}} addrspace(2) constant [2 x float]"
        @check "define void @_Z6kernel7LLVMPtrI7Float32Li1EE5Int64"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Float32,1}, Int};
                        dump_module=true, kernel=true)
    end
end

# Tuples with a dynamic index are lowered to an addrspace(2) constant plus a
# GEP+load. Without InferAddressSpaces propagating AS 2 through the cast to
# the generic AS introduced during `add_global_address_spaces!`, the load
# would end up in AS 0 and Metal's back-end miscompiles it into zeroes.
@testset "dynamic constant global access" begin
    mod = @eval module $(gensym())
        function kernel(ptr, i)
            t = (1.0f0, 2.0f0, 3.0f0, 4.0f0)
            @inbounds unsafe_store!(ptr, t[i])
            return
        end
    end

    @test @filecheck begin
        @check "@{{.+}} ={{.*}} addrspace(2) constant [4 x float]"
        @check_label "define void @_Z6kernel7LLVMPtrI7Float32Li1EE5Int64"
        # the load must happen in addrspace(2); Metal miscompiles loads of
        # the constant if they occur via an addrspacecast to the generic AS
        @check cond=opaque_ptrs "load float, ptr addrspace(2)"
        @check cond=typed_ptrs "load float, float addrspace(2)*"
        Metal.code_llvm(mod.kernel, Tuple{Core.LLVMPtr{Float32,1}, Int};
                        dump_module=true, kernel=true)
    end
end

end

@testset "argument address-space narrowing" begin
    # pointer type in address space `as`, typed- and opaque-pointer compatible
    asptr(as) = supports_typed_pointers() ? LLVM.PointerType(LLVM.Int8Type(), as) :
                                            LLVM.PointerType(as)

    # build a module with an internal `callee` that loads through a generic (AS 0) pointer
    # parameter, reached from one `caller` per entry in `caller_src_as`, each passing a
    # constant global in that address space cast to generic.
    function narrowing_module(caller_src_as::Vector{Int};
                              callee_linkage=LLVM.API.LLVMInternalLinkage,
                              recursive=false, address_taken=false)
        mod = LLVM.Module("test")
        i8 = LLVM.Int8Type()
        callee_ft = LLVM.FunctionType(i8, LLVM.LLVMType[asptr(0)])
        callee = LLVM.Function(mod, "callee", callee_ft)
        linkage!(callee, callee_linkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(callee, "entry"))
            v = load!(builder, i8, parameters(callee)[1])
            if recursive
                # a (would-be infinite) self-call passing a constant global, only to
                # exercise the recursion path; not meant to run.
                g = GlobalVariable(mod, i8, "gself", caller_src_as[1])
                initializer!(g, ConstantInt(i8, 7)); constant!(g, true)
                call!(builder, callee_ft, callee, [const_addrspacecast(g, asptr(0))])
            end
            ret!(builder, v)
        end
        for (n, as) in enumerate(caller_src_as)
            g = GlobalVariable(mod, i8, "g$n", as)
            initializer!(g, ConstantInt(i8, n)); constant!(g, true)
            caller = LLVM.Function(mod, "caller$n", LLVM.FunctionType(i8, LLVM.LLVMType[]))
            linkage!(caller, LLVM.API.LLVMInternalLinkage)
            @dispose builder=IRBuilder() begin
                position!(builder, BasicBlock(caller, "entry"))
                ret!(builder, call!(builder, callee_ft, callee,
                                    [const_addrspacecast(g, asptr(0))]))
            end
        end
        if address_taken
            # a non-call use of the callee: stash its address in a global
            initializer!(GlobalVariable(mod, value_type(callee), "fp"), callee)
        end
        return mod
    end

    callee_param_as(mod) = addrspace(parameters(function_type(functions(mod)["callee"]))[1])
    function calls_to(mod, fname)
        f = functions(mod)[fname]
        [inst for g in functions(mod) for bb in blocks(g) for inst in instructions(bb)
              if inst isa LLVM.CallInst && called_operand(inst) == f]
    end

    # all callers agree -> the parameter is narrowed; attributes survive; IR stays valid
    Context() do ctx
        mod = narrowing_module([2, 2])
        callee = functions(mod)["callee"]
        push!(parameter_attributes(callee, 1), EnumAttribute("nonnull", 0))
        push!(function_attributes(callee), EnumAttribute("nounwind", 0))

        @test GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 2
        @test all(c -> addrspace(value_type(arguments(c)[1])) == 2, calls_to(mod, "callee"))

        callee = functions(mod)["callee"]
        @test kind(EnumAttribute("nonnull", 0)) in kind.(collect(parameter_attributes(callee, 1)))
        @test kind(EnumAttribute("nounwind", 0)) in kind.(collect(function_attributes(callee)))
        @test (verify(mod); true)
    end

    # callers disagree on the source address space -> left alone
    Context() do ctx
        mod = narrowing_module([2, 1])
        @test !GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 0
    end

    # the callee's address is taken (a non-call use) -> left alone
    Context() do ctx
        mod = narrowing_module([2]; address_taken=true)
        @test !GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 0
    end

    # externally-visible callee -> left alone (its signature may be observed elsewhere)
    Context() do ctx
        mod = narrowing_module([2]; callee_linkage=LLVM.API.LLVMExternalLinkage)
        @test !GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 0
    end

    # a self-recursive callee is narrowed and the self-call rewritten to stay well-typed:
    # every call to it (recursive included) must now pass the constant-space pointer
    Context() do ctx
        mod = narrowing_module([2]; recursive=true)
        @test GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 2
        @test length(calls_to(mod, "callee")) == 2
        @test all(c -> addrspace(value_type(arguments(c)[1])) == 2, calls_to(mod, "callee"))
        @test (verify(mod); true)
    end

    # the source need not be a global: a device pointer (AS 1) threaded through a helper
    # as a generic pointer is narrowed to AS 1 just the same
    Context() do ctx
        mod = LLVM.Module("test")
        i8 = LLVM.Int8Type()
        callee_ft = LLVM.FunctionType(i8, LLVM.LLVMType[asptr(0)])
        callee = LLVM.Function(mod, "callee", callee_ft)
        linkage!(callee, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(callee, "entry"))
            ret!(builder, load!(builder, i8, parameters(callee)[1]))
        end
        caller = LLVM.Function(mod, "caller", LLVM.FunctionType(i8, LLVM.LLVMType[asptr(1)]))
        linkage!(caller, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(caller, "entry"))
            gen = addrspacecast!(builder, parameters(caller)[1], asptr(0))
            ret!(builder, call!(builder, callee_ft, callee, [gen]))
        end

        @test GPUCompiler.propagate_argument_address_spaces!(mod)
        @test callee_param_as(mod) == 1
        @test (verify(mod); true)
    end

    # a two-level delegation chain (caller -> mid -> leaf) needs the fixpoint: one sweep
    # narrows `mid` (its caller passes a constant global), which only then exposes `leaf`,
    # since `mid` now forwards an addrspacecast-from-constant. iterate until both narrow.
    Context() do ctx
        mod = LLVM.Module("test")
        i8 = LLVM.Int8Type()
        ft = LLVM.FunctionType(i8, LLVM.LLVMType[asptr(0)])
        param_as(name) = addrspace(parameters(function_type(functions(mod)[name]))[1])

        # leaf: loads through its generic pointer parameter
        leaf = LLVM.Function(mod, "leaf", ft)
        linkage!(leaf, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(leaf, "entry"))
            ret!(builder, load!(builder, i8, parameters(leaf)[1]))
        end

        # mid: forwards its generic pointer parameter to leaf
        mid = LLVM.Function(mod, "mid", ft)
        linkage!(mid, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(mid, "entry"))
            ret!(builder, call!(builder, ft, leaf, [parameters(mid)[1]]))
        end

        # caller: passes a constant global (AS 2) cast to generic into mid
        g = GlobalVariable(mod, i8, "g", 2)
        initializer!(g, ConstantInt(i8, 1)); constant!(g, true)
        caller = LLVM.Function(mod, "caller", LLVM.FunctionType(i8, LLVM.LLVMType[]))
        linkage!(caller, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(caller, "entry"))
            ret!(builder, call!(builder, ft, mid, [const_addrspacecast(g, asptr(0))]))
        end

        # a single sweep reaches only `mid`; the fixpoint must then narrow `leaf` too
        @test GPUCompiler.propagate_argument_address_spaces_once!(mod)
        @test param_as("mid") == 2
        @test param_as("leaf") == 0

        @test GPUCompiler.propagate_argument_address_spaces!(mod)
        @test param_as("leaf") == 2
        @test (verify(mod); true)
    end
end

end
