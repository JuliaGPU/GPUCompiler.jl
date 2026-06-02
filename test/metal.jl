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

    # typed-pointer form: a `Ptr` argument crosses the boundary as an integer, so the callee
    # takes an integer it `inttoptr`s and the callers pass `ptrtoint(addrspacecast(<global> ->
    # generic))`. the integer parameter is narrowed to a constant-space pointer just the same,
    # and the call sites pass the bare global. (regression: the pass used to skip integer
    # parameters, leaving a `ptrtoint(addrspacecast(...))` constant that the LLVM-16 Metal
    # bitcode downgrade miscompiles -- device exceptions crashed on Julia 1.11.)
    Context() do ctx
        i8 = LLVM.Int8Type()
        i64 = LLVM.Int64Type()
        mod = LLVM.Module("test")
        callee_ft = LLVM.FunctionType(i8, LLVM.LLVMType[i64])
        callee = LLVM.Function(mod, "callee", callee_ft)
        linkage!(callee, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(callee, "entry"))
            p = inttoptr!(builder, parameters(callee)[1], asptr(0))
            ret!(builder, load!(builder, i8, p))
        end
        for n in 1:2
            g = GlobalVariable(mod, i8, "g$n", 2)
            initializer!(g, ConstantInt(i8, n)); constant!(g, true)
            caller = LLVM.Function(mod, "caller$n", LLVM.FunctionType(i8, LLVM.LLVMType[]))
            linkage!(caller, LLVM.API.LLVMInternalLinkage)
            @dispose builder=IRBuilder() begin
                position!(builder, BasicBlock(caller, "entry"))
                arg = const_ptrtoint(const_addrspacecast(g, asptr(0)), i64)
                ret!(builder, call!(builder, callee_ft, callee, [arg]))
            end
        end

        @test GPUCompiler.propagate_argument_address_spaces!(mod)
        param = parameters(function_type(functions(mod)["callee"]))[1]
        @test param isa LLVM.PointerType && addrspace(param) == 2
        @test all(c -> value_type(arguments(c)[1]) isa LLVM.PointerType &&
                       addrspace(value_type(arguments(c)[1])) == 2, calls_to(mod, "callee"))
        @test (verify(mod); true)
    end

    # an integer parameter used as more than a pointer image (here also in arithmetic) is
    # left alone: narrowing it would lose the integer's other uses
    Context() do ctx
        i8 = LLVM.Int8Type()
        i64 = LLVM.Int64Type()
        mod = LLVM.Module("test")
        callee_ft = LLVM.FunctionType(i8, LLVM.LLVMType[i64])
        callee = LLVM.Function(mod, "callee", callee_ft)
        linkage!(callee, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(callee, "entry"))
            p = inttoptr!(builder, parameters(callee)[1], asptr(0))
            v = load!(builder, i8, p)
            # a second, non-`inttoptr` use of the integer parameter
            extra = trunc!(builder, add!(builder, parameters(callee)[1], parameters(callee)[1]), i8)
            ret!(builder, add!(builder, v, extra))
        end
        g = GlobalVariable(mod, i8, "g", 2)
        initializer!(g, ConstantInt(i8, 1)); constant!(g, true)
        caller = LLVM.Function(mod, "caller", LLVM.FunctionType(i8, LLVM.LLVMType[]))
        linkage!(caller, LLVM.API.LLVMInternalLinkage)
        @dispose builder=IRBuilder() begin
            position!(builder, BasicBlock(caller, "entry"))
            arg = const_ptrtoint(const_addrspacecast(g, asptr(0)), i64)
            ret!(builder, call!(builder, callee_ft, callee, [arg]))
        end

        @test !GPUCompiler.propagate_argument_address_spaces!(mod)
        @test parameters(function_type(functions(mod)["callee"]))[1] == i64
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

@testset "vector min/max scalarization" begin
    # AIR has no vector floating-point min/max intrinsic, only scalar air.fmin/air.fmax. Julia's
    # NaN-propagating min/max lower to llvm.minimum/llvm.maximum (and llvm.minnum/llvm.maxnum),
    # which LLVM's vectorizers can widen to vector form. Lowering a vector intrinsic directly
    # would emit a nonexistent air.fmin.v4f32-style call (minnum/maxnum) or hit an unsupported-
    # type error (minimum/maximum), so we scalarize into element-wise scalar intrinsic calls.
    is_vec_minmax(i) = i isa LLVM.CallBase && value_type(i) isa LLVM.VectorType &&
        called_operand(i) isa LLVM.Function &&
        LLVM.isintrinsic(called_operand(i)) &&
        LLVM.Intrinsic(called_operand(i)) in
            LLVM.Intrinsic.(["llvm.minnum", "llvm.maxnum", "llvm.minimum", "llvm.maximum"])

    Context() do ctx
        ir = """
        declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
        declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
        define <4 x float> @f(<4 x float> %a, <4 x float> %b, <2 x float> %c, <2 x float> %d) {
        entry:
          %mn = call <4 x float> @llvm.minimum.v4f32(<4 x float> %a, <4 x float> %b)
          %mx = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %c, <2 x float> %d)
          %e0 = extractelement <2 x float> %mx, i32 0
          %r = insertelement <4 x float> %mn, float %e0, i32 0
          ret <4 x float> %r
        }
        """
        mod = parse(LLVM.Module, ir)
        f = functions(mod)["f"]
        insts() = [i for bb in blocks(f) for i in instructions(bb)]

        # precondition: a vector minimum (width 4) and a vector maxnum (width 2)
        @test count(is_vec_minmax, insts()) == 2

        @test GPUCompiler.scalarize_vector_minmax!(f)

        # no vector min/max calls survive; each is replaced by per-lane scalar intrinsic calls
        @test count(is_vec_minmax, insts()) == 0
        scalar_calls = filter(insts()) do i
            i isa LLVM.CallBase && value_type(i) == LLVM.FloatType() &&
                called_operand(i) isa LLVM.Function &&
                LLVM.isintrinsic(called_operand(i))
        end
        @test length(scalar_calls) == 6   # 4 from the v4f32 + 2 from the v2f32
        @test Set(LLVM.name(called_operand(c)) for c in scalar_calls) ==
            Set(["llvm.minimum.f32", "llvm.maxnum.f32"])
        @test (verify(mod); true)
    end

    # a scalar min/max is already lowerable and must be left untouched
    Context() do ctx
        ir = """
        declare float @llvm.minimum.f32(float, float)
        define float @g(float %a, float %b) {
          %r = call float @llvm.minimum.f32(float %a, float %b)
          ret float %r
        }
        """
        mod = parse(LLVM.Module, ir)
        @test !GPUCompiler.scalarize_vector_minmax!(functions(mod)["g"])
        @test (verify(mod); true)
    end
end

@testset "math intrinsic lowering" begin
    # Front-ends emit plain LLVM math intrinsics; the back-end lowers them to AIR device
    # functions. Precise `air.<op>` by default; the relaxed, f32-only `air.fast_<op>` when the
    # call is `afn`-flagged (set per-op by @fastmath or module-wide by apply_fastmath!). This
    # lets Metal.jl drop its hand-written air.* overrides for these ops. `round` is included
    # via llvm.rint (Julia lowers round-to-even to it).
    function called_names(f)
        names = String[]
        for bb in blocks(f), i in instructions(bb)
            i isa LLVM.CallBase || continue
            c = called_operand(i)
            c isa LLVM.Function && push!(names, LLVM.name(c))
        end
        names
    end

    Context() do ctx
        ir = """
        declare float @llvm.sqrt.f32(float)
        declare half  @llvm.sqrt.f16(half)
        declare float @llvm.floor.f32(float)
        declare float @llvm.ceil.f32(float)
        declare float @llvm.trunc.f32(float)
        declare float @llvm.rint.f32(float)
        declare float @llvm.fma.f32(float, float, float)
        declare half  @llvm.fma.f16(half, half, half)
        declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
        define void @f(float %x, half %h, <4 x float> %v) {
          %a = call float @llvm.sqrt.f32(float %x)
          %b = call afn float @llvm.sqrt.f32(float %x)
          %c = call afn half @llvm.sqrt.f16(half %h)
          %d = call float @llvm.floor.f32(float %x)
          %d2 = call afn float @llvm.floor.f32(float %x)
          %e = call float @llvm.ceil.f32(float %x)
          %g = call float @llvm.trunc.f32(float %x)
          %i = call float @llvm.rint.f32(float %x)
          %j = call float @llvm.fma.f32(float %x, float %x, float %x)
          %l = call half @llvm.fma.f16(half %h, half %h, half %h)
          %k = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %v)
          ret void
        }
        """
        mod = parse(LLVM.Module, ir)
        f = functions(mod)["f"]
        @test GPUCompiler.lower_math_intrinsics!(f)
        names = called_names(f)

        # precise by default; relaxed only for the f32 afn-flagged sqrt
        @test "air.sqrt.f32" in names
        @test "air.fast_sqrt.f32" in names
        # f16 has no fast variant, so afn still maps to the precise op
        @test "air.sqrt.f16" in names
        @test !("air.fast_sqrt.f16" in names)
        # rounding ops have fast f32 variants too (selected for the afn floor)
        @test "air.floor.f32" in names
        @test "air.fast_floor.f32" in names
        @test "air.ceil.f32" in names
        @test "air.trunc.f32" in names
        @test "air.rint.f32" in names      # Julia's round arrives as llvm.rint
        # fma has both f16 and f32 precise forms, and no fast variant
        @test "air.fma.f32" in names
        @test "air.fma.f16" in names
        @test !("air.fast_fma.f32" in names)
        # no scalar llvm.* math intrinsics survive; the vector one is left (no air.<op>.v4f32)
        @test !any(n -> startswith(n, "llvm.") && !endswith(n, "v4f32"), names)
        @test "llvm.sqrt.v4f32" in names
        @test (verify(mod); true)
    end

    # apply_fastmath! flags every FP op `afn` (target.fastmath path), so even calls written
    # without @fastmath lower to the relaxed variant.
    Context() do ctx
        ir = """
        declare float @llvm.sqrt.f32(float)
        define float @f(float %x) {
          %a = call float @llvm.sqrt.f32(float %x)
          ret float %a
        }
        """
        mod = parse(LLVM.Module, ir)
        GPUCompiler.apply_fastmath!(mod)
        f = functions(mod)["f"]
        @test GPUCompiler.lower_math_intrinsics!(f)
        @test "air.fast_sqrt.f32" in called_names(f)
        @test (verify(mod); true)
    end
end

@testset "integer intrinsic lowering" begin
    # The integer ops Julia emits as llvm.* are lowered to their AIR builtins, so Metal.jl need
    # not wrap them. Names/signatures verified against Apple's frontend:
    #   llvm.abs (value,i1)  -> air.abs.{s,u}.iN(value)   (the i1 poison flag is dropped)
    #   llvm.{s,u}{min,max}  -> air.{min,max}.{s,u}.iN
    #   llvm.ctlz (v,i1)     -> air.clz.iN(v,i1)          (pure renames, i1 kept)
    #   llvm.cttz (v,i1)     -> air.ctz.iN(v,i1)
    #   llvm.ctpop           -> air.popcount.iN
    #   llvm.bitreverse      -> air.reverse_bits.iN
    km = @eval module $(gensym())
        f() = return
    end
    job, _ = Metal.create_job(km.f, Tuple{})

    Context() do ctx
        ir = """
        declare i32 @llvm.abs.i32(i32, i1)
        declare i32 @llvm.smin.i32(i32, i32)
        declare i32 @llvm.umax.i32(i32, i32)
        declare i32 @llvm.ctlz.i32(i32, i1)
        declare i32 @llvm.cttz.i32(i32, i1)
        declare i32 @llvm.ctpop.i32(i32)
        declare i32 @llvm.bitreverse.i32(i32)
        define void @f(i32 %x, i32 %y) {
          %a = call i32 @llvm.abs.i32(i32 %x, i1 true)
          %b = call i32 @llvm.smin.i32(i32 %x, i32 %y)
          %c = call i32 @llvm.umax.i32(i32 %x, i32 %y)
          %d = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
          %e = call i32 @llvm.cttz.i32(i32 %x, i1 false)
          %g = call i32 @llvm.ctpop.i32(i32 %x)
          %h = call i32 @llvm.bitreverse.i32(i32 %x)
          ret void
        }
        """
        mod = parse(LLVM.Module, ir)
        f = functions(mod)["f"]
        GPUCompiler.lower_llvm_intrinsics!(job, f)

        sigs = Dict{String,String}()
        for bb in blocks(f), i in instructions(bb)
            i isa LLVM.CallBase || continue
            c = called_operand(i)
            c isa LLVM.Function && (sigs[LLVM.name(c)] = string(LLVM.function_type(c)))
        end
        # abs drops the i1 poison operand
        @test haskey(sigs, "air.abs.s.i32") && sigs["air.abs.s.i32"] == "i32 (i32)"
        @test haskey(sigs, "air.min.s.i32") && sigs["air.min.s.i32"] == "i32 (i32, i32)"
        @test haskey(sigs, "air.max.u.i32")
        # clz/ctz keep the i1; popcount/reverse_bits are single-operand
        @test haskey(sigs, "air.clz.i32") && sigs["air.clz.i32"] == "i32 (i32, i1)"
        @test haskey(sigs, "air.ctz.i32") && sigs["air.ctz.i32"] == "i32 (i32, i1)"
        @test haskey(sigs, "air.popcount.i32") && sigs["air.popcount.i32"] == "i32 (i32)"
        @test haskey(sigs, "air.reverse_bits.i32")
        # no llvm.* intrinsics survive
        @test !any(startswith(n, "llvm.") for n in keys(sigs))
        @test (verify(mod); true)
    end
end

end
