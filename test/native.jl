@testset "native" begin

include("definitions/native.jl")

############################################################################################

@testset "Compilation" begin
    kernel() = nothing

    output = native_code_execution(kernel, (); validate=false)
    @test occursin("kernel", output[2])
    @test isempty(output[3])
    @test isempty(output[4])

    @testset "Undefined Functions" begin
        function undef_fn()
            ccall("extern somefunc", llvmcall, Cvoid, ())
            nothing
        end

        output = native_code_execution(undef_fn, (); validate=false)
        @test length(output[3]) == 1
        @test output[3][1] == "somefunc"
    end

    @testset "Undefined Globals" begin
        @generated function makegbl(::Val{name}, ::Type{T}, ::Val{isext}) where {name,T,isext}
            JuliaContext() do ctx
                T_gbl = convert(LLVMType, T, ctx)
                T_ptr = convert(LLVMType, Ptr{T}, ctx)
                llvm_f, _ = create_function(T_ptr)
                mod = LLVM.parent(llvm_f)
                gvar = GlobalVariable(mod, T_gbl, string(name))
                isext && extinit!(gvar, true)
                Builder(ctx) do builder
                    entry = BasicBlock(llvm_f, "entry", ctx)
                    position!(builder, entry)
                    result = ptrtoint!(builder, gvar, T_ptr)
                    ret!(builder, result)
                end
                call_function(llvm_f, Ptr{T})
            end
        end
        function undef_gbl()
            ext_ptr = makegbl(Val(:someglobal), Int64, Val(true))
            Base.unsafe_store!(ext_ptr, 1)
            ptr = makegbl(Val(:otherglobal), Float32, Val(false))
            Base.unsafe_store!(ptr, 2f0)
            nothing
        end

        output = native_code_execution(undef_gbl, ())
        @test length(output[4]) == 2
        @test output[4][1].name == "someglobal"
        @test eltype(output[4][1].type) isa LLVM.IntegerType
        @test output[4][1].external
        @test output[4][2].name == "otherglobal"
        @test eltype(output[4][2].type) isa LLVM.LLVMFloat
        @test !output[4][2].external
    end

    @testset "Callable structs" begin
        struct MyCallable end
        (::MyCallable)(a, b) = a+b

        (CI, rt) = native_code_typed(MyCallable(), (Int, Int), kernel=false)[1]
        @test CI.slottypes[1] == Core.Compiler.Const(MyCallable())
    end
end

############################################################################################

@testset "IR" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    ir = sprint(io->native_code_llvm(io, valid_kernel, Tuple{}; optimize=false, dump_module=true))

    # module should contain our function + a generic call wrapper
    @test occursin(r"define\ .* void\ @.*julia_valid_kernel.*\(\)"x, ir)
    @test !occursin("define %jl_value_t* @jlcall_", ir)

    # there should be no debug metadata
    @test !occursin("!dbg", ir)

    @test native_code_llvm(devnull, invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError native_code_llvm(devnull, invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "unbound typevars" begin
    invalid_kernel() where {unbound} = return
    @test_throws KernelError native_code_llvm(devnull, invalid_kernel, Tuple{})
end

@testset "child functions" begin
    # we often test using `@noinline sink` child functions, so test whether these survive
    @noinline child(i) = sink(i)
    parent(i) = child(i)

    ir = sprint(io->native_code_llvm(io, parent, Tuple{Int}))
    @test occursin(r"call .+ @julia_child_", ir)
end

@testset "sysimg" begin
    # bug: use a system image function

    function foobar(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
    end

    ir = sprint(io->native_code_llvm(io, foobar, Tuple{Ptr{Int},Int}))
    @test !occursin("jlsys_", ir)
end

@testset "tracked pointers" begin
    function kernel(a)
        a[1] = 1
        return
    end

    # this used to throw an LLVM assertion (#223)
    native_code_llvm(devnull, kernel, Tuple{Vector{Int}}; kernel=true)
end

@testset "CUDAnative.jl#278" begin
    # codegen idempotency
    # NOTE: this isn't fixed, but surfaces here due to bad inference of checked_sub
    # NOTE: with the fix to print_to_string this doesn't error anymore,
    #       but still have a test to make sure it doesn't regress
    native_code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)
    native_code_llvm(devnull, Base.checked_sub, Tuple{Int,Int}; optimize=false)

    # breaking recursion in print_to_string makes it possible to compile
    # even in the presence of the above bug
    native_code_llvm(devnull, Base.print_to_string, Tuple{Int,Int}; optimize=false)
end

@testset "LLVM D32593" begin
    @eval struct D32593_struct
        foo::Float32
        bar::Float32
    end

    D32593(ptr) = unsafe_load(ptr).foo

    native_code_llvm(devnull, D32593, Tuple{Ptr{D32593_struct}})
end

@testset "Julia-level throw lowering" begin
    kernel(ptr) = (unsafe_store!(ptr, sqrt(unsafe_load(ptr))); nothing)

    native_code_execution(kernel, Tuple{Ptr{Float32}})
end

end

############################################################################################

@testset "assembly" begin

@testset "basic reflection" begin
    valid_kernel() = return
    invalid_kernel() = 1

    @test native_code_native(devnull, valid_kernel, Tuple{}) == nothing
    @test native_code_native(devnull, invalid_kernel, Tuple{}) == nothing
    @test_throws KernelError native_code_native(devnull, invalid_kernel, Tuple{}; kernel=true)
end

@testset "idempotency" begin
    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)

    kernel() = return
    native_code_native(devnull, kernel, Tuple{})
    native_code_native(devnull, kernel, Tuple{})
end

@testset "compile for host after gpu" begin
    # issue #11: re-using host functions after GPU compilation
    @noinline child(i) = sink(i+1)

    function fromhost()
        child(10)
    end

    function fromptx()
        child(10)
        return
    end

    native_code_native(devnull, fromptx, Tuple{})
    @test fromhost() == 11
end

end

############################################################################################

@testset "errors" begin

@eval Main begin
struct CleverType{T}
    x::T
end
Base.unsafe_trunc(::Type{Int}, x::CleverType) = unsafe_trunc(Int, x.x)
end

@testset "non-isbits arguments" begin
    foobar(i) = (sink(unsafe_trunc(Int,i)); return)

    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{BigInt})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("BigInt", msg)
    end

    # test that we can handle abstract types
    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{Any})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Any", msg)
    end

    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{Union{Int32, Int64}})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Union{Int32, Int64}", msg)
    end

    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{Union{Int32, Int64}})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Union{Int32, Int64}", msg)
    end

    # test that we get information about fields and reason why something is not isbits
    @test_throws_message(KernelError,
                         native_code_execution(foobar, Tuple{CleverType{BigInt}})) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("CleverType", msg) &&
        occursin("BigInt", msg)
    end
end

@testset "invalid LLVM IR" begin
    foobar(i) = println(i)

    @test_throws_message(InvalidIRError,
                         native_code_execution(foobar, Tuple{Int})) do msg
        occursin("invalid LLVM IR", msg) &&
        (occursin(GPUCompiler.RUNTIME_FUNCTION, msg) ||
         occursin(GPUCompiler.UNKNOWN_FUNCTION, msg)) &&
        occursin("[1] println", msg) &&
        occursin(r"\[2\] .+foobar", msg)
    end
end

@testset "invalid LLVM IR (ccall)" begin
    foobar(p) = (unsafe_store!(p, ccall(:time, Cint, ())); nothing)

    @test_throws_message(InvalidIRError,
                         native_code_execution(foobar, Tuple{Ptr{Int}})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.POINTER_FUNCTION, msg) &&
        occursin(r"\[1\] .+foobar", msg)
    end
end

@testset "delayed bindings" begin
    kernel() = (undefined; return)

    @test_throws_message(InvalidIRError,
                         native_code_execution(kernel, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DELAYED_BINDING, msg) &&
        occursin("use of 'undefined'", msg) &&
        occursin(r"\[1\] .+kernel", msg)
    end
end

@testset "dynamic call (invoke)" begin
    @eval @noinline nospecialize_child(@nospecialize(i)) = i
    kernel(a, b) = (unsafe_store!(b, nospecialize_child(a)); return)

    @test_throws_message(InvalidIRError,
                         native_code_execution(kernel, Tuple{Int,Ptr{Int}})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to nospecialize_child", msg) &&
        occursin(r"\[1\] .+kernel", msg)
    end
end

@testset "dynamic call (apply)" begin
    func() = println(1)

    @test_throws_message(InvalidIRError,
                         native_code_execution(func, Tuple{})) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to println", msg) &&
        occursin("[2] func", msg)
    end
end

@testset "LazyCodegen" begin
    import .LazyCodegen: call_delayed

    global flag = Ref(false) # otherwise f is a closure and we can't
                             # pass it to `Val`...
    f() = (flag[]=true; nothing)

    function caller()
        call_delayed(f)
    end
    @test caller() === nothing
    @test flag[]

    ir = sprint(io->native_code_llvm(io, caller, Tuple{}, dump_module=true))
    @test occursin(r"define void @julia_f_\d+", ir)

    add(x, y) = x+y
    function call_add(x, y)
        call_delayed(add, x, y)
    end

    @test call_add(1, 3) == 4

    incr(r) = r[] += 1
    function call_incr(r)
        call_delayed(incr, r)
    end
    r = Ref{Int}(0)
    @test call_incr(r) == 1
    @test r[] == 1

    function call_real(c)
        call_delayed(real, c)
    end

    @test call_real(1.0+im) == 1.0

    # Test ABI removal
    ir = sprint(io->native_code_llvm(io, call_real, Tuple{ComplexF64}))
    @test !occursin(r"alloca", ir)

    ghostly_identity(x, y) = y
    @test call_delayed(ghostly_identity, nothing, 1) == 1

    # tests struct return
    @test call_delayed(complex, 1.0, 2.0) == 1.0+2.0im
end

end

############################################################################################

end
