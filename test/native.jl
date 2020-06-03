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
            T_gbl = convert(LLVMType, T)
            T_ptr = convert(LLVMType, Ptr{T})
            llvm_f, _ = create_function(T_ptr)
            mod = LLVM.parent(llvm_f)
            gvar = GlobalVariable(mod, T_gbl, string(name))
            isext && extinit!(gvar, true)
            Builder(JuliaContext()) do builder
                entry = BasicBlock(llvm_f, "entry", JuliaContext())
                position!(builder, entry)
                result = ptrtoint!(builder, gvar, T_ptr)
                ret!(builder, result)
            end
            call_function(llvm_f, Ptr{T})
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

if VERSION >= v"1.0.2"
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
end

@testset "LLVM D32593" begin
    @eval struct D32593_struct
        foo::Float32
        bar::Float32
    end

    D32593(ptr) = unsafe_load(ptr).foo

    native_code_llvm(devnull, D32593, Tuple{Ptr{D32593_struct}})
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

# some validation happens in the emit_function hook, which is called by code_llvm

@testset "base intrinsics" begin
    foobar(i) = sin(i)

    # NOTE: we don't use test_logs in order to test all of the warning (exception, backtrace)
    logs, _ = Test.collect_test_logs(min_level=Info) do
        withenv("JULIA_DEBUG" => nothing) do
            native_code_llvm(devnull, foobar, Tuple{Int})
        end
    end
    @test length(logs) == 1
    record = logs[1]
    @test record.level == Base.CoreLogging.Warn
    @test record.message == "calls to Base intrinsics might be GPU incompatible"
    @test haskey(record.kwargs, :exception)
    err,bt = record.kwargs[:exception]
    err_msg = sprint(showerror, err)
    @test occursin(Regex("You called sin(.+) in Base.Math .+, maybe you intended to call sin(.+) in $TestRuntime .+ instead?"), err_msg)
    bt_msg = sprint(Base.show_backtrace, bt)
    @test occursin("[1] sin", bt_msg)
    @test occursin(r"\[2\] .+foobar", bt_msg)
end

# some validation happens in `compile`

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
        occursin(GPUCompiler.RUNTIME_FUNCTION, msg) &&
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

end

############################################################################################

end
