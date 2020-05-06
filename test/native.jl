@testset "native" begin

include("definitions/native.jl")

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
                         native_code_llvm(foobar, Tuple{BigInt}; strict=true)) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("BigInt", msg)
    end

    # test that we can handle abstract types
    @test_throws_message(KernelError,
                         native_code_llvm(foobar, Tuple{Any}; strict=true)) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Any", msg)
    end

    @test_throws_message(KernelError,
                         native_code_llvm(foobar, Tuple{Union{Int32, Int64}}; strict=true)) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Union{Int32, Int64}", msg)
    end

    @test_throws_message(KernelError,
                         native_code_llvm(foobar, Tuple{Union{Int32, Int64}}; strict=true)) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("Union{Int32, Int64}", msg)
    end

    # test that we get information about fields and reason why something is not isbits
    @test_throws_message(KernelError,
                         native_code_llvm(foobar, Tuple{CleverType{BigInt}}; strict=true)) do msg
        occursin("passing and using non-bitstype argument", msg) &&
        occursin("CleverType", msg) &&
        occursin("BigInt", msg)
    end
end

@testset "invalid LLVM IR" begin
    foobar(i) = println(i)

    @test_throws_message(InvalidIRError,
                         native_code_llvm(foobar, Tuple{Int}; strict=true)) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.RUNTIME_FUNCTION, msg) &&
        occursin("[1] println", msg) &&
        occursin(r"\[2\] .+foobar", msg)
    end
end

@testset "invalid LLVM IR (ccall)" begin
    foobar(p) = (unsafe_store!(p, ccall(:time, Cint, ())); nothing)

    @test_throws_message(InvalidIRError,
                         native_code_llvm(foobar, Tuple{Ptr{Int}}; strict=true)) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.POINTER_FUNCTION, msg) &&
        occursin(r"\[1\] .+foobar", msg)
    end
end

@testset "delayed bindings" begin
    kernel() = (undefined; return)

    @test_throws_message(InvalidIRError,
                         native_code_llvm(kernel, Tuple{}; strict=true)) do msg
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
                         native_code_llvm(kernel, Tuple{Int,Ptr{Int}}; strict=true)) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to nospecialize_child", msg) &&
        occursin(r"\[1\] .+kernel", msg)
    end
end

@testset "dynamic call (apply)" begin
    func() = pointer(1)

    @test_throws_message(InvalidIRError,
                         native_code_llvm(func, Tuple{}; strict=true)) do msg
        occursin("invalid LLVM IR", msg) &&
        occursin(GPUCompiler.DYNAMIC_CALL, msg) &&
        occursin("call to pointer", msg) &&
        occursin("[1] func", msg)
    end
end

end

############################################################################################

end
