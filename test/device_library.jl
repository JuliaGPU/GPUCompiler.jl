using Test

module LibraryFixtures
using GPUCompiler, LLVM, LLVM.IR, LLVM.Build

# a runtime whose allocator doesn't fold away
module Runtime
    malloc(sz) = Ptr{Cvoid}(Int(0xDEADBEEF))
    signal_exception() = return
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

helper(value::Int32) = value + Int32(7)
vector_export(::Vector) = Int32(7)
exported(value::Int32) = helper(value)
caller(value::Int32) = ccall("extern test_export", llvmcall, Int32, (Int32,), value)
alias_caller(value::Int32) =
    ccall("extern test_alias", llvmcall, Int32, (Int32,), value)
other_caller(value::Int32) =
    ccall("extern test_export", llvmcall, Int32, (Int32,), value) +
    ccall("extern other", llvmcall, Int32, (Int32,), value)
const julia_library = GPUCompiler.DeviceLibrary(:test;
    methods=(GPUCompiler.DeviceLibraryMethod(exported, Int32, (Int32,);
                                             llvm_name="test_export"),
             GPUCompiler.DeviceLibraryMethod(exported, Int32, (Int32,);
                                             llvm_name="test_alias")))

# `lower_intrinsics!` replaces `llvm.exp.f32` with a call to `test_exp`
exp_caller(value::Float32) = ccall("llvm.exp.f32", llvmcall, Float32, (Float32,), value)
both_caller(value::Float32) =
    ccall("extern test_exp", llvmcall, Float32, (Float32,), value) + exp_caller(value)
library_exp(value::Float32) = value * 2f0 + 3f0
fma_exp(value::Float32) = fma(value, 2f0, 3f0)
mutable struct Box
    value::Float32
end
@noinline escape(box::Box) = (box.value += 1f0; box.value)
allocating_exp(value::Float32) = escape(Box(value))
exp_library(f; kwargs...) = GPUCompiler.DeviceLibrary(:exp;
    methods=(GPUCompiler.DeviceLibraryMethod(f, Float32, (Float32,);
                                             llvm_name="test_exp"),), kwargs...)

struct Params <: GPUCompiler.AbstractCompilerParams
    libraries::Tuple
end
GPUCompiler.runtime_module(::CompilerJob{<:Any,Params}) = Runtime
GPUCompiler.device_libraries(job::CompilerJob{<:Any,Params}) =
    (job.config.params.libraries..., GPUCompiler.runtime_library(Runtime))
function GPUCompiler.lower_intrinsics!(::CompilerJob{<:Any,Params}, mod::LLVM.Module)
    exp = LLVM.Intrinsic("llvm.exp")
    for fun in mod.functions
        isdeclaration(fun) && continue
        GPUCompiler.lower_intrinsic_calls!(fun) do builder, call, intrinsic
            intrinsic == exp || return nothing
            GPUCompiler.call_declared!(builder, "test_exp", LLVM.FloatType(),
                                       collect(LLVM.Value, call.arguments))
        end
    end
end

function job(func, types; libs=(julia_library,), target=:native, kwargs...)
    source = GPUCompiler.methodinstance(typeof(func), Base.to_tuple_type(types))
    target = target === :native ? NativeCompilerTarget() : PTXCompilerTarget(; cap=v"7.0")
    validate = target isa PTXCompilerTarget
    CompilerJob(source, CompilerConfig(target, Params(libs);
                                       kernel=false, validate, kwargs...))
end
function ir(func, types=(Int32,); kwargs...)
    JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job(func, types; kwargs...))
        try
            @assert LLVM.verify(mod) === nothing
            string(mod)
        finally
            LLVM.dispose(mod)
        end
    end
end

asm(func, types; kwargs...) = JuliaContext() do ctx
    first(GPUCompiler.compile(:asm, job(func, types; kwargs...)))
end

bitcode(text) = JuliaContext() do ctx
    @dispose mod=parse(LLVM.Module, text) begin
        buffer = IOBuffer()
        write(buffer, mod)
        take!(buffer)
    end
end
end

@testset "Julia libraries" begin
    # imported before optimization, and optimized with the caller
    text = LibraryFixtures.ir(LibraryFixtures.caller)
    @test occursin("define internal fastcc i32 @test_export", text)
    @test contains(text, r"add i32 %\S+, 7")
    @test occursin("define internal fastcc i32 @test_alias",
                   LibraryFixtures.ir(LibraryFixtures.alias_caller))

    @test occursin("define", LibraryFixtures.ir(LibraryFixtures.caller; optimize=false))
    @test occursin("declare i32 @test_export",
                   LibraryFixtures.ir(LibraryFixtures.caller; libraries=false))

    # equivalent descriptors share cached artifacts
    vectors(T) = GPUCompiler.DeviceLibrary(:vectors;
        methods=(GPUCompiler.DeviceLibraryMethod(LibraryFixtures.vector_export, Int32,
                                                 (UnionAll(T, Vector{T}),);
                                                 llvm_name="vector_export"),))
    job = LibraryFixtures.job(LibraryFixtures.caller, (Int32,))
    cached = JuliaContext() do ctx
        GPUCompiler.load_julia_library(vectors(TypeVar(:T, Real)), job; lowered=false)
        length(GPUCompiler.device_libs)
    end
    @test cached == JuliaContext() do ctx
        GPUCompiler.load_julia_library(vectors(TypeVar(:S, Real)), job; lowered=false)
        length(GPUCompiler.device_libs)
    end

    # libraries follow method redefinitions
    @eval LibraryFixtures helper(value::Int32) = value + Int32(19)
    @test contains(LibraryFixtures.ir(LibraryFixtures.caller), r"add i32 %\S+, 19")
end

@testset "optimized Julia libraries" begin
    library = GPUCompiler.DeviceLibrary(:optimized;
        methods=LibraryFixtures.julia_library.methods, optimize=true)
    @test !occursin("@test_export",
                    LibraryFixtures.ir(LibraryFixtures.caller; libs=(library,)))

    @test_throws ArgumentError GPUCompiler.DeviceLibrary(:bad; bitcode=UInt8[], optimize=true)
    @test_throws "Duplicate" GPUCompiler.DeviceLibrary(:bad;
        methods=(first(LibraryFixtures.julia_library.methods),
                 first(LibraryFixtures.julia_library.methods)))
end

@testset "bitcode libraries" begin
    bitcode = LibraryFixtures.bitcode
    upstream = GPUCompiler.DeviceLibrary(:upstream; bitcode=bitcode("""
        declare i32 @downstream(i32)
        define i32 @test_export(i32 %value) {
            %result = call i32 @downstream(i32 %value)
            ret i32 %result
        }"""))
    downstream = GPUCompiler.DeviceLibrary(:downstream; bitcode=bitcode("""
        define i32 @downstream(i32 %value) {
            %result = mul i32 %value, 37
            ret i32 %result
        }"""))

    # dependencies are resolved regardless of the order of libraries
    for libraries in ((upstream, downstream), (downstream, upstream))
        text = LibraryFixtures.ir(LibraryFixtures.caller; libs=libraries)
        @test occursin("mul i32 %value, 37", text)
        @test !occursin("declare", text)
    end

    # unresolved references are left to validation
    @test occursin("declare i32 @downstream",
                   LibraryFixtures.ir(LibraryFixtures.caller; libs=(upstream,)))

    # libraries can share linkonce definitions
    shared(name, constant) = GPUCompiler.DeviceLibrary(Symbol(name); bitcode=bitcode("""
        define linkonce_odr i32 @shared() { ret i32 $constant }
        define i32 @$name(i32 %value) {
            %shared = call i32 @shared()
            %result = add i32 %value, %shared
            ret i32 %result
        }"""))
    text = LibraryFixtures.ir(LibraryFixtures.other_caller;
                              libs=(shared("test_export", 41), shared("other", 41)))
    @test occursin("ret i32 41", text)
    @test count("define internal fastcc i32 @shared", text) == 1
    @test !occursin("declare", text)

    # aliases are exports too
    aliased = GPUCompiler.DeviceLibrary(:aliased; bitcode=bitcode("""
        define i32 @implementation(i32 %value) {
            %result = mul i32 %value, 53
            ret i32 %result
        }
        @test_export = alias i32 (i32), i32 (i32)* @implementation"""))
    text = LibraryFixtures.ir(LibraryFixtures.caller; libs=(aliased,))
    @test occursin("mul i32 %value, 53", text)
    @test !occursin("declare", text)
    mismatch = GPUCompiler.DeviceLibrary(:mismatch; bitcode=bitcode("""
        define float @implementation(float %value) { ret float %value }
        @test_export = alias float (float), float (float)* @implementation"""))
    @test_throws "incompatible" LibraryFixtures.ir(LibraryFixtures.caller; libs=(mismatch,))

    # struct types are compared structurally
    returns_struct = GPUCompiler.DeviceLibrary(:returns_struct; bitcode=bitcode("""
        %pair = type { i32, i32 }
        declare %pair @make_pair(i32)
        define i32 @test_export(i32 %value) {
            %pair = call %pair @make_pair(i32 %value)
            %result = extractvalue %pair %pair, 1
            ret i32 %result
        }"""))
    makes_struct = GPUCompiler.DeviceLibrary(:makes_struct; bitcode=bitcode("""
        %pair = type { i32, i32 }
        define %pair @make_pair(i32 %value) {
            %first = insertvalue %pair undef, i32 %value, 0
            %second = insertvalue %pair %first, i32 43, 1
            ret %pair %second
        }"""))
    text = LibraryFixtures.ir(LibraryFixtures.caller; libs=(returns_struct, makes_struct))
    @test occursin("i32 43", text)
    @test !occursin("declare", text)

    # incompatible definitions are rejected
    mismatch = GPUCompiler.DeviceLibrary(:mismatch; bitcode=bitcode("""
        define float @test_export(float %value) { ret float %value }"""))
    @test_throws "incompatible" LibraryFixtures.ir(LibraryFixtures.caller;
                                                   libs=(mismatch,))

    # forced inlining
    noinline = GPUCompiler.DeviceLibrary(:noinline; bitcode=bitcode("""
        define i32 @test_export(i32 %value) noinline {
            %result = mul i32 %value, 47
            ret i32 %result
        }"""))
    inline = GPUCompiler.DeviceLibrary(:inline; bitcode=bitcode("""
        define i32 @test_export(i32 %value) {
            %result = mul i32 %value, 47
            ret i32 %result
        }"""), always_inline=true)
    @test occursin("call fastcc i32 @test_export",
                   LibraryFixtures.ir(LibraryFixtures.caller; libs=(noinline,)))
    @test !occursin("@test_export",
                    LibraryFixtures.ir(LibraryFixtures.caller; libs=(inline,)))
end

@testset "late imports" begin
    exp_library = LibraryFixtures.exp_library

    # lowered intrinsics are imported after optimization
    for library in (exp_library(LibraryFixtures.library_exp),
                    exp_library(LibraryFixtures.library_exp; optimize=true),
                    exp_library(LibraryFixtures.library_exp; optimize=true, always_inline=false))
        text = LibraryFixtures.ir(LibraryFixtures.exp_caller, (Float32,);
                                  libs=(library,))
        @test !occursin("llvm.exp", text)
        @test !occursin("declare float @test_exp", text)
        @test occursin("3.000000e+00", text)
        @test occursin("call fastcc float @test_exp", text) == !library.always_inline
    end
    text = LibraryFixtures.ir(LibraryFixtures.exp_caller, (Float32,);
                              libs=(exp_library(LibraryFixtures.library_exp),),
                              optimize=false)
    @test !occursin("llvm.exp", text)
    @test occursin("define", text)

    # an export can be needed both before and after optimization
    text = LibraryFixtures.ir(LibraryFixtures.both_caller, (Float32,);
                              libs=(exp_library(LibraryFixtures.library_exp),))
    @test !occursin("llvm.exp", text)
    @test !occursin("declare float @test_exp", text)
    @test !occursin("unreachable", text)
    @test count("call fastcc float @test_exp", text) == 2

    # Julia libraries imported late are lowered like any other Julia code
    library = exp_library(LibraryFixtures.allocating_exp)
    text = LibraryFixtures.ir(LibraryFixtures.exp_caller, (Float32,);
                              libs=(library,), target=:ptx)
    @test !occursin("@julia.", text)
    @test occursin("gpu_gc_pool_alloc", text)

    # late imports are processed like early ones
    text = LibraryFixtures.ir(LibraryFixtures.exp_caller, (Float32,);
                              libs=(exp_library(LibraryFixtures.fma_exp),), target=:ptx)
    @test !occursin("@julia.cpu", text)
    fence = GPUCompiler.DeviceLibrary(:fence; bitcode=LibraryFixtures.bitcode("""
        define float @test_exp(float %value) {
            fence syncscope("workgroup") seq_cst
            ret float %value
        }"""))
    text = LibraryFixtures.ir(LibraryFixtures.exp_caller, (Float32,);
                              libs=(fence,), target=:ptx)
    @test occursin("syncscope(\"block\")", text)

    # late imports are internalized, including aliases (which need PTX 6.3)
    if NVPTX_LLVM_Backend_jll.is_available()
        aliased = GPUCompiler.DeviceLibrary(:aliased; bitcode=LibraryFixtures.bitcode("""
            define float @implementation(float %value) { ret float %value }
            @test_exp = alias float (float), float (float)* @implementation"""))
        text = LibraryFixtures.asm(LibraryFixtures.exp_caller, (Float32,);
                                   libs=(aliased,), target=:ptx)
        @test !occursin(".alias", text)
    end
end
