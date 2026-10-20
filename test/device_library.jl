using Test

module LibraryFixtures
using GPUCompiler, LLVM, LLVM.IR, LLVM.Build

helper(value::Int32) = value + Int32(7)
exported(value::Int32) = helper(value)
caller(value::Int32) = ccall("extern test_export", llvmcall, Int32, (Int32,), value)
alias_caller(value::Int32) =
    ccall("extern test_alias", llvmcall, Int32, (Int32,), value)
intrinsic(value::Float32) = ccall("llvm.exp.f32", llvmcall, Float32, (Float32,), value)
const selection = Ref{Tuple}((GPUCompiler.DeviceLibrary(:test;
    methods=(GPUCompiler.DeviceLibraryMethod(exported, Int32, (Int32,);
                                            llvm_name="test_export"),
             GPUCompiler.DeviceLibraryMethod(exported, Int32, (Int32,);
                                            llvm_name="test_alias"))),))
struct Params <: GPUCompiler.AbstractCompilerParams
    libraries::Tuple
    Params() = new(selection[])
end
GPUCompiler.runtime_module(::CompilerJob{<:Any,Params}) = @__MODULE__
GPUCompiler.device_libraries(job::CompilerJob{<:Any,Params}) = job.config.params.libraries
const after_import = Ref{Any}(nothing)
function GPUCompiler.finish_imports!(::CompilerJob{<:Any,Params}, mod::LLVM.Module, imported)
    after_import[] === nothing || after_import[]()
    return
end

function job(func, types=(Int32,); kwargs...)
    source = GPUCompiler.methodinstance(typeof(func), Tuple{types...})
    CompilerJob(source, CompilerConfig(NativeCompilerTarget(), Params();
                                       kernel=false, validate=false, kwargs...))
end
function ir(func; kwargs...)
    JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job(func; kwargs...))
        try
            @assert LLVM.verify(mod) === nothing
            string(mod)
        finally
            LLVM.dispose(mod)
        end
    end
end
end

mutable struct LibraryTestResults
    value::Int
    LibraryTestResults() = new(0)
end

@testset "kernel cache follows library dependencies" begin
    original = LibraryFixtures.selection[]
    try
        LibraryFixtures.ir(LibraryFixtures.caller)
        job = LibraryFixtures.job(LibraryFixtures.caller)
        results = GPUCompiler.cached_results(LibraryTestResults, job)
        results.value = 42
        @test GPUCompiler.cached_results(LibraryTestResults, job) === results
        @eval LibraryFixtures helper(value::Int32) = value + Int32(23)
        new_job = LibraryFixtures.job(LibraryFixtures.caller)
        @test GPUCompiler.cached_results(LibraryTestResults, new_job).value == 0
        LibraryFixtures.ir(LibraryFixtures.caller)
        results = GPUCompiler.cached_results(LibraryTestResults, new_job)
        results.value = 17
        @test GPUCompiler.cached_results(LibraryTestResults, new_job).value == 17
        LibraryFixtures.selection[] = ()
        empty_job = LibraryFixtures.job(LibraryFixtures.caller)
        @test GPUCompiler.cached_results(LibraryTestResults, empty_job) === nothing
    finally
        LibraryFixtures.selection[] = original
    end
end

@testset "optimized library" begin
    original = LibraryFixtures.selection[]
    try
        LibraryFixtures.selection[] = (GPUCompiler.DeviceLibrary(:optimized;
            methods=first(original).methods, optimize=true),)
        text = LibraryFixtures.ir(LibraryFixtures.caller)
        @test !occursin("call i32 @test_export", text)
        @test occursin("23", text)
    finally
        LibraryFixtures.selection[] = original
    end
end

@testset "Julia device libraries" begin
    @test occursin("23", LibraryFixtures.ir(LibraryFixtures.caller))
    @test !occursin("declare i32 @test_export", LibraryFixtures.ir(LibraryFixtures.caller))
    @test !occursin("declare i32 @test_alias", LibraryFixtures.ir(LibraryFixtures.alias_caller))
    @test occursin("define", LibraryFixtures.ir(LibraryFixtures.caller; optimize=false))
    @test occursin("declare i32 @test_export",
                   LibraryFixtures.ir(LibraryFixtures.caller; libraries=false))

    @eval LibraryFixtures helper(value::Int32) = value + Int32(19)
    @test occursin("19", LibraryFixtures.ir(LibraryFixtures.caller))
end

@testset "dependency order and selection" begin
    original = LibraryFixtures.selection[]
    try
        JuliaContext() do ctx
            downstream = parse(LLVM.Module, """
                define i32 @downstream(i32 %value) { ret i32 %value }
                """)
            upstream = parse(LLVM.Module, """
                declare i32 @downstream(i32)
                define i32 @test_export(i32 %value) {
                    %result = call i32 @downstream(i32 %value)
                    ret i32 %result
                }
                """)
            bytes(mod) = (io = IOBuffer(); write(io, mod); take!(io))
            first_lib = GPUCompiler.DeviceLibrary(:upstream; bitcode=bytes(upstream))
            last_lib = GPUCompiler.DeviceLibrary(:downstream; bitcode=bytes(downstream))
            dispose(upstream)
            dispose(downstream)
            LibraryFixtures.selection[] = (first_lib, last_lib)
            @test !occursin("declare i32 @downstream", LibraryFixtures.ir(LibraryFixtures.caller))
            LibraryFixtures.selection[] = (last_lib, first_lib)
            @test_throws "earlier library" LibraryFixtures.ir(LibraryFixtures.caller)
            LibraryFixtures.selection[] = (first_lib,)
            @test_throws "no later library" LibraryFixtures.ir(LibraryFixtures.caller)
            LibraryFixtures.selection[] = (first_lib, first_lib)
            @test_throws "Duplicate" LibraryFixtures.ir(LibraryFixtures.caller)
        end
    finally
        LibraryFixtures.selection[] = original
    end
end

@testset "linkonce library exports" begin
    original = LibraryFixtures.selection[]
    try
        JuliaContext() do ctx
            function bitcode(text)
                @dispose mod=parse(LLVM.Module, text) begin
                    buffer = IOBuffer()
                    write(buffer, mod)
                    take!(buffer)
                end
            end
            functions = GPUCompiler.DeviceLibrary(:functions; bitcode=bitcode("""
                @control = external constant i32
                define linkonce_odr i32 @test_export(i32 %value) {
                    %control = load i32, i32* @control
                    %result = add i32 %value, %control
                    ret i32 %result
                }
                define linkonce_odr i32 @unused() { ret i32 0 }
                """))
            constants = GPUCompiler.DeviceLibrary(:constants; bitcode=bitcode("""
                @control = linkonce_odr constant i32 137
                """))
            LibraryFixtures.selection[] = (functions, constants)
            snapshot = GPUCompiler.device_library_bitcode(functions)
            @test snapshot.bytes == functions.bitcode
            @test snapshot.digest == functions.digest
            @test GPUCompiler.device_library_bitcode(functions).bytes === snapshot.bytes
            @test_throws ArgumentError GPUCompiler.device_library_bitcode(first(original))
            text = LibraryFixtures.ir(LibraryFixtures.caller)
            @test occursin("137", text)
            @test !occursin("declare i32 @test_export", text)
            @test !occursin("@control", text)
            @test !occursin("@unused", text)
        end
    finally
        LibraryFixtures.selection[] = original
    end
end

@testset "bitcode replacement invalidates caches" begin
    original = LibraryFixtures.selection[]
    try
        mktempdir() do directory
            path = joinpath(directory, "library.bc")
            function replace_library(constant)
                JuliaContext() do ctx
                    @dispose mod=parse(LLVM.Module, """
                        define i32 @test_export(i32 %value) { ret i32 $constant }
                        """) begin
                        write(path * ".new", mod)
                    end
                end
                mv(path * ".new", path; force=true)
            end
            replace_library(31)
            LibraryFixtures.selection[] = (GPUCompiler.DeviceLibrary(:file; bitcode=path),)
            @test occursin("31", LibraryFixtures.ir(LibraryFixtures.caller))
            job = LibraryFixtures.job(LibraryFixtures.caller)
            results = GPUCompiler.cached_results(LibraryTestResults, job)
            results.value = 42
            replace_library(103)
            @test GPUCompiler.cached_results(LibraryTestResults, job).value == 0
            @test occursin("103", LibraryFixtures.ir(LibraryFixtures.caller))
            GPUCompiler.cached_results(LibraryTestResults, job).value = 43
            LibraryFixtures.after_import[] = () -> replace_library(211)
            @test occursin("103", LibraryFixtures.ir(LibraryFixtures.caller))
            LibraryFixtures.after_import[] = nothing
            @test GPUCompiler.cached_results(LibraryTestResults, job).value == 0
        end
    finally
        LibraryFixtures.after_import[] = nothing
        LibraryFixtures.selection[] = original
    end
end

@testset "elementwise scalarization" begin
    JuliaContext() do ctx
        @dispose mod=parse(LLVM.Module, """
            declare <2 x float> @llvm.powi.v2f32.i32(<2 x float>, i32)
            define <2 x float> @test(<2 x float> %value, i32 %power) {
                %result = call afn <2 x float> @llvm.powi.v2f32.i32(<2 x float> %value, i32 %power)
                ret <2 x float> %result
            }
            """) begin
            @test GPUCompiler.scalarize_vector_calls!((call, intrinsic) -> true,
                                                       mod.functions["test"])
            @test verify(mod) === nothing
            @test count("call afn float @llvm.powi.f32.i32", string(mod)) == 2
        end
    end
end
