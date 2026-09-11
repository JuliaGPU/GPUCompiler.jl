# job-scoped device-library providers

using LLVM, Test

module DeviceLibraryFixtures

using GPUCompiler, LLVM

increment(x::UInt64) = UInt64(1)
add_one(x::Unsigned) = x + increment(x)
times_two(x::UInt64) = x * UInt64(2)

const METHODS = (
    GPUCompiler.DeviceLibraryMethod(add_one, UInt64, (UInt64,); llvm_name="gpu_provider_add_one"),
    GPUCompiler.DeviceLibraryMethod(times_two, UInt64, (UInt64,)),
)

struct Provider <: GPUCompiler.AbstractDeviceLibraryProvider end
GPUCompiler.device_library_methods(::Provider, ::GPUCompiler.CompilerJob) = METHODS

const prepare_calls = Ref(0)
function GPUCompiler.prepare_device_library!(::Provider, ::GPUCompiler.CompilerJob,
                                             mod::LLVM.Module, entry::LLVM.Function)
    prepare_calls[] += 1
    entry
end

function call_provider(x::UInt64)
    Base.llvmcall(("""
        declare i64 @gpu_provider_add_one(i64)
        declare i64 @gpu_times_two(i64)

        define i64 @entry(i64 %x) {
          %a = call i64 @gpu_provider_add_one(i64 %x)
          %b = call i64 @gpu_times_two(i64 %a)
          ret i64 %b
        }
        """, "entry"), UInt64, Tuple{UInt64}, x)
end
no_provider_calls(x::UInt64) = x + 1

end

GPUCompiler.device_library_providers(job::Native.NativeCompilerJob) =
    job.source.def.module === DeviceLibraryFixtures ? (DeviceLibraryFixtures.Provider(),) : ()

@testset "device-library providers" begin
    empty!(GPUCompiler.device_libs)
    DeviceLibraryFixtures.prepare_calls[] = 0
    ir = sprint() do io
        Native.code_llvm(io, DeviceLibraryFixtures.call_provider, Tuple{UInt64};
                         dump_module=true, optimize=false, cleanup=false, validate=false)
    end
    @test occursin("define internal i64 @gpu_provider_add_one", ir)
    @test occursin("define internal i64 @gpu_times_two", ir)
    @test DeviceLibraryFixtures.prepare_calls[] == 1
    @test length(GPUCompiler.device_libs) == 1

    # the library is not even loaded when unused
    empty!(GPUCompiler.device_libs)
    Native.code_llvm(devnull, DeviceLibraryFixtures.no_provider_calls, Tuple{UInt64})
    @test DeviceLibraryFixtures.prepare_calls[] == 2
    @test isempty(GPUCompiler.device_libs)

    # a different job receives no provider merely because the fixture module was loaded
    plain, _ = Native.create_job(identity, (UInt64,))
    @test isempty(GPUCompiler.device_library_providers(plain))

end

# compile, load and call a fixture through the native JIT
function run_native(f, x)
    job, _ = Base.invokelatest(Native.create_job, f, (UInt64,))
    obj, entry, relocs = GPUCompiler.JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)
        obj, LLVM.name(meta.entry), meta.relocations
    end
    ptr, jit, _ = Native.load(Vector{UInt8}(codeunits(obj)), entry, relocs)
    try
        ccall(ptr, UInt64, (UInt64,), x)
    finally
        LLVM.dispose(jit)
    end
end

@testset "device-library execution and invalidation" begin
    @test run_native(DeviceLibraryFixtures.call_provider, UInt64(5)) == 12

    # redefining a transitive dependency of a library method must rebuild the library
    @eval DeviceLibraryFixtures increment(x::UInt64) = UInt64(3)
    @test run_native(DeviceLibraryFixtures.call_provider, UInt64(5)) == 16
end
