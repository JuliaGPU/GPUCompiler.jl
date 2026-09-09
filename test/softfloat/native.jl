# execution of the emulated binary64 routines through the native back-end's JIT

using GPUCompiler.SoftFloat: SoftFloat64Provider
using LLVM, Test

module SoftFloatFixtures
soft_widen16(a::UInt64, b::UInt64) =
    reinterpret(UInt64, Float64(reinterpret(Float16, a % UInt16)))
soft_widen32(a::UInt64, b::UInt64) =
    reinterpret(UInt64, Float64(reinterpret(Float32, a % UInt32)))
soft_add(a::UInt64, b::UInt64) =
    reinterpret(UInt64, reinterpret(Float64, a) + reinterpret(Float64, b))
# (using `sqrt_llvm` directly, as the native target has no runtime to throw a DomainError)
soft_math(a::UInt64, b::UInt64) =
    reinterpret(UInt64, Base.sqrt_llvm(abs(reinterpret(Float64, a))) * reinterpret(Float64, b) -
                        fma(reinterpret(Float64, a), 2.0, 0.5))
end

# the provider is only selected for the fixtures, not for any other native job
GPUCompiler.device_library_providers(job::Native.NativeCompilerJob) =
    job.source.def.module === SoftFloatFixtures ? (SoftFloat64Provider(),) : ()

function compile_and_load(f)
    job, _ = Native.create_job(f, (UInt64, UInt64))
    obj, entry, relocs = GPUCompiler.JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)
        obj, LLVM.name(meta.entry), meta.relocations
    end
    Native.load(Vector{UInt8}(codeunits(obj)), entry, relocs)
end

@testset "emulated binary64 on the native back-end" begin
    ir = sprint(io -> Native.code_llvm(io, SoftFloatFixtures.soft_add, Tuple{UInt64, UInt64};
                                       dump_module=true))
    @test !occursin(r"\bdouble\b", ir)
    @test occursin("define internal fastcc i64 @gpu_softfloat_add64", ir)

    for (f, reference) in ((SoftFloatFixtures.soft_add, (a, b) -> a + b),
                           (SoftFloatFixtures.soft_math, (a, b) -> sqrt(abs(a)) * b - fma(a, 2.0, 0.5)))
        ptr, jit, _ = compile_and_load(f)
        try
            for (a, b) in ((1.0, 2.0), (floatmax(Float64), -floatmax(Float64)),
                           (nextfloat(0.0), nextfloat(0.0)), (-0.0, -0.0), (1e-300, 1e300))
                result = ccall(ptr, UInt64, (UInt64, UInt64), reinterpret(UInt64, a), reinterpret(UInt64, b))
                @test result == reinterpret(UInt64, reference(a, b))
            end
        finally
            LLVM.dispose(jit)
        end
    end

    plain, _ = Native.create_job(identity, (UInt64,))
    @test isempty(GPUCompiler.device_library_providers(plain))
end

@testset "library routines compile to integer code" begin
    # every routine must lower to plain integer arithmetic: no double-precision values,
    # no 128-bit integers, and no calls into the Julia runtime (e.g. to throw)
    for method in GPUCompiler.SoftFloat.METHODS
        ir = sprint(io -> Native.code_llvm(io, method.def, method.types; dump_module=true))
        @test !occursin(r"\bdouble\b", ir)
        @test !occursin(r"(?<![\w.])i128\b(?!:)", ir)
        @test !occursin(r"call .*@i?jl_", ir)
    end
end

@testset "legalized widening with flush-to-zero enabled" begin
    for (f, T, U) in ((SoftFloatFixtures.soft_widen16, Float16, UInt16),
                       (SoftFloatFixtures.soft_widen32, Float32, UInt32))
        ptr, jit, _ = compile_and_load(f)
        old = get_zero_subnormals()
        try
            set_zero_subnormals(false)
            inputs = UInt64[1, Base.significand_mask(T), Base.sign_mask(T) | U(1)]
            expected = [reinterpret(UInt64, Float64(reinterpret(T, x % U))) for x in inputs]
            if set_zero_subnormals(true)
                for (x, y) in zip(inputs, expected)
                    @test ccall(ptr, UInt64, (UInt64, UInt64), x, UInt64(0)) == y
                end
            else
                @test_skip "flush-to-zero is unavailable on this CPU"
            end
        finally
            set_zero_subnormals(old)
            LLVM.dispose(jit)
        end
    end
end
