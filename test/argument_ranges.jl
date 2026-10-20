using GPUCompiler, Test

struct FactArray
    ptr::Ptr{Float32}
    dims::Tuple{Int,Int}
    len::Int
    bytes::Int
end
const array_ranges = ((:dims, 1) => 0:typemax(Int32),
                      (:dims, 2) => 0:typemax(Int32),
                      (:len,) => 0:typemax(Int32))
struct BoundedValue{T,L,H}
    value::T
end
const SignedIndex = BoundedValue{Int,Int(typemin(Int32)),Int(typemax(Int32))}
const signed_ranges = ((:value,) => typemin(Int32):typemax(Int32),)

@testset "compiler integration" begin
    # Assumptions must be emitted before ABI lowering, including for closure fields.
    function exceeds(out, a, n)
        unsafe_store!(out, a.len > typemax(Int32) || n.value < typemin(Int32))
        return
    end
    tt = Tuple{Ptr{Bool},FactArray,SignedIndex}
    specialization = ((), (), array_ranges, signed_ranges)
    job, _ = Native.create_job(exceeds, tt; kernel=true, argument_ranges=specialization)
    cfg = job.config
    @test cfg.argument_ranges === specialization && cfg.narrow_indices
    @test CompilerConfig(cfg).argument_ranges === specialization
    @test CompilerConfig(cfg).narrow_indices
    @test hash(cfg) != hash(CompilerConfig(cfg; argument_ranges=()))
    @test hash(cfg) != hash(CompilerConfig(cfg; narrow_indices=false))
    @test @filecheck begin
        @check "store i8 0"
        Native.code_llvm(exceeds, tt; kernel=true, argument_ranges=specialization)
    end
    if sizeof(Int) == 8
        @test @filecheck begin
            @check "icmp"
            Native.code_llvm(exceeds, tt; kernel=true, argument_ranges=())
        end
    end
    @test @filecheck begin
        @check "call void @llvm.assume"
        Native.code_llvm(exceeds, tt; kernel=true, argument_ranges=specialization, optimize=false)
    end
    # Entry argument facts do not apply to non-toplevel compilation.
    @test @filecheck begin
        @check_not "call void @llvm.assume"
        job = CompilerJob(job; config=CompilerConfig(cfg; toplevel=false))
        GPUCompiler.code_llvm(job)
    end
    a = FactArray(C_NULL, (10, 20), 200, 800)
    captured = out -> exceeds(out, a, SignedIndex(0))
    specialization = (Tuple((:a, path...) => range for (path, range) in array_ranges), ())
    @test @filecheck begin
        @check "store i8 0"
        Native.code_llvm(captured, Tuple{Ptr{Bool}}; kernel=true, argument_ranges=specialization)
    end
end

@testset "target pipelines" begin
    function indexed(out, a, i, j)
        idx = (j - 1) * a.dims[1] + i
        LLVM.Interop.assume(1 <= idx <= a.len)
        unsafe_store!(out, unsafe_load(a.ptr, idx))
        return
    end
    tt = Tuple{Core.LLVMPtr{Float32,1},FactArray,Int,Int}
    specialization = ((), (), array_ranges, (), ())
    for (backend, options) in ((Native, (;)), (PTX, (;)), (GCN, (;)),
                               (Metal, (;)), (SPIRV, (; backend=:llvm)))
        sizeof(Int) == 8 || continue # only 64-bit host indices need narrowing
        backend === GCN && LLVM.is_asserts() && continue
        @testset "$(nameof(backend))" begin
            @test @filecheck begin
                @check "zext {{(nneg )?}}i32"
                backend.code_llvm(indexed, tt; kernel=true, argument_ranges=specialization, options...)
            end
        end
    end
end

@testset "native execution" begin
    # The products may wrap in i32, while their difference is a bounded array index.
    function difference(out, data, i, j, stride)
        idx = i * stride - j * stride
        LLVM.Interop.assume(0 <= idx < 1024)
        unsafe_store!(out, unsafe_load(data, idx + 1))
        return
    end
    tt = Tuple{Ptr{Int},Ptr{Int},Int,Int,Int}
    data = collect(1:1024)
    out = Ref(0)
    for facts in (false, true)
        job, _ = Native.create_job(difference, tt; kernel=true, narrow_indices=facts)
        JuliaContext() do ctx
            obj, meta = GPUCompiler.compile(:obj, job)
            entry = meta.entry.name
            dispose(meta.ir)
            fptr, jit, table = Native.load(Vector{UInt8}(codeunits(obj)), entry, meta.relocations)
            @dispose jit=jit begin
                # Equal large indices and small differences cover intermediate signed and
                # unsigned 32-bit overflow, including negative launch-checked inputs.
                for i in (0, 1, 32768, 65535, typemax(Int32), typemin(Int32) + 1),
                    delta in (0, 1), stride in (1, 7, 1023)
                    j = i - delta
                    GC.@preserve data out table begin
                        ccall(fptr, Cvoid, (Ptr{Int},Ptr{Int},Int,Int,Int),
                              out, data, i, j, stride)
                    end
                    @test out[] == delta * stride + 1
                end
            end
        end
    end
end

# Keep aggregate arguments as references for the native test launcher's C ABI.
struct RangeCompilerParams <: AbstractCompilerParams end
GPUCompiler.pass_by_value(::CompilerJob{NativeCompilerTarget,RangeCompilerParams}) = false

if sizeof(Int) == 8
    @testset "checked fallback" begin
        function outside(out, x)
            unsafe_store!(out, x.value < typemin(Int32) || x.value > typemax(Int32))
            return
        end
        specialization = ((), (), signed_ranges)
        variants = []
        try
            for facts in (false, true)
                job, _ = Native.create_job(outside, Tuple{Ptr{Bool},SignedIndex};
                                            kernel=true, argument_ranges=facts ? specialization : ())
                job = CompilerJob(job; config=CompilerConfig(job.config; params=RangeCompilerParams()))
                JuliaContext() do ctx
                    obj, meta = GPUCompiler.compile(:obj, job)
                    entry = meta.entry.name
                    dispose(meta.ir)
                    push!(variants, Native.load(Vector{UInt8}(codeunits(obj)), entry,
                                                meta.relocations))
                end
            end
            out = Ref(false)
            # A miniature launcher: the compiled fact-bearing variant is reusable, but
            # its caller must recheck every new argument value and select the fallback.
            for x in (0, -1, Int(typemin(Int32)), Int(typemax(Int32)),
                      Int(typemin(Int32)) - 1, Int(typemax(Int32)) + 1,
                      typemin(Int), typemax(Int))
                fits = typemin(Int32) <= x <= typemax(Int32)
                fptr = variants[fits ? 2 : 1][1]
                ccall(fptr, Cvoid, (Ptr{Bool},Ref{SignedIndex}), out, SignedIndex(x))
                @test out[] == !(typemin(Int32) <= x <= typemax(Int32))
            end
        finally
            for (_, jit, _) in variants
                dispose(jit)
            end
        end
    end
end

@testset "emitted integer widths" begin
    function outside_range(out, x::BoundedValue{T,L,H}) where {T,L,H}
        unsafe_store!(out, x.value < L || x.value > H)
        return
    end
    for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128, Bool)
        lo = T <: Signed ? T(-3) : T(1)
        hi = T === Bool ? true : typemax(T)
        tt = Tuple{Ptr{Bool},BoundedValue{T,lo,hi}}
        @test @filecheck begin
            @check "store i8 0"
            specialization = ((), (), ((:value,) => lo:hi,))
            Native.code_llvm(outside_range, tt; kernel=true, argument_ranges=specialization)
        end
    end
end

mutable struct RangeResult
    RangeResult() = new()
end

@testset "caller-owned facts and cache identity" begin
    function exceeds_ten(out, x)
        unsafe_store!(out, x > 10)
        return
    end
    tt = Tuple{Ptr{Bool},Int}
    small = ((), (), (() => 0:10,))
    large = ((), (), (() => 0:100,))
    job, _ = Native.create_job(exceeds_ten, tt; kernel=true, argument_ranges=small)
    cfg = job.config
    larger = CompilerConfig(cfg; argument_ranges=large)
    @test typeof(cfg) === typeof(larger) # facts are values, not type parameters
    @test GPUCompiler.matches_config(cfg, CompilerConfig(cfg; argument_ranges=deepcopy(small)))
    @test !GPUCompiler.matches_config(cfg, larger)
    @test hash(cfg) != hash(larger)
    @test @filecheck begin
        @check "store i8 0"
        Native.code_llvm(exceeds_ten, tt; kernel=true, argument_ranges=small)
    end
    @test @filecheck begin
        @check "icmp"
        Native.code_llvm(exceeds_ten, tt; kernel=true, argument_ranges=large)
    end
    if GPUCompiler.HAS_INTEGRATED_CACHE
        a = GPUCompiler.cached_results(RangeResult, job)
        @test a !== nothing
        @test GPUCompiler.cached_results(RangeResult, CompilerJob(job;
            config=CompilerConfig(cfg; argument_ranges=deepcopy(small)))) === a
        @test GPUCompiler.cached_results(RangeResult, CompilerJob(job; config=larger)) !== a
    end
    # Identical physical fields can be asserted for another function without a policy hook.
    other(out, x) = exceeds_ten(out, x)
    @test @filecheck begin
        @check "store i8 0"
        Native.code_llvm(other, tt; kernel=true, argument_ranges=small)
    end
end
