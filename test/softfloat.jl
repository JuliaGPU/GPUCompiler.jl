# numerical tests of the SoftFloat64 routines on the CPU, against native Float64 arithmetic

using GPUCompiler.SoftFloat: SoftFloat64, unordered, minnum, maxnum, paynehanek
using Random, Test

soft(x::Float64) = SoftFloat64(x)
soft(x::UInt64) = reinterpret(SoftFloat64, x)
# results are compared bit for bit, except that NaNs are canonicalized
canonical(x::Float64) = isnan(x) ? NaN : x
canonical(x::SoftFloat64) = canonical(Float64(x))

const EDGE_BITS = UInt64[
    0x0000_0000_0000_0000, 0x8000_0000_0000_0000,   # ±0
    0x0000_0000_0000_0001, 0x000f_ffff_ffff_ffff,   # smallest and largest subnormal
    0x0010_0000_0000_0000, 0x3ff0_0000_0000_0000,   # smallest normal, 1
    0x3ff0_0000_0000_0001, 0x7fef_ffff_ffff_ffff,   # nextfloat(1), floatmax
    0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000,   # ±Inf
    0x7ff0_0000_0000_0001, 0x7ff8_0000_0000_0042,   # signaling and payloaded NaN
]

rng = Xoshiro(0x5f64)
corpus = reinterpret(Float64, vcat(EDGE_BITS, rand(rng, UInt64, 25_000)))
edges = reinterpret(Float64, EDGE_BITS)
pairs = [(corpus[i], corpus[mod1(7919i, length(corpus))]) for i in eachindex(corpus)]

@testset "arithmetic" begin
    for (a, b) in pairs
        @test canonical(soft(a) + soft(b)) === canonical(a + b)
        @test canonical(soft(a) - soft(b)) === canonical(a - b)
        @test canonical(soft(a) * soft(b)) === canonical(a * b)
        @test canonical(soft(a) / soft(b)) === canonical(a / b)
    end

    # uniformly random bits rarely exercise cancellation or the subnormal boundary
    for i in 1:10_000
        a = rand(rng, UInt64)
        b = (a & 0x7ff0_0000_0000_0000) | (rand(rng, UInt64) & 0x800f_ffff_ffff_ffff)
        for (x, y) in ((a, b), (a & 0x801f_ffff_ffff_ffff, b & 0x801f_ffff_ffff_ffff))
            @test canonical(soft(x) + soft(y)) === canonical(reinterpret(Float64, x) + reinterpret(Float64, y))
            @test canonical(soft(x) - soft(y)) === canonical(reinterpret(Float64, x) - reinterpret(Float64, y))
        end
    end
    for a in edges, b in edges
        @test canonical(soft(a) + soft(b)) === canonical(a + b)
        @test canonical(soft(a) * soft(b)) === canonical(a * b)
        @test canonical(soft(a) / soft(b)) === canonical(a / b)
    end

    for a in corpus
        @test canonical(sqrt(soft(a))) === canonical(a < 0 ? NaN : sqrt(a))
        @test Float64(abs(soft(a))) === abs(a)
        @test Float64(-soft(a)) === -a
    end
    for (a, b) in pairs
        @test Float64(copysign(soft(a), soft(b))) === copysign(a, b)
    end

    for i in eachindex(corpus)
        a = corpus[i]
        b = corpus[mod1(3571i, length(corpus))]
        c = corpus[mod1(12347i, length(corpus))]
        @test canonical(fma(soft(a), soft(b), soft(c))) === canonical(fma(a, b, c))
    end
    for a in edges, b in edges, c in edges
        @test canonical(fma(soft(a), soft(b), soft(c))) === canonical(fma(a, b, c))
    end
end

@testset "rounding modes" begin
    # Directed rounding is checked against a BigFloat computation of the exact result (the
    # precision covers the exact product and sum in an FMA) that is rounded once. The
    # computation itself also happens under the mode, for the sign of exact zero results.
    reference(mode, f, xs...) = setrounding(BigFloat, mode) do
        setprecision(BigFloat, 4300) do
            Float64(f(map(BigFloat, xs)...), mode)
        end
    end
    modes = (RoundNearest, RoundToZero, RoundDown, RoundUp)
    finite = filter(isfinite, corpus)
    # add values near the overflow and underflow boundaries
    boundaries = [floatmax(Float64), -floatmax(Float64), prevfloat(floatmax(Float64)),
                  floatmin(Float64), nextfloat(0.0), prevfloat(floatmin(Float64)), 0.0, -0.0,
                  1.0, -1.0, 2.0^-1074 * 3, 1e-160, -1e-160, 0.5, 3.0]
    inputs = vcat(finite[1:5000], boundaries)
    for mode in modes, i in eachindex(inputs)
        a = inputs[i]
        b = inputs[mod1(7919i, length(inputs))]
        c = inputs[mod1(3571i, length(inputs))]
        @test Float64(+(soft(a), soft(b), mode)) === reference(mode, +, a, b)
        @test Float64(-(soft(a), soft(b), mode)) === reference(mode, -, a, b)
        @test Float64(*(soft(a), soft(b), mode)) === reference(mode, *, a, b)
        iszero(b) || @test Float64(/(soft(a), soft(b), mode)) === reference(mode, /, a, b)
        a < 0 || @test Float64(sqrt(soft(a), mode)) === reference(mode, sqrt, a)
        @test Float64(fma(soft(a), soft(b), soft(c), mode)) === reference(mode, (x, y, z) -> x * y + z, a, b, c)
    end
    for a in boundaries, b in boundaries
        for mode in modes
            @test Float64(+(soft(a), soft(b), mode)) === reference(mode, +, a, b)
            @test Float64(*(soft(a), soft(b), mode)) === reference(mode, *, a, b)
        end
        # exact cancellation gives -0.0 only when rounding down
        @test Float64(+(soft(a), soft(-a), RoundDown)) === -0.0
        @test Float64(+(soft(a), soft(-a), RoundUp)) === 0.0
    end
    for (a, b, c) in ((floatmax(Float64), 2.0, -floatmax(Float64)),
                       (floatmin(Float64), 0.5, -prevfloat(floatmin(Float64))),
                       (prevfloat(1.0), nextfloat(1.0), -1.0))
        for mode in modes
            @test Float64(fma(soft(a), soft(b), soft(c), mode)) ===
                reference(mode, (x, y, z) -> x * y + z, a, b, c)
        end
    end
    # Random triples almost never cancel a product. Use the rounded product as the
    # opposing addend, including cases where the residual is subnormal.
    for i in 1:2000
        a = ldexp(1 + rand(rng), rand(rng, -500:500))
        b = ldexp(1 + rand(rng), rand(rng, -500:500))
        c = -(a * b)
        for mode in modes
            @test Float64(fma(soft(a), soft(b), soft(c), mode)) ===
                reference(mode, (x, y, z) -> x * y + z, a, b, c)
        end
    end
    # overflow rounds to the largest finite value when rounding towards zero
    @test Float64(*(soft(floatmax(Float64)), soft(2.0), RoundToZero)) === floatmax(Float64)
    @test Float64(*(soft(floatmax(Float64)), soft(2.0), RoundDown)) === floatmax(Float64)
    @test Float64(*(soft(floatmax(Float64)), soft(2.0), RoundUp)) === Inf
    @test Float64(*(soft(-floatmax(Float64)), soft(2.0), RoundUp)) === -floatmax(Float64)
    @test Float64(*(soft(-floatmax(Float64)), soft(2.0), RoundDown)) === -Inf
end

@testset "comparisons" begin
    for (a, b) in vcat(pairs, [(a, b) for a in edges, b in edges][:])
        x, y = soft(a), soft(b)
        @test (x == y) == (a == b)
        @test (x < y) == (a < b)
        @test (x <= y) == (a <= b)
        @test unordered(x, y) == (isnan(a) || isnan(b))
        @test canonical(min(x, y)) === canonical(min(a, b))
        @test canonical(max(x, y)) === canonical(max(a, b))
        # minnum/maxnum return the number when only one operand is NaN
        @test canonical(minnum(x, y)) === (isnan(a) ? canonical(b) : isnan(b) ? a : canonical(min(a, b)))
        @test canonical(maxnum(x, y)) === (isnan(a) ? canonical(b) : isnan(b) ? a : canonical(max(a, b)))
    end
    for a in corpus
        x = soft(a)
        @test isnan(x) == isnan(a)
        @test isinf(x) == isinf(a)
        @test isfinite(x) == isfinite(a)
        @test iszero(x) == iszero(a)
        @test issubnormal(x) == issubnormal(a)
        @test signbit(x) == signbit(a)
    end
end

@testset "rounding" begin
    # random bits rarely have a fractional part, so also exercise moderate magnitudes
    inputs = vcat(corpus, (rand(rng, 10_000) .- 0.5) .* 2.0 .^ rand(rng, -5:60, 10_000),
                  -4.0:0.25:4.0, [2.0^52 + 0.5, 2.0^52 - 0.5, -2.0^53 + 1])
    for a in inputs
        x = soft(a)
        @test canonical(trunc(x)) === canonical(trunc(a))
        @test canonical(round(x)) === canonical(round(a))
        @test canonical(round(x, RoundNearestTiesAway)) === canonical(round(a, RoundNearestTiesAway))
        @test canonical(floor(x)) === canonical(floor(a))
        @test canonical(ceil(x)) === canonical(ceil(a))
    end
end

@testset "conversions" begin
    for x in vcat(rand(rng, Int64, 25_000), typemin(Int64), typemax(Int64), 0, -1, 1)
        @test Float64(SoftFloat64(x)) === Float64(x)
        for mode in (RoundToZero, RoundDown, RoundUp)
            @test Float64(SoftFloat64(x, mode)) === Float64(BigFloat(x), mode)
        end
    end
    for x in vcat(rand(rng, UInt64, 25_000), typemax(UInt64), 0, 1, UInt64(1) << 63)
        @test Float64(SoftFloat64(x)) === Float64(x)
        for mode in (RoundToZero, RoundDown, RoundUp)
            @test Float64(SoftFloat64(x, mode)) === Float64(BigFloat(x), mode)
        end
    end
    @test Float64(SoftFloat64(Int32(-7))) === -7.0
    @test Float64(SoftFloat64(UInt8(255))) === 255.0
    # float-to-integer routines implement fptosi/fptoui, i.e. truncation of in-range values
    for x in vcat([(rand(rng) - 0.5) * 2.0^63 for _ in 1:10_000], [rand(rng) * 100 for _ in 1:10_000],
                  -1.5, 0.5, -0.0, -2.0^63, prevfloat(2.0^63))
        @test unsafe_trunc(Int64, soft(x)) === unsafe_trunc(Int64, x)
    end
    for x in vcat([rand(rng) * 2.0^64 for _ in 1:10_000], [rand(rng) * 100 for _ in 1:10_000],
                  0.5, 0.0, prevfloat(2.0^64))
        @test unsafe_trunc(UInt64, soft(x)) === unsafe_trunc(UInt64, x)
    end

    for x in reinterpret(Float32, rand(rng, UInt32, 25_000))
        @test canonical(SoftFloat64(x)) === canonical(Float64(x))
    end
    for h in reinterpret(Float16, UInt16(0):typemax(UInt16))
        @test canonical(SoftFloat64(h)) === canonical(Float64(h))
    end
    for x in corpus
        @test Float32(soft(x)) === (isnan(x) ? NaN32 : Float32(x))
        for mode in (RoundToZero, RoundDown, RoundUp)
            isnan(x) || @test Float32(soft(x), mode) === Float32(BigFloat(x), mode)
        end
    end
    # values above the half-way point in the 2^-150 bin round to the least Float32 subnormal
    @test Float32(soft(0x369c_6d9b_5f38_1bd0)) === reinterpret(Float32, 0x0000_0001)

    # every Float16 round-trips
    for h in reinterpret(Float16, UInt16(0):typemax(UInt16))
        x = Float64(h)
        @test Float16(soft(x)) === (isnan(x) ? NaN16 : h)
    end
    # both sides of every positive finite Float16 midpoint, including the normal/subnormal
    # boundary; an intermediate Float32 rounding would lose the side
    for h in UInt16(0):UInt16(0x7bfe)
        x = Float64(reinterpret(Float16, h))
        y = Float64(reinterpret(Float16, h + UInt16(1)))
        midpoint = (x + y) / 2
        for z in (prevfloat(midpoint), midpoint, nextfloat(midpoint))
            @test Float16(soft(z)) === Float16(z)
            @test Float16(soft(-z)) === Float16(-z)
            for mode in (RoundToZero, RoundDown, RoundUp)
                @test Float16(soft(z), mode) === Float16(BigFloat(z), mode)
                @test Float16(soft(-z), mode) === Float16(BigFloat(-z), mode)
            end
        end
    end
end

@testset "paynehanek" begin
    # bit-for-bit identical to Base's UInt128-based implementation
    for x in vcat([2.0^20 * pi/2, 1e20, 1e100, floatmax(Float64), 2.0^20, nextfloat(2.0^20)],
                  [rand(rng) * 2.0^rand(rng, 21:1023) for _ in 1:10_000])
        for y in (x, -x)
            @test paynehanek(y) === Base.Math.paynehanek(y)
        end
    end
end

@testset "number interface" begin
    for x in corpus
        sx = soft(x)
        @test isequal(eps(sx), soft(eps(x)))
        @test hash(sx) == hash(x)
        @test isequal(sx, x)
    end
    @test precision(SoftFloat64) == 53
    @test precision(soft(1.0); base=10) == precision(1.0; base=10)
    @test eps(SoftFloat64) === soft(eps(Float64))
    for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
              Float16, Float32, Float64)
        @test soft(1.5) + one(T) === soft(2.5)
        @test one(T) + soft(1.5) === soft(2.5)
        @test promote_type(SoftFloat64, T) == SoftFloat64
    end
    @test length(Set([soft(-0.0), soft(0.0), soft(NaN), soft(1.5)])) == 4
end

@testset "rounding ties and exact results" begin
    for x in -4.0:0.25:4.0
        @test Float64(round(soft(x), RoundNearestTiesUp)) === round(x, RoundNearestTiesUp)
        @test Float64(round(soft(x), RoundFromZero)) === round(x, RoundFromZero)
    end
    for mode in (RoundNearest, RoundToZero, RoundDown, RoundUp, RoundFromZero,
                 RoundNearestTiesAway, RoundNearestTiesUp)
        for i in 1:1000
            # At most 26 significant bits, so the square is exactly representable.
            x = Float64(rand(rng, UInt32) & 0x03ffffff)
            y = ldexp(1.0, rand(rng, -900:900))
            @test Float64(/(soft(x*y), soft(y), mode)) === x
            @test Float64(sqrt(soft(x*x), mode)) === x
        end
    end
    for mode in (RoundNearestTiesAway, RoundNearestTiesUp)
        @test Float64(+(soft(1.0), soft(2.0^-53), mode)) === nextfloat(1.0)
        @test Float64(+(soft(-1.0), soft(-2.0^-53), mode)) ===
            (mode === RoundNearestTiesUp ? -1.0 : prevfloat(-1.0))
        @test Float64(*(soft(nextfloat(0.0)), soft(0.5), mode)) === nextfloat(0.0)
        @test Float64(*(soft(-nextfloat(0.0)), soft(0.5), mode)) ===
            (mode === RoundNearestTiesUp ? -0.0 : -nextfloat(0.0))
    end
end

@testset "widening with flush-to-zero enabled" begin
    # Pass bits through a call boundary so constant folding cannot hide native FP operations.
    @noinline widen_bits(x::T) where {T<:Union{UInt16,UInt32}} =
        reinterpret(UInt64, SoftFloat64(reinterpret(T === UInt16 ? Float16 : Float32, x)))
    inputs = (UInt16[0, 1, 0x03ff, 0x0400, 0x8001, 0x83ff, 0x7c00, 0x7e01],
              UInt32[0, 1, 0x007fffff, 0x00800000, 0x80000001, 0x807fffff, 0x7f800000, 0x7fc00001])
    old = get_zero_subnormals()
    try
        set_zero_subnormals(false)
        expected = map(xs -> map(widen_bits, xs), inputs)
        if set_zero_subnormals(true)
            for (xs, ys) in zip(inputs, expected), (x, y) in zip(xs, ys)
                @test widen_bits(x) == y
            end
        else
            @test_skip "flush-to-zero is unavailable on this CPU"
        end
    finally
        set_zero_subnormals(old)
    end
end
