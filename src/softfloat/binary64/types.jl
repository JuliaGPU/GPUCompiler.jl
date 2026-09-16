# The SoftFloat64 type, and the representation shared by all operations on it.

"""
    SoftFloat64 <: AbstractFloat

An IEEE 754 binary64 value, represented like `Float64`, whose arithmetic, comparisons and
conversions are implemented with 64-bit integer operations only, for targets without native
double-precision support. `reinterpret` converts to and from `Float64` for free.

Operations round to nearest, ties to even, unless a trailing `RoundingMode` is passed: the
arithmetic operators, `sqrt`, `fma`, `round`, and the conversions from integers and to
narrower floating-point types accept one. Arithmetic and rounding canonicalize NaN results
(`0x7ff8000000000000`).
Bit-preserving operations (`reinterpret`, negation, `abs`, and `copysign`) retain NaN
payloads. There are no exception flags.
"""
primitive type SoftFloat64 <: AbstractFloat 64 end

SoftFloat64(x::Float64) = reinterpret(SoftFloat64, x)
Base.Float64(x::SoftFloat64) = reinterpret(Float64, x)
Base.show(io::IO, x::SoftFloat64) = print(io, "SoftFloat64(", Float64(x), ")")

# the format traits of Float64 apply
for f in (:sign_mask, :exponent_mask, :significand_mask, :exponent_one, :exponent_half,
          :significand_bits, :exponent_bits, :exponent_bias, :exponent_max, :exponent_raw_max)
    @eval Base.$f(::Type{SoftFloat64}) = Base.$f(Float64)
end
Base.uinttype(::Type{SoftFloat64}) = UInt64
Base.precision(::Type{SoftFloat64}; base::Integer=2) = precision(Float64; base)
Base.eps(::Type{SoftFloat64}) = SoftFloat64(eps(Float64))
Base.eps(x::SoftFloat64) = abs(x - reinterpret(SoftFloat64, bits(x) ⊻ UInt64(1)))

# Keep mixed arithmetic in software. Larger integer and arbitrary-precision types need
# their own conversion and promotion policies before they can participate.
Base.promote_rule(::Type{SoftFloat64}, ::Type{T}) where
    {T<:Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float16,Float32,Float64}} = SoftFloat64

# The helpers below apply to any IEEE format `T` these traits describe, i.e. SoftFloat64
# and the native Float16 and Float32, so that conversions between them can share code.

bits(x::T) where {T} = reinterpret(uinttype(T), x)

Base.signbit(x::SoftFloat64) = reinterpret(Int64, x) < 0
Base.isnan(x::SoftFloat64) = bits(x) & ~sign_mask(SoftFloat64) > exponent_mask(SoftFloat64)
Base.isinf(x::SoftFloat64) = bits(x) & ~sign_mask(SoftFloat64) == exponent_mask(SoftFloat64)
Base.isfinite(x::SoftFloat64) = bits(x) & exponent_mask(SoftFloat64) != exponent_mask(SoftFloat64)
Base.iszero(x::SoftFloat64) = bits(x) & ~sign_mask(SoftFloat64) == 0
Base.issubnormal(x::SoftFloat64) = (bits(x) & exponent_mask(SoftFloat64) == 0) & !iszero(x)

# special values of format `T`
signed_zero(::Type{T}, sign::Bool) where {T} =
    reinterpret(T, uinttype(T)(sign) << (8sizeof(T) - 1))
infinity(::Type{T}, sign::Bool) where {T} =
    reinterpret(T, bits(signed_zero(T, sign)) | exponent_mask(T))
largest_finite(::Type{T}, sign::Bool) where {T} =
    reinterpret(T, bits(infinity(T, sign)) - one(uinttype(T)))
quiet_nan(::Type{T}) where {T} =
    reinterpret(T, exponent_mask(T) | (one(uinttype(T)) << (significand_bits(T) - 1)))

Base.zero(::Type{SoftFloat64}) = signed_zero(SoftFloat64, false)
Base.one(::Type{SoftFloat64}) = reinterpret(SoftFloat64, exponent_one(SoftFloat64))
Base.typemin(::Type{SoftFloat64}) = infinity(SoftFloat64, true)
Base.typemax(::Type{SoftFloat64}) = infinity(SoftFloat64, false)
Base.floatmax(::Type{SoftFloat64}) = largest_finite(SoftFloat64, false)
Base.floatmin(::Type{SoftFloat64}) = reinterpret(SoftFloat64, significand_mask(SoftFloat64) + 1)

"""
    unpack(x) -> (sign, exp, sig)

Decompose a finite, nonzero value into its sign, binary exponent and integral significand,
such that `x == ±sig * 2^(exp - significand_bits(T))`: the significand's implicit leading
bit is made explicit, and subnormals are normalized.
"""
@inline function unpack(x::T) where {T}
    u = bits(x)
    sign = u & sign_mask(T) != 0
    exp = ((u & exponent_mask(T)) >> significand_bits(T)) % Int
    sig = u & significand_mask(T)
    if exp == 0  # subnormal
        shift = leading_zeros(sig) - exponent_bits(T)
        return sign, 1 - exponent_bias(T) - shift, sig << shift
    end
    return sign, exp - exponent_bias(T), sig | (significand_mask(T) + one(u))
end

# Base's numeric hashing uses an exact numerator * 2^exponent / denominator decomposition.
function Base.decompose(x::SoftFloat64)
    isnan(x) && return Int64(0), 0, 0
    isinf(x) && return Int64(signbit(x) ? -1 : 1), 0, 0
    iszero(x) && return Int64(0), 0, signbit(x) ? -1 : 1
    sign, exp, sig = unpack(x)
    return Int64(sig), exp - significand_bits(SoftFloat64), sign ? -1 : 1
end
