# Conversions between binary64 and integers, integral values, and the narrower floating-point
# formats, which reuse `round_pack` for their rounding.


## integers

@inline SoftFloat64(x::Int64, mode::RoundingMode=RoundNearest) =
    # `abs(typemin(Int64))` wraps, but its reinterpretation is the right magnitude
    iszero(x) ? zero(SoftFloat64) :
    norm_round_pack(SoftFloat64, x < 0, 0, reinterpret(UInt64, abs(x)), mode)
@inline SoftFloat64(x::UInt64, mode::RoundingMode=RoundNearest) =
    iszero(x) ? zero(SoftFloat64) : norm_round_pack(SoftFloat64, false, 0, x, mode)
SoftFloat64(x::Integer, mode::RoundingMode=RoundNearest) =
    SoftFloat64((x isa Unsigned ? UInt64 : Int64)(x), mode)

# The truncating conversions implement LLVM's `fptosi` and `fptoui`, which are only
# defined for values that fit; other values saturate here, but that must not be relied on.
function Base.unsafe_trunc(::Type{Int64}, x::SoftFloat64)
    (isnan(x) | iszero(x)) && return Int64(0)
    isinf(x) && return signbit(x) ? typemin(Int64) : typemax(Int64)
    sign, exp, sig = unpack(x)
    exp < 0 && return Int64(0)
    exp >= 63 && return sign ? typemin(Int64) : typemax(Int64)
    magnitude = sig << (exp - significand_bits(SoftFloat64))  # a negative shift is to the right
    sign ? -(magnitude % Int64) : magnitude % Int64
end
function Base.unsafe_trunc(::Type{UInt64}, x::SoftFloat64)
    (isnan(x) | iszero(x) | signbit(x)) && return UInt64(0)
    isinf(x) && return typemax(UInt64)
    _, exp, sig = unpack(x)
    exp < 0 && return UInt64(0)
    exp >= 64 && return typemax(UInt64)
    sig << (exp - significand_bits(SoftFloat64))
end


## rounding to an integral value

# (per mode, as Base has fallbacks for AbstractFloat with some of them, which a method
# for all modes would be ambiguous with)
for mode in (:Nearest, :NearestTiesAway, :NearestTiesUp, :ToZero, :Up, :Down, :FromZero)
    @eval Base.round(x::SoftFloat64, mode::RoundingMode{$(QuoteNode(mode))}) =
        round_integral(x, mode)
end
Base.round(x::SoftFloat64) = round(x, RoundNearest)

function round_integral(x::SoftFloat64, mode::RoundingMode)
    isnan(x) && return quiet_nan(SoftFloat64)
    iszero(x) && return x
    u = bits(x)
    sign = signbit(x)
    exp = ((u & exponent_mask(SoftFloat64)) >> significand_bits(SoftFloat64)) % Int -
          exponent_bias(SoftFloat64)
    exp >= significand_bits(SoftFloat64) && return x  # integral already, or infinite
    if exp < 0
        # a magnitude below one rounds to zero or one; its leading bit is the half bit when
        # the exponent is -1, and any lower bit is a rest bit
        up = round_up(mode, sign, false, exp == -1,
                      (exp < -1) | (u & significand_mask(SoftFloat64) != 0))
        return up ? copysign(one(SoftFloat64), x) : signed_zero(SoftFloat64, sign)
    end
    # the weight of one in the bit pattern: adding it to the integral part increments the
    # value, carrying into the exponent when required
    unit = one(UInt64) << (significand_bits(SoftFloat64) - exp)
    fraction = u & (unit - 1)
    iszero(fraction) && return x
    up = round_up(mode, sign, u & unit != 0, fraction & (unit >> 1) != 0,
                  fraction & ((unit >> 1) - 1) != 0)
    reinterpret(SoftFloat64, (u - fraction) + up * unit)
end


## narrower floating-point formats

@inline function SoftFloat64(x::T) where {T<:Union{Float16,Float32}}
    # Native comparisons and intermediate floating-point conversions may flush subnormals
    # to zero on the target. Classify and widen the original bits instead.
    magnitude = bits(x) & ~sign_mask(T)
    sign = bits(x) & sign_mask(T) != 0
    magnitude > exponent_mask(T) && return quiet_nan(SoftFloat64)
    magnitude == exponent_mask(T) && return infinity(SoftFloat64, sign)
    iszero(magnitude) && return signed_zero(SoftFloat64, sign)
    _, exp, sig = unpack(x)
    # exact, so the rounding bits are zero
    round_pack(SoftFloat64, sign, exp, UInt64(sig) << (62 - significand_bits(T)),
               RoundNearest)
end

@inline function narrow(::Type{T}, x::SoftFloat64, mode::RoundingMode) where {T<:Union{Float16,Float32}}
    isnan(x) && return quiet_nan(T)
    isinf(x) && return infinity(T, signbit(x))
    iszero(x) && return signed_zero(T, signbit(x))
    sign, exp, sig = unpack(x)
    # move the leading bit to where `round_pack` expects it, jamming the bits that fall off
    shift = significand_bits(SoftFloat64) - (8sizeof(T) - 2)
    round_pack(T, sign, exp, shift_right_jam(sig, shift) % uinttype(T), mode)
end
@inline Base.Float32(x::SoftFloat64, mode::RoundingMode=RoundNearest) = narrow(Float32, x, mode)
# Float16 is rounded directly: rounding through Float32 could round twice
@inline Base.Float16(x::SoftFloat64, mode::RoundingMode=RoundNearest) = narrow(Float16, x, mode)
