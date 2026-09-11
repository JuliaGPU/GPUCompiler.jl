# Binary64 arithmetic: addition, multiplication, division, square root and fused
# multiply-add, following Berkeley SoftFloat 3e (BSD-3-Clause; see
# LICENSES/berkeley-softfloat-BSD-3-Clause.txt), by way of its port to Metal in
# metal-softfloat (MIT; see LICENSES/metal-softfloat-MIT.txt).
#
# Every operation first handles the special values, then computes a significand for the
# result that is either exact or extended with a sticky bit, and hands it to `round_pack`.
# The operators without a rounding-mode argument are the device library's entry points:
# they are `@noinline` so that the generated device code shares them between call sites.

Base.:-(x::SoftFloat64) = reinterpret(SoftFloat64, bits(x) ⊻ sign_mask(SoftFloat64))
Base.abs(x::SoftFloat64) = reinterpret(SoftFloat64, bits(x) & ~sign_mask(SoftFloat64))
Base.copysign(x::SoftFloat64, y::SoftFloat64) =
    reinterpret(SoftFloat64, (bits(x) & ~sign_mask(SoftFloat64)) | (bits(y) & sign_mask(SoftFloat64)))


## addition

Base.@noinline Base.:+(a::SoftFloat64, b::SoftFloat64) = +(a, b, RoundNearest)
Base.:-(a::SoftFloat64, b::SoftFloat64) = a + -b
Base.:-(a::SoftFloat64, b::SoftFloat64, mode::RoundingMode) = +(a, -b, mode)

@inline function Base.:+(a::SoftFloat64, b::SoftFloat64, mode::RoundingMode)
    if !isfinite(a) | !isfinite(b)
        (isnan(a) | isnan(b)) && return quiet_nan(SoftFloat64)
        (isinf(a) & isinf(b) & (signbit(a) != signbit(b))) && return quiet_nan(SoftFloat64)
        return isinf(a) ? a : b
    end
    if iszero(a) | iszero(b)
        (iszero(a) & iszero(b)) && return signbit(a) == signbit(b) ? a : cancellation_zero(mode)
        return iszero(a) ? b : a
    end

    sign, exp, sig = unpack(a)
    sign_b, exp_b, sig_b = unpack(b)
    # order the operands by magnitude, so that only the smaller one needs to be aligned
    if (exp, sig) < (exp_b, sig_b)
        sign, exp, sig, sign_b, exp_b, sig_b = sign_b, exp_b, sig_b, sign, exp, sig
    end
    # ten bits below the significands hold the rounding bits, and the sticky bit from aligning
    sig <<= 10
    sig_b = shift_right_jam(sig_b << 10, exp - exp_b)
    if sign == sign_b
        sum = sig + sig_b
        if sum >= one(UInt64) << 63
            sum = shift_right_jam(sum, 1)
            exp += 1
        end
        return round_pack(SoftFloat64, sign, exp, sum, mode)
    else
        diff = sig - sig_b
        iszero(diff) && return cancellation_zero(mode)
        # the leading bit of the difference can be anywhere; its lowest bit has weight
        # 2^(exp - 62), the significand's leading bit having been at 62
        return norm_round_pack(SoftFloat64, sign, exp - 62, diff, mode)
    end
end


## multiplication

Base.@noinline Base.:*(a::SoftFloat64, b::SoftFloat64) = *(a, b, RoundNearest)

@inline function Base.:*(a::SoftFloat64, b::SoftFloat64, mode::RoundingMode)
    sign = signbit(a) ⊻ signbit(b)
    if !isfinite(a) | !isfinite(b)
        (isnan(a) | isnan(b) | iszero(a) | iszero(b)) && return quiet_nan(SoftFloat64)
        return infinity(SoftFloat64, sign)
    end
    (iszero(a) | iszero(b)) && return signed_zero(SoftFloat64, sign)

    _, exp_a, sig_a = unpack(a)
    _, exp_b, sig_b = unpack(b)
    # the top word of the 106-bit product has its leading bit at 62 or 61, at weight
    # 2^(exp_a + exp_b + 1); the low word only contributes to the sticky bit
    sig = sticky_high(widemul(sig_a << 10, sig_b << 11))
    exp = exp_a + exp_b + 1
    if sig < one(UInt64) << 62
        sig <<= 1
        exp -= 1
    end
    round_pack(SoftFloat64, sign, exp, sig, mode)
end


## division

# Approximate reciprocals from the tables of Berkeley SoftFloat 3e: `a` is a fixed-point
# number with its leading bit set, with one integer bit (`1 <= a < 2`), or two for the
# square root when `two_integer_bits` is set (`2 <= a < 4`). The result is a pure fraction
# that never exceeds the true reciprocal, to within about 2 ulps.

const APPROX_RECIP_K0S = (0xffc4, 0xf0be, 0xe363, 0xd76f, 0xccad, 0xc2f0, 0xba16, 0xb201,
                          0xaa97, 0xa3c6, 0x9d7a, 0x97a6, 0x923c, 0x8d32, 0x887e, 0x8417)
const APPROX_RECIP_K1S = (0xf0f1, 0xd62c, 0xbfa1, 0xac77, 0x9c0a, 0x8ddb, 0x8185, 0x76ba,
                          0x6d3b, 0x64d4, 0x5d5c, 0x56b1, 0x50b6, 0x4b55, 0x4679, 0x4211)

@inline function approx_recip32(a::UInt32)
    index = (a >> 27) & 0xf
    eps = (a >> 11) % UInt16
    @inbounds r0 = APPROX_RECIP_K0S[index + 1] -
                   ((UInt32(APPROX_RECIP_K1S[index + 1]) * eps) >> 20) % UInt16
    sigma0 = ~((widemul(UInt32(r0), a) >> 7) % UInt32)
    r = (UInt32(r0) << 16) + (widemul(UInt32(r0), sigma0) >> 24) % UInt32
    sqr_sigma0 = (widemul(sigma0, sigma0) >> 32) % UInt32
    r + (widemul(r, sqr_sigma0) >> 48) % UInt32
end

const APPROX_RECIP_SQRT_K0S = (0xb4c9, 0xffab, 0xaa7d, 0xf11c, 0xa1c5, 0xe4c7, 0x9a43, 0xda29,
                               0x93b5, 0xd0e5, 0x8ded, 0xc8b7, 0x88c6, 0xc16d, 0x8424, 0xbae1)
const APPROX_RECIP_SQRT_K1S = (0xa5a5, 0xea42, 0x8c21, 0xc62d, 0x788f, 0xaa7f, 0x6928, 0x94b6,
                               0x5cc7, 0x8335, 0x52a6, 0x74e2, 0x4a3e, 0x68fe, 0x432b, 0x5efd)

@inline function approx_recip_sqrt32(a::UInt32, two_integer_bits::Bool)
    index = ((a >> 27) & 0xe) + !two_integer_bits
    eps = (a >> 12) % UInt16
    @inbounds r0 = APPROX_RECIP_SQRT_K0S[index + 1] -
                   ((UInt32(APPROX_RECIP_SQRT_K1S[index + 1]) * eps) >> 20) % UInt16
    e_sqr_r0 = UInt32(r0) * UInt32(r0)
    two_integer_bits && (e_sqr_r0 <<= 1)
    sigma0 = ~((widemul(e_sqr_r0, a) >> 23) % UInt32)
    r = (UInt32(r0) << 16) + (widemul(UInt32(r0), sigma0) >> 25) % UInt32
    sqr_sigma0 = (widemul(sigma0, sigma0) >> 32) % UInt32
    r += (widemul((r >> 1) + (r >> 3) - (UInt32(r0) << 14), sqr_sigma0) >> 48) % UInt32
    r & 0x8000_0000 == 0 ? UInt32(0x8000_0000) : r
end

Base.@noinline Base.:/(a::SoftFloat64, b::SoftFloat64) = /(a, b, RoundNearest)

@inline function Base.:/(a::SoftFloat64, b::SoftFloat64, mode::RoundingMode)
    sign = signbit(a) ⊻ signbit(b)
    if !isfinite(a) | !isfinite(b) | iszero(a) | iszero(b)
        (isnan(a) | isnan(b)) && return quiet_nan(SoftFloat64)
        ((isinf(a) & isinf(b)) | (iszero(a) & iszero(b))) && return quiet_nan(SoftFloat64)
        (isinf(a) | iszero(b)) && return infinity(SoftFloat64, sign)
        return signed_zero(SoftFloat64, sign)
    end

    _, exp_a, sig_a = unpack(a)
    _, exp_b, sig_b = unpack(b)
    # scale the dividend such that the quotient's leading bit lands at 62
    exp = exp_a - exp_b
    if sig_a < sig_b
        exp -= 1
        sig_a <<= 11
    else
        sig_a <<= 10
    end
    sig_b <<= 11

    # the upper half of the quotient from an approximate reciprocal of the divisor, then the
    # lower half from the remainder, which is finally used to correct the last few bits and
    # to set the sticky bit
    recip = approx_recip32((sig_b >> 32) % UInt32) - 0x2
    sig_b_hi = (sig_b >> 32) % UInt32
    sig_b_lo = (sig_b % UInt32) >> 4
    q_hi = (widemul((sig_a >> 32) % UInt32, recip) >> 32) % UInt32
    double_term = q_hi << 1
    rem = ((sig_a - widemul(double_term, sig_b_hi)) << 28) - widemul(double_term, sig_b_lo)
    q_lo = (widemul((rem >> 32) % UInt32, recip) >> 32) % UInt32 + 0x4
    sig = (UInt64(q_hi) << 32) + (UInt64(q_lo) << 4)
    if sig & 0x1ff < 0x40
        q_lo &= ~UInt32(7)
        sig &= ~UInt64(0x7f)
        double_term = q_lo << 1
        rem = ((rem - widemul(double_term, sig_b_hi)) << 28) - widemul(double_term, sig_b_lo)
        if rem & sign_mask(SoftFloat64) != 0
            sig -= 0x80
        elseif rem != 0
            sig |= 1
        end
    end
    round_pack(SoftFloat64, sign, exp, sig, mode)
end


## square root

Base.@noinline Base.sqrt(x::SoftFloat64) = sqrt(x, RoundNearest)

@inline function Base.sqrt(x::SoftFloat64, mode::RoundingMode)
    if !isfinite(x) | iszero(x) | signbit(x)
        (isnan(x) | (signbit(x) & !iszero(x))) && return quiet_nan(SoftFloat64)
        return x  # ±0 and +Inf
    end

    _, exp_a, sig_a = unpack(x)
    # halving an odd exponent leaves a factor of two in the significand
    exp = exp_a >> 1
    odd_exp = isodd(exp_a)

    # the upper half of the root from an approximate reciprocal square root, then the lower
    # half from the remainder, which is finally used to correct the last few bits and to
    # set the sticky bit
    sig32_a = (sig_a >> 21) % UInt32
    recip = approx_recip_sqrt32(sig32_a, odd_exp)
    sig32_z = (widemul(sig32_a, recip) >> 32) % UInt32
    if odd_exp
        sig_a <<= 9
    else
        sig_a <<= 8
        sig32_z >>= 1
    end
    rem = sig_a - widemul(sig32_z, sig32_z)
    q = (widemul((rem >> 2) % UInt32, recip) >> 32) % UInt32
    sig = ((UInt64(sig32_z) << 32) | (UInt64(1) << 5)) + (UInt64(q) << 3)
    if sig & 0x1ff < 0x22
        sig &= ~UInt64(0x3f)
        shifted = sig >> 6
        rem = (sig_a << 52) - shifted * shifted
        if rem & sign_mask(SoftFloat64) != 0
            sig -= 1
        elseif rem != 0
            sig |= 1
        end
    end
    round_pack(SoftFloat64, false, exp, sig, mode)
end


## fused multiply-add

Base.@noinline Base.fma(a::SoftFloat64, b::SoftFloat64, c::SoftFloat64) =
    fma(a, b, c, RoundNearest)

@inline function Base.fma(a::SoftFloat64, b::SoftFloat64, c::SoftFloat64, mode::RoundingMode)
    sign = signbit(a) ⊻ signbit(b)
    if !isfinite(a) | !isfinite(b) | !isfinite(c)
        (isnan(a) | isnan(b) | isnan(c)) && return quiet_nan(SoftFloat64)
        if isinf(a) | isinf(b)
            (iszero(a) | iszero(b)) && return quiet_nan(SoftFloat64)
            (isinf(c) & (signbit(c) != sign)) && return quiet_nan(SoftFloat64)
            return infinity(SoftFloat64, sign)
        end
        return c
    end
    if iszero(a) | iszero(b)
        # the product is an exact zero
        (iszero(c) & (signbit(c) != sign)) && return cancellation_zero(mode)
        return c
    end

    _, exp_a, sig_a = unpack(a)
    _, exp_b, sig_b = unpack(b)
    # the exact product, normalized to have its leading bit at 125, at weight 2^exp
    product = widemul(sig_a << 10, sig_b << 10)
    exp = exp_a + exp_b + 1
    if product.hi < one(UInt64) << 61
        product <<= 1
        exp -= 1
    end
    iszero(c) && return round_pack(SoftFloat64, sign, exp, sticky_high(product) << 1, mode)

    # align the addend's significand to the product's, shifting whichever is smaller
    sign_c, exp_c, sig_c = unpack(c)
    addend = U128(sig_c << 9, 0)
    if exp >= exp_c
        addend = shift_right_jam(addend, exp - exp_c)
    else
        product = shift_right_jam(product, exp_c - exp)
        exp = exp_c
    end

    if sign == sign_c
        sig = sticky_high(product + addend)  # leading bit at 62 or 61
        if sig >= one(UInt64) << 62
            exp += 1
        else
            sig <<= 1
        end
        return round_pack(SoftFloat64, sign, exp, sig, mode)
    else
        # the larger magnitude determines the sign; cancellation can move the leading bit
        # of the difference anywhere, so normalize from its lowest bit, at weight 2^(exp - 125)
        if product < addend
            sign = sign_c
            product, addend = addend, product
        end
        diff = product - addend
        iszero(diff) && return cancellation_zero(mode)
        return norm_round_pack(SoftFloat64, sign, exp - 125, diff, mode)
    end
end
