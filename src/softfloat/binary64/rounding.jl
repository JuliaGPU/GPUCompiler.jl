# Rounding: the decision shared by all operations, parameterized on the rounding mode, and
# the rounding and packing of results.

# rounding-mode predicates (Base.Rounding has similar ones from Julia 1.11)
rounds_to_nearest(::RoundingMode) = false
rounds_to_nearest(::RoundingMode{:Nearest}) = true
rounds_to_nearest(::RoundingMode{:NearestTiesAway}) = true
rounds_to_nearest(::RoundingMode{:NearestTiesUp}) = true
round_tie(::RoundingMode{:Nearest}, sign::Bool, odd::Bool) = odd
round_tie(::RoundingMode{:NearestTiesAway}, sign::Bool, odd::Bool) = true
round_tie(::RoundingMode{:NearestTiesUp}, sign::Bool, odd::Bool) = !sign
rounds_away_from_zero(::RoundingMode{:ToZero}, sign::Bool) = false
rounds_away_from_zero(::RoundingMode{:FromZero}, sign::Bool) = true
rounds_away_from_zero(::RoundingMode{:Up}, sign::Bool) = !sign
rounds_away_from_zero(::RoundingMode{:Down}, sign::Bool) = sign
overflows_to_infinity(mode::RoundingMode, sign::Bool) =
    rounds_to_nearest(mode) || rounds_away_from_zero(mode, sign)

# the zero produced by an exact cancellation, whose sign depends on the rounding mode
cancellation_zero(::RoundingMode) = signed_zero(SoftFloat64, false)
cancellation_zero(::RoundingMode{:Down}) = signed_zero(SoftFloat64, true)

"""
    round_up(mode, sign, odd, half, rest) -> Bool

Whether rounding a truncated magnitude must increment its lowest kept bit, given that bit
(`odd`), the highest discarded bit (`half`), and whether any lower discarded bit is set
(`rest`).
"""
@inline function round_up(mode::RoundingMode, sign::Bool, odd::Bool, half::Bool, rest::Bool)
    if rounds_to_nearest(mode)
        half & (rest | round_tie(mode, sign, odd))
    else
        rounds_away_from_zero(mode, sign) & (half | rest)
    end
end

"""
    round_pack(T, sign, exp, sig, mode) -> T

Round `±sig * 2^(exp - (8sizeof(T) - 2))` to format `T`: `sig::uinttype(T)` has its leading
bit at position `8sizeof(T) - 2` (so `exp` is the binary exponent of the value), followed by
the significand, and then `exponent_bits(T) - 1` rounding bits, the lowest of which must be
sticky, i.e. set whenever any lower-order bit of the exact result was dropped.
"""
@inline function round_pack(::Type{T}, sign::Bool, exp::Int, sig::U,
                            mode::RoundingMode) where {T, U<:Unsigned}
    nbits = 8sizeof(U)
    nround = nbits - 2 - significand_bits(T)
    biased = exp + exponent_bias(T)
    if biased < 1
        # subnormal: scale to the subnormal grid before rounding, to avoid rounding twice
        sig = shift_right_jam(sig, 1 - biased)
        biased = 1
    end
    odd = sig & (one(U) << nround) != 0
    half = sig & (one(U) << (nround - 1)) != 0
    rest = sig & ((one(U) << (nround - 1)) - one(U)) != 0
    sig = (sig >> nround) + round_up(mode, sign, odd, half, rest)

    # The rounded significand still contains its leading bit, which adds one to the exponent
    # field when packed. A subnormal result, whose leading bit is lower, thus leaves the field
    # at zero, and a carry out of the rounding increments the field, as required.
    field = biased - 1 + (sig >> significand_bits(T)) % Int
    if field >= exponent_raw_max(T)
        return overflows_to_infinity(mode, sign) ? infinity(T, sign) : largest_finite(T, sign)
    end
    reinterpret(T, (U(sign) << (nbits - 1)) + (((biased - 1) % U) << significand_bits(T)) + sig)
end

"""
    norm_round_pack(T, sign, exp, sig, mode) -> T

Like `round_pack`, for a nonzero `sig` (a `UInt64` or `U128`) whose leading bit can be
anywhere, and whose lowest bit has weight `2^exp`.
"""
@inline function norm_round_pack(::Type{T}, sign::Bool, exp::Int, sig,
                                 mode::RoundingMode) where {T}
    shift = leading_zeros(sig) - 1
    sig = shift >= 0 ? sig << shift : shift_right_jam(sig, -shift)
    round_pack(T, sign, exp + 8sizeof(sig) - 2 - shift, narrow(sig), mode)
end
@inline narrow(sig::UInt64) = sig
@inline narrow(sig::U128) = sticky_high(sig)
