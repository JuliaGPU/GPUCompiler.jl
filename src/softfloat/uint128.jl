# A two-word unsigned 128-bit integer, for the significand products of binary64 emulation.
#
# Many GPU back-ends cannot lower `UInt128` arithmetic, so the few operations the emulation
# needs are defined here on a pair of `UInt64`s, with the semantics of the corresponding
# `UInt128` operations: shifts by 128 bits or more yield zero, and negative shift counts
# shift the other way, as for Base's integers.

struct U128
    hi::UInt64
    lo::UInt64
end
U128(x::UInt64) = U128(0, x)

Base.zero(::Type{U128}) = U128(0, 0)
Base.iszero(a::U128) = (a.hi | a.lo) == 0
Base.:(==)(a::U128, b::U128) = (a.hi == b.hi) & (a.lo == b.lo)
Base.:<(a::U128, b::U128) = (a.hi < b.hi) | ((a.hi == b.hi) & (a.lo < b.lo))
Base.:<=(a::U128, b::U128) = !(b < a)

function Base.:+(a::U128, b::U128)
    lo = a.lo + b.lo
    U128(a.hi + b.hi + (lo < a.lo), lo)
end
function Base.:-(a::U128, b::U128)
    lo = a.lo - b.lo
    U128(a.hi - b.hi - (a.lo < b.lo), lo)
end
Base.:-(a::U128) = zero(U128) - a

# (the operators are not defined in terms of each other, which would make them recursive
# and thus prevent them from being inlined)
shift_left(a::U128, n::Int) =
    n < 64 ? U128((a.hi << n) | (a.lo >> (64 - n)), a.lo << n) : U128(a.lo << (n - 64), 0)
shift_right(a::U128, n::Int) =
    n < 64 ? U128(a.hi >> n, (a.lo >> n) | (a.hi << (64 - n))) : U128(0, a.hi >> (n - 64))
Base.:<<(a::U128, n::Int) = n >= 0 ? shift_left(a, n) : shift_right(a, -n)
Base.:>>(a::U128, n::Int) = n >= 0 ? shift_right(a, n) : shift_left(a, -n)

Base.leading_zeros(a::U128) = a.hi == 0 ? 64 + leading_zeros(a.lo) : leading_zeros(a.hi)

"""
    widemul(a, b)

The full product of two unsigned integers, like `Base.widemul`, except that the product of
two `UInt64`s is a `U128`.
"""
widemul(a::UInt32, b::UInt32) = UInt64(a) * UInt64(b)
function widemul(a::UInt64, b::UInt64)
    # schoolbook multiplication of 32-bit halves, as in Base's 32-bit `widemul` fallback
    a0, a1 = a & 0xffff_ffff, a >> 32
    b0, b1 = b & 0xffff_ffff, b >> 32
    p00 = a0 * b0
    mid = a1 * b0 + (p00 >> 32)
    p01 = a0 * b1 + (mid & 0xffff_ffff)
    U128(a1 * b1 + (mid >> 32) + (p01 >> 32), (p01 << 32) | (p00 & 0xffff_ffff))
end

"""
    shift_right_jam(a, n)

Shift `a` right by `n` bits, "jamming" any nonzero bits shifted out into the lowest bit of
the result, which thereby becomes a sticky bit for rounding. `n` may exceed the width of
`a`, in which case the result is `0` or `1`.
"""
@inline shift_right_jam(a::T, n::Int) where {T<:Unsigned} =
    n < 8sizeof(T) ? (a >> n) | T((a << (8sizeof(T) - n)) != 0) : T(a != 0)
@inline function shift_right_jam(a::U128, n::Int)
    n >= 128 && return U128(UInt64(!iszero(a)))
    shifted = a >> n
    U128(shifted.hi, shifted.lo | !iszero(a << (128 - n)))
end

# the top word of a 128-bit significand, with the rest jammed into its lowest bit
sticky_high(a::U128) = a.hi | (a.lo != 0)
