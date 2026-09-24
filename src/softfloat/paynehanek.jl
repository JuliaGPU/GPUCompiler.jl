# Payne-Hanek argument reduction without 128-bit integers.
#
# Adapted from Julia's base/special/rem_pio2.jl (MIT, see LICENSES/julia-MIT.txt), which
# uses `UInt128` arithmetic that many GPU back-ends cannot lower. This version replaces it
# by the two-word `U128`, and is otherwise identical, so that back-ends can override
# `Base.Math.paynehanek(::Float64)` with it to make Base's trigonometric functions compile
# for large arguments. The results are bit-for-bit equal to Base's.

# convert a signed 128-bit fraction to two Float64s, the first truncated to 26 bits
function fromfraction(f::U128)
    iszero(f) && return (0.0, 0.0)

    s = f.hi & sign_mask(Float64)  # sign bit
    x = s == 0 ? f : -f            # magnitude

    # 1. get leading term truncated to 26 bits
    n1 = 128 - leading_zeros(x)
    m1 = (x >> (n1 - 26)).lo << 27
    d1 = ((n1 - 128 + 1021) % UInt64) << 52
    z1 = reinterpret(Float64, s | (d1 + m1))

    # 2. compute remaining term
    x2 = x - (U128(m1) << (n1 - 53))
    iszero(x2) && return (z1, 0.0)
    n2 = 128 - leading_zeros(x2)
    m2 = (x2 >> (n2 - 53)).lo
    d2 = ((n2 - 128 + 1021) % UInt64) << 52
    z2 = reinterpret(Float64, s | (d2 + m2))
    return (z1, z2)
end

# see `Base.Math.paynehanek` for the derivation; only the arithmetic differs
function paynehanek(x::Float64)
    # 1. convert to form x = X * 2^k, where X is a 53-bit integer
    u = reinterpret(UInt64, x)
    X = (u & significand_mask(Float64)) | (one(UInt64) << significand_bits(Float64))
    raw_exponent = ((u & exponent_mask(Float64)) >> significand_bits(Float64)) % Int
    k = raw_exponent - exponent_bias(Float64) - significand_bits(Float64)

    # 2. extract the relevant 192-bit window of 1/2π (the caller guarantees |x| >= 2^20·π/2,
    #    so the table indices are in bounds, as in Base)
    idx = k >> 6
    shift = k - (idx << 6)
    INV_2PI = Base.Math.INV_2PI
    @inbounds if shift == 0
        a1 = INV_2PI[idx+1]
        a2 = INV_2PI[idx+2]
        a3 = INV_2PI[idx+3]
    else
        a1 = (idx < 0 ? zero(UInt64) : INV_2PI[idx+1] << shift) | (INV_2PI[idx+2] >> (64 - shift))
        a2 = (INV_2PI[idx+2] << shift) | (INV_2PI[idx+3] >> (64 - shift))
        a3 = (INV_2PI[idx+3] << shift) | (INV_2PI[idx+4] >> (64 - shift))
    end

    # 3. multiply, keeping the fractional 128 bits of the quotient after division by 2π
    w1 = U128(X * a1, 0)  # overflow becomes integer
    w2 = widemul(X, a2)
    w3 = widemul(X, a3) >> 64
    w = w1 + w2 + w3
    signbit(x) && (w = -w)

    # 4. convert to quadrant, and the quotient fraction after division by π/2
    q = ((w >> 125).lo % Int + 1) >> 1
    f = w << 2

    # 5. convert quotient fraction to split precision Float64
    z_hi, z_lo = fromfraction(f)

    # 6. multiply by π/2
    pio2 = 1.5707963267948966
    pio2_hi = 1.5707963407039642
    pio2_lo = -1.3909067614167116e-8
    y_hi = (z_hi + z_lo) * pio2
    y_lo = (((z_hi * pio2_hi - y_hi) + z_hi * pio2_lo) + z_lo * pio2_hi) + z_lo * pio2_lo
    return q, Base.Math.DoubleFloat64(y_hi, y_lo)
end
