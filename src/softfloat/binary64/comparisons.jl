# Binary64 comparisons and extrema, on the bit patterns.

Base.:(==)(a::SoftFloat64, b::SoftFloat64) =
    !(isnan(a) | isnan(b)) & ((bits(a) == bits(b)) | (iszero(a) & iszero(b)))

function Base.:<(a::SoftFloat64, b::SoftFloat64)
    (isnan(a) | isnan(b)) && return false
    sign_a, sign_b = signbit(a), signbit(b)
    if sign_a != sign_b
        sign_a & !(iszero(a) & iszero(b))
    else
        # the magnitudes order like their bit patterns, in reverse for negative values
        (bits(a) != bits(b)) & (sign_a ⊻ (bits(a) < bits(b)))
    end
end

function Base.:<=(a::SoftFloat64, b::SoftFloat64)
    (isnan(a) | isnan(b)) && return false
    sign_a, sign_b = signbit(a), signbit(b)
    if sign_a != sign_b
        sign_a | (iszero(a) & iszero(b))
    else
        (bits(a) == bits(b)) | (sign_a ⊻ (bits(a) < bits(b)))
    end
end

unordered(a::SoftFloat64, b::SoftFloat64) = isnan(a) | isnan(b)

# `min` and `max` propagate NaNs and order -0.0 below +0.0, like Base's for Float64 and
# LLVM's `minimum` and `maximum` intrinsics
function Base.min(a::SoftFloat64, b::SoftFloat64)
    unordered(a, b) && return quiet_nan(SoftFloat64)
    (iszero(a) & iszero(b)) && return reinterpret(SoftFloat64, bits(a) | bits(b))
    a < b ? a : b
end
function Base.max(a::SoftFloat64, b::SoftFloat64)
    unordered(a, b) && return quiet_nan(SoftFloat64)
    (iszero(a) & iszero(b)) && return reinterpret(SoftFloat64, bits(a) & bits(b))
    a < b ? b : a
end

# LLVM's `minnum` and `maxnum` intrinsics return the number when only one operand is NaN
minnum(a::SoftFloat64, b::SoftFloat64) = isnan(a) ? (isnan(b) ? quiet_nan(SoftFloat64) : b) :
                                         isnan(b) ? a : min(a, b)
maxnum(a::SoftFloat64, b::SoftFloat64) = isnan(a) ? (isnan(b) ? quiet_nan(SoftFloat64) : b) :
                                         isnan(b) ? a : max(a, b)
