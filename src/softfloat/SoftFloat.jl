# software floating-point support for targets without native double precision

module SoftFloat

using LLVM
import ..GPUCompiler
using Base: sign_mask, exponent_mask, significand_mask, exponent_one, significand_bits,
            exponent_bits, exponent_bias, exponent_raw_max, uinttype

include("uint128.jl")
include("binary64/types.jl")
include("binary64/rounding.jl")
include("binary64/arithmetic.jl")
include("binary64/comparisons.jl")
include("binary64/conversions.jl")
include("paynehanek.jl")
include("provider.jl")
include("legalize.jl")

export SoftFloat64, SoftFloat64Provider

end
