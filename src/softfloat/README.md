# Software floating point

This directory implements IEEE 754 binary64 (`Float64`) support for GPU targets without
native double-precision arithmetic, using only 64-bit integer operations. Back-ends opt in
by returning a `SoftFloat64Provider` from `GPUCompiler.device_library_providers(job)`.

The emulation operates on ordinary `Float64` code in the final LLVM module, including
Base's pure-Julia math functions. The usual GPU restrictions still apply: external
floating-point library calls need backend overrides, and unsupported LLVM operations
are rejected. `legalize.jl` first outlines every floating-point operation on
`double` values into a call to one of the integer-only routines, and then rebuilds the
module with `double` replaced by `i64` everywhere (signatures, aggregates, constants,
globals, attributes). The routines are linked in afterwards, through the device-library
machinery in `../rtlib.jl`, from the table in `provider.jl`.

The routines themselves are the methods of `SoftFloat64`, a `primitive type` with the bit
representation of `Float64` that can be used as an ordinary number type on the CPU as well
(which is how `test/softfloat.jl` validates it, bit for bit against native arithmetic).
`binary64/types.jl` defines the type, using Base's format traits (`significand_bits`,
`exponent_mask`, ...), and `unpack`, which yields the sign, exponent and normalized
significand of a value. Every operation in `arithmetic.jl` and `conversions.jl` then
computes an exact or sticky-extended significand for its result and hands it to
`round_pack` in `rounding.jl`, which is where rounding happens: a single `round_up` decision
implements all `RoundingMode`s, for `round_pack`, the narrowing conversions to `Float32` and
`Float16`, and `round` to integral values. The operators default to round-to-nearest-even
but accept a trailing rounding mode, e.g. `+(a, b, RoundUp)`, which specializes at compile
time. `uint128.jl` provides the two-word `U128` that significand products need, since many
back-ends cannot lower `UInt128`.

The large routines (`+`, `*`, `/`, `sqrt`, `fma`) are `@noinline` so that they are shared
between call sites; the others are meant to be inlined.

Semantics, as used by the legalizer: round-to-nearest-even, gradual underflow, signed zeros,
infinities, quiet comparisons, and canonical quiet NaN arithmetic results
(`0x7ff8000000000000`). Sign operations preserve NaN payloads. Not supported: exception
flags, `Float64` atomics and `frem`
(Julia's `rem` does not use it). `Float64`→`Float16` rounds directly rather than through
`Float32`, which would double-round. Widening `Float16` and `Float32` uses their bits
directly, preserving subnormals even when native arithmetic flushes them to zero.

`paynehanek.jl` is unrelated to the emulation itself: it is a version of Base's Payne-Hanek
argument reduction (used by `sin`, `cos`, and `tan` for large arguments) that avoids the
`UInt128` arithmetic many GPU back-ends cannot lower. Back-ends should override
`Base.Math.paynehanek(::Float64)` with it.

## Package boundary

The numerical core is `uint128.jl` and `binary64/`; it depends only on Base and can move
into a standalone package. `provider.jl` and `legalize.jl` are GPUCompiler integration,
while `paynehanek.jl` is a backend workaround for Base's implementation.

`SoftFloat64` currently supplies the arithmetic and conversions needed by that integration,
plus basic numeric traits, promotion and hashing. A standalone package should complete
and test the remaining number interface (checked integer conversions, adjacent values,
parsing and broader promotion) before presenting it as a general replacement for `Float64`.

## Provenance

The algorithms in `binary64/` are those of
[Berkeley SoftFloat 3e](https://github.com/ucb-bar/berkeley-softfloat-3) (BSD-3-Clause),
including its reciprocal tables, by way of
[metal-softfloat](https://github.com/guyfischman/metal-softfloat)
(commit `8b6c592e2e383040fe2778bed8dda7904df284b1`, MIT). `paynehanek.jl` is adapted from
Julia's `base/special/rem_pio2.jl` (MIT). The notices are in `../../LICENSES/`.
