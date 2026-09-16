# device-library provider for binary64 emulation

"""
    SoftFloat64Provider()

Device-library provider implementing IEEE 754 binary64 (`Float64`) arithmetic using only
64-bit integer operations, for targets without native double-precision support.

When selected for a job, all `double` values in the LLVM module are replaced by `i64` bit
patterns, and floating-point operations on them by calls to the [`SoftFloat64`](@ref)
routines (see `legalize_module!`). This is transparent to Julia code: ordinary `Float64`
code, including the pure-Julia math functions in Base, compiles as usual.

Supported semantics: round-to-nearest-even, gradual underflow, signed zeros, infinities,
quiet comparisons, and canonical quiet NaN results (`0x7ff8000000000000`). Not supported:
floating-point exception flags, other rounding modes, and `Float64` atomics.
"""
struct SoftFloat64Provider <: GPUCompiler.AbstractDeviceLibraryProvider end

# name of the LLVM function implementing one of the library's routines
helper_name(name::Symbol) = "gpu_softfloat_$name"

# The library's ABI exchanges binary64 values as `i64`, other floating-point values as
# their bits too, and `Bool` as `i8` like any Julia function. These shims adapt the
# SoftFloat64 methods where their signature differs.
rint(x::SoftFloat64) = round(x, RoundNearest)
round_ties_away(x::SoftFloat64) = round(x, RoundNearestTiesAway)
i64_to_f64(x::Int64) = SoftFloat64(x)
u64_to_f64(x::UInt64) = SoftFloat64(x)
f64_to_i64(x::SoftFloat64) = unsafe_trunc(Int64, x)
f64_to_u64(x::SoftFloat64) = unsafe_trunc(UInt64, x)
f16_to_f64(x::UInt16) = SoftFloat64(reinterpret(Float16, x))
f32_to_f64(x::UInt32) = SoftFloat64(reinterpret(Float32, x))
f64_to_f32(x::SoftFloat64) = reinterpret(UInt32, Float32(x))
f64_to_f16(x::SoftFloat64) = reinterpret(UInt16, Float16(x))

const METHODS = map([
        # arithmetic
        (:add64, +, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:mul64, *, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:div64, /, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:sqrt64, sqrt, SoftFloat64, (SoftFloat64,)),
        (:fma64, fma, SoftFloat64, (SoftFloat64, SoftFloat64, SoftFloat64)),
        (:neg64, -, SoftFloat64, (SoftFloat64,)),
        (:abs64, abs, SoftFloat64, (SoftFloat64,)),
        (:copysign64, copysign, SoftFloat64, (SoftFloat64, SoftFloat64)),
        # comparisons
        (:eq64, ==, Bool, (SoftFloat64, SoftFloat64)),
        (:lt64, <, Bool, (SoftFloat64, SoftFloat64)),
        (:le64, <=, Bool, (SoftFloat64, SoftFloat64)),
        (:unordered64, unordered, Bool, (SoftFloat64, SoftFloat64)),
        (:minimum64, min, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:maximum64, max, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:minnum64, minnum, SoftFloat64, (SoftFloat64, SoftFloat64)),
        (:maxnum64, maxnum, SoftFloat64, (SoftFloat64, SoftFloat64)),
        # rounding to integral values
        (:trunc64, trunc, SoftFloat64, (SoftFloat64,)),
        (:rint64, rint, SoftFloat64, (SoftFloat64,)),
        (:round64, round_ties_away, SoftFloat64, (SoftFloat64,)),
        (:floor64, floor, SoftFloat64, (SoftFloat64,)),
        (:ceil64, ceil, SoftFloat64, (SoftFloat64,)),
        # conversions
        (:i64_to_f64, i64_to_f64, SoftFloat64, (Int64,)),
        (:u64_to_f64, u64_to_f64, SoftFloat64, (UInt64,)),
        (:f64_to_i64, f64_to_i64, Int64, (SoftFloat64,)),
        (:f64_to_u64, f64_to_u64, UInt64, (SoftFloat64,)),
        (:f16_to_f64, f16_to_f64, SoftFloat64, (UInt16,)),
        (:f32_to_f64, f32_to_f64, SoftFloat64, (UInt32,)),
        (:f64_to_f32, f64_to_f32, UInt32, (SoftFloat64,)),
        (:f64_to_f16, f64_to_f16, UInt16, (SoftFloat64,)),
    ]) do (name, f, return_type, argument_types)
    GPUCompiler.DeviceLibraryMethod(f, return_type, argument_types;
                                    name, llvm_name=helper_name(name))
end

GPUCompiler.device_library_methods(::SoftFloat64Provider, ::GPUCompiler.CompilerJob) = METHODS

GPUCompiler.prepare_device_library!(::SoftFloat64Provider, ::GPUCompiler.CompilerJob,
                                    mod::LLVM.Module, entry::LLVM.Function) =
    legalize_module!(mod, entry)
