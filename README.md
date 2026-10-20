# GPUCompiler.jl

*Reusable compiler infrastructure for Julia GPU backends.*

| **Build Status**                                                                                   | **Coverage**                    |
|:--------------------------------------------------------------------------------------------------:|:-------------------------------:|
| [![][buildkite-img]][buildkite-url] [![][gha-img]][gha-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url] |

[buildkite-img]: https://badge.buildkite.com/512eb7dd35ca5b427ddf3240e2b4b3022f0c4f9925f1bdafa8.svg?branch=main
[buildkite-url]: https://buildkite.com/julialang/gpucompiler-dot-jl

[gha-img]: https://github.com/JuliaGPU/GPUCompiler.jl/actions/workflows/Test.yml/badge.svg?branch=main
[gha-url]: https://github.com/JuliaGPU/GPUCompiler.jl/actions?query=workflow%3ACI

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUCompiler.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUCompiler.html

[codecov-img]: https://codecov.io/gh/JuliaGPU/GPUCompiler.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/GPUCompiler.jl

This package offers reusable compiler infrastructure and tooling for
implementing GPU compilers in Julia. **It is not intended for end users!**
Instead, you should use one of the packages that builds on GPUCompiler.jl, such
as [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), [Metal.jl](https://github.com/JuliaGPU/Metal.jl), [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), or [OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl).


## Narrowing index arithmetic

Back-ends can opt into computing bounded 64-bit index expressions in 32 bits with
`CompilerConfig(...; narrow_indices=true)`. The pass runs after optimization and target
finalization, using LLVM range analyses and `llvm.assume` bounds. It leaves loop-affine
indices wide for strength reduction. It can also be run directly as
`GPUCompiler.IndexNarrowing.narrow_indices!(mod, tm)` (`tm` may be `nothing`).

The caller owns specialization policy: select meaningful axes, bin argument values, and
supply facts that hold for the selected bin. GPUCompiler neither inspects runtime values
nor chooses or guards variants. For example, after establishing bounds for a device array
argument with two dimensions:

```julia
ranges = (
    (),  # function object (captured fields can have bounds too)
    ((:dims, 1) => 0:typemax(Int32),
     (:dims, 2) => 0:typemax(Int32),
     (:len,) => 0:typemax(Int32)),
)
config = CompilerConfig(target, params; argument_ranges=ranges)
```

`argument_ranges` is an ordinary immutable tuple with one entry for the function and each
argument, including ghost arguments. Each entry contains `path => range` pairs. A path
uses physical field names or indices; `()` refers to the argument itself. Bounds are
nonempty integer `UnitRange`s whose endpoints fit the fixed-width integer leaf. Signed and
unsigned integers through 128 bits and `Bool` are supported. Properties, array indexing,
pointer dereferences, and stepped ranges are not supported. Ranges are never enumerated.

These facts are **caller assertions**. Passing incorrect facts can produce incorrect code.
GPUCompiler includes them in the configuration's cache identity alongside the existing
signature, world, target, and options. On a cache miss it resolves the field paths and emits
assumptions before argument ABI lowering. There is no runtime validation, type-based
specialization hook, or range metadata encoded in Julia type parameters.

An empty outer tuple, `argument_ranges=()`, supplies no additional facts. Nonempty ranges
enable narrowing by default; `narrow_indices` can be set independently. The caller decides
how many variants to create and how to reuse them safely. For example, a launcher can use
one aggregate “selected indices fit Int32” bin and one unbounded bin, reusing precomputed
range tuples instead of constructing metadata on each launch. A retained kernel must only
be invoked with values satisfying its facts; guards and fallback selection, if needed,
belong to the launcher. Type-only precompilation may supply facts under the same contract.

Device array accessors may additionally emit `LLVM.Interop.assume(1 <= index <= length(A))`
where out-of-bounds access is already undefined. These bounds allow the pass to narrow a
linearized index even when its intermediate products could overflow 32 bits. Arrays created
inside a kernel do not automatically inherit argument facts. The bounds do not select an
arithmetic width: narrowing currently targets 32 bits independently. Assumptions can also
inhibit single-use optimizations, so back-ends should measure their workloads before enabling
this.


## Development

GPUCompiler.jl's own test suite runs on GitHub Actions. In addition, Buildkite runs the test
suites of the GPU back-ends that build on GPUCompiler (CUDA.jl, AMDGPU.jl, Metal.jl, oneAPI.jl
and OpenCL.jl) against the version of GPUCompiler being tested. These back-end tests run on
dedicated hardware, of which there is little, so they are skipped where possible:

- On pull requests, each back-end is only tested when the changes can affect it. Changes to
  back-end-specific code (e.g. `src/metal.jl`) only test that back-end, changes to other
  files in `src/`, `ext/`, `Project.toml` or the CI configuration test all back-ends, and
  changes that only touch tests, documentation or GitHub workflows test none.
- Draft pull requests do not test any back-end, unless asked to by name (see below).
- Pushes to `main`, and tags, test all back-ends.

This selection can be overridden by adding one of the following tags to the message of the
last commit of a pull request:

| Tag                  | Effect                                                             |
|:-------------------- |:------------------------------------------------------------------ |
| `[only metal]`       | Test only the listed back-ends, regardless of the files changed, and also on draft pull requests. Multiple back-ends can be listed, e.g., `[only cuda, amdgpu]`. |
| `[skip amdgpu]`      | Do not test the listed back-ends, even when the changes affect them. |
| `[run all]`          | Test all back-ends, regardless of the files changed.               |
| `[skip tests]`       | Do not test any back-end.                                          |

Back-ends are named `cuda`, `amdgpu`, `metal`, `oneapi` and `opencl` (all lowercase), or by
their target: `ptx` for CUDA.jl, `gcn` for AMDGPU.jl, and `spirv` for both oneAPI.jl and
OpenCL.jl.
