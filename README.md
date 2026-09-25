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
