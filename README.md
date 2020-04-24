# GPUCompiler.jl

*Reusable compiler infrastructure for Julia GPU backends.*

| **Build Status**                                                                                   | **Coverage**                    |
|:--------------------------------------------------------------------------------------------------:|:-------------------------------:|
| [![][gitlab-img]][gitlab-url] [![][travis-img]][travis-url] [![PkgEval][pkgeval-img]][pkgeval-url] | [![][codecov-img]][codecov-url] |

[gitlab-img]: https://gitlab.com/JuliaGPU/GPUCompiler.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/GPUCompiler.jl/commits/master

[travis-img]: https://api.travis-ci.com/JuliaGPU/GPUCompiler.jl.svg?branch=master
[travis-url]: https://travis-ci.com/JuliaGPU/GPUCompiler.jl

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUCompiler.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/G/GPUCompiler.html

[codecov-img]: https://codecov.io/gh/JuliaGPU/GPUCompiler.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/GPUCompiler.jl

This package offers reusable compiler infrastructure and tooling for
implementing GPU compilers in Julia. **It is not intended for end users!**
Instead, you should use one of the packages that builds on GPUCompiler.jl, such
as [CUDAnative](https://github.com/JuliaGPU/CUDAnative.jl).
