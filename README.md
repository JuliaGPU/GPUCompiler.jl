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

### Device libraries

Back-ends select an ordered tuple with `GPUCompiler.device_libraries(job)`. Each
`DeviceLibrary(:name; methods=(DeviceLibraryMethod(...), ...))` describes Julia
specializations and their exported ABIs; `DeviceLibrary(:name; bitcode=path)` describes
vendor bitcode. GPUCompiler prepends its core runtime when Julia's runtime is unavailable.
`runtime_library(MyRuntime)` describes the target's runtime implementations using
GPUCompiler's signatures. Existing `runtime_module(job)` implementations remain supported.

The compiler keeps LLVM intrinsics through the main optimizer, then calls
`lower_intrinsics!(job, mod, entry)` after GC-frame lowering. This hook preserves exported
names and returns the current entry, including when it rebuilds functions. Libraries are
linked once in selection order: every live import must resolve to a later library, a
registered relocation, or a target intrinsic. Duplicate exports, backward dependencies,
and missing implementations produce errors naming the library and symbol. Unused Julia
libraries are not built. Overrides of the legacy `link_libraries!` hook retain their
early and late calls as an opaque compatibility boundary (some back-ends inspect globals
before optimization). Migrating to `device_libraries` removes that double linking;
early inspection belongs in `finish_module!`, not in a linker.

Unoptimized Julia libraries may still contain Julia runtime operations that later passes
remove. These, and missing optional target-runtime implementations, go through final IR
validation so an invalid kernel retains its original diagnostic and backtrace.

`finish_imports!(job, mod, imported_names)` applies target policy to late definitions.
Runtime-intrinsic and kernel-state lowering follow. Post-import inlining and cleanup run
after Julia's GC lowering, when aggregate loads are safe to inline. `libraries=false`
skips resolution and intrinsic lowering; `optimize=false, libraries=true` still resolves
libraries.

Julia libraries reuse the runtime's relocatable per-specialization cache and are lowered
before caching. `optimize=true` additionally optimizes the library, checks that it has no
Julia address spaces, runtime operations, relocations, mutable globals or indirect calls,
and marks exports for late inlining. It is intended for standalone numerical routines;
`always_inline=false` keeps larger routines, such as software division, out of line.
Unoptimized runtime libraries retain Julia's address spaces until the caller's pipeline
removes them. Library selection and target lowering policy must depend on compiler
configuration, which is also used for building library jobs.

Kernel caches record library dependencies at compilation, validate method identities and
CodeInstance world ranges on reuse, and fingerprint vendor bitcode by SHA-256. File bytes
and symbol manifests are cached using file metadata; changing a library selection or
replacing its bitcode invalidates dependent kernel results. Export names are part of the
per-function cache, so one specialization can supply multiple symbols.
Replace bitcode files atomically. Each compilation uses one content snapshot for export
discovery, linking, and dependency recording, even if a file is replaced during compilation.
Selection is cached per configuration and world; hooks must not consult mutable selection
state. File metadata is still checked on reuse, without rebuilding descriptors or scanning
Julia methods in an unchanged world. Alternating old and new worlds can invalidate a
shared kernel result repeatedly; this favors correctness over retaining multiple versions.
Byte-backed libraries copy and hash their input once; treat the descriptor's payload as
read-only and construct a new descriptor to replace it.

### Testing

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
