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


## LLVM toolchain

By default, GPUCompiler optimizes code with Julia's own LLVM, including the LLVM passes of
Julia's code generator, and generates code with back-end libraries (e.g.
NVPTX_LLVM_Backend_jll). The experimental `llvmgpu` toolchain instead moves the code of
every top-level job for a GPU target (PTX, GCN, SPIR-V, Metal and BPF) to libLLVMGPU, a
newer, standalone LLVM from LLVMGPU_jll, once it is linked and post-processed (e.g. by
`finish_linked_module!`), and optimizes it there with
[JuliaIR.jl](https://github.com/JuliaLLVM/JuliaIR.jl)'s ports of Julia's passes.
libLLVMGPU then also generates code for the PTX, GCN, SPIR-V and BPF targets, except for
the alternative code generators: libNVVM (`backend=:nvvm`), which gets the optimized code
downgraded with LLVMDowngrader_jll, and the Khronos SPIR-V translator (`backend=:khronos`).
Metal code is optimized there too, and downgraded to AIR as before. The toolchain is
selected with a preference, which takes effect after restarting Julia:

```julia
using Preferences, GPUCompiler
set_preferences!(GPUCompiler, "toolchain" => "llvmgpu")
```

The IR that `GPUCompiler.compile` returns for jobs that moved to libLLVMGPU (the module,
`meta.ir` and `meta.entry`) then belongs to that library, and LLVM.jl only works with it
in its own context. The do-block form of `compile` takes care of that, with either
toolchain: it compiles in contexts of its own, runs the block in the context of the IR,
and frees both afterwards.

```julia
entry = GPUCompiler.compile(:obj, job) do obj, meta
    meta.entry.name
end
```

Code that calls `compile(target, job)` inside a `JuliaContext() do ... end` block, which
owns and frees the contexts, has to work with the IR inside
`LLVM.context!(LLVM.context(ir)) do ... end`. (As with Julia's LLVM, the IR belongs to the
context of libLLVMGPU that was activated last, normally the block's.) Hooks that run during or after optimization
run in that context too: e.g. `optimize_module!`, `finish_ir!`, and `link_libraries!`,
which optimization can call again after it ran with Julia's LLVM. `code_llvm` prints that
IR with LLVM's printer, which shows all metadata.

Code for other targets, like the host's or ones defined elsewhere, stays with Julia's LLVM.
Targets defined elsewhere can move to libLLVMGPU too, by defining
`GPUCompiler.llvm_library(::MyTarget) = GPUCompiler.toolchain_library()`.
`GPUCompiler.backend_llvm_version(target)` tells which LLVM generates the machine code.

To run the test suite with the llvmgpu toolchain, set the preference in
`test/LocalPreferences.toml` (on Julia 1.10 and 1.11, in `LocalPreferences.toml`).
