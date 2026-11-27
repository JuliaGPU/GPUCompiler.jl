# whether GPUCompiler uses the `llvmgpu` toolchain (see src/toolchain.jl), selected by its
# `toolchain` preference: top-level jobs of every target but the host are optimized with
# libLLVMGPU, using JuliaIR.jl's ports of Julia's passes, so their IR differs in places
const llvmgpu = GPUCompiler.TOOLCHAIN === :llvmgpu

# whether the IR of top-level GPU jobs uses typed pointers, for conditional FileCheck checks:
# with the llvmgpu toolchain, it belongs to libLLVMGPU, which only supports opaque pointers
const gpu_typed_ptrs = typed_ptrs && !llvmgpu
const gpu_opaque_ptrs = !gpu_typed_ptrs

# the LLVM library of the IR that top-level GPU jobs return, e.g. to parse its text again
gpu_library() = llvmgpu ? GPUCompiler.llvmgpu_state().library : LLVM.Library(:julia)
