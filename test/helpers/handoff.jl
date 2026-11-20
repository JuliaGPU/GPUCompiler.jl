# the late hand-off (with GPUCompiler's `handoff` preference set to "late") uses the LLVM
# library registered as `:gpu`: a local build at `GPUCOMPILER_LIBLLVMGPU`, or LLVMGPU_jll's
if haskey(ENV, "GPUCOMPILER_LIBLLVMGPU")
    LLVM.register_library!(:gpu, ENV["GPUCOMPILER_LIBLLVMGPU"])
elseif Base.find_package("LLVMGPU_jll") !== nothing
    @eval using LLVMGPU_jll
end

# whether jobs for targets that support it are handed off, and optimized a second time,
# which changes some of the IR tests look at: e.g., the test runtimes' empty exception
# reporting functions are inlined then
const late_handoff = GPUCompiler.default_handoff() === :late
