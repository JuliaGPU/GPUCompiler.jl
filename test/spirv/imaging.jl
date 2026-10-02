# Julia generates code in imaging mode when precompiling a package, which on 1.10 refers to
# `Bool` through `jl_small_typeof`, a global in the cross-workgroup address space. Force that
# mode in a separate process, because code coverage disables package images, and with them
# imaging mode during precompilation.
script = """
    using GPUCompiler, LLVM
    include($(repr(joinpath(@__DIR__, "..", "helpers", "runtime.jl"))))
    include($(repr(joinpath(@__DIR__, "..", "helpers", "spirv.jl"))))

    function kernel(out::Core.LLVMPtr{UInt,1})
        T = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), Bool)
        Base.unsafe_store!(out, UInt(T))
        return
    end

    job, _ = SPIRV.create_job(kernel, (Core.LLVMPtr{UInt,1},); backend=:llvm, kernel=true)
    JuliaContext() do ctx
        ir, _ = GPUCompiler.compile(:llvm, job)
        verify(ir)
    end
    """
# Base.julia_cmd() inherits coverage from the test runner, which conflicts with imaging mode.
cmd = `$(Base.julia_cmd()) --code-coverage=none --image-codegen --project=$(Base.active_project()) -e $script`
@test success(pipeline(cmd; stdout, stderr))
