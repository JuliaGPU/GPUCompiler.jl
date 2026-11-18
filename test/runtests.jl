using ParallelTestRunner
using Test
import GPUCompiler, LLVM
using GPUCompiler, LLVM
using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll
using NVPTX_LLVM_Backend_jll
using AMDGPU_LLVM_Backend_jll

const init_code = quote
    using GPUCompiler, LLVM, LLVM.IR, LLVM.Build, LLVM.Passes, LLVM.ORC
    using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll
    using LLVMDowngrader_jll
    using NVPTX_LLVM_Backend_jll
    using AMDGPU_LLVM_Backend_jll
    using ScopedValues: with

    # include all helpers
    include(joinpath(@__DIR__, "helpers", "runtime.jl"))
    for file in readdir(joinpath(@__DIR__, "helpers"))
        if endswith(file, ".jl") && file != "runtime.jl"
            include(joinpath(@__DIR__, "helpers", file))
        end
    end
    using FileCheck
end

testsuite = find_tests(@__DIR__)
args = parse_args(ARGS)

# captured variables that get boxed (`Core.Box`) lose their type
if isdefined(Test, :detect_closure_boxes)
    testsuite["closure_boxes"] = quote
        @test Test.detect_closure_boxes(GPUCompiler, GPUCompiler.Runtime) == []
    end
end

if filter_tests!(testsuite, args)
    helperkeys = String[]
    for key in collect(keys(testsuite))
        startswith(key, "helpers/") && push!(helperkeys, key)
    end
    for key in helperkeys
        delete!(testsuite, key)
    end

    if LLVM.is_asserts()
        @warn "LLVM with assertions; skipping GCN tests"
        delete!(testsuite, "gcn")
    end
    if VERSION < v"1.11"
        @warn "Julia 1.11+ required for precompile tests; skipping"
        delete!(testsuite, "ptx/precompile")
        delete!(testsuite, "native/precompile")
    end
    if !SPIRV_LLVM_Backend_jll.is_available() || !SPIRV_LLVM_Translator_jll.is_available() || !SPIRV_Tools_jll.is_available()
        @warn "SPIRV back-end not available; skipping SPIRV tests"
        for key in collect(keys(testsuite))
            startswith(key, "spirv") && delete!(testsuite, key)
        end
    end
    if !NVPTX_LLVM_Backend_jll.is_available()
        @warn "NVPTX back-end not available; skipping PTX tests"
        for key in collect(keys(testsuite))
            startswith(key, "ptx") && delete!(testsuite, key)
        end
    end
    if !AMDGPU_LLVM_Backend_jll.is_available()
        @warn "AMDGPU back-end not available; skipping GCN tests"
        delete!(testsuite, "gcn")
    end
    if Sys.WORD_SIZE == 32
        # Metal and BPF targets are 64-bit only, and Julia's codegen assumes pointers as
        # large as the host's
        @warn "32-bit host; skipping Metal and BPF tests"
        for key in collect(keys(testsuite))
            (startswith(key, "metal") || startswith(key, "bpf")) && delete!(testsuite, key)
        end
    end
end

runtests(GPUCompiler, args; testsuite, init_code)
