using ParallelTestRunner
import GPUCompiler, LLVM
using GPUCompiler, LLVM
# JuliaCodegen.jl (Julia 1.12 and later), when the environment provides it: loading it
# changes nothing unless it is selected (with the `codegen` preference, or per test)
const HAVE_JULIACODEGEN = VERSION >= v"1.12" && Base.find_package("JuliaCodegen") !== nothing
HAVE_JULIACODEGEN && @eval using JuliaCodegen
using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll
using NVPTX_LLVM_Backend_jll
using AMDGPU_LLVM_Backend_jll

const init_code = quote
    using GPUCompiler, LLVM, LLVM.IR, LLVM.Build, LLVM.Passes, LLVM.ORC
    $(HAVE_JULIACODEGEN ? :(using JuliaCodegen) : nothing)
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

if filter_tests!(testsuite, args)
    helperkeys = String[]
    for key in collect(keys(testsuite))
        startswith(key, "helpers/") && push!(helperkeys, key)
    end
    for key in helperkeys
        delete!(testsuite, key)
    end

    if !HAVE_JULIACODEGEN
        @warn "JuliaCodegen.jl not available; skipping its tests"
        delete!(testsuite, "juliacodegen")
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
end

runtests(GPUCompiler, args; testsuite, init_code)
