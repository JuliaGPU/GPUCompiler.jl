using ParallelTestRunner
import GPUCompiler, LLVM

const init_code = quote
    using GPUCompiler, LLVM
    using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

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

    if LLVM.is_asserts()
        delete!(testsuite, "gcn")
    end
    if VERSION < v"1.11"
        delete!(testsuite, "ptx/precompile")
        delete!(testsuite, "native/precompile")
    end
end

runtests(GPUCompiler, args; testsuite, init_code)
