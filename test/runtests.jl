using ParallelTestRunner 

const init_code = quote
    using Test, GPUCompiler, LLVM
    using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

    # include all helpers
    include(joinpath(@__DIR__, "helpers", "runtime.jl"))
    for file in readdir(joinpath(@__DIR__, "helpers"))
        if endswith(file, ".jl") && file != "runtime.jl"
            include(joinpath(@__DIR__, "helpers", file))
        end
    end
    using .FileCheck
end

function testfilter(test)
    if startswith(test, "helpers/")
        return false
    end
    return true
end

runtests(ARGS; init_code, testfilter)
