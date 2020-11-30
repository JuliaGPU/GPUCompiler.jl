using GPUCompiler

include("test/definitions/native.jl")

original() = :original_value

kernel() = original()

replaced() = :replaced_value
GPUCompiler.CI_CACHE.overrides[typeof(original)] = [typeof(replaced)]

function main()
    @show kernel()

    empty!(GPUCompiler.CI_CACHE.dict)
    native_code_llvm(kernel, Tuple{}; debuginfo=:none)
end

isinteractive() || main()
