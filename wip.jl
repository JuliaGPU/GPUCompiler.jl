using GPUCompiler

include("test/definitions/native.jl")

original() = 1

using Base.Experimental: @overlay

# TODO: short function def
@overlay GPUCompiler.mt function original()
    2
end

kernel() = original()

function main()
    @show kernel()

    empty!(GPUCompiler.CI_CACHE.dict)
    native_code_llvm(kernel, Tuple{}; debuginfo=:none)
end

isinteractive() || main()
