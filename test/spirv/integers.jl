# legalization of nonstandard integer widths by the LLVM SPIR-V back-end

@testset "narrow integer switch" begin
    # LLVM folds a mask followed by a small switch into a switch on a narrow integer, here
    # `switch i2 (trunc i64 %x to i2)`. The SPIR-V back-end legalizes that truncation to a
    # plain `OpUConvert` to 8 bits, which does not drop the high bits: selector values
    # outside 0:3 then take the default case. This breaks e.g. the quadrant dispatch of
    # Base's `sin`, `cos`, `sinpi` and `cospi` for larger arguments (observed with PoCL).
    # The truncation should become `OpBitwiseAnd` (or the switch should be on the wide type).
    function quadrant(x::Int64)
        n = x & 3
        n == 0 ? 10 : n == 1 ? 20 : n == 2 ? 30 : 40
    end
    job, _ = SPIRV.create_job(quadrant, (Int64,); backend=:llvm)
    ir, asm = GPUCompiler.JuliaContext() do ctx
        string(first(GPUCompiler.compile(:llvm, job))), first(GPUCompiler.compile(:asm, job))
    end
    if occursin(r"switch i2 ", ir)
        selector = match(r"OpSwitch (%\w+)", asm)
        @test selector !== nothing
        definition = match(Regex("$(selector[1]) = (Op\\w+)"), asm)
        @test definition !== nothing
        @test_broken definition[1] != "OpUConvert"
    end
end
