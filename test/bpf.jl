@testset "eBPF" begin

include("definitions/bpf.jl")

############################################################################################

@testset "No-op" begin
    kernel() = 0

    output = bpf_code_native(kernel, ())[1]
    @test occursin("\tr0 = 0\n\texit", output)
end
@testset "Return argument" begin
    kernel(x) = x

    output = bpf_code_native(kernel, (UInt64,); strip=true)[1]
    @test occursin("\tr0 = r1\n\texit", output)
end
@testset "Addition" begin
    kernel(x) = x+1

    output = bpf_code_native(kernel, (UInt64,); strip=true)[1]
    @test occursin("\tr0 = r1\n\tr0 += 1\n\texit", output)
end

end
