@testitem "util" begin

@testset "split_kwargs" begin
    kwargs = [:(a=1), :(b=2), :(c=3), :(d=4)]
    groups = GPUCompiler.split_kwargs(kwargs, [:a], [:b, :c])
    @test length(groups) == 3
    @test groups[1] == [:(a=1)]
    @test groups[2] == [:(b=2), :(c=3)]
    @test groups[3] == [:(d=4)]
end

@testset "mangle" begin
    struct XX{T} end
    # values checked with c++filt / cu++filt
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{false}})      == "_Z3sin2XXILb0EE"    # "sin(XX<false>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{true}})       == "_Z3sin2XXILb1EE"    # "sin(XX<true>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Int64(10)}})  == "_Z3sin2XXILl10EE"   # "sin(XX<10l>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Int64(0)}})   == "_Z3sin2XXILl0EE"    # "sin(XX<0l>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Int64(-10)}}) == "_Z3sin2XXILln10EE"  # "sin(XX<-10l>)"
end

end
