@testset "split_kwargs" begin
    kwargs = [:(a=1), :(b=2), :(c=3), :(d=4)]
    groups = GPUCompiler.split_kwargs(kwargs, [:a], [:b, :c])
    @test length(groups) == 3
    @test groups[1] == [:(a=1)]
    @test groups[2] == [:(b=2), :(c=3)]
    @test groups[3] == [:(d=4)]
end

@testset "mangling" begin
    using demumble_jll

    function mangle(f, argtyps...)
        mangled = GPUCompiler.mangle_sig(Tuple{typeof(f), argtyps...})
        chomp(read(`$(demumble_jll.demumble()) $mangled`, String))
    end

    # basic stuff
    @test mangle(identity) == "identity"
    @test mangle(identity, Nothing) == "identity()"

    # primitive types
    @test mangle(identity, Int32) == "identity(Int32)"
    @test mangle(identity, Int64) == "identity(Int64)"

    # literals
    @test mangle(identity, Val{1}) == "identity(Val<1>)"
    @test mangle(identity, Val{-1}) == "identity(Val<-1>)"
    @test mangle(identity, Val{Cshort(1)}) == "identity(Val<(short)1>)"
    @test mangle(identity, Val{1.0}) == "identity(Val<0x1p+0>)"
    @test mangle(identity, Val{1f0}) == "identity(Val<0x1p+0f>)"

    # unions
    @test mangle(identity, Union{Int32, Int64}) == "identity(Union<Int32, Int64>)"

    # union alls
    @test mangle(identity, Array) == "identity(Array<T, N>)"

    # many substitutions
    @test mangle(identity, Val{1}, Val{2}, Val{3}, Val{4}, Val{5}, Val{6}, Val{7}, Val{8},
                           Val{9}, Val{10}, Val{11}, Val{12}, Val{13}, Val{14}, Val{15},
                           Val{16}, Val{16}) ==
          "identity(Val<1>, Val<2>, Val<3>, Val<4>, Val<5>, Val<6>, Val<7>, Val<8>, Val<9>, Val<10>, Val<11>, Val<12>, Val<13>, Val<14>, Val<15>, Val<16>, Val<16>)"

    # problematic examples
    @test mangle(identity, String, Matrix{Float32}, Broadcast.Broadcasted{Broadcast.ArrayStyle{Matrix{Float32}}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(Base.literal_pow), Tuple{Base.RefValue{typeof(sin)}, Broadcast.Extruded{Matrix{Float32}, Tuple{Bool, Bool}, Tuple{Int64, Int64}}}}) == "identity(String, Array<Float32, 2>, Broadcasted<ArrayStyle<Array<Float32, 2>>, Tuple<OneTo<Int64>, OneTo<Int64>>, literal_pow, Tuple<RefValue<sin>, Extruded<Array<Float32, 2>, Tuple<Bool, Bool>, Tuple<Int64, Int64>>>>)"
end

@testset "highlighting" begin
    ansi_color = "\x1B[3"  # beginning of any foreground color change

    @testset "PTX" begin
        sample = """
            max.s64         %rd24, %rd18, 0;
            add.s64         %rd7, %rd23, -1;
            setp.lt.u64     %p2, %rd7, %rd24;
            @%p2 bra        \$L__BB0_3;
        """
        can_highlight = GPUCompiler.pygmentize_support("ptx")
        highlighted = sprint(GPUCompiler.highlight, sample, "ptx"; context = (:color => true))
        @test occursin(ansi_color, highlighted) skip = !can_highlight
    end

    @testset "GCN" begin
        sample = """
            v_add_u32     v3, vcc, s0, v0
            v_mov_b32     v4, s1
            v_addc_u32    v4, vcc, v4, 0, vcc
        """
        can_highlight = GPUCompiler.pygmentize_support("gcn")
        highlighted = sprint(GPUCompiler.highlight, sample, "gcn"; context = (:color => true))
        @test occursin(ansi_color, highlighted) skip = !can_highlight
    end
end
