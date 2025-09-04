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
    @test mangle(identity, Val{'a'}) == "identity(Val<97u>)"
    @test mangle(identity, Val{'∅'}) == "identity(Val<8709u>)"

    # unions
    @test mangle(identity, Union{Int32, Int64}) == "identity(Union<Int32, Int64>)"
    @test mangle(identity, Union, Int, Union{Int64, Int32}) == "identity(Union, Int64, Union<Int32, Int64>)"

    # union alls
    @test mangle(identity, Array) == "identity(Array<T, N>)"
    @test mangle(identity, Tuple) == "identity(Tuple)"
    @test mangle(identity, Vector) == "identity(Array<T, 1>)"
    @test mangle(identity, NTuple{2, I} where {I <: Integer}) == "identity(Tuple<I__Integer, I__Integer>)"

    # Vararg
    @test mangle(identity, Vararg{Int}) == "identity(Int64, ...)"
    @test mangle(identity, Vararg{Int, 2}) == "identity(Int64, Int64)"
    @test mangle(identity, Tuple{1, 2}, Tuple{}, Tuple) == "identity(Tuple<1, 2>, Tuple<>, Tuple)"
    @test mangle(identity, NTuple{2, Int}) == "identity(Tuple<Int64, Int64>)"
    @test mangle(identity, Tuple{Vararg{Int}}) == "identity(Tuple<>)"

    # many substitutions
    @test mangle(identity, Val{1}, Val{2}, Val{3}, Val{4}, Val{5}, Val{6}, Val{7}, Val{8},
                           Val{9}, Val{10}, Val{11}, Val{12}, Val{13}, Val{14}, Val{15},
                           Val{16}, Val{16}) ==
          "identity(Val<1>, Val<2>, Val<3>, Val<4>, Val<5>, Val<6>, Val<7>, Val<8>, Val<9>, Val<10>, Val<11>, Val<12>, Val<13>, Val<14>, Val<15>, Val<16>, Val<16>)"

    # intertwined substitutions
    @test mangle(
        identity, Val{1}, Ptr{Tuple{Ptr{Int}}}, Ptr{Int}, Val{1}, Val{2},
        Tuple{Ptr{Int}}, Tuple{Int8}, Int64, Int8
    ) ==
        "identity(Val<1>, Tuple<Int64*>*, Int64*, Val<1>, Val<2>, Tuple<Int64*>, Tuple<Int8>, Int64, Int8)"

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


import GPUCompiler: StackedMethodTable
import Core.Compiler: findsup, findall, isoverlayed

Base.Experimental.@MethodTable(LayerMT)
Base.Experimental.@MethodTable(OtherMT)

OverlayMT() = Core.Compiler.OverlayMethodTable(Base.get_world_counter(), LayerMT)
StackedMT() = StackedMethodTable(Base.get_world_counter(), LayerMT)
DoubleStackedMT() = StackedMethodTable(Base.get_world_counter(), OtherMT, LayerMT)

@testset "StackedMethodTable -- Unoverlayed" begin
    if VERSION >= v"1.11.0-DEV.363"
        @test isoverlayed(OverlayMT()) == true
        @test isoverlayed(StackedMT()) == true
        @test isoverlayed(DoubleStackedMT()) == true
    end

    o_sin  = findsup(Tuple{typeof(sin), Float64}, OverlayMT())
    s_sin  = findsup(Tuple{typeof(sin), Float64}, StackedMT())
    ss_sin = findsup(Tuple{typeof(sin), Float64}, DoubleStackedMT())
    @test s_sin == o_sin
    @test ss_sin == o_sin

    o_sin  = findall(Tuple{typeof(sin), Float64}, OverlayMT())
    s_sin  = findall(Tuple{typeof(sin), Float64}, StackedMT())
    ss_sin = findall(Tuple{typeof(sin), Float64}, DoubleStackedMT())
    if VERSION >= v"1.11.0-DEV.363"
        @test o_sin.matches == s_sin.matches
        @test o_sin.matches == ss_sin.matches
    else
        @test o_sin.matches.matches == s_sin.matches.matches
        @test o_sin.matches.matches == ss_sin.matches.matches
        @test o_sin.overlayed == s_sin.overlayed
        @test o_sin.overlayed == ss_sin.overlayed
        @test o_sin.overlayed == false
    end
end

# Note: This must be a top-level otherwise the tests below will not see the new function.
prev_world = Base.get_world_counter()
Base.Experimental.@overlay LayerMT function Base.sin(x::Float64) end
next_world = Base.get_world_counter()

@test next_world > prev_world

@testset "StackedMethodTable -- Overlayed" begin
    o_sin = findsup(Tuple{typeof(sin), Float64}, OverlayMT())
    s_sin = findsup(Tuple{typeof(sin), Float64}, StackedMT())
    ss_sin = findsup(Tuple{typeof(sin), Float64}, DoubleStackedMT())
    @test s_sin == o_sin
    @test ss_sin == o_sin

    worlds = o_sin[2]
    @test worlds.min_world > prev_world
    @test worlds.max_world == typemax(typeof(next_world))

    o_sin  = findall(Tuple{typeof(sin), Float64}, OverlayMT())
    s_sin  = findall(Tuple{typeof(sin), Float64}, StackedMT())
    ss_sin = findall(Tuple{typeof(sin), Float64}, DoubleStackedMT())
    if VERSION >= v"1.11.0-DEV.363"
        @test o_sin.matches == s_sin.matches
        @test o_sin.matches == ss_sin.matches
    else
        @test o_sin.matches.matches == s_sin.matches.matches
        @test o_sin.matches.matches == ss_sin.matches.matches
        @test o_sin.overlayed == s_sin.overlayed
        @test o_sin.overlayed == ss_sin.overlayed
        @test o_sin.overlayed == true
    end
end

# Test FileCheck
@testset "FileCheck" begin
    @test @filecheck begin
        check"CHECK: works"
        println("works")
    end

    @test_throws "expected string not found in input" @filecheck begin
        check"CHECK: works"
        println("doesn't work")
    end

    @test @filecheck begin
        check"CHECK: errors"
        error("errors")
    end

    @test_throws "expected string not found in input" @filecheck begin
        check"CHECK: works"
        error("errors")
    end
end

@testset "Mock Enzyme" begin
    Enzyme.deferred_codegen_id(typeof(identity), Tuple{Vector{Float64}})
    # Check that we can call this function from the CPU, to support deferred codegen for Enzyme.
    @test ccall("extern deferred_codegen", llvmcall, UInt, (UInt,), 3) == 3
end
