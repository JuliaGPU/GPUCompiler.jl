@testitem "util" begin

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

@testset "safe loggers" begin
    using Logging: Logging

    struct YieldingLogger <: Logging.AbstractLogger
        logger::Logging.AbstractLogger
        YieldingLogger() = new(Logging.current_logger())
    end

    function Logging.handle_message(logger::YieldingLogger, args...)
        yield()
        return Logging.handle_message(logger.logger, args...)
    end

    Logging.shouldlog(::YieldingLogger, ::Any...) = true
    Logging.min_enabled_level(::YieldingLogger) = Logging.Debug

    GPUCompiler.@locked function f()
        GPUCompiler.@safe_debug "safe_debug"
        GPUCompiler.@safe_info "safe_info"
        GPUCompiler.@safe_warn "safe_warn"
        GPUCompiler.@safe_error "safe_error"
        GPUCompiler.@safe_show "safe_show"
    end

    @test begin
        @sync begin
            Threads.@spawn begin
                sleep(0.1)
                @debug "debug"
                sleep(0.1)
                @info "info"
                sleep(0.1)
                @warn "warn"
                sleep(0.1)
                @error "error"
                sleep(0.1)
                @show "show"
                sleep(0.1)
            end
            pipe = Pipe()
            Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
            Threads.@spawn print(stdout, read(pipe, String))
            Threads.@spawn Logging.with_logger(YieldingLogger()) do
                sleep(0.1)
                redirect_stdout(f, pipe)
                close(pipe)
            end
        end
        true
    end
end

end
