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
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{false}})       == "_Z3sin2XXILb0EE"    # "sin(XX<false>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{true}})        == "_Z3sin2XXILb1EE"    # "sin(XX<true>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Cshort(10)}})  == "_Z3sin2XXILs10EE"   # "sin(XX<(short)10>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Cshort(0)}})   == "_Z3sin2XXILs0EE"    # "sin(XX<(short)l>)"
    @test GPUCompiler.mangle_sig(Tuple{typeof(sin), XX{Cshort(-10)}}) == "_Z3sin2XXILsn10EE"  # "sin(XX<(short)-10>)"
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
