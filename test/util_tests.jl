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
            end
            Threads.@spawn Logging.with_logger(YieldingLogger()) do
                sleep(0.1)
                f()
            end
        end
        true
    end
end

end
