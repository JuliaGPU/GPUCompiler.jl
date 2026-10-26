@testset "cached_lookup" begin
    mod = @eval module $(gensym())
    using GPUCompiler

    # stands in for a back-end's kernel lookup: returns the method that the kernel call
    # dispatches to, which is what a compiler job for the kernel would be built from
    const lookups = Ref(0)
    function lookup_method(f::F, tt::TT=Tuple{}; key=nothing) where {F,TT}
        GPUCompiler.cached_lookup(Method, F, tt, key) do
            lookups[] += 1
            methodinstance(F, tt).def
        end
    end

    @noinline helper(x) = x + 1
    kernel(x::Number) = helper(x)
    launch(x; key=nothing) = lookup_method(kernel, Tuple{typeof(x)}; key)

    other_lookup(f::F, tt::TT) where {F,TT} =
        GPUCompiler.cached_lookup(() -> nothing, Method, F, tt, nothing)
    other_launch(x) = other_lookup(kernel, Tuple{typeof(x)})
    end

    lookups() = mod.lookups[]
    launch(x; kwargs...) = Base.invokelatest(mod.launch, x; kwargs...)

    m = launch(1)
    @test m.sig == Tuple{typeof(mod.kernel), Number}
    n = lookups()
    @test launch(1) === m
    @test lookups() == n

    # in another world the lookup runs again, but finds the same method
    @eval mod unrelated() = nothing
    @test launch(1) === m
    @test lookups() == n + 1
    @test launch(1) === m
    @test lookups() == n + 1

    # redefining a callee doesn't change the method, but it is looked up again
    @eval mod @noinline helper(x) = x + 2
    @test launch(1) === m
    @test lookups() == n + 2

    # a more specific method is used, although the code for the other one is still valid
    world = Base.get_world_counter()
    @eval mod kernel(x::Int) = -x
    m′ = launch(1)
    @test m′.sig == Tuple{typeof(mod.kernel), Int}
    @test lookups() == n + 3

    # lookups in an older world see the methods of that world, and are still cached
    @test Base.invoke_in_world(world, mod.launch, 1) === m
    @test launch(1) === m′
    @test lookups() == n + 3

    # other argument types and keys are looked up separately
    @test launch(1.0) === m
    @test lookups() == n + 4
    @test launch(1; key=:a) === m′
    @test lookups() == n + 5
    @test launch(1; key=:b) === m′
    @test lookups() == n + 6
    for _ in 1:2, key in (:a, :b)
        @test launch(1; key) === m′
    end
    @test lookups() == n + 6

    # keys are compared with `isequal`
    @test launch(1; key=NaN) === m′
    @test launch(1; key=NaN) === m′
    @test lookups() == n + 7
    @test launch(1; key=-0.0) === m′
    @test launch(1; key=0.0) === m′
    @test lookups() == n + 9

    # only the most recent results are kept
    keys = 1:GPUCompiler.LOOKUP_CACHE_SIZE
    for key in keys
        launch(1; key)
    end
    n′ = lookups()
    for key in keys
        launch(1; key)
    end
    @test lookups() == n′
    launch(1; key=0)
    launch(1; key=first(keys))
    @test lookups() == n′ + 2

    # different lookups with the same types don't share results
    @test launch(1) === m′
    @test_throws TypeError Base.invokelatest(mod.other_launch, 1)
end

@testset "cached_lookup: compile hooks" begin
    mod = @eval module $(gensym())
    using GPUCompiler
    const lookups = Ref(0)
    kernel() = nothing
    function launch(key)
        GPUCompiler.cached_lookup(Int, typeof(kernel), Tuple{}, key) do
            lookups[] += 1
            GPUCompiler.run_compile_hook(key)
            key
        end
    end
    function hooked_launch(keys)
        seen = Int[]
        GPUCompiler.with_compile_hook(key -> push!(seen, key)) do
            foreach(launch, keys)
        end
        return seen
    end
    end

    lookups() = mod.lookups[]
    launch(key) = Base.invokelatest(mod.launch, key)

    @test launch(1) == 1
    n = lookups()

    # a hook sees every lookup, also of results that were cached before
    @test Base.invokelatest(mod.hooked_launch, [1, 1, 2]) == [1, 1, 2]
    @test lookups() == n + 3

    # and nothing that is looked up while it is active is cached
    @test launch(1) == 1
    @test lookups() == n + 3
    @test launch(2) == 2
    @test lookups() == n + 4
end

@testset "cached_lookup: lifetime and errors" begin
    mod = @eval module $(gensym())
    using GPUCompiler
    const lookups = Ref(0)

    mutable struct Callable
        x::Int
    end
    (c::Callable)() = c.x

    # the result only depends on the type of the callable; its values come from the call
    function launch(f::F, key=nothing) where {F}
        code = GPUCompiler.cached_lookup(Int, F, Tuple{}, key) do
            lookups[] += 1
            # capture the callable, which mustn't keep it alive
            f.x
            100
        end
        return code + f()
    end

    # without leaving a reference to the callable on the caller's stack
    @noinline function launch_new(finalized)
        c = Callable(3)
        finalizer(_ -> finalized[] = true, c)
        launch(c, :new)
        return
    end

    const fail = Ref(false)
    function failing_launch()
        GPUCompiler.cached_lookup(Int, typeof(failing_launch), Tuple{}, nothing) do
            lookups[] += 1
            fail[] && error("lookup failed")
            42
        end
    end
    end

    lookups() = mod.lookups[]

    @test Base.invokelatest(mod.launch, mod.Callable(1)) == 101
    n = lookups()
    @test Base.invokelatest(mod.launch, mod.Callable(2)) == 102
    @test lookups() == n

    finalized = Threads.Atomic{Bool}(false)
    Base.invokelatest(mod.launch_new, finalized)
    # finalizers may run on another thread, or be delayed by a concurrent collection
    for _ in 1:10
        GC.gc(true)
        yield()
        finalized[] && break
        sleep(0.1)
    end
    @test finalized[]

    # failed lookups aren't cached
    mod.fail[] = true
    n = lookups()
    @test_throws ErrorException Base.invokelatest(mod.failing_launch)
    @test_throws ErrorException Base.invokelatest(mod.failing_launch)
    @test lookups() == n + 2
    mod.fail[] = false
    @test Base.invokelatest(mod.failing_launch) == 42
    @test Base.invokelatest(mod.failing_launch) == 42
    @test lookups() == n + 3
end

@testset "cached_lookup: concurrency" begin
    mod = @eval module $(gensym())
    using GPUCompiler
    const lookups = Threads.Atomic{Int}(0)
    kernel() = nothing
    function launch(key)
        GPUCompiler.cached_lookup(Int, typeof(kernel), Tuple{}, key) do
            Threads.atomic_add!(lookups, 1)
            yield()
            key
        end
    end
    end

    # more keys than the cache holds, so that tasks keep replacing each other's results
    results = Base.invokelatest() do
        tasks = map(1:8 * Threads.nthreads()) do i
            Threads.@spawn begin
                ok = true
                for j in 1:1000
                    key = (i + j) % 8
                    ok &= mod.launch(key) == key
                end
                ok
            end
        end
        fetch.(tasks)
    end
    @test all(results)
    @test mod.lookups[] > 0
end

precompile_test_harness("cached_lookup: precompilation") do load_path
    write(joinpath(load_path, "LookupCacheUser.jl"), """
        module LookupCacheUser
        using GPUCompiler
        kernel() = nothing
        lookup_kernel() = 42
        launch() = GPUCompiler.cached_lookup(lookup_kernel, Int, typeof(kernel), Tuple{}, nothing)
        # serializes the code of `launch`, including its cache
        launch()
        end
        """)
    Base.compilecache(Base.PkgId("LookupCacheUser"), stderr, stdout)

    @eval let
        using LookupCacheUser
        # results looked up while precompiling, like session handles, don't survive
        cache = GPUCompiler.lookup_cache(typeof(LookupCacheUser.lookup_kernel),
                                         typeof(LookupCacheUser.kernel), Tuple{}, Nothing, Int)
        @test (@atomic cache.head) === nothing
        @test LookupCacheUser.launch() == 42
    end
end
