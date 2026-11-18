# Cross-session reproducibility of Metal output. Back-ends content-key machine-code caches on
# the emitted AIR (Metal.jl's binary archives key on the metallib bytes), which only pays off
# when a kernel compiles to the same bytes in every session. Two fresh processes compile the
# same kernels and their output is compared byte for byte; the in-process variant in
# test/metal.jl catches most regressions more cheaply, but session-dependent module layout
# (Julia's per-CodeInstance link order on older versions, see src/determinism.jl) and
# counter leaks only show up across processes.

helpers = joinpath(@__DIR__, "..", "helpers")
script = """
    using GPUCompiler, LLVM, LLVMDowngrader_jll
    include(joinpath($(repr(helpers)), "runtime.jl"))
    include(joinpath($(repr(helpers)), "metal.jl"))

    function plain_kernel(ptr, x)
        unsafe_store!(ptr, x * 2f0 + 1f0, 1)
        return
    end

    # several CodeInstances, the exception machinery, and the runtime library
    @noinline function checked(x)
        x < 0 && throw(ArgumentError("negative"))
        return sqrt(x)
    end
    @noinline function bounded(x)
        x > 100 && throw(BoundsError())
        return x * 2
    end
    function throwing_kernel(ptr, x)
        unsafe_store!(ptr, checked(x) + bounded(x), 1)
        return
    end

    # a private constant shared by several CodeInstances
    const table = (1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0, 8f0)
    @noinline lookup_a(i) = i > 8 ? throw(ArgumentError("a")) : table[i]
    @noinline lookup_b(i) = i > 8 ? throw(ArgumentError("b")) : table[9 - i]
    function constant_kernel(ptr, i)
        unsafe_store!(ptr, lookup_a(i) + lookup_b(i), 1)
        return
    end

    # relocations through the kernel state's table
    function reloc_kernel(ptr, sym)
        unsafe_store!(ptr, sym === :a ? 1f0 : 2f0, 1)
        return
    end

    outdir = ARGS[1]
    for (name, f, tt) in [("plain",    plain_kernel,    Tuple{Core.LLVMPtr{Float32,1}, Float32}),
                          ("throwing", throwing_kernel, Tuple{Core.LLVMPtr{Float32,1}, Float32}),
                          ("constant", constant_kernel, Tuple{Core.LLVMPtr{Float32,1}, Int})]
        write(joinpath(outdir, name * ".air"), first(Metal.code_execution(f, tt)))
    end
    if LLVM.version() >= v"17"
        job, _ = Metal.create_table_job(reloc_kernel, Tuple{Core.LLVMPtr{Float32,1}, Symbol};
                                        kernel=true)
        air = JuliaContext() do _
            first(GPUCompiler.compile(:asm, job))
        end
        write(joinpath(outdir, "reloc.air"), air)
    end
    """

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
mktempdir() do dir_a
    mktempdir() do dir_b
        proc_a = run(pipeline(`$cmd $dir_a`; stdout=devnull); wait=false)
        proc_b = run(pipeline(`$cmd $dir_b`; stdout=devnull); wait=false)
        wait(proc_a); wait(proc_b)
        @test success(proc_a) && success(proc_b)

        files = readdir(dir_a)
        @test !isempty(files)
        @test files == readdir(dir_b)
        for file in files
            @testset "$file" begin
                @test read(joinpath(dir_a, file)) == read(joinpath(dir_b, file))
            end
        end
    end
end
