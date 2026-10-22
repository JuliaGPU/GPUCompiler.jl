@testset "configuration" begin
    job, _ = PTX.create_job(identity, (Int,))
    @test job.config.codegen === GPUCompiler.CODEGEN

    julia, _ = PTX.create_job(identity, (Int,); codegen=:julia)
    jc, _ = PTX.create_job(identity, (Int,); codegen=:juliacodegen)
    @test julia.config.codegen === :julia
    @test jc.config.codegen === :juliacodegen
    # part of the identity of cached artifacts
    @test julia.config != jc.config
    @test hash(julia.config) != hash(jc.config)
    # preserved by derived configs, e.g., of nested jobs
    @test CompilerConfig(jc.config; toplevel=false).codegen === :juliacodegen

    @test_throws "Unknown codegen=llvm" PTX.create_job(identity, (Int,); codegen=:llvm)
end

@testset "requirements" begin
    kernel() = nothing
    job, _ = PTX.create_job(kernel, (); codegen=:juliacodegen)

    provider = GPUCompiler.codegen_provider[]
    try
        GPUCompiler.codegen_provider[] = nothing
        JuliaContext() do ctx
            @test_throws "requires JuliaCodegen.jl to be loaded" GPUCompiler.compile(:llvm, job)
        end
    finally
        GPUCompiler.codegen_provider[] = provider
    end
end

@testset "loading" begin
    # with the preference selecting JuliaCodegen.jl, loading GPUCompiler loads it from the
    # environment, and the extension registers the code generator
    code = "import GPUCompiler; print(GPUCompiler.codegen_provider[] !== nothing)"
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`, String)
    @test out == string(GPUCompiler.CODEGEN === :juliacodegen)
end

# Packages that do not depend on JuliaCodegen.jl may compile code while precompiling, e.g.,
# in their workload, in a process without it.
precompile_test_harness("precompilation without JuliaCodegen.jl") do load_path
    create_standalone(load_path, "JuliaCodegenCompiler", "native.jl")
    write(joinpath(load_path, "JuliaCodegenWorkload.jl"), :(
        module JuliaCodegenWorkload
        import JuliaCodegenCompiler
        using JuliaCodegenCompiler.GPUCompiler

        kernel(x) = x + 1
        const codegen = let
            job, _ = JuliaCodegenCompiler.Native.create_job(kernel, (Int,))
            JuliaContext() do ctx
                GPUCompiler.compile(:llvm, job)
            end
            job.config.codegen
        end
        end) |> string)
    Base.compilecache(Base.PkgId("JuliaCodegenWorkload"))

    # such compilations use Julia's code generator; that is part of their configuration,
    # so results cached for them are not used for configurations with JuliaCodegen.jl
    mod = Base.require(Base.PkgId("JuliaCodegenWorkload"))
    @test Base.invokelatest(getglobal, mod, :codegen) === :julia
end

module Corpus
    mutable struct Box; x::Int; end
    struct Pair2; a::Float32; b::Int128; end

    alloc(p, i) = (b = Box(i); unsafe_store!(p, b.x + 1); nothing)
    ref(p, i) = (r = Ref(i); unsafe_store!(p, r[] * 2); nothing)
    function simd(p, n)
        s = 0.0f0
        @simd for i in 1:n
            @inbounds s += unsafe_load(p, i)
        end
        unsafe_store!(p, s)
        nothing
    end
    f16(p::Ptr{Float16}, x::Float16) = (unsafe_store!(p, x * x + Float16(1)); nothing)
    throws(p, i) = (i > 10 && throw(BoundsError()); unsafe_store!(p, i); nothing)
    i128(p::Ptr{Int128}, x::Int128) = (unsafe_store!(p, x * x); nothing)
    tuple(p, t::NTuple{4,Float32}) = (unsafe_store!(p, sum(t)); nothing)
    aggregate(p::Ptr{Pair2}, x::Pair2) = (unsafe_store!(p, Pair2(x.a + 1f0, x.b)); nothing)
    function loop(p, n)
        for i in 1:n, j in 1:n
            unsafe_store!(p, unsafe_load(p, i) + j, i)
        end
        nothing
    end
    closure(p, x) = (f = y -> y + x; unsafe_store!(p, f(1)); nothing)
    complex(p::Ptr{ComplexF32}, z::ComplexF32) = (unsafe_store!(p, z * z + abs(z)); nothing)
    math(p::Ptr{Float64}, x::Float64) =
        (unsafe_store!(p, sqrt(abs(x)) + muladd(x, x, 1.0) + copysign(x, -1.0)); nothing)
    div(p::Ptr{Int}, a::Int, b::Int) = (unsafe_store!(p, Base.div(a, b) + rem(a, b)); nothing)

    # with device pointers, for targets that reject host pointers in kernels
    device_alloc(p::Core.LLVMPtr{Int,1}, i) = (b = Box(i); unsafe_store!(p, b.x + 1); nothing)
    device_tuple(p::Core.LLVMPtr{Float32,1}, t::NTuple{4,Float32}) =
        (unsafe_store!(p, sum(t)); nothing)
    function device_simd(p::Core.LLVMPtr{Float32,1}, n::Int32)
        s = 0.0f0
        @simd for i in Int32(1):n
            @inbounds s += unsafe_load(p, i)
        end
        unsafe_store!(p, s)
        nothing
    end

    const DEVICE_KERNELS = [
        (device_alloc, (Core.LLVMPtr{Int,1}, Int)),
        (device_tuple, (Core.LLVMPtr{Float32,1}, NTuple{4,Float32})),
        (device_simd, (Core.LLVMPtr{Float32,1}, Int32)),
    ]

    const KERNELS = [
        (alloc, (Ptr{Int}, Int)), (ref, (Ptr{Int}, Int)), (simd, (Ptr{Float32}, Int)),
        (f16, (Ptr{Float16}, Float16)), (throws, (Ptr{Int}, Int)),
        (i128, (Ptr{Int128}, Int128)), (tuple, (Ptr{Float32}, NTuple{4,Float32})),
        (aggregate, (Ptr{Pair2}, Pair2)), (loop, (Ptr{Int}, Int)),
        (closure, (Ptr{Int}, Int)), (complex, (Ptr{ComplexF32}, ComplexF32)),
        (math, (Ptr{Float64}, Float64)), (div, (Ptr{Int}, Int, Int)),
    ]

    # arguments to run the kernels with on the host: a buffer for the first argument
    # (whose contents are the result), then the others
    const ARGS = Dict(
        alloc => (Int, 4, (7,)), ref => (Int, 1, (21,)), simd => (Float32, 8, (8,)),
        f16 => (Float16, 1, (Float16(1.5),)), throws => (Int, 1, (3,)),
        i128 => (Int128, 1, (Int128(1) << 40,)), tuple => (Float32, 1, ((1f0, 2f0, 3f0, 4f0),)),
        aggregate => (Pair2, 1, (Pair2(1.5f0, Int128(-3)),)), loop => (Int, 4, (4,)),
        closure => (Int, 1, (41,)), complex => (ComplexF32, 1, (ComplexF32(1, 2),)),
        math => (Float64, 1, (-2.25,)), div => (Int, 1, (17, 5)),
    )
end

# The function signature of a kernel's entry after GPUCompiler's processing, with the IR
# of the generator `codegen`. Without validation, which rejects some of the corpus on some
# targets (e.g. `Float64` on Metal) after optimization.
function entry_with(codegen, backend, f, tt; kwargs...)
    JuliaContext() do ctx
        job, _ = backend.create_job(f, tt; codegen, validate=false, kwargs...)
        ir, meta = GPUCompiler.compile(:llvm, job)
        sig = string(LLVM.function_type(meta.entry))
        cc = LLVM.callconv(meta.entry)
        LLVM.dispose(ir)
        (sig, cc)
    end
end

describe(kwargs) = join(("$k=$v" for (k, v) in pairs(kwargs)), " ")

function corpus_cases()
    cases = Any[]
    backends = Any[("PTX", PTX, (;)), ("SPIR-V", SPIRV, (; backend=:llvm)),
                   ("Metal", Metal, (;)), ("native", Native, (;)), ("BPF", BPF, (;))]
    # see runtests.jl
    if !LLVM.is_asserts() && :AMDGPU in LLVM.backends()
        push!(backends, ("GCN", GCN, (;)))
    end
    for (name, backend, kwargs) in backends,
        (f, tt) in [Corpus.KERNELS; Corpus.DEVICE_KERNELS], kernel in (false, true)
        # Metal kernels can't take host pointers
        name == "Metal" && kernel && (f, tt) ∉ Corpus.DEVICE_KERNELS && continue
        push!(cases, (name, backend, f, tt, (; kernel, kwargs...)))
    end
    return cases
end

@testset "the same entries" begin
    # JuliaCodegen.jl generates functions in Julia's specsig convention, which GPUCompiler
    # then rewrites the same way for every back end
    @testset "$name $(nameof(f)) $(describe(kwargs))" for (name, backend, f, tt, kwargs) in corpus_cases()
        @test entry_with(:juliacodegen, backend, f, tt; kwargs...) ==
              entry_with(:julia, backend, f, tt; kwargs...)
    end
end

# The kernels of the corpus with host pointers, compiled with the generator `codegen`,
# loaded into a JIT and run on a buffer, whose contents are returned.
function run_with(codegen, f, tt)
    T, n, args = Corpus.ARGS[f]
    job, _ = Native.create_job(f, tt; codegen, entry_abi=:func, relocations=:patch,
                               validate=false)
    JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)
        fptr, lljit, _table = Native.load(Vector{UInt8}(codeunits(obj)), meta.entry.name,
                                          meta.relocations)
        @dispose lljit=lljit begin
            buf = T <: Number ? zeros(T, n) : fill(T(map(zero, fieldtypes(T))...), n)
            f === Corpus.simd && (buf .= 1:n)
            boxed = Any[pointer(buf), args...]
            GC.@preserve buf boxed ccall(fptr, Any, (Any, Ptr{Any}, Int32), f,
                                         pointer(boxed), length(boxed))
            buf
        end
    end
end

@testset "the same results" begin
    @testset "$(nameof(f))" for (f, tt) in Corpus.KERNELS
        @test run_with(:juliacodegen, f, tt) == run_with(:julia, f, tt)
    end
end

@testset "JuliaCodegen.jl runs" begin
    # the extension records every compilation in its ledger
    function compilations(codegen, f, tt)
        ledger = tempname()
        withenv("JULIACODEGEN_LEDGER" => ledger) do
            entry_with(codegen, PTX, f, tt; kernel=true)
        end
        isfile(ledger) || return 0
        return countlines(ledger)
    end
    @test compilations(:julia, Corpus.simd, (Ptr{Float32}, Int)) == 0
    @test compilations(:juliacodegen, Corpus.simd, (Ptr{Float32}, Int)) == 1
end
