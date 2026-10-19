@testset "configuration" begin
    job, _ = PTX.create_job(identity, (Int,))
    @test job.config.julia_passes === GPUCompiler.JULIA_PASSES

    cpp, _ = PTX.create_job(identity, (Int,); julia_passes=:cpp)
    ports, _ = PTX.create_job(identity, (Int,); julia_passes=:ports)
    @test cpp.config.julia_passes === :cpp
    @test ports.config.julia_passes === :ports
    # part of the identity of cached artifacts
    @test cpp.config != ports.config
    @test hash(cpp.config) != hash(ports.config)
    # preserved by derived configs, e.g., of nested jobs
    @test CompilerConfig(ports.config; toplevel=false).julia_passes === :ports

    @test_throws "Unknown julia_passes=llvm" PTX.create_job(identity, (Int,); julia_passes=:llvm)
end

@testset "providers" begin
    cpp, _ = PTX.create_job(identity, (Int,); julia_passes=:cpp)
    ports, _ = PTX.create_job(identity, (Int,); julia_passes=:ports)
    @test string(GPUCompiler.julia_pass(cpp, :AllocOptPass)) == "AllocOpt"
    @test GPUCompiler.julia_pass(ports, :AllocOptPass) == "jl-alloc-opt"
    @test_throws "Unknown Julia pass" GPUCompiler.julia_pass(ports, :NoSuchPass)

    # the passes JuliaIR.jl does not port yet, which run as the C++ pass in either mode
    names = GPUCompiler.pass_ports[].names
    @test sort([name for (name, port) in names if port === nothing]) ==
          [:FinalLowerGCPass, :JuliaLICMPass, :LowerPTLSPass, :RemoveJuliaAddrspacesPass]
    @test string(GPUCompiler.julia_pass(ports, :JuliaLICMPass)) == "JuliaLICM"
end

@testset "requirements" begin
    kernel() = nothing
    job, _ = PTX.create_job(kernel, (); julia_passes=:ports)

    ports = GPUCompiler.pass_ports[]
    try
        GPUCompiler.pass_ports[] = nothing
        JuliaContext(; opaque_pointers=true) do ctx
            @test_throws "requires JuliaIR.jl to be loaded" GPUCompiler.compile(:llvm, job)
        end
    finally
        GPUCompiler.pass_ports[] = ports
    end

    if VERSION < v"1.12"
        JuliaContext(; opaque_pointers=false) do ctx
            @test_throws "requires a context with opaque pointers" GPUCompiler.compile(:llvm, job)
        end
    end

    # reflection picks a suitable context
    @test occursin("define", sprint(io -> GPUCompiler.code_llvm(io, job)))
end

# Packages that do not depend on JuliaIR.jl may compile code while precompiling, e.g., in
# their workload, in a process without it.
VERSION >= v"1.11" && precompile_test_harness("precompilation without JuliaIR.jl") do load_path
    create_standalone(load_path, "NativeCompiler", "native.jl")
    write(joinpath(load_path, "PortsWorkload.jl"), :(
        module PortsWorkload
        import NativeCompiler
        using NativeCompiler.GPUCompiler

        kernel(x) = x + 1
        const julia_passes = let
            job, _ = NativeCompiler.Native.create_job(kernel, (Int,))
            JuliaContext() do ctx
                GPUCompiler.compile(:llvm, job)
            end
            job.config.julia_passes
        end
        end) |> string)
    Base.compilecache(Base.PkgId("PortsWorkload"))

    # such compilations use the C++ passes; that is part of their configuration, so
    # results cached for them are not used for configurations with the ports
    mod = Base.require(Base.PkgId("PortsWorkload"))
    @test Base.invokelatest(getglobal, mod, :julia_passes) === :cpp
end

## differential: the C++ passes and the ports give the same IR

# Two compilations differ in the counters codegen appends to the names of functions (and
# of the symbols derived from them in assembly), e.g. `julia_kernel_123`.
normalize_ir(ir::String) =
    replace(ir, r"(?<![\w.])(\.L)?(julia|j|jfptr)_(\w*?)_\d+(?=_param|\b)" => s"\1\2_\3_N")

# The IR of a kernel. Without validation, which rejects some of the corpus on some targets
# (e.g. `Float64` on Metal) after optimization.
function compile_with(julia_passes, backend, f, tt; kwargs...)
    JuliaContext(; opaque_pointers=true) do ctx
        job, _ = backend.create_job(f, tt; julia_passes, validate=false, kwargs...)
        ir, meta = GPUCompiler.compile(:llvm, job)
        str = string(ir)
        LLVM.dispose(ir)
        normalize_ir(str)
    end
end

function same_ir(backend, f, tt; kwargs...)
    cpp = compile_with(:cpp, backend, f, tt; kwargs...)
    ports = compile_with(:ports, backend, f, tt; kwargs...)
    cpp == ports && return true
    @error "IR differs between the C++ passes and the ports" f tt kwargs cpp ports
    return false
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
end

@testset "ports run" begin
    # JuliaIR.jl records every pass it runs in its ledger
    function passes_run(julia_passes, f, tt)
        ledger = tempname()
        old = JuliaIR.LEDGER[]
        try
            JuliaIR.LEDGER[] = ledger
            compile_with(julia_passes, PTX, f, tt; kernel=true)
        finally
            JuliaIR.LEDGER[] = old
        end
        isfile(ledger) || return Set{String}()
        Set(split(line, '\t')[3] for line in eachline(ledger))
    end

    @test isempty(passes_run(:cpp, Corpus.simd, (Ptr{Float32}, Int)))
    ran = union(passes_run(:ports, Corpus.simd, (Ptr{Float32}, Int)),
                passes_run(:ports, Corpus.alloc, (Ptr{Int}, Int)))
    expected = ["jl-alloc-opt", "jl-demote-float16", "jl-gc-invariant-verifier",
                "jl-late-lower-gc-frame", "jl-lower-simdloop", "jl-propagate-addrspaces",
                "jl-remove-ni"]
    VERSION < v"1.12.0-DEV.1390" && push!(expected, "jl-combine-muladd")
    VERSION >= v"1.13.0-DEV.321" && push!(expected, "jl-expand-atomic-modify")
    @test sort!(collect(ran)) == sort!(expected)
end

# Compiling with both providers in opaque-pointer contexts, so that the comparison is
# independent of the typed pointers of Julia 1.10 and 1.11 (see below).
describe(kwargs) = join(("$k=$v" for (k, v) in pairs(kwargs)), " ")

function differential_cases()
    cases = Any[]
    backends = Any[("PTX", PTX, (;)), ("SPIR-V", SPIRV, (; backend=:llvm)),
                   ("Metal", Metal, (;)), ("native", Native, (;)), ("BPF", BPF, (;))]
    # see runtests.jl
    LLVM.is_asserts() || push!(backends, ("GCN", GCN, (;)))
    for (name, backend, kwargs) in backends,
        (f, tt) in [Corpus.KERNELS; Corpus.DEVICE_KERNELS], kernel in (false, true)
        # Metal kernels can't take host pointers
        name == "Metal" && kernel && (f, tt) ∉ Corpus.DEVICE_KERNELS && continue
        push!(cases, (name, backend, f, tt, (; kernel, kwargs...)))
    end
    return cases
end

@testset "differential" begin
    @testset "$name $(nameof(f)) $(describe(kwargs))" for (name, backend, f, tt, kwargs) in differential_cases()
        @test same_ir(backend, f, tt; kwargs...)
    end
    @testset "PTX opt_level=$opt_level" for opt_level in (0, 1, 3)
        @test same_ir(PTX, Corpus.simd, (Ptr{Float32}, Int); kernel=true, opt_level)
        @test same_ir(PTX, Corpus.alloc, (Ptr{Int}, Int); kernel=true, opt_level)
    end
end

# The ports require opaque pointers, so with them, Julia 1.10 and 1.11 generate code in
# opaque-pointer contexts instead of typed ones. Judge that switch separately, with the C++
# passes and down to machine code. Not when Julia itself uses opaque pointers
# (`--opaque-pointers`), as its code generator then does not support typed contexts.
@static if VERSION < v"1.12" && Context(LLVM.supports_typed_pointers)
@testset "typed and opaque pointers" begin
    function asm_with(opaque_pointers, backend, f, tt; kwargs...)
        JuliaContext(; opaque_pointers) do ctx
            job, _ = backend.create_job(f, tt; julia_passes=:cpp, validate=false, kwargs...)
            asm, meta = GPUCompiler.compile(:asm, job)
            LLVM.dispose(meta.ir)
            # without comments, which name basic blocks
            replace(normalize_ir(String(asm)), r"//.*" => "", r"[ \t]+" => " ")
        end
    end

    # the back-ends that generate machine code outside of Julia's LLVM, see runtests.jl
    have_ptx = NVPTX_LLVM_Backend_jll.is_available()
    have_gcn = AMDGPU_LLVM_Backend_jll.is_available() && !LLVM.is_asserts()
    have_spirv = SPIRV_LLVM_Backend_jll.is_available() &&
                 SPIRV_LLVM_Translator_jll.is_available() && SPIRV_Tools_jll.is_available()

    have_ptx && @testset "PTX $(nameof(f)) kernel=$kernel" for (f, tt) in [Corpus.KERNELS;
                                                                         Corpus.DEVICE_KERNELS],
                                                              kernel in (false, true)
        @test asm_with(false, PTX, f, tt; kernel) == asm_with(true, PTX, f, tt; kernel)
    end

    # elsewhere, the code differs in register allocation (native), debug locations (GCN)
    # or the form of instructions (Metal's downgraded IR), so only require that it is
    # generated. Metal and SPIR-V kernels can't take host pointers.
    cases = Any[]
    for (f, tt) in Corpus.KERNELS, kernel in (false, true)
        push!(cases, ("native", Native, f, tt, (; kernel)))
        have_gcn && push!(cases, ("GCN", GCN, f, tt, (; kernel)))
        kernel || push!(cases, ("Metal", Metal, f, tt, (; kernel)))
    end
    for (f, tt) in Corpus.DEVICE_KERNELS, kernel in (false, true)
        push!(cases, ("Metal", Metal, f, tt, (; kernel)))
        if have_spirv
            push!(cases, ("SPIR-V", SPIRV, f, tt, (; kernel, backend=:llvm)))
            push!(cases, ("SPIR-V", SPIRV, f, tt, (; kernel, backend=:khronos)))
        end
    end
    @testset "$name $(nameof(f)) $(describe(kwargs))" for (name, backend, f, tt, kwargs) in cases
        @test asm_with(false, backend, f, tt; kwargs...) isa String
        @test asm_with(true, backend, f, tt; kwargs...) isa String
    end
end
end
