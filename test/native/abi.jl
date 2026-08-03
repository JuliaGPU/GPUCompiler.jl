@testset "ABI layout" begin

# Signatures chosen to exercise the parts of the specsig ABI that the legacy
# lockstep walk in `_classify_arguments_legacy` gets right, plus the two it does
# not: the leading return slots, and the `.roots.` shadow parameter.
struct SmallImm
    a::Int
    b::Int
end
struct BigImm
    a::NTuple{8,Int}
end
# inline-allocated, but only *some* fields are tracked pointers, so codegen adds
# a `.roots.` shadow parameter after the value pointer
struct SomeRoots
    a::Int
    b::String
end

kernel_int(x::Int) = nothing
kernel_ghost(::Nothing, x::Int) = nothing
kernel_agg(x::SmallImm) = nothing
kernel_many(a::Int, b::Float64, c::SmallImm, d::Ptr{Int}) = nothing

SIGNATURES = Any[
    (kernel_int, (Int,)),
    (kernel_ghost, (Nothing, Int)),
    (kernel_agg, (SmallImm,)),
    (kernel_many, (Int, Float64, SmallImm, Ptr{Int})),
]

@testset "differential vs. the legacy classification" begin
    if !GPUCompiler.HAS_ABI_LAYOUT
        @test_skip "jl_get_specsig_layout unavailable"
    else
        for (f, tt) in SIGNATURES
            job, _ = Native.create_job(f, tt)
            JuliaContext() do ctx
                _, meta = GPUCompiler.compile(:llvm, job)
                ft = LLVM.function_type(meta.entry)
                new = GPUCompiler._classify_arguments_abi(job)
                old = GPUCompiler._classify_arguments_legacy(job, ft)
                @test length(new) == length(old)
                for (n, o) in zip(new, old)
                    @test n.cc == o.cc
                    @test n.typ == o.typ
                    @test n.name == o.name
                    @test n.idx == o.idx
                end
                # and the indices actually address the emitted function
                for n in new
                    n.idx === nothing || @test n.idx <= length(LLVM.parameters(ft))
                end
            end
        end
    end
end

@testset "coverage the legacy path lacks" begin
    if !GPUCompiler.HAS_ABI_LAYOUT
        @test_skip "jl_get_specsig_layout unavailable"
    else
        # An aggregate with *some* tracked pointers gets an extra `.roots.` slot
        # that the legacy walk never accounts for, shifting every later index.
        # (Such kernels are rejected by `check_invocation`, but the
        # classification still has to be right for non-kernel jobs.)
        roots_fn(x::SomeRoots, y::Int) = nothing
        job, _ = Native.create_job(roots_fn, (SomeRoots, Int))
        layout, _ = GPUCompiler.abi_layout(job)
        args = GPUCompiler._classify_arguments_abi(job)
        @test args[2].cc == GPUCompiler.BITS_REF
        @test args[2].roots_idx !== nothing
        @test args[2].roots_idx == args[2].idx + 1
        # the trailing Int comes after the shadow slot; the legacy walk puts it
        # in the shadow slot's place
        @test args[3].idx == args[2].roots_idx + 1
        @test args[3].idx == layout.nparams

        # A `:specfunc` entry returning a large immutable takes a leading sret
        # pointer, which the legacy walk never counts.
        sret_fn(x::Int) = BigImm(ntuple(i -> x + i, 8))
        job, _ = Native.create_job(sret_fn, (Int,); entry_abi=:specfunc)
        layout, _ = GPUCompiler.abi_layout(job)
        @test layout.rettype_cc == GPUCompiler.JL_ABI_RET_SRET
        @test layout.sret_idx == 0
        @test layout.nprefix_params == 1
        args = GPUCompiler._classify_arguments_abi(job)
        @test args[2].cc == GPUCompiler.BITS_VALUE
        @test args[2].idx == 2  # after the sret pointer
        JuliaContext() do ctx
            _, meta = GPUCompiler.compile(:llvm, job)
            ft = LLVM.function_type(meta.entry)
            @test length(LLVM.parameters(ft)) == layout.nparams
            # what the legacy path used to report, for the record
            old = GPUCompiler._classify_arguments_legacy(job, ft)
            @test old[2].idx == 1
            @test old[2].idx != args[2].idx
        end
    end
end

@testset "GPUCompiler compiles without a gcstack argument" begin
    if GPUCompiler.HAS_ABI_LAYOUT
        for (f, tt) in SIGNATURES
            job, _ = Native.create_job(f, tt)
            layout, _ = GPUCompiler.abi_layout(job)
            @test layout.pgcstack_idx == -1
            @test layout.specsig == 1  # prefer_specsig is always set
        end
    end
end

@testset "abi_signature" begin
    plus(x::Int, y::Int) = x + y
    mi = Base.method_instance(plus, (Int, Int))
    @test GPUCompiler.abi_signature(mi) === Tuple{typeof(plus),Int,Int}
end

@testset "boxing predicates agree with Julia" begin
    for T in Any[Int, Float64, Nothing, SmallImm, BigImm, SomeRoots, Ptr{Int},
                 Vector{Int}, Any, Integer, Tuple{Int,Int}, Tuple{Int,Any}]
        @test GPUCompiler.deserves_argbox(T) == !GPUCompiler.deserves_stack(T)
        @test GPUCompiler.deserves_retbox(T) == GPUCompiler.deserves_argbox(T)
    end
    @test GPUCompiler.deserves_stack(Int)
    @test GPUCompiler.deserves_stack(SmallImm)
    @test GPUCompiler.deserves_stack(SomeRoots)
    @test GPUCompiler.deserves_argbox(Vector{Int})   # mutable
    @test GPUCompiler.deserves_argbox(Any)           # abstract
    @test GPUCompiler.deserves_argbox(Tuple{Int,Any}) # not inline-allocatable
end

end
