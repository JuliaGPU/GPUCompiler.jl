module SPIRV

using ..GPUCompiler
import LLVM
import ..TestRuntime

struct CompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,CompilerParams}) = TestRuntime

function create_job(@nospecialize(func), @nospecialize(types);
                   supports_fp16=true, supports_fp64=true, supports_bfloat16=false,
                   backend::Symbol, driver::Symbol=:generic, extensions::String="",
                   atomics::SPIRVAtomics=SPIRVAtomics(), kwargs...)
    config_kwargs, kwargs = split_kwargs(kwargs, GPUCompiler.CONFIG_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types), Base.get_world_counter())
    target = SPIRVCompilerTarget(; backend, validate=true, optimize=true, extensions, atomics,
                                   supports_fp16, supports_fp64, supports_bfloat16, driver)
    params = CompilerParams()
    config = CompilerConfig(target, params; kernel=false, config_kwargs...)
    CompilerJob(source, config), kwargs
end

function code_typed(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end

function code_warntype(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_warntype(io, job; kwargs...)
end

function code_llvm(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_llvm(io, job; kwargs...)
end

function code_native(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kwargs...)
    GPUCompiler.code_native(io, job; kwargs...)
end

# aliases without ::IO argument
for method in (:code_warntype, :code_llvm, :code_native)
    method = Symbol("$(method)")
    @eval begin
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

# LLVM atomics on `LLVMPtr`s, like UnsafeAtomics emits them, with the operation, ordering and
# synchronization scope (`:system` for LLVM's default one) passed as `Val`s
module Atomics
    using LLVM, LLVM.IR, LLVM.Build, LLVM.Interop
    const Ordering = LLVM.AtomicOrdering
    const Op = LLVM.AtomicRMWBinOp

    # (typed pointers point to `i8`)
    typed_ptr(builder, p, T) = bitcast!(builder, p, LLVM.PointerType(T, p.value_type.addrspace))

    @inline @llvmgenerated builder function modify!(p::Core.LLVMPtr{T}, x::T, ::Val{op},
            ::Val{order}=Val(Ordering.Monotonic), ::Val{scope}=Val(:system))::T where {T,op,order,scope}
        atomic_rmw!(builder, op, typed_ptr(builder, p, x.value_type), x, order;
                    scope=String(scope))
    end

    # returns the old value
    @inline @llvmgenerated builder function cas!(p::Core.LLVMPtr{T}, cmp::T, new::T,
            ::Val{success}=Val(Ordering.Monotonic), ::Val{failure}=Val(Ordering.Monotonic),
            ::Val{scope}=Val(:system), ::Val{weak}=Val(false))::T where {T,success,failure,scope,weak}
        res = atomic_cmpxchg!(builder, typed_ptr(builder, p, cmp.value_type), cmp, new, success,
                              failure; scope=String(scope), weak)
        extract_value!(builder, res, 0)
    end

    @inline @llvmgenerated builder function load(p::Core.LLVMPtr{T}, ::Type{T},
            ::Val{order}=Val(Ordering.Monotonic), ::Val{scope}=Val(:system))::T where {T,order,scope}
        T_val = convert(LLVMType, T)
        load!(builder, T_val, typed_ptr(builder, p, T_val); ordering=order,
              scope=String(scope), align=sizeof(T))
    end

    @inline @llvmgenerated builder function store!(p::Core.LLVMPtr{T}, x::T,
            ::Val{order}=Val(Ordering.Monotonic), ::Val{scope}=Val(:system))::Nothing where {T,order,scope}
        LLVM.store!(builder, x, typed_ptr(builder, p, x.value_type); ordering=order,
               scope=String(scope), align=sizeof(T))
        return
    end

    @inline @llvmgenerated builder function fence(::Val{order},
            ::Val{scope}=Val(:system))::Nothing where {order,scope}
        fence!(builder, order; scope=String(scope))
        return
    end
end

# simulates codegen for a kernel function: validates by default. Returns the assembly and
# the metadata without the IR (and its entry function), which only lives as long as the
# context, and is disposed of here.
function code_execution(@nospecialize(func), @nospecialize(types); kwargs...)
    job, kwargs = create_job(func, types; kernel=true, kwargs...)
    JuliaContext() do ctx
        asm, meta = GPUCompiler.compile(:asm, job; kwargs...)
        LLVM.dispose(meta.ir)
        asm, Base.structdiff(meta, NamedTuple{(:ir, :entry)})
    end
end

end
