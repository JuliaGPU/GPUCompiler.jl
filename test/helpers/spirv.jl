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


## textual IR with atomics, to test their lowering

# a job for a SPIR-V target with the given atomics, to process textual IR with
function atomics_job(; backend=:llvm, atomics=SPIRVAtomics(), validate=true)
    source = methodinstance(typeof(identity), Tuple{Int}, Base.get_world_counter())
    target = SPIRVCompilerTarget(; backend, atomics, validate=true)
    config = CompilerConfig(target, CompilerParams(); kernel=true, validate)
    CompilerJob(source, config)
end

# a kernel with global, local, generic, private and constant pointers, and `globals`
atomics_kernel(body; globals="") = """
    target datalayout = "$(GPUCompiler.llvm_datalayout(SPIRVCompilerTarget()))"
    $globals
    define spir_kernel void @kernel(ptr addrspace(1) %g, ptr addrspace(3) %l, ptr %p,
                                    ptr addrspace(2) %constant) {
      %generic = addrspacecast ptr addrspace(1) %g to ptr addrspace(4)
    $body
      ret void
    }
    """

# the reasons `validate_ir` gives for rejecting the atomics in `body`
function atomics_errors(body; globals="", kwargs...)
    job = atomics_job(; kwargs...)
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, atomics_kernel(body; globals))
        map(first, GPUCompiler.validate_ir(job, mod))
    end
end

# lower the atomics in `body` like `finish_ir!` does, returning the IR, and translate it to
# SPIR-V with the job's back-end, returning the disassembly (validated by `spirv-val`), with
# integer constants replaced by their value, e.g. `OpAtomicIAdd %uint %g 1 896 1`
function lower_atomics(body; globals="", kwargs...)
    job = atomics_job(; kwargs...)
    LLVM.Context(; opaque_pointers=true) do ctx
        mod = parse(LLVM.Module, atomics_kernel(body; globals))
        GPUCompiler.lower_atomics!(job, mod)
        GPUCompiler.lower_minimum_maximum!(mod)
        LLVM.verify(mod)
        ir = string(mod)
        # no plain atomic or fence may reach the back-end
        for f in mod.functions, bb in f.blocks, inst in bb.instructions
            if GPUCompiler.is_atomic_memop(inst) || inst isa LLVM.FenceInst
                error("Atomic operation was not lowered: $inst")
            end
        end

        mod.triple = GPUCompiler.llvm_triple(job.config.target)
        # (before Julia 1.12, `mcgen` releases the type-inference lock its caller holds)
        locked = VERSION < v"1.12.0-DEV.769"
        locked && ccall(:jl_typeinf_lock_begin, Cvoid, ())
        asm = try
            GPUCompiler.mcgen(job, mod)
        finally
            locked && ccall(:jl_typeinf_lock_end, Cvoid, ())
        end
        constants = Dict{String,String}()
        for m in eachmatch(r"^\s*(%\S+) = OpConstant %u?(?:int|long) (\d+)$"m, asm)
            constants[m[1]] = m[2]
        end
        for m in eachmatch(r"^\s*(%\S+) = OpConstantNull %u?(?:int|long)$"m, asm)
            constants[m[1]] = "0"
        end
        asm = replace(asm, r"%[\w.]+" => id -> get(constants, id, id))
        ir, asm
    end
end

# Check that every combination of operations, types, orderings, synchronization scopes and
# address spaces that used to make the back-ends miscompile, error or exit the process is
# either rejected by `validate_ir`, or lowered to SPIR-V that `spirv-val` accepts. Returns the
# cases that failed, and the number of rejected ones.
function check_atomics_matrix(backend)
    cases = let
        Op = LLVM.AtomicRMWBinOp
        intops = ["xchg", "add", "sub", "and", "or", "xor", "max", "min", "umax", "umin", "nand"]
        LLVM.isavailable(Op.UIncWrap) && append!(intops, ["uinc_wrap", "udec_wrap"])
        fpops = ["xchg", "fadd", "fsub", "fmax", "fmin"]
        LLVM.isavailable(Op.FMaximum) && append!(fpops, ["fmaximum", "fminimum"])
        values = Dict("i8" => "1", "i16" => "1", "i32" => "1", "i64" => "1", "half" => "0xH3C00",
                      "bfloat" => "0xR3F80", "float" => "1.0", "double" => "1.0", "ptr" => "null")
        sizes = Dict("i8" => 1, "i16" => 2, "half" => 2, "bfloat" => 2, "i32" => 4, "float" => 4)
        align(T) = get(sizes, T, 8)
        syncscope(scope) = isempty(scope) ? "" : "syncscope(\"$scope\") "
        ptr(as) = as == 1 ? "%g" : as == 3 ? "%l" : "%generic"
        rmw(op, T, order, scope, as=1) =
            "%r = atomicrmw $op ptr addrspace($as) $(ptr(as)), $T $(values[T]) $(syncscope(scope))$order, align $(align(T))"
        failure(order) = order in ("monotonic", "release") ? "monotonic" :
                         order in ("acquire", "acq_rel") ? "acquire" : "seq_cst"
        cas(T, order, scope, as=1; weak=false) =
            "%r = cmpxchg $(weak ? "weak " : "")ptr addrspace($as) $(ptr(as)), $T $(values[T]), $T $(values[T]) $(syncscope(scope))$order $(failure(order)), align $(align(T))"
        load(T, order, scope, as=1) =
            "%r = load atomic $T, ptr addrspace($as) $(ptr(as)) $(syncscope(scope))$order, align $(align(T))"
        store(T, order, scope, as=1) =
            "store atomic $T $(values[T]), ptr addrspace($as) $(ptr(as)) $(syncscope(scope))$order, align $(align(T))"

        types = ["i8", "i16", "i32", "i64", "half", "bfloat", "float", "double"]
        cases = String[]
        for scope in ("", "workgroup"), order in ("monotonic",)
            for T in types, op in (T in ("half", "bfloat", "float", "double") ? fpops : intops)
                push!(cases, rmw(op, T, order, scope))
            end
            push!(cases, rmw("xchg", "ptr", order, scope))
            for T in ["i8", "i16", "i32", "i64", "ptr"], weak in (false, true)
                push!(cases, cas(T, order, scope; weak))
            end
            for T in [types; "ptr"]
                push!(cases, load(T, order, scope), store(T, order, scope))
            end
        end
        orders = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]
        for scope in ("", "system", "device", "workgroup", "subgroup", "singlethread")
            for as in (1, 3, 4), order in orders
                push!(cases, rmw("add", "i32", order, scope, as), rmw("fadd", "float", order, scope, as),
                      cas("i32", order, scope, as))
                order in ("release", "acq_rel") || push!(cases, load("i32", order, scope, as))
                order in ("acquire", "acq_rel") || push!(cases, store("i32", order, scope, as))
            end
            for order in orders[2:end]
                push!(cases, "fence $(syncscope(scope))$order")
            end
        end
        cases
    end

    all_fadd = (; fadd_f32_global=true, fadd_f32_local=true, fadd_f64_global=true,
                fadd_f64_local=true)
    failures = String[]
    rejected = 0
    for atomics in (SPIRVAtomics(), SPIRVAtomics(; all_fadd...),
                    SPIRVAtomics(; all_fadd..., fadd_f16_global=true, fadd_f16_local=true))
        for body in cases
            if isempty(atomics_errors(body; atomics))
                try
                    lower_atomics(body; backend, atomics)
                catch err
                    push!(failures, "$body with $atomics: $(sprint(showerror, err))")
                end
            else
                rejected += 1
            end
        end
    end
    return failures, rejected
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
