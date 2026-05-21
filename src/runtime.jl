# GPU runtime library
#
# This module defines method instances that will be compiled into a target-specific image
# and will be available to the GPU compiler to call after Julia has generated code.
#
# Most functions implement, or are used to support Julia runtime functions that are expected
# by the Julia compiler to be available at run time, e.g., to dynamically allocate memory,
# box values, etc.

module Runtime

using ..GPUCompiler
using LLVM
using LLVM.Interop


## representation of a runtime method instance

struct RuntimeMethodInstance
    # either a function defined here, or a symbol to fetch a target-specific definition
    def::Union{Function,Symbol}

    return_type::Type
    types::Tuple
    name::Symbol

    # LLVM types cannot be cached, so we can't put them in the runtime method instance.
    # the actual types are constructed upon accessing them, based on a sentinel value:
    #  - nothing: construct the LLVM type based on its Julia counterparts
    #  - function: call this generator to get the type (when more control is needed)
    llvm_return_type::Union{Nothing, Function}
    llvm_types::Union{Nothing, Function}
    llvm_name::String
end

function Base.convert(::Type{LLVM.FunctionType}, rt::RuntimeMethodInstance)
    types = if rt.llvm_types === nothing
        LLVMType[convert(LLVMType, typ; allow_boxed=true) for typ in rt.types]
    else
        rt.llvm_types()
    end

    return_type = if rt.llvm_return_type === nothing
        convert(LLVMType, rt.return_type; allow_boxed=true)
    else
        rt.llvm_return_type()
    end

    LLVM.FunctionType(return_type, types)
end

const methods = Dict{Symbol,RuntimeMethodInstance}()
function get(name::Symbol)
    methods[name]
end

# Register a Julia function `def` as a runtime library function identified by `name`. The
# function will be compiled upon first use for argument types `types` and should return
# `return_type`. Use `Runtime.get(name)` to get a reference to this method instance.
#
# The corresponding LLVM types `llvm_types` and `llvm_return_type` will be deduced from
# their Julia counterparts. To influence that conversion, pass a callable object instead;
# this object will be evaluated at run-time and the returned value will be used instead.
#
# When generating multiple runtime functions from a single definition, make sure to specify
# different values for `name`. The LLVM function name will be deduced from that name, but
# you can always specify `llvm_name` to influence that. Never use an LLVM name that starts
# with `julia_` or the function might clash with other compiled functions.
function compile(def, return_type, types, llvm_return_type=nothing, llvm_types=nothing;
                 name=isa(def,Symbol) ? def : nameof(def), llvm_name="gpu_$name")
    meth = RuntimeMethodInstance(def,
                                 return_type, types, name,
                                 llvm_return_type, llvm_types, llvm_name)
    if haskey(methods, name)
        error("Runtime function $name has already been registered!")
    end
    methods[name] = meth

    # Symbolic `def` means the runtime symbol is provided by the back-end. Emit
    # an `llvmcall` whose IR declares `gpu_<name>` as a *weak* definition with
    # a CPU-safe no-op body, and calls it. The weak body satisfies the JIT's
    # symbol resolution on CPU (so AOT pipelines — juliac, sysimage
    # `compile=all`, PrecompileTools — don't fail trying to link an undefined
    # `gpu_<name>`). The GPU runtime library, when linked in, provides the
    # real strong definition; LLVM's linker semantics replace the weak with
    # the strong.
    if def isa Symbol
        args = [gensym() for typ in types]
        stub = LLVM.Context() do _
            build_runtime_stub(llvm_name, return_type, types, args)
        end
        @eval @inline $def($(args...)) = $stub
    end

    return
end

# Build an `llvmcall` expression for a back-end-provided runtime symbol:
#
#   define weak <rt> @gpu_<name>(<args>) { ret <fake> }
#   define <rt> @entry(<args>) { %r = call <rt> @gpu_<name>(<args>); ret <rt> %r }
#
# Returns the `Base.llvmcall(...)`-shaped quote produced by `LLVM.Interop.call_function`,
# suitable for splicing as the stub's body.
function build_runtime_stub(llvm_name::String, @nospecialize(return_type::Type),
                            @nospecialize(types::Tuple), args::Vector)
    rt = convert(LLVMType, return_type; allow_boxed=true)
    arg_tys = LLVMType[convert(LLVMType, t; allow_boxed=true) for t in types]

    # entry function (`call_function` puts the module on it)
    entry, entry_ft = create_function(rt, arg_tys)
    mod = LLVM.parent(entry)

    # weak definition of `gpu_<name>` that returns a harmless placeholder on CPU
    extern = LLVM.Function(mod, llvm_name, LLVM.FunctionType(rt, arg_tys))
    linkage!(extern, LLVM.API.LLVMWeakAnyLinkage)
    @dispose builder=IRBuilder() begin
        position!(builder, BasicBlock(extern, "entry"))
        emit_fake_return!(builder, rt)
    end

    # entry: call the weak symbol, return its result
    @dispose builder=IRBuilder() begin
        position!(builder, BasicBlock(entry, "entry"))
        result = call!(builder, LLVM.function_type(extern), extern,
                       collect(parameters(entry)))
        if rt isa LLVM.VoidType
            ret!(builder)
        else
            ret!(builder, result)
        end
    end

    return call_function(entry, return_type, Tuple{types...}, args...)
end

# Emit a placeholder return of the given LLVM type — a sentinel value that
# never escapes (the stub is never meant to actually run on CPU; this only
# satisfies materialization).
function emit_fake_return!(builder::IRBuilder, rt::LLVMType)
    if rt isa LLVM.VoidType
        ret!(builder)
    elseif rt isa LLVM.PointerType
        # Use Int64(1), not 0, so `Ptr(Int64(...))` doesn't get lowered to C_NULL.
        i64 = LLVM.IntType(64)
        ret!(builder, const_inttoptr(ConstantInt(i64, 1), rt))
    elseif rt isa LLVM.IntegerType
        ret!(builder, ConstantInt(rt, 0))
    elseif rt isa LLVM.LLVMFloat || rt isa LLVM.LLVMDouble
        ret!(builder, ConstantFP(rt, 0.0))
    else
        error("Unsupported runtime stub return type: $rt")
    end
end


## exception handling

# expected functions for exception signalling
compile(:signal_exception, Nothing, ())

# expected functions for simple exception handling
compile(:report_exception, Nothing, (Ptr{Cchar},))
compile(:report_oom, Nothing, (Csize_t,))

# expected functions for verbose exception handling
compile(:report_exception_frame, Nothing, (Cint, Ptr{Cchar}, Ptr{Cchar}, Cint))
compile(:report_exception_name, Nothing, (Ptr{Cchar},))

# NOTE: no throw functions are provided here, but replaced by an LLVM pass instead
#       in order to provide some debug information without stack unwinding.


## GC

# FIXME: get rid of this and allow boxed types
T_prjlvalue() = convert(LLVMType, Any; allow_boxed=true)

function gc_pool_alloc(sz::Csize_t)
    ptr = malloc(sz)
    if ptr == C_NULL
        report_oom(sz)
        throw(OutOfMemoryError())
    end
    return unsafe_pointer_to_objref(ptr)
end

compile(gc_pool_alloc, Any, (Csize_t,), T_prjlvalue)

# expected functions for GC support
compile(:malloc, Ptr{Nothing}, (Csize_t,))


## boxing and unboxing

const tag_type = UInt
const tag_size = sizeof(tag_type)

const gc_bits = 0x3 # FIXME

# get the type tag of a type at run-time
@generated function type_tag(::Val{type_name}) where type_name
    @dispose ctx=Context() begin
        T_tag = convert(LLVMType, tag_type)
        T_ptag = LLVM.PointerType(T_tag)

        T_pjlvalue = convert(LLVMType, Any; allow_boxed=true)

        # create function
        llvm_f, _ = create_function(T_tag)
        mod = LLVM.parent(llvm_f)

        # this isn't really a function, but we abuse it to get the JIT to resolve the address
        typ = LLVM.Function(mod, "jl_" * String(type_name) * "_type",
                            LLVM.FunctionType(T_pjlvalue))

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            typ_var = bitcast!(builder, typ, T_ptag)

            tag = load!(builder, T_tag, typ_var)

            ret!(builder, tag)
        end

        call_function(llvm_f, tag_type)
    end
end

# we use `jl_value_ptr`, a Julia pseudo-intrinsic that can be used to box and unbox values

@inline @generated function box(val, ::Val{type_name}) where type_name
    sz = sizeof(val)
    allocsz = sz + tag_size

    # type-tags are ephemeral, so look them up at run time
    #tag = unsafe_load(convert(Ptr{tag_type}, type_name))
    tag = :( type_tag(Val(type_name)) )

    quote
        ptr = malloc($(Csize_t(allocsz)))

        # store the type tag
        ptr = convert(Ptr{tag_type}, ptr)
        Core.Intrinsics.pointerset(ptr, $tag | $gc_bits, #=index=# 1, #=align=# $tag_size)

        # store the value
        ptr = convert(Ptr{$val}, ptr+tag_size)
        Core.Intrinsics.pointerset(ptr, val, #=index=# 1, #=align=# $sz)

        unsafe_pointer_to_objref(ptr)
    end
end

@inline function unbox(obj, ::Type{T}) where T
    ptr = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)

    # load the value
    ptr = convert(Ptr{T}, ptr)
    Core.Intrinsics.pointerref(ptr, #=index=# 1, #=align=# sizeof(T))
end

# generate functions functions that exist in the Julia runtime (see julia/src/datatype.c)
for (T, t) in [Int8   => :int8,  Int16  => :int16,  Int32  => :int32,  Int64  => :int64,
               UInt8  => :uint8, UInt16 => :uint16, UInt32 => :uint32, UInt64 => :uint64,
               Bool => :bool, Float32 => :float32, Float64 => :float64]
    box_fn   = Symbol("box_$t")
    unbox_fn = Symbol("unbox_$t")
    @eval begin
        $box_fn(val)   = box($T(val), Val($(QuoteNode(t))))
        $unbox_fn(obj) = unbox(obj, $T)

        compile($box_fn, Any, ($T,), T_prjlvalue; llvm_name=$"ijl_$box_fn")
        compile($unbox_fn, $T, (Any,); llvm_name=$"ijl_$unbox_fn")
    end
end


end
