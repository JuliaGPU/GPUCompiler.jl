# LLVM legalization of binary64 operations and values
#
# This happens in two steps. First, every floating-point operation on `double` values is
# outlined into a call to a placeholder function that still has the original `double`
# signature, e.g. `fadd double %a, %b` becomes `call double @gpu_softfloat_add64(double %a,
# double %b)`. Then, the module is rebuilt with every `double` type replaced by `i64`
# (including in aggregates, globals, constants, signatures and attributes), which turns the
# placeholders into declarations of the integer-only routines that are linked afterwards.
#
# Splitting the semantic and representational changes keeps each step simple: outlining
# only looks at individual instructions, and the rebuild is a type-preserving clone that
# does not need to understand any operation.

mutable struct Float64TypeRemapper
    cache::Dict{LLVM.LLVMType,LLVM.LLVMType}
    suffix::String
end
Float64TypeRemapper(; suffix=".softfloat64") =
    Float64TypeRemapper(Dict{LLVM.LLVMType,LLVM.LLVMType}(), suffix)

function remap_type(remapper::Float64TypeRemapper, typ::LLVM.LLVMType)
    haskey(remapper.cache, typ) && return remapper.cache[typ]
    if typ == LLVM.DoubleType()
        mapped = LLVM.Int64Type()
    elseif typ isa LLVM.ArrayType
        mapped = LLVM.ArrayType(remap_type(remapper, eltype(typ)), length(typ))
    elseif typ isa LLVM.VectorType
        mapped = LLVM.VectorType(remap_type(remapper, eltype(typ)), length(typ))
    elseif typ isa LLVM.FunctionType
        mapped = LLVM.FunctionType(remap_type(remapper, LLVM.return_type(typ)),
            LLVM.LLVMType[remap_type(remapper, t) for t in LLVM.parameters(typ)];
            vararg=LLVM.isvararg(typ))
    elseif typ isa LLVM.PointerType
        mapped = LLVM.is_opaque(typ) ? typ :
            LLVM.PointerType(remap_type(remapper, eltype(typ)), LLVM.addrspace(typ))
    elseif typ isa LLVM.StructType
        if LLVM.isopaque(typ)
            mapped = typ
        elseif LLVM.name(typ) === nothing
            mapped = LLVM.StructType(
                LLVM.LLVMType[remap_type(remapper, t) for t in LLVM.elements(typ)];
                packed=LLVM.ispacked(typ))
        else
            mapped = LLVM.StructType(LLVM.name(typ) * remapper.suffix)
            remapper.cache[typ] = mapped
            LLVM.elements!(mapped,
                LLVM.LLVMType[remap_type(remapper, t) for t in LLVM.elements(typ)],
                LLVM.ispacked(typ))
        end
    else
        mapped = typ
    end
    remapper.cache[typ] = mapped
    mapped
end

function contains_double(typ::LLVM.LLVMType, seen=Set{LLVM.LLVMType}())
    typ == LLVM.DoubleType() && return true
    typ in seen && return false
    push!(seen, typ)
    if typ isa Union{LLVM.ArrayType,LLVM.VectorType}
        return contains_double(eltype(typ), seen)
    elseif typ isa LLVM.FunctionType
        return contains_double(LLVM.return_type(typ), seen) ||
               any(t -> contains_double(t, seen), LLVM.parameters(typ))
    elseif typ isa LLVM.PointerType
        return !LLVM.is_opaque(typ) && contains_double(eltype(typ), seen)
    elseif typ isa LLVM.StructType && !LLVM.isopaque(typ)
        return any(t -> contains_double(t, seen), LLVM.elements(typ))
    end
    false
end

function placeholder!(mod::LLVM.Module, symbol::String, return_type::LLVM.LLVMType,
                      argument_types::Vector{<:LLVM.LLVMType})
    ft = LLVM.FunctionType(return_type, argument_types)
    if haskey(LLVM.functions(mod), symbol)
        fn = LLVM.functions(mod)[symbol]
        LLVM.function_type(fn) == ft || error("soft-float placeholder ABI mismatch for $symbol")
        return fn
    end
    LLVM.Function(mod, symbol, ft)
end

function copy_semantic_metadata!(new::LLVM.Instruction, old::LLVM.Instruction)
    # LLVM.jl cannot enumerate instruction metadata keys, so copy the standardized kinds
    # relevant to arithmetic and memory semantics explicitly. Debug location is installed by
    # the IRBuilder before construction.
    # `fpmath` describes accuracy of a floating-point result and is invalid after
    # the call's result becomes an integer bit pattern.
    for key in ("prof", "range", "tbaa", "tbaa.struct", "alias.scope",
                "noalias", "nontemporal", "invariant.group")
        kind = LLVM.MDKind(key)
        haskey(LLVM.metadata(old), kind) || continue
        LLVM.metadata(new)[kind] = LLVM.metadata(old)[kind]
    end
    new
end

function outlined_call!(builder::LLVM.IRBuilder, inst::LLVM.Instruction, symbol::String,
                        args::Vector{<:LLVM.Value}, return_type::LLVM.LLVMType)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    fn = placeholder!(mod, symbol, return_type, LLVM.LLVMType[LLVM.value_type(x) for x in args])
    LLVM.position!(builder, inst)
    LLVM.debuglocation!(builder, inst)
    call = LLVM.call!(builder, LLVM.function_type(fn), fn, args, LLVM.name(inst))
    copy_semantic_metadata!(call, inst)
    call
end

# apply `f` per lane when `result_type` is a vector, extracting the lanes of vector `args`
function scalarize!(f, builder, result_type, args...)
    result_type isa LLVM.VectorType || return f(args...)
    result = LLVM.UndefValue(result_type)
    for lane in 0:length(result_type)-1
        index = LLVM.ConstantInt(Int32(lane))
        lane_args = map(args) do arg
            LLVM.value_type(arg) isa LLVM.VectorType ?
                LLVM.extract_element!(builder, arg, index) : arg
        end
        result = LLVM.insert_element!(builder, result, f(lane_args...), index)
    end
    result
end

scalar_type(typ) = typ isa LLVM.VectorType ? eltype(typ) : typ
is_binary64_type(typ) = scalar_type(typ) == LLVM.DoubleType()

function comparison_call!(builder, inst, f, a, b)
    # Julia's device ABI represents `Bool` returns as zero-extended i8, while LLVM `fcmp`
    # produces i1. Make that boundary explicit; calling an i8 helper through an i1 prototype
    # happens to verify with opaque pointers but returns incorrect predicates on Metal.
    raw = outlined_call!(builder, inst, helper_name(f), LLVM.Value[a, b], LLVM.Int8Type())
    LLVM.icmp!(builder, LLVM.API.LLVMIntNE, raw, LLVM.ConstantInt(UInt8(0)))
end

# lower an `fcmp` predicate to the eq/lt/le/unordered helpers
function comparison_value!(builder, inst, pred, a, b)
    P = LLVM.API
    eq() = comparison_call!(builder, inst, :eq64, a, b)
    lt() = comparison_call!(builder, inst, :lt64, a, b)
    le() = comparison_call!(builder, inst, :le64, a, b)
    gt() = comparison_call!(builder, inst, :lt64, b, a)
    ge() = comparison_call!(builder, inst, :le64, b, a)
    uno() = comparison_call!(builder, inst, :unordered64, a, b)
    if pred == P.LLVMRealPredicateFalse
        LLVM.ConstantInt(false)
    elseif pred == P.LLVMRealPredicateTrue
        LLVM.ConstantInt(true)
    elseif pred == P.LLVMRealOEQ
        eq()
    elseif pred == P.LLVMRealOGT
        gt()
    elseif pred == P.LLVMRealOGE
        ge()
    elseif pred == P.LLVMRealOLT
        lt()
    elseif pred == P.LLVMRealOLE
        le()
    elseif pred == P.LLVMRealONE
        LLVM.and!(builder, LLVM.not!(builder, eq()), LLVM.not!(builder, uno()))
    elseif pred == P.LLVMRealORD
        LLVM.not!(builder, uno())
    elseif pred == P.LLVMRealUNO
        uno()
    elseif pred == P.LLVMRealUEQ
        LLVM.or!(builder, eq(), uno())
    elseif pred == P.LLVMRealUGT
        LLVM.or!(builder, gt(), uno())
    elseif pred == P.LLVMRealUGE
        LLVM.or!(builder, ge(), uno())
    elseif pred == P.LLVMRealULT
        LLVM.or!(builder, lt(), uno())
    elseif pred == P.LLVMRealULE
        LLVM.or!(builder, le(), uno())
    elseif pred == P.LLVMRealUNE
        LLVM.not!(builder, eq())
    else
        error("unsupported binary64 fcmp predicate $pred")
    end
end

function extend_integer!(builder, value::LLVM.Value, signed::Bool)
    typ = LLVM.value_type(value)
    typ isa LLVM.IntegerType || error("binary64 conversion from non-integer $typ")
    width = LLVM.width(typ)
    width > 64 && error("binary64 conversion from i$width is unsupported")
    width == 64 && return value
    signed ? LLVM.sext!(builder, value, LLVM.Int64Type()) :
             LLVM.zext!(builder, value, LLVM.Int64Type())
end

# LLVM intrinsics on double values, by name prefix (covering both scalar and vector forms)
const INTRINSICS = [
    "llvm.sqrt." => :sqrt64,
    "llvm.fma." => :fma64,
    "llvm.fabs." => :abs64,
    "llvm.copysign." => :copysign64,
    "llvm.trunc." => :trunc64,
    "llvm.rint." => :rint64,
    "llvm.nearbyint." => :rint64,
    "llvm.roundeven." => :rint64,
    "llvm.round." => :round64,
    "llvm.floor." => :floor64,
    "llvm.ceil." => :ceil64,
    "llvm.minimum." => :minimum64,
    "llvm.maximum." => :maximum64,
    "llvm.minnum." => :minnum64,
    "llvm.maxnum." => :maxnum64,
]

# replace a binary64 operation with calls to the emulation routines, returning whether
# the instruction was outlined
function outline_instruction!(builder, inst::LLVM.Instruction)
    double = LLVM.DoubleType()
    result_type = LLVM.value_type(inst)
    operands = collect(LLVM.operands(inst))
    input_type = isempty(operands) ? nothing : LLVM.value_type(first(operands))
    LLVM.position!(builder, inst)
    LLVM.debuglocation!(builder, inst)
    call(f, args...) = outlined_call!(builder, inst, helper_name(f), LLVM.Value[args...], double)

    replacement = if inst isa LLVM.FAddInst && is_binary64_type(result_type)
        scalarize!((a, b) -> call(:add64, a, b), builder, result_type, operands...)
    elseif inst isa LLVM.FSubInst && is_binary64_type(result_type)
        # subtraction is addition of the negated operand (a sign flip, after inlining)
        scalarize!((a, b) -> call(:add64, a, call(:neg64, b)), builder, result_type, operands...)
    elseif inst isa LLVM.FMulInst && is_binary64_type(result_type)
        scalarize!((a, b) -> call(:mul64, a, b), builder, result_type, operands...)
    elseif inst isa LLVM.FDivInst && is_binary64_type(result_type)
        scalarize!((a, b) -> call(:div64, a, b), builder, result_type, operands...)
    elseif inst isa LLVM.FNegInst && is_binary64_type(result_type)
        scalarize!(a -> call(:neg64, a), builder, result_type, operands...)
    elseif inst isa LLVM.FRemInst && is_binary64_type(result_type)
        error("binary64 remainder (frem) is not supported")
    elseif inst isa LLVM.FCmpInst && is_binary64_type(input_type)
        pred = LLVM.predicate(inst)
        scalarize!((a, b) -> comparison_value!(builder, inst, pred, a, b),
                   builder, result_type, operands...)
    elseif inst isa Union{LLVM.SIToFPInst,LLVM.UIToFPInst} && is_binary64_type(result_type)
        signed = inst isa LLVM.SIToFPInst
        scalarize!(builder, result_type, operands...) do x
            call(signed ? :i64_to_f64 : :u64_to_f64, extend_integer!(builder, x, signed))
        end
    elseif inst isa Union{LLVM.FPToSIInst,LLVM.FPToUIInst} && is_binary64_type(input_type)
        T = scalar_type(result_type)
        T isa LLVM.IntegerType && LLVM.width(T) <= 64 ||
            error("unsupported conversion from Float64 to $T")
        f = inst isa LLVM.FPToSIInst ? :f64_to_i64 : :f64_to_u64
        scalarize!(builder, result_type, operands...) do x
            wide = outlined_call!(builder, inst, helper_name(f), LLVM.Value[x], LLVM.Int64Type())
            LLVM.width(T) == 64 ? wide : LLVM.trunc!(builder, wide, T)
        end
    elseif inst isa LLVM.FPExtInst && is_binary64_type(result_type)
        T = scalar_type(input_type)
        T in (LLVM.HalfType(), LLVM.FloatType()) ||
            error("unsupported conversion from $T to Float64")
        scalarize!(builder, result_type, operands...) do x
            # Widen the bits directly; native fpext may flush a half subnormal to zero.
            f, bits_type = T == LLVM.HalfType() ? (:f16_to_f64, LLVM.Int16Type()) :
                                                  (:f32_to_f64, LLVM.Int32Type())
            call(f, LLVM.bitcast!(builder, x, bits_type))
        end
    elseif inst isa LLVM.FPTruncInst && is_binary64_type(input_type)
        T = scalar_type(result_type)
        T in (LLVM.HalfType(), LLVM.FloatType()) ||
            error("unsupported conversion from Float64 to $T")
        f, bits_type = T == LLVM.HalfType() ? (:f64_to_f16, LLVM.Int16Type()) :
                                              (:f64_to_f32, LLVM.Int32Type())
        scalarize!(builder, result_type, operands...) do x
            bits = outlined_call!(builder, inst, helper_name(f), LLVM.Value[x], bits_type)
            LLVM.bitcast!(builder, bits, T)
        end
    elseif inst isa LLVM.CallInst && LLVM.called_operand(inst) isa LLVM.Function &&
           is_binary64_type(result_type)
        callee = LLVM.name(LLVM.called_operand(inst))
        i = findfirst(((prefix, _),) -> startswith(callee, prefix), INTRINSICS)
        i === nothing && return false
        f = INTRINSICS[i][2]
        scalarize!((args...) -> call(f, args...), builder, result_type,
                   collect(LLVM.arguments(inst))...)
    else
        return false
    end
    LLVM.replace_uses!(inst, replacement)
    LLVM.erase!(inst)
    true
end

function outline_float64!(mod::LLVM.Module)
    @dispose builder=LLVM.IRBuilder() begin
        worklist = LLVM.Instruction[]
        for fn in LLVM.functions(mod), bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            push!(worklist, inst)
        end
        for inst in worklist
            outline_instruction!(builder, inst)
        end
    end
    mod
end

function remap_attribute(remapper, attr::LLVM.Attribute)
    attr isa LLVM.TypeAttribute || return attr
    LLVM.TypeAttribute(LLVM.API.LLVMCreateTypeAttribute(
        LLVM.context(), LLVM.kind(attr), remap_type(remapper, LLVM.value(attr))))
end

function copy_global_properties!(new::LLVM.GlobalValue, old::LLVM.GlobalValue)
    LLVM.linkage!(new, LLVM.linkage(old))
    LLVM.visibility!(new, LLVM.visibility(old))
    LLVM.dllstorage!(new, LLVM.dllstorage(old))
    LLVM.unnamed_addr!(new, LLVM.unnamed_addr(old))
    LLVM.local_unnamed_addr!(new, LLVM.local_unnamed_addr(old))
    isempty(LLVM.section(old)) || LLVM.section!(new, LLVM.section(old))
    new
end

function copy_function_properties!(new::LLVM.Function, old::LLVM.Function)
    copy_global_properties!(new, old)
    LLVM.callconv!(new, LLVM.callconv(old))
    isempty(LLVM.gc(old)) || LLVM.gc!(new, LLVM.gc(old))
    for (old_arg, new_arg) in zip(LLVM.parameters(old), LLVM.parameters(new))
        LLVM.name!(new_arg, LLVM.name(old_arg))
    end
    new
end

# `nofpclass` constrains floating-point values, not their integer representations.
# Keep it for native floating-point arguments and results, but drop it for mapped ones.
function copy_attributes!(dest, source, remapper, typ=nothing)
    attrs = collect(source)  # source and destination may be the same call-site set
    for attr in collect(dest)
        delete!(dest, attr)
    end
    for attr in attrs
        if typ !== nothing && scalar_type(typ) isa LLVM.IntegerType &&
           attr isa LLVM.EnumAttribute &&
           LLVM.kind(attr) == LLVM.API.LLVMGetEnumAttributeKindForName("nofpclass", 9)
            continue
        end
        push!(dest, remap_attribute(remapper, attr))
    end
end

function copy_function_attributes!(new::LLVM.Function, old::LLVM.Function, remapper)
    # CloneFunctionInto copies attributes verbatim, including their embedded types.
    copy_attributes!(LLVM.function_attributes(new), LLVM.function_attributes(old), remapper)
    copy_attributes!(LLVM.return_attributes(new), LLVM.return_attributes(old), remapper,
                     LLVM.return_type(LLVM.function_type(new)))
    for (i, arg) in enumerate(LLVM.parameters(new))
        copy_attributes!(LLVM.parameter_attributes(new, i), LLVM.parameter_attributes(old, i),
                         remapper, LLVM.value_type(arg))
    end
    new
end

function map_constant(remapper, value_map::Dict{LLVM.Value,LLVM.Value}, value::LLVM.Value)
    haskey(value_map, value) && return value_map[value]
    value isa LLVM.Constant || error("cannot materialize non-constant LLVM value $value")
    typ = remap_type(remapper, LLVM.value_type(value))
    if value isa LLVM.ConstantFP && LLVM.value_type(value) == LLVM.DoubleType()
        return LLVM.const_bitcast(value, LLVM.Int64Type())
    elseif value isa LLVM.ConstantAggregateZero
        return LLVM.null(typ)
    elseif value isa LLVM.PointerNull
        return LLVM.PointerNull(typ)
    elseif value isa LLVM.UndefValue
        return LLVM.UndefValue(typ)
    elseif value isa LLVM.PoisonValue
        return LLVM.PoisonValue(typ)
    elseif value isa LLVM.ConstantArray
        # LLVM.jl's array interface descends to scalar leaves. Preserve one LLVM
        # aggregate level here, including zero/undef subarrays, and recurse ourselves.
        vals = LLVM.Constant[map_constant(remapper, value_map, x) for x in LLVM.operands(value)]
        return LLVM.ConstantArray(eltype(typ), vals)
    elseif value isa LLVM.ConstantDataArray
        vals = LLVM.Constant[map_constant(remapper, value_map, x) for x in collect(value)]
        return LLVM.ConstantArray(eltype(typ), vals)
    elseif value isa Union{LLVM.ConstantVector,LLVM.ConstantDataVector}
        vals = LLVM.Constant[map_constant(remapper, value_map, x) for x in collect(value)]
        return LLVM.Value(LLVM.API.LLVMConstVector(vals, length(vals)))
    elseif value isa LLVM.ConstantStruct
        vals = LLVM.Constant[map_constant(remapper, value_map, x) for x in LLVM.operands(value)]
        return LLVM.name(typ) === nothing ? LLVM.ConstantStruct(vals; packed=LLVM.ispacked(typ)) :
                                           LLVM.ConstantStruct(typ, vals)
    elseif value isa LLVM.ConstantExpr
        ops = LLVM.Constant[map_constant(remapper, value_map, x) for x in LLVM.operands(value)]
        opcode = LLVM.opcode(value)
        if opcode == LLVM.API.LLVMBitCast
            return LLVM.value_type(first(ops)) == typ ? first(ops) : LLVM.const_bitcast(first(ops), typ)
        elseif opcode == LLVM.API.LLVMAddrSpaceCast
            return LLVM.const_addrspacecast(first(ops), typ)
        elseif opcode == LLVM.API.LLVMPtrToInt
            return LLVM.const_ptrtoint(first(ops), typ)
        elseif opcode == LLVM.API.LLVMIntToPtr
            return LLVM.const_inttoptr(first(ops), typ)
        elseif opcode == LLVM.API.LLVMGetElementPtr
            source_type = remap_type(remapper,
                LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(value)))
            return LLVM.const_gep(source_type, first(ops), ops[2:end])
        end
        error("unsupported constant expression during binary64 legalization: opcode $opcode")
    elseif remap_type(remapper, LLVM.value_type(value)) != LLVM.value_type(value)
        error("unsupported changed constant $(typeof(value)) during binary64 legalization")
    end
    value
end

function rebuild_module!(mod::LLVM.Module, entry::LLVM.Function)
    LLVM.API.LLVMGetFirstGlobalAlias(mod) == C_NULL ||
        error("binary64 legalization does not yet support LLVM global aliases")
    LLVM.API.LLVMGetFirstGlobalIFunc(mod) == C_NULL ||
        error("binary64 legalization does not yet support LLVM ifuncs")
    remapper = Float64TypeRemapper()
    value_map = Dict{LLVM.Value,LLVM.Value}()
    entry_name = LLVM.name(entry)

    # Only globals whose type changes, and functions with a body, need to be rebuilt.
    # Other declarations are kept as they are: recreating an intrinsic declaration, in
    # particular, is not safe with typed pointers. This does require that any remaining
    # double-typed intrinsic declaration is unused (see `check_float64_calls`).
    for fn in collect(LLVM.functions(mod))
        if LLVM.isdeclaration(fn) && startswith(LLVM.name(fn), "llvm.") &&
           contains_double(LLVM.function_type(fn))
            isempty(LLVM.uses(fn)) || error("unsupported binary64 intrinsic: $(LLVM.name(fn))")
            LLVM.erase!(fn)
        end
    end
    old_functions = LLVM.Function[]
    for fn in LLVM.functions(mod)
        if LLVM.isdeclaration(fn) && !contains_double(LLVM.function_type(fn))
            value_map[fn] = fn
        else
            push!(old_functions, fn)
        end
    end
    old_globals = LLVM.GlobalVariable[]
    kept_globals = LLVM.GlobalVariable[]
    for gv in LLVM.globals(mod)
        if contains_double(LLVM.global_value_type(gv))
            push!(old_globals, gv)
        else
            value_map[gv] = gv
            push!(kept_globals, gv)
        end
    end

    for old in old_functions
        LLVM.name!(old, LLVM.name(old) * ".softfloat64.old")
    end
    for old in old_globals
        LLVM.name!(old, LLVM.name(old) * ".softfloat64.old")
    end

    for old in old_globals
        final_name = replace(LLVM.name(old), r"\.softfloat64\.old$" => "")
        typ = remap_type(remapper, LLVM.global_value_type(old))
        new = LLVM.GlobalVariable(mod, typ, final_name, LLVM.addrspace(LLVM.value_type(old)))
        copy_global_properties!(new, old)
        LLVM.constant!(new, LLVM.isconstant(old))
        LLVM.threadlocal!(new, LLVM.isthreadlocal(old))
        LLVM.threadlocalmode!(new, LLVM.threadlocalmode(old))
        LLVM.extinit!(new, LLVM.isextinit(old))
        LLVM.alignment!(new, LLVM.alignment(old))
        value_map[old] = new
    end
    for old in old_functions
        final_name = replace(LLVM.name(old), r"\.softfloat64\.old$" => "")
        new = LLVM.Function(mod, final_name, remap_type(remapper, LLVM.function_type(old)))
        copy_function_properties!(new, old)
        value_map[old] = new
        for (old_arg, new_arg) in zip(LLVM.parameters(old), LLVM.parameters(new))
            value_map[old_arg] = new_arg
        end
    end

    materializer(value) = map_constant(remapper, value_map, value)
    for old in old_functions
        LLVM.isdeclaration(old) && continue
        LLVM.clone_into!(value_map[old], old; value_map,
            changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges,
            type_mapper=t -> remap_type(remapper, t), materializer)
    end
    for old in old_functions
        copy_function_attributes!(value_map[old], old, remapper)
        for bb in LLVM.blocks(value_map[old]), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            attrs = LLVM.function_attributes(inst)
            copy_attributes!(attrs, attrs, remapper)
            attrs = LLVM.return_attributes(inst)
            copy_attributes!(attrs, attrs, remapper, LLVM.value_type(inst))
            for (i, arg) in enumerate(LLVM.arguments(inst))
                attrs = LLVM.argument_attributes(inst, i)
                copy_attributes!(attrs, attrs, remapper, LLVM.value_type(arg))
            end
        end
    end
    for old in old_globals
        init = LLVM.initializer(old)
        init === nothing || LLVM.initializer!(value_map[old], materializer(init))
    end
    # initializers of unchanged globals may still refer to rebuilt functions or globals
    for gv in kept_globals
        init = LLVM.initializer(gv)
        init === nothing && continue
        new_init = materializer(init)
        new_init == init || LLVM.initializer!(gv, new_init)
    end

    # Rewrite kernel annotations and other named metadata while both generations exist.
    for old in old_functions
        LLVM.replace_metadata_uses!(old, value_map[old])
    end
    for old in old_globals
        LLVM.replace_metadata_uses!(old, value_map[old])
    end

    # Remove all old bodies/initializers first so cross-references among old definitions no
    # longer prevent safe erasure.
    for old in old_functions
        LLVM.isdeclaration(old) || empty!(old)
    end
    for old in old_globals
        LLVM.initializer!(old, nothing)
    end
    foreach(LLVM.erase!, old_functions)
    foreach(LLVM.erase!, old_globals)

    LLVM.functions(mod)[entry_name]
end

# Opaque pointers hide the allocated/indexed type from result and operand types, and calls
# carry a function type independently of their callee pointer.
function embedded_type(inst::LLVM.Instruction)
    if inst isa LLVM.AllocaInst
        LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
    elseif inst isa LLVM.GetElementPtrInst
        LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
    elseif inst isa LLVM.CallInst
        LLVM.called_type(inst)
    else
        nothing
    end
end

typed_attributes(attrs) = [LLVM.value(attr) for attr in collect(attrs) if attr isa LLVM.TypeAttribute]

function attribute_types(value::Union{LLVM.Function,LLVM.CallInst})
    types = LLVM.LLVMType[]
    append!(types, typed_attributes(LLVM.function_attributes(value)))
    append!(types, typed_attributes(LLVM.return_attributes(value)))
    if value isa LLVM.Function
        for i in eachindex(LLVM.parameters(value))
            append!(types, typed_attributes(LLVM.parameter_attributes(value, i)))
        end
    else
        for i in eachindex(LLVM.arguments(value))
            append!(types, typed_attributes(LLVM.argument_attributes(value, i)))
        end
    end
    types
end

# call `f(typ, describe)` for every type in the module, where `describe()` returns a
# description of where the type occurs
function foreach_type(f, mod::LLVM.Module)
    for gv in LLVM.globals(mod)
        f(LLVM.global_value_type(gv), () -> "global $(LLVM.name(gv))")
    end
    for fn in LLVM.functions(mod)
        f(LLVM.function_type(fn), () -> "function $(LLVM.name(fn))")
        for typ in attribute_types(fn)
            f(typ, () -> "attribute of function $(LLVM.name(fn))")
        end
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            describe() = "instruction in $(LLVM.name(fn)): $inst"
            f(LLVM.value_type(inst), describe)
            typ = embedded_type(inst)
            typ === nothing || f(typ, describe)
            for op in LLVM.operands(inst)
                f(LLVM.value_type(op), describe)
            end
            inst isa LLVM.CallInst || continue
            for typ in attribute_types(inst)
                f(typ, describe)
            end
        end
    end
end

function uses_float64(mod::LLVM.Module)
    found = false
    foreach_type(mod) do typ, _
        found |= contains_double(typ)
    end
    found
end

# check that no trace of binary64 remains, returning a list of problems
function validate_module(mod::LLVM.Module)
    problems = String[]
    foreach_type(mod) do typ, describe
        contains_double(typ) && push!(problems, describe() * " retains double")
    end
    for fn in LLVM.functions(mod)
        if LLVM.isdeclaration(fn) && startswith(LLVM.name(fn), "llvm.") &&
           occursin(".f64", LLVM.name(fn)) && !isempty(LLVM.uses(fn))
            push!(problems, "f64 intrinsic survived: $(LLVM.name(fn))")
        end
    end
    problems
end

# reject calls with a binary64 ABI to functions we do not control
function check_float64_calls(mod::LLVM.Module)
    for fn in LLVM.functions(mod), bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        # byval/sret and other typed attributes are part of the ABI even when every
        # explicit argument is an opaque pointer. Check the declaration as well, since
        # not all attributes have to be repeated at the call site.
        (contains_double(LLVM.called_type(inst)) ||
         any(contains_double, attribute_types(inst)) ||
         (callee isa LLVM.Function && any(contains_double, attribute_types(callee)))) || continue
        # internal functions are rebuilt with their callers, but external functions,
        # indirect calls and inline assembly have no such ABI agreement
        if callee isa LLVM.Function
            !LLVM.isdeclaration(callee) && continue
            name = LLVM.name(callee)
            any(m -> m.llvm_name == name, METHODS) && continue
        end
        error("unsupported binary64 call during software legalization: $inst")
    end
end

# replace all binary64 operations and values in `mod`, returning the (rebuilt) entry
function legalize_module!(mod::LLVM.Module, entry::LLVM.Function)
    uses_float64(mod) || return entry
    outline_float64!(mod)
    check_float64_calls(mod)
    new_entry = rebuild_module!(mod, entry)
    LLVM.verify(mod)
    problems = validate_module(mod)
    isempty(problems) || error("incomplete binary64 legalization:\n" * join(problems, "\n"))
    new_entry
end
