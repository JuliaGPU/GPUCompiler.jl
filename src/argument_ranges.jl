# Caller-supplied range assumptions. Their truth is the caller's responsibility.

const RangeInteger = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128}

# Resolve physical fields only on a compilation miss. This does not check the truth
# of the bounds: they are assumptions supplied by the caller.
function resolve_argument_range(@nospecialize(T), path::Tuple,
                                range::UnitRange{<:RangeInteger})
    offset = 0
    for key in path
        isbitstype(T) || throw(ArgumentError("argument ranges require isbits fields"))
        i = key isa Symbol ? Base.fieldindex(T, key) : key
        offset += Int(fieldoffset(T, i))
        T = fieldtype(T, i)
    end
    T <: RangeInteger || throw(ArgumentError("argument ranges require fixed-width integer leaves"))
    (offset=offset, lower=T(first(range)), upper=T(last(range)))
end

# Called before finish_module! lowers Julia's argument ABI. Load the integer fields of
# aggregates passed by reference, or spill aggregates passed by value to address their
# physical fields without depending on LLVM aggregate indices (which can omit ghosts).
function emit_argument_ranges!(@nospecialize(job::CompilerJob), entry::LLVM.Function)
    job.config.entry_abi === :specfunc ||
        throw(ArgumentError("argument ranges require entry_abi=:specfunc"))
    args = classify_arguments(job, entry.function_type)
    @dispose builder=IRBuilder() begin
        position!(builder, LLVM.after_phis(entry.entry))
        length(job.config.argument_ranges) == length(args) ||
            throw(ArgumentError("argument ranges must have one entry for the function and each argument"))
        for (arg, T, ranges) in zip(args, job.source.specTypes.parameters, job.config.argument_ranges)
            isempty(ranges) && continue
            arg.cc == GHOST && continue
            arg.cc in (BITS_VALUE, BITS_REF) ||
                throw(ArgumentError("argument ranges require isbits arguments"))
            for (path, range) in ranges
                fact = resolve_argument_range(T, path, range)
                leaf = typeof(fact.lower)
                # Use the storage width for memory accesses, including Bool.
                ty = LLVM.IntType(8 * sizeof(leaf))
                val = entry.parameters[arg.idx]
                if arg.cc == BITS_REF
                    offset = ConstantInt(LLVM.IntType(8 * sizeof(Int)), fact.offset)
                    ptr = inbounds_gep!(builder, LLVM.Int8Type(), val, [offset])
                    val = load!(builder, ty, ptr)
                    val.alignment = 1
                elseif !isempty(path)
                    slot = alloca!(builder, val.value_type)
                    store!(builder, val, slot)
                    offset = ConstantInt(LLVM.IntType(8 * sizeof(Int)), fact.offset)
                    ptr = inbounds_gep!(builder, LLVM.Int8Type(), slot, [offset])
                    val = load!(builder, ty, ptr)
                    val.alignment = 1
                end
                # Use the actual value type (not its storage width, notably for Bool).
                ty = val.value_type
                signed = leaf <: Signed
                lo = ConstantInt(ty, fact.lower)
                hi = ConstantInt(ty, fact.upper)
                if fact.lower == fact.upper
                    assume!(builder, icmp!(builder, LLVM.IntPredicate.EQ, val, lo))
                else
                    if fact.lower != typemin(leaf)
                        assume!(builder, icmp!(builder, signed ? LLVM.IntPredicate.SGE : LLVM.IntPredicate.UGE,
                                              val, lo))
                    end
                    if fact.upper != typemax(leaf)
                        assume!(builder, icmp!(builder, signed ? LLVM.IntPredicate.SLE : LLVM.IntPredicate.ULE,
                                              val, hi))
                    end
                end
            end
        end
    end
    return
end
