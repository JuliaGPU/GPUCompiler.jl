# alias scopes for kernel arguments
#
# A caller that inspects the arguments of a kernel launch (e.g. the arrays a kernel is
# called with) can establish facts that no compiler can prove: which pointers in the
# arguments reach disjoint memory. `kernel_argument_alias_classes` reports those facts, and
# the pass below attaches them to the memory accesses of the kernel as scoped `noalias`
# metadata, which is what `noalias` arguments become when LLVM inlines a function.
#
# The facts are only valid for the arguments the kernel was compiled for, so the caller is
# responsible for keying compilation on them (e.g. by including them in the job's params)
# and for not reusing the kernel with arguments that alias differently.

# Trace a pointer back to the pointer it was derived from in the arguments of `f`. Returns
# `(param, offset)`, identifying the pointer stored at byte `offset` of LLVM parameter
# `param`, or `nothing` when the pointer cannot be traced back to an argument.
struct ArgumentOrigins
    dl::LLVM.DataLayout
    params::Dict{LLVM.Argument,Int}
    cache::Dict{LLVM.Value,Any}
end

const PENDING = :pending

# where in the arguments does this aggregate value come from?
function aggregate_origin(o::ArgumentOrigins, val::LLVM.Value)
    if val isa LLVM.Argument
        idx = get(o.params, val, nothing)
        idx === nothing && return nothing
        return (idx, 0)
    elseif val isa LLVM.ExtractValueInst
        agg = val.operands[1]
        base = aggregate_origin(o, agg)
        base === nothing && return nothing
        offset = aggregate_offset(o.dl, agg.value_type, val.indices)
        offset === nothing && return nothing
        return (base[1], base[2] + offset)
    else
        return nothing
    end
end

function aggregate_offset(dl::LLVM.DataLayout, typ::LLVMType, idxs)
    offset = 0
    for idx in idxs
        if typ isa LLVM.StructType
            offset += LLVM.offsetof(dl, typ, idx + 1)
            typ = typ.elements[idx + 1]
        elseif typ isa LLVM.ArrayType
            typ = typ.element_type
            offset += idx * LLVM.abi_size(dl, typ)
        else
            return nothing
        end
    end
    return offset
end

# where in the (by-reference) arguments is this address?
function address_origin(o::ArgumentOrigins, ptr::LLVM.Value)
    offset = 0
    while true
        if ptr isa LLVM.Argument
            idx = get(o.params, ptr, nothing)
            idx === nothing && return nothing
            return (idx, offset)
        elseif ptr isa LLVM.GetElementPtrInst
            off = LLVM.constant_offset(ptr, o.dl)
            off === nothing && return nothing
            offset += Int(off)
            ptr = ptr.operands[1]
        elseif ptr isa LLVM.BitCastInst || ptr isa LLVM.AddrSpaceCastInst
            ptr = ptr.operands[1]
        else
            return nothing
        end
    end
end

function pointer_origin(o::ArgumentOrigins, val::LLVM.Value)
    cached = get(o.cache, val, nothing)
    cached === PENDING && return PENDING
    haskey(o.cache, val) && return cached
    o.cache[val] = PENDING
    origin = _pointer_origin(o, val)
    if origin === PENDING
        # only derived from a value that is still being traced; trace it again once that
        # value's origin is known
        delete!(o.cache, val)
    else
        o.cache[val] = origin
    end
    return origin
end

function merge_origins(o::ArgumentOrigins, vals)
    origin = PENDING
    for val in vals
        other = pointer_origin(o, val)
        # a cycle back to a value that is being traced adds no new origin
        other === PENDING && continue
        other === nothing && return nothing
        if origin === PENDING
            origin = other
        elseif origin != other
            return nothing
        end
    end
    return origin === PENDING ? nothing : origin
end

function _pointer_origin(o::ArgumentOrigins, val::LLVM.Value)
    if val isa LLVM.GetElementPtrInst || val isa LLVM.BitCastInst ||
       val isa LLVM.AddrSpaceCastInst || val isa LLVM.PtrToIntInst ||
       val isa LLVM.IntToPtrInst || val isa LLVM.FreezeInst
        return pointer_origin(o, val.operands[1])
    elseif val isa LLVM.ConstantExpr
        return nothing
    elseif val isa LLVM.AddInst || val isa LLVM.SubInst || val isa LLVM.OrInst ||
           val isa LLVM.AndInst
        # pointer arithmetic on integers: exactly one side may be derived from a pointer
        # XXX: this assumes the other side does not carry provenance of another argument
        lhs, rhs = val.operands
        a = pointer_origin(o, lhs)
        val isa LLVM.SubInst && return a
        b = pointer_origin(o, rhs)
        a === PENDING && return b
        b === PENDING && return a
        a === nothing && b === nothing && return nothing
        a !== nothing && b !== nothing && return nothing
        return something(a, b)
    elseif val isa LLVM.PHIInst
        return merge_origins(o, [first(x) for x in val.incoming])
    elseif val isa LLVM.SelectInst
        return merge_origins(o, val.operands[2:3])
    elseif val isa LLVM.ExtractValueInst
        return aggregate_origin(o, val)
    elseif val isa LLVM.LoadInst
        return address_origin(o, val.operands[1])
    else
        return nothing
    end
end

# the pointer operands of a memory-accessing instruction, or `nothing` if it isn't one
function accessed_pointers(inst::LLVM.Instruction)
    if inst isa LLVM.LoadInst
        return (inst.operands[1],)
    elseif inst isa LLVM.StoreInst
        return (inst.operands[2],)
    elseif inst isa LLVM.AtomicRMWInst || inst isa LLVM.AtomicCmpXchgInst
        return (inst.operands[1],)
    elseif inst isa LLVM.CallInst
        callee = inst.called_operand
        callee isa LLVM.Function || return nothing
        name = callee.name
        if startswith(name, "llvm.memcpy") || startswith(name, "llvm.memmove")
            return (inst.operands[1], inst.operands[2])
        elseif startswith(name, "llvm.memset")
            return (inst.operands[1],)
        end
    end
    return nothing
end

function new_scope_node(operands...)
    TemporaryMDNode() do temp
        replace_temporary!(temp, MDNode([temp.node, operands...]))
    end
end

function append_metadata!(inst::LLVM.Instruction, kind, nodes)
    md = LLVM.metadata(inst)
    existing = haskey(md, kind) ? collect(md[kind].operands) : Metadata[]
    md[kind] = MDNode(unique(vcat(existing, nodes)))
end

struct ArgumentAliasScopes
    job::CompilerJob
    entry::String
end

function (self::ArgumentAliasScopes)(mod::LLVM.Module)
    job = self.job
    spec = kernel_argument_alias_classes(job)
    spec === nothing && return false
    haskey(mod.functions, self.entry) || return false
    f = mod.functions[self.entry]
    isdeclaration(f) && return false

    # map Julia-level arguments to LLVM parameters
    args = classify_arguments(job, f.function_type)
    params = Dict{LLVM.Argument,Int}()
    for arg in args
        arg.idx === nothing && continue
        params[f.parameters[arg.idx]] = arg.idx
    end
    classes = Dict{Tuple{Int,Int},Int}()
    for (argidx, offset, class) in spec
        idx = args[argidx].idx
        idx === nothing && continue
        classes[(idx, offset)] = class
    end
    all_classes = sort!(unique(values(classes)))
    length(all_classes) >= 2 || return false

    # find the class of every access we can trace to an argument
    o = ArgumentOrigins(mod.datalayout, params, Dict{LLVM.Value,Any}())
    tagged = Tuple{LLVM.Instruction,Vector{Int}}[]
    for bb in f.blocks, inst in bb.instructions
        ptrs = accessed_pointers(inst)
        ptrs === nothing && continue
        inst_classes = Int[]
        for ptr in ptrs
            origin = pointer_origin(o, ptr)
            origin isa Tuple || (empty!(inst_classes); break)
            class = get(classes, origin, nothing)
            class === nothing && (empty!(inst_classes); break)
            push!(inst_classes, class)
        end
        isempty(inst_classes) && continue
        push!(tagged, (inst, unique!(inst_classes)))
    end
    isempty(tagged) && return false

    domain = new_scope_node(MDString("$(self.entry) arguments"))
    scopes = Dict(class => new_scope_node(domain, MDString("$(self.entry) argument class $class"))
                  for class in all_classes)
    for (inst, inst_classes) in tagged
        append_metadata!(inst, LLVM.MD_alias_scope, Metadata[scopes[c] for c in inst_classes])
        others = Metadata[scopes[c] for c in all_classes if !(c in inst_classes)]
        isempty(others) || append_metadata!(inst, LLVM.MD_noalias, others)
    end

    # memory of a class that the kernel doesn't write to is invariant for its duration,
    # which lets back-ends use non-coherent loads (`ld.global.nc` on NVPTX). that requires
    # every write in the kernel to be attributed to a class.
    kernel_argument_invariant_loads(job) || return true
    written = Set{Int}()
    tagged_classes = Dict{LLVM.Instruction,Vector{Int}}(tagged)
    all_attributed = true
    global_as = Set(LLVM.addrspace(first(accessed_pointers(inst)).value_type) for (inst, _) in tagged)
    for bb in f.blocks, inst in bb.instructions
        may_write(inst, global_as) || continue
        inst_classes = get(tagged_classes, inst, nothing)
        if inst_classes === nothing
            all_attributed = false
            break
        end
        union!(written, inst_classes)
    end
    if all_attributed
        for (inst, inst_classes) in tagged
            inst isa LLVM.LoadInst || continue
            any(in(written), inst_classes) && continue
            LLVM.metadata(inst)[LLVM.MD_invariant_load] = MDNode(Metadata[])
        end
    end

    return true
end

# can this instruction write memory that is visible to the kernel's arguments?
#
# XXX: this assumes that distinct non-generic address spaces are disjoint, which holds for
#      PTX (where `global_as` is the global address space) but not for every target.
function may_write(inst::LLVM.Instruction, global_as)
    if inst isa LLVM.StoreInst || inst isa LLVM.AtomicRMWInst ||
       inst isa LLVM.AtomicCmpXchgInst
        ptr = inst isa LLVM.StoreInst ? inst.operands[2] : inst.operands[1]
        as = LLVM.addrspace(ptr.value_type)
        return as == 0 || as in global_as
    elseif inst isa LLVM.FenceInst
        return false
    elseif inst isa LLVM.LoadInst
        # atomic and volatile loads order other memory operations
        return LLVM.API.LLVMGetVolatile(inst) != 0 ||
               LLVM.API.LLVMGetOrdering(inst) != LLVM.API.LLVMAtomicOrderingNotAtomic
    elseif inst isa LLVM.CallBase
        accessed_pointers(inst) !== nothing && return true
        callee = inst.called_operand
        callee isa LLVM.Function || return true
        # barriers order memory operations, but don't write memory themselves
        startswith(callee.name, "llvm.nvvm.barrier") && return false
        # LLVM's memory effects: 2 bits (ref, mod) per location, where argument memory
        # is location 0 and other memory location 2; inaccessible memory can't be ours
        for attrs in (LLVM.function_attributes(callee),
                      LLVM.function_attributes(inst))
            for attr in collect(attrs)
                attr isa LLVM.EnumAttribute || continue
                LLVM.kind(attr) == LLVM.kind(EnumAttribute("memory", 0)) || continue
                return (LLVM.value(attr) & ((0x2 << 0) | (0x2 << 4))) != 0
            end
        end
        return true
    end
    return false
end

ArgumentAliasScopesPass(job, entry) =
    ModulePass("GPUArgumentAliasScopes", ArgumentAliasScopes(job, entry))
