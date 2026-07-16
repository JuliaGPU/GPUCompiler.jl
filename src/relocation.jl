# Host-reference relocation lifecycle
#
#   produce ──▶ merge (on link) ──▶ prune (after DCE) ──▶ lower (at emit_asm)
#
# Julia values produce two relocation kinds: word-sized declaration slots and patches at a
# byte offset inside a defined global. Runtime globals produce slots. Names are globally
# unique and assigned once, so linking can merge IR and metadata without renaming.
#
# The two producers are `collect_julia_value_references!` during IR generation and
# `collect_runtime_global_references!` immediately before backend lowering.
#
# Generated code is portable only when `supports_relocatable_ir()` and the backend preserves
# these relocations for its loader. The default eager lowering bakes current-session values.
#
# This mirrors Julia's own mechanisms: codegen's identity-keyed global_targets slots, the
# sysimage jl_gvars table patched by jl_update_all_gvars, and JIT absoluteSymbols definitions.

"""
    JuliaValueRef(value)

A Julia value with a stable address (a heap object, symbol, or singleton), used as the
serializable identity of a host reference. Resolve it in the active session with
[`resolve_host_reference`](@ref); a loader that writes the resulting address into device
storage must keep `value` rooted for as long as that storage remains reachable.
"""
struct JuliaValueRef
    value::Any

    function JuliaValueRef(value)
        isbitstype(typeof(value)) && sizeof(value) > 0 &&
            error("JuliaValueRef requires an object with a stable address")
        new(value)
    end
end

"""
    CGlobalRef(symbol)

A named libjulia C data global. Resolution returns the word stored in that global in the
current Julia process.
"""
struct CGlobalRef
    symbol::Symbol
end

"""
    HostReference

A serializable source for a host-derived word: either a [`JuliaValueRef`](@ref) or a
[`CGlobalRef`](@ref).
"""
const HostReference = Union{JuliaValueRef,CGlobalRef}

"""
    HostReferences(slots, patches)

Host-reference metadata accompanying a module. `slots` maps word-sized declaration symbols to
the word resolved by a loader. `patches` maps a global name and byte offset to a word patched
after loading.
"""
struct HostReferences
    slots::Dict{String,HostReference}
    patches::Dict{Tuple{String,Int},HostReference}
end

HostReferences() = HostReferences(Dict{String,HostReference}(),
                                  Dict{Tuple{String,Int},HostReference}())

same_host_reference(a::JuliaValueRef, b::JuliaValueRef) = a.value === b.value
same_host_reference(a::CGlobalRef, b::CGlobalRef) = a.symbol === b.symbol
same_host_reference(::HostReference, ::HostReference) = false

"""
    resolve_host_reference(ref) -> UInt

Resolve a host reference to its word in the current Julia process.
"""
function resolve_host_reference(ref::JuliaValueRef)
    box = Any[ref.value]
    GC.@preserve box begin
        return unsafe_load(Base.unsafe_convert(Ptr{UInt}, pointer(box)))
    end
end
function resolve_host_reference(ref::CGlobalRef)
    address = ccall(:jl_cglobal, Any, (Any, Any), String(ref.symbol), UInt)
    return unsafe_load(address)
end

"""
    resolved_relocations(refs)

Resolve relocation metadata for a loader. Returns resolved `slots`, resolved `patches`, and
Julia `roots` that must stay alive while the loaded code can access them.
"""
function resolved_relocations(refs::HostReferences)
    slots = Pair{String,UInt}[]
    patches = Pair{Tuple{String,Int},UInt}[]
    roots = Any[]
    for (name, ref) in refs.slots
        push!(slots, name => resolve_host_reference(ref))
        ref isa JuliaValueRef && push!(roots, ref.value)
    end
    for (key, ref) in refs.patches
        push!(patches, key => resolve_host_reference(ref))
        ref isa JuliaValueRef && push!(roots, ref.value)
    end
    return (; slots, patches, roots)
end

"""
    lower_host_references!(job, mod, refs)

Backend hook for lowering live host references before object emission.

The default implementation resolves them in the current Julia process, making the result
session-dependent. Loaders may instead emit module-owned definitions that are patched after
loading, or external declarations that are defined before loading.

Backends using eager lowering must not persist generated code in `cached_results`. Generated
code is session-portable only when [`supports_relocatable_ir`](@ref) and the backend preserves
all slots and patches with definition- or declaration-based lowering, then applies them at
load time.
"""
function lower_host_references!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                refs::HostReferences)
    resolve_host_references!(mod, refs)
end


function collect_julia_value_references!(mod::LLVM.Module,
                                         gv_to_value::Dict{String, Ptr{Cvoid}})
    refs = HostReferences()
    mod_gvs = globals(mod)
    for (name, init) in gv_to_value
        haskey(mod_gvs, name) || continue
        gv = mod_gvs[name]
        cur = initializer(gv)
        if !(cur === nothing || LLVM.isnull(cur))
            @assert !supports_relocatable_ir()
            continue
        end

        # jl_get_llvm_gvs and jl_get_llvm_gv_inits report an initializer for every
        # mapped global, so a null here means those maps are out of sync.
        init == C_NULL && error("Missing Julia object for global '$name'")
        obj = Base.unsafe_pointer_to_objref(init)
        if isbitstype(typeof(obj)) && sizeof(obj) > 0 && !(obj isa Bool)
            val = materialize_box!(mod, refs, gv, obj, init)
            initializer!(gv, val)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        else
            host_reference_slot_size(mod, gv, name)
            slot_name = safe_name(name) * "_" * string(objectid(obj); base=16)
            LLVM.name!(gv, slot_name)
            LLVM.name(gv) == slot_name ||
                error("Host reference slot name '$slot_name' is already in use")
            refs.slots[slot_name] = JuliaValueRef(obj)
        end
    end

    # Bool JuliaVariables are absent from `gv_to_value`; define one device box per module.
    for (name, obj) in ("jl_true" => true, "jl_false" => false)
        haskey(mod_gvs, name) || continue
        gv = mod_gvs[name]
        cur = initializer(gv)
        if !(cur === nothing || LLVM.isnull(cur))
            @assert !supports_relocatable_ir()
            continue
        end

        init = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)
        val = materialize_box!(mod, refs, gv, obj, init)
        initializer!(gv, val)
        constant!(gv, true)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
    end
    return refs
end

host_reference_word_type() = LLVM.IntType(8sizeof(UInt))

function host_reference_slot_size(mod::LLVM.Module, gv::GlobalVariable, name::String)
    size = abi_size(datalayout(mod), global_value_type(gv))
    size == sizeof(UInt) ||
        error("Host reference slot '$name' has size $size, expected $(sizeof(UInt))")
    return
end

function host_reference_slot_initializer(gv::GlobalVariable, value::UInt)
    T = global_value_type(gv)
    if T isa LLVM.PointerType
        return const_inttoptr(ConstantInt(UInt64(value)), T)
    elseif T isa LLVM.IntegerType && width(T) == 8sizeof(UInt)
        return ConstantInt(T, value)
    end
    error("Host reference slot '$(LLVM.name(gv))' has unsupported LLVM type $T")
end

function foreach_host_reference_slot(f, mod::LLVM.Module, refs::HostReferences)
    mod_gvs = globals(mod)
    for (name, ref) in refs.slots
        haskey(mod_gvs, name) || error("Missing host reference slot '$name'")
        gv = mod_gvs[name]
        host_reference_slot_size(mod, gv, name)
        f(name, gv, ref)
    end
    return
end

function foreach_host_reference_patch(f, mod::LLVM.Module, refs::HostReferences)
    mod_gvs = globals(mod)
    for ((name, offset), ref) in refs.patches
        haskey(mod_gvs, name) || error("Missing host reference patch global '$name'")
        gv = mod_gvs[name]
        init = initializer(gv)
        init === nothing && error("Host reference patch global '$name' has no initializer")
        T = value_type(init)
        T isa LLVM.StructType ||
            error("Host reference patch global '$name' has non-struct initializer $T")
        size = abi_size(datalayout(mod), T)
        0 <= offset && offset + sizeof(UInt) <= size ||
            error("Host reference patch '$name+$offset' is outside its $size-byte global")
        f(name, offset, gv, ref)
    end
    return
end

function prune_dead_host_references!(mod::LLVM.Module, refs::HostReferences)
    mod_gvs = globals(mod)
    for name in collect(keys(refs.slots))
        haskey(mod_gvs, name) || delete!(refs.slots, name)
    end
    for ((name, offset), _) in collect(refs.patches)
        if !haskey(mod_gvs, name)
            delete!(refs.patches, (name, offset))
        elseif isempty(uses(mod_gvs[name]))
            delete!(refs.patches, (name, offset))
            erase!(mod_gvs[name])
        end
    end
    return
end

function resolve_host_references!(mod::LLVM.Module, refs::HostReferences)
    foreach_host_reference_slot(mod, refs) do name, gv, ref
        value = resolve_host_reference(ref)
        val = host_reference_slot_initializer(gv, value)
        initializer!(gv, val)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        constant!(gv, true)
    end
    foreach_host_reference_patch(mod, refs) do name, offset, gv, ref
        init = initializer(gv)
        T = value_type(init)::LLVM.StructType
        idx = Int(element_at(datalayout(mod), T, offset)) + 1
        fields = LLVM.Constant[operands(init)...]
        fields[idx] = ConstantInt(value_type(fields[idx]), resolve_host_reference(ref))
        initializer!(gv, ConstantStruct(T, fields))
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        extinit!(gv, false)
        constant!(gv, true)
        unnamed_addr!(gv, true)
    end
    empty!(refs.slots)
    empty!(refs.patches)
    return
end

"""
    emit_host_reference_definitions!(mod, refs)

Emit host-reference slots as writable, null-initialized definitions. The loader must patch each
definition and interior relocation after loading the object. This requires a per-object symbol
namespace.
"""
function emit_host_reference_definitions!(mod::LLVM.Module, refs::HostReferences)
    slots = GlobalVariable[]
    foreach_host_reference_slot(mod, refs) do _, gv, _
        initializer!(gv, null(global_value_type(gv)))
        constant!(gv, false)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
        extinit!(gv, true)
        push!(slots, gv)
    end
    foreach_host_reference_patch(mod, refs) do _, _, gv, _
        push!(slots, gv)
    end
    isempty(slots) || set_used!(mod, slots...)
    return
end

"""
    emit_host_reference_declarations!(mod, refs)

Prepare host references for a loader that defines symbols before loading an object.

Every slot remains an external word-sized declaration. The loader must define each symbol to
point at a cell containing [`resolve_host_reference`](@ref), and keep the cell and any
referenced Julia value alive while the code is executable. Interior patches cannot be defined
before load and must always be written afterward.
"""
function emit_host_reference_declarations!(mod::LLVM.Module, refs::HostReferences)
    used = GlobalVariable[]
    foreach_host_reference_slot(mod, refs) do name, gv, _
        isdeclaration(gv) || error("Host reference slot '$name' must be a declaration")
        constant!(gv, false)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
        extinit!(gv, false)
    end
    foreach_host_reference_patch(mod, refs) do _, _, gv, _
        push!(used, gv)
    end
    isempty(used) || set_used!(mod, used...)
    return
end

function link_with_host_references!(dest_mod::LLVM.Module, dest_refs::HostReferences,
                                    src_mod::LLVM.Module, src_refs::HostReferences;
                                    only_needed=false)
    # Patch globals are definitions, unlike slots. Make an identical source definition a
    # declaration so LLVM can resolve it to the destination definition while linking.
    for (key, ref) in src_refs.patches
        existing = get(dest_refs.patches, key, nothing)
        existing === nothing && continue
        name, offset = key
        same_host_reference(existing, ref) ||
            error("Host reference patch '$name+$offset' refers to conflicting values")
        haskey(globals(dest_mod), name) ||
            error("Missing destination host reference patch global '$name'")
        haskey(globals(src_mod), name) ||
            error("Missing source host reference patch global '$name'")
        gv = globals(src_mod)[name]
        initializer!(gv, nothing)
        extinit!(gv, false)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
    end

    link!(dest_mod, src_mod; only_needed)
    for (name, ref) in src_refs.slots
        # A slot absent from the linked module was dead (DCE'd or not imported under
        # `only_needed`); its relocation dies with it.
        haskey(globals(dest_mod), name) || continue
        existing = get(dest_refs.slots, name, nothing)
        existing === nothing || same_host_reference(existing, ref) ||
            error("Host reference slot '$name' refers to conflicting values")
        dest_refs.slots[name] = ref
    end
    for ((name, offset), ref) in src_refs.patches
        haskey(globals(dest_mod), name) || continue
        key = (name, offset)
        existing = get(dest_refs.patches, key, nothing)
        existing === nothing || same_host_reference(existing, ref) ||
            error("Host reference patch '$name+$offset' refers to conflicting values")
        dest_refs.patches[key] = ref
    end
    return
end

# emit a device-resident constant replica of the box holding `obj`; returns
# the constant to store in the slot, and the (gcbits-masked) header word
function materialize_box!(mod::LLVM.Module, refs::HostReferences, gv::GlobalVariable,
                          @nospecialize(obj), init::Ptr{Cvoid})
    @assert isbitstype(typeof(obj)) && sizeof(obj) > 0

    W = sizeof(Int)
    hdr, bytes = GC.@preserve obj begin
        # the header word transparently yields the smalltag immediate for
        # smalltag types and the host type pointer otherwise; drop the gcbits
        hdr = unsafe_load(Ptr{UInt}(init - W)) & ~UInt(15)
        bytes = [unsafe_load(Ptr{UInt8}(init), i) for i in 1:sizeof(obj)]
        hdr, bytes
    end

    T_word = LLVM.IntType(8W)
    T_byte = LLVM.Int8Type()
    patch_header = hdr >= UInt(64 << 4)   # jl_max_tags << 4
    fields = LLVM.Constant[ConstantInt(T_word, patch_header ? 0 : hdr),
                           ConstantDataArray(T_byte, bytes)]
    header_idx = 0
    payload_idx = 1
    if Base.datatype_alignment(typeof(obj)) > W
        # pad so the payload lands at a 16-byte offset (JL_HEAP_ALIGNMENT max)
        pushfirst!(fields, ConstantDataArray(T_byte, zeros(UInt8, 16 - W)))
        header_idx = 1
        payload_idx = 2
    end
    boxinit = ConstantStruct(fields)
    boxty = value_type(boxinit)

    box_name = if patch_header
        safe_name(LLVM.name(gv)) * "_" * string(objectid(obj); base=16) * "_box"
    else
        safe_name(LLVM.name(gv)) * "_box"
    end
    box = GlobalVariable(mod, boxty, box_name)
    LLVM.name(box) == box_name || error("Host reference patch global '$box_name' is already in use")
    initializer!(box, boxinit)
    alignment!(box, 16)
    if patch_header
        constant!(box, false)
        linkage!(box, LLVM.API.LLVMExternalLinkage)
        extinit!(box, true)
        offset = Int(offsetof(datalayout(mod), boxty, header_idx))
        refs.patches[(box_name, offset)] = JuliaValueRef(typeof(obj))
    else
        constant!(box, true)
        linkage!(box, LLVM.API.LLVMPrivateLinkage)
        unnamed_addr!(box, true)
    end

    idx(i) = ConstantInt(LLVM.Int32Type(), i)
    payload = const_gep(boxty, box, LLVM.Constant[idx(0), idx(payload_idx)])
    slotty = global_value_type(gv)
    val = value_type(payload) == slotty ? payload : const_addrspacecast(payload, slotty)
    return val
end


# some Julia code contains references to objects in the CPU run-time,
# without actually using the contents or functionality of those objects.
#
# prime example are type tags, which reference the address of the allocated type.
# since those references are ephemeral, we can't eagerly resolve and emit them in the IR,
# but at the same time the GPU can't resolve them at run-time.
#
# this collection performs that resolution at object-emission time.
function is_runtime_global_candidate(value, refs::HostReferences)
    name = LLVM.name(value)
    value isa LLVM.GlobalVariable && haskey(refs.slots, name) && return false
    isdeclaration(value) || return false
    value isa LLVM.Function && LLVM.isintrinsic(value) && return false
    return startswith(name, "jl_")
end

function collect_runtime_global_references!(mod::LLVM.Module, refs::HostReferences)
    changed = false

    for f in [collect(functions(mod)); collect(globals(mod))]
        is_runtime_global_candidate(f, refs) || continue
        fn = LLVM.name(f)
        slot = nothing
        function runtime_global_slot()
            if slot === nothing
                name = "gpu_" * fn
                slot = GlobalVariable(mod, host_reference_word_type(), name)
                LLVM.name(slot) == name ||
                    error("Julia runtime global slot name '$name' is already in use")
                refs.slots[name] = CGlobalRef(Symbol(fn))
            end
            slot
        end

        function replace_bindings!(value)
            changed = false
            for use in collect(uses(value))
                val = user(use)
                if isa(val, LLVM.ConstantExpr)
                    # recurse
                    changed |= replace_bindings!(val)
                elseif isa(val, LLVM.LoadInst)
                    T = value_type(val)
                    if !(T isa LLVM.PointerType ||
                         (T isa LLVM.IntegerType && width(T) == 8sizeof(UInt)))
                        error("Unsupported Julia runtime global '$fn' load of LLVM type $T")
                    end
                    @dispose builder=IRBuilder() begin
                        position!(builder, val)
                        replacement = load!(builder, host_reference_word_type(),
                                            runtime_global_slot())
                        if T isa LLVM.PointerType
                            replacement = inttoptr!(builder, replacement, T)
                        end
                        replace_uses!(val, replacement)
                    end
                    erase!(val)
                    changed = true
                end
            end
            changed
        end

        changed |= replace_bindings!(f)
    end

    return changed
end

function has_unresolved_runtime_global_loads(mod::LLVM.Module, refs::HostReferences)
    function has_load(value)
        for use in uses(value)
            val = user(use)
            val isa LLVM.LoadInst && return true
            val isa LLVM.ConstantExpr && has_load(val) && return true
        end
        return false
    end

    for value in [collect(functions(mod)); collect(globals(mod))]
        is_runtime_global_candidate(value, refs) || continue
        has_load(value) && return true
    end
    return false
end


function referenced_object(value, refs::HostReferences)
    # This is best-effort: optimized shapes fall back to the unknown-binding error path.
    while value isa ConstantExpr &&
          opcode(value) in (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast)
        value = first(operands(value))
    end
    if value isa LLVM.LoadInst
        source = first(operands(value))
        while source isa ConstantExpr &&
              opcode(source) in (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast)
            source = first(operands(source))
        end
        if source isa GlobalVariable
            ref = get(refs.slots, LLVM.name(source), nothing)
            ref isa JuliaValueRef && return Some(ref.value)
        end
    elseif value isa ConstantExpr && opcode(value) == LLVM.API.LLVMIntToPtr
        ptr = Ptr{Cvoid}(convert(Int, first(operands(value))))
        return Some(Base.unsafe_pointer_to_objref(ptr))
    end
    return nothing
end
