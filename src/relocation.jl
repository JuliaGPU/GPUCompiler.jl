# Host-reference relocation lifecycle
#
#   produce ──▶ merge (on link) ──▶ prune (after DCE) ──▶ lower (at emit_asm)
#
# Julia value globals and libjulia runtime globals are represented by symbolic slots until
# the backend lowers them. Slot names identify their HostReference and, once assigned, must
# remain globally unique and stable through linking.
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
    HostReferences(slots, embedded_pointer)

Host-reference metadata accompanying a module. `slots` maps each object-code symbol to the
source of the word that must be written there. `embedded_pointer` conservatively records that
an unavoidable session pointer has been embedded, so the artifact cannot be serialized into a
package image even if `slots` is empty.
"""
mutable struct HostReferences
    slots::Dict{String,HostReference}
    embedded_pointer::Bool
end

HostReferences() = HostReferences(Dict{String,HostReference}(), false)
copy_host_references(refs::HostReferences) =
    HostReferences(copy(refs.slots), refs.embedded_pointer)

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
    lower_host_references!(job, mod, refs)

Backend hook for lowering live host references before object emission.

The default implementation resolves them in the current Julia process, making the result
session-dependent. Loaders may instead emit module-owned definitions that are patched after
loading, or external declarations that are defined before loading.
"""
function lower_host_references!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                refs::HostReferences)
    resolve_host_reference_slots!(mod, refs)
end


function classify_gvs!(mod::LLVM.Module, gv_to_value::Dict{String, Ptr{Cvoid}})
    refs = HostReferences()
    mod_gvs = globals(mod)
    for (name, init) in gv_to_value
        haskey(mod_gvs, name) || continue
        gv = mod_gvs[name]
        cur = initializer(gv)
        if !(cur === nothing || LLVM.isnull(cur))
            refs.embedded_pointer = true
            continue
        end

        # jl_get_llvm_gvs and jl_get_llvm_gv_inits report an initializer for every
        # mapped global, so a null here means those maps are out of sync.
        init == C_NULL && error("Missing Julia object for global '$name'")
        obj = Base.unsafe_pointer_to_objref(init)
        if isbitstype(typeof(obj)) && sizeof(obj) > 0 && !(obj isa Bool)
            val, hdr = materialize_box!(mod, gv, obj, init)
            initializer!(gv, val)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
            refs.embedded_pointer |= hdr >= UInt(64 << 4)
        else
            host_reference_slot_size(mod, gv, name)
            refs.slots[name] = JuliaValueRef(obj)
        end
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

function check_host_reference_slots!(mod::LLVM.Module, refs::HostReferences)
    mod_gvs = globals(mod)
    for name in keys(refs.slots)
        haskey(mod_gvs, name) || error("Missing host reference slot '$name'")
    end
    return
end

function prune_dead_host_reference_slots!(mod::LLVM.Module, refs::HostReferences)
    mod_gvs = globals(mod)
    for name in collect(keys(refs.slots))
        haskey(mod_gvs, name) || delete!(refs.slots, name)
    end
    return
end

function resolve_host_reference_slots!(mod::LLVM.Module, refs::HostReferences)
    check_host_reference_slots!(mod, refs)
    mod_gvs = globals(mod)
    for (name, ref) in refs.slots
        gv = mod_gvs[name]
        host_reference_slot_size(mod, gv, name)
        value = resolve_host_reference(ref)
        val = host_reference_slot_initializer(gv, value)
        initializer!(gv, val)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        constant!(gv, true)
        refs.embedded_pointer = true
    end
    empty!(refs.slots)
    return
end

"""
    emit_host_reference_definitions!(mod, refs)

Emit host-reference slots as writable, null-initialized definitions. The loader must patch each
definition after loading the object. This requires a per-object symbol namespace.
"""
function emit_host_reference_definitions!(mod::LLVM.Module, refs::HostReferences)
    check_host_reference_slots!(mod, refs)
    mod_gvs = globals(mod)
    slots = GlobalVariable[]
    for name in keys(refs.slots)
        gv = mod_gvs[name]
        host_reference_slot_size(mod, gv, name)
        initializer!(gv, null(global_value_type(gv)))
        constant!(gv, false)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
        extinit!(gv, true)
        push!(slots, gv)
    end
    isempty(slots) || set_used!(mod, slots...)
    return
end

"""
    emit_host_reference_declarations!(mod, refs)

Prepare host references for a loader that defines symbols before loading an object.

Runtime-global slots are restored to direct references to their named libjulia globals. Julia
value slots remain external word-sized declarations; the loader must define each symbol to
point at a cell containing [`resolve_host_reference`](@ref) and keep the cell and referenced
value alive while the code is executable.

Slot names are unique only within one compilation. A shared JIT namespace must uniquify them
or use a separate namespace for each object.
"""
function emit_host_reference_declarations!(mod::LLVM.Module, refs::HostReferences)
    check_host_reference_slots!(mod, refs)
    mod_gvs = globals(mod)
    for (name, ref) in collect(refs.slots)
        gv = mod_gvs[name]
        host_reference_slot_size(mod, gv, name)

        if ref isa CGlobalRef
            symbol = String(ref.symbol)
            if symbol == name
                initializer!(gv, nothing)
                constant!(gv, false)
                linkage!(gv, LLVM.API.LLVMExternalLinkage)
                extinit!(gv, false)
            else
                replacement = if haskey(globals(mod), symbol)
                    globals(mod)[symbol]
                elseif haskey(functions(mod), symbol)
                    functions(mod)[symbol]
                else
                    GlobalVariable(mod, host_reference_word_type(), symbol)
                end
                replacement = value_type(replacement) == value_type(gv) ? replacement :
                              const_pointercast(replacement, value_type(gv))
                replace_uses!(gv, replacement)
                erase!(gv)
            end
            delete!(refs.slots, name)
        else
            isdeclaration(gv) ||
                error("Julia value host reference slot '$name' must be a declaration")
            linkage!(gv, LLVM.API.LLVMExternalLinkage)
        end
    end
    return
end

function link_with_host_references!(dest_mod::LLVM.Module, dest_refs::HostReferences,
                                    src_mod::LLVM.Module, src_refs::HostReferences;
                                    only_needed=false)
    # Link-time optimization may already have removed an unreferenced global. This is the
    # last point where its absence is provably dead rather than a broken relocation.
    prune_dead_host_reference_slots!(dest_mod, dest_refs)
    prune_dead_host_reference_slots!(src_mod, src_refs)
    src_gvs = globals(src_mod)
    dest_names = Set{String}(LLVM.name(v) for v in [collect(globals(dest_mod));
                                                      collect(functions(dest_mod))])
    src_names = Set{String}(LLVM.name(v) for v in [collect(globals(src_mod));
                                                     collect(functions(src_mod))])

    function fresh_slot_name(name)
        base = safe_name(name) * "_gpucompiler"
        candidate = base
        suffix = 0
        while candidate in dest_names || candidate in src_names
            suffix += 1
            candidate = base * "_" * string(suffix)
        end
        return candidate
    end

    for (name, ref) in collect(src_refs.slots)
        dest_ref = get(dest_refs.slots, name, nothing)
        haskey(src_gvs, name) || error("Missing source host reference slot '$name'")
        if dest_ref !== nothing && same_host_reference(dest_ref, ref)
            haskey(globals(dest_mod), name) ||
                error("Missing destination host reference slot '$name'")
            continue
        end

        gv = src_gvs[name]
        if name in dest_names || dest_ref !== nothing
            delete!(src_names, name)
            LLVM.name!(gv, fresh_slot_name(name))
            push!(src_names, LLVM.name(gv))
        else
            continue
        end
        actual_name = LLVM.name(gv)
        delete!(src_refs.slots, name)
        haskey(src_refs.slots, actual_name) &&
            error("Duplicate source host reference slot '$actual_name'")
        src_refs.slots[actual_name] = ref
    end

    link!(dest_mod, src_mod; only_needed)
    for (name, ref) in src_refs.slots
        if !haskey(globals(dest_mod), name)
            only_needed && continue
            error("Linked host reference slot '$name' is missing")
        end
        dest_ref = get(dest_refs.slots, name, nothing)
        dest_ref === nothing && (dest_refs.slots[name] = ref)
        dest_ref === nothing || same_host_reference(dest_ref, ref) ||
            error("Conflicting linked host reference slot '$name'")
    end
    for name in keys(dest_refs.slots)
        haskey(globals(dest_mod), name) ||
            error("Merged host reference slot '$name' is missing")
    end
    dest_refs.embedded_pointer |= src_refs.embedded_pointer
    return
end

function relocate_gvs!(mod::LLVM.Module, gv_to_value::Dict{String, Ptr{Cvoid}})
    refs = classify_gvs!(mod, gv_to_value)
    refs.embedded_pointer |= !materialize_bool_singletons!(mod)
    resolve_host_reference_slots!(mod, refs)
    return !refs.embedded_pointer
end

# Bool JuliaVariables are absent from `gv_to_value`; define one device box per name.
function materialize_bool_singletons!(mod::LLVM.Module)
    portable = true
    mod_gvs = globals(mod)
    for (name, obj) in ("jl_true" => true, "jl_false" => false)
        haskey(mod_gvs, name) || continue
        gv = mod_gvs[name]
        cur = initializer(gv)
        if !(cur === nothing || LLVM.isnull(cur))
            # Existing definitions may contain session-specific addresses.
            portable = false
            continue
        end

        init = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)
        val, hdr = materialize_box!(mod, gv, obj, init)
        initializer!(gv, val)
        constant!(gv, true)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)

        # Stay conservative if Bool stops using a smalltag.
        portable &= hdr < UInt(64 << 4)   # jl_max_tags << 4
    end
    return portable
end

# emit a device-resident constant replica of the box holding `obj`; returns
# the constant to store in the slot, and the (gcbits-masked) header word
function materialize_box!(mod::LLVM.Module, gv::GlobalVariable, @nospecialize(obj),
                          init::Ptr{Cvoid})
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
    fields = LLVM.Constant[ConstantInt(T_word, hdr), ConstantDataArray(T_byte, bytes)]
    payload_idx = 1
    if Base.datatype_alignment(typeof(obj)) > W
        # pad so the payload lands at a 16-byte offset (JL_HEAP_ALIGNMENT max)
        pushfirst!(fields, ConstantDataArray(T_byte, zeros(UInt8, 16 - W)))
        payload_idx = 2
    end
    boxinit = ConstantStruct(fields)
    boxty = value_type(boxinit)

    box = GlobalVariable(mod, boxty, safe_name(LLVM.name(gv)) * "_box")
    initializer!(box, boxinit)
    constant!(box, true)
    linkage!(box, LLVM.API.LLVMPrivateLinkage)
    alignment!(box, 16)
    unnamed_addr!(box, true)

    idx(i) = ConstantInt(LLVM.Int32Type(), i)
    payload = const_gep(boxty, box, LLVM.Constant[idx(0), idx(payload_idx)])
    slotty = global_value_type(gv)
    val = value_type(payload) == slotty ? payload : const_addrspacecast(payload, slotty)
    return val, hdr
end


# some Julia code contains references to objects in the CPU run-time,
# without actually using the contents or functionality of those objects.
#
# prime example are type tags, which reference the address of the allocated type.
# since those references are ephemeral, we can't eagerly resolve and emit them in the IR,
# but at the same time the GPU can't resolve them at run-time.
#
# this pass performs that resolution at link time.
struct CollectRuntimeGlobalReferences
    job::CompilerJob
    refs::HostReferences
end
function (self::CollectRuntimeGlobalReferences)(mod::LLVM.Module)
    changed = false

    for f in [collect(functions(mod)); collect(globals(mod))]
        fn = LLVM.name(f)
        # Julia value globals are already represented by an identity in `refs`; they may
        # use a `jl_*` spelling but are not exported libjulia runtime globals.
        f isa LLVM.GlobalVariable && haskey(self.refs.slots, fn) && continue
        if isdeclaration(f) && (!(f isa LLVM.Function) || !LLVM.isintrinsic(f)) &&
           startswith(fn, "jl_")
            slot = nothing
            function runtime_global_slot()
                if slot === nothing
                    name = safe_name("gpu_" * fn)
                    slot = GlobalVariable(mod, host_reference_word_type(), name)
                    actual_name = LLVM.name(slot)
                    haskey(self.refs.slots, actual_name) &&
                        error("Duplicate Julia runtime global slot '$actual_name'")
                    self.refs.slots[actual_name] = CGlobalRef(Symbol(fn))
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
    end

    return changed
end
collect_runtime_global_references!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                   refs::HostReferences) =
    CollectRuntimeGlobalReferences(job, refs)(mod)
CollectRuntimeGlobalReferencesPass(job, refs=HostReferences()) =
    NewPMModulePass("CollectRuntimeGlobalReferences", CollectRuntimeGlobalReferences(job, refs))


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
