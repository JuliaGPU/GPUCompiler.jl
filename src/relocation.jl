# Relocations name words in a module that hold a host address: a reference to a Julia value,
# or a word read from a named C global. Each is recorded as a typed [`Relocation`](@ref) whose
# `kind` says what the site *is*; no lowering infers that from the IR shape.
#
#   produce ──▶ merge (on link) ──▶ prune (after DCE) ──▶ lower
#
# Site names are fixed at creation and namespaced by their producer, so IR and metadata can
# be linked without renaming. Unrelated jobs use distinct names; runtime functions linked
# into several outputs reuse names whose definitions and targets are identical.


## targets

"""
    JuliaValueRef(value)

A Julia value used as the serializable identity of a relocation target. Resolving it with
[`resolve_relocation_target`](@ref) permanently roots the value and returns the address of
its canonical representation. This also gives nonzero-sized `isbits` values a stable box.
"""
struct JuliaValueRef
    value::Any
end

"""
    CGlobalRef(symbol, library=nothing; offset=0)

A named C data global. With `library === nothing`, resolution uses `jl_cglobal`'s
process-wide lookup. Otherwise it looks up `symbol` in `library`. Resolution returns the
word stored at byte `offset`.
"""
struct CGlobalRef
    symbol::Symbol
    library::Union{Nothing,String}
    offset::Int

    function CGlobalRef(symbol::Symbol, library::Union{Nothing,String}=nothing;
                        offset::Integer=0)
        offset >= 0 || throw(ArgumentError("cglobal offset must be nonnegative"))
        new(symbol, library, Int(offset))
    end
end

"""
    RelocationTarget

A serializable target for a relocated word: either a [`JuliaValueRef`](@ref) or a
[`CGlobalRef`](@ref).
"""
const RelocationTarget = Union{JuliaValueRef,CGlobalRef}

same_relocation_target(a::JuliaValueRef, b::JuliaValueRef) = a.value === b.value
same_relocation_target(a::CGlobalRef, b::CGlobalRef) =
    a.symbol === b.symbol && a.library == b.library && a.offset == b.offset
same_relocation_target(::RelocationTarget, ::RelocationTarget) = false

# Permanently root a value in the current process and return the canonical rooted
# instance, exactly as Julia's own codegen does for values referenced from native code
# (`jl_ensure_rooted` on 1.10/1.11, `aot_optimize_roots` on 1.12+). Values compiled in
# this session are already rooted this way, making this a cheap lookup; it matters for
# metadata deserialized from a cache, whose values codegen never saw. Rooting by egal
# identity also folds such duplicates onto the instance native code already uses.
function root_relocation_target(target::JuliaValueRef)
    @static if VERSION >= v"1.11-"
        ccall(:jl_as_global_root, Any, (Any, Cint), target.value, 1)
    else
        ccall(:jl_as_global_root, Any, (Any,), target.value)
    end
end

value_pointer(@nospecialize(value)) = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), value))

"""
    resolve_relocation_target(target) -> UInt

Resolve a relocation target to its word in the current Julia process. Julia values are
permanently rooted (and canonicalized by egal identity) as part of resolution, so the
returned address cannot dangle.
"""
function resolve_relocation_target(target::JuliaValueRef)
    value_pointer(root_relocation_target(target))
end
function resolve_relocation_target(target::CGlobalRef)
    if target.library === nothing
        # `jl_cglobal` accepts the symbol directly and does the process-wide `jl_dlfind`.
        address = ccall(:jl_cglobal, Any, (Any, Any), target.symbol, UInt)
        return unsafe_load(address + target.offset)
    end
    handle = Libdl.dlopen(target.library)
    address = Libdl.dlsym(handle, target.symbol)
    return unsafe_load(Ptr{UInt}(address) + target.offset)
end


## the table

"""
    RelocationSiteKind

What a relocation record points at, and hence how it is lowered:

- `SlotSite`: a word-sized global the code loads through (GOT-style). Produced for
  references to Julia values and for `cglobal` words.
- `InteriorSite`: a word inside a definition's initializer, namely the header of a
  materialized box (see `materialize_box!`).
"""
@enum RelocationSiteKind SlotSite InteriorSite

"""
    Relocation(kind, name, offset, target)

One word to relocate: a global `name` (unique to the job that produced it), a byte `offset`
within that global (always zero for a [`SlotSite`](@ref RelocationSiteKind)), the site
`kind`, and the [`RelocationTarget`](@ref) whose address belongs there.
"""
struct Relocation
    kind::RelocationSiteKind
    name::String
    offset::Int
    target::RelocationTarget

    function Relocation(kind::RelocationSiteKind, name::String, offset::Int,
                        target::RelocationTarget)
        offset >= 0 || throw(ArgumentError("relocation offset must be nonnegative"))
        kind === SlotSite && offset != 0 &&
            throw(ArgumentError("a relocation slot must have offset zero"))
        new(kind, name, offset, target)
    end
end

# Records are kept sorted by this key, which is also their identity: at most one record per
# word. Ordering makes the record vector a deterministic manifest, which is what lets the
# `:table` lowering index the words by rank reproducibly.
relocation_key(rec::Relocation) = (rec.name, rec.offset)

"""
    Relocations(records)

Relocation metadata accompanying a module: [`Relocation`](@ref) records sorted by
`(name, offset)`. See [`resolved_relocations`](@ref) and
[`resolved_relocation_table`](@ref) for handing them to a loader.

Lowering is the end of the manifest's mutable life (`produce → merge → prune → lower →
freeze`): from then on it describes emitted code, which cannot be renegotiated, so the
mutators refuse to touch it. Work on a [`copy`](@ref) if you need a mutable one.
"""
struct Relocations
    records::Vector{Relocation}
    # The `:table` lowering's word order, materialized by it so that the delivered words
    # cannot be desynced from the indices it baked into the code (see
    # `emit_table_relocations!`). Empty for every other strategy.
    table::Vector{RelocationTarget}
    frozen::Base.RefValue{Bool}
end

Relocations(records::Vector{Relocation}) =
    Relocations(records, RelocationTarget[], Ref(false))
Relocations() = Relocations(Relocation[])

# Resolving into IR consumes the records; loaders (and anything else working after lowering)
# copy cached metadata first, which is also how they get a mutable manifest back.
Base.copy(relocs::Relocations) =
    Relocations(copy(relocs.records), copy(relocs.table), Ref(false))
Base.isempty(relocs::Relocations) = isempty(relocs.records)
Base.length(relocs::Relocations) = length(relocs.records)

# Lowering has committed the manifest to emitted code: adding, removing or reordering a
# record now silently desyncs it from that code — a `:table` index shifts onto the wrong
# word, a `:patch` definition is left holding a zero. Refuse instead.
freeze!(relocs::Relocations) = (relocs.frozen[] = true; relocs)

function check_mutable(relocs::Relocations, what::String)
    relocs.frozen[] &&
        error("""Cannot $what a relocation manifest that has already been lowered: its
                 records describe emitted code. Work on a `copy` instead.""")
    return
end

# Binary-search `records` for `key`: the index it occupies, or the index it would be
# inserted at, plus whether it is present.
function relocation_index(records::Vector{Relocation}, key::Tuple{String,Int})
    lo, hi = 1, length(records)
    while lo <= hi
        mid = (lo + hi) >>> 1
        found = relocation_key(records[mid])
        if found < key
            lo = mid + 1
        elseif found > key
            hi = mid - 1
        else
            return mid, true
        end
    end
    return lo, false
end

# Record `rec`, keeping `records` sorted. A record for the same word must agree on
# everything but is otherwise accepted, so that linking two modules that both reference a
# value merges their metadata.
function add_relocation!(relocs::Relocations, rec::Relocation)
    check_mutable(relocs, "add to")
    records = relocs.records
    idx, present = relocation_index(records, relocation_key(rec))
    if present
        existing = records[idx]
        same_relocation_target(existing.target, rec.target) ||
            error("Relocation '$(rec.name)+$(rec.offset)' refers to conflicting values")
        existing.kind === rec.kind ||
            error("Relocation '$(rec.name)+$(rec.offset)' is recorded as both " *
                  "$(existing.kind) and $(rec.kind)")
        return existing
    end
    insert!(records, idx, rec)
    return rec
end

add_relocation!(relocs::Relocations, kind::RelocationSiteKind, name::String, offset::Int,
                target::RelocationTarget) =
    add_relocation!(relocs, Relocation(kind, name, offset, target))

# The record for `(name, offset)`, or `nothing`.
function find_relocation(relocs::Relocations, name::String, offset::Int=0)
    idx, present = relocation_index(relocs.records, (name, offset))
    return present ? relocs.records[idx] : nothing
end

"""
    resolved_relocations(relocs) -> Vector{Pair{Relocation,UInt}}

Resolve relocation metadata for a `:patch` loader, returning each record with its resolved
word. Resolution permanently roots referenced Julia values in the process, so the addresses
stay valid for the lifetime of the session.
"""
function resolved_relocations(relocs::Relocations)
    return Pair{Relocation,UInt}[rec => resolve_relocation_target(rec.target)
                                 for rec in relocs.records]
end

"""
    resolved_relocation_table(relocs) -> Vector{UInt}

Resolve relocation metadata for a `:table` loader, returning the words in the order the
`:table` lowering indexed them by. Resolution permanently roots referenced Julia values in
the process, so the addresses stay valid for the lifetime of the session.
"""
function resolved_relocation_table(relocs::Relocations)
    isempty(relocs.table) && !isempty(relocs.records) &&
        error("""This manifest has $(length(relocs.records)) relocation record(s) but no
                 lowered table, so the code that reads it was never rewritten. Hand the
                 manifest to `emit_asm` (the 4-argument form) rather than emitting the
                 module with an empty one.""")
    return UInt[resolve_relocation_target(target) for target in relocs.table]
end

relocation_word_type() = LLVM.IntType(8sizeof(UInt))

function check_slot_size(mod::LLVM.Module, gv::GlobalVariable, name::String)
    size = LLVM.abi_size(mod.datalayout, gv.global_value_type)
    size == sizeof(UInt) ||
        error("Relocation slot '$name' has size $size, expected $(sizeof(UInt))")
    return
end

function slot_initializer(gv::GlobalVariable, value::UInt)
    T = gv.global_value_type
    if T isa LLVM.PointerType
        return const_inttoptr(ConstantInt(UInt64(value)), T)
    elseif T isa LLVM.IntegerType && T.width == 8sizeof(UInt)
        return ConstantInt(T, value)
    end
    error("Relocation slot '$(gv.name)' has unsupported LLVM type $T")
end

# Validate `gv` against what `rec` says it is. The record is authoritative: a mismatch means
# the metadata and the IR have drifted apart, which every lowering would otherwise turn into
# a silently wrong word.
function check_relocation(mod::LLVM.Module, rec::Relocation, gv::GlobalVariable)
    if rec.kind === SlotSite
        check_slot_size(mod, gv, rec.name)
    else
        isdeclaration(gv) &&
            error("Interior relocation '$(rec.name)' is a declaration")
        init = gv.initializer
        init === nothing && error("Relocation global '$(rec.name)' has no initializer")
        T = init.value_type
        T isa LLVM.StructType ||
            error("Relocation global '$(rec.name)' has non-struct initializer $T")
        size = LLVM.abi_size(mod.datalayout, T)
        rec.offset + sizeof(UInt) <= size ||
            error("Relocation '$(rec.name)+$(rec.offset)' is outside its $size-byte global")
    end
    return
end

function foreach_relocation(f, mod::LLVM.Module, relocs::Relocations)
    mod_gvs = mod.globals
    for rec in relocs.records
        gv = get(mod_gvs, rec.name, nothing)
        gv === nothing && error("Missing relocation global '$(rec.name)'")
        check_relocation(mod, rec, gv)
        f(rec, gv)
    end
    return
end


## producers

# Julia names value globals `<base>_<counter>`, where `<counter>` comes from a process-global
# codegen sequence and so differs from one session to the next. Drop it from relocation slot and
# box names: the target's `objectid` (appended alongside) is the stable per-target identity that
# disambiguates them, so any bitcode keyed on these names stays reproducible across sessions.
# `objectid` is content-stable for the interned symbols and `isbits`/`DataType` values that
# appear as relocation targets.
strip_codegen_counter(name::AbstractString) = replace(name, r"_[0-9]+$" => "")

# A namespace for this job's relocation site names, so that no two kernels — nor a kernel and
# the runtime library linked into it — can ever define the same site symbol. That makes the
# names globally unique by construction, which is what lets `:patch` loaders share one symbol
# namespace (e.g. an ORC `JITDylib` holding several compiled functions) without renaming.
#
# The job's entry name is the natural discriminator, but it is only fixed later in `irgen`, and
# for an unnamed non-kernel job it would carry Julia's per-session codegen counter. Derive the
# same name deterministically instead: the configured name if there is one, otherwise the
# mangled signature (which is literally the entry name for kernels).
relocation_namespace(@nospecialize(job::CompilerJob)) =
    job.config.name !== nothing ? safe_name(job.config.name) :
                                  mangle_sig(job.source.specTypes)

# Site names are used as symbols by loaders, so they must survive every back-end's assembler
# (`ptxas` in particular rejects anything outside `[A-Za-z0-9_$]`); `safe_name` guarantees that
# and `_` is the only separator available.
namespaced_name(namespace::String, base::AbstractString) = namespace * "_" * base

function collect_julia_value_relocations!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                         gv_to_value::Dict{String, Ptr{Cvoid}})
    relocs = Relocations()
    namespace = relocation_namespace(job)
    mod_gvs = mod.globals

    # Device jobs cannot refer to host boxes, so embed copies of boxed isbits constants.
    # Runtime jobs must keep the GC-managed boxes because returned values may re-enter Julia.
    materialize_boxes = !uses_julia_runtime(job)
    for (name, init) in gv_to_value
        gv = get(mod_gvs, name, nothing)
        gv === nothing && continue
        cur = gv.initializer
        if !(cur === nothing || LLVM.isnull(cur))
            @assert !supports_relocatable_ir()
            continue
        end

        # jl_get_llvm_gvs and jl_get_llvm_gv_inits report an initializer for every
        # mapped global, so a null here means those maps are out of sync.
        init == C_NULL && error("Missing Julia object for global '$name'")
        obj = Base.unsafe_pointer_to_objref(init)
        if materialize_boxes && isbitstype(typeof(obj)) && sizeof(typeof(obj)) > 0 &&
           !(obj isa Bool)
            val = materialize_box!(mod, relocs, namespace, gv, obj, init)
            gv.initializer = val
            gv.linkage = LLVM.Linkage.Private
        else
            check_slot_size(mod, gv, name)
            slot_name = namespaced_name(namespace,
                strip_codegen_counter(safe_name(name)) * "_" *
                string(objectid(obj); base=16))
            # Codegen can emit several slots for one value in a module (observed on 1.11,
            # whose backported GV API does not deduplicate them), and their content-derived
            # names collide by construction. Alias later slots onto the first: an equal name
            # means an equal referenced value, and `add_relocation!` below degenerates into
            # its agreeing-duplicate no-op (or errors on the astronomically unlikely
            # `objectid` collision between distinct values).
            existing = get(mod_gvs, slot_name, nothing)
            if existing !== nothing && existing !== gv
                @assert existing.value_type == gv.value_type
                replace_uses!(gv, existing)
                erase!(gv)
            else
                gv.name = slot_name
                gv.name == slot_name ||
                    error("Relocation slot name '$slot_name' is already in use")
            end
            add_relocation!(relocs, SlotSite, slot_name, 0, JuliaValueRef(obj))
        end
    end

    # Bool globals are absent from `gv_to_value`. Device jobs need local boxes; runtime jobs
    # leave them as external cglobals for `collect_cglobal_relocations!`.
    if materialize_boxes
        for (name, obj) in ("jl_true" => true, "jl_false" => false)
            gv = get(mod_gvs, name, nothing)
            gv === nothing && continue
            cur = gv.initializer
            if !(cur === nothing || LLVM.isnull(cur))
                @assert !supports_relocatable_ir()
                continue
            end

            init = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)
            val = materialize_box!(mod, relocs, namespace, gv, obj, init)
            gv.initializer = val
            gv.constant = true
            gv.linkage = LLVM.Linkage.Private
        end
    end
    return relocs
end

# Emit a device-resident constant replica of the box holding `obj` and return
# the constant to store in its slot. Any relocatable header is recorded in `relocs`.
function materialize_box!(mod::LLVM.Module, relocs::Relocations, namespace::String,
                          gv::GlobalVariable, @nospecialize(obj), init::Ptr{Cvoid})
    obj_type = typeof(obj)
    @assert isbitstype(obj_type)
    obj_size = sizeof(obj_type)
    @assert obj_size > 0

    W = sizeof(Int)
    hdr, bytes = GC.@preserve obj begin
        # the header word transparently yields the smalltag immediate for
        # smalltag types and the host type pointer otherwise; drop the gcbits
        hdr = unsafe_load(Ptr{UInt}(init - W)) & ~UInt(15)
        bytes = [unsafe_load(Ptr{UInt8}(init), i) for i in 1:obj_size]
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
    boxty = boxinit.value_type

    # Only a relocatable box needs a namespaced name: its header is a site loaders address by
    # name. A fully-materialized box is a private constant, so LLVM uniques it on its own.
    box_name = if patch_header
        namespaced_name(namespace,
            strip_codegen_counter(safe_name(gv.name)) * "_" *
            string(objectid(obj); base=16) * "_box")
    else
        safe_name(gv.name) * "_box"
    end
    box = GlobalVariable(mod, boxty, box_name)
    box.name == box_name || error("Interior relocation global '$box_name' is already in use")
    box.initializer = boxinit
    box.alignment = 16
    if patch_header
        box.constant = false
        box.linkage = LLVM.Linkage.External
        box.externally_initialized = true
        # `header_idx` is a zero-based field index, `LLVM.offsetof` numbers fields from 1
        offset = LLVM.offsetof(mod.datalayout, boxty, header_idx + 1)
        add_relocation!(relocs, InteriorSite, box_name, offset, JuliaValueRef(typeof(obj)))
    else
        box.constant = true
        box.linkage = LLVM.Linkage.Private
        box.unnamed_addr = LLVM.UnnamedAddr.Global
    end

    idx(i) = ConstantInt(LLVM.Int32Type(), i)
    payload = const_gep(boxty, box, LLVM.Constant[idx(0), idx(payload_idx)])
    slotty = gv.global_value_type
    val = payload.value_type == slotty ? payload : const_addrspacecast(payload, slotty)
    return val
end

# Return the byte offset added by a constant cast or GEP, or `nothing` if it is not static.
function constexpr_byte_offset(ce::LLVM.ConstantExpr, dl::LLVM.DataLayout)
    op = ce.opcode
    if op == LLVM.Opcode.BitCast || op == LLVM.Opcode.AddrSpaceCast
        return 0
    elseif op == LLVM.Opcode.GetElementPtr
        offset = LLVM.constant_offset(ce, dl)
        return offset === nothing ? nothing : Int(offset)
    end
    return nothing
end

is_word_type(T::LLVMType) =
    T isa LLVM.PointerType || (T isa LLVM.IntegerType && T.width == 8sizeof(UInt))

# The addresses an instruction merely forwards: those a `phi` or `select` picks from, or the
# operand of a pointer cast. `nothing` for any other instruction.
function forwarded_addresses(inst::LLVM.Instruction)
    if inst isa LLVM.PHIInst
        return LLVM.Value[value for (value, _) in inst.incoming]
    elseif inst isa LLVM.SelectInst
        return LLVM.Value[inst.operands[2], inst.operands[3]]
    elseif inst isa LLVM.BitCastInst || inst isa LLVM.AddrSpaceCastInst
        return LLVM.Value[inst.operands[1]]
    end
    return nothing
end

# Check forwarded addresses and relax load alignment to what a word slot guarantees.
# Cycles can arise from loop PHIs.
function check_word_loads!(value, seen=Set{LLVM.Value}())
    value in seen && return true
    push!(seen, value)
    for use in value.uses
        val = use.user
        if val isa LLVM.LoadInst
            is_word_type(val.value_type) || return false
            val.alignment > sizeof(UInt) && (val.alignment = sizeof(UInt))
        elseif val isa LLVM.Instruction && forwarded_addresses(val) !== nothing
            check_word_loads!(val, seen) || return false
        else
            return false
        end
    end
    return true
end

# Substitute a slot holding the same word for each loaded cglobal address. Replacing the
# address preserves the loads and any PHIs/selects LLVM has introduced between them.
function redirect_word_addresses!(slot_address, @nospecialize(value), what::String;
                                  offset::Union{Int,Nothing}=0,
                                  dl::LLVM.DataLayout=(value.parent::LLVM.Module).datalayout)
    changed = false
    for use in collect(value.uses)
        val = use.user
        if val isa LLVM.ConstantExpr
            delta = constexpr_byte_offset(val, dl)
            inner = (offset === nothing || delta === nothing) ? nothing : offset + delta
            changed |= redirect_word_addresses!(slot_address, val, what; offset=inner, dl)
            continue
        elseif val isa LLVM.LoadInst
            offset === nothing &&
                error("Unsupported $what load through constant expression $(val.pointer_operand)")
            is_word_type(val.value_type) ||
                error("Unsupported $what load of LLVM type $(val.value_type)")
            val.alignment > sizeof(UInt) && (val.alignment = sizeof(UInt))
        elseif val isa LLVM.Instruction && forwarded_addresses(val) !== nothing
            offset === nothing && continue
            # Slots live in the default address space.
            value.value_type.addrspace == 0 || continue
            check_word_loads!(val) || continue
        else
            continue
        end
        slot = const_pointercast(slot_address(offset), value.value_type)
        replace!(val.operands, value => slot)
        changed = true
    end
    return changed
end

# Record loaded words from libjulia globals as one relocation slot per symbol and offset.
function is_cglobal_candidate(value, relocs::Relocations)
    name = value.name
    value isa LLVM.GlobalVariable &&
        find_relocation(relocs, name) !== nothing && return false
    isdeclaration(value) || return false
    value isa LLVM.Function && LLVM.isintrinsic(value) && return false
    return startswith(name, "jl_")
end

function collect_cglobal_relocations!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                     relocs::Relocations)
    changed = false
    namespace = relocation_namespace(job)

    for f in [collect(mod.functions); collect(mod.globals)]
        is_cglobal_candidate(f, relocs) || continue
        fn = f.name
        slots = Dict{Int,GlobalVariable}()
        function cglobal_slot(offset::Int)
            get!(slots, offset) do
                # Including zero distinguishes `symbol` at N from `symbol_N` at zero.
                name = namespaced_name(namespace, "gpu_$(fn)_$(offset)")
                slot = GlobalVariable(mod, relocation_word_type(), name)
                slot.name == name ||
                    error("cglobal slot name '$name' is already in use")
                add_relocation!(relocs, SlotSite, name, 0, CGlobalRef(Symbol(fn); offset))
                slot
            end
        end

        changed |= redirect_word_addresses!(cglobal_slot, f, "cglobal '$fn'")
    end

    return changed
end

function has_unresolved_cglobal_loads(mod::LLVM.Module, relocs::Relocations)
    # also through merged addresses that `redirect_word_addresses!` had to leave alone
    function has_load(value, seen=Set{LLVM.Value}())
        for use in value.uses
            val = use.user
            val isa LLVM.LoadInst && return true
            if val isa LLVM.ConstantExpr ||
               (val isa LLVM.Instruction && forwarded_addresses(val) !== nothing)
                val in seen && continue
                push!(seen, val)
                has_load(val, seen) && return true
            end
        end
        return false
    end

    for value in [collect(mod.functions); collect(mod.globals)]
        is_cglobal_candidate(value, relocs) || continue
        has_load(value) && return true
    end
    return false
end


## bookkeeping

# Merge `src_mod` into `dest_mod` and carry its relocation metadata across. A site name always
# denotes the same word and target; duplicate declarations or definitions may therefore be
# coalesced by LLVM before their agreeing records are merged.
function link_relocatable!(dest_mod::LLVM.Module, dest_relocs::Relocations,
                            src_mod::LLVM.Module, src_relocs::Relocations;
                            only_needed=false)
    link!(dest_mod, src_mod; only_needed)
    for rec in src_relocs.records
        # A site absent from the linked module was dead (DCE'd or not imported under
        # `only_needed`); its relocation dies with it.
        haskey(dest_mod.globals, rec.name) || continue
        add_relocation!(dest_relocs, rec)
    end
    return
end

function prune_dead_relocations!(mod::LLVM.Module, relocs::Relocations)
    check_mutable(relocs, "prune")
    mod_gvs = mod.globals
    dead_names = Set{String}()
    for rec in relocs.records
        gv = get(mod_gvs, rec.name, nothing)
        if gv === nothing || (!isdeclaration(gv) && isempty(gv.uses))
            push!(dead_names, rec.name)
        end
    end
    filter!(rec -> !(rec.name in dead_names), relocs.records)
    for name in dead_names
        gv = get(mod_gvs, name, nothing)
        gv === nothing || isdeclaration(gv) || erase!(gv)
    end
    return
end


## lowering

# Lower live relocations before object emission, dispatching on the back-end's
# `relocation_lowering` strategy. Internal: back-ends select a strategy through the trait
# rather than overriding this.
function lower_relocations!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                            relocs::Relocations)
    strategy = relocation_lowering(job)
    if strategy === :bake
        bake_relocations!(mod, relocs)
    elseif strategy === :patch
        emit_patchable_relocations!(mod, relocs)
    elseif strategy === :table
        emit_table_relocations!(job, mod, relocs)
    else
        error("Unknown relocation lowering strategy :$strategy")
    end
    return
end

# Overwrite the word at `offset` in `gv`'s struct initializer with `word`.
function patch_initializer_word!(mod::LLVM.Module, gv::GlobalVariable, offset::Int,
                                 word::UInt)
    init = gv.initializer
    T = init.value_type::LLVM.StructType
    idx = LLVM.element_at(mod.datalayout, T, offset)
    # (`elements` also covers an all-zero box, e.g. a patchable header over a zero payload,
    # which LLVM folds to a `zeroinitializer`)
    fields = LLVM.Constant[init.elements...]
    fields[idx] = ConstantInt(fields[idx].value_type, word)
    gv.initializer = ConstantStruct(T, fields)
    return
end

"""
    bake_relocations!(mod, relocs)

Resolve every record in the current Julia process and write the resulting words into the IR,
leaving `relocs` empty. The module then embeds session-local addresses and must not be
persisted across sessions. Drop dead records first with [`prune_dead_relocations!`](@ref).
"""
function bake_relocations!(mod::LLVM.Module, relocs::Relocations)
    check_mutable(relocs, "resolve into IR")
    foreach_relocation(mod, relocs) do rec, gv
        word = resolve_relocation_target(rec.target)
        if rec.kind === SlotSite
            gv.initializer = slot_initializer(gv, word)
            gv.linkage = LLVM.Linkage.Private
            gv.constant = true
        else
            patch_initializer_word!(mod, gv, rec.offset, word)
            gv.linkage = LLVM.Linkage.Private
            gv.externally_initialized = false
            gv.constant = true
            gv.unnamed_addr = LLVM.UnnamedAddr.Global
        end
    end
    empty!(relocs.records)
    return
end

"""
    emit_patchable_relocations!(mod, relocs)

Emit slots as writable, null-initialized definitions, and leave interior records as the
`extinit` definitions they already are. Every record global is a weak, protected-visibility
definition. The loader must patch every record by `(name, offset)` after loading the object
([`resolved_relocations`](@ref)).
"""
function emit_patchable_relocations!(mod::LLVM.Module, relocs::Relocations)
    used = GlobalVariable[]
    foreach_relocation(mod, relocs) do rec, gv
        if rec.kind === SlotSite
            gv.initializer = null(gv.global_value_type)
            gv.constant = false
            gv.externally_initialized = true
        end
        # Two objects can define the same record: a relocation-carrying runtime-library
        # function keeps its own job's namespace in every kernel it is linked into. A loader
        # holding both in one symbol namespace (an ORC `JITDylib`) would see a duplicate
        # definition, so define them weakly and let it coalesce. That is sound rather than
        # merely quiet: a shared name means a shared producing job, hence the same target
        # (`add_relocation!` enforces agreement), so whichever definition survives gets
        # patched with the word every object referencing it expects. `llvm.used` below still
        # anchors them against DCE, and `externally_initialized` still stops the optimizer
        # from believing the null initializer.
        gv.linkage = LLVM.Linkage.WeakODR
        # Julia emits these globals `dso_local`, so backends address them PC-relatively
        # (e.g. `@rel32` on AMDGPU). A weak definition with default visibility is however
        # preemptible in an ELF shared link, which `ld.lld` rejects ("recompile with -fPIC").
        # Protected visibility keeps the symbol in the dynamic symbol table, so loaders can
        # still find it by name, while honouring the non-preemptible promise.
        gv.visibility = LLVM.Visibility.Protected
        push!(used, gv)
    end
    isempty(used) || union!(mod.used, used)
    return
end

# The functions whose bodies use `value`, following constant expressions (a `getelementptr`
# onto a global, an isbits union's `{ptr, i8}` aggregate) through to the instructions they end
# up in.
function using_functions!(fns::Set{LLVM.Function}, @nospecialize(value))
    for use in value.uses
        val = use.user
        if val isa LLVM.Instruction
            push!(fns, val.parent.parent)
        elseif val isa LLVM.Constant
            using_functions!(fns, val)
        end
    end
    return fns
end

# A relocation word comes out of a table whose base the back-end derives from per-dispatch
# state, which only an entry point can reach (a kernel's state argument, typically). So hoist
# every *other* function still holding a relocation use into its caller(s): mark it
# `alwaysinline` and run the inliner, until only entry points hold one. Mirrors
# `inline_unreachable_control_flow!`, and handles the `entry → A → B` case for the same reason:
# `A` gets marked on the next round, once `B` has been inlined into it.
function inline_relocation_users!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                 relocs::Relocations)
    # by name: a function we mark may well be gone by the next round
    hoisted = Set{String}()
    while true
        users = Set{LLVM.Function}()
        for rec in relocs.records
            gv = get(mod.globals, rec.name, nothing)
            gv === nothing || using_functions!(users, gv)
        end

        marked = false
        for f in users
            # an entry point has no call sites, and is where the state arrives
            isempty(f.uses) && continue
            fn = f.name
            fn in hoisted &&
                error("""Function `$fn` uses a relocation but could not be inlined into an
                         entry point (it is likely recursive or address-taken), so it cannot
                         reach the relocation table.""")
            push!(hoisted, fn)
            attrs = f.function_attributes
            delete!(attrs, :noinline)
            push!(attrs, EnumAttribute(:alwaysinline))
            marked = true
        end
        marked || break

        @dispose pb=PassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            with_llvm_machine(job.config.target) do tm
                run!(pb, mod, tm)
            end
        end
    end
    return
end

"""
    emit_table_relocations!(job, mod, relocs)

Rewrite every record into an indexed load from a back-end-provided table of words, the
`:table` strategy's lowering. A record's index is its rank in `relocs`, which the lowering
copies into `relocs.table` so that [`resolved_relocation_table`](@ref) delivers the words in
that same order regardless of what happens to the records afterwards.

Slots become word loads `load(gep(base, index))`, converted back with `inttoptr` where a
slot was loaded as a pointer, and are erased. Interior boxes cannot be patched after load —
the platforms needing this have no writable program-scope storage — so each is demoted to a
per-function stack copy whose header word comes from the table.
[`relocation_table_pointer`](@ref) supplies the base pointer; since it can only do so where
the state is available, callees still holding a relocation use are inlined first.
"""
function emit_table_relocations!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                 relocs::Relocations)
    isempty(relocs) && return
    LLVM.version() >= v"17" ||
        error("The `:table` relocation lowering requires LLVM 17 or later (Julia 1.12+)")

    # Fix the word order up front, in its own vector: the indices below are baked into the
    # code, so what the loader delivers must not depend on the manifest still being in this
    # order afterwards.
    empty!(relocs.table)
    append!(relocs.table, (rec.target for rec in relocs.records))

    inline_relocation_users!(job, mod, relocs)

    T_word = relocation_word_type()

    # One base pointer per function, materialized at the top of its entry block (the state
    # it derives from is a function argument, so it dominates every use).
    bases = Dict{LLVM.Function, Tuple{LLVM.Value, LLVM.Instruction}}()
    function table_base(f::LLVM.Function)
        get!(bases, f) do
            entry = first(f.entry.instructions)
            @dispose builder=IRBuilder() begin
                position!(builder, LLVM.before(entry))
                relocation_table_pointer(job, builder, f), entry
            end
        end
    end
    function table_address(builder::IRBuilder, base::LLVM.Value, index::Int)
        inbounds_gep!(builder, T_word, base, [ConstantInt(LLVM.Int32Type(), index - 1)])
    end
    function table_word(builder::IRBuilder, index::Int)
        f = builder.insert_block.parent
        base, _ = table_base(f)
        load!(builder, T_word, table_address(builder, base, index))
    end

    mod_gvs = mod.globals
    slots = LLVM.GlobalVariable[mod_gvs[rec.name] for rec in relocs.records
                       if rec.kind === SlotSite && haskey(mod_gvs, rec.name)]
    loads = check_relocation_slot_uses!(mod, slots)
    load_relocation_words!(loads, T_word)
    # Expand all constant users before choosing entry insertion points. Expanding a later
    # slot could otherwise insert a use before the entry instruction saved for an earlier one.
    convert_users_to_instructions!(slots)

    for (index, rec) in enumerate(relocs.records)
        gv = get(mod_gvs, rec.name, nothing)
        gv === nothing && error("Missing relocation global '$(rec.name)'")
        check_relocation(mod, rec, gv)

        if rec.kind === SlotSite
            addresses = Dict{LLVM.Function, LLVM.Value}()
            function slot_address(f::LLVM.Function)
                get!(addresses, f) do
                    base, entry = table_base(f)
                    @dispose builder=IRBuilder() begin
                        # After the base, but before any original instruction or PHI edge use.
                        position!(builder, LLVM.before(entry))
                        ptr = table_address(builder, base, index)
                        pointercast!(builder, ptr, gv.value_type)
                    end
                end
            end
            replace_global_with_local!(gv, slot_address)
        else
            demote_relocatable_box!(mod, gv, rec, table_base, table_word, index)
        end
    end

    # Table addresses may be in a different address space from the original slots.
    # Let LLVM propagate that space through the existing PHIs, selects and casts.
    @dispose pb=PassBuilder() begin
        tti = llvm_targetinfo(job.config.target)
        tti === nothing || LLVM.target_transform_info!(pb, tti)
        add!(pb, FunctionPassManager()) do fpm
            add!(fpm, InferAddressSpacesPass())
        end
        with_llvm_machine(job.config.target) do tm
            run!(pb, mod, tm)
        end
    end
    return
end

# Slots denote read-only words, not general storage. In particular, don't merge a table
# address with an unrelated pointer: a back-end may not have a common address space for them.
# Returns the loads of the words, however their address was forwarded.
function check_relocation_slot_uses!(mod::LLVM.Module, slots::Vector{LLVM.GlobalVariable})
    dl = mod.datalayout
    loads = LLVM.LoadInst[]
    seen = Set{LLVM.Value}(slots)
    worklist = LLVM.Value[slots...]
    while !isempty(worklist)
        for use in pop!(worklist).uses
            val = use.user
            if val isa LLVM.LoadInst
                is_word_type(val.value_type) ||
                    error("Unsupported relocation slot load of LLVM type $(val.value_type)")
                push!(loads, val)
                # Julia names these loads after globals with session-specific counters.
                val.name = ""
                # The packed table guarantees word alignment, even if the old global had more.
                val.alignment > sizeof(UInt) && (val.alignment = sizeof(UInt))
                continue
            elseif val isa LLVM.ConstantExpr
                constexpr_byte_offset(val, dl) == 0 ||
                    error("Unsupported relocation slot address $val")
            elseif !(val isa LLVM.Instruction && forwarded_addresses(val) !== nothing)
                error("Unsupported use of relocation slot address: $val")
            end
            val in seen && continue
            push!(seen, val)
            push!(worklist, val)
        end
    end
    for val in seen
        val isa LLVM.Instruction || continue
        for address in forwarded_addresses(val)
            address in seen ||
                error("Relocation slot address merged with unsupported address $address in $val")
        end
    end
    return loads
end

# Metadata that describes a loaded pointer, and so does not apply to a loaded word.
const PointerLoadMetadataKinds =
    (MD_nonnull, MD_dereferenceable, MD_dereferenceable_or_null, MD_align)

# Load the slots' values as the words the table holds. A pointer-typed load would read a
# host address as a pointer into the target's default address space, which on Metal is
# thread memory, and shader validation does not preserve such a value's bits. Converting
# the word back with `inttoptr` keeps the IR valid for existing users, while comparisons
# fold to integer ones.
function load_relocation_words!(loads::Vector{LLVM.LoadInst}, T_word::LLVMType)
    @dispose builder=IRBuilder() begin
        for load in loads
            load.value_type isa LLVM.PointerType || continue
            position!(builder, LLVM.before(load))
            word = load!(builder, T_word, load.pointer_operand;
                         align=load.alignment, volatile=load.volatile)
            if isatomic(load)
                word.ordering = load.ordering
                word.syncscope = load.syncscope
            end
            for (kind, md) in load.metadata
                kind in PointerLoadMetadataKinds || (word.metadata[kind] = md)
            end
            replace_uses!(load, inttoptr!(builder, word, load.value_type))
            erase!(load)
        end
    end
    return
end

# Copy a relocatable box into a per-function stack slot and fill its header from the
# relocation table. Sound because a box address carries no identity of its own: `isbits` egal
# compares by content, so a per-invocation copy is indistinguishable from a shared one.
function demote_relocatable_box!(mod::LLVM.Module, gv::GlobalVariable, rec::Relocation,
                                 table_base, table_word, index::Int)
    boxty = gv.global_value_type::LLVM.StructType
    init = gv.initializer
    # zero-based, for `struct_gep!` (`LLVM.element_at` numbers fields from 1)
    header_idx = LLVM.element_at(mod.datalayout, boxty, rec.offset) - 1

    allocas = Dict{LLVM.Function, LLVM.Value}()
    function box_alloca(f::LLVM.Function)
        get!(allocas, f) do
            _, entry = table_base(f)
            @dispose builder=IRBuilder() begin
                # a static alloca, which dominates the users of the box
                position!(builder, LLVM.at_begin(f.entry))
                # keep Julia's heap alignment, which the payload's `isbits` layout assumes
                ptr = alloca!(builder, boxty; align=max(gv.alignment, 16))

                # initialize it after the table base (the header load below uses it), but
                # before any original instruction, like the slot addresses
                position!(builder, LLVM.before(entry))
                store!(builder, init, ptr)
                # overwrite the (zeroed) header field with the resolved relocation word
                word = table_word(builder, index)
                store!(builder, word, struct_gep!(builder, boxty, ptr, header_idx))
                ptr
            end
        end
    end
    replace_global_with_local!(gv, box_alloca)
    return
end

"""
    apply_relocations!(mod, relocs)

Resolve every live record into `mod` without consuming `relocs`, so cached metadata can be
reused. Records whose global was optimized away are skipped. Resolution permanently roots
referenced Julia values in the process. Apply once per parsed module.

For consumers that need a session-resolved copy of a module whose cached form is symbolic —
e.g. to read a type tag out of the IR — alongside the symbolic one they cache.
"""
function apply_relocations!(mod::LLVM.Module, relocs::Relocations)
    live = copy(relocs)
    prune_dead_relocations!(mod, live)
    bake_relocations!(mod, live)
    return
end


## introspection

function referenced_object(value, relocs::Relocations)
    # This is best-effort: optimized shapes fall back to the unknown-binding error path.
    value = strip_pointer_casts(value)
    if value isa LLVM.LoadInst
        source = strip_pointer_casts(value.pointer_operand)
        if source isa GlobalVariable
            rec = find_relocation(relocs, source.name)
            if rec !== nothing && rec.target isa JuliaValueRef
                return Some(rec.target.value)
            end
        end
    elseif value isa ConstantExpr && value.opcode == LLVM.Opcode.IntToPtr
        addr = first(value.operands)
        addr isa ConstantInt || return nothing
        addr = convert(UInt, addr)
        addr < UInt(64 << 4) && return small_typeof(addr)   # jl_max_tags << 4
        return Some(Base.unsafe_pointer_to_objref(Ptr{Cvoid}(addr)))
    end
    return nothing
end

# Codegen refers to some types by their small tag instead of their address, e.g., in the
# type operand of `julia.gc_alloc_obj`. Such a tag cannot be a heap address; like
# `jl_to_typeof`, resolve it through Julia's table, whose unused entries are null.
function small_typeof(tag::UInt)
    table = cglobal(:jl_small_typeof, Ptr{Cvoid})
    ptr = unsafe_load(table, tag ÷ sizeof(Ptr{Cvoid}) + 1)
    ptr == C_NULL && return nothing
    return Some(Base.unsafe_pointer_to_objref(ptr))
end
