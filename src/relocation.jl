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

A Julia value with a stable address (a heap object, symbol, or singleton), used as the
serializable identity of a relocation target. Resolve it in the active session with
[`resolve_relocation_target`](@ref), which permanently roots the value in the process so
the resolved address stays valid for as long as the session lives.
"""
struct JuliaValueRef
    value::Any

    function JuliaValueRef(value)
        value_type = typeof(value)
        isbitstype(value_type) && sizeof(value_type) > 0 &&
            error("JuliaValueRef requires an object with a stable address")
        new(value)
    end
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
    size = abi_size(datalayout(mod), global_value_type(gv))
    size == sizeof(UInt) ||
        error("Relocation slot '$name' has size $size, expected $(sizeof(UInt))")
    return
end

function slot_initializer(gv::GlobalVariable, value::UInt)
    T = global_value_type(gv)
    if T isa LLVM.PointerType
        return const_inttoptr(ConstantInt(UInt64(value)), T)
    elseif T isa LLVM.IntegerType && width(T) == 8sizeof(UInt)
        return ConstantInt(T, value)
    end
    error("Relocation slot '$(LLVM.name(gv))' has unsupported LLVM type $T")
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
        init = initializer(gv)
        init === nothing && error("Relocation global '$(rec.name)' has no initializer")
        T = value_type(init)
        T isa LLVM.StructType ||
            error("Relocation global '$(rec.name)' has non-struct initializer $T")
        size = abi_size(datalayout(mod), T)
        rec.offset + sizeof(UInt) <= size ||
            error("Relocation '$(rec.name)+$(rec.offset)' is outside its $size-byte global")
    end
    return
end

function foreach_relocation(f, mod::LLVM.Module, relocs::Relocations)
    mod_gvs = globals(mod)
    for rec in relocs.records
        haskey(mod_gvs, rec.name) || error("Missing relocation global '$(rec.name)'")
        gv = mod_gvs[rec.name]
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
        if isbitstype(typeof(obj)) && sizeof(typeof(obj)) > 0 && !(obj isa Bool)
            val = materialize_box!(mod, relocs, namespace, gv, obj, init)
            initializer!(gv, val)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
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
            existing = haskey(mod_gvs, slot_name) ? mod_gvs[slot_name] : nothing
            if existing !== nothing && existing !== gv
                @assert value_type(existing) == value_type(gv)
                replace_uses!(gv, existing)
                erase!(gv)
            else
                LLVM.name!(gv, slot_name)
                LLVM.name(gv) == slot_name ||
                    error("Relocation slot name '$slot_name' is already in use")
            end
            add_relocation!(relocs, SlotSite, slot_name, 0, JuliaValueRef(obj))
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
        val = materialize_box!(mod, relocs, namespace, gv, obj, init)
        initializer!(gv, val)
        constant!(gv, true)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
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
    boxty = value_type(boxinit)

    # Only a relocatable box needs a namespaced name: its header is a site loaders address by
    # name. A fully-materialized box is a private constant, so LLVM uniques it on its own.
    box_name = if patch_header
        namespaced_name(namespace,
            strip_codegen_counter(safe_name(LLVM.name(gv))) * "_" *
            string(objectid(obj); base=16) * "_box")
    else
        safe_name(LLVM.name(gv)) * "_box"
    end
    box = GlobalVariable(mod, boxty, box_name)
    LLVM.name(box) == box_name || error("Interior relocation global '$box_name' is already in use")
    initializer!(box, boxinit)
    alignment!(box, 16)
    if patch_header
        constant!(box, false)
        linkage!(box, LLVM.API.LLVMExternalLinkage)
        extinit!(box, true)
        offset = Int(offsetof(datalayout(mod), boxty, header_idx))
        add_relocation!(relocs, InteriorSite, box_name, offset, JuliaValueRef(typeof(obj)))
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

# Return the byte offset added by a constant cast or GEP, or `nothing` if it is not static.
function constexpr_byte_offset(ce::LLVM.ConstantExpr, dl::DataLayout)
    op = opcode(ce)
    if op == LLVM.API.LLVMBitCast || op == LLVM.API.LLVMAddrSpaceCast
        return 0
    elseif op == LLVM.API.LLVMGetElementPtr
        ops = operands(ce)
        indices = ops[2:end]
        all(idx -> idx isa LLVM.ConstantInt, indices) || return nothing
        T = LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ce))
        offset = convert(Int, indices[1]) * Int(abi_size(dl, T))
        for idx in indices[2:end]
            i = convert(Int, idx)
            if T isa LLVM.StructType
                offset += Int(offsetof(dl, T, i))
                T = elements(T)[i+1]
            elseif T isa LLVM.ArrayType || T isa LLVM.VectorType
                T = eltype(T)
                offset += i * Int(abi_size(dl, T))
            else
                return nothing
            end
        end
        return offset
    end
    return nothing
end

# Rewrite word-sized loads derived through constant casts or GEPs from `value`. The producer
# receives the byte offset; reject paths whose offset is not static.
function rewrite_word_loads!(produce_word, @nospecialize(value), what::String;
                             offset::Union{Int,Nothing}=0,
                             dl::DataLayout=datalayout(LLVM.parent(value)::LLVM.Module))
    changed = false
    for use in collect(uses(value))
        val = user(use)
        if isa(val, LLVM.ConstantExpr)
            delta = constexpr_byte_offset(val, dl)
            inner = (offset === nothing || delta === nothing) ? nothing : offset + delta
            changed |= rewrite_word_loads!(produce_word, val, what; offset=inner, dl)
        elseif isa(val, LLVM.LoadInst)
            offset === nothing &&
                error("Unsupported $what load through constant expression $(operands(val)[1])")
            T = value_type(val)
            (T isa LLVM.PointerType ||
             (T isa LLVM.IntegerType && width(T) == 8sizeof(UInt))) ||
                error("Unsupported $what load of LLVM type $T")
            @dispose builder=IRBuilder() begin
                position!(builder, val)
                replacement = produce_word(builder, offset)
                T isa LLVM.PointerType &&
                    (replacement = inttoptr!(builder, replacement, T))
                replace_uses!(val, replacement)
            end
            erase!(val)
            changed = true
        end
    end
    return changed
end

# Record loaded words from libjulia globals as one relocation slot per symbol and offset.
function is_cglobal_candidate(value, relocs::Relocations)
    name = LLVM.name(value)
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

    for f in [collect(functions(mod)); collect(globals(mod))]
        is_cglobal_candidate(f, relocs) || continue
        fn = LLVM.name(f)
        slots = Dict{Int,GlobalVariable}()
        function cglobal_slot(offset::Int)
            get!(slots, offset) do
                # Including zero distinguishes `symbol` at N from `symbol_N` at zero.
                name = namespaced_name(namespace, "gpu_$(fn)_$(offset)")
                slot = GlobalVariable(mod, relocation_word_type(), name)
                LLVM.name(slot) == name ||
                    error("cglobal slot name '$name' is already in use")
                add_relocation!(relocs, SlotSite, name, 0, CGlobalRef(Symbol(fn); offset))
                slot
            end
        end

        changed |= rewrite_word_loads!(f, "cglobal '$fn'") do builder, offset
            load!(builder, relocation_word_type(), cglobal_slot(offset))
        end
    end

    return changed
end

function has_unresolved_cglobal_loads(mod::LLVM.Module, relocs::Relocations)
    function has_load(value)
        for use in uses(value)
            val = user(use)
            val isa LLVM.LoadInst && return true
            val isa LLVM.ConstantExpr && has_load(val) && return true
        end
        return false
    end

    for value in [collect(functions(mod)); collect(globals(mod))]
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
        haskey(globals(dest_mod), rec.name) || continue
        add_relocation!(dest_relocs, rec)
    end
    return
end

function prune_dead_relocations!(mod::LLVM.Module, relocs::Relocations)
    check_mutable(relocs, "prune")
    mod_gvs = globals(mod)
    dead_names = Set{String}()
    for rec in relocs.records
        gv = haskey(mod_gvs, rec.name) ? mod_gvs[rec.name] : nothing
        if gv === nothing || (!isdeclaration(gv) && isempty(uses(gv)))
            push!(dead_names, rec.name)
        end
    end
    filter!(rec -> !(rec.name in dead_names), relocs.records)
    for name in dead_names
        gv = haskey(mod_gvs, name) ? mod_gvs[name] : nothing
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
    init = initializer(gv)
    T = value_type(init)::LLVM.StructType
    idx = Int(element_at(datalayout(mod), T, offset)) + 1
    # An all-zero box (e.g. a patchable header over a zero payload) is folded to a
    # `zeroinitializer`, a `ConstantAggregateZero` that reports no operands; rebuild
    # the explicit per-field constants from the struct's element types.
    fields = if init isa LLVM.ConstantAggregateZero
        LLVM.Constant[null(elty) for elty in elements(T)]
    else
        LLVM.Constant[operands(init)...]
    end
    fields[idx] = ConstantInt(value_type(fields[idx]), word)
    initializer!(gv, ConstantStruct(T, fields))
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
            initializer!(gv, slot_initializer(gv, word))
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
            constant!(gv, true)
        else
            patch_initializer_word!(mod, gv, rec.offset, word)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
            extinit!(gv, false)
            constant!(gv, true)
            unnamed_addr!(gv, true)
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
            initializer!(gv, null(global_value_type(gv)))
            constant!(gv, false)
            extinit!(gv, true)
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
        linkage!(gv, LLVM.API.LLVMWeakODRLinkage)
        # Julia emits these globals `dso_local`, so backends address them PC-relatively
        # (e.g. `@rel32` on AMDGPU). A weak definition with default visibility is however
        # preemptible in an ELF shared link, which `ld.lld` rejects ("recompile with -fPIC").
        # Protected visibility keeps the symbol in the dynamic symbol table, so loaders can
        # still find it by name, while honouring the non-preemptible promise.
        visibility!(gv, LLVM.API.LLVMProtectedVisibility)
        push!(used, gv)
    end
    isempty(used) || set_used!(mod, used...)
    return
end

# The functions whose bodies use `value`, following constant expressions (a `getelementptr`
# onto a global, an isbits union's `{ptr, i8}` aggregate) through to the instructions they end
# up in.
function using_functions!(fns::Set{LLVM.Function}, @nospecialize(value))
    for use in uses(value)
        val = user(use)
        if val isa LLVM.Instruction
            push!(fns, LLVM.parent(LLVM.parent(val)))
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
    alwaysinline_attr = EnumAttribute("alwaysinline", 0)
    noinline_attr = EnumAttribute("noinline", 0)
    # by name: a function we mark may well be gone by the next round
    hoisted = Set{String}()
    while true
        users = Set{LLVM.Function}()
        for rec in relocs.records
            haskey(globals(mod), rec.name) || continue
            using_functions!(users, globals(mod)[rec.name])
        end

        marked = false
        for f in users
            # an entry point has no call sites, and is where the state arrives
            isempty(uses(f)) && continue
            fn = LLVM.name(f)
            fn in hoisted &&
                error("""Function `$fn` uses a relocation but could not be inlined into an
                         entry point (it is likely recursive or address-taken), so it cannot
                         reach the relocation table.""")
            push!(hoisted, fn)
            attrs = function_attributes(f)
            delete!(attrs, noinline_attr)
            push!(attrs, alwaysinline_attr)
            marked = true
        end
        marked || break

        @dispose pb=NewPMPassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            run!(pb, mod, llvm_machine(job.config.target))
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

Slots become `load(gep(base, index))` and are erased. Interior boxes cannot be patched
after load — the platforms needing this have no writable program-scope storage — so each is
demoted to a per-function stack copy whose header word comes from the table.
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
    bases = Dict{LLVM.Function, LLVM.Value}()
    function table_base(f::LLVM.Function)
        get!(bases, f) do
            @dispose builder=IRBuilder() begin
                position!(builder, first(instructions(first(blocks(f)))))
                relocation_table_pointer(job, builder, f)
            end
        end
    end
    function table_word(builder::IRBuilder, index::Int)
        f = LLVM.parent(position(builder))
        ptr = inbounds_gep!(builder, T_word, table_base(f),
                            [ConstantInt(LLVM.Int32Type(), index - 1)])
        load!(builder, T_word, ptr)
    end

    mod_gvs = globals(mod)
    for (index, rec) in enumerate(relocs.records)
        haskey(mod_gvs, rec.name) || error("Missing relocation global '$(rec.name)'")
        gv = mod_gvs[rec.name]
        check_relocation(mod, rec, gv)

        if rec.kind === SlotSite
            rewrite_word_loads!(gv, "relocation slot '$(rec.name)'") do builder, offset
                offset == 0 ||
                    error("Relocation slot '$(rec.name)' is loaded at offset $offset")
                table_word(builder, index)
            end
            prune_constexpr_uses!(gv)
            isempty(uses(gv)) ||
                error("Relocation slot '$(rec.name)' still has uses after redirection")
            erase!(gv)
        else
            demote_relocatable_box!(mod, gv, rec, table_word, index)
        end
    end
    return
end

# Copy a relocatable box into a per-function stack slot and fill its header from the
# relocation table. Sound because a box address carries no identity of its own: `isbits` egal
# compares by content, so a per-invocation copy is indistinguishable from a shared one.
function demote_relocatable_box!(mod::LLVM.Module, gv::GlobalVariable, rec::Relocation,
                                 table_word, index::Int)
    boxty = global_value_type(gv)::LLVM.StructType
    init = initializer(gv)
    header_idx = Int(element_at(datalayout(mod), boxty, rec.offset))

    allocas = Dict{LLVM.Function, LLVM.Value}()
    function box_alloca(f::LLVM.Function)
        get!(allocas, f) do
            @dispose builder=IRBuilder() begin
                position!(builder, first(instructions(first(blocks(f)))))
                ptr = alloca!(builder, boxty)
                # keep Julia's heap alignment, which the payload's `isbits` layout assumes
                alignment!(ptr, max(alignment(gv), 16))
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
            rec = find_relocation(relocs, LLVM.name(source))
            if rec !== nothing && rec.target isa JuliaValueRef
                return Some(rec.target.value)
            end
        end
    elseif value isa ConstantExpr && opcode(value) == LLVM.API.LLVMIntToPtr
        ptr = Ptr{Cvoid}(convert(Int, first(operands(value))))
        return Some(Base.unsafe_pointer_to_objref(ptr))
    end
    return nothing
end
