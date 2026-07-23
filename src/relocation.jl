# Relocations map words in globals to Julia values or named C globals. Declarations can be
# resolved, patched, or imported; sites inside definitions can only be resolved or patched.
#
#   produce ──▶ merge (on link) ──▶ prune (after DCE) ──▶ lower
#
# `:patch` and `:import` need a symbol namespace per object. `:defer` instead applies sites
# directly to each parsed module. Site names are fixed at creation so IR and metadata can be
# linked without renaming.


## targets

"""
    JuliaValueRef(value)

A Julia value with a stable address (a heap object, symbol, or singleton), used as the
serializable identity of a relocation target. Resolve it in the active session with
[`resolve_relocation_target`](@ref); a loader that writes the resulting address into device
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
    CGlobalRef(symbol, library=nothing)

A named C data global. With `library === nothing`, resolution uses `jl_cglobal`'s
process-wide lookup. Otherwise it looks up `symbol` in `library`. Resolution returns the
word stored in the global.
"""
struct CGlobalRef
    symbol::Symbol
    library::Union{Nothing,String}
end
CGlobalRef(symbol::Symbol) = CGlobalRef(symbol, nothing)

"""
    RelocationTarget

A serializable target for a relocated word: either a [`JuliaValueRef`](@ref) or a
[`CGlobalRef`](@ref).
"""
const RelocationTarget = Union{JuliaValueRef,CGlobalRef}

same_relocation_target(a::JuliaValueRef, b::JuliaValueRef) = a.value === b.value
same_relocation_target(a::CGlobalRef, b::CGlobalRef) =
    a.symbol === b.symbol && a.library == b.library
same_relocation_target(::RelocationTarget, ::RelocationTarget) = false

"""
    resolve_relocation_target(target) -> UInt

Resolve a relocation target to its word in the current Julia process.
"""
function resolve_relocation_target(target::JuliaValueRef)
    UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), target.value))
end
function resolve_relocation_target(target::CGlobalRef)
    if target.library === nothing
        # `jl_cglobal` accepts the symbol directly and does the process-wide `jl_dlfind`.
        address = ccall(:jl_cglobal, Any, (Any, Any), target.symbol, UInt)
        return unsafe_load(address)
    end
    handle = Libdl.dlopen(target.library)
    address = Libdl.dlsym(handle, target.symbol)
    return unsafe_load(Ptr{UInt}(address))
end


## the table

"""
    RelocationSite(name, offset)

The location of a word to relocate: a global name and a byte offset within that global.
The global's IR shape determines how the site is lowered. A declaration denotes an
importable word-sized slot and must have offset zero; a definition denotes a post-load
patch within its initializer.
"""
struct RelocationSite
    name::String
    offset::Int

    function RelocationSite(name::String, offset::Int)
        offset >= 0 || throw(ArgumentError("relocation offset must be nonnegative"))
        new(name, offset)
    end
end

"""
    Relocations(sites)

Relocation metadata accompanying a module. `sites` maps [`RelocationSite`](@ref)s to
targets. See [`resolved_relocations`](@ref) for resolving sites for a loader.
"""
struct Relocations
    sites::Dict{RelocationSite,RelocationTarget}
end

Relocations() = Relocations(Dict{RelocationSite,RelocationTarget}())

# Resolving into IR consumes the site table; loaders copy cached metadata first.
Base.copy(relocs::Relocations) = Relocations(copy(relocs.sites))

"""
    resolved_relocations(relocs)

Resolve relocation metadata for a loader. Returns resolved `sites` and Julia `roots` that
must stay alive while the loaded code can access them.
"""
function resolved_relocations(relocs::Relocations)
    sites = Pair{RelocationSite,UInt}[]
    roots = Any[]
    for (site, ref) in relocs.sites
        push!(sites, site => resolve_relocation_target(ref))
        ref isa JuliaValueRef && push!(roots, ref.value)
    end
    return (; sites, roots)
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

function foreach_relocation(f, mod::LLVM.Module, relocs::Relocations)
    mod_gvs = globals(mod)
    for (site, ref) in relocs.sites
        name = site.name
        offset = site.offset
        haskey(mod_gvs, name) || error("Missing relocation global '$name'")
        gv = mod_gvs[name]
        if isdeclaration(gv)
            offset == 0 ||
                error("Relocation declaration '$name' has nonzero offset $offset")
            check_slot_size(mod, gv, name)
        else
            init = initializer(gv)
            init === nothing && error("Relocation global '$name' has no initializer")
            T = value_type(init)
            T isa LLVM.StructType ||
                error("Relocation global '$name' has non-struct initializer $T")
            size = abi_size(datalayout(mod), T)
            offset + sizeof(UInt) <= size ||
                error("Relocation '$name+$offset' is outside its $size-byte global")
        end
        f(site, gv, ref)
    end
    return
end


## producers

function collect_julia_value_relocations!(mod::LLVM.Module,
                                         gv_to_value::Dict{String, Ptr{Cvoid}})
    relocs = Relocations()
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
            val = materialize_box!(mod, relocs, gv, obj, init)
            initializer!(gv, val)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        else
            check_slot_size(mod, gv, name)
            slot_name = safe_name(name) * "_" * string(objectid(obj); base=16)
            LLVM.name!(gv, slot_name)
            LLVM.name(gv) == slot_name ||
                error("Relocation slot name '$slot_name' is already in use")
            relocs.sites[RelocationSite(slot_name, 0)] = JuliaValueRef(obj)
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
        val = materialize_box!(mod, relocs, gv, obj, init)
        initializer!(gv, val)
        constant!(gv, true)
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
    end
    return relocs
end

# Emit a device-resident constant replica of the box holding `obj` and return
# the constant to store in its slot. Any relocatable header is recorded in `relocs`.
function materialize_box!(mod::LLVM.Module, relocs::Relocations, gv::GlobalVariable,
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
    LLVM.name(box) == box_name || error("Interior relocation global '$box_name' is already in use")
    initializer!(box, boxinit)
    alignment!(box, 16)
    if patch_header
        constant!(box, false)
        linkage!(box, LLVM.API.LLVMExternalLinkage)
        extinit!(box, true)
        offset = Int(offsetof(datalayout(mod), boxty, header_idx))
        relocs.sites[RelocationSite(box_name, offset)] = JuliaValueRef(typeof(obj))
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

# Some Julia code loads words from libjulia C globals, for example type tags. Record those
# loads as dedicated zero-offset relocations immediately before object emission.
function is_cglobal_candidate(value, relocs::Relocations)
    name = LLVM.name(value)
    value isa LLVM.GlobalVariable &&
        haskey(relocs.sites, RelocationSite(name, 0)) && return false
    isdeclaration(value) || return false
    value isa LLVM.Function && LLVM.isintrinsic(value) && return false
    return startswith(name, "jl_")
end

function collect_cglobal_relocations!(mod::LLVM.Module, relocs::Relocations)
    changed = false

    for f in [collect(functions(mod)); collect(globals(mod))]
        is_cglobal_candidate(f, relocs) || continue
        fn = LLVM.name(f)
        slot = nothing
        function cglobal_slot()
            if slot === nothing
                name = "gpu_" * fn
                slot = GlobalVariable(mod, relocation_word_type(), name)
                LLVM.name(slot) == name ||
                    error("cglobal slot name '$name' is already in use")
                relocs.sites[RelocationSite(name, 0)] = CGlobalRef(Symbol(fn))
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
                        error("Unsupported cglobal '$fn' load of LLVM type $T")
                    end
                    @dispose builder=IRBuilder() begin
                        position!(builder, val)
                        replacement = load!(builder, relocation_word_type(),
                                            cglobal_slot())
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

function link_relocatable!(dest_mod::LLVM.Module, dest_relocs::Relocations,
                            src_mod::LLVM.Module, src_relocs::Relocations;
                            only_needed=false)
    # Make an identical source definition a declaration so LLVM can resolve it to the
    # destination while linking. Declaration sites already merge without modification.
    for (site, ref) in src_relocs.sites
        existing = get(dest_relocs.sites, site, nothing)
        existing === nothing && continue
        name = site.name
        offset = site.offset
        same_relocation_target(existing, ref) ||
            error("Relocation '$name+$offset' refers to conflicting values")
        haskey(globals(src_mod), name) ||
            error("Missing source relocation global '$name'")
        gv = globals(src_mod)[name]
        isdeclaration(gv) && continue
        haskey(globals(dest_mod), name) ||
            error("Missing destination relocation global '$name'")
        initializer!(gv, nothing)
        extinit!(gv, false)
        linkage!(gv, LLVM.API.LLVMExternalLinkage)
    end

    link!(dest_mod, src_mod; only_needed)
    for (site, ref) in src_relocs.sites
        name = site.name
        # A site absent from the linked module was dead (DCE'd or not imported under
        # `only_needed`); its relocation dies with it.
        haskey(globals(dest_mod), name) || continue
        existing = get(dest_relocs.sites, site, nothing)
        existing === nothing || same_relocation_target(existing, ref) ||
            error("Relocation '$(site.name)+$(site.offset)' refers to conflicting values")
        dest_relocs.sites[site] = ref
    end
    return
end

function prune_dead_relocations!(mod::LLVM.Module, relocs::Relocations)
    mod_gvs = globals(mod)
    dead_names = Set{String}()
    for site in keys(relocs.sites)
        gv = haskey(mod_gvs, site.name) ? mod_gvs[site.name] : nothing
        if gv === nothing || (!isdeclaration(gv) && isempty(uses(gv)))
            push!(dead_names, site.name)
        end
    end
    for site in collect(keys(relocs.sites))
        site.name in dead_names && delete!(relocs.sites, site)
    end
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
    elseif strategy === :import
        emit_imported_relocations!(mod, relocs)
    elseif strategy === :defer
        # the consumer resolves relocations itself at the `:llvm` level; emitting an
        # object here would leave its sites permanently unresolved
        isempty(relocs.sites) ||
            error("Job defers relocation lowering to its consumer, so code with live " *
                  "relocations cannot be emitted. Use `apply_relocations!` on the " *
                  "`:llvm` result, or a `:patch`/`:import` lowering strategy.")
    else
        error("Unknown relocation lowering strategy :$strategy")
    end
    return
end

"""
    bake_relocations!(mod, relocs)

Resolve every site in the current Julia process and write the resulting words into the IR,
leaving `relocs` empty. The module then embeds session-local addresses and must not be
persisted across sessions. Drop dead sites first with [`prune_dead_relocations!`](@ref).
"""
function bake_relocations!(mod::LLVM.Module, relocs::Relocations)
    foreach_relocation(mod, relocs) do site, gv, ref
        if isdeclaration(gv)
            value = resolve_relocation_target(ref)
            val = slot_initializer(gv, value)
            initializer!(gv, val)
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
            constant!(gv, true)
        else
            init = initializer(gv)
            T = value_type(init)::LLVM.StructType
            idx = Int(element_at(datalayout(mod), T, site.offset)) + 1
            # An all-zero box (e.g. a patchable header over a zero payload) is folded to a
            # `zeroinitializer`, a `ConstantAggregateZero` that reports no operands; rebuild
            # the explicit per-field constants from the struct's element types.
            fields = if init isa LLVM.ConstantAggregateZero
                LLVM.Constant[null(elty) for elty in elements(T)]
            else
                LLVM.Constant[operands(init)...]
            end
            fields[idx] = ConstantInt(value_type(fields[idx]), resolve_relocation_target(ref))
            initializer!(gv, ConstantStruct(T, fields))
            linkage!(gv, LLVM.API.LLVMPrivateLinkage)
            extinit!(gv, false)
            constant!(gv, true)
            unnamed_addr!(gv, true)
        end
    end
    empty!(relocs.sites)
    return
end

"""
    emit_patchable_relocations!(mod, relocs)

Emit declarations as writable, null-initialized definitions. The loader must patch every
relocation after loading the object. This requires a per-object symbol namespace.
"""
function emit_patchable_relocations!(mod::LLVM.Module, relocs::Relocations)
    used = GlobalVariable[]
    foreach_relocation(mod, relocs) do _, gv, _
        if isdeclaration(gv)
            initializer!(gv, null(global_value_type(gv)))
            constant!(gv, false)
            linkage!(gv, LLVM.API.LLVMExternalLinkage)
            extinit!(gv, true)
        end
        push!(used, gv)
    end
    isempty(used) || set_used!(mod, used...)
    return
end

"""
    emit_imported_relocations!(mod, relocs)

Leave declaration slots external so a JIT loader resolves them at link time (e.g. ORC
`absoluteSymbols`); a word inside a definition cannot be imported, so those sites stay
`llvm.used` for the loader to patch after loading. Requires a per-object symbol namespace.
"""
function emit_imported_relocations!(mod::LLVM.Module, relocs::Relocations)
    used = GlobalVariable[]
    foreach_relocation(mod, relocs) do _, gv, _
        if isdeclaration(gv)
            constant!(gv, false)
            linkage!(gv, LLVM.API.LLVMExternalLinkage)
            extinit!(gv, false)
        else
            push!(used, gv)
        end
    end
    isempty(used) || set_used!(mod, used...)
    return
end

"""
    apply_relocations!(mod, relocs) -> roots

Loader entry point for `:defer` back-ends. Resolves every live site into `mod` without
consuming `relocs`, so cached metadata can be reused. Sites whose global was optimized away
are skipped.

Returns the Julia `roots` that must stay alive for as long as the module's code can run.
Apply once per parsed module.
"""
function apply_relocations!(mod::LLVM.Module, relocs::Relocations)
    live = copy(relocs)
    prune_dead_relocations!(mod, live)
    roots = Any[ref.value for ref in values(live.sites) if ref isa JuliaValueRef]
    bake_relocations!(mod, live)
    return roots
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
            ref = get(relocs.sites, RelocationSite(LLVM.name(source), 0), nothing)
            ref isa JuliaValueRef && return Some(ref.value)
        end
    elseif value isa ConstantExpr && opcode(value) == LLVM.API.LLVMIntToPtr
        ptr = Ptr{Cvoid}(convert(Int, first(operands(value))))
        return Some(Base.unsafe_pointer_to_objref(ptr))
    end
    return nothing
end
