## exception sites

# with `uses_exception_sites`, a throw reports nothing but the id of its site, by calling the
# `exception_site` runtime function, and the host maps that id to the exception's name,
# reason and backtrace using the table that compilation returns.
#
# ids stay symbolic until the module is final: the reporting call takes the result of the
# `julia.gpu.exception_site_id` intrinsic, whose operands are the exception's name and
# reason. those are ordinary string globals, so they survive caching of the runtime
# library's bitcode, deferred compilation and linking. `finalize_exception_sites!` then
# numbers the sites, at debug level 2 together with a backtrace derived from the location of
# each reporting call after inlining. lastly, `merge_exception_sites!` gives every function a
# single reporting call that all of its throw sites branch to: on some back-ends a throw site
# costs per call or per block with side effects, not per instruction.

"""
    ExceptionFrame(func, file, line)

A frame of the backtrace of an [`ExceptionSite`](@ref).
"""
struct ExceptionFrame
    func::String
    file::String
    line::Int
end

"""
    ExceptionSite(name, reason, frames)

A throw site in device code, as reported to the host: the exception's `name`, the `reason`
(possibly empty), and at debug level 2 the `frames` of the backtrace, innermost first.

Compiling a job that [`uses_exception_sites`](@ref) returns these as
`meta.exception_sites`, indexed by the id that device code passes to the `exception_site`
runtime function. Id 0 means that nothing is known about the exception.
"""
struct ExceptionSite
    name::String
    reason::String
    frames::Vector{ExceptionFrame}
end

Base.:(==)(a::ExceptionSite, b::ExceptionSite) =
    a.name == b.name && a.reason == b.reason && a.frames == b.frames
Base.hash(site::ExceptionSite, h::UInt) =
    hash(site.frames, hash(site.reason, hash(site.name, h)))

function Base.show(io::IO, ::MIME"text/plain", site::ExceptionSite)
    print(io, site.name)
    isempty(site.reason) || print(io, ": ", site.reason)
    for (i, frame) in enumerate(site.frames)
        print(io, "\n [", i, "] ", frame.func, " at ", frame.file, ":", frame.line)
    end
end


## naming exceptions

const EXCEPTION_INFO_MARKER = "julia.gpu.exception_info"

function exception_info_marker(name::String, reason::String)
    generate_llvmcall(Nothing, Tuple{}) do builder
        mod = current_module(builder)
        # anonymous, as Julia 1.10 merges named globals of `llvmcall`s by name
        name_ptr = globalstring_ptr!(builder, name)
        reason_ptr = globalstring_ptr!(builder, reason)
        ft = LLVM.FunctionType(LLVM.VoidType(), [name_ptr.value_type, reason_ptr.value_type])
        marker = get!(mod.functions, EXCEPTION_INFO_MARKER) do
            LLVM.Function(mod, EXCEPTION_INFO_MARKER, ft)
        end
        call!(builder, ft, marker, [name_ptr, reason_ptr])
        nothing
    end
end

@inline @generated exception_info(::Val{name}, ::Val{reason}) where {name, reason} =
    exception_info_marker(String(name), String(reason))

"""
    GPUCompiler.@gputhrow name reason

Throw an exception that is reported as `name` with `reason`, both string literals, rather
than as the generic exception that GPUCompiler can derive from a throw. Back-ends that don't
[`uses_exception_sites`](@ref) only report the name.
"""
macro gputhrow(name::String, reason::String)
    # without line numbers, so that backtraces start at the caller
    Base.remove_linenums!(quote
        $exception_info($(Val(Symbol(name))), $(Val(Symbol(reason))))
        throw(nothing)
    end)
end

# the name and reason that `@gputhrow` gave the throw `call`, if any
function thrown_exception_info(call::LLVM.CallInst)
    inst = call
    while (inst = LLVM.prev(inst)) !== nothing
        inst isa LLVM.CallInst || continue
        callee = inst.called_operand
        callee isa LLVM.Function && callee.name == EXCEPTION_INFO_MARKER || continue
        name, reason = constant_string(inst.arguments[1]), constant_string(inst.arguments[2])
        name === nothing && return nothing
        return String(name), String(something(reason, ""))
    end
    return nothing
end

# the markers are only needed by `lower_throw!`, and jobs that can throw don't run it
function erase_exception_info!(mod::LLVM.Module)
    marker = get(mod.functions, EXCEPTION_INFO_MARKER, nothing)
    marker === nothing && return false
    for use in collect(marker.uses)
        erase!(use.user::LLVM.CallInst)
    end
    erase!(marker)
    return true
end


## reporting

const EXCEPTION_SITE_ID = "julia.gpu.exception_site_id"

function exception_site_id_intr(mod::LLVM.Module, T_ptr::LLVMType)
    intr = get!(mod.functions, EXCEPTION_SITE_ID) do
        ft = LLVM.FunctionType(LLVM.Int32Type(), [T_ptr, T_ptr])
        LLVM.Function(mod, EXCEPTION_SITE_ID, ft)
    end
    intr.memory_effects = MemoryEffects(:none)
    return intr
end

# report the exception thrown by `inst`, the builder being positioned before it
function emit_exception_site!(@nospecialize(job::CompilerJob), builder::IRBuilder,
                              name::String, reason::String, inst::LLVM.Instruction)
    mod = LLVM.parent(LLVM.parent(builder.insert_block))

    # the location of the reporting call is what gives the site its backtrace
    builder.debug_location = inst.debug_location

    id = if job.config.debug_level >= 1
        name_ptr = globalstring_ptr!(builder, name, "exception")
        reason_ptr = globalstring_ptr!(builder, reason, "exception_reason")
        intr = exception_site_id_intr(mod, name_ptr.value_type)
        call = call!(builder, intr.function_type, intr, [name_ptr, reason_ptr])
        # merging these would merge the throw sites they identify
        push!(call.function_attributes, EnumAttribute("nomerge"))
        call
    else
        ConstantInt(Int32(0))
    end

    call = call!(builder, Runtime.get(:exception_site), [id]; job)
    # until the site is resolved, LLVM must neither merge reporting calls nor inline them
    # (which would separate the id from the call that identifies the site)
    push!(call.function_attributes, EnumAttribute("nomerge"))
    push!(call.function_attributes, EnumAttribute("noinline"))

    return call
end

reporting_function(mod::LLVM.Module) =
    get(mod.functions, Runtime.get(:exception_site).llvm_name, nothing)

# the calls to the reporting function in `mod`
function reporting_calls(mod::LLVM.Module)
    f = reporting_function(mod)
    f === nothing && return LLVM.CallInst[]
    calls = LLVM.CallInst[]
    for use in f.uses
        call = use.user
        call isa LLVM.CallInst && call.called_operand == f || continue
        push!(calls, call)
    end
    return calls
end


## the table of sites

# kept in the module between the calls to `finalize_exception_sites!`, as `finish_ir!` runs
# for every entry point
const EXCEPTION_SITES_MD = "julia.gpu.exception_sites"

function read_exception_sites(mod::LLVM.Module)
    nmd = get(mod.metadata, EXCEPTION_SITES_MD, nothing)
    nmd === nothing && return ExceptionSite[]
    map(collect(nmd.operands)) do node
        fields = String[convert(String, op::LLVM.MDString) for op in node.operands]
        frames = ExceptionFrame[ExceptionFrame(fields[i], fields[i+1], parse(Int, fields[i+2]))
                                for i in 3:3:length(fields)]
        ExceptionSite(fields[1], fields[2], frames)
    end
end

function write_exception_sites!(mod::LLVM.Module, sites::Vector{ExceptionSite})
    delete!(mod.metadata, EXCEPTION_SITES_MD)
    nmd = get!(mod.metadata, EXCEPTION_SITES_MD)
    for site in sites
        fields = LLVM.Metadata[MDString(site.name), MDString(site.reason)]
        for frame in site.frames
            push!(fields, MDString(frame.func), MDString(frame.file),
                  MDString(string(frame.line)))
        end
        push!(nmd.operands, MDNode(fields))
    end
    return
end

# remove the table from the module, returning it
function take_exception_sites!(mod::LLVM.Module)
    sites = read_exception_sites(mod)
    delete!(mod.metadata, EXCEPTION_SITES_MD)
    return sites
end


## resolving sites

"""
    finalize_exception_sites!(job, mod)

Give the throw sites in `mod` their ids. This needs to happen once `mod` is final: after
linking, and after inlining that duplicates throw sites (which gives each copy its own
backtrace). It can happen again, e.g. after inlining more, keeping the ids it already gave.
"""
function finalize_exception_sites!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    intr = get(mod.functions, EXCEPTION_SITE_ID, nothing)
    intr === nothing && return false

    sites = read_exception_sites(mod)
    ids = Dict{ExceptionSite,Int32}(site => i for (i, site) in enumerate(sites))
    function site_id(name, reason, frames)
        get!(ids, ExceptionSite(name, reason, frames)) do
            push!(sites, ExceptionSite(name, reason, frames))
            length(sites) % Int32
        end
    end

    @dispose builder=IRBuilder() begin
        for call in reporting_calls(mod)
            # back-ends may have added arguments (the kernel state, Metal's kernel inputs)
            i = findfirst(arg -> depends_on_site_id(arg, intr), collect(call.arguments))
            i === nothing && continue

            # identical sites that LLVM merged anyway share this call and its location
            frames = if job.config.debug_level >= 2
                ExceptionFrame[ExceptionFrame(String(frame.func), String(frame.file),
                                              frame.line) for frame in backtrace(call)]
            else
                ExceptionFrame[]
            end
            id = resolve_site_id!(builder, call.arguments[i], intr, frames, site_id)
            LLVM.operands(call)[i] = id
        end
    end

    # every site is identified by a reporting call, so nothing else should use its id
    strings = Set{LLVM.GlobalVariable}()
    for use in collect(intr.uses)
        inst = use.user::LLVM.CallInst
        erase_dead_selection!(inst)
        isempty(inst.uses) ||
            error("Exception site id is used other than by a call to report it: ", string(inst))
        for arg in inst.arguments
            gv = arg
            while gv isa LLVM.ConstantExpr
                gv = first(gv.operands)
            end
            gv isa LLVM.GlobalVariable && push!(strings, gv)
        end
        erase!(inst)
    end
    erase!(intr)
    for gv in strings
        isempty(gv.uses) && LLVM.linkage(gv) == LLVM.Linkage.Private && erase!(gv)
    end

    write_exception_sites!(mod, sites)
    return true
end

function depends_on_site_id(val::LLVM.Value, intr::LLVM.Function, seen=Set{LLVM.Value}())
    val in seen && return false
    push!(seen, val)
    if val isa LLVM.CallInst
        return val.called_operand == intr
    elseif val isa LLVM.PHIInst
        return any(((v, _),) -> depends_on_site_id(v, intr, seen), val.incoming)
    elseif val isa LLVM.SelectInst
        return any(v -> depends_on_site_id(v, intr, seen), LLVM.operands(val)[2:3])
    end
    return false
end

# the id of the site(s) that `val` identifies. LLVM may have merged the blocks that call
# the reporting function, selecting the id with a phi or select, which we then mirror.
function resolve_site_id!(builder::IRBuilder, val::LLVM.Value, intr::LLVM.Function, frames,
                          site_id, resolved=Dict{LLVM.Value,LLVM.Value}())
    haskey(resolved, val) && return resolved[val]
    if val isa LLVM.ConstantInt
        return val
    elseif val isa LLVM.CallInst && val.called_operand == intr
        name = something(constant_string(val.arguments[1]), "exception")
        reason = something(constant_string(val.arguments[2]), "")
        return ConstantInt(site_id(name, reason, frames))
    elseif val isa LLVM.PHIInst
        position!(builder, LLVM.before(val))
        phi = phi!(builder, LLVM.Int32Type())
        resolved[val] = phi
        for (v, bb) in collect(val.incoming)
            push!(phi.incoming,
                  (resolve_site_id!(builder, v, intr, frames, site_id, resolved), bb))
        end
        return phi
    elseif val isa LLVM.SelectInst
        cond, a, b = LLVM.operands(val)
        a = resolve_site_id!(builder, a, intr, frames, site_id, resolved)
        b = resolve_site_id!(builder, b, intr, frames, site_id, resolved)
        position!(builder, LLVM.before(val))
        sel = select!(builder, cond, a, b)
        resolved[val] = sel
        return sel
    end
    error("Cannot determine the exception site identified by ", string(val))
end

# erase the phis and selects that only served to select a now-resolved site id
function erase_dead_selection!(inst::LLVM.Instruction)
    for use in collect(inst.uses)
        user = use.user
        if (user isa LLVM.PHIInst || user isa LLVM.SelectInst) && user != inst
            erase_dead_selection!(user)
        end
    end
    if (inst isa LLVM.PHIInst || inst isa LLVM.SelectInst) && isempty(inst.uses)
        erase!(inst)
    end
    return
end


## merging sites

# whether `call` is all that is left of a throw: it is followed by nothing but a trap
function is_exception_exit(call::LLVM.CallInst)
    inst = call
    while (inst = LLVM.next(inst)) !== nothing
        inst isa LLVM.UnreachableInst && return true
        inst isa LLVM.CallInst || return false
        callee = inst.called_operand
        callee isa LLVM.Function && LLVM.isintrinsic(callee, Intrinsic("llvm.trap")) ||
            return false
    end
    return false
end

"""
    merge_exception_sites!(job, mod)

Make the throw sites in every function of `mod` branch to a single call reporting the
exception, passing it the site's id, and inline that call. The sites must have been
finalized with [`finalize_exception_sites!`](@ref).
"""
function merge_exception_sites!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    f = reporting_function(mod)
    f === nothing && return false
    ft = f.function_type
    changed = false

    sites = Dict{LLVM.Function, Vector{LLVM.CallInst}}()
    for call in reporting_calls(mod)
        is_exception_exit(call) || continue
        push!(get!(Vector{LLVM.CallInst}, sites, LLVM.parent(LLVM.parent(call))), call)
    end

    @dispose builder=IRBuilder() begin
        for (fun, calls) in sites
            length(calls) >= 2 || continue
            first_call = first(calls)

            block = BasicBlock(fun, "exception")
            position!(builder, LLVM.at_end(block))
            builder.debug_location = first_call.debug_location
            args = LLVM.Value[]
            phis = Tuple{LLVM.PHIInst,Int}[]
            for i in 1:length(ft.parameters)
                vals = [call.arguments[i] for call in calls]
                if allequal(vals)
                    push!(args, first(vals))
                else
                    phi = phi!(builder, ft.parameters[i])
                    push!(phis, (phi, i))
                    push!(args, phi)
                end
            end
            merged = call!(builder, ft, f, args)
            merged.callconv = first_call.callconv
            if !(LLVM.next(first_call) isa LLVM.UnreachableInst)
                trap = LLVM.Function(mod, Intrinsic("llvm.trap"))
                call!(builder, trap.function_type, trap)
            end
            unreachable!(builder)

            for call in calls
                bb = LLVM.parent(call)
                for (phi, i) in phis
                    push!(phi.incoming, (call.arguments[i], bb))
                end
                rest = LLVM.Instruction[call]
                inst = call
                while (inst = LLVM.next(inst)) !== nothing
                    push!(rest, inst)
                end
                position!(builder, LLVM.before(call))
                builder.debug_location = call.debug_location
                br!(builder, block)
                foreach(erase!, reverse(rest))
            end
            changed = true
        end
    end

    # a call is expensive on some back-ends (e.g. IGC compiles kernels containing calls with
    # fewer lanes, and AMDGPU spills around them), so inline the one copy left per function
    if !isdeclaration(f)
        for call in reporting_calls(mod)
            delete!(call.function_attributes, :noinline)
        end
        delete!(f.function_attributes, :noinline)
        push!(f.function_attributes, EnumAttribute(:alwaysinline))
        @dispose pb=PassBuilder() begin
            add!(pb, AlwaysInlinerPass())
            with_llvm_machine(job.config.target) do tm
                run!(pb, mod, tm)
            end
        end
        changed = true
    end

    return changed
end
