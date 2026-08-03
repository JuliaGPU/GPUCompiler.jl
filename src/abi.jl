# Julia's specsig ABI, as reported by Julia itself.
#
# Everything here mirrors the `jl_get_specsig_layout` interface in `src/julia.h`.
# Where it is available we ask Julia how it lowers a signature instead of
# re-deriving the rules; the fallbacks in `irgen.jl` implement the same rules by
# hand for older Julia versions.

const JL_ABI_LAYOUT_VERSION = UInt32(1)

# jl_abi_retcc_t
const JL_ABI_RET_BOXED    = Int32(0)
const JL_ABI_RET_REGISTER = Int32(1)
const JL_ABI_RET_SRET     = Int32(2)
const JL_ABI_RET_UNION    = Int32(3)
const JL_ABI_RET_GHOSTS   = Int32(4)

# jl_abi_argcc_t
const JL_ABI_ARG_ELIDED   = Int32(0)
const JL_ABI_ARG_VALUE    = Int32(1)
const JL_ABI_ARG_INDIRECT = Int32(2)
const JL_ABI_ARG_BOXED    = Int32(3)

# jl_abi_elide_t
const JL_ABI_ELIDE_NONE      = Int32(0)
const JL_ABI_ELIDE_GHOST     = Int32(1)
const JL_ABI_ELIDE_UNIQUEREP = Int32(2)

struct JLAbiArgInfo
    typ::Ptr{Cvoid}
    cc::Int32
    param_idx::Int32
    roots_idx::Int32
    elide_reason::Int32
    _reserved::Int32
end

struct JLAbiLayout
    version::UInt32
    specsig::Int32
    needsparams::Int32
    sigt::Ptr{Cvoid}
    rettype::Ptr{Cvoid}
    rettype_cc::Int32
    return_roots::UInt32
    all_roots::Int32
    union_bytes::Csize_t
    union_align::Csize_t
    union_minalign::Csize_t
    sret_idx::Int32
    return_roots_idx::Int32
    pgcstack_idx::Int32
    nprefix_params::Int32
    nargs::Int32
    nparams::Int32
end
JLAbiLayout() = JLAbiLayout(JL_ABI_LAYOUT_VERSION, 0, 0, C_NULL, C_NULL, 0, 0, 0,
                            0, 0, 0, -1, -1, -1, 0, 0, 0)

struct JLAbiQuery
    version::UInt32
    ci::Ptr{Cvoid}
    sigt::Ptr{Cvoid}
    rt::Ptr{Cvoid}
    is_opaque_closure::Int32
    cgparams::Ptr{Base.CodegenParams}
    mod::Ptr{Cvoid}
    datalayout::Ptr{UInt8}
    triple::Ptr{UInt8}
    name::Ptr{UInt8}
    decl_out::Ptr{Ptr{Cvoid}}
end

function _have_symbol(lib::String, sym::Symbol)
    handle = try
        Libdl.dlopen(Libdl.dlpath(lib))
    catch
        return false
    end
    return Libdl.dlsym(handle, sym; throw_error=false) !== nothing
end

const _LIBJULIA_CODEGEN = Base.isdebugbuild() ? "libjulia-codegen-debug" : "libjulia-codegen"
const _LIBJULIA_INTERNAL = Base.isdebugbuild() ? "libjulia-internal-debug" : "libjulia-internal"

"""
Whether this Julia can report its own specsig ABI. Probed rather than
version-gated so that the feature is picked up by backports too.
"""
const HAS_ABI_LAYOUT = _have_symbol(_LIBJULIA_CODEGEN, :jl_get_specsig_layout)

"""
Whether Julia exports the boxing predicates that decide the specsig ABI; if not,
`irgen.jl` falls back to its own copy of the rules.
"""
const HAS_DESERVES_CCALL = _have_symbol(_LIBJULIA_INTERNAL, :jl_deserves_stack)

# jl_value_t* of an arbitrary object, including immutable ones like Types, which
# `pointer_from_objref` refuses
_value_ptr(@nospecialize x) = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x)

"""
    abi_layout(job; mod=nothing)

Ask Julia for the specsig layout of `job.source`, using this job's
[`codegen_params`](@ref) — `gcstack_arg=false` removes a leading parameter and
so shifts every index. Returns `(layout::JLAbiLayout, args::Vector{JLAbiArgInfo})`.

Pass `mod` to have the declaration built with the target's data layout and
triple. That only changes the address spaces of the emitted pointers, not the
parameter count or ordering, so callers that just want the index mapping (such
as [`classify_arguments`](@ref)) can leave it out.
"""
function abi_layout(@nospecialize(job::CompilerJob); mod::Union{Nothing,LLVM.Module}=nothing)
    sigt = abi_signature(job.source)
    rt = typeinf_type(job.source; interp=get_interpreter(job))
    return abi_layout(sigt, rt; params=codegen_params(job), mod)
end

function abi_layout(@nospecialize(sigt), @nospecialize(rt);
                    params::Base.CodegenParams,
                    mod::Union{Nothing,LLVM.Module}=nothing,
                    is_opaque_closure::Bool=false)
    HAS_ABI_LAYOUT ||
        error("this Julia does not provide jl_get_specsig_layout")
    nargs = length((sigt::DataType).parameters)
    args = Vector{JLAbiArgInfo}(undef, max(nargs, 1))
    layout = Ref(JLAbiLayout())
    pparams = Ref(params)
    local ret
    GC.@preserve sigt rt args layout pparams begin
        query = Ref(JLAbiQuery(JL_ABI_LAYOUT_VERSION,
                               C_NULL, _value_ptr(sigt), _value_ptr(rt),
                               Int32(is_opaque_closure),
                               Base.unsafe_convert(Ptr{Base.CodegenParams}, pparams),
                               mod === nothing ? C_NULL : convert(Ptr{Cvoid}, mod.ref),
                               C_NULL, C_NULL, C_NULL, C_NULL))
        ret = @ccall jl_get_specsig_layout(query::Ptr{JLAbiQuery}, layout::Ptr{JLAbiLayout},
                                           pointer(args)::Ptr{JLAbiArgInfo},
                                           Int32(nargs)::Int32)::Cint
    end
    ret == 0 || error("jl_get_specsig_layout failed for $sigt -> $rt (code $ret)")
    l = layout[]
    return l, args[1:l.nargs]
end

"""
    abi_signature(source)

The signature `source` is compiled against. This is its `specTypes` unless a
`Core.ABIOverride` replaces it, which `specTypes` alone would miss.
"""
abi_signature(mi::Core.MethodInstance) = mi.specTypes
@static if isdefined(Core, :ABIOverride)
    abi_signature(ci::Core.CodeInstance) =
        @static if isdefined(Base, :get_ci_abi)
            Base.get_ci_abi(ci)
        else
            def = ci.def
            def isa Core.ABIOverride ? def.abi : (def::Core.MethodInstance).specTypes
        end
else
    abi_signature(ci::Core.CodeInstance) = (ci.def::Core.MethodInstance).specTypes
end
