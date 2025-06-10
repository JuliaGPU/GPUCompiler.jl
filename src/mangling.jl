# name mangling

# safe name generation

# LLVM doesn't like names with special characters, so we need to sanitize them.
# note that we are stricter than LLVM, because of `ptxas`.

safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9]"=>"_")

safe_name(t::DataType) = safe_name(String(nameof(t)))
function safe_name(t::Type{<:Function})
    # like Base.nameof, but for function types
    fn = @static if !hasfield(Core.TypeName, :mt)
        t.name.singletonname
    else
        mt = t.name.mt
        if mt === Symbol.name.mt
            # uses shared method table, so name is not unique to this function type
            nameof(t)
        else
            mt.name
        end
    end
    safe_name(string(fn))
end
safe_name(::Type{Union{}}) = "Bottom"

safe_name(x) = safe_name(repr(x))


# C++ mangling

# we generate function names that look like C++ functions, because many tools, like NVIDIA's
# profilers, support them (grouping different instantiations of the same kernel together).

function mangle_param(t, substitutions = Any[], top = false)
    t == Nothing && return "v"

    function find_substitution(x)
        sub = findfirst(isequal(x), substitutions)
        res = if sub === nothing
            nothing
        elseif sub == 1
            "S_"
        else
            seq_id = uppercase(string(sub-2; base=36))
            "S$(seq_id)_"
        end
        return res
    end

    # check if we already know this type
    str = find_substitution(t)
    if str !== nothing
        return str
    end

    if isa(t, DataType) && t <: Ptr
        tn = mangle_param(eltype(t), substitutions)
        push!(substitutions, t)
        "P$tn"
    elseif isa(t, DataType)
        # check if we already know this base type
        str = find_substitution(t.name.wrapper)
        if str === nothing
            tn = safe_name(t)
            str = "$(length(tn))$tn"
            push!(substitutions, t.name.wrapper)
        end

        if t.name.wrapper == t && !isa(t.name.wrapper, UnionAll)
            # a type with no typevars
            str
        else
            # encode typevars as template parameters
            if isempty(t.parameters)
                w_types = t.name.wrapper.types
                if !isempty(w_types) && !isempty(w_types[end] isa Core.TypeofVararg)
                    # If the type accepts a variable amount of parameters,
                    # e.g. `Tuple{}`, then we mark it as empty: "Tuple<>"
                    str *= "IJEE"
                end
            else
                str *= "I"
                for tp in t.parameters
                    str *= mangle_param(tp, substitutions)
                end
                str *= "E"
            end
            push!(substitutions, t)
            str
        end
    elseif isa(t, Union)
        # check if we already know the Union name
        str = find_substitution(Union)
        if str === nothing
            tn = "Union"
            str = "$(length(tn))$tn"
            push!(substitutions, Union)
        end

        # encode union types as template parameters
        str *= "I"
        for tp in Base.uniontypes(t)  # cannot be empty as `Union{}` is not a `Union`
            str *= mangle_param(tp, substitutions)
        end
        str *= "E"

        push!(substitutions, t)
        str
    elseif isa(t, UnionAll)
        mangle_param(Base.unwrap_unionall(t), substitutions)
    elseif isa(t, Core.TypeofVararg)
        T = isdefined(t, :T) ? t.T : Any
        if isdefined(t, :N)
            # For NTuple, repeat the type as needed
            str = ""
            for _ in 1:t.N
                str *= mangle_param(T, substitutions)
            end
            str
        elseif top
            # Variadic arguments only make sense for function arguments
            mangle_param(T, substitutions) * "z"  # T...
        else
            # Treat variadic arguments for a type as no arguments
            ""
        end
    elseif isa(t, Char)
        mangle_param(UInt32(t), substitutions)
    elseif isa(t, Union{Bool, Cchar, Cuchar, Cshort, Cushort, Cint, Cuint, Clong, Culong, Clonglong, Culonglong, Int128, UInt128})
        ts = t isa Bool       ? 'b' : # bool
             t isa Cchar      ? 'a' : # signed char
             t isa Cuchar     ? 'h' : # unsigned char
             t isa Cshort     ? 's' : # short
             t isa Cushort    ? 't' : # unsigned short
             t isa Cint       ? 'i' : # int
             t isa Cuint      ? 'j' : # unsigned int
             t isa Clong      ? 'l' : # long
             t isa Culong     ? 'm' : # unsigned long
             t isa Clonglong  ? 'x' : # long long, __int64
             t isa Culonglong ? 'y' : # unsigned long long, __int64
             t isa Int128     ? 'n' : # __int128
             t isa UInt128    ? 'o' : # unsigned __int128
             error("Invalid type")
        tn = string(abs(t), base=10)
        # for legibility, encode Julia-native integers as C-native integers, if possible
        if t isa Int && typemin(Cint) <= t <= typemax(Cint)
            ts = 'i'
        end
        if t < 0
            tn = 'n'*tn
        end
        "L$(ts)$(tn)E"
    elseif t isa Float32
        bits = string(reinterpret(UInt32, t); base=16)
        "Lf$(bits)E"
    elseif t isa Float64
        bits = string(reinterpret(UInt64, t); base=16)
        "Ld$(bits)E"
    else
        tn = safe_name(t)   # TODO: actually does support digits...
        if startswith(tn, r"\d")
            # C++ classes cannot start with a digit, so mangling doesn't support it
            tn = "_$(tn)"
        end
        "$(length(tn))$tn"
    end
end

function mangle_sig(sig)
    ft, tt... = sig.parameters

    # mangle the function name
    fn = safe_name(ft)
    str = "_Z$(length(fn))$fn"

    # mangle each parameter
    substitutions = []
    for t in tt
        str *= mangle_param(t, substitutions, true)
    end

    return str
end
