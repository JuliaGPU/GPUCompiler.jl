# name mangling

# safe name generation

# LLVM doesn't like names with special characters, so we need to sanitize them.
# note that we are stricter than LLVM, because of `ptxas`.

safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9]"=>"_")

safe_name(t::DataType) = safe_name(String(nameof(t)))
function safe_name(t::Type{<:Function})
    # like Base.nameof, but for function types
    mt = t.name.mt
    fn = if mt === Symbol.name.mt
        # uses shared method table, so name is not unique to this function type
        nameof(t)
    else
        mt.name
    end
    safe_name(string(fn))
end
safe_name(::Type{Union{}}) = "Bottom"

safe_name(x) = safe_name(repr(x))


# C++ mangling

# we generate function names that look like C++ functions, because many tools, like NVIDIA's
# profilers, support them (grouping different instantiations of the same kernel together).

function mangle_param(t, substitutions=Any[])
    t == Nothing && return "v"

    function find_substitution(x)
        sub = findfirst(isequal(x), substitutions)
        if sub === nothing
            nothing
        elseif sub == 1
            "S_"
        else
            seq_id = uppercase(string(sub-2; base=36))
            "S$(seq_id)_"
        end
    end

    if isa(t, DataType) && t <: Ptr
        tn = mangle_param(eltype(t), substitutions)
        "P$tn"
    elseif isa(t, DataType)
        # check if we already know this type
        str = find_substitution(t)
        if str !== nothing
            return str
        end

        # check if we already know this base type
        str = find_substitution(t.name.wrapper)
        if str === nothing
            tn = safe_name(t)
            str = "$(length(tn))$tn"
            push!(substitutions, t.name.wrapper)
        end

        # encode typevars as template parameters
        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t, substitutions)
            end
            str *= "E"

            push!(substitutions, t)
        end

        str
    elseif isa(t, Union)
        # check if we already know this union type
        str = find_substitution(t)
        if str !== nothing
            return str
        end

        # check if we already know the Union name
        str = find_substitution(Union)
        if str === nothing
            tn = "Union"
            str = "$(length(tn))$tn"
            push!(substitutions, tn)
        end

        # encode union types as template parameters
        if !isempty(Base.uniontypes(t))
            str *= "I"
            for t in Base.uniontypes(t)
                str *= mangle_param(t, substitutions)
            end
            str *= "E"

            push!(substitutions, t)
        end

        str
    elseif isa(t, UnionAll)
        mangle_param(t.body, substitutions)
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
        str *= mangle_param(t, substitutions)
    end

    return str
end
