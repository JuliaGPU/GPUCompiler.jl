function static_eval(mod, name)
    if Base.isbindingresolved(mod, name) && Base.isdefined(mod, name)
        return Some(getfield(mod, name))
    else
        return nothing
    end
end
static_eval(gr::GlobalRef) = static_eval(gr.mod, gr.name)

function ir_element(x, code::Vector)
    while isa(x, Core.SSAValue)
        x = code[x.id]
    end
    return x
end

"""
    is_ir_element(x, y, code::Vector)

Return `true` if `x === y` or if `x` is an `SSAValue` such that
`is_ir_element(code[x.id], y, code)` is `true`.
"""
function is_ir_element(x, y, code::Vector)
    result = false
    while true # break by default
        if x === y #
            result = true
            break
        elseif isa(x, Core.SSAValue)
            x = code[x.id]
        else
            break
        end
    end
    return result
end


function early_transform!(mi, src)
    for (i, x) in enumerate(src.code)
        stmt = Base.Meta.isexpr(x, :(=)) ? x.args[2] : x
        if stmt isa GlobalRef
            @show static_eval(stmt)
        end
        # TODO: Walk stmt.args and find other uses of `:GlobalRef`
        # TODO: decide which GlobalRef to rewrite?
    end
    return nothing
end