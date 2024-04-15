@testsetup module Precompile

using Test
using ReTestItems

export precompile_test_harness, check_presence, create_standalone

function precompile_test_harness(@nospecialize(f), testset::String)
    @testset "$testset" begin
        precompile_test_harness(f, true)
    end
end
function precompile_test_harness(@nospecialize(f), separate::Bool)
    load_path = mktempdir()
    load_cache_path = separate ? mktempdir() : load_path
    try
        pushfirst!(LOAD_PATH, load_path)
        pushfirst!(DEPOT_PATH, load_cache_path)
        f(load_path)
    finally
        try
            rm(load_path, force=true, recursive=true)
        catch err
            @show err
        end
        if separate
            try
                rm(load_cache_path, force=true, recursive=true)
            catch err
                @show err
            end
        end
        filter!((≠)(load_path), LOAD_PATH)
        separate && filter!((≠)(load_cache_path), DEPOT_PATH)
    end
    nothing
end

function check_presence(mi, token)
    found = false
    ci = isdefined(mi, :cache) ? mi.cache : nothing
    while ci !== nothing
        if ci.owner === token && ci.max_world == typemax(UInt)
            found = true
            break
        end
        ci = isdefined(ci, :next) ? ci.next : nothing
    end
    return found
end

function create_standalone(load_path, name::String, file)
    cp(joinpath(@__DIR__, "runtime.jl"), joinpath(load_path, "runtime.jl"), force=true)

    TS = include(file)
    code = TS.code
    if code.head == :begin
        code.head = :block
    end
    @assert code.head == :block
    code = Expr(:module, true, Symbol(name), code)

    # Write out the test setup as a micro package
    write(joinpath(load_path, "$name.jl"), string(code))
    Base.compilecache(Base.PkgId(name)))
end
