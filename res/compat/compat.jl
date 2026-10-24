# Turn the results of matrix.jl into narrowed `julia` compat bounds, and rewrite
# `G/GPUCompiler/Compat.toml` in a checkout of General accordingly.
#
# Usage: julia --project=<env with RegistryTools> compat.jl [--workdir=DIR] [--registry=PATH]
#
# Only versions with a failing cell are changed. An untested patch release is assumed to
# behave like the closest tested patch below it (matrix.jl tests all patches of a minor
# version when its first and last disagree); untested releases below every tested one,
# as when only a new release was tested, are left alone. Cells that can't be resolved are
# no evidence either way, and open upper bounds (e.g. `1.10 - 1`) stay open unless the
# newest release fails.
# A summary of every change and the failures behind it is written to DIR/report.md.

include("matrix.jl")
using RegistryTools: Compress

# a stand-in for unreleased Julia versions, to keep open upper bounds open
const FUTURE = v"1.9999.0"

function analyze(workdir, claims, releases)
    changes = Dict{VersionNumber,Any}()
    broken = Dict{VersionNumber,Any}()
    for (pkgver, spec) in sort(collect(claims); by=first)
        claimed = sort(filter(in(spec), collect(keys(releases))))
        failing = VersionNumber[]
        passing = false
        passing_deps = Set{Any}()
        evidence = Dict{VersionNumber,Any}()
        for minor in unique(VersionNumber.(getfield.(claimed, :major), getfield.(claimed, :minor)))
            patches = filter(v -> (v.major, v.minor) == (minor.major, minor.minor), claimed)
            results = Dict(v => load_result(workdir, pkgver, v) for v in patches)
            tested = filter(v -> results[v] !== nothing, patches)
            isempty(tested) && continue
            for v in patches
                # an untested patch takes the verdict of the closest tested patch below it;
                # those below every tested patch (when only a new release was tested) stay
                below = filter(<=(v), tested)
                isempty(below) && continue
                ref = last(below)
                result = results[ref]
                if verdict(result) === true
                    passing = true
                    push!(passing_deps, get(result, "deps", Dict()))
                end
                if verdict(result) === false
                    push!(failing, v)
                    evidence[ref] = result
                end
            end
        end
        isempty(failing) && continue
        # failing on every Julia version points at a too-wide dependency bound or a gap in
        # smoke.jl rather than at the julia bound, so leave those for a human to look at.
        # that only follows when the oldest claimed release was tested, i.e. not when only
        # a new release was.
        oldest_tested = load_result(workdir, pkgver, first(claimed)) !== nothing
        if !passing && oldest_tested
            broken[pkgver] = (; old=spec, failing, evidence)
            continue
        end

        allowed = setdiff(claimed, failing)
        pool = [collect(keys(releases)); FUTURE]
        # keep an open upper bound open, unless the newest release is one that fails
        FUTURE in spec && !(last(claimed) in failing) && push!(allowed, FUTURE)
        newspec = Compress.compress_versions(sort(pool), sort(allowed))
        # a failure that comes with different dependency versions than the passing cells may
        # be caused by a dependency, and then needs a narrower bound on that dependency instead
        suspect = any(r -> !(get(r, "deps", Dict()) in passing_deps), values(evidence))
        changes[pkgver] = (; old=spec, new=newspec, failing, evidence, suspect)
    end
    return changes, broken
end

spaced(s::AbstractString) = replace(s, r"\s*-\s*" => " - ")
unspaced(s::AbstractString) = replace(s, " - " => "-")
unspaced(s::AbstractVector) = unspaced.(s)
compat_value(spec::VersionSpec) =
    length(spec.ranges) == 1 ? spaced(string(only(spec.ranges))) :
                               spaced.(string.(spec.ranges))

# `updates` maps package versions to new compat entries. Only the affected keys are
# recompressed; every other entry is kept exactly as written.
function rewrite_compat(path, updates)
    original = TOML.parsefile(path)
    versions = Compress.load_versions(path)
    data = Compress.load(path, versions)
    keys_changed = unique(k for entries in values(updates) for k in keys(entries))

    final = Dict{String,Any}()
    keys_by_range = Dict{String,String}()
    for (range, table) in original
        rest = filter(p -> !(p.first in keys_changed), table)
        isempty(rest) || (final[range] = rest)
        keys_by_range[unspaced(range)] = range
    end
    for key in keys_changed
        # values spelled as in the original file where possible
        spelling = Dict(unspaced(table[key]) => table[key]
                        for table in values(original) if haskey(table, key))
        groups = Dict{Any,Vector{VersionNumber}}()
        for v in versions
            value = get(get(updates, v, Dict()), key, get(data[v], key, nothing))
            value === nothing && continue
            push!(get!(groups, unspaced(value), VersionNumber[]), v)
        end
        for (value, vers) in groups
            for range in Compress.compress_versions(versions, sort(vers)).ranges
                range = get(keys_by_range, unspaced(string(range)), spaced(string(range)))
                get!(final, range, Dict{String,Any}())[key] =
                    get(spelling, value, value isa AbstractString ? spaced(value) : spaced.(value))
            end
        end
    end
    open(io -> TOML.print(io, final; sorted=true), path, "w")
end

# results obtained with `--pin=NAME@VERSION` show that capping that dependency at VERSION
# makes the package version work; turn those pins into narrowed compat entries
function pinned_caps(workdir, path, claims)
    data = Compress.load(path)
    caps = Dict{VersionNumber,Dict{String,Any}}()
    for pkgver in keys(claims)
        dir = joinpath(workdir, "results", string(pkgver))
        isdir(dir) || continue
        for file in readdir(dir), pin in get(TOML.parsefile(joinpath(dir, file)), "pins", [])
            name, version = split(pin, '@')
            old = VersionSpec(get(data[pkgver], name, "*"))
            new = intersect(old, VersionSpec("0 - $version"))
            new == old || (get!(caps, pkgver, Dict{String,Any}())[name] = compat_value(new))
        end
    end
    return caps
end

function failure_summary(result)
    result["resolve"] == "ok" || return "resolve: $(result["resolve"])"
    failed = sort([k for (k, v) in result["steps"] if !(v in ("ok", "skip"))])
    msg = get(get(result, "messages", Dict()), first(failed), "")
    msg = first(split(replace(msg, "\\n" => "\n"), '\n'))
    return "$(join(failed, ", ")): $(first(msg, 160))"
end

function deps_summary(result)
    deps = join(["$k $v" for (k, v) in sort(collect(get(result, "deps", Dict())); by=first)], ", ")
    pins = get(result, "pins", String[])
    isempty(pins) ? deps : "$deps (pinned: $(join(pins, ", ")))"
end

function report(io, changes, broken)
    println(io, "## Narrowed julia compat\n")
    println(io, "Versions marked with * failed with other dependency versions than they passed with.\n")
    println(io, "| Version | Claimed | Proposed | Failing Julia | Failure (first tested) | Dependencies |")
    println(io, "|---|---|---|---|---|---|")
    for (pkgver, change) in sort(collect(changes); by=first)
        failing = join(unique(["$(v.major).$(v.minor)" for v in change.failing]), ", ")
        ref = first(sort(collect(keys(change.evidence))))
        mark = change.suspect ? "*" : ""
        println(io, "| $pkgver$mark | `$(change.old)` | `$(change.new)` | $failing (e.g. $ref) | ",
                replace(failure_summary(change.evidence[ref]), '|' => "\\|"), " | ",
                deps_summary(change.evidence[ref]), " |")
    end
    println(io, "\n## Failing on every claimed Julia version (not changed)\n")
    println(io, "| Version | Claimed | Failure (first tested) | Dependencies |")
    println(io, "|---|---|---|---|")
    for (pkgver, b) in sort(collect(broken); by=first)
        ref = first(sort(collect(keys(b.evidence))))
        println(io, "| $pkgver | `$(b.old)` | ",
                replace(failure_summary(b.evidence[ref]), '|' => "\\|"), " | ",
                deps_summary(b.evidence[ref]), " |")
    end
end

function main_compat(args=ARGS)
    opts = Dict("workdir" => "/var/tmp/gpucompiler-compat", "registry" => "")
    for arg in args
        m = match(r"^--(\w+)=(.*)$", arg)
        (m === nothing || !haskey(opts, m[1])) && error("Unknown argument: $arg")
        opts[m[1]] = m[2]
    end
    workdir = opts["workdir"]
    registry = isempty(opts["registry"]) ? joinpath(workdir, "General") : opts["registry"]

    claims = claimed_compat(registry)
    releases = julia_releases(workdir)
    changes, broken = analyze(workdir, claims, releases)

    open(joinpath(workdir, "report.md"), "w") do io
        report(io, changes, broken)
    end
    report(stdout, changes, broken)
    path = joinpath(package_dir(registry), "Compat.toml")
    updates = pinned_caps(workdir, path, claims)
    for (pkgver, change) in changes
        get!(updates, pkgver, Dict{String,Any}())["julia"] = compat_value(change.new)
    end
    for (pkgver, caps) in sort(collect(updates); by=first), (name, value) in caps
        name == "julia" || println("$pkgver: $name = $(repr(value)) (from pinned results)")
    end
    rewrite_compat(path, updates)
    println("\nRewrote $path ($(length(updates)) versions changed)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_compat()
end
