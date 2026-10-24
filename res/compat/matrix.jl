# Check every registered GPUCompiler version against the Julia releases its registry entry
# claims to support, to find `julia` compat bounds in General that are too wide.
#
# For each (package version, Julia version) pair, a fresh environment adds exactly that
# package version (resolve), loads it (precompile), and runs `smoke.jl` (code generation for
# the native and PTX targets). Of every Julia minor version a package version claims, the
# first and last patch release are tested; all patches in between only when those disagree.
#
# Usage: julia matrix.jl [--jobs=N] [--workdir=DIR] [--only=VERSIONSPEC] [--julia=VERSIONSPEC]
#                        [--pin=NAME@VERSION,...]
#
# `--pin` adds dependencies at exact versions, to check whether narrowing a dependency's
# compat bound would fix a failure. Results record the pins they were obtained with.
#
# Results are written to DIR/results/<version>/<julia>.toml, and existing results are
# reused, so the script can be interrupted and restarted. Delete a result (or the whole
# directory) to rerun it. Use `compat.jl` to turn the results into Compat.toml changes.

using Pkg, TOML, Downloads, Dates
using Pkg.Versions: VersionRange, VersionSpec, semver_spec

const PACKAGE = "GPUCompiler"
const PACKAGE_UUID = "61eb1bfa-7361-4325-ad38-22787b887f55"
const SMOKE = joinpath(@__DIR__, "smoke.jl")
const STEPS = ["load", "native", "native_jlruntime", "ptx", "ptx_throw"]

const RESOLVE_TIMEOUT = 20*60
const SMOKE_TIMEOUT = 45*60

function parse_args(args=ARGS)
    opts = Dict("jobs" => string(max(1, Sys.CPU_THREADS ÷ 2)),
                "workdir" => "/var/tmp/gpucompiler-compat",
                "only" => ">= 0", "julia" => ">= 0", "pin" => "")
    for arg in args
        m = match(r"^--(\w+)=(.*)$", arg)
        m === nothing && error("Unknown argument: $arg")
        haskey(opts, m[1]) || error("Unknown option: --$(m[1])")
        opts[m[1]] = m[2]
    end
    return opts
end


## registry

function update_registry(dir)
    if isdir(dir)
        run(`git -C $dir fetch --quiet --depth=1 origin master`)
        run(`git -C $dir reset --quiet --hard FETCH_HEAD`)
    else
        run(`git clone --quiet --depth=1 https://github.com/JuliaRegistries/General.git $dir`)
    end
    return dir
end

package_dir(registry) = joinpath(registry, string(uppercase(PACKAGE[1])), PACKAGE)

# the julia compat claimed by every non-yanked version of the package
function claimed_compat(registry)
    dir = package_dir(registry)
    versions = TOML.parsefile(joinpath(dir, "Versions.toml"))
    compat = TOML.parsefile(joinpath(dir, "Compat.toml"))
    claims = Dict{VersionNumber,VersionSpec}()
    for (v, info) in versions
        get(info, "yanked", false) && continue
        v = VersionNumber(v)
        # without julia compat, the registry doesn't restrict the Julia version
        claims[v] = VersionSpec("1")
        for (range, deps) in compat
            if v in VersionRange(range) && haskey(deps, "julia")
                # registry files store `VersionSpec` strings, not `[compat]` entries
                claims[v] = VersionSpec(deps["julia"])
            end
        end
    end
    return claims
end


## Julia releases

struct JuliaRelease
    version::VersionNumber
    url::String
    sha256::String
end

# stable x86_64 Linux releases, from the same index juliaup uses
function julia_releases(workdir)
    path = joinpath(workdir, "versions.json")
    Downloads.download("https://julialang-s3.julialang.org/bin/versions.json", path)
    json = read(path, String)
    releases = Dict{VersionNumber,JuliaRelease}()
    # every file entry is a flat JSON object, so there's no need for a JSON parser
    for m in eachmatch(r"\{[^{}]*\}", json)
        obj = m.match
        fields = Dict(k => v for (k, v) in
                      (mm.captures for mm in eachmatch(r"\"(\w+)\":\s*\"([^\"]*)\"", obj)))
        get(fields, "triplet", "") == "x86_64-linux-gnu" || continue
        get(fields, "kind", "") == "archive" || continue
        v = VersionNumber(fields["version"])
        isempty(v.prerelease) || continue
        releases[v] = JuliaRelease(v, fields["url"], fields["sha256"])
    end
    return releases
end

function install_julia(workdir, release::JuliaRelease)
    dir = joinpath(workdir, "julia", string(release.version))
    julia = joinpath(dir, "bin", "julia")
    isfile(julia) && return julia
    tarball = joinpath(workdir, "julia", "$(release.version).tar.gz")
    mkpath(dirname(tarball))
    Downloads.download(release.url, tarball)
    hash = first(split(read(`sha256sum $tarball`, String)))
    hash == release.sha256 || error("Checksum mismatch for Julia $(release.version)")
    tmp = mktempdir(dirname(dir))
    run(`tar -xzf $tarball -C $tmp --strip-components=1`)
    mv(tmp, dir; force=true)
    rm(tarball)
    return julia
end


## running one cell

result_path(workdir, pkgver, jlver) =
    joinpath(workdir, "results", string(pkgver), "$jlver.toml")

# run a command with a time limit, returning the process (or nothing on timeout)
function run_limited(cmd, log, timeout)
    proc = open(log, "a") do io
        run(pipeline(cmd; stdout=io, stderr=io); wait=false)
    end
    timer = Timer(timeout) do _
        process_running(proc) && kill(proc, Base.SIGKILL)
    end
    wait(proc)
    timedout = !isopen(timer)
    close(timer)
    return timedout ? nothing : proc
end

# runs on every Julia version since 1.3, so avoid newer syntax and Pkg APIs
const RESOLVE_SCRIPT = """
    using Pkg
    name, version = ARGS[1:2]
    pins = Pkg.Types.PackageSpec[Pkg.PackageSpec(name=String(pin[1]), version=VersionNumber(pin[2]))
            for pin in split.(ARGS[3:end], '@')]
    try
        Pkg.add([Pkg.PackageSpec(name=name, version=VersionNumber(version)); pins])
    catch err
        if nameof(typeof(err)) == :ResolverError
            println("RESOLVE unsatisfiable")
            exit(0)
        end
        rethrow()
    end
    # PTX code generation needs the back-end from a weak dependency (since v1.18)
    entry = Base.locate_package(Base.identify_package(name))
    project = Pkg.TOML.parsefile(joinpath(dirname(dirname(entry)), "Project.toml"))
    if haskey(get(project, "weakdeps", Dict()), "NVPTX_LLVM_Backend_jll")
        Pkg.add("NVPTX_LLVM_Backend_jll")
    end
    manifest = Pkg.TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    for (dep, entries) in get(manifest, "deps", manifest)
        entries isa Vector || continue
        haskey(entries[1], "version") && println("DEP ", dep, " ", entries[1]["version"])
    end
    println("RESOLVE ok")
    """

function run_cell(workdir, julia, pkgver, jlver, slot, pins)
    depot = joinpath(workdir, "depots", "$jlver-$slot")
    mkpath(depot)
    env = mktempdir(mkpath(joinpath(workdir, "envs")))
    logdir = mkpath(joinpath(workdir, "logs", string(pkgver)))
    log = joinpath(logdir, "$jlver.log")
    rm(log; force=true)

    # cells run concurrently, so set the environment per command rather than with `withenv`
    vars = filter(p -> !startswith(p.first, "JULIA_"), ENV)
    vars["JULIA_DEPOT_PATH"] = depot
    vars["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    vars["JULIA_NUM_PRECOMPILE_TASKS"] = "2"
    # glibc 2.41+ refuses libraries that request an executable stack, like the libopenlibm
    # of older Julia releases
    vars["GLIBC_TUNABLES"] = "glibc.rtld.execstack=2"
    let
        result = Dict{String,Any}("package" => string(pkgver), "julia" => string(jlver),
                                  "pins" => pins,
                                  "date" => string(now()), "log" => log)
        t0 = time()

        cmd = `$julia --startup-file=no --project=$env -e $RESOLVE_SCRIPT $PACKAGE $pkgver $pins`
        resolve = run_limited(setenv(cmd, vars), log, RESOLVE_TIMEOUT)
        output = read(log, String)
        deps = Dict(String(m[1]) => String(m[2])
                    for m in eachmatch(r"^DEP (\S+) (\S+)$"m, output))
        result["deps"] = filter(p -> p.first in ("LLVM", "LLVMExtra_jll",
                                                 "NVPTX_LLVM_Backend_jll",
                                                 "CompilerCaching"), deps)
        if resolve === nothing
            result["resolve"] = "timeout"
        elseif occursin("RESOLVE unsatisfiable", output)
            result["resolve"] = "unsatisfiable"
        elseif occursin("RESOLVE ok", output) && success(resolve)
            result["resolve"] = "ok"
        else
            result["resolve"] = "error"
        end

        if result["resolve"] == "ok"
            write(log, output, "\n### smoke test\n")
            # runtime libraries are cached per slug, which doesn't identify the package
            # version, so don't let cells share them
            rm(joinpath(depot, "scratchspaces", PACKAGE_UUID);
               recursive=true, force=true)
            cmd = `$julia --startup-file=no --project=$env $SMOKE`
            smoke = run_limited(setenv(cmd, vars), log, SMOKE_TIMEOUT)
            # don't record a cell that was interrupted from outside (e.g. by stopping this script)
            if smoke !== nothing && smoke.termsignal in (Base.SIGTERM, Base.SIGINT)
                error("Smoke test was interrupted")
            end
            output = read(log, String)
            steps = Dict{String,Any}()
            messages = Dict{String,Any}()
            for m in eachmatch(r"^RESULT (\S+) (\S+) ?(.*)$"m, output)
                steps[m[1]] = String(m[2])
                isempty(m[3]) || (messages[m[1]] = first(String(m[3]), 500))
            end
            # a crash or timeout leaves the remaining steps without a result
            for step in STEPS
                get!(steps, step, smoke === nothing ? "timeout" : "crash")
            end
            result["steps"] = steps
            result["messages"] = messages
        end

        result["duration"] = round(Int, time() - t0)
        rm(env; recursive=true, force=true)
        path = result_path(workdir, pkgver, jlver)
        mkpath(dirname(path))
        open(io -> TOML.print(io, result; sorted=true), path, "w")
        return result
    end
end

# `true` if the package version works, `false` if it doesn't, `nothing` if it can't be
# installed (then the claim doesn't matter) or smoke.jl doesn't cover its API; such a cell
# is no evidence either way
function verdict(result)
    result["resolve"] == "unsatisfiable" && return nothing
    result["resolve"] == "ok" || return false
    any(==("unsupported"), values(result["steps"])) && return nothing
    return all(in(("ok", "skip")), values(result["steps"]))
end


## scheduling

timedout(result) = result["resolve"] == "timeout" ||
                   any(==("timeout"), values(get(result, "steps", Dict())))

function run_cells(workdir, releases, cells, jobs, pins)
    isempty(cells) && return
    # install the necessary Julia versions upfront, sequentially
    julias = Dict(jlver => install_julia(workdir, releases[jlver])
                  for jlver in unique(last.(cells)))

    # group cells by Julia version so that workers reuse depots (and their compile caches)
    queue = Channel{Tuple{VersionNumber,VersionNumber}}(Inf)
    foreach(cell -> put!(queue, cell), sort(cells; by=reverse))
    close(queue)

    total = length(cells)
    done = Threads.Atomic{Int}(0)
    @sync for slot in 1:jobs
        @async for (pkgver, jlver) in queue
            result = try
                result = run_cell(workdir, julias[jlver], pkgver, jlver, slot, pins)
                # timeouts are usually a hang under load, so give those a second chance
                if timedout(result)
                    result = run_cell(workdir, julias[jlver], pkgver, jlver, slot, pins)
                end
                result
            catch err
                @error "Failed to run $PACKAGE $pkgver on Julia $jlver" exception=(err, catch_backtrace())
                continue
            end
            n = Threads.atomic_add!(done, 1) + 1
            status = something(verdict(result), "n/a")
            println("[$n/$total] $PACKAGE $pkgver on Julia $jlver: $status ",
                    "($(result["duration"])s)")
            flush(stdout)
        end
    end
end

load_result(workdir, pkgver, jlver) =
    let path = result_path(workdir, pkgver, jlver)
        isfile(path) ? TOML.parsefile(path) : nothing
    end

function main(args=ARGS)
    opts = parse_args(args)
    workdir = mkpath(opts["workdir"])
    jobs = parse(Int, opts["jobs"])
    only = semver_spec(opts["only"])
    julia_filter = semver_spec(opts["julia"])
    pins = String.(filter(!isempty, split(opts["pin"], ',')))

    registry = update_registry(joinpath(workdir, "General"))
    claims = filter(p -> p.first in only, claimed_compat(registry))
    releases = filter(p -> p.first in julia_filter, julia_releases(workdir))

    # phase 1: the first and last patch of every claimed minor version
    patches = Dict{Tuple{VersionNumber,VersionNumber},Vector{VersionNumber}}()
    for (pkgver, spec) in claims
        for jlver in sort(collect(keys(releases)))
            jlver in spec || continue
            push!(get!(patches, (pkgver, VersionNumber(jlver.major, jlver.minor)), []), jlver)
        end
    end
    cells = Tuple{VersionNumber,VersionNumber}[]
    for ((pkgver, _), vs) in patches, jlver in unique([first(vs), last(vs)])
        load_result(workdir, pkgver, jlver) === nothing && push!(cells, (pkgver, jlver))
    end
    println("Phase 1: $(length(cells)) cells to run")
    run_cells(workdir, releases, cells, jobs, pins)

    # phase 2: all patches of minor versions whose first and last patch disagree
    cells = empty(cells)
    for ((pkgver, _), vs) in patches
        length(vs) > 2 || continue
        a, b = load_result(workdir, pkgver, first(vs)), load_result(workdir, pkgver, last(vs))
        (a === nothing || b === nothing) && continue
        verdict(a) == verdict(b) && continue
        for jlver in vs[2:end-1]
            load_result(workdir, pkgver, jlver) === nothing && push!(cells, (pkgver, jlver))
        end
    end
    println("Phase 2: $(length(cells)) cells to run")
    run_cells(workdir, releases, cells, jobs, pins)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
