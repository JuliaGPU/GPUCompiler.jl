# TEMPORARY: sets up CI environments with JuliaIR.jl, which the `julia_passes = "ports"`
# preference needs and which is not registered yet. Drop this file, and its uses in
# `.github/workflows/Test.yml` and `.buildkite/downstream.yml`, once it is.
#
# JuliaIR.jl comes from its `[sources]` entry in `test/Project.toml`, so that its revision
# is set in one place.
#
#     julia .ci/juliair.jl test (cpp|ports)
#
# Prepares GPUCompiler's test environment (`test/`), for `julia --project=test
# test/runtests.jl`, with the `julia_passes` preference set to the given provider. Julia
# 1.10 ignores `[sources]`, so JuliaIR.jl is added there explicitly.
#
#     julia .ci/juliair.jl downstream PACKAGE [PACKAGES...] [--workspace]
#
# Develops GPUCompiler and the downstream PACKAGE (and further PACKAGES, e.g. packages of
# PACKAGE's repository) into the active environment, for `Pkg.test(PACKAGE)` with the
# ports: JuliaIR.jl is added too, and to PACKAGE's test environment, where GPUCompiler
# loads it. The preference goes into the active environment (in CI, a temporary one),
# which `Pkg.test` passes on to the test environment; GPUCompiler is added to PACKAGE's
# test dependencies so that it applies there. `--workspace` also instantiates PACKAGE's
# workspace against GPUCompiler (Metal.jl's tests run examples in its workspace
# environments).
#
#     julia .ci/juliair.jl check-ledger FILE
#
# Fails unless FILE, the JuliaIR.jl ledger (`JULIAIR_LEDGER`) of a run with the ports,
# shows that the core ports ran.

using Pkg, TOML

const ROOT = dirname(@__DIR__)
const GPUCOMPILER_UUID = "61eb1bfa-7361-4325-ad38-22787b887f55"
const JULIAIR_UUID = "3913dba9-7ba9-44aa-bf76-1cfa3b3073b1"

function juliair_source()
    toml = TOML.parsefile(joinpath(ROOT, "test", "Project.toml"))
    source = toml["sources"]["JuliaIR"]
    return PackageSpec(url=source["url"], rev=source["rev"])
end

function write_preferences(dir, mode)
    # an environment that doesn't exist yet (Pkg creates it on the first change)
    mkpath(dir)
    file = joinpath(dir, "LocalPreferences.toml")
    open(io -> TOML.print(io, Dict("GPUCompiler" => Dict("julia_passes" => mode))), file, "w")
    println("$file:")
    print(read(file, String))
end

function test(mode)
    mode in ("cpp", "ports") || error("Unknown provider $mode; expected cpp or ports")
    write_preferences(joinpath(ROOT, "test"), mode)
    Pkg.activate(joinpath(ROOT, "test"))
    # first, as JuliaIR.jl is not registered: any other operation would need to resolve it
    VERSION < v"1.11" && Pkg.add(juliair_source())
    # Julia 1.10 and 1.11 have no workspaces, so the test environment needs GPUCompiler
    VERSION < v"1.12" && Pkg.develop(path=ROOT)
    Pkg.instantiate()
    Pkg.status()
end

function downstream(packages, workspace)
    package = first(packages)
    env = dirname(Base.active_project())
    write_preferences(env, "ports")

    Pkg.develop(path=ROOT)
    Pkg.add(juliair_source())
    Pkg.develop([PackageSpec(name=name) for name in packages])

    checkout = joinpath(Pkg.devdir(), package)
    if workspace
        # before adding JuliaIR.jl to the test environment, which is part of the workspace
        # and could then not be resolved; the examples that need the workspace run in the
        # test process, which has loaded GPUCompiler and JuliaIR.jl already
        Pkg.activate(checkout)
        Pkg.develop(path=ROOT)
        Pkg.instantiate(workspace=true)
        Pkg.activate(env)
    end
    test_project = joinpath(checkout, "test", "Project.toml")
    isfile(test_project) || error("$package has no test/Project.toml")
    add = Dict("GPUCompiler" => GPUCOMPILER_UUID, "JuliaIR" => JULIAIR_UUID)
    # by editing the text, as rewriting it with TOML.print reorders the file
    deps = get(TOML.parsefile(test_project), "deps", Dict())
    lines = [k * " = \"" * v * "\"" for (k, v) in add if !haskey(deps, k)]
    if !isempty(lines)
        text = read(test_project, String)
        occursin(r"^\[deps\]$"m, text) || (text = "[deps]\n\n" * text)
        text = replace(text, r"^\[deps\]$"m => join(["[deps]"; lines], "\n"); count=1)
        write(test_project, text)
    end

    Pkg.status()
end

# the ports that run for every kernel on every target and Julia version, and JuliaLICM,
# which runs for every loop (that test suites have)
const CORE_PORTS = ["jl-alloc-opt", "jl-late-lower-gc-frame", "jl-propagate-addrspaces",
                    "jl-remove-ni", "jl-demote-float16", "jl-remove-addrspaces", "jl-licm"]

function check_ledger(file)
    isfile(file) || error("No JuliaIR.jl ledger at $file: the ports did not run")
    counts = Dict{String,Int}()
    pids = Set{String}()
    for (i, line) in enumerate(eachline(file))
        # PID, "pass", the pass, the Julia dialect and the LLVM version
        fields = split(line, '\t')
        if !(length(fields) == 5 && all(isdigit, fields[1]) && fields[2] == "pass" &&
             startswith(fields[3], "jl-") && startswith(fields[4], "julia=") &&
             startswith(fields[5], "llvm="))
            error("Malformed line $i in the ledger at $file: $(repr(line))")
        end
        counts[fields[3]] = get(counts, fields[3], 0) + 1
        push!(pids, fields[1])
    end
    for (name, count) in sort!(collect(counts))
        println(rpad(name, 32), count)
    end
    println("in $(length(pids)) processes")
    missing_ports = setdiff(CORE_PORTS, keys(counts))
    isempty(missing_ports) ||
        error("The ports $(join(missing_ports, ", ")) did not run; is GPUCompiler using the C++ passes?")
end

function main(args)
    isempty(args) && error("Usage: julia .ci/juliair.jl (test MODE | downstream PACKAGE... [--workspace] | check-ledger FILE)")
    command = args[1]
    if command == "test"
        test(args[2])
    elseif command == "downstream"
        downstream(filter(!=("--workspace"), args[2:end]), "--workspace" in args)
    elseif command == "check-ledger"
        check_ledger(args[2])
    else
        error("Unknown command $command")
    end
end

main(ARGS)
