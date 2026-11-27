# Set up a CI environment to test a downstream package with the llvmgpu toolchain.
#
#     julia .ci/toolchain.jl PACKAGE [PACKAGES...] [--workspace]
#
# Develops GPUCompiler and the downstream PACKAGE (and further PACKAGES, e.g. packages of
# PACKAGE's repository) into the active environment, for `Pkg.test(PACKAGE)`. The
# preference goes into the active environment (in CI, a temporary one), which `Pkg.test`
# passes on to the test environment; GPUCompiler is added to PACKAGE's test dependencies so
# that it applies there. `--workspace` also instantiates PACKAGE's workspace against
# GPUCompiler (Metal.jl's tests run examples in its workspace environments).

using Pkg, TOML

const ROOT = dirname(@__DIR__)
const GPUCOMPILER_UUID = "61eb1bfa-7361-4325-ad38-22787b887f55"

function downstream(packages, workspace)
    package = first(packages)
    env = dirname(Base.active_project())

    # an environment that doesn't exist yet (Pkg creates it on the first change)
    mkpath(env)
    file = joinpath(env, "LocalPreferences.toml")
    open(io -> TOML.print(io, Dict("GPUCompiler" => Dict("toolchain" => "llvmgpu"))), file, "w")
    println("$file:")
    print(read(file, String))

    Pkg.develop(path=ROOT)
    Pkg.develop([PackageSpec(name=name) for name in packages])

    checkout = joinpath(Pkg.devdir(), package)
    if workspace
        # before editing the test environment, which is part of the workspace and could
        # then not be resolved; the examples that need the workspace run in the test
        # process, which has loaded GPUCompiler already
        Pkg.activate(checkout)
        Pkg.develop(path=ROOT)
        Pkg.instantiate(workspace=true)
        Pkg.activate(env)
    end
    test_project = joinpath(checkout, "test", "Project.toml")
    isfile(test_project) || error("$package has no test/Project.toml")
    # by editing the text, as rewriting it with TOML.print reorders the file
    deps = get(TOML.parsefile(test_project), "deps", Dict())
    if !haskey(deps, "GPUCompiler")
        text = read(test_project, String)
        occursin(r"^\[deps\]$"m, text) || (text = "[deps]\n\n" * text)
        text = replace(text, r"^\[deps\]$"m => "[deps]\nGPUCompiler = \"$GPUCOMPILER_UUID\"";
                       count=1)
        write(test_project, text)
    end

    Pkg.status()
end

packages = filter(!=("--workspace"), ARGS)
isempty(packages) && error("Usage: julia .ci/toolchain.jl PACKAGE [PACKAGES...] [--workspace]")
downstream(packages, "--workspace" in ARGS)
