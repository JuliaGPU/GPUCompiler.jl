# TEMPORARY: sets up CI environments with the unreleased LLVM.jl, JuliaIR.jl and
# JuliaCodegen.jl that the `codegen=:juliacodegen` option needs, until those are released.
# Drop this file, and its uses in `.github/workflows/Test.yml`, once GPUCompiler can depend
# on registered versions.
#
# LLVM.jl's branch comes from `[sources]` in `Project.toml`; it is checked out (to `LLVM.jl/`
# in this repository) and developed explicitly rather than through `[sources]`, because its
# LLVMExtra library has to be built from the same checkout, and because Julia 1.10 ignores
# `[sources]`. JuliaCodegen.jl and JuliaIR.jl (which it depends on) come from the branches
# below; they are added to the test environment only on Julia 1.12 and later, which
# JuliaCodegen.jl requires.
#
#     julia .ci/juliacodegen.jl build
#
# Checks out LLVM.jl and builds its LLVMExtra library if the checkout has wrapper changes
# that no LLVMExtra_jll release has (LLVM.jl's `deps/build_ci.jl`). Needs git, cmake (or
# downloads CMake_jll) and a C++ compiler; on Windows, run it in an MSYS2 shell.
#
#     julia .ci/juliacodegen.jl test (julia|juliacodegen)
#
# Prepares GPUCompiler's test environment (`test/`), for `julia --project=test
# test/runtests.jl`, with the `codegen` preference set to the given code generator. Also
# develops LLVM.jl into the default environment (`@v#.#`), with its preferences.
#
#     julia .ci/juliacodegen.jl check-ledger FILE
#
# Fails unless FILE, the ledger of the JuliaCodegenExt extension (`JULIACODEGEN_LEDGER`)
# of a run with JuliaCodegen.jl, shows that it compiled code, and reports how much of it
# fell back to native code.

using Pkg, TOML

const ROOT = dirname(@__DIR__)
const LLVM_DIR = joinpath(ROOT, "LLVM.jl")

# the branches of the unreleased packages
const JULIACODEGEN = (; url="https://github.com/JuliaLLVM/JuliaCodegen.jl", rev="vc/jc-24-gpu-gaps")
const JULIAIR = (; url="https://github.com/JuliaLLVM/JuliaIR.jl", rev="vc/runtime-cfunction")

function source(project, name)
    toml = TOML.parsefile(joinpath(ROOT, project, "Project.toml"))
    return toml["sources"][name]
end

# remove the `[sources]` entries of the packages this script develops, which Pkg 1.11
# refuses to replace ("`path` and `url` are conflicting")
function strip_sources(project)
    file = joinpath(ROOT, project, "Project.toml")
    toml = TOML.parsefile(file)
    sources = get(toml, "sources", Dict())
    delete!(sources, "LLVM")
    isempty(sources) && delete!(toml, "sources")
    open(io -> TOML.print(io, toml), file, "w")
end

run_git(args) = run(`git $args`)

function build()
    llvm = source(".", "LLVM")
    if !isdir(LLVM_DIR)
        # the full history, as build_ci.jl compares it with LLVMExtra_jll's releases
        run_git(`clone --quiet --branch $(llvm["rev"]) $(llvm["url"]) $LLVM_DIR`)
    end
    run_git(`-C $LLVM_DIR log -1 --format='LLVM.jl %H %s'`)
    # left by a build for another Julia version, which build_ci.jl may not replace
    rm(joinpath(LLVM_DIR, "LocalPreferences.toml"); force=true)
    Pkg.activate(LLVM_DIR)
    Pkg.instantiate()

    if Sys.which("cmake") === nothing
        # a wrapper on PATH, as build_local.jl looks for `cmake` there
        Pkg.activate(; temp=true)
        Pkg.add("CMake_jll")
        @eval using CMake_jll
        bin = mktempdir()
        libpath = Base.invokelatest(() -> CMake_jll.LIBPATH[])
        exe = Base.invokelatest(() -> CMake_jll.cmake_path)
        write(joinpath(bin, "cmake"), """
            #!/bin/sh
            LD_LIBRARY_PATH="$libpath:\$LD_LIBRARY_PATH" exec "$exe" "\$@"
            """)
        chmod(joinpath(bin, "cmake"), 0o755)
        ENV["PATH"] = bin * (Sys.iswindows() ? ";" : ":") * ENV["PATH"]
    end

    run(`$(Base.julia_cmd()) --project=$(joinpath(LLVM_DIR, "deps")) $(joinpath(LLVM_DIR, "deps", "build_ci.jl"))`)
    prefs = joinpath(LLVM_DIR, "LocalPreferences.toml")
    if isfile(prefs)
        print(read(prefs, String))
    else
        println("Using LLVMExtra_jll")
    end
end

# the preferences for an environment: LLVM.jl's LLVMExtra build, if any, and the code
# generator
function write_preferences(dir, mode)
    prefs = Dict{String,Any}()
    llvm_prefs = joinpath(LLVM_DIR, "LocalPreferences.toml")
    isfile(llvm_prefs) && merge!(prefs, TOML.parsefile(llvm_prefs))
    prefs["GPUCompiler"] = Dict("codegen" => mode)
    file = joinpath(dir, "LocalPreferences.toml")
    open(io -> TOML.print(io, prefs), file, "w")
    println("$file:")
    print(read(file, String))
end

function check_mode(mode)
    mode in ("julia", "juliacodegen") ||
        error("Unknown code generator $mode; expected julia or juliacodegen")
    mode == "juliacodegen" && VERSION < v"1.12" &&
        error("JuliaCodegen.jl requires Julia 1.12 or later")
end

function test(mode)
    check_mode(mode)
    isdir(LLVM_DIR) || error("Run `julia .ci/juliacodegen.jl build` first")
    strip_sources(".")
    write_preferences(joinpath(ROOT, "test"), mode)

    Pkg.activate(joinpath(ROOT, "test"))
    if VERSION >= v"1.12"
        # first, as they are not registered: any other operation would need to resolve them
        Pkg.add([PackageSpec(; JULIAIR...), PackageSpec(; JULIACODEGEN...)])
    end
    specs = [PackageSpec(path=LLVM_DIR)]
    # Julia 1.10 and 1.11 have no workspaces, so the test environment needs GPUCompiler
    VERSION < v"1.12" && pushfirst!(specs, PackageSpec(path=ROOT))
    Pkg.develop(specs)
    Pkg.status()

    # Aqua's persistent-task check loads GPUCompiler in a new environment, stacked only on
    # the default one, so that needs LLVM.jl and the preference of its LLVMExtra too
    Pkg.activate()
    Pkg.develop(path=LLVM_DIR)
    write_preferences(dirname(Base.active_project()), mode)
end

function check_ledger(file)
    isfile(file) || error("No JuliaCodegen.jl ledger at $file: it did not run")
    jobs = 0
    functions = 0
    fallbacks = 0
    pids = Set{String}()
    for (i, line) in enumerate(eachline(file))
        # PID, "juliacodegen", the method instance, the functions, the fallbacks
        fields = split(line, '\t')
        if !(length(fields) == 5 && all(isdigit, fields[1]) && fields[2] == "juliacodegen" &&
             all(isdigit, fields[4]) && all(isdigit, fields[5]))
            error("Malformed line $i in the ledger at $file: $(repr(line))")
        end
        jobs += 1
        functions += parse(Int, fields[4])
        fallbacks += parse(Int, fields[5])
        push!(pids, fields[1])
    end
    println("$jobs compilations of $functions functions, $fallbacks of which call native code, in $(length(pids)) processes")
    jobs > 0 || error("JuliaCodegen.jl compiled nothing; is GPUCompiler using Julia's code generator?")
end

function main(args)
    # the environments are only complete at the end, e.g., JuliaIR.jl fails to precompile
    # with the registered LLVM.jl that is used until the checkout is developed
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    isempty(args) && error("Usage: julia .ci/juliacodegen.jl (build | test MODE | check-ledger FILE)")
    command = args[1]
    if command == "build"
        build()
    elseif command == "test"
        test(args[2])
    elseif command == "check-ledger"
        check_ledger(args[2])
    else
        error("Unknown command $command")
    end
end

main(ARGS)
