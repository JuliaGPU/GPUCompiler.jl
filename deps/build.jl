# TEMPORARY, with .ci/juliacodegen.jl: the assert jobs run TestAsserts.yml from the default
# branch (it is triggered by `workflow_run`), which only runs julia-buildpkg and
# julia-runtest. Pkg.build runs this file, so set up the environments here the way
# Test.yml does, with LLVMExtra built from the LLVM.jl branch. Delete with
# .ci/juliacodegen.jl.
if get(ENV, "GITHUB_WORKFLOW", "") == "TestAsserts"
    script = joinpath(dirname(@__DIR__), ".ci", "juliacodegen.jl")
    # not in Pkg.build's sandbox environment
    julia = addenv(Base.julia_cmd(), "JULIA_PROJECT" => nothing, "JULIA_LOAD_PATH" => nothing)
    run(`$julia --color=yes $script build`)
    run(`$julia --color=yes $script test $(VERSION >= v"1.12" ? "juliacodegen" : "julia")`)
end
