using Test, Base.CoreLogging
import Base.CoreLogging: Info

using GPUCompiler

import LLVM

include("util.jl")

@testset "GPUCompiler" begin

GPUCompiler.reset_runtime()

GPUCompiler.enable_timings()

include("native.jl")
include("ptx.jl")
if !parse(Bool, get(ENV, "CI_ASSERTS", "false")) && VERSION < v"1.4"
  include("gcn.jl")
end

haskey(ENV, "CI") && GPUCompiler.timings()

end

using LLVM_jll
function lit(f; adjust_PATH=true, adjust_LIBPATH=true)
    PATH_SEP = Sys.iswindows() ? ';' : ':'
    env_mapping = Dict{String,String}()
    PATH = string(LLVM_jll.PATH, PATH_SEP, Base.Sys.BINDIR) # To get the right julia
    if adjust_PATH
        if !isempty(get(ENV, "PATH", ""))
            env_mapping["PATH"] = string(PATH, PATH_SEP, ENV["PATH"])
        else
            env_mapping["PATH"] = PATH
        end
    end
    LIBPATH=LLVM_jll.LIBPATH
    LIBPATH_env=LLVM_jll.LIBPATH_env
    if adjust_LIBPATH
        if !isempty(get(ENV, LIBPATH_env, ""))
            env_mapping[LIBPATH_env] = string(LIBPATH, PATH_SEP, ENV[LIBPATH_env])
        else
            env_mapping[LIBPATH_env] = LIBPATH
        end
    end
    if Sys.iswindows()
        PYTHONPATH = joinpath(LLVM_jll.artifact_dir, "tools", "lit")
        if haskey(ENV, "PYTHONPATH")
            env_mapping["PYTHONPATH"] = string(PYTHONPATH, PATH_SEP, ENV["PYTHONPATH"])
        else
            env_mapping["PYTHONPATH"] = PYTHONPATH
        end
    end
    # set JULIA_PROJECT to the `test/Project.toml`
    env_mapping["JULIA_PROJECT"] = Base.current_project()
    env_mapping["LLVM_TOOLS_DIR"] = joinpath(LLVM_jll.artifact_dir, "tools")
    lit_path = joinpath(LLVM_jll.artifact_dir, "tools", "lit", "lit.py")
    withenv(env_mapping...) do
        f(lit_path)
    end
end

lit() do lit_path
    if Sys.iswindows()
        run(`python $lit_path -va codegen`)
    else
        run(`$lit_path -va codegen`)
    end
end
