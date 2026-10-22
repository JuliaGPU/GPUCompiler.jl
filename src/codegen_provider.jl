# The code generator that lowers Julia IR to LLVM IR: Julia's own (`jl_emit_native`), or
# JuliaCodegen.jl, a code generator written in Julia

# The default of `CompilerConfig`'s `codegen`. Set with
# `Preferences.set_preferences!(GPUCompiler, "codegen" => "juliacodegen")`.
const CODEGEN = Symbol(@load_preference("codegen", "julia"))
CODEGEN in (:julia, :juliacodegen) ||
    error("Invalid codegen preference $(repr(String(CODEGEN))); expected \"julia\" or \"juliacodegen\"")

# JuliaCodegen.jl's code generator, supplied by the JuliaCodegenExt extension: a function
# with the contract of `compile_method_instance`'s code generation, from the job and the
# `(CodeInstance, CodeInfo)` pairs inference produced to the LLVM module, the functions of
# each method instance, and the values of the module's global variables.
struct CodegenProvider
    emit::Function
end
const codegen_provider = Ref{Union{Nothing,CodegenProvider}}(nothing)

# The default of `CompilerConfig`'s `codegen`: the preference, except while precompiling
# without JuliaCodegen.jl loaded, e.g., in the workload of a package that does not depend
# on it. That uses Julia's code generator; since the setting is part of the configuration,
# what it caches is not used for configurations that select JuliaCodegen.jl.
function default_codegen()
    if CODEGEN === :juliacodegen && codegen_provider[] === nothing &&
       ccall(:jl_generating_output, Cint, ()) == 1
        return :julia
    end
    return CODEGEN
end

const JULIACODEGEN = Base.PkgId(Base.UUID("f68e5268-cafc-4116-b772-1c15436a0999"), "JuliaCodegen")

# When the preference selects JuliaCodegen.jl, load it if the environment provides it, so
# that code using GPUCompiler through other packages (e.g. their test suites) gets it
# without loading it itself. Its extension then registers the code generator, once
# GPUCompiler has finished loading.
function load_codegen()
    CODEGEN === :juliacodegen || return
    ccall(:jl_generating_output, Cint, ()) == 1 && return
    haskey(Base.loaded_modules, JULIACODEGEN) && return
    Base.locate_package(JULIACODEGEN) === nothing && return
    Base.require(JULIACODEGEN)
    return
end

function check_codegen(@nospecialize(job::CompilerJob))
    job.config.codegen === :juliacodegen || return
    VERSION >= v"1.12" || error("codegen=:juliacodegen requires Julia 1.12 or later")
    codegen_provider[] === nothing &&
        error("codegen=:juliacodegen requires JuliaCodegen.jl to be loaded (`using JuliaCodegen`)")
    return
end

# generate the LLVM IR of the `(CodeInstance, CodeInfo)` pairs `codeinfos` with
# JuliaCodegen.jl, see `compile_method_instance`
function emit_juliacodegen(@nospecialize(job::CompilerJob), codeinfos, root_ci)
    check_codegen(job)
    provider = codegen_provider[]::CodegenProvider
    return Base.invokelatest(provider.emit, job, codeinfos, root_ci)
end
