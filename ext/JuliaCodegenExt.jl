# JuliaCodegen.jl, a code generator written in Julia, for jobs that select it with
# `codegen=:juliacodegen`. Loading this extension alone does not change generated code.
module JuliaCodegenExt

import GPUCompiler
using GPUCompiler: CompilerJob, llvm_triple, julia_datalayout, can_safepoint,
                   dwarf_version, llvm_debug_info
import JuliaCodegen
using LLVM

const CC = Base.Compiler

# The objects the generated code refers to, which GPUCompiler resolves to relocations from
# their addresses after code generation: kept alive for as long as the session, like the
# roots of Julia's own code generator.
const ROOTS = IdSet{Any}()
const ROOTS_LOCK = ReentrantLock()

# with `JULIACODEGEN_LEDGER=<file>`, every compilation appends a line to the file: the
# process, `juliacodegen`, the method instance, the number of functions, and the number
# of functions that call native code instead (fallbacks), tab-separated
function record(@nospecialize(job::CompilerJob), nfunctions, nfallbacks)
    ledger = get(ENV, "JULIACODEGEN_LEDGER", "")
    isempty(ledger) && return
    open(ledger, "a") do io
        println(io, join((getpid(), "juliacodegen", repr(job.source), nfunctions, nfallbacks), '\t'))
    end
end

# JuliaCodegen.jl's name of a `LLVM.DebugEmissionKind`: line tables, in a compile unit of
# that kind, but for `NoDebug` at debug level 0 (as Julia's code generator, which emits
# the locations unless Julia runs with `-g0`)
function debug_info(kind, level)
    kind == LLVM.DebugEmissionKind.NoDebug && return level == 0 ? :none : :no_debug
    kind == LLVM.DebugEmissionKind.LineTablesOnly && return :line_tables
    kind == LLVM.DebugEmissionKind.DebugDirectivesOnly && return :directives
    return :full
end

# the configuration of JuliaCodegen.jl for `job`: its target, and the code generation
# parameters GPUCompiler gives Julia's code generator (see `compile_method_instance`)
function codegen_config(@nospecialize(job::CompilerJob))
    target = job.config.target
    dl = julia_datalayout(target)
    datalayout = dl === nothing ? nothing : (s = string(dl); LLVM.dispose(dl); s)
    cpu = hasproperty(target, :cpu) ? String(target.cpu) : ""
    features = hasproperty(target, :features) ? String(target.features) : ""
    jctarget = JuliaCodegen.Target(; triple=llvm_triple(target), cpu, features, datalayout)
    # unsupported constructs call native code, which validation reports where the target
    # can't (with `JULIACODEGEN_UNSUPPORTED=error`, they throw, which says what they are);
    # errors of the code generator itself are reported as such
    on_unsupported = get(ENV, "JULIACODEGEN_UNSUPPORTED", "") == "error" ? :error : :fallback
    debug = debug_info(llvm_debug_info(job), job.config.debug_level)
    return JuliaCodegen.CodegenConfig(; target=jctarget, gcstack_arg=false,
                                      safepoint_on_entry=can_safepoint(job), debug_info=debug,
                                      on_unsupported, on_internal_error=:error)
end

# The entry of the job in the specsig convention as Julia's code generator names it,
# `julia_f_N` (back ends and tests refer to it): JuliaCodegen.jl names its function so,
# and the entry `j_f_N`, which inlines it. Swap the names.
function name_entry!(mod::LLVM.Module, functions, root_ci)
    fs = get(functions, root_ci, nothing)
    (fs === nothing || fs.specfunc === nothing) && return
    base = chopprefix(fs.specfunc, "j_")
    body = get(LLVM.functions(mod), "julia_" * base, nothing)
    entry = LLVM.functions(mod)[fs.specfunc]
    body === nothing || LLVM.name!(body, "jlbody_" * base)
    LLVM.name!(entry, "julia_" * base)
    functions[root_ci] = JuliaCodegen.CompiledFunctions(fs.func, "julia_" * base)
    return
end

# `compile_method_instance`'s code generation with JuliaCodegen.jl, from the sources of the
# `(CodeInstance, CodeInfo)` pairs inference produced: the module, the functions of each
# method instance, and the objects its global variables refer to
function emit(@nospecialize(job::CompilerJob), codeinfos, root_ci)
    codeinfos === nothing && error("codegen=:juliacodegen requires the sources from inference")
    config = codegen_config(job)
    mod = LLVM.Module("start")
    mod.flags["Dwarf Version", LLVM.ModuleFlagBehavior.Warning] =
        LLVM.Metadata(LLVM.ConstantInt(dwarf_version(job.config.target)))
    mod.flags["Debug Info Version", LLVM.ModuleFlagBehavior.Warning] =
        LLVM.Metadata(LLVM.ConstantInt(LLVM.DEBUG_METADATA_VERSION()))
    functions, output = JuliaCodegen.emit_into!(mod, codeinfos; config, world=job.world)
    root_ci === nothing || name_entry!(mod, functions, root_ci)

    compiled = Dict()
    for (ci, src) in codeinfos
        mi = CC.get_ci_mi(ci)
        fs = functions[ci]
        compiled[mi] = (; ci, func=fs.func, specfunc=fs.specfunc,
                        src=job.config.validate ? src : nothing)
    end
    if !haskey(compiled, job.source) && root_ci !== nothing && haskey(compiled, CC.get_ci_mi(root_ci))
        # the root can be less specific than `job.source` (see `compile_method_instance`)
        compiled[job.source] = compiled[CC.get_ci_mi(root_ci)]
    end
    @assert haskey(compiled, job.source)

    gv_to_value = Dict{String,Ptr{Cvoid}}()
    @lock ROOTS_LOCK for (name, x) in output.literals
        push!(ROOTS, x)
        gv_to_value[name] = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), x)
    end
    record(job, length(functions), length(output.fallbacks))
    return mod, compiled, gv_to_value
end

function __init__()
    GPUCompiler.codegen_provider[] = GPUCompiler.CodegenProvider(emit)
end

end # module JuliaCodegenExt
