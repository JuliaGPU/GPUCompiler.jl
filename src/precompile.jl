struct PrecompilationCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::Compiler{<:Any,PrecompilationCompilerParams}) = GPUCompiler

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    # simple compilation
    target = NativeCompilerTarget()
    params = PrecompilationCompilerParams()
    compiler = Compiler(target, params)
    source = FunctionSpec(identity, Tuple{Nothing}, true, "dummy")
    job = CompilerJob(compiler, source)
    compile(:asm, job; strip=true, only_entry=true)
    empty!(GLOBAL_CI_CACHE) # JuliaLang/julia#41714

    precompile(emit_llvm, (CompilerJob, MethodInstance))
end
