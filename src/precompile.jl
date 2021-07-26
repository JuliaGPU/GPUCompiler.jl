function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(process_entry!),CompilerJob{PTXCompilerTarget},LLVM.Module,LLVM.Function})
    Base.precompile(Tuple{Core.kwftype(typeof(load_runtime)),NamedTuple{(:ctx,), Tuple{Context}},typeof(load_runtime),CompilerJob})
    Base.precompile(Tuple{typeof(lower_byval),CompilerJob,LLVM.Module,LLVM.Function})
    Base.precompile(Tuple{typeof(lower_ptls!),LLVM.Module})
    Base.precompile(Tuple{typeof(call!),Builder,GPUCompiler.Runtime.RuntimeMethodInstance,Vector{ConstantExpr}})
    Base.precompile(Tuple{typeof(emit_function!),LLVM.Module,CompilerJob,Function,GPUCompiler.Runtime.RuntimeMethodInstance})
    Base.precompile(Tuple{typeof(mangle_param),Type,Vector{String}})
    Base.precompile(Tuple{typeof(process_module!),CompilerJob{PTXCompilerTarget},LLVM.Module})
    Base.precompile(Tuple{typeof(resolve_cpu_references!),LLVM.Module})
    precompile(emit_llvm, (CompilerJob, Core.MethodInstance, Bool, Bool, Bool, Bool))
end
