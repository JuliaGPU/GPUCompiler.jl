function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(lower_throw!),LLVM.Module})   # time: 0.17275752
    Base.precompile(Tuple{typeof(lower_byval),CompilerJob,LLVM.Module,LLVM.Function})   # time: 0.055481136
    Base.precompile(Tuple{typeof(call!),Builder,GPUCompiler.Runtime.RuntimeMethodInstance,Vector{ConstantExpr}})   # time: 0.022024114
    Base.precompile(Tuple{Type{GPUInterpreter},CodeCache,Core.MethodTable,UInt64})   # time: 0.014925144
    Base.precompile(Tuple{typeof(resolve_cpu_references!),LLVM.Module})   # time: 0.014757595
    Base.precompile(Tuple{typeof(lower_ptls!),LLVM.Module})   # time: 0.013304743
    Base.precompile(Tuple{Core.kwftype(typeof(load_runtime)),NamedTuple{(:ctx,), Tuple{Context}},typeof(load_runtime),CompilerJob})   # time: 0.012303286
    Base.precompile(Tuple{typeof(lower_gc_frame!),LLVM.Function})   # time: 0.008227327
    Base.precompile(Tuple{typeof(llvm_machine),PTXCompilerTarget})   # time: 0.008051048
    Base.precompile(Tuple{typeof(hide_trap!),LLVM.Module})   # time: 0.007331288
    Base.precompile(Tuple{typeof(mangle_param),Type,Vector{String}})   # time: 0.003355128
    isdefined(GPUCompiler, Symbol("#55#59")) && Base.precompile(Tuple{getfield(GPUCompiler, Symbol("#55#59")),ModulePassManager})   # time: 0.001026008
    Base.precompile(Tuple{typeof(julia_datalayout),PTXCompilerTarget})   # time: 0.001025637

    precompile(emit_llvm, (CompilerJob, Core.MethodInstance))
end
