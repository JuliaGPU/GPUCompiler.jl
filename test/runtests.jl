using GPUCompiler, LLVM
GPUCompiler.reset_runtime()

using InteractiveUtils
@info "System information:\n" * sprint(io->versioninfo(io; verbose=true))

using ReTestItems
runtests(GPUCompiler; nworkers=min(Sys.CPU_THREADS,4), nworker_threads=1,
                      testitem_timeout=60) do ti
    if ti.name == "GCN" && !LLVM.is_asserts()
        # XXX: GCN's non-0 stack address space triggers LLVM assertions due to Julia bugs
        return false
    end

    true
end
