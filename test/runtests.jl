using GPUCompiler, LLVM
GPUCompiler.reset_runtime()

using InteractiveUtils
@info "System information:\n" * sprint(io->versioninfo(io; verbose=true))

import SPIRV_LLVM_Translator_unified_jll
import SPIRV_Tools_jll

using ReTestItems
runtests(GPUCompiler; nworkers=min(Sys.CPU_THREADS,4), nworker_threads=1,
                      testitem_timeout=120) do ti
    if ti.name == "GCN" && LLVM.is_asserts()
        # XXX: GCN's non-0 stack address space triggers LLVM assertions due to Julia bugs
        return false
    end

    @dispose ctx=Context() begin
        # XXX: some back-ends do not support opaque pointers
        if ti.name in ["Metal"] && !supports_typed_pointers(ctx)
            return false
        end
    end

    if ti.name in ["PTX", "GCN", "PTX precompile"] && Sys.isapple()
        # support for AMDGPU and NVTX on macOS has been removed from Julia's LLVM build
        return false
    end


    if ti.name in ["SPIRV"] && !(SPIRV_LLVM_Translator_unified_jll.is_available() && SPIRV_Tools_jll.is_available())
        # SPIRV needs it's tools to be available
        return false
    end

    if ti.name in ["PTX precompile", "native precompile"] && VERSION < v"1.11-"
        # precompile needs v1.11
        return false
    end

    true
end
