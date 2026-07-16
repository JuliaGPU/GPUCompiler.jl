# machine code generation

# final preparations for the module to be compiled to machine code
# these passes should not be run when e.g. compiling to write to disk.
function prepare_execution!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                            refs::HostReferences=HostReferences())
    prune_dead_host_reference_slots!(mod, refs)
    collect_runtime_global_references!(job, mod, refs)
    lower_host_references!(job, mod, refs)
    @dispose pb=NewPMPassBuilder() begin
        register!(pb, CollectRuntimeGlobalReferencesPass(job, refs))

        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        add!(pb, CollectRuntimeGlobalReferencesPass(job, refs))
        add!(pb, GlobalDCEPass())
        add!(pb, StripDeadPrototypesPass())

        run!(pb, mod, llvm_machine(job.config.target))
    end

    prune_dead_host_reference_slots!(mod, refs)
    lower_host_references!(job, mod, refs)
    return
end

function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
