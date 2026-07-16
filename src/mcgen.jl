# machine code generation

# final preparations for the module to be compiled to machine code
# these passes should not be run when e.g. compiling to write to disk.
function run_cleanup_pipeline!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    @dispose pb=NewPMPassBuilder() begin
        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        add!(pb, GlobalDCEPass())
        add!(pb, StripDeadPrototypesPass())
        run!(pb, mod, llvm_machine(job.config.target))
    end
    return
end

function prepare_execution!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                            refs::HostReferences=HostReferences())
    # Clean up first so only live references get slots and get lowered.
    run_cleanup_pipeline!(job, mod)
    prune_dead_host_reference_slots!(mod, refs)

    collect_runtime_global_references!(job, mod, refs)
    lower_host_references!(job, mod, refs)

    # Fold constants exposed by eager lowering, and discard slots made dead by either
    # lowering strategy.
    run_cleanup_pipeline!(job, mod)
    prune_dead_host_reference_slots!(mod, refs)

    has_unresolved_runtime_global_loads(mod, refs) &&
        error("Unresolved Julia runtime global load after host-reference lowering")
    return
end

function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
