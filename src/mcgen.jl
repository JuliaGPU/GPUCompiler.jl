# machine code generation

# Finalize the module for backend emission by collecting and lowering all live relocations.
function prepare_execution!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                            relocs::Relocations=Relocations())
    # Clean up first so only live relocations get lowered.
    function cleanup()
        @dispose pb=NewPMPassBuilder() begin
            add!(pb, RecomputeGlobalsAAPass())
            add!(pb, GlobalOptPass())
            add!(pb, GlobalDCEPass())
            add!(pb, StripDeadPrototypesPass())
            run!(pb, mod, llvm_machine(job.config.target))
        end
    end
    cleanup()
    prune_dead_relocations!(mod, relocs)

    collect_cglobal_relocations!(mod, relocs)
    lower_relocations!(job, mod, relocs)

    # Fold constants exposed by eager lowering, and discard slots made dead by
    # either lowering strategy.
    cleanup()
    prune_dead_relocations!(mod, relocs)

    has_unresolved_cglobal_loads(mod, relocs) &&
        error("Unresolved cglobal load after relocation lowering")
    return
end

function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
