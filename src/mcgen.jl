# machine code generation

# Finalize the module for backend emission by collecting and lowering all live relocations.
function prepare_execution!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                            relocs::Relocations=Relocations())
    # Clean up first so only live relocations get lowered.
    function cleanup(; fold_instructions=false)
        @dispose pb=NewPMPassBuilder() begin
            if fold_instructions
                add!(pb, NewPMFunctionPassManager()) do fpm
                    add!(fpm, instcombine_pass(job))
                end
            end
            add!(pb, RecomputeGlobalsAAPass())
            add!(pb, GlobalOptPass())
            add!(pb, GlobalDCEPass())
            add!(pb, StripDeadPrototypesPass())
            run!(pb, mod, llvm_machine(job.config.target))
        end
    end
    cleanup()
    prune_dead_relocations!(mod, relocs)

    # Linking left one DICompileUnit per linked module; fold the copies so the debug-info
    # graph does not depend on link order (see determinism.jl).
    dedup_compile_units!(mod)

    # For non-`:bake` strategies this already ran at the end of `emit_llvm` (so the
    # `:llvm`-level metadata is complete); re-running is a no-op since rewritten loads
    # target namespaced `gpu_jl_*` slots, which are not collection candidates. It remains
    # load-bearing for `:bake` and for direct `emit_asm` callers that pass fresh
    # `Relocations`.
    collect_cglobal_relocations!(job, mod, relocs)

    # Lower, then freeze: from here the manifest describes emitted code, so no record may be
    # added, dropped or reordered. Nothing can die on its own either — a record that survived
    # the prune above is either baked into an initializer, anchored in `llvm.used`, or
    # rewritten into a table load. Freezing after lowering, not before, leaves
    # `bake_relocations!` free to consume the records as its final act.
    lower_relocations!(job, mod, relocs)
    freeze!(relocs)

    # Fold constants exposed by eager lowering, and drop globals the lowering left dead.
    cleanup(; fold_instructions=true)

    has_unresolved_cglobal_loads(mod, relocs) &&
        error("Unresolved cglobal load after relocation lowering")
    return
end

function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
