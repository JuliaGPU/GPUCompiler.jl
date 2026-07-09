# machine code generation

# final preparations for the module to be compiled to machine code
# these passes should not be run when e.g. compiling to write to disk.
function prepare_execution!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    @dispose pb=NewPMPassBuilder() begin
        register!(pb, ResolveCPUReferencesPass(job))

        add!(pb, RecomputeGlobalsAAPass())
        add!(pb, GlobalOptPass())
        add!(pb, ResolveCPUReferencesPass(job))
        add!(pb, GlobalDCEPass())
        add!(pb, StripDeadPrototypesPass())

        run!(pb, mod, llvm_machine(job.config.target))
    end

    return
end

# some Julia code contains references to objects in the CPU run-time,
# without actually using the contents or functionality of those objects.
#
# prime example are type tags, which reference the address of the allocated type.
# since those references are ephemeral, we can't eagerly resolve and emit them in the IR,
# but at the same time the GPU can't resolve them at run-time.
#
# this pass performs that resolution at link time.
struct ResolveCPUReferences
    job::CompilerJob
end
function (self::ResolveCPUReferences)(mod::LLVM.Module)
    changed = false

    for f in functions(mod)
        fn = LLVM.name(f)
        if isdeclaration(f) && !LLVM.isintrinsic(f) && startswith(fn, "jl_")
            # lazily resolve the address of the binding; some symbols only exist
            # within the JIT (e.g. `jl_get_pgcstack_resolved`) and cannot be looked up,
            # but such symbols are only ever called, not loaded from.
            dereferenced = nothing
            function resolve_binding()
                if dereferenced === nothing
                    address = ccall(:jl_cglobal, Any, (Any, Any), fn, UInt)
                    dereferenced = LLVM.ConstantInt(unsafe_load(address))
                end
                dereferenced
            end

            function replace_bindings!(value)
                changed = false
                for use in uses(value)
                    val = user(use)
                    if isa(val, LLVM.ConstantExpr)
                        # recurse
                        changed |= replace_bindings!(val)
                    elseif isa(val, LLVM.LoadInst)
                        # resolve
                        replace_uses!(val, resolve_binding())
                        erase!(val)
                        # FIXME: iterator invalidation?
                        changed = true
                    end
                end
                changed
            end

            changed |= replace_bindings!(f)
        end
    end

    return changed
end
ResolveCPUReferencesPass(job) =
    NewPMModulePass("ResolveCPUReferences", ResolveCPUReferences(job))


function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
