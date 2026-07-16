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

# some Julia code contains references to objects in the CPU run-time,
# without actually using the contents or functionality of those objects.
#
# prime example are type tags, which reference the address of the allocated type.
# since those references are ephemeral, we can't eagerly resolve and emit them in the IR,
# but at the same time the GPU can't resolve them at run-time.
#
# this pass performs that resolution at link time.
struct CollectRuntimeGlobalReferences
    job::CompilerJob
    refs::HostReferences
end
function (self::CollectRuntimeGlobalReferences)(mod::LLVM.Module)
    changed = false

    for f in [collect(functions(mod)); collect(globals(mod))]
        fn = LLVM.name(f)
        # Julia value globals are already represented by an identity in `refs`; they may
        # use a `jl_*` spelling but are not exported libjulia runtime globals.
        f isa LLVM.GlobalVariable && haskey(self.refs.slots, fn) && continue
        if isdeclaration(f) && (!(f isa LLVM.Function) || !LLVM.isintrinsic(f)) &&
           startswith(fn, "jl_")
            slot = nothing
            function runtime_global_slot()
                if slot === nothing
                    name = safe_name("gpu_" * fn)
                    slot = GlobalVariable(mod, host_reference_word_type(), name)
                    initializer!(slot, null(global_value_type(slot)))
                    linkage!(slot, LLVM.API.LLVMExternalLinkage)
                    extinit!(slot, true)
                    actual_name = LLVM.name(slot)
                    haskey(self.refs.slots, actual_name) &&
                        error("Duplicate Julia runtime global slot '$actual_name'")
                    self.refs.slots[actual_name] = CGlobalRef(Symbol(fn))
                end
                slot
            end

            function replace_bindings!(value)
                changed = false
                for use in collect(uses(value))
                    val = user(use)
                    if isa(val, LLVM.ConstantExpr)
                        # recurse
                        changed |= replace_bindings!(val)
                    elseif isa(val, LLVM.LoadInst)
                        T = value_type(val)
                        if !(T isa LLVM.PointerType ||
                             (T isa LLVM.IntegerType && width(T) == 8sizeof(UInt)))
                            error("Unsupported Julia runtime global '$fn' load of LLVM type $T")
                        end
                        @dispose builder=IRBuilder() begin
                            position!(builder, val)
                            replacement = load!(builder, host_reference_word_type(),
                                                runtime_global_slot())
                            if T isa LLVM.PointerType
                                replacement = inttoptr!(builder, replacement, T)
                            end
                            replace_uses!(val, replacement)
                        end
                        erase!(val)
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
collect_runtime_global_references!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                   refs::HostReferences) =
    CollectRuntimeGlobalReferences(job, refs)(mod)
CollectRuntimeGlobalReferencesPass(job, refs=HostReferences()) =
    NewPMModulePass("CollectRuntimeGlobalReferences", CollectRuntimeGlobalReferences(job, refs))


function mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module, format=LLVM.API.LLVMAssemblyFile)
    tm = llvm_machine(job.config.target)

    return String(emit(tm, mod, format))
end
