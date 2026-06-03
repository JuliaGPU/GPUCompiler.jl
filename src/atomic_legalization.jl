# in bytes
function smallest_atomic_size(job)
    return 4
end

# 1. Legalize sizes
# 2. Legalize ordering through fences
# 3. Legalize operations through cmpswp

function legalize_atomics!(job, ir)
    dl = datalayout(ir)
    for f in functions(ir), bb in blocks(f), inst in instructions(bb)
        if inst isa LLVM.LoadInst && is_atomic(inst)
            typ = value_type(inst)
            if sizeof(dl, typ) < smallest_atomic_size(job)
                # Replace with a larger atomic type
                @dispose builder = IRBuilder() begin
                    position!(builder, inst)
                    ptr = only(operands(inst))
                    load = load!(builder, LLVM.IntType(smallest_atomic_size(job) * 8), ptr)
                    # TODO: alignment, ordering, etc.
                    # TODO: Handle floats and other types appropriately
                    # TODO: Do we need to shift the loaded value?
                    new_inst = trunc!(builder, load, typ)

                    replace_uses!(inst, new_inst)
                    erase!(inst)
                end
            end
        elseif inst isa LLVM.StoreInst && is_atomic(inst)
        end
    end
    return ir
end
