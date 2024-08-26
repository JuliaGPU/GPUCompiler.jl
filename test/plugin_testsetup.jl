@testsetup module Plugin

using Test
using ReTestItems
import LLVM
import GPUCompiler

function mark(x)
    ccall("extern gpucompiler.mark", llvmcall, Nothing, (Int,), x)
end

function remove_mark!(@nospecialize(job), intrinsic, mod::LLVM.Module)
    changed = false

    for use in LLVM.uses(intrinsic)
        val = LLVM.user(use)
        if isempty(LLVM.uses(val))
            LLVM.erase!(val)
            changed = true
        else
            # the validator will detect this
        end
    end

    return changed
end

GPUCompiler.register_plugin!("gpucompiler.mark", false, 
                             pipeline_callback=remove_mark!)

end