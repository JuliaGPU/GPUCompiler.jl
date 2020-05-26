# implementation of the GPUCompiler interfaces for generating SPIR-V code

# https://github.com/llvm/llvm-project/blob/master/clang/lib/Basic/Targets/SPIR.h
# https://github.com/KhronosGroup/LLVM-SPIRV-Backend/blob/master/llvm/docs/SPIR-V-Backend.rst
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

using SPIRV_LLVM_Translator_jll


## target

export SPIRVCompilerTarget

Base.@kwdef struct SPIRVCompilerTarget <: AbstractCompilerTarget
end

llvm_triple(::SPIRVCompilerTarget) = Int===Int64 ? "spir64-unknown-unknown" : "spirv-unknown-unknown"

# SPIRV is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::SPIRVCompilerTarget) = nothing

llvm_datalayout(::SPIRVCompilerTarget) = Int===Int64 ?
    "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024" :
    "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{SPIRVCompilerTarget}) = "spirv"

function process_kernel!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module, kernel::LLVM.Function)
    # calling convention
    for fun in functions(mod)
        callconv!(kernel, LLVM.API.LLVMSPIRFUNCCallConv)
    end
    callconv!(kernel, LLVM.API.LLVMSPIRKERNELCallConv)

    return kernel
end

function add_lowering_passes!(job::CompilerJob{SPIRVCompilerTarget}, pm::LLVM.PassManager)
    add!(pm, ModulePass("RemoveTrap", rm_trap!))
end

function mcgen(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module, f::LLVM.Function,
               format=LLVM.API.LLVMAssemblyFile)
    # write the bitcode to a temporary file (the SPIRV Translator library doesn't have a C API)
    mktemp() do input, input_io
        write(input_io, mod)
        flush(input_io)

        # compile to SPIR-V
        mktemp() do output, output_io
            llvm_spirv() do translator
                cmd = `$translator`
                if format == LLVM.API.LLVMAssemblyFile
                    cmd = `$cmd -spirv-text`
                end
                cmd = `$cmd -o $output $input`
                run(cmd)
            end

            # read back the file
            if format == LLVM.API.LLVMAssemblyFile
                read(output_io, String)
            else
                read(output_io)
            end
        end
    end
end


## LLVM passes

# SPIR-V does not support trap, and has no mechanism to abort compute kernels
# (OpKill is only available in fragment execution mode)
function rm_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "hide trap" begin

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end
    end

    end
    return changed
end
