# implementation of the GPUCompiler interfaces for generating SPIR-V code

# https://github.com/llvm/llvm-project/blob/master/clang/lib/Basic/Targets/SPIR.h
# https://github.com/KhronosGroup/LLVM-SPIRV-Backend/blob/master/llvm/docs/SPIR-V-Backend.rst
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

const SPIRV_LLVM_Translator_jll = LazyModule("SPIRV_LLVM_Translator_jll", UUID("4a5d46fc-d8cf-5151-a261-86b458210efb"))
const SPIRV_Tools_jll = LazyModule("SPIRV_Tools_jll", UUID("6ac6d60f-d740-5983-97d7-a4482c0689f4"))


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

function process_module!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module)
    # calling convention
    for f in functions(mod)
        # JuliaGPU/GPUCompiler.jl#97
        #callconv!(f, LLVM.API.LLVMSPIRFUNCCallConv)
    end
end

function process_entry!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    if job.source.kernel
        # HACK: Intel's compute runtime doesn't properly support SPIR-V's byval attribute.
        #       they do support struct byval, for OpenCL, so wrap byval parameters in a struct.
        entry = wrap_byval(job, mod, entry)

        # calling convention
        callconv!(entry, LLVM.API.LLVMSPIRKERNELCallConv)
    end

    return entry
end

function finish_module!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module)
    # SPIR-V does not support trap, and has no mechanism to abort compute kernels
    # (OpKill is only available in fragment execution mode)
    ModulePassManager() do pm
        add!(pm, ModulePass("RemoveTrap", rm_trap!))
        add!(pm, ModulePass("RemoveFreeze", rm_freeze!))
        run!(pm, mod)
    end
end

@unlocked function mcgen(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMAssemblyFile)
    # The SPIRV Tools don't handle Julia's debug info, rejecting DW_LANG_Julia...
    strip_debuginfo!(mod)

    # translate to SPIR-V
    input = tempname(cleanup=false) * ".bc"
    translated = tempname(cleanup=false) * ".spv"
    write(input, mod)
    SPIRV_LLVM_Translator_jll.llvm_spirv() do translator
        proc = run(ignorestatus(`$translator --spirv-debug-info-version=ocl-100 -o $translated $input`))
        if !success(proc)
            error("""Failed to translate LLVM code to SPIR-V.
                     If you think this is a bug, please file an issue and attach $(input).""")
        end
    end

    # validate
    # XXX: parameterize this on the `validate` driver argument
    # XXX: our code currently doesn't pass the validator
    if Base.JLOptions().debug_level >= 2 && false
        SPIRV_Tools_jll.spirv_val() do validator
            proc = run(ignorestatus(`$validator $translated`))
            if !success(proc)
                error("""Failed to validate generated SPIR-V.
                         If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
            end
        end
    end

    # optimize
    # XXX: parameterize this on the `optimize` driver argument
    # XXX: the optimizer segfaults on some of our code
    optimized = tempname(cleanup=false) * ".spv"
    if false
        SPIRV_Tools_jll.spirv_opt() do optimizer
            proc = run(ignorestatus(`$optimizer -O --skip-validation $translated -o $optimized`))
            if !success(proc)
                error("""Failed to optimize generated SPIR-V.
                         If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
            end
        end
    end

    output = if format == LLVM.API.LLVMObjectFile
        read(translated)
    else
        # disassemble
        SPIRV_Tools_jll.spirv_dis() do disassembler
            read(`$disassembler $optimized`, String)
        end
    end

    rm(input)
    rm(translated)
    #rm(optimized)

    return output
end

# reimplementation that uses `spirv-dis`, giving much more pleasant output
function code_native(io::IO, job::CompilerJob{SPIRVCompilerTarget}; raw::Bool=false, dump_module::Bool=false)
    obj, _ = codegen(:obj, job; strip=!raw, only_entry=!dump_module, validate=false)
    mktemp() do input_path, input_io
        write(input_io, obj)
        flush(input_io)

        SPIRV_Tools_jll.spirv_dis() do disassembler
            if io == stdout
                run(`$disassembler $input_path`)
            else
                mktemp() do output_path, output_io
                    run(`$disassembler $input_path -o $output_path`)
                    asm = read(output_io, String)
                    print(io, asm)
                end
            end
        end
    end
end


## LLVM passes

# remove llvm.trap and its uses from a module
function rm_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "remove trap" begin

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end

        @compiler_assert isempty(uses(trap)) job
        unsafe_delete!(mod, trap)
    end

    end
    return changed
end

# remove freeze and replace uses by the original value
# (KhronosGroup/SPIRV-LLVM-Translator#1140)
function rm_freeze!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "remove freeze" begin

    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        if inst isa LLVM.FreezeInst
            orig = first(operands(inst))
            replace_uses!(inst, orig)
            @compiler_assert isempty(uses(inst)) job
            unsafe_delete!(bb, inst)
            changed = true
        end
    end

    end
    return changed
end

# wrap byval pointers in a single-value struct
function wrap_byval(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry_f::LLVM.Function)
    ctx = context(mod)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType
    @compiler_assert return_type(entry_ft) == LLVM.VoidType(ctx) job

    args = classify_arguments(job, entry_f)
    filter!(args) do arg
        arg.cc != GHOST
    end

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    for arg in args
        typ = if arg.cc == BITS_REF
            st = LLVM.StructType([eltype(arg.codegen.typ)]; ctx)
            LLVM.PointerType(st, addrspace(arg.codegen.typ))
        else
            convert(LLVMType, arg.typ; ctx)
        end
        push!(wrapper_types, typ)
    end
    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, wrapper_fn * ".inner")
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(ctx), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    let builder = Builder(ctx)
        entry = BasicBlock(wrapper_f, "entry"; ctx)
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        # perform argument conversions
        for arg in args
            param = parameters(wrapper_f)[arg.codegen.i]
            attrs = parameter_attributes(wrapper_f, arg.codegen.i)
            if arg.cc == BITS_REF
                if LLVM.version() >= v"12"
                    push!(attrs, TypeAttribute("byval", eltype(wrapper_types[arg.codegen.i]); ctx))
                else
                    push!(attrs, EnumAttribute("byval", 0; ctx))
                end
                ptr = struct_gep!(builder, param, 0)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, param)
                for attr in collect(attrs)
                    push!(parameter_attributes(wrapper_f, arg.codegen.i), attr)
                end
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)

        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    delete!(function_attributes(entry_f), EnumAttribute("noinline", 0; ctx))
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0; ctx))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end
