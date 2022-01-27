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
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # calling convention
        callconv!(entry, LLVM.API.LLVMSPIRKERNELCallConv)
    end

    return entry
end

function finish_module!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    ctx = context(mod)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # HACK: Intel's compute runtime doesn't properly support SPIR-V's byval attribute.
        #       they do support struct byval, for OpenCL, so wrap byval parameters in a struct.
        entry = wrap_byval(job, mod, entry)
    end

    # add module metadata
    ## OpenCL 2.0
    push!(metadata(mod)["opencl.ocl.version"],
          MDNode([ConstantInt(Int32(2); ctx),
                  ConstantInt(Int32(0); ctx)]; ctx))
    ## SPIR-V 1.5
    push!(metadata(mod)["opencl.spirv.version"],
          MDNode([ConstantInt(Int32(1); ctx),
                  ConstantInt(Int32(5); ctx)]; ctx))

    return entry
end

@unlocked function mcgen(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMAssemblyFile)
    # The SPIRV Tools don't handle Julia's debug info, rejecting DW_LANG_Julia...
    strip_debuginfo!(mod)

    ModulePassManager() do pm
        # SPIR-V does not support trap, and has no mechanism to abort compute kernels
        # (OpKill is only available in fragment execution mode)
        add!(pm, ModulePass("RemoveTrap", rm_trap!))

        # the LLVM to SPIR-V translator does not support the freeze instruction
        # (SPIRV-LLVM-Translator#1140)
        add!(pm, ModulePass("RemoveFreeze", rm_freeze!))

        run!(pm, mod)
    end

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
function wrap_byval(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ctx = context(mod)
    ft = eltype(llvmtype(f)::LLVM.PointerType)::LLVM.FunctionType
    @compiler_assert return_type(ft) == LLVM.VoidType(ctx) job

    # find the byval parameters
    byval = BitVector(undef, length(parameters(ft)))
    if LLVM.version() >= v"12"
        for i in 1:length(byval)
            attrs = collect(parameter_attributes(f, i))
            byval[i] = any(attrs) do attr
                kind(attr) == kind(EnumAttribute("byval", 0; ctx))
            end
        end
    else
        # XXX: byval is not round-trippable on LLVM < 12 (see maleadt/LLVM.jl#186)
        has_kernel_state = kernel_state_type(job) !== Nothing
        orig_ft = if has_kernel_state
            # the kernel state has been added here already, so strip the first parameter
            LLVM.FunctionType(return_type(ft), parameters(ft)[2:end]; vararg=isvararg(ft))
        else
            ft
        end
        args = classify_arguments(job, orig_ft)
        filter!(args) do arg
            arg.cc != GHOST
        end
        for arg in args
            if arg.cc == BITS_REF
                # NOTE: +1 since this pass runs after introducing the kernel state
                byval[arg.codegen.i+has_kernel_state] = true
            end
        end
        if has_kernel_state
            byval[1] = true
        end
    end

    # generate the wrapper function type & definition
    new_types = LLVM.LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        typ = if byval[i]
            st = LLVM.StructType([eltype(param)]; ctx)
            LLVM.PointerType(st, addrspace(param))
        else
            param
        end
        push!(new_types, typ)
    end
    new_ft = LLVM.FunctionType(LLVM.VoidType(ctx), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = Vector{LLVM.Value}()
    Builder(ctx) do builder
        entry = BasicBlock(new_f, "entry"; ctx)
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(new_f))
            if byval[i]
                ptr = struct_gep!(builder, param, 0)
                push!(new_args, ptr)
            else
                push!(new_args, param)
            end
        end

        # inline the old IR
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)
        # NOTE: we need global changes because LLVM 12 wants to clone debug metadata

        # apply byval attributes again (`clone_into!` didn't due to the type mismatch)
        for i in 1:length(byval)
            attrs = parameter_attributes(new_f, i)
            if byval[i]
                if LLVM.version() >= v"12"
                    push!(attrs, TypeAttribute("byval", eltype(new_types[i]); ctx))
                else
                    push!(attrs, EnumAttribute("byval", 0; ctx))
                end
            end
        end

        # fall through
        br!(builder, collect(blocks(new_f))[2])
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    # XXX: there may still be metadata using this function. RAUW updates those,
    #      but asserts on a debug build due to the updated function type.
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    # clean-up
    # NOTE: byval wrapping happens very late, after optimization
    ModulePassManager() do pm
        # merge GEPs
        instruction_combining!(pm)

        # fold the entry bb into the rest of the function
        cfgsimplification!(pm)

        run!(pm, mod)
    end

    return new_f
end
