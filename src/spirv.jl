# implementation of the GPUCompiler interfaces for generating SPIR-V code

# https://github.com/llvm/llvm-project/blob/master/clang/lib/Basic/Targets/SPIR.h
# https://github.com/KhronosGroup/LLVM-SPIRV-Backend/blob/master/llvm/docs/SPIR-V-Backend.rst
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

const SPIRV_LLVM_Translator_unified_jll = LazyModule("SPIRV_LLVM_Translator_unified_jll", UUID("85f0d8ed-5b39-5caa-b1ae-7472de402361"))
const SPIRV_Tools_jll = LazyModule("SPIRV_Tools_jll", UUID("6ac6d60f-d740-5983-97d7-a4482c0689f4"))


## target

export SPIRVCompilerTarget

Base.@kwdef struct SPIRVCompilerTarget <: AbstractCompilerTarget
    extensions::Vector{String} = []
    supports_fp16::Bool = true
    supports_fp64::Bool = true
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

function finish_module!(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    # update calling convention
    for f in functions(mod)
        # JuliaGPU/GPUCompiler.jl#97
        #callconv!(f, LLVM.API.LLVMSPIRFUNCCallConv)
    end
    if job.config.kernel
        callconv!(entry, LLVM.API.LLVMSPIRKERNELCallConv)
    end

    # HACK: Intel's compute runtime doesn't properly support SPIR-V's byval attribute.
    #       they do support struct byval, for OpenCL, so wrap byval parameters in a struct.
    if job.config.kernel
        entry = wrap_byval(job, mod, entry)
    end

    # add module metadata
    ## OpenCL 2.0
    push!(metadata(mod)["opencl.ocl.version"],
          MDNode([ConstantInt(Int32(2)),
                  ConstantInt(Int32(0))]))
    ## SPIR-V 1.5
    push!(metadata(mod)["opencl.spirv.version"],
          MDNode([ConstantInt(Int32(1)),
                  ConstantInt(Int32(5))]))

    return entry
end

function validate_ir(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module)
    errors = IRError[]

    # support for half and double depends on the target
    if !job.config.target.supports_fp16
        append!(errors, check_ir_values(mod, LLVM.HalfType()))
    end
    if !job.config.target.supports_fp64
        append!(errors, check_ir_values(mod, LLVM.DoubleType()))
    end

    return errors
end

@unlocked function mcgen(job::CompilerJob{SPIRVCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMAssemblyFile)
    # The SPIRV Tools don't handle Julia's debug info, rejecting DW_LANG_Julia...
    strip_debuginfo!(mod)

    # SPIR-V does not support trap, and has no mechanism to abort compute kernels
    # (OpKill is only available in fragment execution mode)
    rm_trap!(mod)

    # the LLVM to SPIR-V translator does not support the freeze instruction
    # (SPIRV-LLVM-Translator#1140)
    rm_freeze!(mod)


    # translate to SPIR-V
    input = tempname(cleanup=false) * ".bc"
    translated = tempname(cleanup=false) * ".spv"
    options = `--spirv-debug-info-version=ocl-100`
    if !isempty(job.config.target.extensions)
        str = join(map(ext->"+$ext", job.config.target.extensions), ",")
        options = `$options --spirv-ext=$str`
    end
    write(input, mod)
    let cmd = `$(SPIRV_LLVM_Translator_unified_jll.llvm_spirv()) $options -o $translated $input`
        proc = run(ignorestatus(cmd))
        if !success(proc)
            error("""Failed to translate LLVM code to SPIR-V.
                     If you think this is a bug, please file an issue and attach $(input).""")
        end
    end

    # validate
    # XXX: parameterize this on the `validate` driver argument
    # XXX: our code currently doesn't pass the validator
    #if Base.JLOptions().debug_level >= 2
    #    cmd = `$(SPIRV_Tools_jll.spirv_val()) $translated`
    #    proc = run(ignorestatus(cmd))
    #    if !success(proc)
    #        error("""Failed to validate generated SPIR-V.
    #                 If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
    #    end
    #end

    # optimize
    # XXX: parameterize this on the `optimize` driver argument
    # XXX: the optimizer segfaults on some of our code
    optimized = tempname(cleanup=false) * ".spv"
    #let cmd = `$(SPIRV_Tools_jll.spirv_opt()) -O --skip-validation $translated -o $optimized`
    #    proc = run(ignorestatus(cmd))
    #    if !success(proc)
    #        error("""Failed to optimize generated SPIR-V.
    #                 If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
    #    end
    #end

    output = if format == LLVM.API.LLVMObjectFile
        read(translated)
    else
        # disassemble
        read(`$(SPIRV_Tools_jll.spirv_dis()) $translated`, String)
    end

    rm(input)
    rm(translated)
    #rm(optimized)

    return output
end

# reimplementation that uses `spirv-dis`, giving much more pleasant output
function code_native(io::IO, job::CompilerJob{SPIRVCompilerTarget}; raw::Bool=false, dump_module::Bool=false)
    obj, _ = JuliaContext() do ctx
        compile(:obj, job; strip=!raw, only_entry=!dump_module, validate=false)
    end
    mktemp() do input_path, input_io
        write(input_io, obj)
        flush(input_io)

        disassembler = SPIRV_Tools_jll.spirv_dis()
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
    ft = function_type(f)::LLVM.FunctionType

    args = classify_arguments(job, ft)
    filter!(args) do arg
        arg.cc != GHOST
    end

    # find the byval parameters
    byval = BitVector(undef, length(parameters(ft)))
    if LLVM.version() >= v"12"
        for i in 1:length(byval)
            attrs = collect(parameter_attributes(f, i))
            byval[i] = any(attrs) do attr
                kind(attr) == kind(TypeAttribute("byval", LLVM.VoidType()))
            end
        end
    else
        # XXX: byval is not round-trippable on LLVM < 12 (see maleadt/LLVM.jl#186)
        for arg in args
            byval[arg.idx] = (arg.cc == BITS_REF)
        end
    end

    # generate the wrapper function type & definition
    new_types = LLVM.LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        typ = if byval[i]
            llvm_typ = convert(LLVMType, args[i].typ)
            st = LLVM.StructType([llvm_typ])
            LLVM.PointerType(st, addrspace(param))
        else
            param
        end
        push!(new_types, typ)
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = Vector{LLVM.Value}()
    @dispose builder=IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(new_f))
            if byval[i]
                llvm_typ = convert(LLVMType, args[i].typ)
                ptr = struct_gep!(builder, LLVM.StructType([llvm_typ]), param, 0)
                push!(new_args, ptr)
            else
                push!(new_args, param)
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # apply byval attributes again (`clone_into!` didn't due to the type mismatch)
        for i in 1:length(byval)
            attrs = parameter_attributes(new_f, i)
            if byval[i]
                llvm_typ = convert(LLVMType, args[i].typ)
                push!(attrs, TypeAttribute("byval", LLVM.StructType([llvm_typ])))
            end
        end

        # fall through
        br!(builder, collect(blocks(new_f))[2])
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    return new_f
end
