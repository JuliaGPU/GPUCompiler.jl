# implementation of the GPUCompiler interfaces for generating SPIR-V code

# https://github.com/llvm/llvm-project/blob/master/clang/lib/Basic/Targets/SPIR.h
# https://github.com/KhronosGroup/LLVM-SPIRV-Backend/blob/master/llvm/docs/SPIR-V-Backend.rst
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

const SPIRV_LLVM_Backend_jll =
    LazyModule("SPIRV_LLVM_Backend_jll",
               UUID("4376b9bf-cff8-51b6-bb48-39421dff0d0c"))
const SPIRV_LLVM_Translator_unified_jll =
    LazyModule("SPIRV_LLVM_Translator_unified_jll",
               UUID("85f0d8ed-5b39-5caa-b1ae-7472de402361"))
const SPIRV_LLVM_Translator_jll =
    LazyModule("SPIRV_LLVM_Translator_jll",
               UUID("4a5d46fc-d8cf-5151-a261-86b458210efb"))
const SPIRV_Tools_jll =
    LazyModule("SPIRV_Tools_jll",
               UUID("6ac6d60f-d740-5983-97d7-a4482c0689f4"))


## target

export SPIRVCompilerTarget

Base.@kwdef struct SPIRVCompilerTarget <: AbstractCompilerTarget
    version::Union{Nothing,VersionNumber} = nothing
    extensions::Vector{String} = []
    supports_fp16::Bool = true
    supports_fp64::Bool = true
    replace_copysign_f16 = false

    backend::Symbol = isavailable(SPIRV_LLVM_Backend_jll) ? :llvm : :khronos
    # XXX: these don't really belong in the _target_ struct
    validate::Bool = false
    optimize::Bool = false
end

function llvm_triple(target::SPIRVCompilerTarget)
    if target.backend == :llvm
        architecture = Int===Int64 ? "spirv64" : "spirv32"  # could also be "spirv" for logical addressing
        subarchitecture = target.version === nothing ? "" : "v$(target.version.major).$(target.version.minor)"
        vendor = "unknown"  # could also be AMD
        os = "unknown"
        environment = "unknown"
        return "$architecture$subarchitecture-$vendor-$os-$environment"
    elseif target.backend == :khronos
        return Int===Int64 ? "spir64-unknown-unknown" : "spirv-unknown-unknown"
    end
end

# SPIRV is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::SPIRVCompilerTarget) = nothing

llvm_datalayout(::SPIRVCompilerTarget) = Int===Int64 ?
    "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1" :
    "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{SPIRVCompilerTarget}) =
    "spirv-" * String(job.config.target.backend)

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

    # replace copysign.f16 intrinsic with manual implementation since pocl doesn't support it
    job.config.target.replace_copysign_f16 && replace_copysign_f16!(mod)

    # translate to SPIR-V
    input = tempname(cleanup=false) * ".bc"
    translated = tempname(cleanup=false) * ".spv"
    write(input, mod)
    if job.config.target.backend === :llvm
        cmd = `$(SPIRV_LLVM_Backend_jll.llc()) $input -filetype=obj -o $translated`

        if !isempty(job.config.target.extensions)
            str = join(map(ext->"+$ext", job.config.target.extensions), ",")
            cmd = `$(cmd) -spirv-ext=$str`
        end
    elseif job.config.target.backend === :khronos
        translator = if isavailable(SPIRV_LLVM_Translator_jll)
            SPIRV_LLVM_Translator_jll.llvm_spirv()
        elseif isavailable(SPIRV_LLVM_Translator_unified_jll)
            SPIRV_LLVM_Translator_unified_jll.llvm_spirv()
        else
            error("This functionality requires the SPIRV_LLVM_Translator_jll or SPIRV_LLVM_Translator_unified_jll package, which should be installed and loaded first.")
        end
        cmd = `$translator -o $translated $input --spirv-debug-info-version=ocl-100`

        if !isempty(job.config.target.extensions)
            str = join(map(ext->"+$ext", job.config.target.extensions), ",")
            cmd = `$(cmd) --spirv-ext=$str`
        end

        if job.config.target.version !== nothing
            cmd = `$(cmd) --spirv-max-version=$(job.config.target.version.major).$(job.config.target.version.minor)`
        end
    end
    try
        run(cmd)
    catch e
        error("""Failed to translate LLVM code to SPIR-V.
                 If you think this is a bug, please file an issue and attach $(input).""")
    end

    # validate
    if job.config.target.validate
        try
            run(`$(SPIRV_Tools_jll.spirv_val()) $translated`)
        catch e
            error("""Failed to validate generated SPIR-V.
                     If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
        end
    end

    # optimize
    optimized = tempname(cleanup=false) * ".spv"
    if job.config.target.optimize
        try
            run(```$(SPIRV_Tools_jll.spirv_opt()) -O --skip-validation
                                                  $translated -o $optimized```)
        catch
            error("""Failed to optimize generated SPIR-V.
                     If you think this is a bug, please file an issue and attach $(input) and $(translated).""")
        end
    else
        cp(translated, optimized)
    end

    output = if format == LLVM.API.LLVMObjectFile
        read(optimized)
    else
        # disassemble
        read(`$(SPIRV_Tools_jll.spirv_dis()) $optimized`, String)
    end

    rm(input)
    rm(translated)
    rm(optimized)

    return output
end

# reimplementation that uses `spirv-dis`, giving much more pleasant output
function code_native(io::IO, job::CompilerJob{SPIRVCompilerTarget}; raw::Bool=false, dump_module::Bool=false)
    config = CompilerConfig(job.config; strip=!raw, only_entry=!dump_module, validate=false)
    obj, _ = JuliaContext() do ctx
        compile(:obj, CompilerJob(job; config))
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
    @tracepoint "remove trap" begin

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                erase!(val)
                changed = true
            end
        end

        @compiler_assert isempty(uses(trap)) job
        erase!(trap)
    end

    end
    return changed
end

# remove freeze and replace uses by the original value
# (KhronosGroup/SPIRV-LLVM-Translator#1140)
function rm_freeze!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @tracepoint "remove freeze" begin

    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        if inst isa LLVM.FreezeInst
            orig = first(operands(inst))
            replace_uses!(inst, orig)
            @compiler_assert isempty(uses(inst)) job
            erase!(inst)
            changed = true
        end
    end

    end
    return changed
end

# wrap byval pointers in a single-value struct
function wrap_byval(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ft = function_type(f)::LLVM.FunctionType

    # find the byval parameters
    byval = BitVector(undef, length(parameters(ft)))
    types = Vector{LLVMType}(undef, length(parameters(ft)))
    for i in 1:length(byval)
        byval[i] = false
        for attr in collect(parameter_attributes(f, i))
            if kind(attr) == kind(TypeAttribute("byval", LLVM.VoidType()))
                byval[i] = true
                types[i] = value(attr)
            end
        end
    end

    # generate the wrapper function type & definition
    new_types = LLVM.LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        typ = if byval[i]
            llvm_typ = convert(LLVMType, types[i])
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
                llvm_typ = convert(LLVMType, types[i])
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
                llvm_typ = convert(LLVMType, types[i])
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
    erase!(f)
    LLVM.name!(new_f, fn)

    return new_f
end

# replace llvm.copysign.f16 calls with manual implementation
function replace_copysign_f16!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @tracepoint "replace copysign f16" begin

    # Find llvm.copysign.f16 intrinsic
    copysign_name = "llvm.copysign.f16"
    if haskey(functions(mod), copysign_name)
        copysign_fn = functions(mod)[copysign_name]

        # Process all uses of the intrinsic
        for use in uses(copysign_fn)
            call_inst = user(use)
            if isa(call_inst, LLVM.CallInst)
                @dispose builder=IRBuilder() begin
                    # Position builder before the call
                    position!(builder, call_inst)

                    # Get operands (x and y)
                    x = operands(call_inst)[1]  # magnitude
                    y = operands(call_inst)[2]  # sign source

                    # Create the replacement implementation
                    i16_type = LLVM.IntType(16)

                    # Bitcast half values to i16
                    x_bits = bitcast!(builder, x, i16_type)
                    y_bits = bitcast!(builder, y, i16_type)

                    # XOR the bit patterns and check if result is negative
                    xor_result = xor!(builder, y_bits, x_bits)
                    is_negative = icmp!(builder, LLVM.API.LLVMIntSLT, xor_result,
                                      ConstantInt(i16_type, 0))

                    # Create fneg of x
                    neg_x = fneg!(builder, x)

                    # Select between neg_x and x based on the sign test
                    result = select!(builder, is_negative, neg_x, x)

                    # Replace uses and erase the original call
                    replace_uses!(call_inst, result)
                    erase!(call_inst)
                    changed = true
                end
            end
        end

        # Remove the intrinsic declaration if no longer used
        if isempty(uses(copysign_fn))
            erase!(copysign_fn)
        end
    end

    end
    return changed
end

