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
        lower_unreachable_to_return!(job, mod, entry)
        verify(mod)
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
    erase!(f)
    LLVM.name!(new_f, fn)

    return new_f
end


## LLVM IR passes

# lower unreachable instructions to returns with error flags
#
# SPIR-V does not have a trap instruction, so the common trap + unreachable sequence
# results in `OpUnreachable` actually getting executed, which is undefined behavior.
# Instead, we transform unreachable instructions to returns with an error flag that's
# checked by the caller.
function lower_unreachable_to_return!(@nospecialize(job::CompilerJob),
                                      mod::LLVM.Module, entry::LLVM.Function)
    job = current_job::CompilerJob
    changed = false
    @tracepoint "lower unreachable to return" begin

    already_transformed_functions = Set{LLVM.Function}()

    # The pass runs until all unreachable instructions are transformed. During each
    # iteration, we transform all unreachable instructions to returns, and transform all
    # callers to handle the flag, generating a new unreachable when it is set.
    while true
        # Find all functions with unreachable instructions
        functions_with_unreachable = Set{LLVM.Function}()
        for f in functions(mod)
            for bb in blocks(f), inst in instructions(bb)
                if inst isa LLVM.UnreachableInst
                    push!(functions_with_unreachable, f)
                    break
                end
            end
        end
        isempty(functions_with_unreachable) && break

        # Transform functions with unreachable to return a flag next to the original value
        transformed_functions = Dict{LLVM.Function, LLVM.Function}()
        for f in functions_with_unreachable
            ft = function_type(f)
            ret_type = return_type(ft)
            fn = LLVM.name(f)

            # in the case of the entry-point function, we cannot touch its type or returned
            # value, so simply replace the unreachable with a return.
            if f == entry
                @compiler_assert ret_type == LLVM.VoidType() job

                # find un reachables
                unreachables = LLVM.Value[]
                for bb in blocks(f), inst in instructions(bb)
                    if inst isa LLVM.UnreachableInst
                        push!(unreachables, inst)
                    end
                end

                # transform unreachable to return
                @dispose builder=IRBuilder() begin
                    for inst in unreachables
                        position!(builder, inst)
                        ret!(builder)
                        erase!(inst)
                    end
                end

                continue
            end

            # If this is the first time looking at this function, we need to change its type
            if !in(f, already_transformed_functions)
                # Create new return type: {i1, original_type}
                new_ret_type = if ret_type == LLVM.VoidType()
                    LLVM.StructType([LLVM.Int1Type()])
                else
                    LLVM.StructType([LLVM.Int1Type(), ret_type])
                end

                LLVM.name!(f, fn * ".old")
                new_ft = LLVM.FunctionType(new_ret_type, parameters(ft))
                new_f = LLVM.Function(mod, fn, new_ft)
                linkage!(new_f, linkage(f))
                for (i, param) in enumerate(parameters(f))
                    LLVM.name!(parameters(new_f)[i], LLVM.name(param))
                end

                # clone the IR
                value_map = Dict{LLVM.Value, LLVM.Value}(
                    param => parameters(new_f)[i] for (i,param) in enumerate(parameters(f))
                )
                clone_into!(new_f, f; value_map,
                            changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

                # rewrite return instructions
                returns = LLVM.Value[]
                for bb in blocks(new_f), inst in instructions(bb)
                    if inst isa LLVM.RetInst
                        push!(returns, inst)
                    end
                end
                @dispose builder=IRBuilder() begin
                    for inst in returns
                        position!(builder, inst)
                        if ret_type == LLVM.VoidType()
                            # void function: return {false}
                            flag_and_val =
                                insert_value!(builder, UndefValue(new_ret_type),
                                              ConstantInt(LLVM.Int1Type(), false), 0)
                        else
                            # non-void function: return {false, val}
                            val = only(operands(inst))
                            flag_and_val =
                                insert_value!(builder, UndefValue(new_ret_type),
                                              ConstantInt(LLVM.Int1Type(), false), 0)
                            flag_and_val = insert_value!(builder, flag_and_val, val, 1)
                        end
                        ret!(builder, flag_and_val)
                        erase!(inst)
                    end
                end

                transformed_functions[f] = new_f
                push!(already_transformed_functions, new_f)
                f = new_f
            end

            # rewrite unreachable instructions
            ret_type = return_type(function_type(f))
            unreachables = LLVM.Value[]
            for bb in blocks(f), inst in instructions(bb)
                if inst isa LLVM.UnreachableInst
                    push!(unreachables, inst)
                end
            end
            @dispose builder=IRBuilder() begin
                for inst in unreachables
                    position!(builder, inst)
                    if length(elements(ret_type)) == 1
                        # void function: return {true}
                        flag_and_val = insert_value!(builder, UndefValue(ret_type),
                                                     ConstantInt(LLVM.Int1Type(), true), 0)
                    else
                        # non-void function: return {true, undef}
                        val_type = elements(ret_type)[2]
                        flag_and_val = insert_value!(builder, UndefValue(ret_type),
                                                     ConstantInt(LLVM.Int1Type(), true), 0)
                        flag_and_val = insert_value!(builder, flag_and_val,
                                                     UndefValue(val_type), 1)
                    end
                    ret!(builder, flag_and_val)
                    erase!(inst)
                end
            end

            changed = true
        end

        # Rewrite calls
        for (old_f, new_f) in transformed_functions
            calls_to_rewrite = LLVM.CallInst[]
            for use in uses(old_f)
                call_inst = user(use)
                if call_inst isa LLVM.CallInst && called_operand(call_inst) == old_f
                    push!(calls_to_rewrite, call_inst)
                end
            end

            @dispose builder=IRBuilder() begin
                for call_inst in calls_to_rewrite
                    f = LLVM.parent(LLVM.parent(call_inst))
                    position!(builder, call_inst)

                    # Call the new function
                    new_call = call!(builder, function_type(new_f), new_f, arguments(call_inst))
                    callconv!(new_call, callconv(call_inst))

                    # Split the block and branch based on the flag
                    flag = extract_value!(builder, new_call, 0)
                    error_block = BasicBlock(f, "error")
                    move_after(error_block, LLVM.parent(call_inst))
                    continue_block = BasicBlock(f, "continue")
                    move_after(continue_block, error_block)
                    br_inst = br!(builder, flag, error_block, continue_block)

                    # Extract the returned value in the continue block
                    position!(builder, continue_block)
                    if value_type(call_inst) != LLVM.VoidType()
                        value = extract_value!(builder, new_call, 1)
                        replace_uses!(call_inst, value)
                    end
                    @compiler_assert isempty(uses(call_inst)) job
                    erase!(call_inst)

                    # Move the remaining instructions over to the continue block
                    while true
                        inst = LLVM.nextinst(br_inst)
                        inst === nothing && break
                        remove!(inst)
                        insert!(builder, inst)
                    end

                    # Generate an unreachable in the error block
                    position!(builder, error_block)
                    unreachable!(builder)
                end
            end

            @compiler_assert isempty(uses(old_f)) job
            erase!(old_f)
        end
    end

    # Get rid of `llvm.trap` and `noreturn` to prevent reconstructing `unreachable`
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
    for f in functions(mod)
        delete!(function_attributes(f), EnumAttribute("noreturn", 0))
    end

    end
    return changed
end
