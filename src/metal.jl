# implementation of the GPUCompiler interfaces for generating Metal code

const Metal_LLVM_Tools_jll = LazyModule("Metal_LLVM_Tools_jll", UUID("0418c028-ff8c-56b8-a53e-0f9676ed36fc"))

## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    macos::VersionNumber
end

function Base.hash(target::MetalCompilerTarget, h::UInt)
    h = hash(target.macos, h)
end

source_code(target::MetalCompilerTarget) = "text"

# Metal is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::MetalCompilerTarget) = nothing

llvm_triple(target::MetalCompilerTarget) = "air64-apple-macosx$(target.macos)"

llvm_datalayout(target::MetalCompilerTarget) =
    "e-p:64:64:64"*
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
    "-f32:32:32-f64:64:64"*
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"*
    "-n8:16:32"

needs_byval(job::CompilerJob{MetalCompilerTarget}) = false


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{MetalCompilerTarget}) = "metal-macos$(job.config.target.macos)"

isintrinsic(@nospecialize(job::CompilerJob{MetalCompilerTarget}), fn::String) =
    return startswith(fn, "air.")

const LLVMMETALFUNCCallConv   = LLVM.API.LLVMCallConv(102)
const LLVMMETALKERNELCallConv = LLVM.API.LLVMCallConv(103)

function process_module!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module)
    # calling convention
    for f in functions(mod)
        #callconv!(f, LLVMMETALFUNCCallConv)
        # XXX: this makes InstCombine erase kernel->func calls.
        #      do we even need this? if we do, do so in metallib-instead.
    end
end

function process_entry!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.config.kernel
        # calling convention
        callconv!(entry, LLVMMETALKERNELCallConv)
    end

    return entry
end

function validate_module(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module)
    errors = IRError[]

    T_double = LLVM.DoubleType(context(mod))

    for fun in functions(mod), bb in blocks(fun), inst in instructions(bb)
        if value_type(inst) == T_double || any(param->value_type(param) == T_double, operands(inst))
            bt = backtrace(inst)
            push!(errors, ("use of double floating-point value", bt, inst))
        end
    end

    return errors
end

# TODO: why is this done in finish_module? maybe just in process_entry?
function finish_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.config.kernel
        entry = pass_by_reference!(job, mod, entry)

        add_input_arguments!(job, mod, entry)
        entry = LLVM.functions(mod)[entry_fn]
    end

    return functions(mod)[entry_fn]
end

function finish_ir!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module,
                                  entry::LLVM.Function)
    entry = invoke(finish_ir!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.config.kernel
        entry = add_address_spaces!(job, mod, entry)

        add_argument_metadata!(job, mod, entry)

        add_module_metadata!(job, mod)
    end

    return functions(mod)[entry_fn]
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
    strip_debuginfo!(mod)  # XXX: is this needed?

    # translate to metallib
    input = tempname(cleanup=false) * ".bc"
    translated = tempname(cleanup=false) * ".metallib"
    write(input, mod)
    Metal_LLVM_Tools_jll.metallib_as() do assembler
        proc = run(ignorestatus(`$assembler -o $translated $input`))
        if !success(proc)
            error("""Failed to translate LLVM code to MetalLib.
                     If you think this is a bug, please file an issue and attach $(input).""")
        end
    end

    output = if format == LLVM.API.LLVMObjectFile
        read(translated)
    else
        # disassemble
        Metal_LLVM_Tools_jll.metallib_dis() do disassembler
            read(`$disassembler -o - $translated`, String)
        end
    end

    rm(input)
    rm(translated)

    return output
end


# generic pointer removal
#
# every pointer argument (i.e. byref objs) to a kernel needs an address space attached.
# this pass rewrites pointers to reference arguments to be located in address space 1.
#
# NOTE: this pass only rewrites byref objs, not plain pointers being passed; the user is
# responsible for making sure these pointers have an address space attached (using LLVMPtr).
#
# NOTE: this pass also only rewrites pointers _without_ address spaces, which requires it to
# be executed after optimization (where Julia's address spaces are stripped). If we ever
# want to execute it earlier, adapt remapType to rewrite all pointer types.
function add_address_spaces!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ctx = context(mod)
    ft = function_type(f)

    # find the byref parameters
    byref = BitVector(undef, length(parameters(ft)))
    let args = classify_arguments(job, ft)
        filter!(args) do arg
            arg.cc != GHOST
        end
        for arg in args
            byref[arg.codegen.i] = (arg.cc == BITS_REF)
        end
    end

    function remapType(src)
        # TODO: shouldn't we recurse into structs here, making sure the parent object's
        #       address space matches the contained one? doesn't matter right now as we
        #       only use LLVMPtr (i.e. no rewriting of contained pointers needed) in the
        #       device addrss space (i.e. no mismatch between parent and field possible)
        dst = if src isa LLVM.PointerType && addrspace(src) == 0
            LLVM.PointerType(remapType(eltype(src)), #=device=# 1)
        else
            src
        end
        return dst
    end

    # generate the new function type & definition
    new_types = LLVMType[]
    for (i, param) in enumerate(parameters(ft))
        if byref[i]
            push!(new_types, remapType(param::LLVM.PointerType))
        else
            push!(new_types, param)
        end
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # we cannot simply remap the function arguments, because that will not propagate the
    # address space changes across, e.g, bitcasts (the dest would still be in AS 0).
    # using a type remapper on the other hand changes too much, including unrelated insts.
    # so instead, we load the arguments in stack slots and dereference them so that we can
    # keep on using the original IR that assumed pointers without address spaces
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder(ctx) begin
        entry = BasicBlock(new_f, "conversion"; ctx)
        position!(builder, entry)

        # perform argument conversions
        for (i, param) in enumerate(parameters(ft))
            if byref[i]
                # load the argument in a stack slot
                val = load!(builder, eltype(parameters(new_ft)[i]), parameters(new_f)[i])
                ptr = alloca!(builder, eltype(param))
                store!(builder, val, ptr)
                push!(new_args, ptr)
            else
                push!(new_args, parameters(new_f)[i])
            end
            for attr in collect(parameter_attributes(f, i))
                push!(parameter_attributes(new_f, i), attr)
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        # fall through
        br!(builder, blocks(new_f)[2])
    end

    # remove the old function
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    # clean-up after this pass (which runs after optimization)
    @dispose pm=ModulePassManager() begin
        cfgsimplification!(pm)
        scalar_repl_aggregates!(pm)
        early_cse!(pm)
        instruction_combining!(pm)

        run!(pm, mod)
    end

    return new_f
end


# value-to-reference conversion
#
# Metal doesn't support passing valuse, so we need to convert those to references instead
function pass_by_reference!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ctx = context(mod)
    ft = function_type(f)
    @compiler_assert return_type(ft) == LLVM.VoidType(ctx) job

    # generate the new function type & definition
    args = classify_arguments(job, ft)
    new_types = LLVM.LLVMType[]
    bits_as_reference = BitVector(undef, length(parameters(ft)))
    for arg in args
        if arg.cc == BITS_VALUE && !(arg.typ <: Ptr || arg.typ <: Core.LLVMPtr)
            # pass the value as a reference instead
            push!(new_types, LLVM.PointerType(arg.codegen.typ, #=Constant=# 1))
            bits_as_reference[arg.codegen.i] = true
        elseif arg.cc != GHOST
            push!(new_types, arg.codegen.typ)
            bits_as_reference[arg.codegen.i] = false
        end
    end
    new_ft = LLVM.FunctionType(return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (i, (arg, new_arg)) in enumerate(zip(parameters(f), parameters(new_f)))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=IRBuilder(ctx) begin
        entry = BasicBlock(new_f, "entry"; ctx)
        position!(builder, entry)

        # perform argument conversions
        for arg in args
            if arg.cc != GHOST
                if bits_as_reference[arg.codegen.i]
                    # load the reference to get a value back
                    val = load!(builder, eltype(parameters(new_ft)[arg.codegen.i]),
                                parameters(new_f)[arg.codegen.i])
                    push!(new_args, val)
                else
                    push!(new_args, parameters(new_f)[arg.codegen.i])
                end
            end
        end

        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i,param) in enumerate(parameters(f))
        )

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)

        # fall through
        br!(builder, blocks(new_f)[2])
    end

    # set the attributes (needs to happen _after_ cloning)
    # TODO: verify that clone copies other attributes,
    #       and that other uses of clone don't set parameters before cloning
    for i in 1:length(parameters(new_f))
        if bits_as_reference[i]
            # add appropriate attributes
            # TODO: other attributes (nonnull, readonly, align, dereferenceable)?
            ## we've just emitted a load, so the pointer itself cannot be captured
            push!(parameter_attributes(new_f, i), EnumAttribute("nocapture", 0; ctx))
            ## Metal.jl emits separate buffers for each scalar argument
            push!(parameter_attributes(new_f, i), EnumAttribute("noalias", 0; ctx))
        end
    end

    # remove the old function
    # NOTE: if we ever have legitimate uses of the old function, create a shim instead
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    return new_f
end


# kernel input arguments
#
# hardware index counters (thread id, group id, etc) aren't accessed via intrinsics,
# but using special arguments to the kernel function.

const kernel_intrinsics = Dict()
for intr in [
        "dispatch_quadgroups_per_threadgroup", "dispatch_simdgroups_per_threadgroup",
        "quadgroup_index_in_threadgroup", "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup", "simdgroups_per_threadgroup",
        "thread_index_in_quadgroup", "thread_index_in_simdgroup",
        "thread_index_in_threadgroup", "thread_execution_width", "threads_per_simdgroup"],
    (llvm_typ, julia_typ) in [
        ("i32",  UInt32),
        ("i16",  UInt16),
    ]
    push!(kernel_intrinsics, "julia.air.$intr.$llvm_typ" =>  (name=intr, typ=julia_typ))
end
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"],
    (llvm_typ, julia_typ) in [
        ("i32",   UInt32),
        ("v2i32", NTuple{2, VecElement{UInt32}}),
        ("v3i32", NTuple{3, VecElement{UInt32}}),
        ("i16",   UInt16),
        ("v2i16", NTuple{2, VecElement{UInt16}}),
        ("v3i16", NTuple{3, VecElement{UInt16}}),
    ]
    push!(kernel_intrinsics, "julia.air.$intr.$llvm_typ" => (name=intr, typ=julia_typ))
end

function argument_type_name(typ)
    if typ isa LLVM.IntegerType && width(typ) == 16
        "ushort"
    elseif typ isa LLVM.IntegerType && width(typ) == 32
        "uint"
    elseif typ isa LLVM.VectorType
         argument_type_name(eltype(typ)) * string(Int(size(typ)))
    else
        error("Cannot encode unknown type `$typ`")
    end
end

function add_input_arguments!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                              entry::LLVM.Function)
    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    # figure out which intrinsics are used and need to be added as arguments
    used_intrinsics = filter(keys(kernel_intrinsics)) do intr_fn
        haskey(functions(mod), intr_fn)
    end |> collect
    nargs = length(used_intrinsics)

    # determine which functions need these arguments
    worklist = Set{LLVM.Function}([entry])
    for intr_fn in used_intrinsics
        push!(worklist, functions(mod)[intr_fn])
    end
    worklist_length = 0
    while worklist_length != length(worklist)
        # iteratively discover functions that use an intrinsic or any function calling it
        worklist_length = length(worklist)
        additions = LLVM.Function[]
        for f in worklist, use in uses(f)
            inst = user(use)::Instruction
            bb = LLVM.parent(inst)
            new_f = LLVM.parent(bb)
            in(new_f, worklist) || push!(additions, new_f)
        end
        for f in additions
            push!(worklist, f)
        end
    end
    for intr_fn in used_intrinsics
        delete!(worklist, functions(mod)[intr_fn])
    end

    # add the arguments
    # NOTE: we don't need to be fine-grained here, as unused args will be removed during opt
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = LLVM.name(f)
        ft = function_type(f)
        LLVM.name!(f, fn * ".orig")
        # create a new function
        new_param_types = LLVMType[parameters(ft)...]

        for intr_fn in used_intrinsics
            llvm_typ = convert(LLVMType, kernel_intrinsics[intr_fn].typ; ctx)
            push!(new_param_types, llvm_typ)
        end
        new_ft = LLVM.FunctionType(return_type(ft), new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        linkage!(new_f, linkage(f))
        for (arg, new_arg) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_arg, LLVM.name(arg))
        end
        for (intr_fn, new_arg) in zip(used_intrinsics, parameters(new_f)[end-nargs+1:end])
            LLVM.name!(new_arg, kernel_intrinsics[intr_fn].name)
        end

        workmap[f] = new_f
    end

    # clone and rewrite the function bodies.
    # we don't need to rewrite much as the arguments are added last.
    for (f, new_f) in workmap
        # map the arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_param, LLVM.name(param))
            value_map[param] = new_param
        end

        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)

        # we can't remove this function yet, as we might still need to rewrite any called,
        # but remove the IR already
        empty!(f)
    end

    # drop unused constants that may be referring to the old functions
    # XXX: can we do this differently?
    for f in worklist
        for use in uses(f)
            val = user(use)
            if val isa LLVM.ConstantExpr && isempty(uses(val))
                LLVM.unsafe_destroy!(val)
            end
        end
    end

    # update other uses of the old function, modifying call sites to pass the arguments
    function rewrite_uses!(f, new_f)
        # update uses
        @dispose builder=IRBuilder(ctx) begin
            for use in uses(f)
                val = user(use)
                if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                    callee_f = LLVM.parent(LLVM.parent(val))
                    # forward the arguments
                    position!(builder, val)
                    new_val = if val isa LLVM.CallInst
                        call!(builder, function_type(new_f), new_f,
                              [arguments(val)..., parameters(callee_f)[end-nargs+1:end]...],
                              operand_bundles(val))
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.ConstantExpr && opcode(val) == LLVM.API.LLVMBitCast
                    # XXX: why isn't this caught by the value materializer above?
                    target = operands(val)[1]
                    @assert target == f
                    new_val = LLVM.const_bitcast(new_f, value_type(val))
                    rewrite_uses!(val, new_val)
                    # we can't simply replace this constant expression, as it may be used
                    # as a call, taking arguments (so we need to rewrite it to pass the input arguments)

                    # drop the old constant if it is unused
                    # XXX: can we do this differently?
                    if isempty(uses(val))
                        LLVM.unsafe_destroy!(val)
                    end
                else
                    error("Cannot rewrite unknown use of function: $val")
                end
            end
        end
    end
    for (f, new_f) in workmap
        rewrite_uses!(f, new_f)
        @assert isempty(uses(f))
        unsafe_delete!(mod, f)
    end

    # replace uses of the intrinsics with references to the input arguments
    for (i, intr_fn) in enumerate(used_intrinsics)
        intr = functions(mod)[intr_fn]
        for use in uses(intr)
            val = user(use)
            callee_f = LLVM.parent(LLVM.parent(val))
            if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                replace_uses!(val, parameters(callee_f)[end-nargs+i])
            else
                error("Cannot rewrite unknown use of function: $val")
            end

            @assert isempty(uses(val))
            unsafe_delete!(LLVM.parent(val), val)
        end
        @assert isempty(uses(intr))
        unsafe_delete!(mod, intr)
    end

    return
end


# argument metadata generation
#
# module metadata is used to identify buffers that are passed as kernel arguments.

function add_argument_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                entry::LLVM.Function)
    ctx = context(mod)

    ## argument info
    arg_infos = Metadata[]

    # Iterate through arguments and create metadata for them
    args = classify_arguments(job, function_type(entry))
    i = 1
    for arg in args
        haskey(arg, :codegen) || continue
        @assert arg.codegen.typ isa LLVM.PointerType

        # NOTE: we emit the bare minimum of argument metadata to support
        #       bindless argument encoding. Actually using the argument encoder
        #       APIs (deprecated in Metal 3) turned out too difficult, given the
        #       undocumented nature of the argument metadata, and the complex
        #       arguments we encounter with typical Julia kernels.

        md = Metadata[]

        # argument index
        @assert arg.codegen.i == i
        push!(md, Metadata(ConstantInt(Int32(i-1); ctx)))

        push!(md, MDString("air.buffer"; ctx))

        push!(md, MDString("air.location_index"; ctx))
        push!(md, Metadata(ConstantInt(Int32(i-1); ctx)))

        # XXX: unknown
        push!(md, Metadata(ConstantInt(Int32(1); ctx)))

        push!(md, MDString("air.read_write"; ctx)) # TODO: Check for const array

        push!(md, MDString("air.address_space"; ctx))
        push!(md, Metadata(ConstantInt(Int32(addrspace(arg.codegen.typ)); ctx)))

        arg_type = if arg.typ <: Core.LLVMPtr
            arg.typ.parameters[1]
        else
            arg.typ
        end

        push!(md, MDString("air.arg_type_size"; ctx))
        push!(md, Metadata(ConstantInt(Int32(sizeof(arg_type)); ctx)))

        push!(md, MDString("air.arg_type_align_size"; ctx))
        push!(md, Metadata(ConstantInt(Int32(Base.datatype_alignment(arg_type)); ctx)))

        push!(arg_infos, MDNode(md; ctx))

        i += 1
    end

    # Create metadata for argument intrinsics last
    for intr_arg in parameters(entry)[i:end]
        intr_fn = LLVM.name(intr_arg)

        arg_info = Metadata[]

        push!(arg_info, Metadata(ConstantInt(Int32(i-1); ctx)))
        push!(arg_info, MDString("air.$intr_fn" ; ctx))

        push!(arg_info, MDString("air.arg_type_name" ; ctx))
        push!(arg_info, MDString(argument_type_name(value_type(intr_arg)); ctx))

        arg_info = MDNode(arg_info; ctx)
        push!(arg_infos, arg_info)

        i += 1
    end
    arg_infos = MDNode(arg_infos; ctx)


    ## stage info
    stage_infos = Metadata[]
    stage_infos = MDNode(stage_infos; ctx)

    kernel_md = MDNode([entry, stage_infos, arg_infos]; ctx)
    push!(metadata(mod)["air.kernel"], kernel_md)

    return
end


# module-level metadata

# TODO: determine limits being set dynamically
function add_module_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    ctx = context(mod)

    # register max device buffer count
    max_buff = Metadata[]
    push!(max_buff, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_buff, MDString("air.max_device_buffers"; ctx))
    push!(max_buff, Metadata(ConstantInt(Int32(31); ctx)))
    max_buff = MDNode(max_buff; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_buff)

    # register max constant buffer count
    max_const_buff_md = Metadata[]
    push!(max_const_buff_md, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_const_buff_md, MDString("air.max_constant_buffers"; ctx))
    push!(max_const_buff_md, Metadata(ConstantInt(Int32(31); ctx)))
    max_const_buff_md = MDNode(max_const_buff_md; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_const_buff_md)

    # register max threadgroup buffer count
    max_threadgroup_buff_md = Metadata[]
    push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_threadgroup_buff_md, MDString("air.max_threadgroup_buffers"; ctx))
    push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(31); ctx)))
    max_threadgroup_buff_md = MDNode(max_threadgroup_buff_md; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_threadgroup_buff_md)

    # register max texture buffer count
    max_textures_md = Metadata[]
    push!(max_textures_md, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_textures_md, MDString("air.max_textures"; ctx))
    push!(max_textures_md, Metadata(ConstantInt(Int32(128); ctx)))
    max_textures_md = MDNode(max_textures_md; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_textures_md)

    # register max write texture buffer count
    max_rw_textures_md = Metadata[]
    push!(max_rw_textures_md, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_rw_textures_md, MDString("air.max_read_write_textures"; ctx))
    push!(max_rw_textures_md, Metadata(ConstantInt(Int32(8); ctx)))
    max_rw_textures_md = MDNode(max_rw_textures_md; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_rw_textures_md)

    # register max sampler count
    max_samplers_md = Metadata[]
    push!(max_samplers_md, Metadata(ConstantInt(Int32(7); ctx)))
    push!(max_samplers_md, MDString("air.max_samplers"; ctx))
    push!(max_samplers_md, Metadata(ConstantInt(Int32(16); ctx)))
    max_samplers_md = MDNode(max_samplers_md; ctx)
    push!(metadata(mod)["llvm.module.flags"], max_samplers_md)

    # add compiler identification
    llvm_ident_md = Metadata[]
    push!(llvm_ident_md, MDString("Julia $(VERSION) with Metal.jl"; ctx))
    llvm_ident_md = MDNode(llvm_ident_md; ctx)
    push!(metadata(mod)["llvm.ident"], llvm_ident_md)

    # add AIR version
    air_md = Metadata[]
    push!(air_md, Metadata(ConstantInt(Int32(2); ctx)))
    push!(air_md, Metadata(ConstantInt(Int32(4); ctx)))
    push!(air_md, Metadata(ConstantInt(Int32(0); ctx)))
    air_md = MDNode(air_md; ctx)
    push!(metadata(mod)["air.version"], air_md)

    # add language version
    air_lang_md = Metadata[]
    push!(air_lang_md, MDString("Metal"; ctx))
    push!(air_lang_md, Metadata(ConstantInt(Int32(2); ctx)))
    push!(air_lang_md, Metadata(ConstantInt(Int32(4); ctx)))
    push!(air_lang_md, Metadata(ConstantInt(Int32(0); ctx)))
    air_lang_md = MDNode(air_lang_md; ctx)
    push!(metadata(mod)["air.language_version"], air_lang_md)

    # set sdk version
    sdk_version!(mod, job.config.target.macos)

    return
end
