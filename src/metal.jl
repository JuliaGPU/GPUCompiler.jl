# implementation of the GPUCompiler interfaces for generating Metal code

const Metal_LLVM_Tools_jll = LazyModule("Metal_LLVM_Tools_jll", UUID("0418c028-ff8c-56b8-a53e-0f9676ed36fc"))

## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    macos::VersionNumber
end

function Base.hash(target::MetalCompilerTarget, h::UInt)
    hash(target.macos, h)
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
runtime_slug(job::CompilerJob{MetalCompilerTarget}) = "metal-macos$(job.target.macos)"

isintrinsic(@nospecialize(job::CompilerJob{MetalCompilerTarget}), fn::String) =
    return startswith(fn, "air.")

const LLVMMETALFUNCCallConv   = LLVM.API.LLVMCallConv(102)
const LLVMMETALKERNELCallConv = LLVM.API.LLVMCallConv(103)

const metal_struct_names = [:MtlDeviceArray, :MtlDeviceMatrix, :MtlDeviceVector]

# Initial mapping of types - There has to be a better way
const jl_type_to_c = Dict(
                    Float32 => "float",
                    Float16 => "half",
                    Int64   => "long",
                    UInt64  => "ulong",
                    Int32   => "int",
                    UInt32  => "uint",
                    Int16   => "short",
                    UInt16  => "ushort",
                    Int8    => "char",
                    UInt8   => "uchar",
                    Bool    => "char" # xxx: ?
                )

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

    if job.source.kernel
        # calling convention
        callconv!(entry, LLVMMETALKERNELCallConv)
    end

    return entry
end

# TODO: why is this done in finish_module? maybe just in process_entry?
function finish_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.source.kernel
        entry = pass_by_reference!(job, mod, entry)

        arguments = add_input_arguments!(job, mod, entry)
        entry = LLVM.functions(mod)[entry_fn]

        add_argument_metadata!(job, mod, entry, arguments)

        add_module_metadata!(job, mod)
    end

    return functions(mod)[entry_fn]
end

function finish_ir!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module,
                                  entry::LLVM.Function)
    entry = invoke(finish_ir!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.source.kernel
        add_address_spaces!(mod, entry)
    end

    return functions(mod)[entry_fn]
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
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
# every pointer argument (e.g. byref objs) to a kernel needs an address space attached.
function add_address_spaces!(mod::LLVM.Module, f::LLVM.Function)
    ctx = context(mod)
    ft = eltype(llvmtype(f))

    function remapType(src)
        # TODO: recurse in structs
        dst = if src isa LLVM.PointerType && addrspace(src) == 0
            LLVM.PointerType(remapType(eltype(src)), #=device=# 1)
        else
            src
        end
        # TODO: cache
        return dst
    end

    # generate the new function type & definition
    new_types = LLVMType[remapType(typ) for typ in parameters(ft)]
    new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(parameters(f), parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # map the parameters
    value_map = Dict{LLVM.Value, LLVM.Value}(
        param => new_param for (param, new_param) in zip(parameters(f), parameters(new_f))
    )
    value_map[f] = new_f

    # before D96531 (part of LLVM 13), clone_into! wants to duplicate debug metadata
    # when the functions are part of the same module. that is invalid, because it
    # results in desynchronized debug intrinsics (GPUCompiler#284), so remove those.
    if LLVM.version() < v"13"
        removals = LLVM.Instruction[]
        for bb in blocks(f), inst in instructions(bb)
            if inst isa LLVM.CallInst && LLVM.name(called_value(inst)) == "llvm.dbg.declare"
                push!(removals, inst)
            end
        end
        for inst in removals
            @assert isempty(uses(inst))
            unsafe_delete!(LLVM.parent(inst), inst)
        end
        changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges
    else
        changes = LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly
    end

    function type_mapper(typ)
        remapType(typ)
    end

    clone_into!(new_f, f; value_map, changes, type_mapper)

    # update calls to overloaded intrinsic, re-mangling their names
    # XXX: shouldn't clone_into! do this?
    LLVM.@dispose builder=Builder(ctx) begin
        for bb in blocks(new_f), inst in instructions(bb)
            if inst isa LLVM.CallBase
                callee_f = called_value(inst)
                LLVM.isintrinsic(callee_f) || continue
                intr = Intrinsic(callee_f)
                isoverloaded(intr) || continue

                # get an appropriately-overloaded intrinsic instantiation
                # XXX: apparently it differs per intrinsics which arguments to take into
                #      consideration when generating an overload? for example, with memcpy
                #      the trailing i1 argument is not included in the overloaded name.
                intr_f = if intr == Intrinsic("llvm.memcpy")
                    LLVM.Function(mod, intr, llvmtype.(arguments(inst)[1:end-1]))
                else
                    error("Unsupported intrinsic; please file an issue.")
                end

                # create a call to the new intrinsic
                # TODO: wrap setCalledFunction instead of using an IRBuilder
                position!(builder, inst)
                new_inst = if inst isa LLVM.CallInst
                    call!(builder, intr_f, arguments(inst), operand_bundles(inst))
                else
                    # TODO: invoke and callbr
                    error("Rewrite of $(typeof(inst))-based calls is not implemented: $inst")
                end
                callconv!(new_inst, callconv(inst))

                # replace the old call
                replace_uses!(inst, new_inst)
                @assert isempty(uses(inst))
                unsafe_delete!(LLVM.parent(inst), inst)
            end
        end
    end

    # remove the old function
    fn = LLVM.name(f)
    @assert isempty(uses(f))
    replace_metadata_uses!(f, new_f)
    unsafe_delete!(mod, f)
    LLVM.name!(new_f, fn)

    return new_f
end


# value-to-reference conversion
#
# Metal doesn't support passing valuse, so we need to convert those to references instead
function pass_by_reference!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ctx = context(mod)
    ft = eltype(llvmtype(f))
    @compiler_assert LLVM.return_type(ft) == LLVM.VoidType(ctx) job

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
    new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (i, (arg, new_arg)) in enumerate(zip(parameters(f), parameters(new_f)))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    # emit IR performing the "conversions"
    new_args = LLVM.Value[]
    @dispose builder=Builder(ctx) begin
        entry = BasicBlock(new_f, "entry"; ctx)
        position!(builder, entry)

        # perform argument conversions
        for arg in args
            if arg.cc != GHOST
                if bits_as_reference[arg.codegen.i]
                    # load the reference to get a value back
                    val = load!(builder, parameters(new_f)[arg.codegen.i])
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
        "thread_index_in_quadgroup", "thread_index_in_simdgroup", "thread_index_in_threadgroup",
        "thread_execution_width", "threads_per_simdgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("i16",   "ushort",  UInt16),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("v2i32", "uint2", NTuple{2, VecElement{UInt32}}),
        ("v3i32", "uint3", NTuple{3, VecElement{UInt32}}),
        ("i16",   "ushort",  UInt16),
        ("v2i16", "ushort2", NTuple{2, VecElement{UInt16}}),
        ("v3i16", "ushort3", NTuple{3, VecElement{UInt16}}),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end

function add_input_arguments!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                              entry::LLVM.Function)
    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    # figure out which intrinsics are used and need to be added as arguments
    used_intrinsics = filter(keys(kernel_intrinsics)) do intr_fn
        haskey(functions(mod), intr_fn)
    end |> collect
    # TODO: Figure out how to not be inefficient with these changes
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
    # NOTE: we could be more fine-grained, only adding the specific intrinsics used by this function.
    #       not sure if that's worth it though.
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = LLVM.name(f)
        ft = eltype(llvmtype(f))
        LLVM.name!(f, fn * ".orig")
        # create a new function
        new_param_types = LLVMType[parameters(ft)...]

        for intr_fn in used_intrinsics
            llvm_typ = convert(LLVMType, kernel_intrinsics[intr_fn].julia_typ; ctx)
            push!(new_param_types, llvm_typ)
        end
        new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        linkage!(new_f, linkage(f))
        for (arg, new_arg) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_arg, LLVM.name(arg))
        end
        for (intr_fn, new_arg) in zip(used_intrinsics, parameters(new_f)[end-nargs+1:end])
            LLVM.name!(new_arg, kernel_intrinsics[intr_fn].air_name)
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
        @dispose builder=Builder(ctx) begin
            for use in uses(f)
                val = user(use)
                if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                    callee_f = LLVM.parent(LLVM.parent(val))
                    # forward the arguments
                    position!(builder, val)
                    new_val = if val isa LLVM.CallInst
                        call!(builder, new_f, [arguments(val)..., parameters(callee_f)[end-nargs+1:end]...], operand_bundles(val))
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
                    new_val = LLVM.const_bitcast(new_f, llvmtype(val))
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

    return used_intrinsics
end


# argument metadata generation
#
# module metadata is used to identify buffers that are passed as kernel arguments.

# TODO: MDString(::Symbol)?

# XXX: we translate Julia types to C typenames. that's not required, but reduces the diff
#      while working on this code. after that, just report the full type name (no `nameof`)
type_mapping = Dict(
    "Int32" => "int",
    "Int64" => "long",
)
function humanize_typename(typ)
    name = string(nameof(typ))
    get(type_mapping, name, name)
end

# return the Julia element type of a type represented by an LLVM ArrayType
# (i.e., not only tuples, but every homogeneous structure)
function generalized_eltype(typ)
    if typ <: Tuple
        eltype(typ)
    else # homogeneous struct
        typs = unique(fieldtypes(typ))
        @assert length(typs) == 1 "LLVM array type used with non-homogeneous struct $typ"
        typs[]
    end
end

# TODO: express different argument types using Julia types
@enum ArgumentKind begin
    GhostArgument
    BufferArgument
    ArrayArgument
    ConstantArgument
    StructArgument
    IndirectStructArgument
end
struct ArgumentInfo
    name::String
    kind::ArgumentKind
    id::Union{Nothing,Int}
    fields::Union{Nothing,Vector{ArgumentInfo}}

    ArgumentInfo(name, kind, id=nothing, fields=nothing) = new(string(name), kind, id, fields)
end

# TODO: split metadata encoding in two separate passes:
# - first build the ArgumentInfo tree
# - then use it to generate metadata and pass the same tree to the front-end

function add_argument_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                entry::LLVM.Function, used_intrinsics::Vector)
    ctx = context(mod)

    ## argument info
    arg_infos = Metadata[]      # information for Metal
    arg_metas = ArgumentInfo[]  # metadata for Metal.jl

    # Iterate through arguments and create metadata for them
    args = classify_arguments(job, eltype(llvmtype(entry)))
    for (i, arg) in enumerate(args)
        if !haskey(arg, :codegen)
            push!(arg_metas, ArgumentInfo(arg.name, GhostArgument))
            continue
        end
        @assert arg.codegen.typ isa LLVM.PointerType

        # if an argument is a structure containing resources (buffers, textures, samplers,
        # constants), as opposed to passing a resource directly, it will be encoded using a
        # argument encoder targeting an argument buffer. this requires the fields of the
        # structure to have a (monotonically increasing) resource identifier. see also:
        # https://developer.apple.com/documentation/metal/buffers/indexing_argument_buffers

        arg_info, arg_meta, _ = encode_argument(arg; ctx)
        push!(arg_infos, MDNode(arg_info; ctx))

        push!(arg_metas, arg_meta)
    end

    # Create metadata for argument intrinsics last
    for (i, intr_fn) in enumerate(used_intrinsics)
        arg_info = Metadata[]
        push!(arg_info, Metadata(ConstantInt(Int32(length(parameters(entry))-i); ctx)))
        push!(arg_info, MDString("air." * kernel_intrinsics[intr_fn].air_name; ctx))

        push!(arg_info, MDString("air.arg_type_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_typ; ctx))

        # NOTE: this is optional
        push!(arg_info, MDString("air.arg_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_name; ctx))

        arg_info = MDNode(arg_info; ctx)
        push!(arg_infos, arg_info)
    end
    arg_infos = MDNode(arg_infos; ctx)
    ## stage info
    stage_infos = Metadata[]
    stage_infos = MDNode(stage_infos; ctx)

    kernel_md = MDNode([entry, stage_infos, arg_infos]; ctx)
    push!(metadata(mod)["air.kernel"], kernel_md)

    job.meta[:arguments] = arg_metas
    return
end

function encode_argument(arg, id=0; ctx)
    md = Metadata[]

    # argument index
    push!(md, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))

    if arg.codegen.typ isa LLVM.PointerType
        eltyp = eltype(arg.codegen.typ)

        push!(md, MDString("air.buffer"; ctx))

        push!(md, MDString("air.location_index"; ctx))
        if id == 0
            # for top-level arguments (which start at id=0), this value seems to represent
            # the index of the argument, whereas for nested arguments it's just the id...
            push!(md, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
        else
            push!(md, Metadata(ConstantInt(Int32(id); ctx)))
        end

        push!(md, Metadata(ConstantInt(Int32(1); ctx))) # XXX: unknown

        push!(md, MDString("air.read_write"; ctx)) # TODO: Check for const array

        if eltyp isa LLVM.StructType
            push!(md, MDString("air.struct_type_info"; ctx))
            nested_info, nested_meta, ids, indirect = encode_struct(arg, id, eltyp; ctx)
            if indirect
                md[2] = MDString("air.indirect_buffer"; ctx)
            end
            push!(md, MDNode(nested_info; ctx))
            info = ArgumentInfo(arg.name,
                                indirect ? IndirectStructArgument : StructArgument,
                                nothing, nested_meta)
        elseif eltyp isa LLVM.ArrayType
            # XXX: duplication with encode_struct
            ids = length(eltyp)
            let eltyp=eltyp
                eltyp = eltype(eltyp)
                while eltyp isa LLVM.ArrayType
                    ids *= length(eltyp)
                    eltyp = eltype(eltyp)
                end
            end
            info = ArgumentInfo(arg.name, ArrayArgument, id)
        else
            ids = 1
            info = ArgumentInfo(arg.name, BufferArgument, id)
        end

        # buffers require to report the alignment of the element type
        # TODO: deduplicate with the same logic below
        # TODO: what to do with array types?
        arg_type = if arg.typ <: Core.LLVMPtr
            arg.typ.parameters[1]
        else
            arg.typ
        end

        push!(md, MDString("air.arg_type_size"; ctx))
        push!(md, Metadata(ConstantInt(Int32(sizeof(arg_type)); ctx)))

        push!(md, MDString("air.arg_type_align_size"; ctx))
        push!(md, Metadata(ConstantInt(Int32(Base.datatype_alignment(arg_type)); ctx)))
    else
        # this can only happen with indirect argument, i.e., as part of a struct

        ids = 1

        push!(md, MDString("air.indirect_constant"; ctx))

        push!(md, MDString("air.location_index"; ctx))
        push!(md, Metadata(ConstantInt(Int32(id); ctx)))
        push!(md, Metadata(ConstantInt(Int32(1); ctx))) # XXX: unknown

        info = ArgumentInfo(arg.name, ConstantArgument, id)
    end

    push!(md, MDString("air.arg_type_name"; ctx))
    if arg.typ <: Core.LLVMPtr
        # in the case of buffers, the element type is encoded
        push!(md, MDString(humanize_typename(arg.typ.parameters[1]); ctx))
    else
        push!(md, MDString(humanize_typename(arg.typ); ctx))
    end
    # TODO: duplication with encode_struct

    # NOTE: this is optional
    push!(md, MDString("air.arg_name"; ctx))
    push!(md, MDString(string(arg.name); ctx))

    return md, info, ids
end

function encode_struct(arg, offset, typ=arg.codegen.typ; indirect=false, ctx)
    md = Metadata[]
    infos = ArgumentInfo[]

    # the `indirect` keyword argument controls whether we should emit `!indirect_argument`
    # metadata for all fields of this structure. this flag is set when encountering a
    # (nested) buffer, in which case earlier fields also need an `!indirect_argument` entry.
    # we implement this by restarting the process and calling `encode_struct` again.

    # the `typ` argument allows overriding the type of this structure. this is needed to
    # overcome the difference between struct arguments (passed by reference, i.e., pointer)
    # and nested structs which are a value contained in another object.

    fields = classify_fields(arg.typ, typ)
    ids = 0
    id = offset
    for (i, field) in enumerate(fields)
        # skip ghost fields
        if !haskey(field, :codegen)
            push!(infos, ArgumentInfo(field.name, GhostArgument))
            continue
        end

        if field.codegen.typ isa LLVM.StructType
            push!(md, MDString("air.struct_type_info"; ctx))
            nested_md, nested_info, field_ids, nested_indirect =
                encode_struct(field, id; ctx, indirect)
            if nested_indirect && !indirect
                # backtrack and restart the process with indirect=true
                return encode_struct(arg, offset, typ; ctx, indirect=true)
            end
            push!(md, MDNode(nested_md; ctx))
            field_info = ArgumentInfo(field.name,
                                      indirect ? IndirectStructArgument : StructArgument,
                                      nothing, nested_info)
        elseif field.codegen.typ isa LLVM.ArrayType
            field_ids = length(field.codegen.typ)
            element_typ = eltype(field.codegen.typ)
            while element_typ isa LLVM.ArrayType
                field_ids *= length(element_typ)
                element_typ = eltype(element_typ)
            end
            field_info = ArgumentInfo(field.name, ArrayArgument, id)
        elseif field.codegen.typ isa LLVM.VectorType
            error("Vector types are not supported")
        elseif field.codegen.typ isa LLVM.PointerType
            if !indirect
                # NOTE: we detect actual LLVM pointers here, meaning we only support buffer
                #       fields encoded by Core.LLVMPtr, not Base.Ptr (encoded as an integer)
                # TODO: can we support regular Julia pointers? if so, we wouldn't have to
                #       look for Ptr/LLVMPtr anymore, but could just use the LLVM type
                #       representation. on the other hand, to support textures etc, we'll
                #       have to look at the Julia type representation anyway.
                return encode_struct(arg, offset, typ; ctx, indirect=true)
            end
            field_ids = 1
            field_info = ArgumentInfo(field.name, BufferArgument, id)
        else
            field_ids = 1
            field_info = ArgumentInfo(field.name, ConstantArgument, indirect ? id : nothing)
        end

        # Offset in bytes from start of struct
        push!(md, Metadata(ConstantInt(Int32(fieldoffset(arg.typ, i)); ctx)))

        # Size of element in bytes, always 8 for buffers (pointer size)
        if field.codegen.typ isa LLVM.ArrayType
            push!(md, Metadata(ConstantInt(Int32(sizeof(generalized_eltype(field.typ))); ctx)))
        else
            push!(md, Metadata(ConstantInt(Int32(sizeof(field.typ)); ctx)))
        end

        # Length of element, always 0 for buffers
        if field.codegen.typ isa LLVM.ArrayType
            push!(md, Metadata(ConstantInt(Int32(length(field.codegen.typ)); ctx)))
        else
            push!(md, Metadata(ConstantInt(Int32(0); ctx)))
        end

        # Field type
        if field.typ <: Core.LLVMPtr
            # in the case of buffers, the element type is encoded
            push!(md, MDString(humanize_typename(field.typ.parameters[1]); ctx))
        else
            push!(md, MDString(humanize_typename(field.typ); ctx))
        end
        # TODO: in the case of LLVM.ArrayType (coming from a tuple, or homogeneous struct)
        #       the element type should be reported as well

        # Field name
        push!(md, MDString(string(field.name); ctx))

        if indirect
            push!(md, MDString("air.indirect_argument"; ctx))
            # numbering of the fields of an indirect arguments begins at id 0 again
            # TODO: does that mean we should reuse code at a higher level?
            indirect_id = id - offset
            if field.codegen.typ isa LLVM.StructType
                push!(md, Metadata(ConstantInt(Int32(indirect_id); ctx)))
            else
                indirect_md, indirect_info, indirect_ids =
                    encode_argument(field, indirect_id; ctx)
                #@assert indirect_ids == field_ids
                # XXX: for non-ptr arguments, encode_argument always assumes a constant.
                #      that isn't true; it could also be an array, with multiple IDs.
                #      enable the assertion above to identify such cases.
                push!(md, MDNode(indirect_md; ctx))
            end
        end

        id += field_ids
        ids += field_ids
        push!(infos, field_info)
    end

    return md, infos, ids, indirect
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

    return
end
