# hand-off of optimized code to a newer, standalone LLVM library
#
# With `handoff=:late`, optimization runs as usual with Julia's LLVM, after which the module
# is transferred as bitcode to a standalone LLVM library with GPU back-ends (libLLVMGPU),
# where it is optimized again. Target finalization (`finish_ir!`), validation, relocation
# collection, and code generation then all happen with that library, which takes the place
# of the per-target back-end libraries (e.g. NVPTX_LLVM_Backend_jll).

export supports_handoff


## mode

"""
    supports_handoff(target::AbstractCompilerTarget) -> Bool

Whether jobs for `target` support the late hand-off to a standalone LLVM library
(`CompilerConfig(...; handoff=:late)`).
"""
supports_handoff(@nospecialize(target::AbstractCompilerTarget)) = false

# the default for `CompilerConfig(...; handoff)`, from the `handoff` preference
const handoff_preference = Ref{Union{Nothing,Symbol}}(nothing)
function default_handoff()
    mode = handoff_preference[]
    if mode === nothing
        mode = Symbol(load_preference(@__MODULE__, "handoff", "none"))
        mode in (:none, :late) ||
            error("Unknown value $(repr(String(mode))) for GPUCompiler's `handoff` preference; expected \"none\" or \"late\"")
        handoff_preference[] = mode
    end
    return mode
end

# the hand-off mode of a job: one that is inherited from the preference only applies to
# targets that support it, and nested jobs are linked into a parent before its hand-off
function resolve_handoff(@nospecialize(target::AbstractCompilerTarget),
                         handoff::Union{Nothing,Symbol}, toplevel::Bool)
    if handoff === nothing
        handoff = supports_handoff(target) ? default_handoff() : :none
    elseif handoff ∉ (:none, :late)
        error("Unknown handoff=$(repr(handoff)); expected :none or :late")
    end
    toplevel || return :none
    if handoff === :late && !supports_handoff(target)
        error("$(nameof(typeof(target))) does not support the late hand-off (handoff=:late)")
    end
    return handoff
end


## library

const LLVMGPU_jll = LazyModule("LLVMGPU_jll", UUID("a5c41ef1-0fc2-5e62-838b-0d6f5f7d5115"))

# the state of a library that code is handed off to
struct HandoffLibrary
    library::LLVM.Library

    # protects the library's command-line options, along with the creation of the target
    # machines that read them, and code generation (which the back-end libraries serialized)
    lock::ReentrantLock

    # the SPIR-V extensions that are currently enabled through `--spirv-ext`
    spirv_extensions::Set{String}
end

const handoff_libraries = Dict{LLVM.Library,HandoffLibrary}()
const handoff_libraries_lock = ReentrantLock()

# the library to hand off to: whichever is registered with LLVM.jl as `:gpu`, or else the
# one LLVMGPU_jll provides (which isn't loaded here, only used once it is)
function handoff_library()
    lib = try
        LLVM.Library(:gpu)
    catch err
        err isa ErrorException || rethrow()
        isavailable(LLVMGPU_jll) ||
            error("""The late hand-off needs a standalone LLVM library with GPU back-ends:
                     load LLVMGPU_jll, or register a library as `:gpu` using `LLVM.register_library!`.""")
        LLVM.register_library!(:gpu, LLVMGPU_jll.libLLVMGPU)
    end
    @lock handoff_libraries_lock begin
        get!(handoff_libraries, lib) do
            initialize_handoff_library(lib)
        end
    end
end

function initialize_handoff_library(lib::LLVM.Library)
    # the target and extension tables below are those of LLVM 23
    LLVM.version(lib).major == 23 ||
        error("The late hand-off supports LLVM 23 libraries, not LLVM $(LLVM.version(lib)) ($(lib.path))")

    # LLVM.jl does not know every back-end, e.g. SPIR-V, so look up the initializers
    handle = Libdl.dlopen(lib.path)
    for backend in ("NVPTX", "AMDGPU", "SPIRV"),
        component in ("TargetInfo", "Target", "TargetMC", "AsmPrinter", "AsmParser")
        f = Libdl.dlsym(handle, "LLVMInitialize$(backend)$(component)"; throw_error=false)
        f === nothing || ccall(f, Cvoid, ())
    end

    LLVM.library!(lib) do
        # like NVPTX_LLVM_Backend_jll did, leave eligible multiply-add pairs contractible
        # for ptxas, like nvcc does
        LLVM.clopts("-nvptx-fma-level=1")
    end

    return HandoffLibrary(lib, ReentrantLock(), Set{String}())
end

# whether `lib` is a library that code is handed off to
is_handoff_library(lib::LLVM.Library=LLVM.library()) =
    @lock handoff_libraries_lock haskey(handoff_libraries, lib)

# the state of the library that the current library is handed off to
function current_handoff_library()
    lib = LLVM.library()
    @lock handoff_libraries_lock begin
        state = get(handoff_libraries, lib, nothing)
        state === nothing && error("LLVM library :$(lib.name) is not used for the late hand-off")
        return state
    end
end


## contexts

# The IR that `compile` returns lives in the context of the enclosing `JuliaContext` block,
# so after a hand-off, that block owns the context of the other library too. Contexts are
# not thread safe, so it belongs to the task that runs the block.
mutable struct ContextOwner
    const task::Task
    handoff_context::Union{Nothing,LLVM.Context}
end

const context_owner = ScopedValue{Union{Nothing,ContextOwner}}(nothing)

function handoff_context(lib::LLVM.Library)
    owner = context_owner[]
    if owner === nothing
        error("The late hand-off (handoff=:late) needs to be used inside a `JuliaContext() do ... end` block, which owns the resulting IR")
    elseif owner.task !== current_task()
        error("The late hand-off can only be used by the task that runs the enclosing `JuliaContext` block; use a block in every task instead")
    end
    ctx = owner.handoff_context
    if ctx === nothing
        ctx = LLVM.Context(; library=lib)
        # (only activated when used, by `context!`)
        deactivate(ctx)
        owner.handoff_context = ctx
    elseif ctx.library !== lib
        error("The late hand-off library changed from :$(ctx.library.name) to :$(lib.name) within one `JuliaContext` block")
    end
    return ctx
end

function dispose_handoff_context(owner::ContextOwner)
    ctx = owner.handoff_context
    ctx === nothing && return
    owner.handoff_context = nothing
    activate(ctx)
    dispose(ctx)
    return
end


## transfer

# Transfer the optimized module `ir` to the hand-off library, returning the new module
# (in a context owned by the enclosing `JuliaContext` block). The original is disposed of.
function late_handoff(@nospecialize(job::CompilerJob), ir::LLVM.Module)
    state = handoff_library()
    ctx = handoff_context(state.library)

    # bitcode is forward compatible, so the newer library can read it
    input = bitcode(ir)
    mod = context!(() -> transfer(job, input), ctx)
    dispose(ir)
    return mod
end

function transfer(@nospecialize(job::CompilerJob), input::Vector{UInt8})
    mod = parse(LLVM.Module, input)
    try
        with_llvm_machine(job.config.target) do tm
            prepare_handoff!(mod, tm)
        end
        if job.config.optimize
            @tracepoint "reoptimization" reoptimize!(job, mod)
        end
    catch
        dispose(mod)
        rethrow()
    end
    return mod
end

function prepare_handoff!(mod::LLVM.Module, tm::Union{Nothing,LLVM.TargetMachine})
    tm === nothing && return

    # what the module was optimized for so far should match what code is generated for:
    # the target machine replaces the module's layout when generating code anyway, but
    # the optimizer of the new library should already use it
    LLVM.DataLayout(tm) do dl
        mod.datalayout = dl
    end
    return
end

# optimize again with the standard pipeline of the new library, to benefit from its
# improvements. it cannot express the restrictions that some targets put on GPUCompiler's
# own pipeline (see `optimization_options`), so those jobs are not optimized again.
function reoptimize!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    all(==(true), values(optimization_options(job))) || return
    vectorize = can_vectorize(job)
    with_llvm_machine(job.config.target) do tm
        @dispose pb=PassBuilder(; loop_vectorization=vectorize,
                                  slp_vectorization=vectorize) begin
            add!(pb, "default<O$(job.config.opt_level)>")
            run!(pb, mod, tm)
        end
    end
    return
end


## code generation

# generate code for `mod` from the late hand-off, with the hand-off library's back-end
function handoff_mcgen(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                       format::LLVM.CodeGenFileType.T)
    # a module that does not verify is a fatal error in code generation
    verify(mod)

    state = current_handoff_library()
    @lock state.lock begin
        with_llvm_machine(job.config.target) do tm
            tm === nothing && error("No target machine for $(job.config.target)")
            set_subtarget_attributes!(mod, tm)
            LLVM.emit(tm, mod, format)
        end
    end
end

# like `llc` and the back-end libraries did, make the functions use the target machine's
# processor and features, unless they specify their own. only when generating code: the
# attributes keep functions without them from being inlined.
function set_subtarget_attributes!(mod::LLVM.Module, tm::LLVM.TargetMachine)
    cpu = tm.cpu
    features = tm.features
    for f in mod.functions
        isdeclaration(f) && continue
        attrs = f.function_attributes
        if !isempty(cpu) && !haskey(attrs, "target-cpu")
            push!(attrs, StringAttribute("target-cpu", cpu))
        end
        if !isempty(features) && !haskey(attrs, "target-features")
            push!(attrs, StringAttribute("target-features", features))
        end
    end
    return
end

# Set the SPIR-V extensions that target machines created afterwards enable to exactly
# `extensions`, a specifier like "+SPV_KHR_bfloat16,-SPV_KHR_float_controls". The SPIR-V
# back-end reads them from its `--spirv-ext` option when a target machine is created. Every
# use of the option replaces the set of extensions, but the ones it disables (with `-NAME`)
# stay disabled for good, so only ever enable the exact set. Must be called with
# `state.lock` held, through the creation of the target machine.
function set_spirv_extensions!(state::HandoffLibrary, extensions::AbstractString)
    wanted = Set{String}()
    for entry in split(extensions, ','; keepempty=false)
        entry = strip(entry)
        flag, name = entry[1], entry[2:end]
        flag in ('+', '-') ||
            error("Invalid SPIR-V extension $(repr(entry)) in $(repr(extensions)); expected +NAME or -NAME")
        names = if name == "all"
            spirv_extension_names
        elseif name in spirv_extension_names
            (name,)
        else
            error("Unknown SPIR-V extension $(repr(name)) in $(repr(extensions))")
        end
        flag == '+' ? union!(wanted, names) : setdiff!(wanted, names)
    end

    if wanted != state.spirv_extensions
        spec = join(("+" * name for name in sort!(collect(wanted))), ',')
        LLVM.library!(state.library) do
            LLVM.clopts("--spirv-ext=" * spec)
        end
        empty!(state.spirv_extensions)
        union!(state.spirv_extensions, wanted)
    end
    return
end

# the extensions that the SPIR-V back-end of LLVM 23.1 knows (`SPIRVExtensionMap` in
# SPIRVCommandLine.cpp). it ignores unknown ones, so they are rejected here.
const spirv_extension_names = (
    "SPV_ALTERA_arbitrary_precision_fixed_point", "SPV_ALTERA_arbitrary_precision_floating_point",
    "SPV_ALTERA_arbitrary_precision_integers", "SPV_ALTERA_blocking_pipes",
    "SPV_AMD_weak_linkage", "SPV_EXT_arithmetic_fence", "SPV_EXT_demote_to_helper_invocation",
    "SPV_EXT_descriptor_indexing", "SPV_EXT_fragment_fully_covered",
    "SPV_EXT_fragment_invocation_density", "SPV_EXT_fragment_shader_interlock",
    "SPV_EXT_image_raw10_raw12", "SPV_EXT_mesh_shader", "SPV_EXT_optnone",
    "SPV_EXT_relaxed_printf_string_address_space", "SPV_EXT_shader_atomic_float16_add",
    "SPV_EXT_shader_atomic_float_add", "SPV_EXT_shader_atomic_float_min_max",
    "SPV_EXT_shader_image_int64", "SPV_EXT_shader_stencil_export",
    "SPV_EXT_shader_viewport_index_layer", "SPV_GOOGLE_hlsl_functionality1",
    "SPV_GOOGLE_user_type", "SPV_INTEL_16bit_atomics", "SPV_INTEL_2d_block_io",
    "SPV_INTEL_bfloat16_arithmetic", "SPV_INTEL_bfloat16_conversion",
    "SPV_INTEL_bindless_images", "SPV_INTEL_cache_controls", "SPV_INTEL_float_controls2",
    "SPV_INTEL_fp_max_error", "SPV_INTEL_function_pointers",
    "SPV_INTEL_global_variable_fpga_decorations", "SPV_INTEL_global_variable_host_access",
    "SPV_INTEL_inline_assembly", "SPV_INTEL_int4", "SPV_INTEL_joint_matrix",
    "SPV_INTEL_kernel_attributes", "SPV_INTEL_long_composites",
    "SPV_INTEL_masked_gather_scatter", "SPV_INTEL_media_block_io",
    "SPV_INTEL_memory_access_aliasing", "SPV_INTEL_optnone", "SPV_INTEL_predicated_io",
    "SPV_INTEL_split_barrier", "SPV_INTEL_subgroup_matrix_multiply_accumulate",
    "SPV_INTEL_subgroups", "SPV_INTEL_tensor_float32_conversion",
    "SPV_INTEL_ternary_bitwise_function", "SPV_INTEL_unstructured_loop_controls",
    "SPV_INTEL_usm_storage_classes", "SPV_INTEL_variable_length_array",
    "SPV_KHR_16bit_storage", "SPV_KHR_abort", "SPV_KHR_bfloat16", "SPV_KHR_bit_instructions",
    "SPV_KHR_cooperative_matrix", "SPV_KHR_device_group", "SPV_KHR_expect_assume",
    "SPV_KHR_float_controls", "SPV_KHR_float_controls2", "SPV_KHR_fma",
    "SPV_KHR_fragment_shader_barycentric", "SPV_KHR_fragment_shading_rate",
    "SPV_KHR_integer_dot_product", "SPV_KHR_linkonce_odr", "SPV_KHR_maximal_reconvergence",
    "SPV_KHR_multiview", "SPV_KHR_no_integer_wrap_decoration", "SPV_KHR_non_semantic_info",
    "SPV_KHR_physical_storage_buffer", "SPV_KHR_poison_freeze", "SPV_KHR_post_depth_coverage",
    "SPV_KHR_ray_query", "SPV_KHR_ray_tracing", "SPV_KHR_shader_clock",
    "SPV_KHR_shader_draw_parameters", "SPV_KHR_subgroup_rotate",
    "SPV_KHR_uniform_group_instructions", "SPV_KHR_vulkan_memory_model",
    "SPV_NV_shader_atomic_fp16_vector", "SPV_NV_shader_subgroup_partitioned")


## Julia types

# Lower a Julia type to LLVM in the current context, also when that context belongs to
# another library than Julia's, whose code generator lowers types. The type is lowered in a
# Julia context, and transferred through a bitcode module holding a global of that type.
function julia_llvm_type(@nospecialize(T::Type))
    ctx = context()
    ctx.library === LLVM.Library(:julia) && return convert(LLVMType, T)
    input = LLVM.Context(; library=:julia) do _
        @dispose mod=LLVM.Module("type") begin
            GlobalVariable(mod, convert(LLVMType, T), "value")
            bitcode(mod)
        end
    end
    @dispose mod=parse(LLVM.Module, input) begin
        mod.globals["value"].global_value_type
    end
end
