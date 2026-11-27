# the LLVM toolchain that optimizes code and generates machine code
#
# With the default `julia` toolchain, the whole pipeline uses Julia's LLVM, and external
# back-end libraries (e.g. NVPTX_LLVM_Backend_jll) generate code. With the `llvmgpu`
# toolchain, once linked and post-processed, the module of a top-level job moves to a newer,
# standalone LLVM library with GPU back-ends (libLLVMGPU, from LLVMGPU_jll): optimization,
# target finalization (`finish_ir!`), validation and relocation collection happen with that
# library, and so does code generation, except with libNVVM or the Khronos SPIR-V
# translator, which get the optimized bitcode. That is for GPUCompiler's GPU targets; others,
# like the host, stay with Julia's LLVM (see `llvm_library`). Julia's code generator always
# uses Julia's LLVM.
#
# Select it with `Preferences.set_preferences!(GPUCompiler, "toolchain" => "llvmgpu")`.

import JuliaIR
import LLVMGPU_jll

const TOOLCHAIN = Symbol(@load_preference("toolchain", "julia"))
TOOLCHAIN in (:julia, :llvmgpu) ||
    error("Invalid toolchain preference $(repr(String(TOOLCHAIN))); expected \"julia\" or \"llvmgpu\"")


## libLLVMGPU

struct LLVMGPUState
    library::LLVM.Library

    # protects the library's command-line options, along with the creation of the target
    # machines that read them, and code generation (which the back-end libraries serialized)
    lock::ReentrantLock

    # the minimum PTX ISA version of each processor (see `ptx_minimum_version`)
    ptx_minimum_versions::Dict{String,VersionNumber}
end

# (process-local: reset by `__init__`)
const llvmgpu_state_ref = Ref{Union{Nothing,LLVMGPUState}}(nothing)
const llvmgpu_init_lock = ReentrantLock()

# libLLVMGPU, registered with LLVM.jl and initialized on first use
function llvmgpu_state()
    @lock llvmgpu_init_lock begin
        state = llvmgpu_state_ref[]
        if state === nothing
            lib = LLVM.register_library!(:gpu, LLVMGPU_jll.libLLVMGPU)
            state = llvmgpu_state_ref[] = initialize_llvmgpu(lib)
        end
        return state
    end
end

function initialize_llvmgpu(lib::LLVM.Library)
    # (the command-line options and back-end behavior relied on are those of LLVM 23)
    LLVM.version(lib).major == 23 ||
        error("The llvmgpu toolchain supports LLVM 23 libraries, not LLVM $(LLVM.version(lib)) ($(lib.path))")

    # LLVM.jl does not know every back-end, e.g. SPIR-V, so look up the initializers
    handle = Libdl.dlopen(lib.path)
    for backend in ("NVPTX", "AMDGPU", "SPIRV", "BPF"),
        component in ("TargetInfo", "Target", "TargetMC", "AsmPrinter", "AsmParser")
        f = Libdl.dlsym(handle, "LLVMInitialize$(backend)$(component)"; throw_error=false)
        f === nothing || ccall(f, Cvoid, ())
    end

    LLVM.library!(lib) do
        # like NVPTX_LLVM_Backend_jll did, leave eligible multiply-add pairs contractible
        # for ptxas, like nvcc does
        LLVM.clopts("-nvptx-fma-level=1")
    end

    return LLVMGPUState(lib, ReentrantLock(), Dict{String,VersionNumber}())
end

# whether `lib` is libLLVMGPU
function is_llvmgpu(lib::LLVM.Library=LLVM.library())
    TOOLCHAIN === :llvmgpu || return false
    state = @lock llvmgpu_init_lock llvmgpu_state_ref[]
    return state !== nothing && state.library === lib
end

# whether `job` moves to libLLVMGPU. nested jobs are linked into the module of their parent
# before it does.
uses_llvmgpu(@nospecialize(job::CompilerJob)) =
    job.config.toplevel && is_llvmgpu(checked_llvm_library(job.config.target))


## contexts

# The IR that `compile` returns lives in a context of the enclosing `JuliaContext` block,
# which, with the `llvmgpu` toolchain, also has a context of libLLVMGPU for the modules that
# move there (see `JuliaContext`). Like with Julia's LLVM, that is the context of the
# library that was activated last by the task, so a context created inside the block owns
# the modules compiled while it is active.
function llvmgpu_context(lib::LLVM.Library)
    ctx = LLVM.library!(() -> context(; throw_error=false), lib)
    ctx === nothing &&
        error("No active context of libLLVMGPU; compile inside a `JuliaContext() do ... end` block, which owns the resulting IR")
    return ctx
end


## transfer

# Move the linked module `ir` to libLLVMGPU, returning the new module (in a context owned by
# the enclosing `JuliaContext` block). The original is disposed of.
function move_to_llvmgpu!(ir::LLVM.Module)
    state = llvmgpu_state()
    ctx = llvmgpu_context(state.library)

    # bitcode is forward compatible, so the newer library can read it. the module keeps
    # its datalayout, like with Julia's LLVM: Julia's non-integral address spaces keep the
    # optimizer from turning GC-tracked pointers into integers until `RemoveNIPass`, and
    # the layout of types like `Int128` must match the host's.
    input = bitcode(ir)
    mod = context!(() -> parse(LLVM.Module, input), ctx)
    dispose(ir)
    return mod
end

# run `f` in the context of `ir`, which can belong to libLLVMGPU
in_context(f, ir::LLVM.Module) = TOOLCHAIN === :llvmgpu ? context!(f, context(ir)) : f()


## code generation

# generate code for `mod`, a module of libLLVMGPU, with libLLVMGPU's back-end for the target
function llvmgpu_emit(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                      format::LLVM.CodeGenFileType.T)
    # a module that does not verify is a fatal error in code generation
    verify(mod)

    state = llvmgpu_state()
    @lock state.lock begin
        with_llvm_machine(job.config.target) do tm
            tm === nothing && error("No target machine for $(job.config.target)")
            # code generation mutates the module, which is also returned as `meta.ir`
            @dispose mod=copy(mod) begin
                set_subtarget_attributes!(mod, tm)
                LLVM.emit(tm, mod, format)
            end
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
# `extensions`, a specifier like "+SPV_KHR_bfloat16,SPV_KHR_expect_assume" (where, as with
# SPIRV_LLVM_Backend_jll, an entry without a sign enables the extension). The SPIR-V back-end
# reads them from its `--spirv-ext` option when a target machine is created. Every use of the
# option replaces the set of enabled extensions, but the ones it disables (with `-NAME`) stay
# disabled for the rest of the process, so those are rejected. Must be called with
# `state.lock` held, through the creation of the target machine.
function set_spirv_extensions!(state::LLVMGPUState, extensions::AbstractString)
    entries = String[]
    for entry in split(extensions, ','; keepempty=false)
        entry = strip(entry)
        isempty(entry) && continue
        if startswith(entry, '-')
            error("Cannot disable SPIR-V extensions ($entry in $(repr(extensions))) with the llvmgpu toolchain")
        end
        name = lstrip(entry, '+')
        push!(entries, name == "all" ? "all" : "+" * name)
    end
    LLVM.library!(state.library) do
        try
            LLVM.clopts("--spirv-ext=" * join(entries, ','))
        catch err
            err isa LLVM.LLVMException || rethrow()
            error("Invalid SPIR-V extensions $(repr(extensions)): $(err.info)")
        end
    end
    return
end
