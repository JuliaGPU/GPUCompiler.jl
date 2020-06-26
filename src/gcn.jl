# implementation of the GPUCompiler interfaces for generating GCN code

## target

export GCNCompilerTarget

Base.@kwdef struct GCNCompilerTarget <: AbstractCompilerTarget
    dev_isa::String
end

llvm_triple(::GCNCompilerTarget) = "amdgcn-amd-amdhsa"

function llvm_machine(target::GCNCompilerTarget)
    triple = llvm_triple(target)
    t = Target(triple=triple)

    cpu = target.dev_isa
    feat = ""
    optlevel = LLVM.API.LLVMCodeGenLevelDefault
    reloc = LLVM.API.LLVMRelocPIC
    tm = TargetMachine(t, triple, cpu, feat, optlevel, reloc)
    asm_verbosity!(tm, true)

    return tm
end


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{GCNCompilerTarget}) = "gcn-$(job.target.dev_isa)"

const gcn_intrinsics = () # TODO: ("vprintf", "__assertfail", "malloc", "free")
isintrinsic(::CompilerJob{GCNCompilerTarget}, fn::String) = in(fn, gcn_intrinsics)

function process_kernel!(job::CompilerJob{GCNCompilerTarget}, mod::LLVM.Module, kernel::LLVM.Function)
    kernel = wrap_entry!(job, mod, kernel)
    # AMDGPU kernel calling convention
    callconv!(kernel, LLVM.API.LLVMCallConv(91))
    kernel
end

function add_lowering_passes!(job::CompilerJob{GCNCompilerTarget}, pm::LLVM.PassManager)
    add!(pm, ModulePass("LowerThrowExtra", lower_throw_extra!))
end

function lower_throw_extra!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "lower throw (extra)" begin

    throw_functions = [
        r"julia_bounds_error.*",
        r"julia_throw_boundserror.*",
        r"julia_error_if_canonical_getindex.*",
        r"julia_error_if_canonical_setindex.*",
        r"julia___subarray_throw_boundserror.*",
    ]


    for f in functions(mod)
        f_name = LLVM.name(f)
        for fn in throw_functions
            if occursin(fn, f_name)
                for use in uses(f)
                    call = user(use)::LLVM.CallInst

                    # replace the throw with a trap
                    let builder = Builder(JuliaContext())
                        position!(builder, call)
                        emit_exception!(builder, f_name, call)
                        dispose(builder)
                    end

                    # remove the call
                    call_args = collect(operands(call))[1:end-1] # last arg is function itself
                    unsafe_delete!(LLVM.parent(call), call)

                    # HACK: kill the exceptions' unused arguments
                    for arg in call_args
                        # peek through casts
                        if isa(arg, LLVM.AddrSpaceCastInst)
                            cast = arg
                            arg = first(operands(cast))
                            isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                        end

                        if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                            unsafe_delete!(LLVM.parent(arg), arg)
                        end
                    end

                    changed = true
                end

                @compiler_assert isempty(uses(f)) job
            end
        end
    end

    end
    return changed
end

function emit_trap!(job::CompilerJob{GCNCompilerTarget}, builder, mod, inst)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(JuliaContext())))
    end
    if Base.libllvm_version < v"9"
        rl_ft = LLVM.FunctionType(LLVM.Int32Type(JuliaContext()),
                                  [LLVM.Int32Type(JuliaContext())])
        rl = if haskey(functions(mod), "llvm.amdgcn.readfirstlane")
            functions(mod)["llvm.amdgcn.readfirstlane"]
        else
            LLVM.Function(mod, "llvm.amdgcn.readfirstlane", rl_ft)
        end
        # FIXME: Early versions of the AMDGPU target fail to skip machine
        # blocks with certain side effects when EXEC==0, except when certain
        # criteria are met within said block. We emit a v_readfirstlane_b32
        # instruction here, as that is sufficient to trigger a skip. Without
        # this, the target will only attempt to do a "masked branch", which
        # only works on vector instructions (trap is a scalar instruction, and
        # therefore it is executed even when EXEC==0).
        rl_val = call!(builder, rl, [ConstantInt(Int32(32), JuliaContext())])
        rl_bc = inttoptr!(builder, rl_val, LLVM.PointerType(LLVM.Int32Type(JuliaContext())))
        store!(builder, rl_val, rl_bc)
    end
    call!(builder, trap)
end

# manual implementation of byval, as the backend doesn't support it for kernel args
# https://reviews.llvm.org/D79744
function wrapper_type(julia_t::Type, codegen_t::LLVMType)::LLVMType
    if !isbitstype(julia_t)
        # don't pass jl_value_t by value; it's an opaque structure
        return codegen_t
    elseif isa(codegen_t, LLVM.PointerType) && !(julia_t <: Ptr)
        # we didn't specify a pointer, but codegen passes one anyway.
        # make the wrapper accept the underlying value instead.
        return eltype(codegen_t)
    else
        return codegen_t
    end
end
# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(job::CompilerJob, mod::LLVM.Module, entry_f::LLVM.Function)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType
    @compiler_assert return_type(entry_ft) == LLVM.VoidType(JuliaContext()) job

    # filter out types which don't occur in the LLVM function signatures
    sig = Base.signature_type(job.source.f, job.source.tt)::Type
    julia_types = Type[]
    for dt::Type in sig.parameters
        if !isghosttype(dt) && (VERSION < v"1.5.0-DEV.581" || !Core.Compiler.isconstType(dt))
            push!(julia_types, dt)
        end
    end

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[wrapper_type(julia_t, codegen_t)
                                  for (julia_t, codegen_t)
                                  in zip(julia_types, parameters(entry_ft))]
    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, wrapper_fn * ".inner")
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    let builder = Builder(JuliaContext())
        entry = BasicBlock(wrapper_f, "entry", JuliaContext())
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        # perform argument conversions
        codegen_types = parameters(entry_ft)
        wrapper_params = parameters(wrapper_f)
        param_index = 0
        for (julia_t, codegen_t, wrapper_t, wrapper_param) in
            zip(julia_types, codegen_types, wrapper_types, wrapper_params)
            param_index += 1
            if codegen_t != wrapper_t
                # the wrapper argument doesn't match the kernel parameter type.
                # this only happens when codegen wants to pass a pointer.
                @compiler_assert isa(codegen_t, LLVM.PointerType) job
                @compiler_assert eltype(codegen_t) == wrapper_t job

                # copy the argument value to a stack slot, and reference it.
                ptr = alloca!(builder, wrapper_t)
                if LLVM.addrspace(codegen_t) != 0
                    ptr = addrspacecast!(builder, ptr, codegen_t)
                end
                store!(builder, wrapper_param, ptr)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, wrapper_param)
                for attr in collect(parameter_attributes(entry_f, param_index))
                    push!(parameter_attributes(wrapper_f, param_index), attr)
                end
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)

        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, JuliaContext()))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    fixup_metadata!(entry_f)
    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end
# HACK: get rid of invariant.load and const TBAA metadata on loads from pointer args,
#       since storing to a stack slot violates the semantics of those attributes.
# TODO: can we emit a wrapper that doesn't violate Julia's metadata?
function fixup_metadata!(f::LLVM.Function)
    for param in parameters(f)
        if isa(llvmtype(param), LLVM.PointerType)
            # collect all uses of the pointer
            worklist = Vector{LLVM.Instruction}(user.(collect(uses(param))))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = metadata(value)
                if haskey(md, LLVM.MD_invariant_load)
                    delete!(md, LLVM.MD_invariant_load)
                end
                if haskey(md, LLVM.MD_tbaa)
                    delete!(md, LLVM.MD_tbaa)
                end

                # recurse on the output of some instructions
                if isa(value, LLVM.BitCastInst) ||
                   isa(value, LLVM.GetElementPtrInst) ||
                   isa(value, LLVM.AddrSpaceCastInst)
                    append!(worklist, user.(collect(uses(value))))
                end

                # IMPORTANT NOTE: if we ever want to inline functions at the LLVM level,
                # we need to recurse into call instructions here, and strip metadata from
                # called functions (see CUDAnative.jl#238).
            end
        end
    end
end
