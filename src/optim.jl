# LLVM IR optimization

function optimize!(@nospecialize(job::CompilerJob), mod::LLVM.Module; opt_level=2)
    tm = llvm_machine(job.config.target)

    global current_job
    current_job = job

    @dispose pb=NewPMPassBuilder() begin
        register!(pb, GPULowerCPUFeaturesPass())
        register!(pb, GPULowerPTLSPass())
        register!(pb, GPULowerGCFramePass())
        register!(pb, AddKernelStatePass())
        register!(pb, LowerKernelStatePass())
        register!(pb, CleanupKernelStatePass())

        add!(pb, NewPMModulePassManager()) do mpm
            buildNewPMPipeline!(mpm, job, opt_level)
        end
        run!(pb, mod, tm)
    end
    optimize_module!(job, mod)
    run!(DeadArgumentEliminationPass(), mod, tm)
    return
end

function buildNewPMPipeline!(mpm, @nospecialize(job::CompilerJob), opt_level)
    buildEarlySimplificationPipeline(mpm, job, opt_level)
    add!(mpm, AlwaysInlinerPass())
    buildEarlyOptimizerPipeline(mpm, job, opt_level)
    add!(mpm, NewPMFunctionPassManager()) do fpm
        buildLoopOptimizerPipeline(fpm, job, opt_level)
        buildScalarOptimizerPipeline(fpm, job, opt_level)
        if (can_vectorize(job)) && opt_level >= 2
            buildVectorPipeline(fpm, job, opt_level)
        end
        if isdebug(:optim)
            add!(fpm, WarnMissedTransformationsPass())
        end
    end
    buildIntrinsicLoweringPipeline(mpm, job, opt_level)
    buildCleanupPipeline(mpm, job, opt_level)
end

const BasicSimplifyCFGOptions =
    (; switch_range_to_icmp=true,
       switch_to_lookup=true,
       forward_switch_cond=true,
    )
const AggressiveSimplifyCFGOptions =
    (; switch_range_to_icmp=true,
       switch_to_lookup=true,
       forward_switch_cond=true,
       # These mess with loop rotation, so only do them after that
       hoist_common_insts=true,
       # Causes an SRET assertion error in late-gc-lowering
       #sink_common_insts=true
    )

function buildEarlySimplificationPipeline(mpm, @nospecialize(job::CompilerJob), opt_level)
    if should_verify()
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, GCInvariantVerifierPass())
        end
        add!(mpm, VerifierPass())
    end
    add!(mpm, ForceFunctionAttrsPass())
    # TODO invokePipelineStartCallbacks
    add!(mpm, Annotation2MetadataPass())
    add!(mpm, ConstantMergePass())
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, LowerExpectIntrinsicPass())
        if opt_level >= 2
            add!(fpm, PropagateJuliaAddrspacesPass())
        end
        add!(fpm, SimplifyCFGPass(; BasicSimplifyCFGOptions...))
        if opt_level >= 1
            add!(fpm, DCEPass())
            add!(fpm, SROAPass())
        end
    end
    # TODO invokeEarlySimplificationCallbacks
end

function buildEarlyOptimizerPipeline(mpm, @nospecialize(job::CompilerJob), opt_level)
    add!(mpm, NewPMCGSCCPassManager()) do cgpm
        # TODO invokeCGSCCCallbacks
        add!(cgpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, AllocOptPass())
            add!(fpm, Float2IntPass())
            add!(fpm, LowerConstantIntrinsicsPass())
        end
    end
    add!(mpm, GPULowerCPUFeaturesPass())
    if opt_level >= 1
        add!(mpm, NewPMFunctionPassManager()) do fpm
            if opt_level >= 2
                add!(fpm, SROAPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JumpThreadingPass())
                add!(fpm, CorrelatedValuePropagationPass())
                add!(fpm, ReassociatePass())
                add!(fpm, EarlyCSEPass())
                add!(fpm, AllocOptPass())
            else
                add!(fpm, InstCombinePass())
                add!(fpm, EarlyCSEPass())
            end
        end
        # TODO invokePeepholeCallbacks
    end
end

function buildLoopOptimizerPipeline(fpm, @nospecialize(job::CompilerJob), opt_level)
    add!(fpm, NewPMLoopPassManager()) do lpm
        add!(lpm, LowerSIMDLoopPass())
        if opt_level >= 2
            add!(lpm, LoopRotatePass())
        end
        # TODO invokeLateLoopOptimizationCallbacks
    end
    if opt_level >= 2
        add!(fpm, NewPMLoopPassManager(; use_memory_ssa=true)) do lpm
            add!(lpm, LICMPass())
            add!(lpm, JuliaLICMPass())
            add!(lpm, SimpleLoopUnswitchPass(nontrivial=true, trivial=true))
            add!(lpm, LICMPass())
            add!(lpm, JuliaLICMPass())
        end
    end
    if opt_level >= 2
        add!(fpm, IRCEPass())
    end
    add!(fpm, NewPMLoopPassManager()) do lpm
        if opt_level >= 2
            add!(lpm, LoopInstSimplifyPass())
            add!(lpm, LoopIdiomRecognizePass())
            add!(lpm, IndVarSimplifyPass())
            add!(lpm, LoopDeletionPass())
            add!(lpm, LoopFullUnrollPass())
        end
        # TODO invokeLoopOptimizerEndCallbacks
    end
end

function buildScalarOptimizerPipeline(fpm, @nospecialize(job::CompilerJob), opt_level)
    if opt_level >= 2
        add!(fpm, AllocOptPass())
        add!(fpm, SROAPass())
        add!(fpm, InstSimplifyPass())
        add!(fpm, GVNPass())
        add!(fpm, MemCpyOptPass())
        add!(fpm, SCCPPass())
        add!(fpm, CorrelatedValuePropagationPass())
        add!(fpm, DCEPass())
        add!(fpm, IRCEPass())
        add!(fpm, InstCombinePass())
        add!(fpm, JumpThreadingPass())
    end
    if opt_level >= 3
        add!(fpm, GVNPass())
    end
    if opt_level >= 2
        add!(fpm, DSEPass())
        # TODO invokePeepholeCallbacks
        add!(fpm, SimplifyCFGPass(; AggressiveSimplifyCFGOptions...))
        add!(fpm, AllocOptPass())
        add!(fpm, NewPMLoopPassManager()) do lpm
            add!(lpm, LoopDeletionPass())
            add!(lpm, LoopInstSimplifyPass())
        end
        add!(fpm, LoopDistributePass())
    end
    # TODO invokeScalarOptimizerCallbacks
end

function buildVectorPipeline(fpm, @nospecialize(job::CompilerJob), opt_level)
    add!(fpm, InjectTLIMappings())
    add!(fpm, LoopVectorizePass())
    add!(fpm, LoopLoadEliminationPass())
    add!(fpm, InstCombinePass())
    add!(fpm, SimplifyCFGPass(; AggressiveSimplifyCFGOptions...))
    add!(fpm, SLPVectorizerPass())
    add!(fpm, VectorCombinePass())
    # TODO invokeVectorizerCallbacks
    add!(fpm, ADCEPass())
    add!(fpm, LoopUnrollPass(; opt_level))
end

function buildIntrinsicLoweringPipeline(mpm, @nospecialize(job::CompilerJob), opt_level)
    add!(mpm, RemoveNIPass())

    # lower GC intrinsics
    if !uses_julia_runtime(job)
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, GPULowerGCFramePass())
        end
    end

    # lower kernel state intrinsics
    # NOTE: we can only do so here, as GC lowering can introduce calls to the runtime,
    #       and thus additional uses of the kernel state intrinsics.
    if job.config.kernel
        # TODO: now that all kernel state-related passes are being run here, merge some?
        add!(mpm, AddKernelStatePass())
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, LowerKernelStatePass())
        end
        add!(mpm, CleanupKernelStatePass())
    end

    if !uses_julia_runtime(job)
        # remove dead uses of ptls
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, ADCEPass())
        end
        add!(mpm, GPULowerPTLSPass())
    end

    add!(mpm, NewPMFunctionPassManager()) do fpm
        # lower exception handling
        if uses_julia_runtime(job) && VERSION < v"1.13.0-DEV.36"
            add!(fpm, LowerExcHandlersPass())
        end
        add!(fpm, GCInvariantVerifierPass())
        add!(fpm, LateLowerGCPass())
        if uses_julia_runtime(job) && VERSION >= v"1.11.0-DEV.208"
            add!(fpm, FinalLowerGCPass())
        end
    end
    if uses_julia_runtime(job) && VERSION < v"1.11.0-DEV.208"
        add!(mpm, FinalLowerGCPass())
    end

    if opt_level >= 2
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, GVNPass())
            add!(fpm, SCCPPass())
            add!(fpm, DCEPass())
        end
    end

    # lower PTLS intrinsics
    if uses_julia_runtime(job)
        add!(mpm, LowerPTLSPass())
    end

    if opt_level >= 1
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, InstCombinePass())
            add!(fpm, SimplifyCFGPass(; AggressiveSimplifyCFGOptions...))
        end
    end

    # remove Julia address spaces
    add!(mpm, RemoveJuliaAddrspacesPass())

    # Julia's operand bundles confuse the inliner, so repeat here now they are gone.
    # FIXME: we should fix the inliner so that inlined code gets optimized early-on
    add!(mpm, AlwaysInlinerPass())
end

function buildCleanupPipeline(mpm, @nospecialize(job::CompilerJob), opt_level)
    if opt_level >= 2
        add!(mpm, NewPMFunctionPassManager()) do fpm
            if VERSION < v"1.12.0-DEV.1390"
                add!(fpm, CombineMulAddPass())
            end
            add!(fpm, DivRemPairsPass())
        end
    end
    # TODO invokeOptimizerLastCallbacks
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, AnnotationRemarksPass())
    end
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, DemoteFloat16Pass())
        if opt_level >= 1
            add!(fpm, GVNPass())
        end
    end
end



## custom passes

# lowering intrinsics
function cpu_features!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false

    argtyps = Dict(
        "f32" => Float32,
        "f64" => Float64,
    )

    # have_fma
    for f in functions(mod)
        ft = function_type(f)
        fn = LLVM.name(f)
        startswith(fn, "julia.cpu.have_fma.") || continue
        typnam = fn[20:end]

        # determine whether this back-end supports FMA on this type
        has_fma = if haskey(argtyps, typnam)
            typ = argtyps[typnam]
            have_fma(job.config.target, typ)
        else
            # warn?
            false
        end
        has_fma = ConstantInt(return_type(ft), has_fma)

        # substitute all uses of the intrinsic with a constant
        materialized = LLVM.Value[]
        for use in uses(f)
            val = user(use)
            replace_uses!(val, has_fma)
            push!(materialized, val)
        end

        # remove the intrinsic and its uses
        for val in materialized
            @assert isempty(uses(val))
            erase!(val)
        end
        @assert isempty(uses(f))
        erase!(f)
    end

    return changed
end
GPULowerCPUFeaturesPass() = NewPMModulePass("GPULowerCPUFeatures", cpu_features!)

# lower object allocations to to PTX malloc
#
# this is a PoC implementation that is very simple: allocate, and never free. it also runs
# _before_ Julia's GC lowering passes, so we don't get to use the results of its analyses.
# when we ever implement a more potent GC, we will need those results, but the relevant pass
# is currently very architecture/CPU specific: hard-coded pool sizes, TLS references, etc.
# such IR is hard to clean-up, so we probably will need to have the GC lowering pass emit
# lower-level intrinsics which then can be lowered to architecture-specific code.
function lower_gc_frame!(fun::LLVM.Function)
    job = current_job::CompilerJob
    mod = LLVM.parent(fun)
    changed = false

    # plain alloc
    if haskey(functions(mod), "julia.gc_alloc_obj")
        alloc_obj = functions(mod)["julia.gc_alloc_obj"]
        alloc_obj_ft = function_type(alloc_obj)
        T_prjlvalue = return_type(alloc_obj_ft)
        T_pjlvalue = convert(LLVMType, Any; allow_boxed=true)

        for use in uses(alloc_obj)
            call = user(use)::LLVM.CallInst

            # decode the call
            ops = arguments(call)
            sz = ops[2]

            # replace with PTX alloc_obj
            @dispose builder=IRBuilder() begin
                position!(builder, call)
                ptr = call!(builder, Runtime.get(:gc_pool_alloc), [sz])
                replace_uses!(call, ptr)
            end

            erase!(call)

            changed = true
        end

        @compiler_assert isempty(uses(alloc_obj)) job
    end

    # we don't care about write barriers
    if haskey(functions(mod), "julia.write_barrier")
        barrier = functions(mod)["julia.write_barrier"]

        for use in uses(barrier)
            call = user(use)::LLVM.CallInst
            erase!(call)
            changed = true
        end

        @compiler_assert isempty(uses(barrier)) job
    end

    return changed
end
GPULowerGCFramePass() = NewPMFunctionPass("GPULowerGCFrame", lower_gc_frame!)

# lower the `julia.ptls_states` intrinsic by removing it, since it is GPU incompatible.
#
# this assumes and checks that the TLS is unused, which should be the case for most GPU code
# after lowering the GC intrinsics to TLS-less code and having run DCE.
#
# TODO: maybe don't have Julia emit actual uses of the TLS, but use intrinsics instead,
#       making it easier to remove or reimplement that functionality here.
function lower_ptls!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false

    intrinsic = "julia.get_pgcstack"

    if haskey(functions(mod), intrinsic)
        ptls_getter = functions(mod)[intrinsic]

        for use in uses(ptls_getter)
            val = user(use)
            if isempty(uses(val))
                erase!(val)
                changed = true
            else
                # the validator will detect this
            end
        end
     end

    return changed
end
GPULowerPTLSPass() = NewPMModulePass("GPULowerPTLS", lower_ptls!)
