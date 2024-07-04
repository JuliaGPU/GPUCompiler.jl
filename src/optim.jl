# LLVM IR optimization

function optimize!(@nospecialize(job::CompilerJob), mod::LLVM.Module; opt_level=2)
    if use_newpm
        optimize_newpm!(job, mod; opt_level)
    else
        optimize_legacypm!(job, mod; opt_level)
    end
    return
end


## new pm

function optimize_newpm!(@nospecialize(job::CompilerJob), mod::LLVM.Module; opt_level)
    tm = llvm_machine(job.config.target)

    global current_job
    current_job = job

    @dispose pb=NewPMPassBuilder() begin
        register!(pb, CPUFeaturesPass())
        register!(pb, LowerPTLSPass())
        register!(pb, LowerGCFramePass())
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
    if VERSION < v"1.10"
        add!(mpm, LowerSIMDLoopPass())
    end
    add!(mpm, NewPMFunctionPassManager()) do fpm
        buildLoopOptimizerPipeline(fpm, job, opt_level)
        buildScalarOptimizerPipeline(fpm, job, opt_level)
        if uses_julia_runtime(job) && opt_level >= 2
            # XXX: we disable vectorization, as this generally isn't useful for GPU targets
            #      and actually causes issues with some back-end compilers (like Metal).
            # TODO: Make this not dependent on `uses_julia_runtime` (likely CPU), but it's own control
            buildVectorPipeline(fpm, job, opt_level)
        end
        if isdebug(:optim)
            add!(fpm, WarnMissedTransformationsPass())
        end
    end
    buildIntrinsicLoweringPipeline(mpm, job, opt_level)
    buildCleanupPipeline(mpm, job, opt_level)
end

if use_newpm
    const BasicSimplifyCFGOptions =
        (; convert_switch_range_to_icmp=true,
           convert_switch_to_lookup_table=true,
           forward_switch_cond_to_phi=true,
        )
    const AggressiveSimplifyCFGOptions =
        (; convert_switch_range_to_icmp=true,
           convert_switch_to_lookup_table=true,
           forward_switch_cond_to_phi=true,
           # These mess with loop rotation, so only do them after that
           hoist_common_insts=true,
           # Causes an SRET assertion error in late-gc-lowering
           #sink_common_insts=true
        )
end

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
    add!(mpm, CPUFeaturesPass())
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
        if VERSION >= v"1.10"
            add!(lpm, LowerSIMDLoopPass())
        end
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
    add!(mpm, NewPMFunctionPassManager()) do fpm
        if !uses_julia_runtime(job)
            add!(fpm, LowerGCFramePass())
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
        add!(mpm, LowerPTLSPass())
    end

    add!(mpm, NewPMFunctionPassManager()) do fpm
        # lower exception handling
        if uses_julia_runtime(job)
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
            add!(fpm, CombineMulAddPass())
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


## legacy pm

function optimize_legacypm!(@nospecialize(job::CompilerJob), mod::LLVM.Module; opt_level)
    triple = llvm_triple(job.config.target)
    tm = llvm_machine(job.config.target)

    global current_job
    current_job = job

    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)
        addOptimizationPasses!(pm, opt_level)
        run!(pm, mod)
    end

    # NOTE: we need to use multiple distinct pass managers to force pass ordering;
    #       intrinsics should never get lowered before Julia has optimized them.
    # XXX: why doesn't the barrier noop pass work here?

    # lower intrinsics
    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)

        if !uses_julia_runtime(job)
            lower_gc_frame!(pm)
        end

        if job.config.kernel
            # GC lowering is the last pass that may introduce calls to the runtime library,
            # and thus additional uses of the kernel state intrinsic.
            # TODO: now that all kernel state-related passes are being run here, merge some?
            add_kernel_state!(pm)
            lower_kernel_state!(pm)
            cleanup_kernel_state!(pm)
        end

        if !uses_julia_runtime(job)
            # remove dead uses of ptls
            aggressive_dce!(pm)
            lower_ptls!(pm)
        end

        if uses_julia_runtime(job)
            lower_exc_handlers!(pm)
        end
        # the Julia GC lowering pass also has some clean-up that is required
        late_lower_gc_frame!(pm)
        if uses_julia_runtime(job)
            final_lower_gc!(pm)
        end

        remove_ni!(pm)
        remove_julia_addrspaces!(pm)

        if uses_julia_runtime(job)
            # We need these two passes and the instcombine below
            # after GC lowering to let LLVM do some constant propagation on the tags.
            # and remove some unnecessary write barrier checks.
            gvn!(pm)
            sccp!(pm)
            # Remove dead use of ptls
            dce!(pm)
            LLVM.Interop.lower_ptls!(pm, dump_native(job))
            instruction_combining!(pm)
            # Clean up write barrier and ptls lowering
            cfgsimplification!(pm)
        end

        # Julia's operand bundles confuse the inliner, so repeat here now they are gone.
        # FIXME: we should fix the inliner so that inlined code gets optimized early-on
        always_inliner!(pm)

        # some of Julia's optimization passes happen _after_ lowering intrinsics
        combine_mul_add!(pm)
        div_rem_pairs!(pm)

        if VERSION < v"1.10.0-DEV.1144"
            # save function attributes to work around JuliaGPU/GPUCompiler#437
            current_attrs = Dict{String,Any}()
            for f in functions(mod)
                attrs = function_attributes(f)
                length(attrs) == 0 && continue
                current_attrs[LLVM.name(f)] = collect(attrs)
            end
        end

        run!(pm, mod)

        if VERSION < v"1.10.0-DEV.1144"
            # restore function attributes
            for (fn, attrs) in current_attrs
                haskey(functions(mod), fn) || continue
                f = functions(mod)[fn]

                for attr in attrs
                    # NOTE: there's no function attributes that contain a type,
                    #       so we can just blindly add them back
                    push!(function_attributes(f), attr)
                end
            end
        end
    end

    # target-specific optimizations
    optimize_module!(job, mod)

    # we compile a module containing the entire call graph,
    # so perform some interprocedural optimizations.
    #
    # for some reason, these passes need to be distinct from the regular optimization chain,
    # or certain values (such as the constant arrays used to populare llvm.compiler.user ad
    # part of the LateLowerGCFrame pass) aren't collected properly.
    #
    # these might not always be safe, as Julia's IR metadata isn't designed for IPO.
    @dispose pm=ModulePassManager() begin
        addTargetPasses!(pm, tm, triple)

        # simplify function calls that don't use the returned value
        dead_arg_elimination!(pm)

        run!(pm, mod)
    end

    return
end

function addTargetPasses!(pm, tm, triple)
    add_library_info!(pm, triple)
    add_transform_info!(pm, tm)
end

# Based on Julia's optimization pipeline, minus the SLP and loop vectorizers.
function addOptimizationPasses!(pm, opt_level)
    # compare with the using Julia's optimization pipeline directly:
    #ccall(:jl_add_optimization_passes, Cvoid,
    #      (LLVM.API.LLVMPassManagerRef, Cint, Cint),
    #      pm, opt_level, #=lower_intrinsics=# 0)
    #return

    # NOTE: LLVM 12 disabled the hoisting of common instruction
    #       before loop vectorization (https://reviews.llvm.org/D84108).
    #
    #       This is re-enabled with calls to cfg_simplify here,
    #       to merge allocations and sometimes eliminate them,
    #       since AllocOpt does not handle PhiNodes.
    #       Enable this instruction hoisting because of this and Union benchmarks.

    constant_merge!(pm)

    if opt_level < 2
        cpu_features!(pm)
        if opt_level == 1
            instruction_simplify!(pm)
        end
        if LLVM.version() >= v"12"
            cfgsimplification!(pm; hoist_common_insts=true)
        else
            cfgsimplification!(pm)
        end
        if opt_level == 1
            scalar_repl_aggregates!(pm)
            instruction_combining!(pm)
            early_cse!(pm)
            # maybe add GVN?
            # also try GVNHoist and GVNSink
        end
        mem_cpy_opt!(pm)
        always_inliner!(pm) # Respect always_inline
        lower_simdloop!(pm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
        return
    end

    propagate_julia_addrsp!(pm)
    scoped_no_alias_aa!(pm)
    type_based_alias_analysis!(pm)
    if opt_level >= 3
        basic_alias_analysis!(pm)
    end
    if LLVM.version() >= v"12"
        cfgsimplification!(pm; hoist_common_insts=true)
    else
        cfgsimplification!(pm)
    end
    dce!(pm)
    scalar_repl_aggregates!(pm)

    #mem_cpy_opt!(pm)

    always_inliner!(pm) # Respect always_inline

    # Running `memcpyopt` between this and `sroa` seems to give `sroa` a hard
    # time merging the `alloca` for the unboxed data and the `alloca` created by
    # the `alloc_opt` pass.

    alloc_opt!(pm)
    # consider AggressiveInstCombinePass at optlevel > 2
    instruction_combining!(pm)
    if LLVM.version() >= v"12"
        cfgsimplification!(pm; hoist_common_insts=true)
    else
        cfgsimplification!(pm)
    end
    cpu_features!(pm)
    scalar_repl_aggregates!(pm)
    # SROA can duplicate PHI nodes which can block LowerSIMD
    instruction_combining!(pm)
    jump_threading!(pm)
    correlated_value_propagation!(pm)

    reassociate!(pm)

    early_cse!(pm)

    # Load forwarding above can expose allocations that aren't actually used
    # remove those before optimizing loops.
    alloc_opt!(pm)
    loop_rotate!(pm)
    # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)

    # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
    lower_simdloop!(pm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
    licm!(pm)
    julia_licm!(pm)
    if LLVM.version() >= v"15"
        simple_loop_unswitch_legacy!(pm)
    else
        # XXX: simple loop unswitch is available on older versions of LLVM too,
        #      but using this pass instead of the old one breaks Metal.jl.
        loop_unswitch!(pm)
    end
    licm!(pm)
    julia_licm!(pm)
    inductive_range_check_elimination!(pm)
    # Subsequent passes not stripping metadata from terminator
    instruction_simplify!(pm)
    loop_idiom!(pm)
    ind_var_simplify!(pm)
    loop_deletion!(pm)
    loop_unroll!(pm) # TODO: in Julia createSimpleLoopUnroll

    # Run our own SROA on heap objects before LLVM's
    alloc_opt!(pm)
    # Re-run SROA after loop-unrolling (useful for small loops that operate,
    # over the structure of an aggregate)
    scalar_repl_aggregates!(pm)
    # might not be necessary:
    instruction_simplify!(pm)

    gvn!(pm)
    mem_cpy_opt!(pm)
    sccp!(pm)

    # These next two passes must come before IRCE to eliminate the bounds check in #43308
    correlated_value_propagation!(pm)
    dce!(pm)

    inductive_range_check_elimination!(pm)  # Must come between the two GVN passes

    # Run instcombine after redundancy elimination to exploit opportunities
    # opened up by them.
    # This needs to be InstCombine instead of InstSimplify to allow
    # loops over Union-typed arrays to vectorize.
    instruction_combining!(pm)
    jump_threading!(pm)
    if opt_level >= 3
        gvn!(pm)    # Must come after JumpThreading and before LoopVectorize
    end
    dead_store_elimination!(pm)

    # More dead allocation (store) deletion before loop optimization
    # consider removing this:
    alloc_opt!(pm)
    # see if all of the constant folding has exposed more loops
    # to simplification and deletion
    # this helps significantly with cleaning up iteration
    cfgsimplification!(pm)  # See note above, don't hoist instructions before LV
    loop_deletion!(pm)
    instruction_combining!(pm)
    loop_vectorize!(pm)
    loop_load_elimination!(pm)
    # Cleanup after LV pass
    instruction_combining!(pm)
    if LLVM.version() >= v"12"
        cfgsimplification!(pm; # Aggressive CFG simplification
            forward_switch_cond_to_phi=true,
            convert_switch_to_lookup_table=true,
            need_canonical_loop=true,
            hoist_common_insts=true,
            #sink_common_insts=true # FIXME: Causes assertion in llvm-late-lowering
        )
    else
        cfgsimplification!(pm)
    end

    aggressive_dce!(pm)
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
            unsafe_delete!(LLVM.parent(val), val)
        end
        @assert isempty(uses(f))
        unsafe_delete!(mod, f)
    end

    return changed
end
cpu_features!(pm::PassManager) = add!(pm, ModulePass("LowerCPUFeatures", cpu_features!))
if LLVM.has_newpm()
    CPUFeaturesPass() = NewPMModulePass("GPULowerCPUFeatures", cpu_features!)
end

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

            unsafe_delete!(LLVM.parent(call), call)

            changed = true
        end

        @compiler_assert isempty(uses(alloc_obj)) job
    end

    # we don't care about write barriers
    if haskey(functions(mod), "julia.write_barrier")
        barrier = functions(mod)["julia.write_barrier"]

        for use in uses(barrier)
            call = user(use)::LLVM.CallInst
            unsafe_delete!(LLVM.parent(call), call)
            changed = true
        end

        @compiler_assert isempty(uses(barrier)) job
    end

    return changed
end
lower_gc_frame!(pm::PassManager) = add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))
if LLVM.has_newpm()
    LowerGCFramePass() = NewPMFunctionPass("GPULowerGCFrame", lower_gc_frame!)
end

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
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            else
                # the validator will detect this
            end
        end
     end

    return changed
end
lower_ptls!(pm::PassManager) = add!(pm, ModulePass("LowerPTLS", lower_ptls!))
if LLVM.has_newpm()
    LowerPTLSPass() = NewPMModulePass("GPULowerPTLS", lower_ptls!)
end
