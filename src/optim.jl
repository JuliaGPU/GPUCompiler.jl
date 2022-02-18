# LLVM IR optimization

function addTargetPasses!(pm, tm, triple)
    add_library_info!(pm, triple)
    add_transform_info!(pm, tm)
end

# Based on Julia's optimization pipeline, minus the SLP and loop vectorizers.
function addOptimizationPasses!(pm, opt_level=2)
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
    instruction_simplify!(pm)
    jump_threading!(pm)
    correlated_value_propagation!(pm)

    reassociate!(pm)

    early_cse!(pm)

    # Load forwarding above can expose allocations that aren't actually used
    # remove those before optimizing loops.
    alloc_opt!(pm)
    loop_rotate!(pm)
    # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
    loop_idiom!(pm)

    # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
    lower_simdloop!(pm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
    licm!(pm)
    julia_licm!(pm)
    loop_unswitch!(pm)
    licm!(pm)
    julia_licm!(pm)
    inductive_range_check_elimination!(pm)
    # Subsequent passes not stripping metadata from terminator
    instruction_simplify!(pm)
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

    # Run instcombine after redundancy elimination to exploit opportunities
    # opened up by them.
    # This needs to be InstCombine instead of InstSimplify to allow
    # loops over Union-typed arrays to vectorize.
    instruction_combining!(pm)
    jump_threading!(pm)
    correlated_value_propagation!(pm)
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
    if LLVM.version() >= v"12"
        cfgsimplification!(pm; # Aggressive CFG simplification
            forward_switch_cond_to_phi=true,
            convert_switch_to_lookup_table=true,
            need_canonical_loop=true,
            hoist_common_insts=true,
            sink_common_insts=true) # FIXME: Causes assertion in llvm-late-lowering
    else
        cfgsimplification!(pm)
    end

    aggressive_dce!(pm)
end

function optimize!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    triple = llvm_triple(job.target)
    tm = llvm_machine(job.target)

    global current_job
    current_job = job

    ModulePassManager() do pm
        addTargetPasses!(pm, tm, triple)
        addOptimizationPasses!(pm)
        run!(pm, mod)
    end

    # NOTE: we need to use multiple distinct pass managers to force pass ordering;
    #       intrinsics should never get lowered before Julia has optimized them.
    # XXX: why doesn't the barrier noop pass work here?

    # lower intrinsics
    ModulePassManager() do pm
        addTargetPasses!(pm, tm, triple)

        add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))

        if job.source.kernel
            # GC lowering is the last pass that may introduce calls to the runtime library,
            # and thus additional uses of the kernel state intrinsic.
            add!(pm, FunctionPass("LowerKernelState", lower_kernel_state!))
            add!(pm, ModulePass("CleanupKernelState", cleanup_kernel_state!))
        end

        # remove dead uses of ptls
        aggressive_dce!(pm)
        add!(pm, ModulePass("LowerPTLS", lower_ptls!))

        # the Julia GC lowering pass also has some clean-up that is required
        late_lower_gc_frame!(pm)

        remove_ni!(pm)
        remove_julia_addrspaces!(pm)

        # Julia's operand bundles confuse the inliner, so repeat here now they are gone.
        # FIXME: we should fix the inliner so that inlined code gets optimized early-on
        always_inliner!(pm)

        # some of Julia's optimization passes happen _after_ lowering intrinsics
        combine_mul_add!(pm)
        div_rem_pairs!(pm)

        run!(pm, mod)
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
    ModulePassManager() do pm
        addTargetPasses!(pm, tm, triple)

        # - remove unused kernel state arguments
        # - simplify function calls that don't use the returned value
        dead_arg_elimination!(pm)

        run!(pm, mod)
    end

    # compare to Clang by using the pass manager builder APIs:
    #LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
    #ModulePassManager() do pm
    #    addTargetPasses!(pm, tm, triple)
    #    PassManagerBuilder() do pmb
    #        optlevel!(pmb, 2)
    #        populate!(pm, pmb)
    #    end
    #    run!(pm, mod)
    #end

    return
end


## lowering intrinsics
cpu_features!(pm::PassManager) = add!(pm, ModulePass("LowerCPUFeatures", cpu_features!))
function cpu_features!(mod::LLVM.Module)
    job = current_job::CompilerJob
    ctx = context(mod)
    changed = false

    argtyps = Dict(
        "f32" => Float32,
        "f64" => Float64,
    )

    # have_fma
    for f in functions(mod)
        ft = eltype(llvmtype(f))
        fn = LLVM.name(f)
        startswith(fn, "julia.cpu.have_fma.") || continue
        typnam = fn[20:end]

        # determine whether this back-end supports FMA on this type
        has_fma = if haskey(argtyps, typnam)
            typ = argtyps[typnam]
            have_fma(job.target, typ)
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
    ctx = context(fun)
    changed = false

    # plain alloc
    if haskey(functions(mod), "julia.gc_alloc_obj")
        alloc_obj = functions(mod)["julia.gc_alloc_obj"]
        alloc_obj_ft = eltype(llvmtype(alloc_obj))
        T_prjlvalue = return_type(alloc_obj_ft)
        T_pjlvalue = convert(LLVMType, Any; ctx, allow_boxed=true)

        for use in uses(alloc_obj)
            call = user(use)::LLVM.CallInst

            # decode the call
            ops = arguments(call)
            sz = ops[2]

            # replace with PTX alloc_obj
            Builder(ctx) do builder
                # NOTE: this happens late during the pipeline, where we may have to
                #       pass a kernel state arguments to the runtime function.
                state = if job.source.kernel
                    kernel_state_type(job)
                else
                    Nothing
                end

                position!(builder, call)
                ptr = call!(builder, Runtime.get(:gc_pool_alloc), [sz]; state)
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

    intrinsic = VERSION >= v"1.7.0-DEV.1205" ? "julia.get_pgcstack" : "julia.ptls_states"

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
