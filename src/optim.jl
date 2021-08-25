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

    # compate to Clang by using the pass manager builder APIs:
    #LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
    #ModulePassManager() do pm
    #    add_library_info!(pm, triple(mod))
    #    add_transform_info!(pm, tm)
    #    PassManagerBuilder() do pmb
    #        populate!(pm, pmb)
    #    end
    #    run!(pm, mod)
    #end

    # NOTE: LLVM 12 disabled the hoisting of common instruction
    #       before loop vectorization (https://reviews.llvm.org/D84108).
    #
    #       This is re-enabled with calls to cfg_simplify here,
    #       to merge allocations and sometimes eliminate them,
    #       since AllocOpt does not handle PhiNodes.
    #       Enable this instruction hoisting because of this and Union benchmarks.

    constant_merge!(pm)

    if opt_level < 2
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

        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        run!(pm, mod)
    end

    return
end


## lowering intrinsics

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
            ops = operands(call)
            sz = ops[2]

            # replace with PTX alloc_obj
            let builder = Builder(ctx)
                position!(builder, call)
                ptr = call!(builder, Runtime.get(:gc_pool_alloc), [sz])
                replace_uses!(call, ptr)
                dispose(builder)
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
