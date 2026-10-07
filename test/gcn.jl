if :AMDGPU in LLVM.backends()

# XXX: generic `sink` generates an instruction selection error
sink_gcn(i) = sink(i, Val(5))

@testset "backend selector" begin
    # in the test environment AMDGPU_LLVM_Backend_jll is loaded, so the default is :external
    @test GCNCompilerTarget(dev_isa="gfx900").backend === :external

    # both constructor forms accept an explicit backend, alongside the other options
    @test GCNCompilerTarget(dev_isa="gfx900"; backend=:inprocess).backend === :inprocess
    @test GCNCompilerTarget("gfx900"; backend=:inprocess).backend === :inprocess
    let target = GCNCompilerTarget("gfx900"; features="+wavefrontsize64", backend=:external)
        @test target.dev_isa == "gfx900"
        @test target.features == "+wavefrontsize64"
        @test target.backend === :external
    end

    mod = @eval module $(gensym())
        kernel() = return
    end

    # the backend participates in the cache owner, so different back-ends don't share a cache
    job_ext, _ = GCN.create_job(mod.kernel, Tuple{}; backend=:external)
    job_inp, _ = GCN.create_job(mod.kernel, Tuple{}; backend=:inprocess)
    owner_ext = GPUCompiler.cache_owner(job_ext)
    owner_inp = GPUCompiler.cache_owner(job_inp)
    @test owner_ext.target.backend === :external
    @test owner_inp.target.backend === :inprocess
    @test owner_ext != owner_inp

    # the explicit :external backend generates machine code through the external llc
    @test (GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:external); true)

    # the :inprocess backend generates machine code through the in-process LLVM back-end
    @test (GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:inprocess); true)

    # an unknown back-end is rejected at machine-code generation
    @test_throws "Unsupported GCN back-end" GCN.code_native(devnull, mod.kernel, Tuple{}; backend=:bogus)
end

@testset "IR" begin

@testset "fma" begin
    # `fma` uses the hardware instruction, not the Float64-based `fma_emulated`
    mod = @eval module $(gensym())
        kernel(p::Ptr{Float32}, x::Float32, y::Float32, z::Float32) =
            (unsafe_store!(p, fma(x, y, z)); return)
    end

    @test @filecheck begin
        @check_label "define {{.*}} @{{(julia|j)_kernel_[0-9]+}}"
        @check "call float @llvm.fma.f32"
        @check_not "double"
        GCN.code_llvm(mod.kernel, Tuple{Ptr{Float32}, Float32, Float32, Float32})
    end
end

@testset "synchronization scopes" begin
    # front-ends spell scopes like LLVM's SPIR-V back-end; they're renamed for the target
    mod = @eval module $(gensym())
        kernel() = Base.llvmcall("""
            fence syncscope("singlethread") seq_cst
            fence syncscope("subgroup") seq_cst
            fence syncscope("device") seq_cst
            fence syncscope("workgroup") seq_cst
            fence seq_cst
            fence syncscope("agent-one-as") seq_cst
            fence syncscope("device-mem-local") acquire
            ret void
            """, Nothing, Tuple{})
    end
    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel[0-9_]*}}"
        @check "fence syncscope(\"singlethread\") seq_cst"
        @check "fence syncscope(\"wavefront\") seq_cst"
        @check "fence syncscope(\"agent\") seq_cst"
        @check "fence syncscope(\"workgroup\") seq_cst"
        @check "fence seq_cst"
        @check "fence syncscope(\"agent-one-as\") seq_cst"
        # GCN orders all memory, so the memory a scope names is dropped
        @check "fence syncscope(\"agent\") acquire"
        GCN.code_llvm(mod.kernel, Tuple{})
    end
end

@testset "atomic validation" begin
    # validate textual IR, with AMDGPU's synchronization scopes, for a GCN target
    function validate_atomics(body; backend=:inprocess)
        source = methodinstance(typeof(identity), Tuple{Int}, Base.get_world_counter())
        target = GCNCompilerTarget(; dev_isa="gfx90a", backend)
        job = CompilerJob(source, CompilerConfig(target, GCN.CompilerParams(); kernel=true))
        datalayout = @dispose dl=GPUCompiler.llvm_datalayout(target) begin
            string(dl)
        end
        Context(; opaque_pointers=true) do ctx
            mod = parse(LLVM.Module, """
                target datalayout = "$datalayout"
                define void @kernel(ptr addrspace(1) %p, ptr addrspace(3) %s,
                                    ptr addrspace(4) %c, ptr addrspace(5) %l, ptr %g) {
                  $body
                  ret void
                }""")
            join(first.(GPUCompiler.validate_ir(job, mod)), "\n")
        end
    end

    @testset "unsupported" begin
        for (body, reason) in (
                ("%a = atomicrmw xchg ptr addrspace(1) %p, i128 1 syncscope(\"agent\") monotonic, align 16",
                 "128-bit atomic operation (GCN supports atomics of at most 64 bits)"),
                ("%a = cmpxchg ptr %g, i128 0, i128 1 syncscope(\"agent\") monotonic monotonic, align 16",
                 "128-bit atomic operation"),
                ("%a = load atomic i128, ptr addrspace(1) %p syncscope(\"agent\") acquire, align 16",
                 "128-bit atomic operation"),
                ("store atomic fp128 0xL0, ptr addrspace(3) %s syncscope(\"workgroup\") release, align 16",
                 "128-bit atomic operation"),
                ("%a = atomicrmw add ptr addrspace(1) %p, i64 1 syncscope(\"agent\") monotonic, align 4",
                 "atomic operation with alignment 4 (requires at least 8-byte alignment)"),
                ("%a = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope(\"device\") monotonic, align 4",
                 "atomic operation with synchronization scope \"device\""),
                ("%a = load atomic i32, ptr %g syncscope(\"system-one-as\") monotonic, align 4",
                 "atomic operation with synchronization scope \"system-one-as\""),
                ("fence syncscope(\"subgroup\") seq_cst",
                 "fence with synchronization scope \"subgroup\""),
                ("%a = load atomic i32, ptr addrspace(4) %c syncscope(\"agent\") monotonic, align 4",
                 "atomic operation in address space 4"),
            )
            @test occursin(reason, validate_atomics(body))
        end
    end

    @testset "supported" begin
        for body in (
                "%a = atomicrmw add ptr addrspace(1) %p, i64 1 syncscope(\"agent\") monotonic, align 8",
                "%a = cmpxchg ptr %g, i64 0, i64 1 seq_cst seq_cst, align 8",
                "%a = load atomic ptr, ptr addrspace(1) %p unordered, align 8",
                # operations the back-end expands
                "%a = atomicrmw nand ptr addrspace(3) %s, i8 1 syncscope(\"workgroup\") monotonic, align 1",
                "%a = atomicrmw fmax ptr %g, half 1.0 syncscope(\"wavefront\") monotonic, align 2",
                # private memory, where atomics are plain memory operations
                "%a = atomicrmw add ptr addrspace(5) %l, i32 1 syncscope(\"agent\") monotonic, align 4",
                # scopes that only order the address space of the operation
                "%a = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope(\"agent-one-as\") monotonic, align 4",
                "fence syncscope(\"one-as\") seq_cst\nfence syncscope(\"singlethread-one-as\") acquire",
            )
            @test validate_atomics(body) == ""
        end

        # packed floating-point values (LLVM 19 added vector `atomicrmw fadd`)
        if LLVM.version() >= v"19"
            @test validate_atomics("%a = atomicrmw fadd ptr addrspace(1) %p, <2 x half> zeroinitializer syncscope(\"agent\") monotonic, align 4") == ""
        end
    end

    # the cluster scope (the agent on targets without clusters) is only known from LLVM 22
    cluster = "%a = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope(\"cluster\") monotonic, align 4\n" *
              "fence syncscope(\"cluster-one-as\") acquire"
    @test occursin("synchronization scope \"cluster\"", validate_atomics(cluster)) ==
          (LLVM.version() < v"22")
    @test validate_atomics(cluster; backend=:external) == ""
end

@testset "kernel calling convention" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "amdgpu_kernel"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true)
    end

    @test @filecheck begin
        @check "amdgpu_kernel"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
    end
end

@testset "launch bounds" begin
    mod = @eval module $(gensym())
        kernel() = return
    end

    @test @filecheck begin
        @check_not "amdgpu-flat-work-group-size"
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true)
    end

    @test @filecheck begin
        @check "\"amdgpu-flat-work-group-size\"=\"1,42\""
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true, maxthreads=42)
    end

    @test @filecheck begin
        @check "\"amdgpu-flat-work-group-size\"=\"256,256\""
        GCN.code_llvm(mod.kernel, Tuple{}; dump_module=true, kernel=true,
                      minthreads=256, maxthreads=256)
    end

    @test @filecheck begin
        @check ".max_flat_workgroup_size: 42"
        GCN.code_native(mod.kernel, Tuple{}; dump_module=true, kernel=true, maxthreads=42)
    end
end

@testset "bounds errors" begin
    mod = @eval module $(gensym())
        function kernel()
            Base.throw_boundserror(1, 2)
            return
        end
    end

    @test @filecheck begin
        @check_not "{{julia_throw_boundserror_[0-9]+}}"
        @check "@gpu_report_exception"
        @check "@gpu_signal_exception"
        GCN.code_llvm(mod.kernel, Tuple{})
    end
end

@testset "kernarg address space for byref parameters" begin
    mod = @eval module $(gensym())
        struct MyStruct
            x::Float64
            y::Float64
        end

        function kernel(s::MyStruct)
            s.x + s.y
            return
        end
    end

    # byref struct params should be ptr addrspace(4) in kernel IR
    @test @filecheck begin
        @check cond=typed_ptrs "define amdgpu_kernel void @_Z6kernel8MyStruct({{.*}} addrspace(4)*"
        @check cond=opaque_ptrs "define amdgpu_kernel void @_Z6kernel8MyStruct(ptr addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.MyStruct}; dump_module=true, kernel=true)
    end

    # non-kernel should NOT have addrspace(4)
    @test @filecheck begin
        @check_not "addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.MyStruct}; dump_module=true, kernel=false)
    end
end

@testset "byref attribute preserved on kernarg parameters" begin
    mod = @eval module $(gensym())
        struct LargeStruct
            a::Float64
            b::Float64
            c::Float64
            d::Float64
        end

        function kernel(s::LargeStruct, out::Ptr{Float64})
            unsafe_store!(out, s.a + s.b + s.c + s.d)
            return
        end
    end

    # the byref attribute must survive the addrspace rewrite (clone_into! can drop it)
    @test @filecheck begin
        @check "byref"
        @check "addrspace(4)"
        GCN.code_llvm(mod.kernel, Tuple{mod.LargeStruct, Ptr{Float64}};
                       dump_module=true, kernel=true)
    end
end

@testset "mixed byref and scalar kernel parameters" begin
    mod = @eval module $(gensym())
        struct Params
            x::Float64
            y::Float64
        end

        function kernel(a::Float64, s::Params, out::Ptr{Float64})
            unsafe_store!(out, a + s.x + s.y)
            return
        end
    end

    # scalar Float64 should NOT be in addrspace(4),
    # only the struct byref param should be.
    # NOTE: Ptr{Float64} is lowered to i64 on Julia ≤1.11 and ptr on Julia 1.12+.
    @test @filecheck begin
        @check "define amdgpu_kernel void"
        @check_same "double"
        @check_same cond=typed_ptrs "{{.*}} addrspace(4)*"
        @check_same cond=opaque_ptrs "ptr addrspace(4)"
        @check_same "{{(i64|ptr)}}"
        GCN.code_llvm(mod.kernel, Tuple{Float64, mod.Params, Ptr{Float64}};
                       dump_module=true, kernel=true)
    end
end

@testset "add_kernarg_address_spaces! rewrites IR correctly" begin
    mod = @eval module $(gensym())
        struct KernelArgs
            x::Float64
            y::Float64
            z::Float64
        end

        function kernel(s::KernelArgs, scale::Float64, out::Ptr{Float64})
            unsafe_store!(out, (s.x + s.y + s.z) * scale)
            return
        end
    end

    job, _ = GCN.create_job(mod.kernel, Tuple{mod.KernelArgs, Float64, Ptr{Float64}};
                             kernel=true)
    JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        @dispose ir=ir begin
            entry = meta.entry
            ft = entry.function_type
            params = ft.parameters

            # the struct byref param should be ptr addrspace(4)
            has_as4 = any(p -> p isa LLVM.PointerType && p.addrspace == 4, params)
            @test has_as4

            # non-struct params (double, and i64/ptr for Ptr{Float64}) should NOT
            # be in addrspace(4). Ptr{Float64} is i64 on Julia ≤1.11, ptr on 1.12+.
            non_byref = filter(p -> !(p isa LLVM.PointerType && p.addrspace == 4), params)
            @test !isempty(non_byref)  # double (and i64 or ptr) params

            # byref attribute must be present
            ir_str = string(ir)
            @test occursin("byref", ir_str)
        end
    end
end

@testset "https://github.com/JuliaGPU/AMDGPU.jl/issues/846" begin
    ir, rt = GCN.code_typed((Tuple{Tuple{Val{4}}, Tuple{Float32}},); always_inline=true) do t
        t[1]
    end |> only
    @test rt == Tuple{Val{4}}
end

end

############################################################################################
@testset "assembly" begin

@testset "patchable relocation visibility" begin
    # AMDGPU.jl links the object into a shared library with ld.lld. Julia references the
    # record globals `dso_local` (PC-relative `@rel32`), so a weak default-visibility
    # definition would be rejected as preemptible ("recompile with -fPIC"); the symbol
    # must be protected.
    if GPUCompiler.supports_relocatable_ir()
        mod = @eval module $(gensym())
            function kernel(out::Ptr{Bool}, s::Symbol)
                unsafe_store!(out, s === :foo)
                return
            end
        end
        asm = sprint(io->GCN.code_native(io, mod.kernel, Tuple{Ptr{Bool},Symbol};
                                         kernel=true, patch=true))
        m = match(r"(?m)^\s*\.protected\s+(\S+jl_sym_foo\S*)", asm)
        @test m !== nothing
        if m !== nothing
            name = m.captures[1]
            @test occursin(r"(?m)^\s*\.weak\s+" * name, asm)
            @test occursin("$(name)@rel32@lo", asm)
        end
    end
end

@testset "atomic validation" begin
    # atomics in Julia code are reported with the frame that performs them, for both
    # back-ends (Julia's LLVM would emit a libatomic call, failing only at load time)
    function add_kernel(T)
        ptr = typed_ptrs ? "$T addrspace(1)*" : "ptr addrspace(1)"
        ir = """
            define void @entry($ptr %p, $T %x) #0 {
              %old = atomicrmw add $ptr %p, $T %x syncscope("agent") monotonic, align 16
              ret void
            }
            attributes #0 = { alwaysinline }"""
        JT = T == "i128" ? Int128 : Int64
        mod = @eval module $(gensym())
            kernel(p::Core.LLVMPtr{$JT,1}, x::$JT) =
                (Base.llvmcall(($ir, "entry"), Nothing, Tuple{Core.LLVMPtr{$JT,1},$JT}, p, x);
                 return)
        end
        Base.invokelatest(getfield, mod, :kernel), Tuple{Core.LLVMPtr{JT,1},JT}
    end
    f, tt = add_kernel("i128")
    for backend in (:inprocess, :external)
        @test_throws_message(InvalidIRError,
                             Base.invokelatest(GCN.code_execution, f, tt; backend)) do msg
            occursin("Reason: unsupported 128-bit atomic operation", msg) &&
            occursin(r"\[\d+\] kernel", msg)
        end
    end
    asm, _ = Base.invokelatest(GCN.code_execution, add_kernel("i64")...)
    @test occursin("global_atomic_add_x2", asm)
end

LLVM.version() >= v"20" && @testset "usub_sat atomics" begin
    # LLVM 22 fails to select 32-bit `usub_sat` on flat memory, and on local memory before
    # gfx12 (llvm/llvm-project#229442). AMDGPU_LLVM_Backend_jll carries the fix, but with
    # Julia's own LLVM the operation is expanded to a compare-exchange loop
    function rmw_kernel(as, op="usub_sat")
        ir = """
            define void @entry(ptr addrspace($as) %p, i32 %v) #0 {
              %r = atomicrmw $op ptr addrspace($as) %p, i32 %v syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
              ret void
            }
            attributes #0 = { alwaysinline }
            !0 = !{}"""
        mod = @eval module $(gensym())
            kernel(p::Core.LLVMPtr{UInt32,$as}, v::UInt32) =
                (Base.llvmcall(($ir, "entry"), Nothing, Tuple{Core.LLVMPtr{UInt32,$as}, UInt32},
                               p, v); return)
        end
        Base.invokelatest(getfield, mod, :kernel), Tuple{Core.LLVMPtr{UInt32,as},UInt32}
    end

    # (flat pointers in kernel arguments are global pointers to the back-end)
    for (dev_isa, as, affected, instruction) in (
            ("gfx1030", 3, true, "ds_cmpst_rtn_b32"),
            ("gfx1100", 3, true, "ds_cmpstore_rtn_b32"),
            ("gfx10-3-generic", 3, true, "ds_cmpst_rtn_b32"),
            ("gfx11-generic", 3, true, "ds_cmpstore_rtn_b32"),
            ("gfx1200", 3, false, "ds_sub_clamp_u32"),
            ("gfx1030", 0, true, "flat_atomic_cmpswap"),
            ("gfx1100", 0, true, "flat_atomic_cmpswap_b32"),
            ("gfx1200", 0, true, "flat_atomic_sub_clamp_u32"),
            ("gfx90a", 0, true, "flat_atomic_cmpswap"),
            ("gfx1100", 1, false, "global_atomic_csub_u32"),
            ("gfx1200", 1, false, "global_atomic_sub_clamp_u32"),
        )
        f, tt = rmw_kernel(as)
        kernel = as != 0

        # the back-end compiles the operation, expanding it where the hardware lacks it
        ir = sprint(io->GCN.code_llvm(io, f, tt; dev_isa, kernel))
        @test occursin("atomicrmw usub_sat", ir)
        asm = sprint(io->GCN.code_native(io, f, tt; dev_isa, kernel))
        @test occursin(instruction, asm)

        # Julia's LLVM needs the workaround from version 22 on
        ir = sprint(io->GCN.code_llvm(io, f, tt; dev_isa, kernel, backend=:inprocess))
        @test occursin("atomicrmw usub_sat", ir) == (LLVM.version() < v"22" || !affected)

        # other operations are unaffected
        f, tt = rmw_kernel(as, "usub_cond")
        ir = sprint(io->GCN.code_llvm(io, f, tt; dev_isa, kernel, backend=:inprocess))
        @test occursin("atomicrmw usub_cond", ir)
    end

    # the expansion is a workaround, not part of validation
    if LLVM.version() >= v"22"
        f, tt = rmw_kernel(3)
        ir = sprint(io->GCN.code_llvm(io, f, tt; dev_isa="gfx1030", kernel=true,
                                      backend=:inprocess, validate=false))
        @test !occursin("atomicrmw usub_sat", ir)
        asm = sprint(io->GCN.code_native(io, f, tt; dev_isa="gfx1030", kernel=true,
                                         backend=:inprocess, validate=false))
        @test occursin("ds_cmpst_rtn_b32", asm)
    end
end

@testset "sub-word atomics with a used result" begin
    # before LLVM 21, the back-end crashes on 8- and 16-bit atomic operations on uniform
    # addresses whose result is used (llvm/llvm-project#128388), so they are expanded to a
    # compare-exchange loop
    jltype(T) = T == "i8" ? Int8 : T == "i16" ? Int16 : Int32
    function rmw_source(T, op; used=true, as=1)
        align = sizeof(jltype(T))
        ptr = typed_ptrs ? "$T addrspace($as)*" : "ptr addrspace($as)"
        """
        define void @entry($ptr %p, $T %x) #0 {
          %old = atomicrmw $op $ptr %p, $T %x syncscope("agent") monotonic, align $align
          $(used ? "%q = getelementptr inbounds $T, $ptr %p, i64 1" : "")
          $(used ? "store $T %old, $ptr %q, align $align" : "")
          ret void
        }
        attributes #0 = { alwaysinline }"""
    end
    function rmw_ir(T, op; used=true, as=1, backend)
        JT, ir = jltype(T), rmw_source(T, op; used, as)
        mod = @eval module $(gensym())
            kernel(p::Core.LLVMPtr{$JT,$as}, x::$JT) =
                (Base.llvmcall(($ir, "entry"), Nothing, Tuple{Core.LLVMPtr{$JT,$as},$JT}, p, x);
                 return)
        end
        kernel = Base.invokelatest(getfield, mod, :kernel)
        Base.invokelatest(sprint, io->GCN.code_llvm(io, kernel, Tuple{Core.LLVMPtr{JT,as},JT};
                                                    dev_isa="gfx1030", kernel=true, backend))
    end

    for (T, op, as) in (("i8", "add", 1), ("i16", "max", 3), ("i8", "umin", 0))
        @test occursin("atomicrmw $op", rmw_ir(T, op; as, backend=:inprocess)) ==
              (LLVM.version() >= v"21")
        @test occursin("atomicrmw $op", rmw_ir(T, op; as, backend=:external))
    end
    # other operations, unused results and wider atomics are unaffected
    @test occursin("atomicrmw xchg", rmw_ir("i8", "xchg"; backend=:inprocess))
    @test occursin("atomicrmw add", rmw_ir("i8", "add"; used=false, backend=:inprocess))
    @test occursin("atomicrmw add", rmw_ir("i32", "add"; backend=:inprocess))

    # the back-end crashes the process, so generate code in another one
    script = """
        using GPUCompiler, LLVM
        include($(repr(joinpath(@__DIR__, "helpers", "runtime.jl"))))
        include($(repr(joinpath(@__DIR__, "helpers", "gcn.jl"))))
        kernel_i8(p::Core.LLVMPtr{Int8,1}, x::Int8) =
            (Base.llvmcall(($(repr(rmw_source("i8", "add"; as=1))), "entry"), Nothing,
                           Tuple{Core.LLVMPtr{Int8,1},Int8}, p, x); return)
        kernel_i16(p::Core.LLVMPtr{Int16,3}, x::Int16) =
            (Base.llvmcall(($(repr(rmw_source("i16", "max"; as=3))), "entry"), Nothing,
                           Tuple{Core.LLVMPtr{Int16,3},Int16}, p, x); return)
        for (f, tt) in ((kernel_i8, Tuple{Core.LLVMPtr{Int8,1},Int8}),
                        (kernel_i16, Tuple{Core.LLVMPtr{Int16,3},Int16}))
            GCN.code_native(devnull, f, tt; dev_isa="gfx1030", kernel=true, backend=:inprocess)
        end
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    @test success(pipeline(cmd; stdout, stderr))
end

@testset "s_load for kernarg struct access" begin
    mod = @eval module $(gensym())
        struct MyStruct
            x::Float64
            y::Float64
        end

        function kernel(s::MyStruct, out::Ptr{Float64})
            unsafe_store!(out, s.x + s.y)
            return
        end
    end

    # struct field loads from kernarg should use s_load, not flat_load
    @test @filecheck begin
        @check "s_load_dwordx"
        @check_not "flat_load"
        GCN.code_native(mod.kernel, Tuple{mod.MyStruct, Ptr{Float64}}; kernel=true)
    end
end

@testset "no scratch spills for small struct kernarg" begin
    mod = @eval module $(gensym())
        struct SmallStruct
            x::Float64
            y::Float64
        end

        function kernel(s::SmallStruct, out::Ptr{Float64})
            unsafe_store!(out, s.x + s.y)
            return
        end
    end

    # a small struct kernel should not need scratch memory
    @test @filecheck begin
        @check ".private_segment_fixed_size: 0"
        GCN.code_native(mod.kernel, Tuple{mod.SmallStruct, Ptr{Float64}};
                         dump_module=true, kernel=true)
    end
end

@testset "Int128 kernel arguments" begin
    # the back-end must lay out kernel arguments like the host does, which before Julia 1.12
    # aligns Int128 to 8 bytes (JuliaGPU/AMDGPU.jl#1002)
    mod = @eval module $(gensym())
        kernel(a::Int32, b::Int128, c::Tuple{Int32,Int128}, d::Int64) = return
    end
    types = Tuple{Int32, Int128, Tuple{Int32,Int128}, Int64}
    arg_offset(i) = fieldoffset(types, i)
    arg_size(i) = sizeof(fieldtype(types, i))

    @test @filecheck begin
        @check "amdhsa.kernels:"
        @check_next "- .args:"
        @check ".offset: $(arg_offset(1)){{\$}}"
        @check_next ".size: $(arg_size(1)){{\$}}"
        @check_next ".value_kind: by_value"
        @check ".offset: $(arg_offset(2)){{\$}}"
        @check_next ".size: $(arg_size(2)){{\$}}"
        @check_next ".value_kind: by_value"
        @check ".offset: $(arg_offset(3)){{\$}}"
        @check_next ".size: $(arg_size(3)){{\$}}"
        @check_next ".value_kind: by_value"
        @check ".offset: $(arg_offset(4)){{\$}}"
        @check_next ".size: $(arg_size(4)){{\$}}"
        @check_next ".value_kind: by_value"
        GCN.code_native(mod.kernel, types; dump_module=true, kernel=true)
    end
end

@testset "skip scalar trap" begin
    mod = @eval module $(gensym())
        workitem_idx_x() = ccall("llvm.amdgcn.workitem.id.x", llvmcall, Int32, ())
        trap() = ccall("llvm.trap", llvmcall, Nothing, ())

        function kernel()
            if workitem_idx_x() > 1
                trap()
            end
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_kernel_[0-9]+}}:"
        @check "s_cbranch_exec"
        @check "s_trap 2"
        GCN.code_native(mod.kernel, Tuple{})
    end
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child(i) = sink_gcn(i)
        function parent(i)
            child(i)
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_parent_[0-9]+}}:"
        @check "s_add_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
        @check "s_addc_u32 {{.+}} {{(julia|j)_child_[0-9]+}}@rel32@"
        GCN.code_native(mod.parent, Tuple{Int64}; dump_module=true)
    end
end

@testset "kernel functions" begin
    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline nonentry(i) = sink_gcn(i)
        function entry(i)
            nonentry(i)
            return
        end
    end

    @test @filecheck begin
        @check ".type {{(julia|j)_nonentry_[0-9]+}},@function"
        @check ".symbol:{{.*}}_Z5entry5Int64.kd"
        @check_not ".symbol:{{.*}}nonentry"
        GCN.code_native(mod.entry, Tuple{Int64}; dump_module=true, kernel=true)
    end
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child(i) = sink_gcn(i)
        function parent1(i)
            child(i)
            return
        end
        function parent2(i)
            child(i+1)
            return
        end
    end

    @test @filecheck begin
        @check ".type {{(julia|j)_child_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        @check ".type {{(julia|j)_child_[0-9]+}},@function"
        GCN.code_native(mod.parent2, Tuple{Int}; dump_module=true)
    end
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions

    mod = @eval module $(gensym())
        import ..sink_gcn
        @noinline child1(i) = sink_gcn(i)
        @noinline child2(i) = sink_gcn(i+1)
        function parent1(i)
            child1(i) + child2(i)
            return
        end
        function parent2(i)
            child1(i+1) + child2(i+1)
            return
        end
    end

    @test @filecheck begin
        @check_dag ".type {{(julia|j)_child1_[0-9]+}},@function"
        @check_dag ".type {{(julia|j)_child2_[0-9]+}},@function"
        GCN.code_native(mod.parent1, Tuple{Int}; dump_module=true)
    end

    @test @filecheck begin
        @check_dag ".type {{(julia|j)_child1_[0-9]+}},@function"
        @check_dag ".type {{(julia|j)_child2_[0-9]+}},@function"
        GCN.code_native(mod.parent2, Tuple{Int}; dump_module=true)
    end
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the GCN code shouldn't contain a throw)

    # NOTE: Int32 to test for #49

    mod = @eval module $(gensym())
        function kernel(out)
            wid, lane = fldmod1(unsafe_load(out), Int32(32))
            unsafe_store!(out, wid)
            return
        end
    end

    @test @filecheck begin
        @check_label "{{(julia|j)_kernel_[0-9]+}}:"
        @check_not "jl_throw"
        @check_not "jl_invoke"
        GCN.code_native(mod.kernel, Tuple{Ptr{Int32}})
    end
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    mod = @eval module $(gensym())
        function kernel(x)
            unsafe_trunc(Int, x)
            return
        end
    end
    GCN.code_native(devnull, mod.kernel, Tuple{Float64})
    @test "We did not crash!" != ""
end

# FIXME: _ZNK4llvm14TargetLowering20scalarizeVectorStoreEPNS_11StoreSDNodeERNS_12SelectionDAGE
false && @testset "exception arguments" begin
    mod = @eval module $(gensym())
        function kernel(a)
            unsafe_store!(a, trunc(Int, unsafe_load(a)))
            return
        end
    end

    GCN.code_native(devnull, mod.kernel, Tuple{Ptr{Float64}})
end

# FIXME: in function julia_inner_18528 void (%jl_value_t addrspace(10)*): invalid addrspacecast
false && @testset "GC and TLS lowering" begin
    mod = @eval module $(gensym())
        import ..sink_gcn
        mutable struct PleaseAllocate
            y::Csize_t
        end

        # common pattern in Julia 0.7: outlined throw to avoid a GC frame in the calling code
        @noinline function inner(x)
            sink_gcn(x.y)
            nothing
        end

        function kernel(i)
            inner(PleaseAllocate(Csize_t(42)))
            nothing
        end
    end

    @test @filecheck begin
        @check_not "jl_push_gc_frame"
        @check_not "jl_pop_gc_frame"
        @check_not "jl_get_gc_frame_slot"
        @check_not "jl_new_gc_frame"
        @check "gpu_gc_pool_alloc"
        GCN.code_native(mod.kernel, Tuple{Int})
    end

    # make sure that we can still ellide allocations
    function ref_kernel(ptr, i)
        data = Ref{Int64}()
        data[] = 0
        if i > 1
            data[] = 1
        else
            data[] = 2
        end
        unsafe_store!(ptr, data[], i)
        return nothing
    end

    @test @filecheck begin
        @check_not "gpu_gc_pool_alloc"
        GCN.code_native(ref_kernel, Tuple{Ptr{Int64}, Int})
    end
end

@testset "float boxes" begin
    mod = @eval module $(gensym())
        function kernel(a,b)
            # Int32(a) may fail, throwing an `InexactError`, whose `@nospecialize`
            # constructor would box the Float32
            c = Int32(a)
            unsafe_store!(b, c)
            return
        end
    end

    # the exception object isn't constructed, as nothing looks at it
    @test @filecheck begin
        @check_label "define void @{{(julia|j)_kernel_[0-9]+}}"
        @check_not "jl_box_float32"
        @check_not "gpu_gc_pool_alloc"
        GCN.code_llvm(mod.kernel, Tuple{Float32,Ptr{Float32}}; dump_module=true)
    end
    GCN.code_native(devnull, mod.kernel, Tuple{Float32,Ptr{Float32}})
end

@testset "stack allocation intrinsic" begin
    mod = @eval module $(gensym())
        import ..GPUCompiler

        function scratch(x)
            p = GPUCompiler.alloca(Float32, Val(8), Val(5))
            @inbounds unsafe_store!(p, x, 1)
            @inbounds unsafe_store!(p, x, 8)
            return @inbounds unsafe_load(p, 1) + unsafe_load(p, 8)
        end

        # zero-element scratch yields a (null) pointer without emitting an alloca
        empty_scratch() = GPUCompiler.alloca(Float32, Val(0), Val(5)) === reinterpret(Core.LLVMPtr{Float32,5}, C_NULL)
    end

    # AMDGPU uses alloca address space 5, which is exactly what the scratch requests, so the
    # materialized slot lives in AS 5 and no `addrspacecast` is needed.
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check "alloca [8 x i32], align 4, addrspace(5)"
        @check_not "addrspacecast"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.scratch, Tuple{Float32}; optimize=false, dump_module=true)
    end

    # once optimized the slot is promoted away entirely (result is x + x).
    @test @filecheck begin
        @check_label "define float @{{(julia|j)_scratch_[0-9]+}}"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.scratch, Tuple{Float32})
    end

    # a zero-byte allocation lowers to a null pointer rather than a degenerate alloca.
    @test @filecheck begin
        @check_label "define {{.*}}@{{(julia|j)_empty_scratch_[0-9]+}}"
        @check_not "= alloca"
        @check_not "julia.gpu.alloca"
        GCN.code_llvm(mod.empty_scratch, Tuple{})
    end
end

end
end # :AMDGPU in LLVM.backends()
