# Smoke test for one installed GPUCompiler version, run by matrix.jl in an environment that
# has only GPUCompiler added. Works across every API generation since v0.2, and prints one
# `RESULT <step> <ok|fail|skip|unsupported> [message]` line per step.

function report(step, status, msg="")
    msg = replace(first(msg, 2000), '\n' => "\\n")
    println("RESULT $step $status $msg")
    flush(stdout)
end

try
    @eval using GPUCompiler
    report("load", "ok")
catch err
    report("load", "fail", sprint(showerror, err))
    for step in ("native", "native_jlruntime", "ptx", "ptx_throw")
        report(step, "skip")
    end
    exit(0)
end

# v0.1 had per-target job types, which this script doesn't cover
if !isdefined(GPUCompiler, :AbstractCompilerParams)
    for step in ("native", "native_jlruntime", "ptx", "ptx_throw")
        report(step, "unsupported")
    end
    exit(0)
end

module SmokeRuntime
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct SmokeParams <: GPUCompiler.AbstractCompilerParams end

GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{<:Any,SmokeParams}) = SmokeRuntime
# before 2.0, runtime libraries are cached by slug, and older versions have no default one
if isdefined(GPUCompiler, :runtime_slug)
    GPUCompiler.runtime_slug(::GPUCompiler.CompilerJob{GPUCompiler.NativeCompilerTarget,SmokeParams}) =
        "smoke-native"
    GPUCompiler.runtime_slug(::GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget,SmokeParams}) =
        "smoke-ptx"
end

function smoke_job(f, tt, target; kernel)
    if isdefined(GPUCompiler, :CompilerConfig)
        source = if isdefined(GPUCompiler, :methodinstance)
            GPUCompiler.methodinstance(typeof(f), tt)
        else
            GPUCompiler.FunctionSpec(typeof(f), tt)
        end
        config = GPUCompiler.CompilerConfig(target, SmokeParams(); kernel=kernel)
        GPUCompiler.CompilerJob(source, config)
    else
        source = GPUCompiler.FunctionSpec(f, tt, kernel)
        GPUCompiler.CompilerJob(target, source, SmokeParams())
    end
end

function step(name, f, tt, target; kernel=false, expect=nothing)
    try
        job = smoke_job(f, tt, target; kernel=kernel)
        asm = sprint(io -> GPUCompiler.code_native(io, job))
        if expect !== nothing && !occursin(expect, asm)
            report(name, "fail", "unexpected output:\n" * asm)
        else
            report(name, "ok")
        end
    catch err
        report(name, "fail", sprint(showerror, err) * "\n" *
                             sprint(Base.show_backtrace, catch_backtrace()))
    end
end

add_one(x) = x + 1
store_one(p::Ptr{Float32}) = (unsafe_store!(p, 1f0); nothing)
store_div(p::Ptr{Int}, x::Int) = (unsafe_store!(p, div(10, x)); nothing)

step("native", add_one, Tuple{Int}, GPUCompiler.NativeCompilerTarget())
if :jlruntime in fieldnames(GPUCompiler.NativeCompilerTarget)
    step("native_jlruntime", add_one, Tuple{Int},
         GPUCompiler.NativeCompilerTarget(; jlruntime=true))
else
    report("native_jlruntime", "skip")
end
# since v1.18, PTX code generation uses the back-end from this (weak) dependency
if Base.find_package("NVPTX_LLVM_Backend_jll") !== nothing
    @eval using NVPTX_LLVM_Backend_jll
end
ptx = GPUCompiler.PTXCompilerTarget(; cap=v"7.0")
step("ptx", store_one, Tuple{Ptr{Float32}}, ptx; kernel=true, expect=".entry")
step("ptx_throw", store_div, Tuple{Ptr{Int},Int}, ptx; kernel=true, expect=".entry")
