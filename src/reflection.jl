import InteractiveUtils

using UUIDs
const Cthulhu = Base.PkgId(UUID("f68482b8-f384-11e8-15f7-abe071a5a75f"), "Cthulhu")


#
# syntax highlighting
#

import Highlights
using tree_sitter_llvm_jll, tree_sitter_ptx_jll, tree_sitter_spirv_jll,
      tree_sitter_gcn_jll

# tree-sitter grammars for the assembly formats we emit, keyed by the
# `source_code` of the compiler target
const highlight_languages = Dict(
    "llvm"  => tree_sitter_llvm_jll,
    "ptx"   => tree_sitter_ptx_jll,
    "spirv" => tree_sitter_spirv_jll,
    "gcn"   => tree_sitter_gcn_jll,
)

# Set to a Highlights.jl theme, or `nothing` to detect the terminal background.
const highlight_theme = Ref{Any}(nothing)
const detected_highlight_theme = Ref{Union{Nothing,String}}()
const highlight_theme_lock = ReentrantLock()

function environment_highlight_theme(colorfgbg = get(ENV, "COLORFGBG", ""))
    background = tryparse(Int, last(rsplit(colorfgbg, ';'; limit=2)))
    isnothing(background) && return
    rgb = if 0 <= background < 16
        palette = (0x000000, 0x800000, 0x008000, 0x808000,
                   0x000080, 0x800080, 0x008080, 0xc0c0c0,
                   0x808080, 0xff0000, 0x00ff00, 0xffff00,
                   0x0000ff, 0xff00ff, 0x00ffff, 0xffffff)
        color = palette[background + 1]
        ((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff) ./ 255
    elseif 16 <= background < 232
        levels = (0, 95, 135, 175, 215, 255)
        color = background - 16
        (levels[color ÷ 36 + 1], levels[(color ÷ 6) % 6 + 1], levels[color % 6 + 1]) ./ 255
    elseif 232 <= background < 256
        fill((8 + 10 * (background - 232)) / 255, 3)
    else
        return
    end
    light_background(rgb) ? "Monokai Pro Light" : "Monokai Pro"
end

light_background(rgb) = 0.299 * rgb[1] + 0.587 * rgb[2] + 0.114 * rgb[3] > 0.5

function terminal_background(reply)
    match = Base.match(r"\e\]11;rgb:([0-9a-f]{1,4})/([0-9a-f]{1,4})/([0-9a-f]{1,4})(?:\a|\e\\)"i,
                       reply)
    isnothing(match) && return
    map(match.captures) do component
        parse(Int, component; base=16) / (16^length(component) - 1)
    end
end

function terminal_highlight_theme(reply)
    rgb = terminal_background(reply)
    isnothing(rgb) && return
    light_background(rgb) ? "Monokai Pro Light" : "Monokai Pro"
end

function repl_terminal(io)
    repl = isdefined(Base, :active_repl) ? Base.active_repl : nothing
    hasproperty(repl, :t) || return
    terminal = repl.t
    terminal isa Base.Terminals.TTYTerminal || return
    only(Base.unwrapcontext(io)) === terminal.out_stream || return
    terminal.in_stream isa Base.TTY || return
    terminal
end

function query_terminal_theme(terminal; timeout=0.1)
    input = terminal.in_stream
    Base.Terminals.raw!(terminal, true) || return
    try
        Base.start_reading(input)
        print(terminal.out_stream, "\e]11;?\a")
        flush(terminal.out_stream)

        reply = UInt8[]
        deadline = time() + timeout
        while time() < deadline
            timedwait(() -> bytesavailable(input) > 0, max(0, deadline - time()); pollint=0.005) ===
                :timed_out && break
            append!(reply, readavailable(input))
            theme = terminal_highlight_theme(String(reply))
            isnothing(theme) || return theme
        end
    finally
        Base.stop_reading(input)
        Base.Terminals.raw!(terminal, false)
    end
end

function selected_highlight_theme(io)
    theme = highlight_theme[]
    isnothing(theme) || return theme

    theme = environment_highlight_theme()
    isnothing(theme) || return theme

    terminal = repl_terminal(io)
    isnothing(terminal) && return "Monokai Pro"
    lock(highlight_theme_lock) do
        if !isassigned(detected_highlight_theme)
            detected_highlight_theme[] = query_terminal_theme(terminal)
        end
        something(detected_highlight_theme[], "Monokai Pro")
    end
end

function highlight(io::IO, code, lexer)
    have_color = get(io, :color, false)
    language = get(highlight_languages, lexer, nothing)
    if !have_color || language === nothing
        print(io, code)
        return
    end

    # render to a buffer first so that highlighting failures don't result in
    # partial output
    buf = IOBuffer()
    try
        Highlights.highlight(buf, MIME("text/ansi"), code, language,
                             selected_highlight_theme(io))
        write(io, take!(buf))
    catch err
        @warn "Failed to highlight $lexer code" exception=(err, catch_backtrace()) maxlog=1
        print(io, code)
    end
    return
end

#
# Compat shims
#

include("reflection_compat.jl")

#
# code_* replacements
#

function code_lowered(@nospecialize(job::CompilerJob); kwargs...)
    sig = job.source.specTypes  # XXX: can we just use the method instance?
    code_lowered_by_type(sig; kwargs...)
end

function code_typed(@nospecialize(job::CompilerJob); interactive::Bool=false, kwargs...)
    sig = job.source.specTypes  # XXX: can we just use the method instance?
    if interactive
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = get(Base.loaded_modules, Cthulhu, nothing)
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        interp = get_interpreter(job)
        descend_code_typed = getfield(mod, :descend_code_typed)
        descend_code_typed(sig; interp, kwargs...)
    else
        interp = get_interpreter(job)
        Base.code_typed_by_type(sig; interp, kwargs...)
    end
end

function code_warntype(io::IO, @nospecialize(job::CompilerJob); interactive::Bool=false, kwargs...)
    sig = job.source.specTypes  # XXX: can we just use the method instance?
    if interactive
        @assert io == stdout
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = get(Base.loaded_modules, Cthulhu, nothing)
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")

        interp = get_interpreter(job)
        descend_code_warntype = getfield(mod, :descend_code_warntype)
        descend_code_warntype(sig; interp, kwargs...)
    else
        interp = get_interpreter(job)
        code_warntype_by_type(io, sig; interp, kwargs...)
    end
end
code_warntype(@nospecialize(job::CompilerJob); kwargs...) = code_warntype(stdout, job; kwargs...)

InteractiveUtils.code_lowered(err::InvalidIRError; kwargs...) = code_lowered(err.job; kwargs...)
InteractiveUtils.code_typed(err::InvalidIRError; kwargs...) = code_typed(err.job; kwargs...)
InteractiveUtils.code_warntype(err::InvalidIRError; kwargs...) = code_warntype(err.job; kwargs...)

InteractiveUtils.code_lowered(err::KernelError; kwargs...) = code_lowered(err.job; kwargs...)
InteractiveUtils.code_typed(err::KernelError; kwargs...) = code_typed(err.job; kwargs...)
InteractiveUtils.code_warntype(err::KernelError; kwargs...) = code_warntype(err.job; kwargs...)

struct jl_llvmf_dump
    TSM::LLVM.API.LLVMOrcThreadSafeModuleRef
    F::LLVM.API.LLVMValueRef
@static if VERSION >= v"1.14.0-DEV.1823" # JuliaLang/julia#60698
    dump::Ptr{Nothing}
    jl_llvmf_dump(TSM::LLVM.API.LLVMOrcThreadSafeModuleRef, F::LLVM.API.LLVMValueRef) = new(TSM, F, C_NULL)
end
end

"""
    code_llvm([io], job; optimize=job.config.optimize, raw=false, dump_module=false)

Prints the device LLVM IR generated for the given compiler job to `io` (default `stdout`).

The following keyword arguments are supported:

- `optimize`: determines if the code is optimized, which includes kernel-specific
  optimizations if `kernel` is true
- `raw`: return the raw IR including all metadata
- `dump_module`: display the entire module instead of just the function

See also: [`@device_code_llvm`](@ref), `InteractiveUtils.code_llvm`
"""
function code_llvm(io::IO, @nospecialize(job::CompilerJob); optimize::Bool=job.config.optimize, raw::Bool=false,
                   debuginfo::Symbol=:default, dump_module::Bool=false, kwargs...)
    # NOTE: jl_dump_function_ir supports stripping metadata, so don't do it in the driver
    config = CompilerConfig(job.config; validate=false, strip=false, optimize)
    str = JuliaContext() do ctx
        ir, meta = compile(:llvm, CompilerJob(job; config))
        ts_mod = ThreadSafeModule(ir)
        entry_fn = meta.entry
        GC.@preserve ts_mod entry_fn begin
            value = Ref(jl_llvmf_dump(ts_mod.ref, entry_fn.ref))
            ccall(:jl_dump_function_ir, Ref{String},
                    (Ptr{jl_llvmf_dump}, Bool, Bool, Ptr{UInt8}),
                    value, !raw, dump_module, debuginfo)
        end
    end
    highlight(io, str, "llvm")
end
code_llvm(@nospecialize(job::CompilerJob); kwargs...) = code_llvm(stdout, job; kwargs...)

"""
    code_native([io], f, types; cap::VersionNumber, kernel=false, raw=false)

Prints the native assembly generated for the given compiler job to `io` (default `stdout`).

The following keyword arguments are supported:

- `cap` which device to generate code for
- `kernel`: treat the function as an entry-point kernel
- `raw`: return the raw code including all metadata

See also: [`@device_code_native`](@ref), `InteractiveUtils.code_llvm`
"""
function code_native(io::IO, @nospecialize(job::CompilerJob);
                     raw::Bool=false, dump_module::Bool=false)
    config = CompilerConfig(job.config; strip=!raw, only_entry=!dump_module, validate=false)
    asm, meta = JuliaContext() do ctx
        compile(:asm, CompilerJob(job; config))
    end
    highlight(io, asm, source_code(job.config.target))
end
code_native(@nospecialize(job::CompilerJob); kwargs...) =
    code_native(stdout, job; kwargs...)


#
# @device_code_* functions
#

function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        # we only want to invoke the hook once for every compilation job
        jobs = Set()
        function outer_hook(job)
            if !in(job, jobs)
                # the user hook might invoke the compiler again, so disable the hook
                old_hook = $compile_hook[]
                try
                    $compile_hook[] = nothing
                    $inner_hook(job; $(map(esc, user_kwargs)...))
                finally
                    $compile_hook[] = old_hook
                end
                push!(jobs, job)
            end
        end

        # now invoke the user code with this hook in place
        try
            $compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            $compile_hook[] = nothing
        end

        if isempty(jobs)
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

"""
    @device_code_lowered ex

Evaluates the expression `ex` and returns the result of
`InteractiveUtils.code_lowered` for every compiled GPU kernel.

See also: `InteractiveUtils.@code_lowered`
"""
macro device_code_lowered(ex...)
    quote
        buf = Any[]
        function hook(job::CompilerJob)
            append!(buf, code_lowered(job))
        end
        $(emit_hooked_compilation(:hook, ex...))
        buf
    end
end

"""
    @device_code_typed ex

Evaluates the expression `ex` and returns the result of
`InteractiveUtils.code_typed` for every compiled GPU kernel.

See also: `InteractiveUtils.@code_typed`
"""
macro device_code_typed(ex...)
    quote
        output = Dict{CompilerJob,Any}()
        function hook(job::CompilerJob; kwargs...)
            output[job] = code_typed(job; kwargs...)
        end
        $(emit_hooked_compilation(:hook, ex...))
        output
    end
end

"""
    @device_code_warntype [io::IO=stdout] ex

Evaluates the expression `ex` and prints the result of
`InteractiveUtils.code_warntype` to `io` for every compiled GPU kernel.

See also: `InteractiveUtils.@code_warntype`
"""
macro device_code_warntype(ex...)
    function hook(job::CompilerJob; io::IO=stdout, kwargs...)
        println(io, "$job")
        println(io)
        code_warntype(io, job; kwargs...)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_llvm [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of `InteractiveUtils.code_llvm`
to `io` for every compiled GPU kernel. For other supported keywords, see
[`GPUCompiler.code_llvm`](@ref).

See also: InteractiveUtils.@code_llvm
"""
macro device_code_llvm(ex...)
    function hook(job::CompilerJob; io::IO=stdout, kwargs...)
        println(io, "; $job")
        code_llvm(io, job; kwargs...)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_native [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`GPUCompiler.code_native`](@ref) to `io`
for every compiled GPU kernel. For other supported keywords, see
[`GPUCompiler.code_native`](@ref).
"""
macro device_code_native(ex...)
    function hook(job::CompilerJob; io::IO=stdout, kwargs...)
        println(io, "// $job")
        println(io)
        code_native(io, job; kwargs...)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code dir::AbstractString=... [...] ex

Evaluates the expression `ex` and dumps all intermediate forms of code to the directory
`dir`.
"""
macro device_code(ex...)
    localUnique = 1
    function hook(job::CompilerJob; dir::AbstractString)
        name = job.source.def.name
        fn = "$(name)_$(localUnique)"
        mkpath(dir)

        open(joinpath(dir, "$fn.lowered.jl"), "w") do io
            code = only(code_lowered(job))
            println(io, code)
        end

        open(joinpath(dir, "$fn.typed.jl"), "w") do io
            code = only(code_typed(job; debuginfo=:source))
            println(io, code)
        end

        open(joinpath(dir, "$fn.unopt.ll"), "w") do io
            code_llvm(io, job; dump_module=true, raw=true, optimize=false)
        end

        open(joinpath(dir, "$fn.opt.ll"), "w") do io
            code_llvm(io, job; dump_module=true, raw=true, optimize=true)
        end

        open(joinpath(dir, "$fn.asm"), "w") do io
            code_native(io, job; dump_module=true, raw=true)
        end

        localUnique += 1
    end
    emit_hooked_compilation(hook, ex...)
end
