import InteractiveUtils

using UUIDs
const Cthulhu = Base.PkgId(UUID("f68482b8-f384-11e8-15f7-abe071a5a75f"), "Cthulhu")


#
# syntax highlighting
#

const _pygmentize = Ref{Union{String,Nothing}}()
function pygmentize()
    if !isassigned(_pygmentize)
        _pygmentize[] = Sys.which("pygmentize")
    end
    return _pygmentize[]
end

const _pygmentize_version = Ref{Union{VersionNumber, Nothing}}()
function pygmentize_version()
    isassigned(_pygmentize_version) && return _pygmentize_version[]

    pygmentize_cmd = pygmentize()
    if isnothing(pygmentize_cmd)
        return _pygmentize_version[] = nothing
    end

    cmd = `$pygmentize_cmd -V`
    @static if Sys.iswindows()
        # Avoid encoding issues with pipes on Windows by using cmd.exe to capture stdout for us
        cmd = `cmd.exe /C $cmd`
        cmd = addenv(cmd, "PYTHONUTF8" => 1)
    end
    version_str = readchomp(cmd)

    pos = findfirst("Pygments version ", version_str)
    if !isnothing(pos)
        version_start = last(pos) + 1
        version_end = findnext(',', version_str, version_start) - 1
        version = tryparse(VersionNumber, version_str[version_start:version_end])
    else
        version = nothing
    end

    if isnothing(version)
        @warn "Could not parse Pygments version:\n$version_str"
    end

    return _pygmentize_version[] = version
end

function pygmentize_support(lexer)
    highlighter_ver = pygmentize_version()
    if isnothing(highlighter_ver)
        @warn "Syntax highlighting of $lexer code relies on Pygments.\n\
               Use `pip install pygments` to install the lastest version" maxlog = 1
        return false
    elseif lexer == "ptx"
        if highlighter_ver < v"2.16"
            @warn "Pygments supports PTX highlighting starting from version 2.16\n\
                   Detected version $highlighter_ver\n\
                   Please update with `pip install pygments -U`" maxlog = 1
            return false
        end
        return true
    elseif lexer == "gcn"
        if highlighter_ver < v"2.8"
            @warn "Pygments supports GCN highlighting starting from version 2.8\n\
                   Detected version $highlighter_ver\n\
                   Please update with `pip install pygments -U`" maxlog = 1
            return false
        end
        return true
    else
        return false
    end
end

function highlight(io::IO, code, lexer)
    have_color = get(io, :color, false)
    if !have_color
        print(io, code)
    elseif lexer == "llvm"
        InteractiveUtils.print_llvm(io, code)
    elseif pygmentize_support(lexer)
        lexer = lexer == "gcn" ? "amdgpu" : lexer
        pygments_args = String[pygmentize(), "-f", "terminal", "-P", "bg=dark", "-l", lexer]
        @static if Sys.iswindows()
            # Avoid encoding issues with pipes on Windows by using a temporary file
            mktemp() do tmp_path, tmp_io
                println(tmp_io, code)
                close(tmp_io)
                push!(pygments_args, "-o", tmp_path, tmp_path)
                cmd = Cmd(pygments_args)
                wait(open(cmd))  # stdout and stderr go to devnull
                print(io, read(tmp_path, String))
            end
        else
            cmd = Cmd(pygments_args)
            pipe = open(cmd, "r+")
            print(pipe, code)
            close(pipe.in)
            print(io, read(pipe, String))
        end
    else
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
end

"""
    code_llvm([io], job; optimize=true, raw=false, dump_module=false)

Prints the device LLVM IR generated for the given compiler job to `io` (default `stdout`).

The following keyword arguments are supported:

- `optimize`: determines if the code is optimized, which includes kernel-specific
  optimizations if `kernel` is true
- `raw`: return the raw IR including all metadata
- `dump_module`: display the entire module instead of just the function

See also: [`@device_code_llvm`](@ref), `InteractiveUtils.code_llvm`
"""
function code_llvm(io::IO, @nospecialize(job::CompilerJob); optimize::Bool=true, raw::Bool=false,
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
            code_llvm(io, job; dump_module=true, raw=true)
        end

        open(joinpath(dir, "$fn.asm"), "w") do io
            code_native(io, job; dump_module=true, raw=true)
        end

        localUnique += 1
    end
    emit_hooked_compilation(hook, ex...)
end
