import InteractiveUtils

using UUIDs
const Cthulhu = Base.PkgId(UUID("f68482b8-f384-11e8-15f7-abe071a5a75f"), "Cthulhu")


#
# syntax highlighting
#

# https://github.com/JuliaLang/julia/blob/dacd16f068fb27719b31effbe8929952ee2d5b32/stdlib/InteractiveUtils/src/codeview.jl
const hlscheme = Dict{Symbol, Tuple{Bool, Union{Symbol, Int}}}(
    :default     => (false, :normal), # e.g. comma, equal sign, unknown token
    :comment     => (false, :light_black),
    :label       => (false, :light_red),
    :instruction => ( true, :light_cyan),
    :type        => (false, :cyan),
    :number      => (false, :yellow),
    :bracket     => (false, :yellow),
    :variable    => (false, :normal), # e.g. variable, register
    :keyword     => (false, :light_magenta),
    :funcname    => (false, :light_yellow),
)

function highlight(io::IO, code, lexer)
    if !haskey(io, :color)
        print(io, code)
    elseif lexer == "llvm"
        InteractiveUtils.print_llvm(io, code)
    elseif lexer == "ptx"
        highlight_ptx(io, code)
    else
        print(io, code)
    end
end

const ptx_instructions = [
    "abs", "cvt", "min", "shfl", "vadd", "activemask", "cvta", "mma", "shl", "vadd2",
    "add", "discard", "mov", "shr", "vadd4", "addc", "div", "mul", "sin", "vavrg2",
    "alloca", "dp2a", "mul24", "slct", "vavrg4", "and", "dp4a", "nanosleep", "sqrt",
    "vmad", "applypriority", "ex2", "neg", "st", "vmax", "atom", "exit", "not",
    "stackrestore", "vmax2", "bar", "fence", "or", "stacksave", "vmax4", "barrier",
    "fma", "pmevent", "sub", "vmin", "bfe", "fns", "popc", "subc", "vmin2", "bfi",
    "isspacep", "prefetch", "suld", "vmin4", "bfind", "istypep", "prefetchu", "suq",
    "vote", "bmsk", "ld", "prmt", "sured", "vset", "bra", "ldmatrix", "rcp", "sust",
    "vset2", "brev", "ldu", "red", "szext", "vset4", "brkpt", "lg2", "redux", "tanh",
    "vshl", "brx", "lop3", "rem", "testp", "vshr", "call", "mad", "ret", "tex", "vsub",
    "clz", "mad24", "rsqrt", "tld4", "vsub2", "cnot", "madc", "sad", "trap", "vsub4",
    "copysign", "match", "selp", "txq", "wmma", "cos", "max", "set", "vabsdiff", "xor",
    "cp", "mbarrier", "setp", "vabsdiff2", "createpolicy", "membar", "shf", "vabsdiff4"]

# simple regex-based highlighter
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
function highlight_ptx(io::IO, code)
    function get_token(s)
        # TODO: doesn't handle `ret;`, `{1`, etc; not properly tokenizing symbols
        m = match(r"(\s*)([^\s]+)(.*)", s)
        m !== nothing && (return m.captures[1:3])
        return nothing, nothing, nothing
    end
    print_tok(token, type) = Base.printstyled(io,
                                              token,
                                              bold = hlscheme[type][1],
                                              color = hlscheme[type][2])
    buf = IOBuffer(code)
    while !eof(buf)
        line = readline(buf)
        indent, tok, line = get_token(line)
        istok(regex) = match(regex, tok) !== nothing
        isinstr() = first(split(tok, '.')) in ptx_instructions
        while (tok !== nothing)
            print(io, indent)

            # comments
            if istok(r"^\/\/")
                print_tok(tok, :comment)
                print_tok(line, :comment)
                break
            # labels
            elseif istok(r"^[\w]+:")
                print_tok(tok, :label)
            # instructions
            elseif isinstr()
                print_tok(tok, :instruction)
            # directives
            elseif istok(r"^\.[\w]+")
                print_tok(tok, :type)
            # guard predicates
            elseif istok(r"^@!?%p.+")
                print_tok(tok, :keyword)
            # registers
            elseif istok(r"^%[\w]+")
                print_tok(tok, :variable)
            # constants
            elseif istok(r"^0[xX][A-F]+U?") ||      # hexadecimal
                   istok(r"^0[0-8]+U?") ||          # octal
                   istok(r"^0[bB][01]+U?") ||       # binary
                   istok(r"^[0-9]+U?") ||           # decimal
                   istok(r"^0[fF]{hexdigit}{8}") || # single-precision floating point
                   istok(r"^0[dD]{hexdigit}{16}")   # double-precision floating point
                print_tok(tok, :number)
            # TODO: function names
            # TODO: labels as RHS
            else
                print_tok(tok, :default)
            end
            indent, tok, line = get_token(line)
        end
        print(io, '\n')
    end
end

#
# code_* replacements
#

code_lowered(@nospecialize(job::CompilerJob); kwargs...) =
    InteractiveUtils.code_lowered(job.source.f, job.source.tt; kwargs...)

function code_typed(@nospecialize(job::CompilerJob); interactive::Bool=false, kwargs...)
    # TODO: use the compiler driver to get the Julia method instance (we might rewrite it)
    if interactive
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = get(Base.loaded_modules, Cthulhu, nothing)
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        interp = get_interpreter(job)
        descend_code_typed = getfield(mod, :descend_code_typed)
        descend_code_typed(job.source.f, job.source.tt; interp, kwargs...)
    elseif VERSION >= v"1.7-"
        interp = get_interpreter(job)
        InteractiveUtils.code_typed(job.source.f, job.source.tt; interp, kwargs...)
    else
        InteractiveUtils.code_typed(job.source.f, job.source.tt; kwargs...)
    end
end

function code_warntype(io::IO, @nospecialize(job::CompilerJob); interactive::Bool=false, kwargs...)
    # TODO: use the compiler driver to get the Julia method instance (we might rewrite it)
    if interactive
        @assert io == stdout
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = get(Base.loaded_modules, Cthulhu, nothing)
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        interp = get_interpreter(job)
        descend_code_warntype = getfield(mod, :descend_code_warntype)
        descend_code_warntype(job.source.f, job.source.tt; interp, kwargs...)
    elseif VERSION >= v"1.7-"
        interp = get_interpreter(job)
        InteractiveUtils.code_warntype(io, job.source.f, job.source.tt; interp, kwargs...)
    else
        InteractiveUtils.code_warntype(io, job.source.f, job.source.tt; kwargs...)
    end
end
code_warntype(@nospecialize(job::CompilerJob); kwargs...) = code_warntype(stdout, job; kwargs...)

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
                   debuginfo::Symbol=:default, dump_module::Bool=false)
    # NOTE: jl_dump_function_ir supports stripping metadata, so don't do it in the driver
    ir, meta = codegen(:llvm, job; optimize=optimize, strip=false, validate=false)
    str = ccall(:jl_dump_function_ir, Ref{String},
                (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                meta.entry, !raw, dump_module, debuginfo)
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
function code_native(io::IO, @nospecialize(job::CompilerJob); raw::Bool=false, dump_module::Bool=false)
    asm, meta = codegen(:asm, job; strip=!raw, only_entry=!dump_module, validate=false)
    highlight(io, asm, source_code(job.target))
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
        local kernels = Set()
        function outer_hook(job)
            if !in(job, kernels)
                $inner_hook(job; $(map(esc, user_kwargs)...))
                push!(kernels, job)
            end
        end

        if $compile_hook[] !== nothing
            error("Chaining multiple @device_code calls is unsupported")
        end
        try
            $compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            $compile_hook[] = nothing
        end

        if isempty(kernels)
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

Evaluates the expression `ex` and prints the result of InteractiveUtils.code_llvm
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
    only(xs) = (@assert length(xs) == 1; first(xs))
    localUnique = 1
    function hook(job::CompilerJob; dir::AbstractString)
        name = something(job.source.name, nameof(job.source.f))
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
