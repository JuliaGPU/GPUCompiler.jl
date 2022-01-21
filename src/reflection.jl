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

ptx_instructions = ["abs", "activemask", "add", "addc", "alloca", "and",
                          "applypriority", "atom", "bar", "barrier", "bfe", "bfi",
                          "bfind", "bmsk", "bra", "brev", "brkpt", "brx", "call", "clz",
                          "cnot", "copysign", "cos", "cp", "createpolicy", "cvt", "cvta",
                          "discard", "div", "dp2a", "dp4a", "ex2", "exit", "fence",
                          "fma", "fns", "isspacep", "istypep", "ld", "ldmatrix", "ldu",
                          "lg2", "lop3", "mad", "mad24", "madc", "match", "max", "mbarrier",
                          "membar", "min", "mma", "mov", "mul", "mul24", "nanosleep", "neg",
                          "not", "or", "pmevent", "popc", "prefetch", "prefetchu", "prmt",
                          "rcp", "red", "redux", "rem", "ret", "rsqrt", "sad", "selp",
                          "set", "setp", "shf", "shfl", "shl", "shr", "sin", "slct", "sqrt",
                          "st", "stackrestore", "stacksave", "sub", "subc", "suld", "suq",
                          "sured", "sust", "szext", "tanh", "testp", "tex", "tld4", "trap",
                          "txq", "vabsdiff", "vabsdiff2", "vabsdiff4", "vadd", "vadd2", "vadd4",
                          "vavrg2", "vavrg4", "vmad", "vmax", "vmax2", "vmax4", "vmin", "vmin2",
                          "vmin4", "vote", "vset", "vset2", "vset4", "vshl", "vshr", "vsub",
                          "vsub2", "vsub4", "wmma", "xor"]

r_ptx_instruction = join(ptx_instructions, "|")

types = ["s8", "s16", "s32", "s64", "u8,", "u16,", "u32", "u64", "f16", "f16x2", "f32", "f64", "b8,", "b16", "b32", "b64", "pred"]
r_types = join(types, "|")


# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-and-bit-size-comparisons
 operators_comparison_sint = ["eq", "ne", "lt", "le", "gt", "ge"]
 operators_comparison_uint = ["eq", "ne", "lo", "ls", "hi", "hs"]
 operators_comparison_bit  = ["eq", "ne"]

 operators_comparison_float = ["eq", "ne", "lt", "le", "gt", "ge"]
 operators_comparison_nanfloat = ["equ", "neu", "ltu", "leu", "gtu", "geu"]
 operators_comparison_nan = ["num", "nan"]

 modifiers_int = ["rni", "rzi", "rmi", "rpi"]
 modifiers_float = ["rn", "rna", "rz", "rm", "rp"]
 modifiers = sort(unique([modifiers_int...,modifiers_float...]))

 state_spaces = ["reg", "sreg", "const", "global", "local", "param", "shared", "tex"]


 operators = sort(unique([operators_comparison_sint..., operators_comparison_uint...,
                   operators_comparison_bit..., operators_comparison_float...,
                   operators_comparison_nanfloat..., operators_comparison_nan...,
                   modifiers..., state_spaces..., types...]))


r_operators = join(operators, "|")

# We can divide into types of instructions as all combinations of instructions, types and operators are not valid.
r_instruction = "(?:(?:$r_ptx_instruction)\\.(?:(?:$r_operators)(?:\\.))?(?:$(r_types)))"

directives = ["address_size", "align", "branchtargets", "callprototype",
              "calltargets", "const", "entry", "extern", "file", "func", "global",
              "loc", "local", "maxnctapersm", "maxnreg", "maxntid",
              "minnctapersm", "param", "pragma", "reg", "reqntid", "section",
              "shared", "sreg", "target", "tex", "version", "visible", "weak"]

r_directive = join(directives, "|")


r_hex = "0[xX][A-F]+U?"
r_octal = "0[0-8]+U?"
r_binary = "0[bB][01]+U?"
r_decimal = "[0-9]+U?"
r_float = "0[fF]{hexdigit}{8}"
r_double = "0[dD]{hexdigit}{16}"

r_number = join(map(x -> "(?:" * x * ")", [r_hex, r_octal, r_binary, r_decimal, r_float, r_double]), "|")

r_register_special = ["%clock", "%clock64", "%clock_hi", "%ctaid", "%dynamic_smem_size", "%envreg\\d{0,2}", # envreg0-31
                      "%globaltimer", "%globaltimer_hi", "%globaltimer_lo,", "%gridid", "%laneid", "%lanemask_eq",
                      "%lanemask_ge", "%lanemask_gt", "%lanemask_le", "%lanemask_lt", "%nctaid", "%nsmid",
                      "%ntid", "%nwarpid", "%pm\\d,", "%pm\\d_64", "%reserved_smem_offset<2>",
                      "%reserved_smem_offset_begin", "%reserved_smem_offset_cap", "%reserved_smem_offset_end", "%smid",
                      "%tid", "%total_smem_size", "%warpid", "%\\w{1,2}\\d{0,2}"]

r_register = join(r_register_special, "|")


r_followsym = "[a-zA-Z0-9_\$]"
r_identifier=  "[a-zA-Z]{$r_followsym}* | {[_\$%]{$r_followsym}+"

r_guard_predicate = "@!?%p\\d{0,2}"
r_label = "[\\w_]+:"
r_comment = "//"
r_unknown = "[^\\s]*"

r_line = "(?:(?:.$r_directive)|(?:$r_instruction)|(?:$r_register)|(?:$r_number)|(?:$r_label)|(?:$r_guard_predicate)|(?:$r_comment)|(?:$r_identifier)|(?:$r_unknown))"

get_token(n::Nothing) = nothing, nothing, nothing

# simple regex-based highlighter
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
function highlight_ptx(io::IO, code::AbstractString)
    function get_token(s)
        m = match(Regex("^(\\s*)($r_line)([^\\w\\d]+.*)?"), s)
        m !== nothing && (return m.captures[1:3])
        return nothing, nothing, nothing
    end
    get_token(n::Nothing) = nothing, nothing, nothing
    print_tok(token, type) = Base.printstyled(io,
                                              token,
                                              bold = hlscheme[type][1],
                                              color = hlscheme[type][2])
    code = IOBuffer(code)
    while !eof(code)
        line = readline(code)
        indent, tok, line = get_token(line)
        is_tok(regex) = match(Regex("^(" * regex * ")"), tok) !== nothing
        while (tok !== nothing)
            print(io, indent)
            if is_tok(r_comment)
                print_tok(tok, :comment)
                print_tok(line, :comment)
                break
            elseif is_tok(r_label)
                print_tok(tok, :label)
            elseif is_tok(r_instruction)
                print_tok(tok, :instruction)
            elseif is_tok(r_directive)
                print_tok(tok, :type)
            elseif is_tok(r_guard_predicate)
                print_tok(tok, :keyword)
            elseif is_tok(r_register)
                print_tok(tok, :number)
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
