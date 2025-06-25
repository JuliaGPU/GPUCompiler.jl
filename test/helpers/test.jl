# @test_throw, with additional testing for the exception message
macro test_throws_message(f, typ, ex...)
    quote
        msg = ""
        @test_throws $(esc(typ)) try
            $(esc(ex...))
        catch err
            msg = sprint(showerror, err)
            rethrow()
        end

        if !$(esc(f))(msg)
            # @test should return its result, but doesn't
            errmsg = "Failed to validate error message\n" * msg
            @error errmsg
        end
        @test $(esc(f))(msg)
    end
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline @generated function sink(i::T, ::Val{addrspace}=Val(0)) where {T <: Union{Int32,UInt32}, addrspace}
    as_str = addrspace > 0 ? " addrspace($addrspace)" : ""
    llvmcall_str = """%slot = alloca i32$(addrspace > 0 ? ", addrspace($addrspace)" : "")
                     store volatile i32 %0, i32$(as_str)* %slot
                     %value = load volatile i32, i32$(as_str)* %slot
                     ret i32 %value"""
    return :(Base.llvmcall($llvmcall_str, T, Tuple{T}, i))
end
@inline @generated function sink(i::T, ::Val{addrspace}=Val(0)) where {T <: Union{Int64,UInt64}, addrspace}
    as_str = addrspace > 0 ? " addrspace($addrspace)" : ""
    llvmcall_str = """%slot = alloca i64$(addrspace > 0 ? ", addrspace($addrspace)" : "")
                     store volatile i64 %0, i64$(as_str)* %slot
                     %value = load volatile i64, i64$(as_str)* %slot
                     ret i64 %value"""
    return :(Base.llvmcall($llvmcall_str, T, Tuple{T}, i))
end

# filecheck utils

module FileCheck
    import LLVM_jll
    import IOCapture
    using GPUCompiler, LLVM

    export filecheck, @filecheck, @check_str

    global filecheck_path::String
    function __init__()
        global filecheck_path = joinpath(LLVM_jll.artifact_dir, "tools", "FileCheck")
    end

    function filecheck_exe(; adjust_PATH::Bool=true, adjust_LIBPATH::Bool=true)
        env = Base.invokelatest(
            LLVM_jll.JLLWrappers.adjust_ENV!,
            copy(ENV),
            LLVM_jll.PATH[],
            LLVM_jll.LIBPATH[],
            adjust_PATH,
            adjust_LIBPATH
        )

        return Cmd(Cmd([filecheck_path]); env)
    end

    const julia_typed_pointers = JuliaContext() do ctx
        supports_typed_pointers(ctx)
    end

    function filecheck(f, input)
        # FileCheck assumes that the input is available as a file
        mktemp() do path, input_io
            write(input_io, input)
            close(input_io)

            # capture the output of `f` and write it into a temporary buffer
            result = IOCapture.capture(rethrow=Union{}) do
                f(input)
            end
            output_io = IOBuffer()
            write(output_io, result.output)
            println(output_io)

            if result.error
                # if the function errored, also render the exception and backtrace
                showerror(output_io, result.value, result.backtrace)
            elseif result.value !== nothing
                # also show the returned value; some APIs don't print
                write(output_io, string(result.value))
            end

            # determine some useful prefixes for FileCheck
            prefixes = ["CHECK",
                        "JULIA$(VERSION.major)_$(VERSION.minor)",
                        "LLVM$(Base.libllvm_version.major)"]
            ## whether we use typed pointers or opaque pointers
            if julia_typed_pointers
                push!(prefixes, "TYPED")
            else
                push!(prefixes, "OPAQUE")
            end
            ## whether we pass pointers as integers or as actual pointers
            if VERSION >= v"1.12.0-DEV.225"
                push!(prefixes, "PTR_ABI")
            else
                push!(prefixes, "INTPTR_ABI")
            end

            # now pass the collected output to FileCheck
            seekstart(output_io)
            filecheck_io = Pipe()
            cmd = ```$(filecheck_exe())
                     --color
                     --allow-unused-prefixes
                     --check-prefixes $(join(prefixes, ','))
                     $path```
            proc = run(pipeline(ignorestatus(cmd); stdin=output_io, stdout=filecheck_io, stderr=filecheck_io); wait=false)
            close(filecheck_io.in)

            # collect the output of FileCheck
            reader = Threads.@spawn String(read(filecheck_io))
            Base.wait(proc)
            log = strip(fetch(reader))

            # error out if FileCheck did not succeed.
            # otherwise, return true so that `@test @filecheck` works as expected.
            if !success(proc)
                error(log)
            end
            return true
        end
    end

    # collect checks used in the @filecheck block by piggybacking on macro expansion
    const checks = String[]
    macro check_str(str)
        push!(checks, str)
        nothing
    end

    macro filecheck(ex)
        ex = Base.macroexpand(__module__, ex)
        if isempty(checks)
            error("No checks provided within the @filecheck macro block")
        end
        check_str = join(checks, "\n")
        empty!(checks)

        esc(quote
            filecheck($check_str) do _
                $ex
            end
        end)
    end
end
