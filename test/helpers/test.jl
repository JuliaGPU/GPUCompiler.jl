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
