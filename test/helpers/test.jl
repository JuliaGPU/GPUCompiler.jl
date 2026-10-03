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

# helper function for sinking a value to prevent the callee from getting optimized away:
# a volatile round trip through a stack slot (in the given address space)
using LLVM, LLVM.IR, LLVM.Build, LLVM.Interop
@inline @llvmgenerated builder function sink(i::T, ::Val{addrspace}=Val(0))::T where {
        T <: Union{Int32,UInt32,Int64,UInt64}, addrspace}
    slot = alloca!(builder, i.value_type, "slot"; addrspace)
    store!(builder, i, slot).volatile = true
    value = load!(builder, i.value_type, slot, "value")
    value.volatile = true
    value
end

# typed/opaque pointer detection for conditional FileCheck checks

const typed_ptrs = JuliaContext() do ctx
    supports_typed_pointers(ctx)
end
const opaque_ptrs = !typed_ptrs
