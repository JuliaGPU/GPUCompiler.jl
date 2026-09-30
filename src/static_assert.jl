export @static_assert

"""
    @static_assert condition message

Require `condition` to be proven true while compiling device code. Unlike
`Base.@static`, the condition is evaluated after target-specific values have
been propagated through LLVM. A condition that remains dynamic is rejected;
this is not a runtime assertion mechanism.

`message` must be a string literal. It is reported with the device-code
backtrace if the assertion cannot be proven.
"""
macro static_assert(condition, message)
    message isa String || throw(ArgumentError("@static_assert message must be a string literal"))
    marker = static_assert_marker(message)
    return quote
        if !$(esc(condition))
            $marker
        end
        nothing
    end
end

const STATIC_ASSERT_MARKER = "gpu_static_assert"
const STATIC_ASSERTION = "static assertion failed"

function static_assert_marker(message::String)
    generate_llvmcall(Nothing, Tuple{}) do builder
        # LLVM.jl owns the pointer representation and creates an anonymous private
        # string global, just like LLVM's annotation helpers.
        string = globalstring_ptr!(builder, message)
        marker_type = LLVM.FunctionType(LLVM.VoidType(), [string.value_type])
        marker = LLVM.Function(current_module(builder), STATIC_ASSERT_MARKER, marker_type)
        call!(builder, marker_type, marker, [string])
        nothing
    end
end

function static_assert_message(inst::LLVM.CallInst)
    try
        value = only(inst.arguments)
        while value isa ConstantExpr
            value = first(value.operands)
        end
        value isa GlobalVariable || return nothing
        initializer = value.initializer
        initializer isa ConstantDataSequential || initializer isa ConstantArray || return nothing
        values = initializer.elements
        bytes = UInt8[convert(UInt8, byte) for byte in values]
        !isempty(bytes) && bytes[end] == 0x00 && pop!(bytes)
        return String(bytes)
    catch err
        err isa ArgumentError || err isa BoundsError || rethrow()
        return nothing
    end
end
