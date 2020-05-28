@testset "Compilation" begin
    function native_compile(kind::Symbol, func, types, kernel=true; kwargs...)
        source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
        target = NativeCompilerTarget()
        params = TestCompilerParams()
        job = CompilerJob(target, source, params)
        return GPUCompiler.compile(kind, job; kwargs...)
    end

    kernel() = nothing

    llvm_output = native_compile(:llvm, kernel, ())
    @test length(llvm_output) == 2
    @test llvm_output[1] isa LLVM.Module
    @test llvm_output[2] isa LLVM.Function

    asm_output = native_compile(:asm, kernel, ())
    obj_output = native_compile(:obj, kernel, ())
    @test length(asm_output) == length(obj_output) == 4
    @test typeof(asm_output[1]) == typeof(obj_output[1]) == String
    @test typeof(asm_output[2]) == typeof(obj_output[2]) == String
    @test occursin("kernel", asm_output[2])
    @test length(asm_output[3]) == 0
    @test length(asm_output[4]) == 0

    @testset "Undefined Functions" begin
        function undef_fn()
            ccall("extern somefunc", llvmcall, Cvoid, ())
            nothing
        end

        asm_output = native_compile(:asm, undef_fn, (); strict=false)
        @test length(asm_output[3]) == 1
        @test asm_output[3][1] == "somefunc"
    end

    @testset "Undefined Globals" begin
        @generated function makegbl()
            T = Int64
            T_gbl = convert(LLVMType, T)
            T_ptr = convert(LLVMType, Ptr{T})
            llvm_f, _ = create_function(T_ptr)
            mod = LLVM.parent(llvm_f)
            gvar = GlobalVariable(mod, T_gbl, "someglobal")
            Builder(JuliaContext()) do builder
                entry = BasicBlock(llvm_f, "entry", JuliaContext())
                position!(builder, entry)
                result = ptrtoint!(builder, gvar, T_ptr)
                ret!(builder, result)
            end
            call_function(llvm_f, Ptr{T})
        end
        function undef_gbl()
            ptr = makegbl()
            Base.unsafe_store!(ptr, 1)
            nothing
        end

        asm_output = native_compile(:asm, undef_gbl, (); strict=false)
        @test length(asm_output[4]) == 1
        @test asm_output[4][1][1] == "someglobal"
        @test eltype(asm_output[4][1][2]) isa LLVM.IntegerType
    end
end
