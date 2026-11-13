# Packages that add methods to `Base` functions invalidate code that calls those functions
# with arguments of unknown type. When that code is ours, the compiler is recompiled for
# the first kernel after loading such a package, so check that a package known for its
# invalidations leaves the methods of GPUCompiler alone.
#
# This doesn't cover Julia's inference code as specialized for `GPUInterpreter`, which
# Julia versions invalidate through call sites of their own. On Julia 1.11 and older,
# StaticArrays still invalidates some of our methods (the 1.10 code cache, and the
# definition of runtime methods).
VERSION >= v"1.12-" && @testset "invalidations" begin
    script = raw"""
        using GPUCompiler
        logged = ccall(:jl_debug_method_invalidation, Any, (Cint,), 1)
        using StaticArrays
        ccall(:jl_debug_method_invalidation, Any, (Cint,), 0)

        function owned(mod::Module)
            while true
                mod === GPUCompiler && return true
                parent = parentmodule(mod)
                parent === mod && return false
                mod = parent
            end
        end
        seen = Set{Core.MethodInstance}()
        for item in logged
            item isa Core.CodeInstance && (item = item.def)
            item isa Core.MethodInstance || continue
            def = item.def
            def isa Method && owned(def.module) || continue
            item in seen && continue
            push!(seen, item)
            println(item)
        end
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`
    @test readchomp(cmd) == ""
end
