using Distributed, Test, GPUCompiler, LLVM

using SPIRV_LLVM_Backend_jll, SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

# include all helpers
include(joinpath(@__DIR__, "helpers", "runtime.jl"))
for file in readdir(joinpath(@__DIR__, "helpers"))
    if endswith(file, ".jl") && file != "runtime.jl"
        include(joinpath(@__DIR__, "helpers", file))
    end
end
using .FileCheck


## entry point

function runtests(f, name)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, GPUCompiler)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            GC.gc(true)
            Random.seed!(1)

            @timed @testset $"$name" begin
                $f()
            end
        end
        data = Core.eval(mod, ex)
        #data[1] is the testset

        # process results
        cpu_rss = Sys.maxrss()
        if VERSION >= v"1.11.0-DEV.1529"
            tc = Test.get_test_counts(data[1])
            passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
                tc.passes, tc.fails, tc.errors, tc.broken, tc.cumulative_passes,
                tc.cumulative_fails, tc.cumulative_errors, tc.cumulative_broken
        else
            passes,fails,errors,broken,c_passes,c_fails,c_errors,c_broken =
                Test.get_test_counts(data[1])
        end
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5])
        end
        res = vcat(collect(data), cpu_rss)

        GC.gc(true)
        res
    finally
        Test.TESTSET_PRINT_ENABLE[] = old_print_setting
    end
end

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
