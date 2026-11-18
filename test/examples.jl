function find_sources(path::String, sources=String[])
    if isdir(path)
        for entry in readdir(path)
            find_sources(joinpath(path, entry), sources)
        end
    elseif endswith(path, ".jl")
        push!(sources, path)
    end
    sources
end

dir = joinpath(@__DIR__, "..", "examples")
files = find_sources(dir)
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", files)
filter!(file -> !occursin("Kaleidoscope", file), files)

cd(dir) do
    examples = relpath.(files, Ref(dir))
    @testset for example in examples
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project())`
        @test success(pipeline(`$cmd $example`, stderr=stderr))
    end
end
