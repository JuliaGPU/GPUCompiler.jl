abstract type AbstractCompilerTarget end
# expected fields:
# - mod::Base.Module
# - link_libraries::Union{Nothing,Function} (::AbstractCompilerJob, ::LLVM.Module, ::Vector{String})


#
# PTX
#

export PTXCompilerTarget

struct PTXCompilerTarget <: AbstractCompilerTarget
    mod::Base.Module
    link_libraries::Union{Nothing,Function}
end
