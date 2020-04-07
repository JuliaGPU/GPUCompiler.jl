abstract type AbstractCompilerTarget end
# expected fields:
# - triple::String
# - datalayout::String
# - runtime_module::Base.Module
# - link_libraries::Union{Nothing,Function} (::AbstractCompilerJob, ::LLVM.Module, ::Vector{String})


#
# PTX
#

export PTXCompilerTarget

Base.@kwdef struct PTXCompilerTarget <: AbstractCompilerTarget
    triple::String = ifelse(Int===Int64, "nvptx64-nvidia-cuda", "nvptx-nvidia-cuda")
    datalayout::String = ifelse(Int===Int64,
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
        "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")
    runtime_module::Base.Module
    link_libraries::Union{Nothing,Function}
end
