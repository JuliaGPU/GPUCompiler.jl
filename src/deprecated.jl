# Deprecations scheduled for removal in the next major release.

function defs(mod::LLVM.Module)
    Base.depwarn("`GPUCompiler.defs(mod)` is deprecated; inline `filter(f -> !isdeclaration(f), collect(functions(mod)))`.",
                 :defs)
    filter(f -> !isdeclaration(f), collect(functions(mod)))
end

function decls(mod::LLVM.Module)
    Base.depwarn("`GPUCompiler.decls(mod)` is deprecated; inline `filter(f -> isdeclaration(f) && !LLVM.isintrinsic(f), collect(functions(mod)))`.",
                 :decls)
    filter(f -> isdeclaration(f) && !LLVM.isintrinsic(f), collect(functions(mod)))
end

link_library!(mod::LLVM.Module, lib::LLVM.Module) = link_library!(mod, [lib])
function link_library!(mod::LLVM.Module, libs::Vector{LLVM.Module})
    Base.depwarn("`GPUCompiler.link_library!` is deprecated; call `LLVM.link!(mod, copy(lib))` directly, or `LLVM.link!(mod, lib; only_needed=true)` with a freshly-parsed library.",
                 :link_library!)
    libs = [copy(lib) for lib in libs]
    for lib in libs
        link!(mod, lib)
    end
end

# no-op 3-arg fallback so downstream overrides that chain via
# `invoke(GPUCompiler.link_libraries!, Tuple{CompilerJob, Module,
# Vector{String}}, ...)` still resolve.
link_libraries!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                undefined_fns::Vector{String}) = return

# `true` when a downstream package has defined a 3-arg `link_libraries!`
# override for `job`, i.e. the dispatched method isn't our fallback above.
#
# Uses the same `jl_gf_invoke_lookup` path as `Core._hasmethod` rather than
# `which`, so it's safe to call from generated-function-adjacent contexts
# where `Base.get_world_counter()` returns `typemax(UInt)` and reflection
# queries like `which` / `methods` fail (see JuliaLang/julia#48611).
# All this because Enzyme.jl calls GPUCompiler.jl from a generated function.
function has_legacy_link_libraries(@nospecialize(job::CompilerJob))
    tt = Tuple{typeof(link_libraries!), typeof(job),
               LLVM.Module, Vector{String}}
    world = ccall(:jl_get_tls_world_age, UInt, ())
    m = ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), tt, nothing, world)
    return m !== nothing && (m::Method).module !== @__MODULE__
end
