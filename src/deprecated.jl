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
