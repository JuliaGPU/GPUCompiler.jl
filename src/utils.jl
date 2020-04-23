export tbaa_make_child

function tbaa_make_child(name::String, constant::Bool=false; ctx::LLVM.Context=JuliaContext())
    tbaa_root = MDNode([MDString("gputbaa", ctx)], ctx)
    tbaa_struct_type =
        MDNode([MDString("gputbaa_$name", ctx),
                tbaa_root,
                LLVM.ConstantInt(0, ctx)], ctx)
    tbaa_access_tag =
        MDNode([tbaa_struct_type,
                tbaa_struct_type,
                LLVM.ConstantInt(0, ctx),
                LLVM.ConstantInt(constant ? 1 : 0, ctx)], ctx)

    return tbaa_access_tag
end


defs(mod::LLVM.Module)  = filter(f -> !isdeclaration(f), collect(functions(mod)))
decls(mod::LLVM.Module) = filter(f ->  isdeclaration(f) && intrinsic_id(f) == 0,
                                 collect(functions(mod)))

