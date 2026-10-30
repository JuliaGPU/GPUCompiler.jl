using Aqua

# the tree-sitter grammars are only loaded by the HighlightsExt extension
Aqua.test_all(GPUCompiler;
    stale_deps=(ignore=[:Tracy, :tree_sitter_gcn_jll, :tree_sitter_llvm_jll,
                        :tree_sitter_ptx_jll, :tree_sitter_spirv_jll],),)
