# Syntax highlighting of reflection output through Highlights.jl.
#
# This lives in an extension because Highlights.jl's dependency tree (TreeSitter.jl, and
# through it JSON) invalidates enough Base code to noticeably slow down loading of
# packages that depend on GPUCompiler, while highlighting only matters interactively.
module HighlightsExt

using GPUCompiler
import Highlights
using tree_sitter_llvm_jll, tree_sitter_ptx_jll, tree_sitter_spirv_jll,
      tree_sitter_gcn_jll

# tree-sitter grammars for `GPUCompiler.highlight_languages`
const languages = Dict(
    "llvm"  => tree_sitter_llvm_jll,
    "ptx"   => tree_sitter_ptx_jll,
    "spirv" => tree_sitter_spirv_jll,
    "gcn"   => tree_sitter_gcn_jll,
)

render(buf::IO, code, lexer, theme) =
    Highlights.highlight(buf, MIME("text/ansi"), code, languages[lexer], theme)

function __init__()
    GPUCompiler.highlighter[] = render
end

end # module HighlightsExt
