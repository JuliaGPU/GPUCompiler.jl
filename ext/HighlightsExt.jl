# Syntax highlighting of reflection output through Highlights.jl.
#
# This lives in an extension because Highlights.jl's dependency tree (TreeSitter.jl, and
# through it JSON) invalidates enough Base code to noticeably slow down loading of
# packages that depend on GPUCompiler, while highlighting only matters interactively.
module HighlightsExt

using GPUCompiler
import Highlights

render(buf::IO, code, language, theme) =
    Highlights.highlight(buf, MIME("text/ansi"), code, language, theme)

function __init__()
    GPUCompiler.highlighter[] = render
end

end # module HighlightsExt
