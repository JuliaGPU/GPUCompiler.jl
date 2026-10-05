# Interactive reflection (`interactive=true`) with Cthulhu.jl.
module CthulhuExt

using GPUCompiler: GPUInterpreter
import Cthulhu

# Cthulhu 3 needs a provider for every interpreter it descends with; the default one
# wraps the interpreter like Cthulhu 2 did implicitly
if isdefined(Cthulhu, :AbstractProvider)
    Cthulhu.AbstractProvider(interp::GPUInterpreter) = Cthulhu.DefaultProvider(interp)
end

end
