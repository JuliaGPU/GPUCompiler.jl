using Aqua

Aqua.test_all(GPUCompiler;
    stale_deps=(ignore=[:Tracy],),)
