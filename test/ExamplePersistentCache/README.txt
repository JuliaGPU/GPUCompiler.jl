Persistent Cache api:

GPUCompiler.ci_cache_snapshot() -> cache: returns a snapshot of GLOBAL_CI_CACHES used 
as a base point for what will be persistently cached.

GPUCompiler.ci_cache_delta(snapshot::cache) -> cache: takes a snapshot and returns
the cache that represents the difference between (current GLOBAL_CI_CACHES - snapshot)

GPUCompiler.ci_cache_insert(snapshot::cache): inserts snapshot into GLOBAL_CI_CACHES


Usage:
snapshot = GPUCompiler.ci_cache_snapshot()
... precompile work ...
const persistent_cache = GPUCompiler.ci_cache_delta(snapshot)

function __init__()
    GPUCompiler.ci_cache_insert(persistent_cache)
    ... rest of init logic ...
end
