# Registry compat checks

GPUCompiler depends on Julia internals, so a release often stops working on Julia versions
that come out after it, while its registry entry keeps claiming support for them. These
scripts find such versions and fix the `julia` bounds in General.

- `matrix.jl` tests every registered version against every Julia release its `julia` bound
  claims (the first and last patch of each minor version, and all patches in between when
  those two disagree). For each pair it adds exactly that version to a fresh environment,
  loads it, and runs `smoke.jl`, which generates code for the native and PTX targets.
- `compat.jl` turns the results into narrower `julia` bounds, writes them to `Compat.toml`
  in a General checkout, and summarizes every change and the failure behind it in
  `report.md`.

A new Julia release only matters for the versions whose bound includes it: those without
an upper bound for a new minor release, and also those with a closed range for a new patch
release. So after every Julia release, test just that release, in a fresh work directory:

```
julia matrix.jl --julia==1.14.0 --workdir=/var/tmp/compat-1.14.0
julia --project=<env with RegistryTools> compat.jl --workdir=/var/tmp/compat-1.14.0 \
      --registry=<General checkout>
```

That takes minutes. Without `--julia`, `matrix.jl` audits every version against every
claimed release, which takes a few hours on a 32-core machine. Results are written as
cells finish, so an interrupted run picks up where it stopped; delete a result to test it
again.

Check the report before opening a PR against General:

- Versions failing on *every* claimed Julia version aren't changed: that usually means a
  dependency bound is too wide. Confirm by rerunning those versions with the suspected
  dependency pinned (`--only=<versions> --pin=Name@version`); `compat.jl` then caps that
  dependency for them.
- Versions marked `*` failed with other dependency versions than they passed with, so the
  dependency may be at fault rather than Julia.
- Read the logs (next to the results) for anything that looks like a problem with the
  harness rather than with GPUCompiler.
