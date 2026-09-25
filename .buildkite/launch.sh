#!/usr/bin/env bash
# Uploads the downstream test steps. On pull requests, the agent skips each step whose
# `if_changed` patterns match none of the files changed relative to the merge-base with
# the base branch; if it cannot compute that diff, it runs every step. Pushes to branches
# and tags, and commits that select steps explicitly with `[only ...]` or `[run all]`,
# are not filtered by changed files. The `if:` conditions in `downstream.yml` apply
# either way.
set -euo pipefail

if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
    echo "Not a pull request; not selecting steps by changed files"
    export BUILDKITE_AGENT_APPLY_IF_CHANGED=false
elif [[ "${BUILDKITE_MESSAGE:-}" =~ \[(only\ |run\ all\]) ]]; then
    echo "The commit message selects steps explicitly; not selecting steps by changed files"
    export BUILDKITE_AGENT_APPLY_IF_CHANGED=false
fi

# `[julia X]` and `[test_args ...]` select the Julia version and the tests (passed as
# `test_args` to `Pkg.test`) of the downstream steps, e.g. `[julia nightly]` or
# `[test_args core/codegen gpuarrays/broadcasting]`.
if [[ "${BUILDKITE_MESSAGE:-}" =~ \[julia\ ([^]]+)\] ]]; then
    export DOWNSTREAM_JULIA="${BASH_REMATCH[1]}"
    echo "Testing downstream packages on Julia $DOWNSTREAM_JULIA"
fi
if [[ "${BUILDKITE_MESSAGE:-}" =~ \[test_args\ ([^]]+)\] ]]; then
    export DOWNSTREAM_TEST_ARGS="${BASH_REMATCH[1]}"
    echo "Running downstream tests: $DOWNSTREAM_TEST_ARGS"
fi

# fetch the base branch first, as a stale local ref would widen the diff
buildkite-agent pipeline upload --fetch-diff-base .buildkite/downstream.yml
