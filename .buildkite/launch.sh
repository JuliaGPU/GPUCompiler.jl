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

# fetch the base branch first, as a stale local ref would widen the diff
buildkite-agent pipeline upload --fetch-diff-base .buildkite/downstream.yml
