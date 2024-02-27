#!/bin/bash -e

# handle user inputs
[ $# -ne 4 ] && { echo "Usage: $0 <pipeline> <branch> <build_name> <destination_file>" >&2; exit 1; }
PIPELINE="$1"
BRANCH="$2"
BUILD_NAME="$3"
DEST_FILE="$4"
[ -z "$BUILDKITE_TOKEN" ] && { echo "BUILDKITE_TOKEN not set." >&2; exit 1; }

API_BASE="https://api.buildkite.com/v2"
ORG="julialang"

# find the first successful build on the requested branch, and get its artifacts url
ARTIFACTS_URL=$(curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" "$API_BASE/organizations/$ORG/pipelines/$PIPELINE/builds?branch=$BRANCH" | \
    jq -r "first(.[] | .jobs[] | select(.step_key == \"$BUILD_NAME\" and .exit_status == 0) | .artifacts_url)")
[ -z "$ARTIFACTS_URL" ] && { echo "No successful build found."; exit 1; }

# fetch the url of the first artifact
ARTIFACT_URL=$(curl -s -H "Authorization: Bearer $BUILDKITE_TOKEN" "$ARTIFACTS_URL" | \
    jq -r '.[0].download_url')
[ -z "$ARTIFACT_URL" ] && { echo "No artifact found."; exit 1; }

curl -s -L -H "Authorization: Bearer $BUILDKITE_TOKEN" -o "$DEST_FILE" "$ARTIFACT_URL"
echo "Artifact downloaded as $DEST_FILE"
