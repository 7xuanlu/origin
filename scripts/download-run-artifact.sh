#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: download-run-artifact.sh <run-id> <artifact-name> <destination>" >&2
    exit 64
fi

run_id=$1
artifact_name=$2
destination=$3
gh_bin=${GH_BIN:-gh}
attempt_limit=${WENLAN_ARTIFACT_DOWNLOAD_ATTEMPTS:-4}
retry_delay=${WENLAN_ARTIFACT_DOWNLOAD_RETRY_DELAY_SECONDS:-2}

case "$run_id" in
    ''|*[!0-9]*)
        echo "run id must be numeric" >&2
        exit 64
        ;;
esac
if [[ -z "$artifact_name" || -z "$destination" ]]; then
    echo "artifact name and destination must be non-empty" >&2
    exit 64
fi
case "$attempt_limit:$retry_delay" in
    *[!0-9:]*|:*|*:) echo "retry settings must be non-negative integers" >&2; exit 64 ;;
esac
if (( attempt_limit < 1 )); then
    echo "artifact download attempts must be positive" >&2
    exit 64
fi
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

temp_parent=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
scratch=$(mktemp -d "$temp_parent/wenlan-artifact-download.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

for (( attempt = 1; attempt <= attempt_limit; attempt++ )); do
    attempt_dir="$scratch/attempt-$attempt"
    mkdir -p "$attempt_dir"
    if "$gh_bin" run download "$run_id" \
        --repo "$GITHUB_REPOSITORY" \
        --name "$artifact_name" \
        --dir "$attempt_dir"; then
        mkdir -p "$destination"
        cp -R "$attempt_dir"/. "$destination"/
        echo "artifact download succeeded on attempt $attempt/$attempt_limit"
        exit 0
    fi
    echo "artifact download attempt $attempt/$attempt_limit failed" >&2
    if (( attempt < attempt_limit )); then
        sleep "$retry_delay"
    fi
done

echo "artifact download exhausted $attempt_limit attempts" >&2
exit 1
