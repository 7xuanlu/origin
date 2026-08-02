#!/usr/bin/env bash
set -euo pipefail

targets() {
  python3 scripts/release_targets.py preflight-matrix "$@" |
    python3 -c 'import json, sys; print(",".join(item["target"] for item in json.load(sys.stdin)["include"]))'
}

assert_targets() {
  label="$1"
  expected="$2"
  shift 2
  actual="$(targets "$@")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label: expected $expected, got $actual" >&2
    exit 1
  fi
}

non_windows="aarch64-apple-darwin,aarch64-unknown-linux-gnu,x86_64-unknown-linux-gnu"
all_targets="$non_windows,x86_64-pc-windows-msvc"

assert_targets ordinary-pr "$non_windows" \
  --event-name pull_request \
  --ref refs/pull/431/merge \
  --head-ref codex/promote-release-artifacts
assert_targets ordinary-main "x86_64-pc-windows-msvc" \
  --event-name push \
  --ref refs/heads/main
assert_targets release-please "$all_targets" \
  --event-name pull_request \
  --ref refs/pull/430/merge \
  --head-ref release-please--branches--main
assert_targets workflow-dispatch "$all_targets" \
  --event-name workflow_dispatch \
  --ref refs/heads/main

echo "release preflight event matrices: PASS"
