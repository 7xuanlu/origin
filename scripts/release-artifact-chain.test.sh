#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

release_targets="scripts/release_targets.py"
cli_runner="crates/wenlan-cli/npm/run.js"

expected_darwin_asset="wenlan-darwin-arm64.tar.gz"

python3 "$release_targets" matrix \
  | grep -q '"artifact_name":"wenlan-darwin-arm64"' \
  || fail "canonical release target inventory does not produce wenlan-darwin-arm64"

grep -q "const ASSET = \"${expected_darwin_asset}\"" "$cli_runner" \
  || fail "npm wenlan runner does not download ${expected_darwin_asset}"

if grep -q '\${name}-\${TARGET}' "$cli_runner"; then
  fail "npm wenlan runner still downloads per-binary target-name assets"
fi

echo "PASS: npm wenlan runner consumes release.yml darwin-arm64 artifact"
