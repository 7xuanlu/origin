#!/usr/bin/env bash
set -euo pipefail

# bump-version.sh — update the version string across all project files
# Usage: bash scripts/bump-version.sh <NEW_VERSION>
# Example: bash scripts/bump-version.sh 0.2.0
# Example: bash scripts/bump-version.sh 0.2.0-alpha.1

NEW_VERSION="${1:-}"

# Validate argument
if [[ -z "$NEW_VERSION" ]]; then
  echo "Error: version argument required" >&2
  echo "Usage: bash scripts/bump-version.sh <N.N.N[-PRERELEASE]>" >&2
  exit 1
fi

# Validate semver format (N.N.N or N.N.N-prerelease)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Error: version must be in N.N.N or N.N.N-prerelease format (e.g. 0.2.0 or 0.2.0-alpha.1), got: $NEW_VERSION" >&2
  exit 1
fi

# Resolve repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Files to update
CARGO_TOMLS=(
  "$REPO_ROOT/app/Cargo.toml"
  "$REPO_ROOT/crates/origin-core/Cargo.toml"
  "$REPO_ROOT/crates/origin-server/Cargo.toml"
  "$REPO_ROOT/crates/origin-types/Cargo.toml"
)

echo "Bumping version to $NEW_VERSION"
echo ""

# Update Cargo.toml files — only the bare `version = "..."` line (the [package] version).
# Dependency version lines use inline table syntax ({ version = "..." }) and are not affected
# by this pattern which anchors to start-of-line.
for f in "${CARGO_TOMLS[@]}"; do
  sed -i '' -E 's/^version = "[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]*)?"/version = "'"$NEW_VERSION"'"/' "$f"
  echo "  Updated $f"
done

# Update package.json — the top-level "version" field
PKG_JSON="$REPO_ROOT/package.json"
sed -i '' -E 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]*)?"/"version": "'"$NEW_VERSION"'"/' "$PKG_JSON"
echo "  Updated $PKG_JSON"

# Update tauri.conf.json — the "version" field
TAURI_CONF="$REPO_ROOT/app/tauri.conf.json"
sed -i '' -E 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]*)?"/"version": "'"$NEW_VERSION"'"/' "$TAURI_CONF"
echo "  Updated $TAURI_CONF"

# Update npm wrapper package.json
NPM_PKG="$REPO_ROOT/packages/origin-mcp-npm/package.json"
if [ -f "$NPM_PKG" ]; then
  sed -i '' -E 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]*)?"/"version": "'"$NEW_VERSION"'"/' "$NPM_PKG"
  echo "  Updated $NPM_PKG"
fi

echo ""
echo "Regenerating Cargo.lock..."
cargo generate-lockfile
echo "  Updated Cargo.lock"

echo ""
echo "Changed files:"
git diff --stat app/Cargo.toml crates/*/Cargo.toml package.json app/tauri.conf.json packages/origin-mcp-npm/package.json Cargo.lock 2>/dev/null || true

echo ""
echo "Done. Verify with:"
echo '  grep -rn "version" app/Cargo.toml crates/*/Cargo.toml package.json app/tauri.conf.json | grep -E "^\S+:(version|\"version\")"'
