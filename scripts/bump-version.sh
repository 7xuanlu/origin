#!/usr/bin/env bash
# Sync version from version.txt to Cargo.toml workspace.package + npm + plugin manifests.
# Called by release-please.yml on the release branch after release-please opens a release PR.
# Usage: bash scripts/bump-version.sh
#   (no arguments — version is read from version.txt in the repo root)
set -euo pipefail

NEW_VERSION="$(cat version.txt | tr -d '[:space:]')"
[[ -n "$NEW_VERSION" ]] || { echo "ERROR: version.txt is empty"; exit 1; }

# Validate semver format (N.N.N or N.N.N-prerelease)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "ERROR: version.txt must contain N.N.N or N.N.N-prerelease format, got: $NEW_VERSION" >&2
  exit 1
fi

echo "Syncing version from version.txt: ${NEW_VERSION}"
echo ""

# 1. Cargo.toml workspace.package version (propagates to all crates via inheritance)
# Portability: BSD sed (macOS) needs empty -i arg; GNU sed (Linux CI) doesn't.
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s/^(version = \")[^\"]+(\".*x-release-please-version)/\1${NEW_VERSION}\2/" Cargo.toml
else
    sed -i -E "s/^(version = \")[^\"]+(\".*x-release-please-version)/\1${NEW_VERSION}\2/" Cargo.toml
fi
echo "  Updated Cargo.toml (workspace.package.version)"

# 1b. Cargo.toml workspace.dependencies version pins (wenlan-types, wenlan-core)
# These must match workspace.package.version so cargo publish strips the path
# and uses the registry version. Without this, the local member resolves
# `version = "0.X.Y"` against a workspace.package "0.X.Z" and the build fails.
for dep in wenlan-types wenlan-core; do
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' -E "s|^(${dep}[[:space:]]+= \\{ path = \"crates/${dep}\",[[:space:]]+version = \")[^\"]+(\".*)|\\1${NEW_VERSION}\\2|" Cargo.toml
    else
        sed -i -E "s|^(${dep}[[:space:]]+= \\{ path = \"crates/${dep}\",[[:space:]]+version = \")[^\"]+(\".*)|\\1${NEW_VERSION}\\2|" Cargo.toml
    fi
done
echo "  Updated Cargo.toml (workspace.dependencies wenlan-types/wenlan-core)"

# 2. npm wrapper package.json files
(cd crates/wenlan-mcp/npm && npm version "$NEW_VERSION" --no-git-tag-version --allow-same-version --ignore-scripts >/dev/null)
(cd crates/wenlan-cli/npm && npm version "$NEW_VERSION" --no-git-tag-version --allow-same-version --ignore-scripts >/dev/null)
echo "  Updated crates/wenlan-mcp/npm/package.json"
echo "  Updated crates/wenlan-cli/npm/package.json"

# 3. Claude plugin manifest (moved under plugin/ subdir in v0.5.0)
PLUGIN_MANIFEST="plugin/.claude-plugin/plugin.json"
jq ".version = \"$NEW_VERSION\"" "$PLUGIN_MANIFEST" > "${PLUGIN_MANIFEST}.tmp"
mv "${PLUGIN_MANIFEST}.tmp" "$PLUGIN_MANIFEST"
echo "  Updated $PLUGIN_MANIFEST"

CODEX_PLUGIN_MANIFEST="plugin-codex/.codex-plugin/plugin.json"
jq ".version = \"${NEW_VERSION}+codex\"" "$CODEX_PLUGIN_MANIFEST" > "${CODEX_PLUGIN_MANIFEST}.tmp"
mv "${CODEX_PLUGIN_MANIFEST}.tmp" "$CODEX_PLUGIN_MANIFEST"
echo "  Updated $CODEX_PLUGIN_MANIFEST"

# 4. Plugin's MCP server pin — the wrapper script falls back to
# `npx -y wenlan-mcp@^X.Y.Z` so a floating tag can't auto-RCE on every
# Claude Code session. The pin lives in the runner shell script, not
# .mcp.json, so dev users can override the wenlan-mcp binary via
# WENLAN_MCP_DEV_BIN.
CODEX_PLUGIN_MCP_RUNNER="plugin-codex/bin/wenlan-mcp-runner.sh"
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s|(wenlan-mcp@\\^)[0-9]+\\.[0-9]+\\.[0-9]+|\\1${NEW_VERSION}|g" "$CODEX_PLUGIN_MCP_RUNNER"
else
    sed -i -E "s|(wenlan-mcp@\\^)[0-9]+\\.[0-9]+\\.[0-9]+|\\1${NEW_VERSION}|g" "$CODEX_PLUGIN_MCP_RUNNER"
fi
echo "  Updated $CODEX_PLUGIN_MCP_RUNNER (wenlan-mcp pin)"

CODEX_PLUGIN_README="plugin-codex/README.md"
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s|(wenlan-mcp@\\^)[0-9]+\\.[0-9]+\\.[0-9]+|\\1${NEW_VERSION}|g" "$CODEX_PLUGIN_README"
else
    sed -i -E "s|(wenlan-mcp@\\^)[0-9]+\\.[0-9]+\\.[0-9]+|\\1${NEW_VERSION}|g" "$CODEX_PLUGIN_README"
fi
echo "  Updated $CODEX_PLUGIN_README (wenlan-mcp pin)"

# 5. /setup skill install.sh URL pinned to current tag (not `main`), so the
# install one-liner is reproducible at the release boundary.
SETUP_SKILL="plugin/skills/setup/SKILL.md"
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s|(raw\\.githubusercontent\\.com/7xuanlu/wenlan/)(main\|v[0-9]+\\.[0-9]+\\.[0-9]+)(/install\\.sh)|\\1v${NEW_VERSION}\\3|g" "$SETUP_SKILL"
else
    sed -i -E "s|(raw\\.githubusercontent\\.com/7xuanlu/wenlan/)(main\|v[0-9]+\\.[0-9]+\\.[0-9]+)(/install\\.sh)|\\1v${NEW_VERSION}\\3|g" "$SETUP_SKILL"
fi
echo "  Updated $SETUP_SKILL (install.sh tag pin)"

CODEX_SETUP_SKILL="plugin-codex/skills/setup/SKILL.md"
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s|(raw\\.githubusercontent\\.com/7xuanlu/wenlan/)(main\|v[0-9]+\\.[0-9]+\\.[0-9]+)(/install\\.sh)|\\1v${NEW_VERSION}\\3|g" "$CODEX_SETUP_SKILL"
else
    sed -i -E "s|(raw\\.githubusercontent\\.com/7xuanlu/wenlan/)(main\|v[0-9]+\\.[0-9]+\\.[0-9]+)(/install\\.sh)|\\1v${NEW_VERSION}\\3|g" "$CODEX_SETUP_SKILL"
fi
echo "  Updated $CODEX_SETUP_SKILL (install.sh tag pin)"

# 6. Desktop app version trio (app/Cargo.toml, app/tauri.conf.json,
# package.json). Kept in lockstep with the workspace release train here,
# not via release-please extra-files — extra-files collides with the
# closed config-key schema and pinned changed-file set enforced by
# scripts/validate-release-candidate.py.
APP_CARGO_TOML="app/Cargo.toml"
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s/^(version = \")[^\"]+(\".*x-release-please-version)/\1${NEW_VERSION}\2/" "$APP_CARGO_TOML"
else
    sed -i -E "s/^(version = \")[^\"]+(\".*x-release-please-version)/\1${NEW_VERSION}\2/" "$APP_CARGO_TOML"
fi
echo "  Updated $APP_CARGO_TOML (package.version)"

APP_TAURI_CONF="app/tauri.conf.json"
jq ".version = \"$NEW_VERSION\"" "$APP_TAURI_CONF" > "${APP_TAURI_CONF}.tmp"
mv "${APP_TAURI_CONF}.tmp" "$APP_TAURI_CONF"
echo "  Updated $APP_TAURI_CONF"

APP_PACKAGE_JSON="package.json"
jq ".version = \"$NEW_VERSION\"" "$APP_PACKAGE_JSON" > "${APP_PACKAGE_JSON}.tmp"
mv "${APP_PACKAGE_JSON}.tmp" "$APP_PACKAGE_JSON"
echo "  Updated $APP_PACKAGE_JSON"

# 7. Cargo.lock workspace member versions. release-please bumps the manifests
# above but never regenerates the lockfile, so validate-versions.sh (run in
# release.yml on the tag) fails on "Cargo.lock drift" and aborts the release.
# cargo isn't available on the release-please runner, so rewrite the lock
# entries textually — symmetric with validate-versions.sh's reader. Internal
# wenlan deps are listed name-only (no version string) in Cargo.lock, so only
# each member's own version line needs to change. wenlan-app joins the
# daemon crates here since it now rides the same release train (step 6).
awk -v ver="$NEW_VERSION" '
  $0 == "[[package]]" { in_pkg=1; is_member=0; print; next }
  in_pkg && $1 == "name" && $2 == "=" {
    n=$3; gsub(/"/, "", n)
    is_member = (n=="wenlan" || n=="wenlan-app" || n=="wenlan-core" || n=="wenlan-mcp" || n=="wenlan-server" || n=="wenlan-types")
    print; next
  }
  in_pkg && is_member && /^version = / { print "version = \"" ver "\""; in_pkg=0; is_member=0; next }
  { print }
' Cargo.lock > Cargo.lock.tmp && mv Cargo.lock.tmp Cargo.lock
echo "  Updated Cargo.lock (workspace member versions)"

echo ""
echo "Versions synced from version.txt (${NEW_VERSION}) to Cargo.toml + npm + plugin manifests + plugin MCP/skills + desktop app trio + Cargo.lock."
