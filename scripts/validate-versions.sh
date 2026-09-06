#!/usr/bin/env bash
# Pre-flight: assert all version sources match RELEASE_TAG.
set -euo pipefail

[[ -n "${RELEASE_TAG:-}" ]] || { echo "ERROR: RELEASE_TAG env var required"; exit 1; }
TAG_VER="${RELEASE_TAG#v}"

VTXT_VER=$(cat version.txt | tr -d '[:space:]')
WS_VER=$(grep -E '^version = ' Cargo.toml | head -1 | sed -E 's/version = "([^"]+)".*/\1/')
WENLAN_TYPES_DEP_VER=$(grep -E '^wenlan-types[[:space:]]+=' Cargo.toml | sed -E 's/.*version = "([^"]+)".*/\1/')
WENLAN_CORE_DEP_VER=$(grep -E '^wenlan-core[[:space:]]+=' Cargo.toml | sed -E 's/.*version = "([^"]+)".*/\1/')
APP_CARGO_VER=$(grep -E '^version = ' app/Cargo.toml | head -1 | sed -E 's/version = "([^"]+)".*/\1/')
APP_TAURI_VER=$(jq -r .version app/tauri.conf.json)
APP_PKG_VER=$(jq -r .version package.json)
LOCK_VERSIONS=$(awk '
  $0 == "[[package]]" { in_pkg=1; name=""; version=""; next }
  in_pkg && $1 == "name" && $2 == "=" {
    name=$3
    gsub(/"/, "", name)
    next
  }
  in_pkg && $1 == "version" && $2 == "=" {
    version=$3
    gsub(/"/, "", version)
    if (name == "wenlan" || name == "wenlan-app" || name == "wenlan-core" || name == "wenlan-mcp" || name == "wenlan-server" || name == "wenlan-types") {
      print name ":" version
    }
    in_pkg=0
  }
' Cargo.lock | sort)
MCP_NPM_VER=$(jq -r .version crates/wenlan-mcp/npm/package.json)
WENLAN_NPM_VER=$(jq -r .version crates/wenlan-cli/npm/package.json)
PLUGIN_VER=$(jq -r .version plugin/.claude-plugin/plugin.json)
CODEX_PLUGIN_VER_RAW=$(jq -r .version plugin-codex/.codex-plugin/plugin.json)
CODEX_PLUGIN_VER="${CODEX_PLUGIN_VER_RAW%%+*}"
# The Codex runner derives its `npx wenlan-mcp@^X.Y.Z` fallback from the sibling
# plugin.json at run time, so it must carry no hardcoded pin that could drift.
CODEX_RUNNER_HARDCODED_PINS=$(grep -Eo 'wenlan-mcp@\^[0-9]+\.[0-9]+\.[0-9]+' plugin-codex/bin/wenlan-mcp-runner.sh | sed -E 's/.*@\^//' | sort -u || true)
CODEX_RUNNER_DERIVES_PIN=$(grep -c '\.codex-plugin/plugin\.json' plugin-codex/bin/wenlan-mcp-runner.sh || true)
CODEX_SETUP_TAGS=$(grep -Eo '/v[0-9]+\.[0-9]+\.[0-9]+/install\.sh' plugin-codex/skills/setup/SKILL.md | sed -E 's|/v([^/]+)/install\.sh|\1|' | sort -u || true)

echo "Tag:         $TAG_VER"
echo "version.txt: $VTXT_VER"
echo "Cargo:       $WS_VER"
echo "wenlan-types dep: $WENLAN_TYPES_DEP_VER"
echo "wenlan-core dep:  $WENLAN_CORE_DEP_VER"
echo "app/Cargo.toml:   $APP_CARGO_VER"
echo "app/tauri.conf:   $APP_TAURI_VER"
echo "package.json:     $APP_PKG_VER"
echo "Cargo.lock:"
printf '%s\n' "$LOCK_VERSIONS" | sed 's/^/  /'
echo "wenlan-mcp npm: $MCP_NPM_VER"
echo "wenlan npm: $WENLAN_NPM_VER"
echo "Plugin:      $PLUGIN_VER"
echo "Codex plugin: $CODEX_PLUGIN_VER_RAW"
echo "Codex runner hardcoded pins: ${CODEX_RUNNER_HARDCODED_PINS:-none}"
echo "Codex setup tags:"
printf '%s\n' "$CODEX_SETUP_TAGS" | sed 's/^/  /'

if [[ "$VTXT_VER" != "$TAG_VER" || "$WS_VER" != "$TAG_VER" || "$WENLAN_TYPES_DEP_VER" != "$TAG_VER" || "$WENLAN_CORE_DEP_VER" != "$TAG_VER" || "$MCP_NPM_VER" != "$TAG_VER" || "$WENLAN_NPM_VER" != "$TAG_VER" || "$PLUGIN_VER" != "$TAG_VER" || "$CODEX_PLUGIN_VER" != "$TAG_VER" || "$APP_CARGO_VER" != "$TAG_VER" || "$APP_TAURI_VER" != "$TAG_VER" || "$APP_PKG_VER" != "$TAG_VER" ]]; then
    echo "ERROR: version drift — bump-version.sh likely failed in release-please.yml"
    exit 1
fi

for pin in $CODEX_SETUP_TAGS; do
    if [[ "$pin" != "$TAG_VER" ]]; then
        echo "ERROR: Codex plugin release pin drift — ${pin} is not ${TAG_VER}"
        exit 1
    fi
done

if [[ -z "$CODEX_SETUP_TAGS" ]]; then
    echo "ERROR: Codex plugin release pin missing"
    exit 1
fi

if [[ -n "$CODEX_RUNNER_HARDCODED_PINS" ]]; then
    echo "ERROR: Codex runner carries a hardcoded wenlan-mcp pin (${CODEX_RUNNER_HARDCODED_PINS//$'\n'/ }); it must derive the pin from plugin.json"
    exit 1
fi

if [[ "$CODEX_RUNNER_DERIVES_PIN" == "0" ]]; then
    echo "ERROR: Codex runner does not derive its wenlan-mcp pin from .codex-plugin/plugin.json"
    exit 1
fi

for crate in wenlan wenlan-app wenlan-core wenlan-mcp wenlan-server wenlan-types; do
    if ! printf '%s\n' "$LOCK_VERSIONS" | grep -qx "${crate}:${TAG_VER}"; then
        echo "ERROR: Cargo.lock drift — ${crate} is not ${TAG_VER}"
        exit 1
    fi
done

echo "All versions consistent: $TAG_VER"
