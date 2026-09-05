#!/usr/bin/env bash
# Dispatch the Wenlan MCP server for Codex.
#
# Resolution order (most specific first):
#   1. Sibling file `bin/wenlan-mcp.local` next to this script.
#   2. WENLAN_MCP_DEV_BIN env var, for local development.
#   3. ~/.wenlan/bin/wenlan-mcp, installed by install.sh.
#   4. npx -y wenlan-mcp@^<.codex-plugin/plugin.json version>, package fallback.
#      The version is derived from the sibling plugin.json (the single source
#      of truth kept on the release train) rather than a hardcoded pin that
#      silently drifts — same approach as the Claude runner.
#
# The explicit agent name is a Codex plugin requirement: stdio MCP clients may
# send a client name during initialize, but the fallback must not mislabel
# Codex captures as another client.

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd -P)"
local_bin="${here}/wenlan-mcp.local"
agent_name="${WENLAN_MCP_AGENT_NAME:-codex}"

if [ -x "${local_bin}" ]; then
  exec "${local_bin}" --agent-name "${agent_name}" "$@"
fi

dev_bin="${WENLAN_MCP_DEV_BIN:-${ORIGIN_MCP_DEV_BIN:-}}"
if [ -n "${dev_bin}" ] && [ -x "${dev_bin}" ]; then
  exec "${dev_bin}" --agent-name "${agent_name}" "$@"
fi

installed_bin="${HOME}/.wenlan/bin/wenlan-mcp"
if [ -x "${installed_bin}" ]; then
  exec "${installed_bin}" --agent-name "${agent_name}" "$@"
fi

# Derive the npm version from the sibling plugin.json (single source of truth on
# the release train) so the fallback can't drift from a hardcoded pin; @latest if
# it can't be read. Strip any pre-release/build suffix (e.g. "0.17.0+codex").
# ponytail: sed-parse the one "version" key — no python/jq dep in the MCP host shell.
plugin_json="${here}/../.codex-plugin/plugin.json"
ver="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${plugin_json}" 2>/dev/null | head -1)"
ver="${ver%%+*}"
ver="${ver%%-*}"
if [ -n "${ver}" ]; then
  exec npx -y "wenlan-mcp@^${ver}" --agent-name "${agent_name}" "$@"
fi
exec npx -y wenlan-mcp@latest --agent-name "${agent_name}" "$@"
