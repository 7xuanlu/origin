#!/usr/bin/env bash
# First-run gauntlet: `npx -y wenlan-mcp` (macos-15, ubuntu-24.04, ubuntu-24.04-arm).
# A daemon from the release archive listens on 17882; the npm wrapper's
# postinstall download and the MCP round-trip are exercised against it.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
PORT=17882
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-npx-mcp-daemon.XXXXXX")"
DAEMON_PID=""
# Isolated cache so the postinstall really downloads instead of reusing a runner cache.
export npm_config_cache="$GAUNTLET_OUT/npm-cache"
if [ "$IS_LATEST" = true ]; then
    PKG="wenlan-mcp"
else
    PKG="wenlan-mcp@$VERSION"
    info pinned-mode "npx -y $PKG (unpinned npx resolves npm's latest tag, see npm-wenlan-mcp-latest)"
fi

cleanup() {
    trap - EXIT
    if [ -n "$DAEMON_PID" ]; then
        stop_process "$DAEMON_PID"
    fi
    rm -rf "$DATA_DIR"
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

NPM_LATEST="$(npm view wenlan-mcp version 2>/dev/null || true)"
info npm-wenlan-mcp-latest "${NPM_LATEST:-unknown}"
[ "$NPM_LATEST" = "$VERSION" ] || info npm-version-lag "npm latest=${NPM_LATEST:-unknown} release=$VERSION"

DAEMON_LINE="$(TAG="$TAG" PORT="$PORT" DATA_DIR="$DATA_DIR" VERSION="$VERSION" \
    bash "$HELPERS/daemon-from-archive.sh" | tee "$GAUNTLET_OUT/logs/daemon-from-archive.log" | tail -1)"
DAEMON_PID="$(cat "$DATA_DIR/daemon.pid" 2>/dev/null || true)"
info archive-daemon "$DAEMON_LINE"

# First invocation installs the package. npm >= 7 hides lifecycle-script output
# by default; the foreground flag only makes install.js's lines visible, it does
# not change what runs.
info npx-mcp-install-command "npm_config_foreground_scripts=true npx -y $PKG --version"
GAUNTLET_TIMEOUT=600 check_output npx-mcp-install "wenlan-mcp installed successfully" -- \
    env npm_config_foreground_scripts=true npx -y "$PKG" --version
check_output npx-mcp-postinstall-download "Downloading wenlan-mcp $VERSION for" -- \
    grep -F "Downloading wenlan-mcp $VERSION for" "$GAUNTLET_OUT/checks/npx-mcp-install.log"
# The documented command, verbatim, now that the cache is warm.
info npx-mcp-command "npx -y $PKG"
check_output npx-mcp-version "$VERSION" -- npx -y "$PKG" --version

MCP_BIN=npx MCP_ARGS="[\"-y\",\"$PKG\",\"--origin-url\",\"http://127.0.0.1:$PORT\"]" \
    EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief python3 "$HELPERS/mcp-roundtrip.py"
