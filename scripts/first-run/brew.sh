#!/usr/bin/env bash
# First-run gauntlet: Homebrew tap (macos-15, arm64).
# `brew install 7xuanlu/tap/wenlan` ships the CLI only, so the documented next
# step (`wenlan background on`) is expected to fail; that gap is recorded, then
# a daemon from the release archive stands in for the CLI and MCP loops.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
PORT=17881
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-brew-daemon.XXXXXX")"
UID_NUM="$(id -u)"
DAEMON_PID=""

cleanup() {
    trap - EXIT
    if [ -n "$DAEMON_PID" ]; then
        stop_process "$DAEMON_PID"
    fi
    # `background on` should fail before registering, but never trust that: unload anyway.
    launchctl bootout "gui/$UID_NUM/com.wenlan.server" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/com.wenlan.server.plist"
    brew uninstall wenlan wenlan-mcp >"$GAUNTLET_OUT/logs/teardown-brew-uninstall.log" 2>&1 || true
    brew untap 7xuanlu/tap >"$GAUNTLET_OUT/logs/teardown-brew-untap.log" 2>&1 || true
    rm -rf "$DATA_DIR" "$HOME/Library/Application Support/wenlan"
    check brew-wenlan-removed -- bash -c '! command -v wenlan'
    check brew-wenlan-mcp-removed -- bash -c '! command -v wenlan-mcp'
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

# Homebrew cannot pin: the tap serves whatever the formula says (recorded below).
[ "$IS_LATEST" = true ] || info pinned-mode "brew has no pin; installing the tap's current formula and comparing to $VERSION"
GAUNTLET_TIMEOUT=900 check brew-install-wenlan -- brew install 7xuanlu/tap/wenlan
GAUNTLET_TIMEOUT=900 check brew-install-wenlan-mcp -- brew install 7xuanlu/tap/wenlan-mcp
check_output wenlan-version "$VERSION" -- wenlan --version
check_output wenlan-mcp-version "$VERSION" -- wenlan-mcp --version
check brew-test-wenlan -- brew test 7xuanlu/tap/wenlan
check brew-test-wenlan-mcp -- brew test 7xuanlu/tap/wenlan-mcp
FORMULA_VER="$(brew info --json=v2 7xuanlu/tap/wenlan 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["formulae"][0]["versions"]["stable"])' 2>/dev/null || true)"
info brew-formula-version "${FORMULA_VER:-unknown}"
[ "$FORMULA_VER" = "$VERSION" ] || info brew-version-lag "formula=${FORMULA_VER:-unknown} release=$VERSION"
WENLAN_PATH="$(command -v wenlan || true)"
info brew-wenlan-path "${WENLAN_PATH:-not on PATH}"
info brew-server-next-to-cli "$(ls -l "$(dirname "${WENLAN_PATH:-/nonexistent}")/wenlan-server" 2>&1 || true)"

# The documented next step. Today it fails because the formula ships no
# wenlan-server (finding F7) — that is a real user-facing failure, so it is a
# plain check and stays red until packaging or the docs change. Encoding the
# known defect as the success condition would hide the breakage and turn the
# eventual fix into a false red.
check brew-background-on -- wenlan background on
info brew-status-without-daemon "$(WENLAN_NO_AUTOSTART=1 wenlan status 2>&1 || true)"

# Stand-in daemon from the release archive so the brew CLI and MCP can be exercised.
DAEMON_LINE="$(TAG="$TAG" PORT="$PORT" DATA_DIR="$DATA_DIR" VERSION="$VERSION" \
    bash "$HELPERS/daemon-from-archive.sh" | tee "$GAUNTLET_OUT/logs/daemon-from-archive.log" | tail -1)"
DAEMON_PID="$(cat "$DATA_DIR/daemon.pid" 2>/dev/null || true)"
info archive-daemon "$DAEMON_LINE"
WENLAN_BIN="${WENLAN_PATH:-wenlan}" WENLAN_HOST="http://127.0.0.1:$PORT" bash "$HELPERS/cli-roundtrip.sh"
MCP_BIN="$(command -v wenlan-mcp || echo wenlan-mcp)" MCP_ARGS="[\"--origin-url\",\"http://127.0.0.1:$PORT\"]" \
    EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief python3 "$HELPERS/mcp-roundtrip.py"
