#!/usr/bin/env bash
# First-run gauntlet: `npx -y wenlan setup` (macos-15, arm64).
# Runs the README command verbatim (or the pinned form), then proves the
# daemon, CLI, MCP, autostart recovery, and `background off` semantics.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
HEALTH="http://127.0.0.1:7878/api/health"
PLIST="$HOME/Library/LaunchAgents/com.wenlan.server.plist"
DATA_ROOT="$HOME/Library/Application Support/wenlan"
UID_NUM="$(id -u)"
SAFE_TAG="$(printf '%s' "$TAG" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
if [ "$IS_LATEST" = true ]; then
    CMD="npx -y wenlan setup"
    BIN_DIR="$HOME/.wenlan/bin"
else
    CMD="WENLAN_RELEASE_TAG=$TAG npx -y wenlan@$VERSION setup"
    BIN_DIR="$HOME/.wenlan/releases/$SAFE_TAG"
    info pinned-mode "$CMD installs into $BIN_DIR instead of ~/.wenlan/bin (unpinned npx would fetch releases/latest)"
fi
W="$BIN_DIR/wenlan"

cleanup() {
    trap - EXIT
    if [ -x "$W" ]; then
        "$W" background off >"$GAUNTLET_OUT/logs/teardown-background-off.log" 2>&1 || true
    fi
    launchctl bootout "gui/$UID_NUM/com.wenlan.server" 2>/dev/null || true
    daemon_postmortem "$BIN_DIR/wenlan-server" "$DATA_ROOT"
    rm -f "$PLIST"
    rm -rf "$HOME/.wenlan" "$DATA_ROOT"
    # Empty substring: PASS purely on the nonzero exit.
    check_fails no-leftover-service "" -- launchctl print "gui/$UID_NUM/com.wenlan.server"
    check port-7878-closed -- bash -c '! lsof -nP -iTCP:7878 -sTCP:LISTEN'
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

info npx-setup-command "$CMD"
GAUNTLET_TIMEOUT=900 check npx-setup -- bash -c "$CMD"
LOG="$GAUNTLET_OUT/checks/npx-setup.log"
check_output npx-downloading-line "Downloading Wenlan $TAG..." -- grep -F "Downloading Wenlan $TAG..." "$LOG"
check_output npx-binaries-installed-line "Wenlan binaries are installed in" -- grep -F "Wenlan binaries are installed in" "$LOG"
check_output npx-background-on-line "Installed and started com.wenlan.server; daemon healthy at" -- grep -F "Installed and started com.wenlan.server; daemon healthy at" "$LOG"
for b in wenlan wenlan-server wenlan-mcp; do
    check "bin-$b-executable" -- test -x "$BIN_DIR/$b"
done
wait_health "$HEALTH" 240 || true
assert_version "$HEALTH" "$VERSION"
check plist-exists -- test -f "$PLIST"
if [ "$IS_LATEST" != true ]; then
    # run.js runs `background on` even for a tagged install; install.sh says not to.
    if [ -f "$PLIST" ]; then
        info pinned-background-on "run.js registered com.wenlan.server for a pinned install; ProgramArguments=$(plutil -extract ProgramArguments json -o - "$PLIST" 2>&1)"
    else
        info pinned-background-on "no com.wenlan.server.plist written"
    fi
fi

WENLAN_BIN="$W" bash "$HELPERS/cli-roundtrip.sh"
MCP_BIN="$BIN_DIR/wenlan-mcp" MCP_ARGS='[]' EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief \
    python3 "$HELPERS/mcp-roundtrip.py"
check_output doctor "Daemon: running on" -- "$W" doctor

# Autostart recovery: kill the launchd-owned daemon, then any CLI call must restart it.
check launchctl-kill-daemon -- launchctl kill TERM "gui/$UID_NUM/com.wenlan.server"
sleep 2
info health-after-kill "$(curl -sf --max-time 2 "$HEALTH" 2>&1 || echo unreachable)"
check_output autostart-recovery "wenlan: daemon not reachable — starting com.wenlan.server" -- "$W" memories --limit 1

# `background off`: daemon down, registration kept, marker written, CLI explains itself.
check background-off -- "$W" background off
check health-unreachable-after-off -- bash -c '! curl -sf --max-time 2 "$1" >/dev/null' _ "$HEALTH"
check autostart-marker-exists -- test -f "$DATA_ROOT/autostart.off"
check plist-kept-after-off -- test -f "$PLIST"
check_fails stopped-marker-error "daemon stopped by" -- env WENLAN_NO_AUTOSTART= "$W" search x
check background-on-again -- "$W" background on
wait_health "$HEALTH" 120 || true
assert_version "$HEALTH" "$VERSION"
