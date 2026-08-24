#!/usr/bin/env bash
# First-run gauntlet: macOS desktop app (macos-15, arm64).
# Part A mounts the published .dmg and inspects the bundle. Part B runs the
# README one-liner verbatim, lets it launch the app, and proves the daemon,
# bundled CLI, and bundled MCP connector from a new user's seat.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
HEALTH="http://127.0.0.1:7878/api/health"
APP="/Applications/Wenlan.app"
MACOS="$APP/Contents/MacOS"
AGENTS="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
MOUNT="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-dmg.XXXXXX")"
DL="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-dl.XXXXXX")"

cleanup() {
    trap - EXIT
    hdiutil detach "$MOUNT" -quiet 2>/dev/null || true
    pkill -x wenlan-app 2>/dev/null || true
    sleep 3
    if curl -sf --max-time 2 "$HEALTH" >/dev/null 2>&1; then
        info daemon-after-app-quit "still healthy after wenlan-app quit (launchd-owned com.wenlan.server)"
    else
        info daemon-after-app-quit "unreachable after wenlan-app quit"
    fi
    if [ -x "$MACOS/wenlan" ]; then
        "$MACOS/wenlan" background off >"$GAUNTLET_OUT/logs/teardown-background-off.log" 2>&1 || true
    fi
    launchctl bootout "gui/$UID_NUM/com.wenlan.desktop" 2>/dev/null || true
    launchctl bootout "gui/$UID_NUM/com.wenlan.server" 2>/dev/null || true
    rm -f "$AGENTS/com.wenlan.desktop.plist" "$AGENTS/com.wenlan.server.plist"
    rm -rf "$APP" "$HOME/Library/Application Support/wenlan" "$MOUNT" "$DL"
    collect "$HOME/Library/Logs/com.wenlan.desktop/wenlan.log"
    # Empty substring: PASS purely on the nonzero exit.
    check_fails no-leftover-service "" -- launchctl print "gui/$UID_NUM/com.wenlan.server"
    check port-7878-closed -- bash -c '! lsof -nP -iTCP:7878 -sTCP:LISTEN'
    check no-leftover-process -- bash -c '! { pgrep -x wenlan-app; pgrep -x wenlan-server; } | grep -q .'
    evaluate
    exit $?
}
trap cleanup EXIT

# ── Part A: the .dmg a user downloads from the Releases page ─────────────────
DMG="Wenlan_${VERSION}_aarch64.dmg"
check dmg-download -- curl -fsSL --retry 3 -o "$DL/$DMG" "https://github.com/7xuanlu/wenlan/releases/download/$TAG/$DMG"
check dmg-attach -- hdiutil attach -nobrowse -readonly -mountpoint "$MOUNT" "$DL/$DMG"
check dmg-has-app -- test -d "$MOUNT/Wenlan.app"
check_output dmg-bundle-version "$VERSION" -- plutil -extract CFBundleShortVersionString raw "$MOUNT/Wenlan.app/Contents/Info.plist"
info dmg-codesign "$(codesign -dv --verbose=2 "$MOUNT/Wenlan.app" 2>&1 || true)"
info dmg-gatekeeper "$(spctl -a -t exec -vv "$MOUNT/Wenlan.app" 2>&1 || true)"
check dmg-detach -- hdiutil detach "$MOUNT"

# ── Part B: the README one-liner, launching the app ──────────────────────────
ONE_LINER='/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/scripts/install-macos-app.sh)"'
if [ "$IS_LATEST" != true ]; then
    export WENLAN_APP_RELEASE_JSON_URL="https://api.github.com/repos/7xuanlu/wenlan/releases/tags/$TAG"
    info pinned-mode "WENLAN_APP_RELEASE_JSON_URL=$WENLAN_APP_RELEASE_JSON_URL (unpinned installer reads releases/latest)"
fi
info app-install-command "$ONE_LINER"
GAUNTLET_TIMEOUT=600 check app-install -- bash -c "$ONE_LINER"

check app-installed -- test -d "$APP"
check app-no-quarantine -- bash -c '! xattr -l "$1" | grep -q com.apple.quarantine' _ "$APP"
check app-process-started -- bash -c 'for _ in $(seq 1 30); do pgrep -x wenlan-app >/dev/null && exit 0; sleep 1; done; exit 1'
wait_health "$HEALTH" 240 || true
assert_version "$HEALTH" "$VERSION"
check plist-desktop-exists -- test -f "$AGENTS/com.wenlan.desktop.plist"
check plist-server-exists -- test -f "$AGENTS/com.wenlan.server.plist"
if launchctl print "gui/$UID_NUM/com.wenlan.server" >"$GAUNTLET_OUT/checks/launchctl-print-server.log" 2>&1; then
    check launchctl-server-loaded -- launchctl print "gui/$UID_NUM/com.wenlan.server"
else
    # Headless runners may have no usable GUI launchd domain; a runner artifact, not a product finding.
    info launchctl-server-loaded "launchctl print failed: $(head -c 400 "$GAUNTLET_OUT/checks/launchctl-print-server.log")"
fi
SERVER_PID="$(pgrep -x wenlan-server | head -1 || true)"
SERVER_CMD="$(ps -o command= -p "${SERVER_PID:-0}" 2>/dev/null || true)"
info daemon-command "pid=${SERVER_PID:-none} cmd=$SERVER_CMD"
check daemon-binary-in-bundle -- bash -c '[[ "$1" == "$2"/wenlan-server* ]]' _ "$SERVER_CMD" "$MACOS"

WENLAN_BIN="$MACOS/wenlan" bash "$HELPERS/cli-roundtrip.sh"
MCP_BIN="$MACOS/wenlan-mcp" MCP_ARGS='[]' EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief \
    python3 "$HELPERS/mcp-roundtrip.py"
check_output doctor "Daemon: running on" -- "$MACOS/wenlan" doctor

sleep 60
check app-alive-after-60s -- pgrep -x wenlan-app
if screencapture -x "$GAUNTLET_OUT/logs/macos-app.png" 2>/dev/null; then
    info screenshot "logs/macos-app.png"
else
    info screenshot "screencapture failed (no window server session?)"
fi
