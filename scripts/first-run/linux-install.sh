#!/usr/bin/env bash
# First-run gauntlet: install.sh on Linux (ubuntu-24.04, ubuntu-24.04-arm).
# Runs the documented curl | bash flow verbatim (or the pinned form), then the
# systemd user service, CLI, MCP, autostart recovery, `background off`, and an
# offline first-boot probe.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
HEALTH="http://127.0.0.1:7878/api/health"
UNIT="$HOME/.config/systemd/user/wenlan-server.service"
DATA_ROOT="$HOME/.local/share/wenlan"
XDG_RUNTIME_DIR="/run/user/$(id -u)"
export XDG_RUNTIME_DIR
SAFE_TAG="$(printf '%s' "$TAG" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
if [ "$IS_LATEST" = true ]; then
    CMD='curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | bash'
    BIN_DIR="$HOME/.wenlan/bin"
else
    CMD="curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/main/install.sh | WENLAN_RELEASE_TAG=$TAG bash"
    BIN_DIR="$HOME/.wenlan/releases/$SAFE_TAG"
    info pinned-mode "$CMD installs into $BIN_DIR and leaves ~/.bashrc alone (unpinned install.sh fetches releases/latest)"
fi
W="$BIN_DIR/wenlan"

cleanup() {
    trap - EXIT
    if [ -x "$W" ]; then
        "$W" background off >"$GAUNTLET_OUT/logs/teardown-background-off.log" 2>&1 || true
    fi
    systemctl --user disable --now wenlan-server.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload 2>/dev/null || true
    rm -rf "$HOME/.wenlan" "$DATA_ROOT"
    # Drop the two lines install.sh appended to ~/.bashrc.
    if [ -f "$HOME/.bashrc" ]; then
        awk '/^# Added by Wenlan installer$/ { skip = 2 } skip > 0 { skip--; next } { print }' "$HOME/.bashrc" >"$HOME/.bashrc.tmp" \
            && mv "$HOME/.bashrc.tmp" "$HOME/.bashrc"
    fi
    check unit-gone -- test ! -f "$UNIT"
    check port-7878-closed -- bash -c '! ss -ltnH "sport = :7878" | grep -q .'
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

if systemctl --user show-environment >"$GAUNTLET_OUT/checks/systemd-user-manager.log" 2>&1; then
    info systemd-user-manager "available (XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR)"
else
    info systemd-user-manager "unavailable: $(head -c 300 "$GAUNTLET_OUT/checks/systemd-user-manager.log"); service rows below reflect that"
fi

info install-command "$CMD"
GAUNTLET_TIMEOUT=600 check install-sh -- bash -c "$CMD"
LOG="$GAUNTLET_OUT/checks/install-sh.log"
if [ "$IS_LATEST" = true ]; then
    check_output install-latest-release-line "Latest release: $TAG" -- grep -F "Latest release: $TAG" "$LOG"
    check_output bashrc-path-block "# Added by Wenlan installer" -- grep -F "# Added by Wenlan installer" "$HOME/.bashrc"
else
    check_output install-requested-release-line "Requested release: $TAG" -- grep -F "Requested release: $TAG" "$LOG"
    check bashrc-untouched -- bash -c '! grep -qF "# Added by Wenlan installer" "$HOME/.bashrc"'
fi
export PATH="$BIN_DIR:$PATH"
info path-export "export PATH=\"$BIN_DIR:\$PATH\""
check_output setup-basic "Wenlan is set up for local memory." -- wenlan setup --basic
check_output background-on "Installed and started com.wenlan.server." -- wenlan background on
check unit-file-exists -- test -f "$UNIT"
check_output unit-enabled "enabled" -- systemctl --user is-enabled wenlan-server
wait_health "$HEALTH" 240 || true
assert_version "$HEALTH" "$VERSION"
check status -- wenlan status

WENLAN_BIN="$W" bash "$HELPERS/cli-roundtrip.sh"
MCP_BIN="$BIN_DIR/wenlan-mcp" MCP_ARGS='[]' EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief \
    python3 "$HELPERS/mcp-roundtrip.py"
check_output doctor "Daemon: running on" -- wenlan doctor

# Autostart recovery: stop the unit, then any CLI call must start it again.
check systemd-stop -- systemctl --user stop wenlan-server
sleep 2
info health-after-stop "$(curl -sf --max-time 2 "$HEALTH" 2>&1 || echo unreachable)"
check_output autostart-recovery "wenlan: daemon not reachable — starting" -- wenlan memories --limit 1

# `background off`: unit inactive but present, marker written, CLI explains itself.
check background-off -- wenlan background off
check_output unit-inactive-after-off "inactive" -- systemctl --user show wenlan-server --property=ActiveState --value
check unit-kept-after-off -- test -f "$UNIT"
check autostart-marker-exists -- test -f "$DATA_ROOT/autostart.off"
check_fails stopped-marker-error "daemon stopped by" -- wenlan search x
check background-on-again -- wenlan background on
wait_health "$HEALTH" 120 || true
assert_version "$HEALTH" "$VERSION"

# Offline first boot (informational): isolated data dir and port, Hugging Face unreachable.
OFF_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-offline.XXXXXX")"
OFF_LOG="$GAUNTLET_OUT/checks/offline-first-run.log"
OFF_RC=0
HF_ENDPOINT=http://127.0.0.1:9 WENLAN_DATA_DIR="$OFF_DIR" WENLAN_PORT=17999 \
    timeout 90 "$BIN_DIR/wenlan-server" --port 17999 --data-dir "$OFF_DIR" >"$OFF_LOG" 2>&1 || OFF_RC=$?
info offline-first-run "rc=$OFF_RC (124 = still running at 90s); $(tail -n 20 "$OFF_LOG")"
rm -rf "$OFF_DIR"

journalctl --user -u wenlan-server --no-pager -n 200 >"$GAUNTLET_OUT/logs/journal-wenlan-server.log" 2>&1 || true
