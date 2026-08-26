#!/usr/bin/env bash
# First-run gauntlet: Claude Code plugin `/setup` skill (ubuntu-24.04, macos-15).
# Replays plugin/skills/setup/SKILL.md from the tagged checkout ($RELEASE_SRC)
# step by step: hook while down, bootstrap line verbatim, re-probe, version
# drift, doctor, MCP through the plugin's runner, hook while healthy.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}" "${RELEASE_SRC:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
HEALTH="http://127.0.0.1:7878/api/health"
export CLAUDE_PLUGIN_ROOT="$RELEASE_SRC/plugin"
HOOK="$CLAUDE_PLUGIN_ROOT/hooks/check-daemon.sh"
RUNNER="$CLAUDE_PLUGIN_ROOT/bin/wenlan-mcp-runner.sh"
SKILL="$CLAUDE_PLUGIN_ROOT/skills/setup/SKILL.md"
PLUGIN_JSON="$CLAUDE_PLUGIN_ROOT/.claude-plugin/plugin.json"
OS="$(uname -s)"
BIN_DIR="$HOME/.wenlan/bin"
UID_NUM="$(id -u)"
if [ "$OS" = Linux ]; then
    export XDG_RUNTIME_DIR="/run/user/$UID_NUM"
    UNIT="$HOME/.config/systemd/user/wenlan-server.service"
    DATA_ROOT="$HOME/.local/share/wenlan"
else
    PLIST="$HOME/Library/LaunchAgents/com.wenlan.server.plist"
    DATA_ROOT="$HOME/Library/Application Support/wenlan"
fi
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"

strip_installer_block() {
    # Removes the two lines install.sh appends to an rc file. awk, because BSD
    # sed has no `addr,+N` form.
    [ -f "$1" ] || return 0
    awk '/^# Added by Wenlan installer$/ { skip = 2 } skip > 0 { skip--; next } { print }' "$1" >"$1.tmp" && mv "$1.tmp" "$1"
}

best_effort() {
    # INFO row with rc and the output tail; never a FAIL. stdin closed so a
    # login prompt cannot hang the job.
    local name="$1"; shift
    local log="$GAUNTLET_OUT/checks/$name.log" rc=0
    ${TIMEOUT_BIN:+"$TIMEOUT_BIN" 180} "$@" >"$log" 2>&1 </dev/null || rc=$?
    info "$name" "rc=$rc $(tail -c 600 "$log" | tr '\n' ' ')"
}

cleanup() {
    trap - EXIT
    if [ -x "$BIN_DIR/wenlan" ]; then
        "$BIN_DIR/wenlan" background off >"$GAUNTLET_OUT/logs/teardown-background-off.log" 2>&1 || true
    fi
    if [ "$OS" = Linux ]; then
        systemctl --user disable --now wenlan-server.service 2>/dev/null || true
        rm -f "$UNIT"
        systemctl --user daemon-reload 2>/dev/null || true
        journalctl --user -u wenlan-server --no-pager -n 200 >"$GAUNTLET_OUT/logs/journal-wenlan-server.log" 2>&1 || true
        strip_installer_block "$HOME/.bashrc"
        daemon_postmortem "$BIN_DIR/wenlan-server" "$DATA_ROOT"
        rm -rf "$HOME/.wenlan" "$DATA_ROOT"
        check unit-gone -- test ! -f "$UNIT"
        check port-7878-closed -- bash -c '! ss -ltnH "sport = :7878" | grep -q .'
    else
        launchctl bootout "gui/$UID_NUM/com.wenlan.server" 2>/dev/null || true
        rm -f "$PLIST"
        strip_installer_block "$HOME/.bashrc"
        strip_installer_block "$HOME/.zshrc"
        daemon_postmortem "$BIN_DIR/wenlan-server" "$DATA_ROOT"
        rm -rf "$HOME/.wenlan" "$DATA_ROOT"
        # Empty substring: PASS purely on the nonzero exit.
        check_fails no-leftover-service "" -- launchctl print "gui/$UID_NUM/com.wenlan.server"
        check port-7878-closed -- bash -c '! lsof -nP -iTCP:7878 -sTCP:LISTEN'
    fi
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

# Step 0: the marketplace path a real user takes. Installing the claude CLI
# itself is environment, so best effort; but once the CLI exists, the
# published marketplace install must actually work — an exit failure there is
# a product finding, not an environment artifact. stdin stays closed so a
# login prompt fails fast instead of hanging.
best_effort claude-code-install npm i -g @anthropic-ai/claude-code
if command -v claude >/dev/null 2>&1; then
    check claude-marketplace-add -- bash -c 'exec </dev/null; claude plugin marketplace add 7xuanlu/wenlan'
    check claude-plugin-install -- bash -c 'exec </dev/null; claude plugin install wenlan@7xuanlu-wenlan'
else
    info claude-marketplace-add "unchecked: claude CLI unavailable on this runner"
    info claude-plugin-install "unchecked: claude CLI unavailable on this runner"
fi
info plugin-diff-vs-main "$(diff -rq "$RELEASE_SRC/plugin" "$REPO_ROOT/plugin" 2>&1 | head -20)"

# Step 1: SessionStart hook with no runtime.
check_output hook-daemon-down "[wenlan] local runtime not running" -- bash "$HOOK"

# Step 2: the bootstrap block, verbatim from the tagged SKILL.md.
BOOTSTRAP_RE='curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v[0-9][^ ]*/install.sh | bash'
BOOTSTRAP="$(grep -o "$BOOTSTRAP_RE" "$SKILL" | head -1)"
info skill-bootstrap-line "${BOOTSTRAP:-not found in $SKILL}"
check_output skill-pin-matches-release "/v$VERSION/install.sh" -- grep -o "$BOOTSTRAP_RE" "$SKILL"
[ -n "$BOOTSTRAP" ] || BOOTSTRAP='echo "bootstrap line not found in SKILL.md"; false'
[ "$IS_LATEST" = true ] || info pinned-mode "SKILL.md pins install.sh to the tag, but that install.sh has no tag of its own and installs releases/latest; run verbatim anyway"
GAUNTLET_TIMEOUT=600 check skill-bootstrap -- bash -c "$BOOTSTRAP"
export PATH="$HOME/.wenlan/bin:$PATH"
info path-export 'export PATH="$HOME/.wenlan/bin:$PATH"'
check_output setup-basic "Wenlan is set up for local memory." -- wenlan setup --basic
check_output background-on "Installed and started com.wenlan.server." -- wenlan background on
if [ "$OS" = Linux ]; then
    check unit-file-exists -- test -f "$UNIT"
else
    check plist-exists -- test -f "$PLIST"
fi

# Step 3: the skill's 60-second re-probe, as written; then the real first-boot wait.
check skill-reprobe-60s -- bash -c 'for _ in $(seq 1 60); do curl -fsS -m 3 http://127.0.0.1:7878/api/health && exit 0; sleep 1; done; exit 1'
wait_health "$HEALTH" 240 || true
assert_version "$HEALTH" "$VERSION"
DAEMON_VER="$(curl -sf --max-time 5 "$HEALTH" 2>/dev/null | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
EXPECTED_VER="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$PLUGIN_JSON" | head -1)"
info version-compare "daemon=$DAEMON_VER expected=$EXPECTED_VER release=$VERSION"
# SKILL.md compares the release part only (it strips the daemon's +g<sha> suffix).
check daemon-matches-plugin-version -- bash -c '[ -n "$1" ] && [ "${1%%+*}" = "${2%%+*}" ]' _ "$DAEMON_VER" "$EXPECTED_VER"

# Step 4: doctor through the resolved binary, as the skill does.
W="$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")"
info doctor-binary "$W"
check_output doctor "Daemon: running on" -- "$W" doctor

# Step 5: MCP through the plugin's runner; prove it picked the installed binary, not npx.
info runner-candidates "installed=$(ls -l "$BIN_DIR/wenlan-mcp" 2>&1) local=$([ -x "$CLAUDE_PLUGIN_ROOT/bin/wenlan-mcp.local" ] && echo present || echo absent) WENLAN_MCP_DEV_BIN=${WENLAN_MCP_DEV_BIN:-unset}"
check_output runner-uses-installed-binary "exec $BIN_DIR/wenlan-mcp" -- bash -c 'bash -x "$1" --version 2>&1 | grep -F "+ exec"' _ "$RUNNER"
MCP_BIN="$RUNNER" MCP_ARGS='[]' EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief \
    python3 "$HELPERS/mcp-roundtrip.py"

# Step 6: hook again; silent when healthy and versions agree.
info hook-output-when-healthy "$(bash "$HOOK" 2>&1 || true)"
check hook-silent-when-healthy -- bash -c '[ -z "$(bash "$1" 2>&1)" ]' _ "$HOOK"
