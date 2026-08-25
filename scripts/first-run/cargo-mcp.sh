#!/usr/bin/env bash
# First-run gauntlet: `cargo install wenlan-mcp` (ubuntu-24.04).
# Builds the connector from crates.io in $HOME (outside the checkout, so the
# repo's rust-toolchain.toml cannot steer it), then runs the MCP round-trip
# against a daemon from the release archive on 17883.
# No `set -e`: a failed step records a row and the later steps still run.
# shellcheck disable=SC2016  # bash -c snippets take $1/$2 positionally on purpose
set -u -o pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
. "$(dirname "$0")/lib.sh"
: "${TAG:?}" "${VERSION:?}" "${IS_LATEST:?}" "${REPO_ROOT:?}"
HELPERS="$REPO_ROOT/scripts/first-run"
PORT=17883
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-cargo-daemon.XXXXXX")"
MCP="$HOME/.cargo/bin/wenlan-mcp"
DAEMON_PID=""
if [ "$IS_LATEST" = true ]; then
    CMD="cargo install wenlan-mcp"
else
    CMD="cargo install wenlan-mcp --version $VERSION"
    info pinned-mode "$CMD (unpinned cargo install resolves the newest crates.io version)"
fi

cleanup() {
    trap - EXIT
    if [ -n "$DAEMON_PID" ]; then
        stop_process "$DAEMON_PID"
    fi
    (cd "$HOME" && cargo uninstall wenlan-mcp) >"$GAUNTLET_OUT/logs/teardown-cargo-uninstall.log" 2>&1 || true
    rm -rf "$DATA_DIR"
    check cargo-mcp-removed -- test ! -e "$MCP"
    check no-leftover-process -- bash -c '! pgrep -x wenlan-server'
    evaluate
    exit $?
}
trap cleanup EXIT

info cargo-install-command "cd \$HOME && $CMD"
START="$(date +%s)"
GAUNTLET_TIMEOUT=2400 check cargo-install -- bash -c "cd \"\$HOME\" && $CMD"
info cargo-install-seconds "$(( $(date +%s) - START ))"
info cargo-toolchain "$(cd "$HOME" && rustc --version 2>&1; cargo --version 2>&1)"
check_output cargo-mcp-version "$VERSION" -- "$MCP" --version

DAEMON_LINE="$(TAG="$TAG" PORT="$PORT" DATA_DIR="$DATA_DIR" VERSION="$VERSION" \
    bash "$HELPERS/daemon-from-archive.sh" | tee "$GAUNTLET_OUT/logs/daemon-from-archive.log" | tail -1)"
DAEMON_PID="$(cat "$DATA_DIR/daemon.pid" 2>/dev/null || true)"
info archive-daemon "$DAEMON_LINE"

MCP_BIN="$MCP" MCP_ARGS="[\"--origin-url\",\"http://127.0.0.1:$PORT\"]" \
    EXPECT_TOOL_COUNT=29 MCP_TOOLS=capture,recall,brief python3 "$HELPERS/mcp-roundtrip.py"
