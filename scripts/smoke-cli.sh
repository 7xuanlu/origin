#!/usr/bin/env bash
# Smoke test: the shipped `wenlan` CLI binary drives a real daemon over HTTP.
# Black-box per-surface loop: capture -> memories -> search -> status, all
# through the CLI, never curl. Isolated port + data dir + pages dir per repo
# smoke-test policy — never touches prod data (dev/prod share 7878 by default).
#
# The isolated scratch, the asserted teardown, the ownership and isolation
# readbacks and the ledger multiset are shared with `smoke-mcp.sh` and live in
# `lib/smoke-common.sh`.
set -euo pipefail

PORT="${PORT:-17881}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/host-process.sh
. "$ROOT/scripts/lib/host-process.sh"

# BIN preset (e.g. a release-archive extract) skips the cargo build below.
BIN_PRESET="${BIN:-}"
BIN="${BIN:-$ROOT/target/debug}"
SERVER_BIN="$BIN/wenlan-server"
CLI_BIN="$BIN/wenlan"
if (( HOST_IS_WINDOWS == 1 )); then
    SERVER_BIN="${SERVER_BIN}.exe"
    CLI_BIN="${CLI_BIN}.exe"
fi

SMOKE_NAME=smoke-cli
SMOKE_PASS_MESSAGE="PASS: CLI surface smoke (status, capture, memories, search) against isolated daemon"
# shellcheck source=scripts/lib/smoke-common.sh
. "$ROOT/scripts/lib/smoke-common.sh"
trap smoke_cleanup EXIT

smoke_preflight

if [ -z "$BIN_PRESET" ]; then
    echo "==> Building wenlan-server + wenlan"
    (cd "$ROOT" && cargo build -p wenlan-server -p wenlan)
fi
[ -x "$SERVER_BIN" ] || fail "no daemon binary at $SERVER_BIN"
[ -x "$CLI_BIN" ] || fail "no CLI binary at $CLI_BIN"

smoke_scratch
smoke_start_daemon
smoke_assert_ownership
smoke_assert_isolation

# --- drive the surface --------------------------------------------------------
# The CLI loop itself lives in the first-run gauntlet helper (shared with the
# release-artifact workflow); it records every step and never exits early.
GAUNTLET_OUT="$DATA_DIR/gauntlet" GAUNTLET_CHANNEL=smoke-cli \
WENLAN_NO_AUTOSTART=1 WENLAN_BIN="$CLI_BIN" WENLAN_HOST="$HOST" \
    bash "$ROOT/scripts/first-run/cli-roundtrip.sh"

# cli-roundtrip.sh makes four unconditional check/check_output calls and emits
# no conditional row, so the expectation is derived rather than observed.
EXPECTED_ROWS="cli-capture=PASS
cli-memories=PASS
cli-search=PASS
cli-status=PASS"
smoke_assert_ledger "$DATA_DIR/gauntlet/findings.tsv" "$EXPECTED_ROWS" CLI

BODY_OK=1
