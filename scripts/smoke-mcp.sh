#!/usr/bin/env bash
# Smoke test: the shipped `wenlan-mcp` binary bridges a real MCP client to a
# real daemon. Black-box per-surface loop over stdio JSON-RPC: initialize ->
# tools/list -> capture tool call -> recall tool call. Isolated port + data dir
# + pages dir per repo smoke-test policy — never touches prod data.
#
# The isolated scratch, the asserted teardown, the ownership and isolation
# readbacks and the ledger multiset are shared with `smoke-cli.sh` and live in
# `lib/smoke-common.sh`.
#
# Windows note specific to this surface: `mcp-roundtrip.py` runs as a NATIVE
# Windows Python process and `subprocess.Popen`s MCP_BIN itself, so MCP_BIN,
# GAUNTLET_OUT, WENLAN_MCP_CACHE_DIR and the driver's own path all have to be
# native spellings — not just WENLAN_DATA_DIR. The shell keeps the MSYS forms
# for its own `rm -rf` and `cut`.
set -euo pipefail

PORT="${PORT:-17882}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/host-process.sh
. "$ROOT/scripts/lib/host-process.sh"

# BIN preset (e.g. a release-archive extract) skips the cargo build below.
BIN_PRESET="${BIN:-}"
BIN="${BIN:-$ROOT/target/debug}"
SERVER_BIN="$BIN/wenlan-server"
MCP_BIN_PATH="$BIN/wenlan-mcp"
if (( HOST_IS_WINDOWS == 1 )); then
    SERVER_BIN="${SERVER_BIN}.exe"
    MCP_BIN_PATH="${MCP_BIN_PATH}.exe"
fi

SMOKE_NAME=smoke-mcp
SMOKE_PASS_MESSAGE="PASS: MCP surface smoke (initialize, tools/list, capture, recall) against isolated daemon"
# shellcheck source=scripts/lib/smoke-common.sh
. "$ROOT/scripts/lib/smoke-common.sh"
trap smoke_cleanup EXIT

smoke_preflight

# `python3` is not the name the interpreter carries everywhere — the repo's own
# Windows gauntlet uses `python`. Resolve it rather than hardcoding.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi
[ -n "$PYTHON" ] || fail "python3 (or python) is required to drive the MCP round-trip"

if [ -z "$BIN_PRESET" ]; then
    echo "==> Building wenlan-server + wenlan-mcp"
    (cd "$ROOT" && cargo build -p wenlan-server -p wenlan-mcp)
fi
[ -x "$SERVER_BIN" ] || fail "no daemon binary at $SERVER_BIN"
[ -x "$MCP_BIN_PATH" ] || fail "no MCP binary at $MCP_BIN_PATH"

smoke_scratch

NATIVE_MCP_BIN="$(native_path "$MCP_BIN_PATH")" || fail "could not spell $MCP_BIN_PATH for the driver"
NATIVE_GAUNTLET_OUT="$(native_path "$DATA_DIR/gauntlet")" || fail "could not spell the gauntlet output directory"
NATIVE_MCP_CACHE="$(native_path "$DATA_DIR/mcp-cache")" || fail "could not spell the MCP cache directory"
NATIVE_DRIVER="$(native_path "$ROOT/scripts/first-run/mcp-roundtrip.py")" ||
    fail "could not spell the MCP driver path"

smoke_start_daemon
smoke_assert_ownership
smoke_assert_isolation

# --- drive the surface --------------------------------------------------------
echo "==> Driving wenlan-mcp over stdio JSON-RPC"
# The JSON-RPC driver lives in the first-run gauntlet helper (shared with the
# release-artifact workflow); it records every step and never exits early.
# WENLAN_MCP_CACHE_DIR keeps the self-update probe out of the user cache (the
# empty temp cache still forces one GET to github.com; its 3s timeout is
# fail-soft).
#
# MCP_TOOLS is exported here rather than left to the driver's own default, so
# the expectation below is computed from the SAME value the driver reads. The
# row set is a function of that value, so a pasted literal would be wrong for
# any caller that changes it.
MCP_TOOLS="${MCP_TOOLS:-capture,recall}"
export MCP_TOOLS
GAUNTLET_OUT="$NATIVE_GAUNTLET_OUT" GAUNTLET_CHANNEL=smoke-mcp \
WENLAN_NO_AUTOSTART=1 \
WENLAN_MCP_CACHE_DIR="$NATIVE_MCP_CACHE" \
MCP_BIN="$NATIVE_MCP_BIN" \
MCP_ARGS='["--origin-url","'"$HOST"'","--agent-name","smoke-mcp"]' \
    "$PYTHON" "$NATIVE_DRIVER"

# mcp-roundtrip.py always records initialize / tools-list / tool-count(INFO) /
# capture / recall, and emits an `mcp-brief` row only when `brief` is in
# MCP_TOOLS, so the expectation is derived rather than observed.
EXPECTED_ROWS="mcp-capture=PASS
mcp-initialize=PASS
mcp-recall=PASS
mcp-tool-count=INFO
mcp-tools-list=PASS"
case ",$MCP_TOOLS," in
    *,brief,*) EXPECTED_ROWS="$EXPECTED_ROWS
mcp-brief=PASS" ;;
esac
smoke_assert_ledger "$DATA_DIR/gauntlet/findings.tsv" "$EXPECTED_ROWS" MCP

BODY_OK=1
