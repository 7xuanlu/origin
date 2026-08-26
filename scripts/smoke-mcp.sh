#!/usr/bin/env bash
# Smoke test: the shipped `wenlan-mcp` binary bridges a real MCP client to a
# real daemon. Black-box per-surface loop over stdio JSON-RPC: initialize ->
# tools/list -> capture tool call -> recall tool call. Isolated port + data
# dir per repo smoke-test policy — never touches prod data.
set -euo pipefail

PORT="${PORT:-17882}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# BIN preset (e.g. a release-archive extract) skips the cargo build below.
BIN_PRESET="${BIN:-}"
BIN="${BIN:-$ROOT/target/debug}"

# Explicit template: macOS mktemp ignores TMPDIR without one, and the
# sandboxed dev loop can only write under TMPDIR.
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/smoke-mcp.XXXXXX")"
DAEMON_PID=""

cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        kill -9 "$DAEMON_PID" >/dev/null 2>&1 || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    for _ in $(seq 1 10); do
        if ! lsof -ti ":${PORT}" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    rm -rf "$DATA_DIR"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $1" >&2
    echo "--- daemon log tail ---" >&2
    tail -40 "$DATA_DIR/daemon.log" >&2 || true
    exit 1
}

if [ -z "$BIN_PRESET" ]; then
    echo "==> Building wenlan-server + wenlan-mcp"
    (cd "$ROOT" && cargo build -p wenlan-server -p wenlan-mcp)
fi

# Without lsof a failed port check reads as "port free" — fail loud instead.
command -v lsof >/dev/null 2>&1 || fail "lsof is required (port check + cleanup)"

if lsof -ti ":${PORT}" >/dev/null 2>&1; then
    fail "port ${PORT} already in use; set PORT= to another free port"
fi

echo "==> Starting daemon on port ${PORT} (data dir ${DATA_DIR})"
(cd "$ROOT" && WENLAN_PORT="$PORT" WENLAN_DATA_DIR="$DATA_DIR" \
    exec "$BIN/wenlan-server" >"$DATA_DIR/daemon.log" 2>&1) &
DAEMON_PID=$!

echo "==> Waiting for /api/health"
healthy=""
for i in $(seq 1 120); do
    if curl -sf --max-time 2 "$HOST/api/health" >/dev/null 2>&1; then
        echo "    healthy after ${i}s"
        healthy=1
        break
    fi
    kill -0 "$DAEMON_PID" 2>/dev/null || fail "daemon exited during startup"
    sleep 1
done
[ -n "$healthy" ] || fail "daemon did not become healthy within 120s"

echo "==> Driving wenlan-mcp over stdio JSON-RPC"
# The JSON-RPC driver lives in the first-run gauntlet helper (shared with the
# release-artifact workflow); it records every step and never exits early.
# WENLAN_MCP_CACHE_DIR keeps the self-update probe out of the user cache (the
# empty temp cache still forces one GET to github.com; its 3s timeout is
# fail-soft).
GAUNTLET_OUT="$DATA_DIR/gauntlet" GAUNTLET_CHANNEL=smoke-mcp \
WENLAN_MCP_CACHE_DIR="$DATA_DIR/mcp-cache" \
MCP_BIN="$BIN/wenlan-mcp" \
MCP_ARGS='["--origin-url","'"$HOST"'","--agent-name","smoke-mcp"]' \
    python3 "$ROOT/scripts/first-run/mcp-roundtrip.py"
[ -s "$DATA_DIR/gauntlet/findings.tsv" ] || fail "MCP round-trip recorded nothing"
if grep -q $'\tFAIL\t' "$DATA_DIR/gauntlet/findings.tsv"; then
    echo "--- findings.tsv ---" >&2
    cat "$DATA_DIR/gauntlet/findings.tsv" >&2
    fail "MCP stdio round-trip recorded FAIL rows"
fi

echo "PASS: MCP surface smoke (initialize, tools/list, capture, recall) against isolated daemon"
