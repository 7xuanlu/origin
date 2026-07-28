#!/usr/bin/env bash
# Smoke test: the shipped `wenlan-mcp` binary bridges a real MCP client to a
# real daemon. Black-box per-surface loop over stdio JSON-RPC: initialize ->
# tools/list -> capture tool call -> recall tool call. Isolated port + data
# dir per repo smoke-test policy — never touches prod data.
set -euo pipefail

PORT="${PORT:-17882}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug"

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

echo "==> Building wenlan-server + wenlan-mcp"
(cd "$ROOT" && cargo build -p wenlan-server -p wenlan-mcp)

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
# Keep the self-update probe out of the user cache (the empty temp cache
# still forces one GET to api.github.com; its 3s timeout is fail-soft).
WENLAN_MCP_CACHE_DIR="$DATA_DIR/mcp-cache" \
MCP_BIN="$BIN/wenlan-mcp" MCP_URL="$HOST" python3 - <<'EOF' \
    || fail "MCP stdio round-trip failed"
import json, os, select, subprocess, sys, time

SENTINEL = "walrus-observatory-5507"
proc = subprocess.Popen(
    [os.environ["MCP_BIN"], "--origin-url", os.environ["MCP_URL"],
     "--agent-name", "smoke-mcp"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
)

def send(obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode())
    proc.stdin.flush()

buf = b""

def recv(want_id, timeout=60):
    # Frame by hand: select() + buffered readline() can hang on a partial
    # frame or strand a second frame already sitting in Python's buffer
    # while select() waits on the drained fd.
    global buf
    deadline = time.monotonic() + timeout
    while True:
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                return msg
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(f"timeout waiting for response id={want_id}")
        ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1))
        if ready:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise SystemExit("wenlan-mcp closed stdout")
            buf += chunk

def expect_ok(msg, what):
    if "error" in msg:
        raise SystemExit(f"{what} returned JSON-RPC error: {msg['error']}")
    if msg.get("result", {}).get("isError"):
        raise SystemExit(f"{what} returned tool error: {msg['result']}")
    return msg["result"]

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2024-11-05", "capabilities": {},
    "clientInfo": {"name": "smoke", "version": "0"}}})
init = expect_ok(recv(1), "initialize")
print(f"    initialize ok: {init.get('serverInfo', {}).get('name', '?')}")
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
tools = {t["name"] for t in expect_ok(recv(2), "tools/list")["tools"]}
for needed in ("capture", "recall"):
    if needed not in tools:
        raise SystemExit(f"tools/list missing '{needed}' (got {sorted(tools)})")
print(f"    tools/list ok: {len(tools)} tools")

send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
    "name": "capture",
    "arguments": {"content": f"The {SENTINEL} sentinel lives in the MCP smoke."}}})
expect_ok(recv(3), "capture")
print("    capture ok")

# One monotonic window for the whole poll — per-retry timeouts would let
# the advertised 60s stretch to 30 minutes against a hung daemon.
found = False
poll_deadline = time.monotonic() + 60
while time.monotonic() < poll_deadline:
    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "recall",
        "arguments": {"query": "walrus observatory sentinel"}}})
    result = expect_ok(
        recv(4, timeout=max(1, poll_deadline - time.monotonic())), "recall")
    if SENTINEL in json.dumps(result):
        found = True
        break
    time.sleep(2)
if not found:
    raise SystemExit("captured sentinel not present in recall output within 60s")
print("    recall ok: sentinel round-tripped")

proc.stdin.close()
proc.wait(timeout=10)
EOF

echo "PASS: MCP surface smoke (initialize, tools/list, capture, recall) against isolated daemon"
