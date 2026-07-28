#!/usr/bin/env bash
# Smoke test: the shipped `wenlan` CLI binary drives a real daemon over HTTP.
# Black-box per-surface loop: capture -> memories -> search -> status, all
# through the CLI, never curl. Isolated port + data dir per repo smoke-test
# policy — never touches prod data (dev/prod share 7878 by default).
set -euo pipefail

PORT="${PORT:-17881}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug"

# Explicit template: macOS mktemp ignores TMPDIR without one, and the
# sandboxed dev loop can only write under TMPDIR.
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/smoke-cli.XXXXXX")"
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

echo "==> Building wenlan-server + wenlan"
(cd "$ROOT" && cargo build -p wenlan-server -p wenlan)

# Without lsof a failed port check reads as "port free" — fail loud instead.
command -v lsof >/dev/null 2>&1 || fail "lsof is required (port check + cleanup)"

if lsof -ti ":${PORT}" >/dev/null 2>&1; then
    fail "port ${PORT} already in use; set PORT= to another free port"
fi

echo "==> Starting daemon on port ${PORT} (data dir ${DATA_DIR})"
# cwd = repo root so the daemon finds the shared .fastembed_cache.
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

SENTINEL="kumquat-lighthouse-8231"
CLI() { WENLAN_HOST="$HOST" "$BIN/wenlan" --format json "$@"; }

echo "==> wenlan status"
STATUS_OUT="$(CLI status)" || fail "wenlan status exited nonzero"
[ -n "$STATUS_OUT" ] || fail "wenlan status printed nothing"

echo "==> wenlan capture (sentinel)"
CLI capture "The ${SENTINEL} sentinel sentence lives in the CLI smoke." \
    --type fact >/dev/null || fail "wenlan capture exited nonzero"

echo "==> wenlan memories contains the sentinel"
# Capture then match: piping into grep -q can SIGPIPE the CLI under pipefail.
MEMS_OUT="$(CLI memories --limit 20)" || fail "wenlan memories exited nonzero"
case "$MEMS_OUT" in
    *"$SENTINEL"*) ;;
    *) fail "captured sentinel not listed by wenlan memories" ;;
esac

echo "==> wenlan search finds the sentinel"
hit=""
for i in $(seq 1 30); do
    SEARCH_OUT="$(CLI search "kumquat lighthouse sentinel sentence" --limit 5)" \
        || fail "wenlan search exited nonzero"
    case "$SEARCH_OUT" in
        *"$SENTINEL"*)
            echo "    hit after ${i} poll(s)"
            hit=1
            break
            ;;
    esac
    sleep 2
done
[ -n "$hit" ] || fail "sentinel not retrievable via wenlan search within 60s"

echo "PASS: CLI surface smoke (status, capture, memories, search) against isolated daemon"
