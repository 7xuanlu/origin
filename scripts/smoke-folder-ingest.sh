#!/usr/bin/env bash
# Smoke test: the shipped daemon binary ingests a folder over real HTTP.
# Isolated port + data dir per repo smoke-test policy — never touches prod
# data (dev/prod share 7878 + the platform data dir by default).
set -euo pipefail

PORT="${PORT:-17879}"
HOST="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug"

DATA_DIR="$(mktemp -d)"
FIXTURE_DIR="$(mktemp -d)"
DAEMON_PID=""

cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        kill -9 "$DAEMON_PID" >/dev/null 2>&1 || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    # Verify the port actually freed so a rerun cannot collide.
    for _ in $(seq 1 10); do
        if ! lsof -ti ":${PORT}" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    rm -rf "$DATA_DIR" "$FIXTURE_DIR"
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
    if curl -sf "$HOST/api/health" >/dev/null 2>&1; then
        echo "    healthy after ${i}s"
        healthy=1
        break
    fi
    kill -0 "$DAEMON_PID" 2>/dev/null || fail "daemon exited during startup"
    sleep 1
done
[ -n "$healthy" ] || fail "daemon did not become healthy within 120s"

echo "==> Creating fixture folder"
cat >"$FIXTURE_DIR/99-notes.md" <<'EOF'
# Smoke note

Ordinary markdown content to give the chunker something to work with.
The zebra-quokka-7139 sentinel sentence lives in the markdown fixture.
EOF
cat >"$FIXTURE_DIR/00-plain.txt" <<'EOF'
Plain text fixture for the folder ingest smoke test.
The xylophone-birch-4242 sentinel sentence is buried here.
EOF

echo "==> wenlan sources add ${FIXTURE_DIR}"
WENLAN_HOST="$HOST" "$BIN/wenlan" sources add "$FIXTURE_DIR"

echo "==> Confirming source registered with file_count > 0"
# macOS mktemp dirs canonicalize /var -> /private/var; match on the basename.
FIXTURE_KEY="$(basename "$FIXTURE_DIR")"
SOURCES="$(curl -sf "$HOST/api/sources")" || fail "GET /api/sources failed"
echo "$SOURCES" | grep -q "$FIXTURE_KEY" || fail "source not registered: $FIXTURE_KEY not in /api/sources"
# Fresh isolated data dir => ours is the only source; any zero count is a fail.
if echo "$SOURCES" | grep -q '"file_count":0'; then
    fail "source registered but file_count is 0"
fi

echo "==> Searching for the buried sentinel sentence"
hit=""
# First ambient admission is intentionally bounded by the 60s startup delay
# plus two 30s resource samples. Keep two poll windows of CI jitter beyond
# that 120s floor; the loop still exits immediately when the document lands.
for i in $(seq 1 90); do
    RESP="$(curl -sf -X POST "$HOST/api/memory/search" \
        -H 'Content-Type: application/json' \
        -d '{"query":"xylophone birch buried sentinel sentence","limit":5}')" || RESP=""
    if echo "$RESP" | grep -q "xylophone-birch-4242"; then
        echo "    hit after ${i} poll(s)"
        hit=1
        break
    fi
    sleep 2
done
[ -n "$hit" ] || fail "buried sentence not retrievable within 180s"

echo "==> Waiting for the sibling document in the next bounded thermal turn"
sibling_hit=""
for i in $(seq 1 90); do
    RESP="$(curl -sf -X POST "$HOST/api/memory/search" \
        -H 'Content-Type: application/json' \
        -d '{"query":"zebra quokka sentinel markdown fixture","limit":5}')" || RESP=""
    if echo "$RESP" | grep -q "zebra-quokka-7139"; then
        echo "    sibling hit after ${i} poll(s)"
        sibling_hit=1
        break
    fi
    sleep 2
done
[ -n "$sibling_hit" ] || fail "sibling document not retrievable within 180s"

echo "==> Deletion propagation: remove 00-plain.txt, re-sync, sentence must vanish"
rm "$FIXTURE_DIR/00-plain.txt"
WENLAN_HOST="$HOST" "$BIN/wenlan" sources add "$FIXTURE_DIR"
gone=""
for i in $(seq 1 15); do
    RESP="$(curl -sf -X POST "$HOST/api/memory/search" \
        -H 'Content-Type: application/json' \
        -d '{"query":"xylophone birch buried sentinel sentence","limit":5}')" || RESP=""
    if ! echo "$RESP" | grep -q "xylophone-birch-4242"; then
        echo "    reaped after ${i} poll(s)"
        gone=1
        break
    fi
    sleep 2
done
[ -n "$gone" ] || fail "deleted file's chunks still retrievable after re-sync"
# The sibling's indexed chunks must survive deletion of the first file.
RESP="$(curl -sf -X POST "$HOST/api/memory/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"zebra quokka sentinel markdown fixture","limit":5}')" || RESP=""
echo "$RESP" | grep -q "zebra-quokka-7139" || fail "sibling markdown chunks lost after a sibling delete"

echo "==> PASS"
