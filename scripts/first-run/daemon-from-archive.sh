#!/usr/bin/env bash
# First-run gauntlet: fetch the published runtime archive for this platform
# exactly the way install.sh does, unpack it, and start wenlan-server from it.
#
# Env:
#   TAG             release tag, e.g. v0.17.0 (required)
#   PORT            daemon port (required)
#   DATA_DIR        scratch root: bin/, daemon.log, daemon.pid live here (required)
#   VERSION         when set, /api/health .version must equal it (leading v ok)
#   HEALTH_TIMEOUT  seconds to wait for /api/health (default 180)
#   REPO            GitHub repo (default 7xuanlu/wenlan)
#
# Records each stage through lib.sh. Prints `DAEMON_BIN=<dir> DAEMON_PID=<pid>`
# as the last line. Exits 1 when the daemon never became healthy (after
# recording the FAIL and copying the daemon log tail into the check log).
set -uo pipefail
# shellcheck source=scripts/first-run/lib.sh
. "$(dirname "$0")/lib.sh"

: "${TAG:?TAG (release tag, e.g. v0.17.0) is required}"
: "${PORT:?PORT is required}"
: "${DATA_DIR:?DATA_DIR is required}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
REPO="${REPO:-7xuanlu/wenlan}"
BIN="$DATA_DIR/bin"
HEALTH_URL="http://127.0.0.1:$PORT/api/health"

# Same table as install.sh — keep the two in sync.
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
    Darwin-arm64)              ASSET="wenlan-darwin-arm64.tar.gz" ;;
    Linux-aarch64|Linux-arm64) ASSET="wenlan-linux-arm64.tar.gz" ;;
    Linux-x86_64)              ASSET="wenlan-linux-x64.tar.gz" ;;
    *)
        # shellcheck disable=SC2016  # $1 is for the inner bash, on purpose
        check platform-supported -- bash -c 'echo "unsupported platform: $1"; exit 1' _ "$OS-$ARCH"
        exit 1
        ;;
esac
URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"
ARCHIVE="$DATA_DIR/$ASSET"
mkdir -p "$BIN"

echo "==> Downloading $URL"
# One retry after 30s: anonymous GitHub downloads on runners occasionally 5xx
# or rate-limit. The per-check cap is raised because the archive is large.
# shellcheck disable=SC2016  # $1/$2 are for the inner bash, on purpose
GAUNTLET_TIMEOUT=900 check download-archive -- bash -c '
    curl -fL --connect-timeout 15 --max-time 400 -o "$1" "$2" && exit 0
    echo "first download failed; retrying in 30s"
    sleep 30
    curl -fL --connect-timeout 15 --max-time 400 -o "$1" "$2"
' _ "$ARCHIVE" "$URL"
if [ ! -s "$ARCHIVE" ]; then
    echo "==> download failed; nothing to start" >&2
    exit 1
fi
info download-size "$ASSET: $(wc -c <"$ARCHIVE" | tr -d ' ') bytes"

echo "==> Extracting into $BIN"
check extract-archive -- tar -xzf "$ARCHIVE" -C "$BIN"
# Flat layout, same three members install.sh expects.
check archive-members -- ls -l "$BIN/wenlan" "$BIN/wenlan-server" "$BIN/wenlan-mcp"
chmod +x "$BIN/wenlan" "$BIN/wenlan-server" "$BIN/wenlan-mcp" 2>/dev/null || true
if [ "$OS" = Darwin ]; then
    # install.sh clears the quarantine attribute on unsigned binaries.
    xattr -cr "$BIN/wenlan" "$BIN/wenlan-server" "$BIN/wenlan-mcp" 2>/dev/null || true
fi

echo "==> Starting wenlan-server on port $PORT (data dir $DATA_DIR)"
(cd "$DATA_DIR" && WENLAN_PORT="$PORT" WENLAN_DATA_DIR="$DATA_DIR" WENLAN_NO_AUTOSTART=1 \
    exec "$BIN/wenlan-server" >"$DATA_DIR/daemon.log" 2>&1) &
DAEMON_PID=$!
echo "$DAEMON_PID" >"$DATA_DIR/daemon.pid"

echo "==> Waiting for $HEALTH_URL (up to ${HEALTH_TIMEOUT}s)"
if ! wait_health "$HEALTH_URL" "$HEALTH_TIMEOUT"; then
    {
        if kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo "daemon pid $DAEMON_PID is still running but never answered $HEALTH_URL"
        else
            echo "daemon pid $DAEMON_PID exited during startup"
        fi
        echo "--- daemon log tail ---"
        tail -80 "$DATA_DIR/daemon.log" 2>/dev/null
    } | tee "$GAUNTLET_OUT/checks/health-timeout.log" >&2
    collect "$DATA_DIR/daemon.log"
    exit 1
fi

if [ -n "${VERSION:-}" ]; then
    assert_version "$HEALTH_URL" "$VERSION"
fi

echo "DAEMON_BIN=$BIN DAEMON_PID=$DAEMON_PID"
