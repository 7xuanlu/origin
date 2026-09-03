#!/usr/bin/env bash
# Smoke test: the shipped `wenlan-mcp` binary bridges a real MCP client to a
# real daemon. Black-box per-surface loop over stdio JSON-RPC: initialize ->
# tools/list -> capture tool call -> recall tool call. Isolated port + data dir
# + pages dir per repo smoke-test policy — never touches prod data.
#
# Every measurement here is TRI-STATE: measured / negative / could not measure.
# A check that cannot run FAILS; it never reads as a negative. That is what the
# old `command -v lsof || fail` gate was protecting.
#
# Windows notes, all additive — POSIX behaviour is unchanged:
#   * `mcp-roundtrip.py` runs as a NATIVE Windows Python process and
#     `subprocess.Popen`s MCP_BIN itself, so MCP_BIN, GAUNTLET_OUT,
#     WENLAN_MCP_CACHE_DIR and the driver's own path all have to be native
#     spellings — not just WENLAN_DATA_DIR. The shell keeps the MSYS forms for
#     its own `rm -rf` and `cut`.
#   * `$!` is an MSYS job pid; the stop goes through the identity-checked
#     `force_terminate_process`, never pid alone.
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

DATA_DIR=""
PAGES_DIR=""
NATIVE_DATA_DIR=""
NATIVE_PAGES_DIR=""
NATIVE_SERVER_BIN=""
DAEMON_JOB_PID=""
DAEMON_WIN_PID=""
BODY_OK=""

fail() {
    echo "FAIL: $1" >&2
    if [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/daemon.log" ]; then
        echo "--- daemon log tail ---" >&2
        tail -40 "$DATA_DIR/daemon.log" >&2 || true
    fi
    exit 1
}

# Cleanup ASSERTS: the recorded process must be measurably gone AND the port
# measurably closed, and either failure fails the script. The PASS line lives
# here because it used to print before the EXIT trap ran, so a leaked daemon
# could not fail the run. PASS must not outrank a leak.
#
# Final status = the body's status if non-zero, else cleanup's status.
cleanup() {
    local body=$?
    trap - EXIT
    set +e
    local cleanup_rc=0 stop_rc=0 target_pid="" waited="" i

    if (( HOST_IS_WINDOWS == 1 )); then
        target_pid="$DAEMON_WIN_PID"
    else
        target_pid="$DAEMON_JOB_PID"
    fi

    if [ -n "$DAEMON_JOB_PID" ]; then
        if [ -z "$target_pid" ]; then
            echo "FAIL: cleanup cannot stop the daemon: no Windows pid was resolved" >&2
            echo "      end any process running $SERVER_BIN by hand before retrying" >&2
            cleanup_rc=1
        else
            force_terminate_process "$target_pid" "$NATIVE_SERVER_BIN" || stop_rc=$?
        fi

        for i in $(seq 1 30); do
            if ! kill -0 "$DAEMON_JOB_PID" 2>/dev/null; then waited=$i; break; fi
            sleep 1
        done
        if [ -n "$waited" ]; then
            wait "$DAEMON_JOB_PID" 2>/dev/null
        else
            echo "FAIL: cleanup: shell child $DAEMON_JOB_PID still alive after 30s" >&2
            cleanup_rc=1
        fi

        if [ -n "$target_pid" ]; then
            probe_process_alive "$target_pid"
            case "$PROCESS_ALIVE_STATE" in
                gone) ;;
                alive)
                    echo "FAIL: cleanup: daemon pid $target_pid is STILL running (stop status $stop_rc)" >&2
                    cleanup_rc=1
                    ;;
                *)
                    echo "FAIL: cleanup: liveness of pid $target_pid could not be measured" >&2
                    echo "      exit is UNMEASURED, which is not the same as gone" >&2
                    cleanup_rc=1
                    ;;
            esac
        fi

        probe_listener_port "$PORT"
        case "$LISTENER_PROBE_STATE" in
            none) ;;
            found)
                echo "FAIL: cleanup: port $PORT still has a listener (pid $LISTENER_PROBE_PID)" >&2
                cleanup_rc=1
                ;;
            *)
                echo "FAIL: cleanup: port $PORT release check could not be measured" >&2
                echo "      the port is UNMEASURED, which is not the same as released" >&2
                cleanup_rc=1
                ;;
        esac
    fi

    [ -n "$DATA_DIR" ] && rm -rf "$DATA_DIR"

    if [ "$body" -ne 0 ] || [ -z "$BODY_OK" ]; then
        exit $(( body != 0 ? body : 1 ))
    fi
    if [ "$cleanup_rc" -ne 0 ]; then
        exit "$cleanup_rc"
    fi
    echo "PASS: MCP surface smoke (initialize, tools/list, capture, recall) against isolated daemon"
    exit 0
}
trap cleanup EXIT

# --- preflight, BEFORE the build ---------------------------------------------
command -v curl >/dev/null 2>&1 ||
    fail "curl is required (health probe); its absence must not surface as a 120s timeout"

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

probe_listener_port "$PORT"
case "$LISTENER_PROBE_STATE" in
    found) fail "port ${PORT} already in use (pid $LISTENER_PROBE_PID); set PORT= to another free port" ;;
    unmeasured)
        fail "port ${PORT} could not be measured — needs netstat on Windows, lsof on macOS/Linux; an unmeasured port is not a free port"
        ;;
esac

if [ -z "$BIN_PRESET" ]; then
    echo "==> Building wenlan-server + wenlan-mcp"
    (cd "$ROOT" && cargo build -p wenlan-server -p wenlan-mcp)
fi
[ -x "$SERVER_BIN" ] || fail "no daemon binary at $SERVER_BIN"
[ -x "$MCP_BIN_PATH" ] || fail "no MCP binary at $MCP_BIN_PATH"

# --- isolated scratch: port, data dir, AND pages dir --------------------------
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/smoke-mcp.XXXXXX")" || fail "mktemp failed"
PAGES_DIR="$DATA_DIR/pages"
mkdir -p "$PAGES_DIR" || fail "could not create the scratch pages directory"
NATIVE_DATA_DIR="$(native_path "$DATA_DIR")" || fail "could not spell $DATA_DIR for the daemon"
NATIVE_PAGES_DIR="$(native_path "$PAGES_DIR")" || fail "could not spell $PAGES_DIR for the daemon"
NATIVE_SERVER_BIN="$(native_path "$SERVER_BIN")" || fail "could not spell $SERVER_BIN for the daemon"
NATIVE_MCP_BIN="$(native_path "$MCP_BIN_PATH")" || fail "could not spell $MCP_BIN_PATH for the driver"
NATIVE_GAUNTLET_OUT="$(native_path "$DATA_DIR/gauntlet")" || fail "could not spell the gauntlet output directory"
NATIVE_MCP_CACHE="$(native_path "$DATA_DIR/mcp-cache")" || fail "could not spell the MCP cache directory"
NATIVE_DRIVER="$(native_path "$ROOT/scripts/first-run/mcp-roundtrip.py")" ||
    fail "could not spell the MCP driver path"
NATIVE_FASTEMBED="$(native_path "${FASTEMBED_CACHE_DIR:-$ROOT/.fastembed_cache}")" ||
    fail "could not spell the fastembed cache directory for the daemon"

# The default pages folder is `.wenlan/pages` under the OS user-home, NOT under
# WENLAN_DATA_DIR. Written atomically BEFORE the spawn: startup reads
# configuration before it serves HTTP, so a config written afterwards would read
# back green while startup had already touched the default directory.
printf '{"knowledge_path":"%s","setup_completed":true}\n' "$NATIVE_PAGES_DIR" \
    >"$DATA_DIR/config.json.tmp" || fail "could not write the scratch config"
mv "$DATA_DIR/config.json.tmp" "$DATA_DIR/config.json" || fail "could not install the scratch config"
grep -qF "\"knowledge_path\":\"$NATIVE_PAGES_DIR\"" "$DATA_DIR/config.json" ||
    fail "the scratch config does not contain the expected knowledge_path"

echo "==> Starting daemon on port ${PORT} (data dir ${NATIVE_DATA_DIR})"
(cd "$ROOT" && WENLAN_NO_AUTOSTART=1 WENLAN_PORT="$PORT" WENLAN_DATA_DIR="$NATIVE_DATA_DIR" \
    FASTEMBED_CACHE_DIR="$NATIVE_FASTEMBED" \
    exec "$SERVER_BIN" >"$DATA_DIR/daemon.log" 2>&1) &
DAEMON_JOB_PID=$!

if (( HOST_IS_WINDOWS == 1 )); then
    resolve_rc=0
    DAEMON_WIN_PID="$(windows_pid_for_job "$DAEMON_JOB_PID" "$SERVER_BIN")" || resolve_rc=$?
    if (( resolve_rc == 2 )); then
        fail "the Windows pid of the daemon could not be measured (ps/awk failure)"
    elif (( resolve_rc != 0 )); then
        fail "job $DAEMON_JOB_PID never resolved to a $SERVER_BIN Windows pid"
    fi
fi

echo "==> Waiting for /api/health"
healthy=""
for i in $(seq 1 120); do
    if curl -sf --max-time 2 "$HOST/api/health" >/dev/null 2>&1; then
        echo "    healthy after ${i}s"
        healthy=1
        break
    fi
    kill -0 "$DAEMON_JOB_PID" 2>/dev/null || fail "daemon exited during startup"
    sleep 1
done
[ -n "$healthy" ] || fail "daemon did not become healthy within 120s"

# --- ownership: the listener must BE the process we spawned -------------------
probe_listener_port "$PORT"
case "$LISTENER_PROBE_STATE" in
    found) ;;
    none) fail "health succeeded but no listener was found on port $PORT" ;;
    *) fail "listener probe on port $PORT failed after health — ownership indeterminate" ;;
esac
if (( HOST_IS_WINDOWS == 1 )); then
    OWN_PID="$DAEMON_WIN_PID"
else
    OWN_PID="$DAEMON_JOB_PID"
fi
[ "$LISTENER_PROBE_PID" = "$OWN_PID" ] ||
    fail "port $PORT is held by pid $LISTENER_PROBE_PID, not the daemon we spawned ($OWN_PID)"
probe_process_image "$LISTENER_PROBE_PID"
case "$PROCESS_IMAGE_STATE" in
    found) ;;
    none) fail "the listener pid $LISTENER_PROBE_PID has no image — unmeasured, not a match" ;;
    *) fail "the image of listener pid $LISTENER_PROBE_PID could not be measured — not a match" ;;
esac
if (( HOST_IS_WINDOWS == 1 )); then
    listener_image="$(normalize_program_path "$PROCESS_IMAGE_VALUE")" ||
        fail "could not normalize the listener's image path"
    want_image="$(normalize_program_path "$SERVER_BIN")" ||
        fail "could not normalize the daemon binary path"
    [ "$listener_image" = "$want_image" ] ||
        fail "the listener's image is [$listener_image], not [$want_image]"
else
    case "$PROCESS_IMAGE_VALUE" in
        "$SERVER_BIN" | "$SERVER_BIN "*) ;;
        *) fail "the listener's image is [$PROCESS_IMAGE_VALUE], not [$SERVER_BIN]" ;;
    esac
fi

# --- isolation readback, from the LIVE daemon, both halves --------------------
reported="$(curl -sf --max-time 5 "$HOST/api/knowledge/path")" ||
    fail "pages readback request FAILED — unmeasured, not a match"
[ "$reported" = "{\"path\":\"$NATIVE_PAGES_DIR\"}" ] ||
    fail "pages readback mismatch: got [$reported] want [{\"path\":\"$NATIVE_PAGES_DIR\"}]"
logged_root="$(sed -n '/Wenlan data root: /{s/.*Wenlan data root: //;s/\r$//;p;q;}' "$DATA_DIR/daemon.log")" ||
    fail "extracting the logged data root FAILED — unmeasured, not a match"
[ -n "$logged_root" ] || fail "the daemon log has no 'Wenlan data root:' line at all"
if (( HOST_IS_WINDOWS == 1 )); then
    logged_cmp="$(printf '%s' "$logged_root" | tr 'A-Z\\' 'a-z/')"
    want_cmp="$(printf '%s' "$NATIVE_DATA_DIR" | tr 'A-Z\\' 'a-z/')"
else
    logged_cmp="$logged_root"
    want_cmp="$NATIVE_DATA_DIR"
fi
[ "$logged_cmp" = "$want_cmp" ] ||
    fail "data root mismatch: the daemon logged [$logged_root], scratch root is [$NATIVE_DATA_DIR]"
echo "    isolated: pages=$NATIVE_PAGES_DIR data=$NATIVE_DATA_DIR"

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

# --- exact (name,status) multiset against a FRESH ledger -----------------------
# "Not empty and no FAIL row" passed a ledger with an early PASS and every later
# step omitted. Derived from this invocation's own inputs: mcp-roundtrip.py
# always records initialize / tools-list / tool-count(INFO) / capture / recall,
# and emits an `mcp-brief` row only when `brief` is in MCP_TOOLS.
#
# Both sides are status-checked AND required non-empty: two unchecked `sort`s
# once let both sides come out empty and compare EQUAL, passing a wrong ledger.
EXPECTED_ROWS="mcp-capture=PASS
mcp-initialize=PASS
mcp-recall=PASS
mcp-tool-count=INFO
mcp-tools-list=PASS"
case ",$MCP_TOOLS," in
    *,brief,*) EXPECTED_ROWS="$EXPECTED_ROWS
mcp-brief=PASS" ;;
esac
TSV="$DATA_DIR/gauntlet/findings.tsv"
[ -s "$TSV" ] || fail "MCP round-trip recorded nothing"
actual_rows="$(cut -f2,3 "$TSV" | tr '\t' '=' | sort)" ||
    fail "reading the ledger FAILED — unmeasured, not a pass"
want_rows="$(printf '%s\n' "$EXPECTED_ROWS" | sort)" ||
    fail "sorting the expected ledger FAILED — unmeasured, not a pass"
[ -n "$actual_rows" ] && [ -n "$want_rows" ] ||
    fail "ledger comparison degenerated to empty (actual=[$actual_rows] want=[$want_rows])"
if [ "$actual_rows" != "$want_rows" ]; then
    echo "--- findings.tsv ---" >&2
    cat "$TSV" >&2
    echo "--- got ---" >&2; printf '%s\n' "$actual_rows" >&2
    echo "--- want --" >&2; printf '%s\n' "$want_rows" >&2
    fail "MCP round-trip ledger is not the expected (name,status) multiset"
fi

BODY_OK=1
