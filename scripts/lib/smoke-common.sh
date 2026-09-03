#!/usr/bin/env bash
# Shared scaffolding for the surface smokes (`smoke-cli.sh`, `smoke-mcp.sh`).
# Source this file AFTER `lib/host-process.sh`; never run it.
#
# Every measurement here is TRI-STATE: measured / negative / could not measure.
# A check that cannot run FAILS; it never reads as a negative. That is what the
# old `command -v lsof || fail` gate was protecting.
#
# Windows notes, all additive — POSIX behaviour is unchanged:
#   * `mktemp -d` gets an explicit template on every path (macOS mktemp ignores
#     TMPDIR without one, the sandboxed dev loop can only write under TMPDIR,
#     and TMPDIR is unset in Git Bash), and the resulting MSYS path is converted
#     with `native_path` at the DAEMON boundary only: the shell's own `rm -rf`
#     still needs the MSYS spelling. Two variables, never one rewritten variable.
#   * `$!` is an MSYS job pid; `kill -9` on it reaches the MSYS wrapper, not
#     necessarily the native daemon. Both pids are kept and the stop goes through
#     the identity-checked `force_terminate_process`, never pid alone.
#
# The caller provides, before calling anything here:
#   PORT, HOST, ROOT, SERVER_BIN   the run's port, base URL, repo root, daemon
#   SMOKE_NAME                     scratch-dir prefix, e.g. "smoke-cli"
#   SMOKE_PASS_MESSAGE             the single PASS line cleanup prints

DATA_DIR=""
PAGES_DIR=""
NATIVE_DATA_DIR=""
NATIVE_PAGES_DIR=""
NATIVE_SERVER_BIN=""
NATIVE_FASTEMBED=""
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

# Cleanup ASSERTS. It is not enough to try to stop the daemon: the recorded
# process must be measurably gone AND the port measurably closed, and either
# failure fails the script. The PASS line lives in here because it used to print
# before the EXIT trap ran, so a leaked daemon could not fail the run. PASS must
# not outrank a leak.
#
# Final status = the body's status if non-zero, else cleanup's status. A cleanup
# that exits with its own status alone would turn a failed body into a pass
# whenever the teardown happened to work.
smoke_cleanup() {
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
            # Windows only: the native pid never resolved, so there is no
            # identity to kill by and killing the MSYS job pid would be a
            # pid-only kill of the wrong process. Say so instead of pretending.
            echo "FAIL: cleanup cannot stop the daemon: no Windows pid was resolved" >&2
            echo "      end any process running $SERVER_BIN by hand before retrying" >&2
            cleanup_rc=1
        else
            force_terminate_process "$target_pid" "$NATIVE_SERVER_BIN" || stop_rc=$?
        fi

        # Reap the shell child, bounded. An unconditional `wait` here would
        # block forever on a daemon that ignored the stop.
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

        # The stop helper's status is a diagnostic only; the verdict is this
        # measurement. A closed port would not prove it either — that is a
        # separate contract, asserted below.
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

        # Port release is a SEPARATE contract from process exit: a closed port
        # does not prove the process we recorded is gone, and a dead process
        # does not prove the port was released. Asserted only when this run
        # actually spawned something — a preflight refusal has nothing to leak.
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
    echo "$SMOKE_PASS_MESSAGE"
    exit 0
}

# Preflight, run BEFORE the build: a checkout that could not run the smoke at
# all should not pay a full compile before being told so.
smoke_preflight() {
    command -v curl >/dev/null 2>&1 ||
        fail "curl is required (health probe); its absence must not surface as a 120s timeout"

    probe_listener_port "$PORT"
    case "$LISTENER_PROBE_STATE" in
        found) fail "port ${PORT} already in use (pid $LISTENER_PROBE_PID); set PORT= to another free port" ;;
        unmeasured)
            fail "port ${PORT} could not be measured — needs netstat on Windows, lsof on macOS/Linux; an unmeasured port is not a free port"
            ;;
    esac
}

# Isolated scratch: data dir, pages dir, and the native spellings the daemon
# needs. The default pages folder is `.wenlan/pages` under the OS user-home, NOT
# under WENLAN_DATA_DIR, so isolating the data dir alone still writes captures
# into the developer's real pages directory. The config is written atomically
# BEFORE the spawn: startup reads configuration before it serves HTTP, so a
# config written after the spawn would read back green while startup had already
# touched the default directory.
smoke_scratch() {
    DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${SMOKE_NAME}.XXXXXX")" || fail "mktemp failed"
    PAGES_DIR="$DATA_DIR/pages"
    mkdir -p "$PAGES_DIR" || fail "could not create the scratch pages directory"
    NATIVE_DATA_DIR="$(native_path "$DATA_DIR")" || fail "could not spell $DATA_DIR for the daemon"
    NATIVE_PAGES_DIR="$(native_path "$PAGES_DIR")" || fail "could not spell $PAGES_DIR for the daemon"
    NATIVE_SERVER_BIN="$(native_path "$SERVER_BIN")" || fail "could not spell $SERVER_BIN for the daemon"
    NATIVE_FASTEMBED="$(native_path "${FASTEMBED_CACHE_DIR:-$ROOT/.fastembed_cache}")" ||
        fail "could not spell the fastembed cache directory for the daemon"

    printf '{"knowledge_path":"%s","setup_completed":true}\n' "$NATIVE_PAGES_DIR" \
        >"$DATA_DIR/config.json.tmp" || fail "could not write the scratch config"
    mv "$DATA_DIR/config.json.tmp" "$DATA_DIR/config.json" || fail "could not install the scratch config"
    grep -qF "\"knowledge_path\":\"$NATIVE_PAGES_DIR\"" "$DATA_DIR/config.json" ||
        fail "the scratch config does not contain the expected knowledge_path"
}

# Spawn the daemon, resolve the pid the listener table will report, and wait for
# health. The daemon no longer reads a cache relative to its working directory,
# so the repo's shared .fastembed_cache is named explicitly instead of
# downloading 210 MB into every scratch data dir.
smoke_start_daemon() {
    local resolve_rc healthy i
    echo "==> Starting daemon on port ${PORT} (data dir ${NATIVE_DATA_DIR})"
    (cd "$ROOT" && WENLAN_NO_AUTOSTART=1 WENLAN_PORT="$PORT" WENLAN_DATA_DIR="$NATIVE_DATA_DIR" \
        FASTEMBED_CACHE_DIR="$NATIVE_FASTEMBED" \
        exec "$SERVER_BIN" >"$DATA_DIR/daemon.log" 2>&1) &
    DAEMON_JOB_PID=$!

    if (( HOST_IS_WINDOWS == 1 )); then
        # `$!` is the MSYS job pid; every Windows process and listener table
        # reports the Windows one. Both are kept: bash can only `wait` on its
        # own child.
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
}

# Ownership: the listener must BE the process we spawned. Health alone only
# proves that something answers on the port.
smoke_assert_ownership() {
    local own_pid listener_image want_image
    probe_listener_port "$PORT"
    case "$LISTENER_PROBE_STATE" in
        found) ;;
        none) fail "health succeeded but no listener was found on port $PORT" ;;
        *) fail "listener probe on port $PORT failed after health — ownership indeterminate" ;;
    esac
    # The pid this run spawned, in the terms the listener table reports: the
    # Windows pid on Git Bash, and `$!` itself on POSIX, where the subshell
    # execs the daemon and so keeps its own pid.
    if (( HOST_IS_WINDOWS == 1 )); then
        own_pid="$DAEMON_WIN_PID"
    else
        own_pid="$DAEMON_JOB_PID"
    fi
    [ "$LISTENER_PROBE_PID" = "$own_pid" ] ||
        fail "port $PORT is held by pid $LISTENER_PROBE_PID, not the daemon we spawned ($own_pid)"
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
        # `ps -o command=` reports the command line, so the recorded path is
        # either the whole of it or its first word — the same comparison
        # dev-runtime.sh already ships on POSIX.
        case "$PROCESS_IMAGE_VALUE" in
            "$SERVER_BIN" | "$SERVER_BIN "*) ;;
            *) fail "the listener's image is [$PROCESS_IMAGE_VALUE], not [$SERVER_BIN]" ;;
        esac
    fi
}

# Isolation readback, from the LIVE daemon, both halves.
smoke_assert_isolation() {
    local reported logged_root logged_cmp want_cmp
    reported="$(curl -sf --max-time 5 "$HOST/api/knowledge/path")" ||
        fail "pages readback request FAILED — unmeasured, not a match"
    [ "$reported" = "{\"path\":\"$NATIVE_PAGES_DIR\"}" ] ||
        fail "pages readback mismatch: got [$reported] want [{\"path\":\"$NATIVE_PAGES_DIR\"}]"
    # One process, no pipe: `sed … | head -1` CAN SIGPIPE sed, and under
    # pipefail that is indistinguishable from a parse failure.
    logged_root="$(sed -n '/Wenlan data root: /{s/.*Wenlan data root: //;s/\r$//;s/ (database [^)]*)$//;p;q;}' "$DATA_DIR/daemon.log")" ||
        fail "extracting the logged data root FAILED — unmeasured, not a match"
    [ -n "$logged_root" ] || fail "the daemon log has no 'Wenlan data root:' line at all"
    if (( HOST_IS_WINDOWS == 1 )); then
        # Windows spells one directory many ways; compare the way the production
        # guard does, folding separator and case and nothing else.
        logged_cmp="$(printf '%s' "$logged_root" | tr 'A-Z\\' 'a-z/')"
        want_cmp="$(printf '%s' "$NATIVE_DATA_DIR" | tr 'A-Z\\' 'a-z/')"
    else
        logged_cmp="$logged_root"
        want_cmp="$NATIVE_DATA_DIR"
    fi
    [ "$logged_cmp" = "$want_cmp" ] ||
        fail "data root mismatch: the daemon logged [$logged_root], scratch root is [$NATIVE_DATA_DIR]"
    echo "    isolated: pages=$NATIVE_PAGES_DIR data=$NATIVE_DATA_DIR"
}

# Exact (name,status) multiset against a FRESH ledger.
#
# "Not empty and no FAIL row" passes a ledger with an early PASS and every later
# step omitted, so the caller DERIVES its expectation from what this invocation
# asks the driver to do rather than pasting an observed run. The ledger is fresh
# by construction: GAUNTLET_OUT is inside this run's own mktemp directory.
#
# Both sides are status-checked AND required non-empty: two unchecked `sort`s
# once let both sides come out empty and compare EQUAL, passing a wrong ledger.
#
# args: the findings.tsv path, the expected `name=status` lines, and the label
#       used in the failure messages.
smoke_assert_ledger() {
    local tsv="$1" expected="$2" label="$3" actual_rows want_rows
    [ -s "$tsv" ] || fail "$label round-trip recorded nothing"
    actual_rows="$(cut -f2,3 "$tsv" | tr '\t' '=' | sort)" ||
        fail "reading the ledger FAILED — unmeasured, not a pass"
    want_rows="$(printf '%s\n' "$expected" | sort)" ||
        fail "sorting the expected ledger FAILED — unmeasured, not a pass"
    [ -n "$actual_rows" ] && [ -n "$want_rows" ] ||
        fail "ledger comparison degenerated to empty (actual=[$actual_rows] want=[$want_rows])"
    if [ "$actual_rows" != "$want_rows" ]; then
        echo "--- findings.tsv ---" >&2
        cat "$tsv" >&2
        echo "--- got ---" >&2; printf '%s\n' "$actual_rows" >&2
        echo "--- want --" >&2; printf '%s\n' "$want_rows" >&2
        fail "$label round-trip ledger is not the expected (name,status) multiset"
    fi
}
