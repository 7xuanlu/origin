#!/usr/bin/env bash
# First-run gauntlet helpers (bash). Source this file; never run it.
#
# Every channel script records checks into $GAUNTLET_OUT/findings.tsv and never
# exits early on a failed check, so one broken step still lets the later steps
# report. `evaluate` at the end turns any FAIL into a nonzero exit.
#
# TSV columns: channel, name, status (PASS|FAIL|INFO), rc, detail. Detail is a
# single line (tabs and newlines escaped) capped at 2000 characters; the full
# captured output of each check lives in $GAUNTLET_OUT/checks/<name>.log.
#
# Contract (all functions return 0 unless stated):
#   check NAME -- CMD ARGS...            PASS when CMD exits 0
#   check_output NAME SUBSTR -- CMD...   PASS when CMD exits 0 and output contains SUBSTR
#   check_fails NAME SUBSTR -- CMD...    PASS when CMD exits nonzero and output contains SUBSTR
#   info NAME VALUE                      informational row (never fails the run)
#   wait_health URL SECS                 poll /api/health; records seconds-to-health; returns 1 on timeout
#   assert_version URL EXPECTED          PASS when health .version == EXPECTED (leading v stripped)
#   stop_process PID [SECS]              TERM, wait up to SECS (15) for exit, then KILL;
#                                        records seconds-to-exit
#   daemon_postmortem BIN DATA_ROOT       after a failed health wait on a service-managed
#                                        daemon: collect its own logs, the launchd
#                                        record, and a foreground replay of the binary
#   collect FILE...                      copy files into $GAUNTLET_OUT/logs/
#   evaluate                             print the table; return 1 when any FAIL row exists
#
# Environment: GAUNTLET_OUT (default ./gauntlet-out), GAUNTLET_CHANNEL (default
# basename of the calling script), GAUNTLET_TIMEOUT (per-check cap, default 300s).

GAUNTLET_OUT="${GAUNTLET_OUT:-$PWD/gauntlet-out}"
GAUNTLET_CHANNEL="${GAUNTLET_CHANNEL:-$(basename "${0%.*}")}"
GAUNTLET_TIMEOUT="${GAUNTLET_TIMEOUT:-300}"
GAUNTLET_TSV="$GAUNTLET_OUT/findings.tsv"
mkdir -p "$GAUNTLET_OUT/checks" "$GAUNTLET_OUT/logs"

_gauntlet_escape() {
    # One line, no tabs, capped so the TSV stays readable in a step summary.
    # Substring, not `head -c`: head closing the pipe early makes tr/printf
    # print "write error: Broken pipe" into the job log on long outputs.
    local one
    one="$(printf '%s' "$1" | tr '\t\r' '  ' | tr '\n' '|')"
    printf '%s' "${one:0:2000}"
}

_gauntlet_record() {
    local status="$1" name="$2" rc="$3" detail="$4"
    local one
    one="$(_gauntlet_escape "$detail")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$GAUNTLET_CHANNEL" "$name" "$status" "$rc" "$one" >>"$GAUNTLET_TSV"
    printf '[%s] %s %s%s\n' "$status" "$name" "(rc=$rc)" "${one:+ — ${one:0:200}}"
}

_gauntlet_run() {
    # Runs the command with a cap, tees output to the check log, echoes rc.
    local name="$1"; shift
    local log="$GAUNTLET_OUT/checks/$name.log"
    local rc=0
    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM --kill-after=10s "$GAUNTLET_TIMEOUT" "$@" >"$log" 2>&1 || rc=$?
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout --signal=TERM --kill-after=10s "$GAUNTLET_TIMEOUT" "$@" >"$log" 2>&1 || rc=$?
    else
        "$@" >"$log" 2>&1 || rc=$?
    fi
    printf '%s' "$rc"
}

check() {
    local name="$1"; shift
    [ "${1:-}" = "--" ] && shift
    local rc
    rc="$(_gauntlet_run "$name" "$@")"
    local out
    out="$(cat "$GAUNTLET_OUT/checks/$name.log" 2>/dev/null)"
    if [ "$rc" = 0 ]; then
        _gauntlet_record PASS "$name" "$rc" "$out"
    else
        _gauntlet_record FAIL "$name" "$rc" "$out"
    fi
    return 0
}

check_output() {
    local name="$1" want="$2"; shift 2
    [ "${1:-}" = "--" ] && shift
    local rc
    rc="$(_gauntlet_run "$name" "$@")"
    local out
    out="$(cat "$GAUNTLET_OUT/checks/$name.log" 2>/dev/null)"
    if [ "$rc" = 0 ] && [[ "$out" == *"$want"* ]]; then
        _gauntlet_record PASS "$name" "$rc" "$out"
    else
        _gauntlet_record FAIL "$name" "$rc" "expected substring: $want; got: $out"
    fi
    return 0
}

check_fails() {
    local name="$1" want="$2"; shift 2
    [ "${1:-}" = "--" ] && shift
    local rc
    rc="$(_gauntlet_run "$name" "$@")"
    local out
    out="$(cat "$GAUNTLET_OUT/checks/$name.log" 2>/dev/null)"
    if [ "$rc" != 0 ] && [[ "$out" == *"$want"* ]]; then
        _gauntlet_record PASS "$name" "$rc" "$out"
    else
        _gauntlet_record FAIL "$name" "$rc" "expected nonzero exit with substring: $want; got: $out"
    fi
    return 0
}

info() {
    _gauntlet_record INFO "$1" 0 "${2:-}"
    return 0
}

wait_health() {
    local url="$1" secs="${2:-120}" i
    for i in $(seq 1 "$secs"); do
        if curl -sf --connect-timeout 1 --max-time 2 "$url" >/dev/null 2>&1; then
            info "seconds-to-health" "$i"
            return 0
        fi
        sleep 1
    done
    _gauntlet_record FAIL "health-timeout" 1 "no 200 from $url within ${secs}s"
    return 1
}

assert_version() {
    local url="$1" want="${2#v}" got body
    body="$(curl -sf --max-time 5 "$url" 2>/dev/null || true)"
    got="$(printf '%s' "$body" | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
    # Published builds report `X.Y.Z+g<sha>` (build metadata from build.rs);
    # compare the release part, keep the full string in the row.
    if [ "${got%%+*}" = "$want" ]; then
        _gauntlet_record PASS "health-version" 0 "$got"
    else
        _gauntlet_record FAIL "health-version" 1 "expected $want; health body: $body"
    fi
    return 0
}

stop_process() {
    # Graceful stop with a bounded wait. A `kill` followed by an immediate
    # process check races the daemon's own shutdown (seen on the arm runner),
    # and how long a clean exit takes is itself worth a row.
    local pid="$1" secs="${2:-15}" i=0
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt "$secs" ]; do
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
        info seconds-to-exit "still running ${secs}s after TERM; sent KILL"
    else
        info seconds-to-exit "$i"
    fi
}

daemon_postmortem() {
    # Call from teardown, after the service is stopped and before the data
    # root is deleted. The daemon owns its log files (the launchd plist sends
    # stdout/stderr to /dev/null), so those files plus a foreground replay
    # under the service's environment shape (cwd /, minimal PATH, the plist
    # env) are the only way to learn why a managed daemon never got healthy.
    local bin="$1" data_root="$2" log tail_text
    collect "$data_root/logs" "$HOME/Library/Logs/com.wenlan.server-fallback"
    for log in "$data_root/logs/wenlan-server.log" "$data_root/logs/wenlan-server.bootstrap.log"; do
        if [ -f "$log" ]; then
            tail_text="$(tail -c 1500 "$log")"
            info "daemon-log-$(basename "$log" .log)" "$tail_text"
        else
            info "daemon-log-$(basename "$log" .log)" "absent: $log"
        fi
    done
    if [ "$(uname -s)" = Darwin ]; then
        launchctl print "gui/$(id -u)/com.wenlan.server" >"$GAUNTLET_OUT/checks/launchctl-print-server.log" 2>&1 || true
        info launchd-record "$(grep -E 'state|runs|last exit|stdout path|stderr path' "$GAUNTLET_OUT/checks/launchctl-print-server.log" | tr -s '\t ' ' ')"
    fi
    if ! grep -q $'\thealth-timeout\tFAIL' "$GAUNTLET_TSV" 2>/dev/null; then
        return 0
    fi
    [ -x "$bin" ] || { info daemon-replay "skipped: $bin is not executable"; return 0; }
    local replay="$GAUNTLET_OUT/checks/daemon-replay.log" rc=0
    _replay_capped "$bin" "$replay" 127.0.0.1:17917 || rc=$?
    # 124: still running (healthy) at 30s, i.e. the binary is fine and the
    # fault is in how the service runs it. Anything else is the daemon's exit.
    info daemon-replay "rc=$rc (124 = still running at 30s) $(tail -c 1500 "$replay" | tr '\n' '|')"

    # Discriminating replay: identical environment shape, but with an explicit
    # absolute writable embedder cache. Default failing while this one lives
    # past init pins the crash on the cwd-relative fastembed default cache;
    # both failing points away from it (network/TLS under the service env).
    local replay2="$GAUNTLET_OUT/checks/daemon-replay-cache-dir.log" rc2=0
    mkdir -p "$data_root/replay-cache"
    _replay_capped "$bin" "$replay2" 127.0.0.1:17918 \
        FASTEMBED_CACHE_DIR="$data_root/replay-cache" || rc2=$?
    info daemon-replay-cache-dir "rc=$rc2 (124 = still running at 30s) $(tail -c 1500 "$replay2" | tr '\n' '|')"
}

# Run the daemon exactly as launchd would (cwd /, minimal env) for at most
# 30 s, then SIGKILL. `timeout` proved insufficient here: in the from-main
# gauntlet run a replay whose daemon survived init outlived the cap and hung
# every macOS job into its timeout-minutes, losing the verdict row. A KILL
# watchdog cannot be ignored or waited out. Exit 124 = killed while running.
# Usage: _replay_capped <bin> <logfile> <bind_addr> [EXTRA=env ...]
# Reads $data_root from the calling daemon_postmortem scope. The cap is
# GAUNTLET_REPLAY_CAP seconds (default 30; the test shortens it).
_replay_capped() {
    local bin="$1" log="$2" bind="$3" cap="${GAUNTLET_REPLAY_CAP:-30}"
    shift 3
    (
        cd / || exit 125
        env -i HOME="$HOME" USER="${USER:-$(id -un)}" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
            RUST_LOG=info WENLAN_DATA_DIR="$data_root" WENLAN_BIND_ADDR="$bind" \
            "$@" "$bin" >"$log" 2>&1 &
        pid=$!
        for _ in $(seq "$cap"); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            exit 124
        fi
        wait "$pid"
    )
}

collect() {
    local f
    for f in "$@"; do
        [ -e "$f" ] && cp -R "$f" "$GAUNTLET_OUT/logs/" 2>/dev/null
    done
    return 0
}

evaluate() {
    local fails
    # A run that recorded nothing is unchecked, never a pass: a channel that
    # dies before its first check must not evaluate green.
    if [ ! -s "$GAUNTLET_TSV" ]; then
        printf '\n==> no findings recorded (%s missing or empty): unchecked, not a pass\n' "$GAUNTLET_TSV"
        return 1
    fi
    printf '\n==> findings for %s\n' "$GAUNTLET_CHANNEL"
    awk -F'\t' '{ printf "  %-4s %-40s rc=%s\n", $3, $2, $4 }' "$GAUNTLET_TSV" 2>/dev/null
    fails="$(awk -F'\t' '$3 == "FAIL"' "$GAUNTLET_TSV" 2>/dev/null | wc -l | tr -d ' ')"
    printf '==> %s FAIL row(s)\n' "$fails"
    [ "$fails" = 0 ]
}
