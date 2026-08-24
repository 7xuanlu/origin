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
    printf '%s' "$1" | tr '\t\r' '  ' | tr '\n' '|' | head -c 2000
}

_gauntlet_record() {
    local status="$1" name="$2" rc="$3" detail="$4"
    printf '%s\t%s\t%s\t%s\t%s\n' "$GAUNTLET_CHANNEL" "$name" "$status" "$rc" \
        "$(_gauntlet_escape "$detail")" >>"$GAUNTLET_TSV"
    printf '[%s] %s %s%s\n' "$status" "$name" "(rc=$rc)" "${detail:+ — $(printf '%s' "$detail" | head -c 200 | tr '\n' ' ')}"
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

collect() {
    local f
    for f in "$@"; do
        [ -e "$f" ] && cp -R "$f" "$GAUNTLET_OUT/logs/" 2>/dev/null
    done
    return 0
}

evaluate() {
    local fails
    printf '\n==> findings for %s\n' "$GAUNTLET_CHANNEL"
    awk -F'\t' '{ printf "  %-4s %-40s rc=%s\n", $3, $2, $4 }' "$GAUNTLET_TSV" 2>/dev/null
    fails="$(awk -F'\t' '$3 == "FAIL"' "$GAUNTLET_TSV" 2>/dev/null | wc -l | tr -d ' ')"
    printf '==> %s FAIL row(s)\n' "$fails"
    [ "$fails" = 0 ]
}
