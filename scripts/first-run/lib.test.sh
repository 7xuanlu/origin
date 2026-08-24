#!/usr/bin/env bash
# Tests for scripts/first-run/lib.sh and summary.py. Run: bash scripts/first-run/lib.test.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/lib-test.XXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT

failures=0
assert() {
    # assert DESCRIPTION CONDITION...
    local desc="$1"; shift
    if "$@"; then
        echo "  ok   $desc"
    else
        echo "  FAIL $desc" >&2
        failures=$((failures + 1))
    fi
}

# row NAME -> "STATUS<TAB>RC<TAB>DETAIL" for the single TSV row with that name
row() {
    awk -F'\t' -v n="$1" '$2 == n { print $3 "\t" $4 "\t" $5 }' "$GAUNTLET_TSV"
}
row_status() { row "$1" | cut -f1; }
row_rc() { row "$1" | cut -f2; }
row_detail() { row "$1" | cut -f3; }
row_count() { awk -F'\t' -v n="$1" '$2 == n' "$GAUNTLET_TSV" | wc -l | tr -d ' '; }
# [[ is a keyword, not a command, so assert needs real commands for glob tests.
starts_with() { case "$1" in "$2"*) return 0 ;; *) return 1 ;; esac; }
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }

echo "== lib.sh"
export GAUNTLET_OUT="$tmp/run/findings-test-macos"
export GAUNTLET_CHANNEL="unit"
# shellcheck source=scripts/first-run/lib.sh
. "$here/lib.sh" >/dev/null

check ok-true -- true >/dev/null
check bad-false -- false >/dev/null
check exit-7 -- bash -c 'echo boom; exit 7' >/dev/null
assert "check PASS for true"        [ "$(row_status ok-true)" = PASS ]
assert "check FAIL for false"       [ "$(row_status bad-false)" = FAIL ]
assert "check FAIL rc=1 for false"  [ "$(row_rc bad-false)" = 1 ]
assert "check records real rc"      [ "$(row_rc exit-7)" = 7 ]
assert "check detail is the output" [ "$(row_detail exit-7)" = boom ]
assert "check log holds output"     [ "$(cat "$GAUNTLET_OUT/checks/exit-7.log")" = boom ]

check_output co-pass needle -- echo "hay needle stack" >/dev/null
check_output co-miss needle -- echo "hay stack" >/dev/null
check_output co-rc needle -- bash -c 'echo needle; exit 1' >/dev/null
assert "check_output PASS when substring present"  [ "$(row_status co-pass)" = PASS ]
assert "check_output FAIL when substring absent"   [ "$(row_status co-miss)" = FAIL ]
assert "check_output FAIL detail names the want"   starts_with "$(row_detail co-miss)" "expected substring: needle;"
assert "check_output FAIL on nonzero rc"           [ "$(row_status co-rc)" = FAIL ]

check_fails cf-pass boom -- bash -c 'echo boom; exit 3' >/dev/null
check_fails cf-zero boom -- bash -c 'echo boom' >/dev/null
check_fails cf-nosub boom -- bash -c 'echo quiet; exit 3' >/dev/null
assert "check_fails PASS for nonzero + substring" [ "$(row_status cf-pass)" = PASS ]
assert "check_fails keeps the rc"                 [ "$(row_rc cf-pass)" = 3 ]
assert "check_fails FAIL when command exits 0"    [ "$(row_status cf-zero)" = FAIL ]
assert "check_fails FAIL when substring absent"   [ "$(row_status cf-nosub)" = FAIL ]

info note "hello world" >/dev/null
assert "info records INFO"        [ "$(row_status note)" = INFO ]
assert "info detail is the value" [ "$(row_detail note)" = "hello world" ]

check multi -- printf 'a\tb\nc\n' >/dev/null
assert "multi-line output is one TSV row"   [ "$(row_count multi)" = 1 ]
assert "detail is tab-free, newline as |"   [ "$(row_detail multi)" = "a b|c" ]
assert "every TSV row has exactly 5 fields" [ "$(awk -F'\t' 'NF != 5' "$GAUNTLET_TSV" | wc -l | tr -d ' ')" = 0 ]

# 5000 chars of output: the detail is capped at 2000 and nothing on stderr
# (a `head -c` truncation used to print "write error: Broken pipe").
long_err="$(check long -- bash -c 'head -c 5000 /dev/zero | tr "\0" x' 2>&1 >/dev/null)"
assert "long detail capped at 2000 chars" [ "$(row_detail long | wc -c | tr -d ' ')" = 2001 ]
assert "long detail truncation is silent" [ -z "$long_err" ]

# The victim must not carry this script's EXIT trap: a bare `sleep 300 &` forks
# a bash child that, if the TERM below lands before its exec, runs the trap and
# deletes $tmp under the running test. Reset the trap, then exec.
(trap - EXIT; exec sleep 300) &
victim=$!
disown "$victim"  # keep bash's "Terminated" job notice out of the test output
stop_process "$victim" 5 >/dev/null
gone() { ! kill -0 "$1" 2>/dev/null; }
assert "stop_process ends the process"    gone "$victim"
assert "stop_process records seconds-to-exit" [ "$(row_status seconds-to-exit)" = INFO ]
assert "stop_process on a dead pid is a no-op" [ "$(stop_process "$victim" 1 | wc -l | tr -d ' ')" = 0 ]

# port 9 (discard) is closed on every runner; the 1s window keeps this quick.
if wait_health "http://127.0.0.1:9/api/health" 1 >/dev/null; then wh=0; else wh=$?; fi
assert "wait_health returns 1 on timeout"     [ "$wh" = 1 ]
assert "wait_health records health-timeout"   [ "$(row_status health-timeout)" = FAIL ]

if evaluate >/dev/null; then ev=0; else ev=$?; fi
assert "evaluate returns 1 when a FAIL row exists" [ "$ev" = 1 ]

# A fresh sourcing with a clean output dir: no FAIL rows -> evaluate returns 0.
clean_rc="$(
    GAUNTLET_OUT="$tmp/run/findings-clean-linux" GAUNTLET_CHANNEL=clean bash -c '
        . "$1/lib.sh" >/dev/null
        check fine -- true >/dev/null
        info seconds-to-health 3 >/dev/null
        if evaluate >/dev/null; then echo 0; else echo $?; fi
    ' _ "$here"
)"
assert "evaluate returns 0 without FAIL rows" [ "$clean_rc" = 0 ]

# A channel that recorded nothing (missing or empty TSV) is unchecked, never a
# pass: evaluate must fail rather than read an absent file as zero FAIL rows.
empty_rc="$(
    GAUNTLET_OUT="$tmp/run/findings-empty-linux" GAUNTLET_CHANNEL=empty bash -c '
        . "$1/lib.sh" >/dev/null
        rm -f "$GAUNTLET_TSV"
        if evaluate >/dev/null; then echo 0; else echo $?; fi
    ' _ "$here"
)"
assert "evaluate returns 1 when no findings were recorded" [ "$empty_rc" = 1 ]

# The post-mortem replay watchdog must SIGKILL a daemon that ignores TERM and
# report 124: a TERM-only cap let a recovered daemon hang the job past its
# timeout-minutes and lose the verdict row (from-main run, all macOS legs).
cat >"$tmp/stubborn.sh" <<'EOS'
#!/bin/sh
trap '' TERM
while :; do sleep 1; done
EOS
chmod +x "$tmp/stubborn.sh"
watchdog_rc="$(
    GAUNTLET_OUT="$tmp/run/findings-watchdog-macos" GAUNTLET_CHANNEL=watchdog \
    GAUNTLET_REPLAY_CAP=2 bash -c '
        . "$1/lib.sh" >/dev/null
        data_root="$GAUNTLET_OUT"
        if _replay_capped "$2" "$GAUNTLET_OUT/checks/stubborn.log" 127.0.0.1:17999; then
            echo 0
        else
            echo $?
        fi
    ' _ "$here" "$tmp/stubborn.sh"
)"
assert "replay watchdog kills a TERM-ignoring daemon with rc 124" [ "$watchdog_rc" = 124 ]

echo "== summary.py"
summary="$(python3 "$here/summary.py" "$tmp/run")"
assert "table header"              grep -q '^| Channel | Label | PASS | FAIL | INFO | seconds-to-health | Worst |$' <<<"$summary"
assert "unit row is Worst=FAIL"    grep -q '^| unit | findings-test-macos | .* | FAIL |$' <<<"$summary"
assert "clean row is Worst=INFO with health seconds" grep -q '^| clean | findings-clean-linux | 1 | 0 | 1 | 3 | INFO |$' <<<"$summary"
assert "Findings section present" grep -q '^### Findings$' <<<"$summary"
assert "FAIL row listed with rc"   grep -q '^- \*\*unit / exit-7\*\* (findings-test-macos, rc=7): boom$' <<<"$summary"
assert "INFO row listed"           grep -q '^- \*\*unit / note\*\* (findings-test-macos, rc=0): hello world$' <<<"$summary"
fail_line="$(grep -n "unit / bad-false" <<<"$summary" | cut -d: -f1)"
info_line="$(grep -n "unit / note" <<<"$summary" | cut -d: -f1)"
assert "FAIL rows precede INFO rows" [ "$fail_line" -lt "$info_line" ]
empty="$(python3 "$here/summary.py" "$tmp/nothing-here")"
assert "empty dir still prints a line" contains "$empty" "No findings.tsv"

if [ "$failures" != 0 ]; then
    echo "FAIL: lib.test.sh ($failures assertion(s))" >&2
    exit 1
fi
echo "PASS: lib.test.sh"
