#!/usr/bin/env bash
# Behaviour cases and negative controls for scripts/first-run/port-precheck.sh.
#
# Round 13d, new finding 2: the port precheck was rewritten to stop reporting a
# probe that could not run as "port free", and to stop losing the verdict when
# the ledger cannot be written -- and nothing anywhere in the repository would
# have noticed if either half were reverted. The only check ever run against it
# was `bash -n`, which proves the file parses.
#
# So this does two things, in the order that matters:
#
#   1. CASES -- drive the shipped script for real, over a stubbed listener
#      table, and assert the row it writes and the status it exits with. These
#      are the tests the script did not have.
#   2. CONTROLS -- revert one half of one remedy in a COPY of the script, rerun
#      the cases, and FAIL if the case that defends it still passes. A green
#      case list does not prove the cases would notice the bug.
#
# The shipped script and library are never written to; both are asserted
# unchanged at the end.
#
# Run: bash scripts/negative-controls/port-precheck-controls.sh
set -uo pipefail

# ROUND 3 (Codex Sol), FINDING N7. A terminal completion marker, printed last
# and only after every control has been scored. Without one, a harness killed
# partway and a harness that found nothing leave the same tail, and the
# aggregate runner cannot tell "0 control failures" from "it never got there".
MARKER="NEGATIVE-CONTROL COMPLETE"
HARNESS="port-precheck-controls.sh"
started=$SECONDS

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$root/scripts/first-run/port-precheck.sh"
lib="$root/scripts/lib/host-process.sh"
logs="$root/target/negative-control-logs"
mkdir -p "$logs"

script_before="$(cat "$script")"
lib_before="$(cat "$lib")"

work="$(mktemp -d "${TMPDIR:-/tmp}/port-precheck-controls-XXXXXX")" || exit 1
reached_end=0
finish() {
  local rc=$?
  rm -rf "$work"
  if (( reached_end )); then
    echo "$MARKER $HARNESS failures=$failures elapsed=$((SECONDS - started))s"
  else
    # Every way out that is not the bottom of the file lands here: a FATAL
    # refusal, a `set -u` abort, a signal, a watchdog kill. Saying so is the
    # point -- a run that stopped early must not be readable as a run that
    # scored every control and found nothing.
    # ...and it must not exit 0 either. Falling off the file early -- a `return`
    # at top level, a subshell that ended cleanly, a SIGTERM that lands after a
    # successful command -- reaches here with rc=0, and a supervisor reading
    # only the status would call that a clean run.
    #
    # ROUND 4: settle the number BEFORE printing it. The rewrite used to happen
    # after, so on the one path where it matters the line said `rc=0` while the
    # process exited 1, and the line exists precisely so a reader does not have
    # to reconcile those two.
    local inherited=$rc
    (( rc )) || rc=1
    echo "NEGATIVE-CONTROL ABORTED $HARNESS rc=$rc elapsed=$((SECONDS - started))s"
    if (( inherited != rc )); then
      echo "  (the run ended with status $inherited; a run that did not score"
      echo "   every control must not exit 0, so this exits $rc)"
    fi
    exit "$rc"
  fi
}
trap finish EXIT

failures=0
PORT=17931
# The row this script writes is read back by a DIFFERENT process: the channel
# starts afterwards, finds the row above its own window mark, and restates its
# verdict as one of its own rows. In a reused GAUNTLET_OUT the carried region
# holds every earlier run's rows too, so without something in the row naming the
# run that took the measurement, one PASS left behind by an earlier run is
# carried into every later one. `free` asserts the token reaches the detail
# column; `no-run-token` asserts an unbindable row is a FAIL and not a pass.
RUN_TOKEN="control-run-77"

# --- the listener table the probe reads --------------------------------------
# Real netstat shape on Windows; the POSIX branch of the probe reads lsof, so
# both stubs are written and whichever the host reaches is the one that answers.
#
# The UDP row is not decoration. `netstat -ano` prints the whole TCP table and
# then the whole UDP table, so a UDP row is the probe's witness that the TCP
# section ENDED rather than stopped early -- without one, a fixture stands for a
# TRUNCATED table and the probe answers "could not measure" for every port in
# it, which is a different case from the two these tables are for.
table_head=$'\nActive Connections\n\n  Proto  Local Address          Foreign Address        State           PID\n  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576\n  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4\n'
table_udp=$'  UDP    0.0.0.0:5353           *:*                                    9528\n'
table_without_port="$table_head$table_udp"
table_with_port="$table_head"'  TCP    127.0.0.1:'"$PORT"'         0.0.0.0:0              LISTENING       4242
'"$table_udp"

write_stubs() { # dir, netstat-body, lsof-body
  mkdir -p "$1"
  printf '#!/usr/bin/env bash\n%s\n' "$2" >"$1/netstat"
  printf '#!/usr/bin/env bash\n%s\n' "$3" >"$1/lsof"
  chmod 0755 "$1/netstat" "$1/lsof"
}

emit() { # a bash body that prints "$1" verbatim
  printf 'cat <<'"'"'TABLE'"'"'\n%sTABLE\n' "$1"
}

# --- one case ----------------------------------------------------------------
# Prints "ok" / "FAIL ..." and returns 0 when the case passed.
run_case() { # name, script_under_test, mode, expect_rc, expect_row, expect_stderr
  local name="$1" subject="$2" mode="$3" want_rc="$4" want_row="$5" want_err="$6"
  local case_dir="$work/case-$RANDOM$RANDOM"
  local bin="$case_dir/bin" out="$case_dir/out" rc=0 row="" stderr_text=""
  local token="$RUN_TOKEN"
  mkdir -p "$bin"

  case "$mode" in
    free)
      write_stubs "$bin" "$(emit "$table_without_port")" 'exit 1'
      mkdir -p "$out"
      ;;
    busy)
      write_stubs "$bin" "$(emit "$table_with_port")" \
        'echo 4242; exit 0'
      mkdir -p "$out"
      ;;
    unmeasurable)
      # A listener table that cannot run at all. This is the state the old
      # `|| true` spelled as "free".
      write_stubs "$bin" 'exit 1' 'exit 1'
      mkdir -p "$out"
      ;;
    ledger-unwritable)
      write_stubs "$bin" "$(emit "$table_without_port")" 'exit 1'
      mkdir -p "$out/findings.tsv"   # a DIRECTORY where the ledger goes
      ;;
    out-uncreatable)
      write_stubs "$bin" "$(emit "$table_without_port")" 'exit 1'
      mkdir -p "$case_dir"
      printf 'not a directory\n' >"$case_dir/blocker"
      out="$case_dir/blocker/nested"
      ;;
    no-run-token)
      # The port is free and perfectly measurable. The only thing missing is the
      # value that ties the row to THIS run -- and a row nothing downstream can
      # attribute is not evidence, so it must FAIL rather than leave one more
      # anonymous PASS for the next run into this directory to inherit.
      write_stubs "$bin" "$(emit "$table_without_port")" 'exit 1'
      mkdir -p "$out"
      token=""
      ;;
    *) printf '  FAIL %-22s unknown mode %s\n' "$name" "$mode"; return 1 ;;
  esac

  local shim_path="$bin"
  if command -v cygpath >/dev/null 2>&1; then shim_path="$(cygpath -u "$bin")"; fi
  local err_file="$case_dir/stderr"
  # `env -u` and not an empty assignment: this harness can itself be run from a
  # shell that exports GAUNTLET_RUN_TOKEN, and a case about an UNSET variable
  # that quietly inherits one measures nothing.
  if [[ -n "$token" ]]; then
    PATH="$shim_path:$PATH" GAUNTLET_OUT="$out" GAUNTLET_CHANNEL="control" \
      GAUNTLET_RUN_TOKEN="$token" \
      bash "$subject" "$PORT" >"$case_dir/stdout" 2>"$err_file" || rc=$?
  else
    PATH="$shim_path:$PATH" GAUNTLET_OUT="$out" GAUNTLET_CHANNEL="control" \
      env -u GAUNTLET_RUN_TOKEN \
      bash "$subject" "$PORT" >"$case_dir/stdout" 2>"$err_file" || rc=$?
  fi
  stderr_text="$(cat "$err_file" 2>/dev/null || true)"
  if [[ -f "$out/findings.tsv" ]]; then row="$(cat "$out/findings.tsv")"; fi

  local why=""
  [[ "$rc" == "$want_rc" ]] || why="exit $rc, wanted $want_rc"
  if [[ -z "$why" && -n "$want_row" ]]; then
    [[ "$row" == *"$want_row"* ]] || why="row [$row] does not contain [$want_row]"
  fi
  if [[ -z "$why" && -n "$want_err" ]]; then
    [[ "$stderr_text" == *"$want_err"* ]] ||
      why="stderr [$stderr_text] does not contain [$want_err]"
  fi
  if [[ -n "$why" ]]; then
    printf '  FAIL %-22s %s\n' "$name" "$why"
    return 1
  fi
  printf '  ok   %-22s\n' "$name"
  return 0
}

# name | mode | rc | row substring | stderr substring
#
# `busy` and `unmeasurable` exit 1, not 0. The script used to end in an
# unconditional `exit 0`, which made the workflow's "Port 7878 precheck" step
# green on a busy port and green on a port that could not be measured -- the row
# said FAIL and nothing read it, because a precheck runs before any Evaluate and
# the POSIX channels have none. The row and the exit status now carry the same
# verdict, and these two rows are what pin that.
#
# `free` asserts the RUN TOKEN in the detail column, not merely the word "free".
# The row is read back by the channel process that starts after this one, out of
# a carried region that in a reused GAUNTLET_OUT also holds every earlier run's
# rows; the token is the only thing in the row that says which run measured the
# port. Drop it and this case still says "measured free" while the row it
# describes is indistinguishable from a corpse.
CASES=(
  "free|free|0|	PASS	0	measured free; run=control-run-77|"
  "busy|busy|1|	FAIL	0	BUSY: pid 4242|"
  "unmeasurable|unmeasurable|1|	FAIL	0	could not measure|"
  "no-run-token|no-run-token|1|	FAIL	0	no run token to bind this precheck to|"
  "ledger-unwritable|ledger-unwritable|3||was measured and LOST"
  "out-uncreatable|out-uncreatable|3||cannot create"
)

run_all() { # subject, indent-label -> prints, sets PASSED_CASES
  local subject="$1"
  PASSED_CASES=()
  FAILED_CASES=()
  local spec name mode rc rowtext errtext
  for spec in "${CASES[@]}"; do
    IFS='|' read -r name mode rc rowtext errtext <<<"$spec"
    if run_case "$name" "$subject" "$mode" "$rc" "$rowtext" "$errtext"; then
      PASSED_CASES+=("$name")
    else
      FAILED_CASES+=("$name")
    fi
  done
}

echo "port-precheck-controls"
echo "cases against the shipped script:"
run_all "$script"
if (( ${#FAILED_CASES[@]} )); then
  failures=$(( failures + ${#FAILED_CASES[@]} ))
  # ROUND 4 (Codex Sol). This used to say the controls below "would measure
  # nothing" and then run them anyway. A case that is ALREADY red on the
  # shipped script comes back red under every mutation too, so every control
  # naming it prints `ok   caught:` and is credited for a failure it did not
  # cause -- which is finding N1 from round 3, in the harness that was written
  # to answer it. Refuse instead of narrate: a baseline that is not green makes
  # the controls unscoreable, not merely doubtful.
  echo "  the shipped script does not pass its own cases, so no mutation below"
  echo "  could be told apart from the red that is already there:"
  for _red in "${FAILED_CASES[@]}"; do echo "    $_red"; done
  echo "  refusing to score any control against a baseline that is not green"
  # A refusal is a completed measurement, not an abort: every control this run
  # will ever score has been scored, and the answer is that they cannot be. So
  # the terminal marker is owed -- `reached_end` lets the EXIT trap print it,
  # and the non-zero status carries the verdict.
  echo "CONTROL FAILURES: $failures"
  reached_end=1
  exit 1
fi

# --- the controls ------------------------------------------------------------
# Each reverts ONE half of ONE remedy in a copy, then requires that the named
# case notices. `must_survive` is the guard against a mutation that simply
# breaks everything: a control that reddens the whole list has not localised
# anything.
# How many times a literal string occurs in another.
count_occurrences() { # haystack, needle -> the count on stdout
  local rest="$1" needle="$2" n=0
  [[ -n "$needle" ]] || { printf '0'; return; }
  while [[ "$rest" == *"$needle"* ]]; do
    n=$((n + 1))
    rest="${rest#*"$needle"}"
  done
  printf '%s' "$n"
}

mutate() { # old, new -> writes $work/subject/first-run/port-precheck.sh
  # The copy keeps the script's own directory layout, because the script finds
  # the library through "$here/../lib/host-process.sh"; the library is copied
  # too, so nothing under scripts/ is opened for writing by this harness.
  local old="$1" new="$2" src="$work/subject" text head tail hits
  rm -rf "$src"
  mkdir -p "$src/first-run" "$src/lib"
  cp "$lib" "$src/lib/host-process.sh"
  text="$(cat "$script")"
  # An anchor that matched zero times, or more than once, would silently mutate
  # nothing or the wrong thing -- a control that measures nothing while
  # reporting that it did.
  #
  # That is what this comment has always said and what the test below did not
  # do. `head="${text%%"$old"*}"` stops at the FIRST occurrence, so for a
  # doubled anchor `head$old$tail` still recombines into `$text` and `$head` is
  # still shorter than it: the check saw "at least once" and called it "exactly
  # once". The first copy was mutated, the second left alone, and the subject
  # was a third program that nobody wrote -- reported on as if it were the
  # reverted one. Counted now.
  hits="$(count_occurrences "$text" "$old")"
  if [[ "$hits" != 1 ]]; then
    printf '    FAIL anchor matched %s times, wanted exactly 1; this control would test nothing\n' \
      "$hits"
    return 1
  fi
  # ROUND 4 (Codex Sol), FINDING C5.1. Matching once is not the same as
  # changing something. These control blocks are written by copying the one
  # above and editing two strings, and editing neither leaves a mutation that
  # replaces a real span of the file with exactly what was already there. It
  # then runs, every case passes, and the control is reported as "the suite
  # PASSED with the fix reverted" -- true, and about the wrong file.
  if [[ "$old" == "$new" ]]; then
    printf '    FAIL the replacement is identical to the anchor; this control reverts nothing\n'
    return 1
  fi
  head="${text%%"$old"*}"
  tail="${text#*"$old"}"
  printf '%s\n' "$head$new$tail" >"$src/first-run/port-precheck.sh"
  return 0
}

control() { # name, why, old, new, must_fail..., --, must_survive...
  local name="$1" why="$2" old="$3" new="$4"; shift 4
  local -a must_fail=() must_survive=() bucket=must_fail
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then bucket=must_survive; continue; fi
    if [[ "$bucket" == must_fail ]]; then must_fail+=("$arg"); else must_survive+=("$arg"); fi
  done
  printf '  %s  (%s)\n' "$name" "$why"
  if ! mutate "$old" "$new"; then failures=$((failures + 1)); return; fi
  run_all "$work/subject/first-run/port-precheck.sh" >"$logs/$name.log" 2>&1
  local want
  for want in "${must_fail[@]}"; do
    if printf '%s\n' "${FAILED_CASES[@]:-}" | grep -qx -- "$want"; then
      printf '    ok   caught:   %s\n' "$want"
    else
      printf '    FAIL survived: %s -- the case does not defend this fix\n' "$want"
      failures=$((failures + 1))
    fi
  done
  # ROUND 3 (Codex Sol), FINDING N2. This walked `must_survive` and nothing
  # else. Today every control happens to name all five cases, so the two lists
  # partition the suite -- but that is a coincidence of the current case count,
  # not a property of the code. A sixth case added to CASES would be scored by
  # nothing, and a mutation that reddened it would still be reported as a
  # control that fired on exactly the case it names.
  #
  # The survivor set is now taken from CASES minus this control's own must_fail
  # list, so it cannot fall behind the suite.
  local case_spec case_name
  for case_spec in "${CASES[@]}"; do
    case_name="${case_spec%%|*}"
    if printf '%s\n' "${must_fail[@]:-}" | grep -qx -- "$case_name"; then continue; fi
    if printf '%s\n' "${PASSED_CASES[@]:-}" | grep -qx -- "$case_name"; then
      printf '    ok   survived: %s\n' "$case_name"
    else
      printf '    FAIL also failed: %s -- the control is not pinned to the fix\n' "$case_name"
      failures=$((failures + 1))
    fi
  done
  # `must_survive` stays as the control's own statement of what it is pinned
  # to; it must agree with the computed set rather than replace it.
  for want in "${must_survive[@]}"; do
    if printf '%s\n' "${must_fail[@]:-}" | grep -qx -- "$want"; then
      printf '    FAIL %s is named in both must_fail and must_survive\n' "$want"
      failures=$((failures + 1))
    elif ! printf '%s\n' "${CASES[@]%%|*}" | grep -qx -- "$want"; then
      printf '    FAIL must_survive names %s, which is not a case in CASES\n' "$want"
      failures=$((failures + 1))
    fi
  done
}

echo "controls:"

control nc-unmeasurable-is-free \
  'a listener table that could not run is recorded as a free port' \
  '    *)
      st=FAIL
      detail="could not measure whether $port is free; recorded as unusable, not as free; run=$run_token"
      ;;' \
  '    *)
      st=PASS
      detail="measured free; run=$run_token"
      ;;' \
  unmeasurable -- free busy no-run-token ledger-unwritable out-uncreatable

control nc-busy-is-informational \
  'a busy shared port is recorded as INFO, which no ledger rule counts' \
  '      st=FAIL
      detail="BUSY: pid $LISTENER_PROBE_PID is listening on $port; run=$run_token"' \
  '      st=INFO
      detail="BUSY: pid $LISTENER_PROBE_PID is listening on $port; run=$run_token"' \
  busy -- free unmeasurable no-run-token ledger-unwritable out-uncreatable

control nc-ledger-loss-is-not-fatal \
  'the measured verdict row is lost and the script still exits 0' \
  '  echo "[LEDGER] exiting 3 rather than letting an unrecorded FAIL read as a pass" >&2
  exit 3
fi' \
  '  echo "[LEDGER] exiting 3 rather than letting an unrecorded FAIL read as a pass" >&2
  :
fi' \
  ledger-unwritable -- free busy unmeasurable no-run-token out-uncreatable

control nc-verdict-swallowed-at-exit \
  'the measured verdict is recorded and then thrown away by an unconditional exit 0' \
  '[ "$st" = PASS ]' \
  'exit 0' \
  busy unmeasurable no-run-token -- free ledger-unwritable out-uncreatable

control nc-outdir-unchecked \
  'the output directory cannot be created and the script says the wrong thing' \
  'if ! mkdir -p "$GAUNTLET_OUT" 2>/dev/null; then
  echo "[LEDGER] cannot create $GAUNTLET_OUT; the port row cannot be recorded" >&2
  exit 3
fi' \
  'mkdir -p "$GAUNTLET_OUT" 2>/dev/null || true' \
  out-uncreatable -- free busy unmeasurable no-run-token ledger-unwritable

# The half of the run-token remedy that lives in THIS script: an unset token is
# a FAIL, not a shrug. Revert it and the precheck reports a pass for a row no
# later reader can attribute to a run -- which is the corpse the channel then
# carries in, exactly as it did before the token existed.
control nc-unbound-precheck-is-a-pass \
  'a precheck with no run token to bind it to is recorded as a passing measurement' \
  '  st=FAIL
  detail="no run token to bind this precheck to: GAUNTLET_RUN_TOKEN is unset, so this row cannot be told from one an earlier run left in this reused GAUNTLET_OUT"' \
  '  st=PASS
  detail="measured free"' \
  no-run-token -- free busy unmeasurable ledger-unwritable out-uncreatable

# --- the harness's own aim ---------------------------------------------------
if [[ "$(cat "$script")" != "$script_before" ]]; then
  echo "FATAL: scripts/first-run/port-precheck.sh changed during the run"
  exit 1
fi
if [[ "$(cat "$lib")" != "$lib_before" ]]; then
  echo "FATAL: scripts/lib/host-process.sh changed during the run"
  exit 1
fi

echo "CONTROL FAILURES: $failures"
reached_end=1
(( failures == 0 )) || exit 1
