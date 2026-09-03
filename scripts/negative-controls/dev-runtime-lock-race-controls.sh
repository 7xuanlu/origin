#!/usr/bin/env bash
# The stale-lock break in `acquire_runtime_lock`, driven by two REAL runs of the
# shipped script against one state directory.
#
# THE RACE. A run whose `mkdir` failed reads the owner file, measures that pid
# dead, and then removes the owner file and the directory and recreates them as
# its own. Between the read and the removals the dead holder's lock can be
# released and a LIVE run can take it -- `mkdir` succeeds the moment the
# directory is gone. The removals then destroy that live run's lock, the
# breaker's `mkdir` succeeds, and both runs believe they hold the worktree's
# isolated port and data directory.
#
# HOW IT IS DRIVEN, and why it is not sampled. The window is microseconds wide
# in ordinary running. `DEV_RUNTIME_RACE_SLEEP` (0 in every real run) widens it
# to seconds, so the interleaving is arranged rather than waited for. Each run
# appends ENTER before its hold and LEAVE after it to one witness file; a depth
# of 2 over that log is two runs inside the lock at the same moment.
#
# WHAT IS MEASURED, per arm:
#   shipped  -- every round must have exactly ONE run inside the lock. Both
#               halves matter: 0 double-holds AND no round where the refusal
#               locked everybody out.
#   reverted -- the re-read is deleted from a copy of the shipped file and every
#               round must double-hold. Without it a green shipped arm is
#               consistent with a harness that cannot see the defect at all.
#
# NOT COVERED. The residual window between the re-read and the `rm`/`rmdir` two
# lines later. `mkdir` has no primitive that tests and removes in one step, so
# that window cannot be closed, and `DEV_RUNTIME_RACE_SLEEP` does not sit inside
# it -- nothing here drives it.
#
# Run: bash scripts/negative-controls/dev-runtime-lock-race-controls.sh
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
runtime="$root/scripts/dev-runtime.sh"

runtime_before="$(cat "$runtime")"
failures=0

work="$(mktemp -d "${TMPDIR:-/tmp}/dev-runtime-lock-race-XXXXXX")" || exit 1
trap 'rm -rf "$work"' EXIT

# --- timings ------------------------------------------------------------------
# Wall clock, and every one of them is a margin over process startup on this
# platform rather than a tuned minimum. The breaker must reach the widened
# window before the harness releases the stale lock, and the live holder must
# still be inside its hold when the breaker wakes. A round that misses either is
# reported void rather than scored, so a host slower than these margins loses
# rounds loudly instead of passing quietly.
WIDEN_S=10       # DEV_RUNTIME_RACE_SLEEP for the breaker
RELEASE_AT_S=5   # when the dead holder "releases" and the live run starts
HOLDER_HOLD_S=13 # the live holder's time inside the lock
BREAKER_HOLD_S=2
ROUNDS="${WENLAN_NC_LOCK_RACE_ROUNDS:-6}"

# A pid the kernel answers ESRCH for, checked rather than assumed: an owner that
# is merely unmeasurable takes a different branch, and this harness would then
# drive nothing.
DEAD_PID=999999
# shellcheck source=../lib/host-process.sh
. "$root/scripts/lib/host-process.sh"
if ! errno_says_no_such_process "$DEAD_PID"; then
  echo "FATAL: pid $DEAD_PID is not measurably dead on this host; every round" >&2
  echo "       would refuse at the liveness test and drive no race" >&2
  exit 1
fi
if [[ "$runtime_before" != *'DEV_RUNTIME_RACE_SLEEP'* ]]; then
  echo "FATAL: dev-runtime.sh has no DEV_RUNTIME_RACE_SLEEP hook; the window" >&2
  echo "       cannot be widened and no round would be conclusive" >&2
  exit 1
fi

# --- the drivers --------------------------------------------------------------
# The subject is the WHOLE shipped file, not an extract: its traps, its globals
# and its library are what the race runs through. Only the final `case` dispatch
# is replaced, with an arm that takes the lock, records that it is inside it,
# holds, and leaves. The copy sits beside a copy of `lib/` because the script
# resolves its library from its own `BASH_SOURCE`.
cp -r "$root/scripts/lib" "$work/lib" || exit 1
if [[ "$(grep -c '^case "\${1:-}" in$' "$runtime")" != "1" ]]; then
  echo "FATAL: could not find exactly one dispatch in dev-runtime.sh" >&2
  exit 1
fi

build_driver() { # source-text, out-path
  # The three file-scope `refuse_production_path` calls are dropped, identically
  # from both arms. Each canonicalizes its argument and eleven production roots
  # through a separate `node -e`, which is about twelve seconds per run on this
  # host and would dominate every window this harness has to hit. They run
  # before anything the lock touches and neither arm differs in them.
  # The cut is made on the dispatch LINE ITSELF and not on a line number taken
  # from the shipped file: the reverted arm is shorter, and a fixed number would
  # chop two lines off the end of it and produce a driver that does not parse.
  printf '%s\n' "$1" \
    | awk '$0 == "case \"${1:-}\" in" { exit } { print }' \
    | grep -v '^refuse_production_path ' >"$2"
  cat >>"$2" <<'ARM'
acquire_runtime_lock || { RESULT_KIND=safety-refusal; printf 'RACE-REFUSED\n'; exit 3; }
printf 'ENTER %s\n' "$RACE_NAME" >>"$RACE_WITNESS"
sleep "$RACE_HOLD_S"
printf 'LEAVE %s\n' "$RACE_NAME" >>"$RACE_WITNESS"
RESULT_KIND=ok
printf 'RACE-HELD %s\n' "$RACE_NAME"
exit 0
ARM
}

# The re-read, deleted. This is the control: the anchor is the shipped text, it
# must occur exactly once, and the reverted arm below must double-hold every
# round or this harness is not watching the thing it names.
REREAD_ANCHOR='      recheck=0
      owner_again="$(sed -n '"'"'1p'"'"' "$LOCK_OWNER_FILE")" || recheck=$?
      if (( recheck != 0 )) || [[ "$owner_again" != "$owner" ]]; then'
occurrences=0
rest="$runtime_before"
while [[ "$rest" == *"$REREAD_ANCHOR"* ]]; do
  occurrences=$(( occurrences + 1 ))
  rest="${rest#*"$REREAD_ANCHOR"}"
done
if (( occurrences != 1 )); then
  echo "FATAL: the owner re-read anchor matched $occurrences times in" >&2
  echo "       dev-runtime.sh, wanted exactly 1; the reverted arm would test" >&2
  echo "       nothing" >&2
  exit 1
fi
# `if (( 0 ))` keeps the block's shape -- its `return 1` and its messages -- and
# removes only the QUESTION, so the arm differs from the shipped file in the
# measurement and in nothing else.
REVERTED="${runtime_before%%"$REREAD_ANCHOR"*}      if (( 0 )); then${runtime_before#*"$REREAD_ANCHOR"}"

build_driver "$runtime_before" "$work/driver-shipped.sh"
build_driver "$REVERTED" "$work/driver-reverted.sh"
for d in shipped reverted; do
  if ! bash -n "$work/driver-$d.sh"; then
    echo "FATAL: the $d driver does not parse" >&2
    exit 1
  fi
done

# --- one round ----------------------------------------------------------------
# Prints one word: double, single, lockout, or void.
run_round() { # driver, round-number
  local driver="$1" n="$2" base witness pa pb ev depth=0 max=0 enters=0
  base="$work/round-$n-$RANDOM"
  witness="$base.witness"
  mkdir -p "$base/runtime.lock" || { printf 'void\n'; return; }
  printf '%s\n' "$DEAD_PID" >"$base/runtime.lock/pid" || { printf 'void\n'; return; }
  : >"$witness" || { printf 'void\n'; return; }

  WENLAN_DEV_STATE_DIR="$base" RACE_NAME=breaker RACE_WITNESS="$witness" \
    RACE_HOLD_S="$BREAKER_HOLD_S" DEV_RUNTIME_RACE_SLEEP="$WIDEN_S" \
    bash "$driver" >"$base.breaker.log" 2>&1 &
  pa=$!
  sleep "$RELEASE_AT_S"
  # The dead holder releases and a live run takes the lock, both inside the
  # breaker's widened window. Removing the lock by hand is exactly what an
  # ordinary `release_runtime_lock` does one line at a time.
  rm -f "$base/runtime.lock/pid" && rmdir "$base/runtime.lock" 2>/dev/null
  WENLAN_DEV_STATE_DIR="$base" RACE_NAME=holder RACE_WITNESS="$witness" \
    RACE_HOLD_S="$HOLDER_HOLD_S" \
    bash "$driver" >"$base.holder.log" 2>&1 &
  pb=$!
  wait "$pa"
  wait "$pb"

  # The breaker reached the widened window only if its `mkdir` lost. If it was
  # slow enough to create the lock cleanly instead, the live holder refuses on a
  # LIVE owner and nothing about the stale break was exercised. That round is
  # void and is reported as such rather than scored.
  if grep -q "another dev runtime command is active" "$base.holder.log" 2>/dev/null; then
    printf 'void\n'
    return
  fi
  while read -r ev _; do
    case "$ev" in
      ENTER) depth=$(( depth + 1 )); enters=$(( enters + 1 ))
             if (( depth > max )); then max="$depth"; fi ;;
      LEAVE) depth=$(( depth - 1 )) ;;
    esac
  done <"$witness"
  if (( max >= 2 )); then printf 'double\n'
  elif (( enters == 1 )); then printf 'single\n'
  elif (( enters == 0 )); then printf 'lockout\n'
  else printf 'void\n'
  fi
}

run_arm() { # driver, label, expected-word
  local driver="$1" label="$2" want="$3" n verdict
  local double=0 single=0 lockout=0 void=0
  for (( n = 1; n <= ROUNDS; n++ )); do
    verdict="$(run_round "$driver" "$n")"
    case "$verdict" in
      double)  double=$(( double + 1 )) ;;
      single)  single=$(( single + 1 )) ;;
      lockout) lockout=$(( lockout + 1 )) ;;
      *)       void=$(( void + 1 )) ;;
    esac
    printf '    round %d: %s\n' "$n" "$verdict"
  done
  printf '  %s: double-holds %d/%d, single %d, lockout %d, void %d\n' \
    "$label" "$double" "$ROUNDS" "$single" "$lockout" "$void"
  local got=0
  case "$want" in
    single) got="$single" ;;
    double) got="$double" ;;
  esac
  if (( got != ROUNDS )); then
    printf '  FAIL %s: wanted %d/%d rounds to be "%s"\n' "$label" "$ROUNDS" "$ROUNDS" "$want"
    failures=$(( failures + 1 ))
  else
    printf '  ok   %s: %d/%d rounds "%s"\n' "$label" "$ROUNDS" "$ROUNDS" "$want"
  fi
}

echo "two real runs against one stale dev runtime lock:"
run_arm "$work/driver-shipped.sh" "shipped" single
echo "lock controls:"
echo "  nc-lock-stale-break-without-the-re-read  (the owner is not read again before the removals)"
run_arm "$work/driver-reverted.sh" "reverted" double

if [[ "$(cat "$runtime")" != "$runtime_before" ]]; then
  echo "FATAL: scripts/dev-runtime.sh changed during the run"; exit 1
fi

echo "CONTROL FAILURES: $failures"
(( failures == 0 )) || exit 1
