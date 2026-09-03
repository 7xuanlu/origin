#!/usr/bin/env bash
# The stale-lock break in `acquire_runtime_lock`, driven by two REAL runs of the
# shipped script against one state directory.
#
# THE RACE. A run whose `mkdir` failed reads the owner file, measures that pid
# dead, and then destroys the lock and recreates it as its own. Between the read
# and the destruction the dead holder's lock can be released and a LIVE run can
# take it -- `mkdir` succeeds the moment the directory is gone. A destruction
# made on the old measurement then destroys that live run's lock, the breaker's
# `mkdir` succeeds, and both runs believe they hold the worktree's isolated port
# and data directory. Worse, the breaker cannot tell: its own release compares
# its own token, finds it, and prints `DEV_RUNTIME_RESULT: ok`. Only the victim
# ever notices.
#
# TWO WINDOWS, AND THEY ARE DEFENDED BY DIFFERENT THINGS. The shipped code
# re-reads the owner immediately before it breaks, and it breaks by ATOMIC
# RENAME rather than by `rm` plus `rmdir`.
#
#   window 1, before the re-read -- the re-read closes it: the lock now names
#             somebody else, so this run's measurement is stale and it refuses.
#   window 2, between the re-read and the break -- only the rename closes it. A
#             read is not a removal, so no amount of re-reading can make two
#             destructive steps safe; `rename` is the removal AND the test in
#             one step, so at most one process can ever destroy a given
#             GENERATION of the lock, and a breaker that finds it renamed away
#             somebody else's generation puts it back and goes round.
#
# HOW IT IS DRIVEN, and why it is not sampled. Both windows are microseconds
# wide in ordinary running. `DEV_RUNTIME_RACE_SLEEP` (window 1) and
# `DEV_RUNTIME_RACE_SLEEP_BREAK` (window 2) are 0 in every real run and widen
# one window each to seconds, so the interleaving is arranged rather than waited
# for. Each run appends ENTER before its hold and LEAVE after it to one witness
# file; a depth of 2 over that log is two runs inside the lock at the same
# moment.
#
# WHAT IS MEASURED, per arm. Each shipped arm is read against the revert that
# defends ITS window, because a remedy is only shown to work where its own
# absence is shown to break:
#   shipped/window-1  -- exactly ONE run inside the lock, every round.
#   shipped/window-2  -- the same, for the window the re-read cannot close.
#   reverted/window-1 -- the re-read deleted AND the rename put back to
#                        `rm`+`rmdir`: the original defect, and it must
#                        double-hold every round. Deleting the re-read ALONE no
#                        longer double-holds here, and that is not a gap: the
#                        rename defends window 1 as well, so the arm has to take
#                        both halves away to expose the window.
#   reverted/window-2 -- the rename put back to `rm`+`rmdir`, re-read intact:
#                        this is the arm that measures the rename on its own,
#                        and it must double-hold every round. Without it a green
#                        shipped/window-2 arm is consistent with a harness that
#                        cannot see the defect at all.
# Both halves of every verdict matter: 0 double-holds AND no round where the
# refusal locked everybody out.
#
# THE RESIDUAL, STATED. The rename is not free. For the microseconds a live
# generation is moved aside and put back, its own holder can reach
# `release_runtime_lock` and find its lock directory or its owner file missing,
# and report the lock STOLEN. That is a FALSE ALARM -- the lock was handed
# straight back and nothing was shared -- and it is a refusal, never a
# double-hold, so nothing this harness scores can be reached through it. What
# is genuinely not closed is a THIRD run `mkdir`ing the lock in the instant it
# is aside; that window is microseconds, neither hook sits inside it, and
# `mkdir` offers no primitive that would close it. It is narrower than the
# `rm`+`rmdir` window it replaces, and `acquire_runtime_lock` says so at the
# rename.
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
WIDEN_S=10       # the widening hook's value, whichever window the arm widens
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
# Both hooks, checked separately: `DEV_RUNTIME_RACE_SLEEP_BREAK` contains
# `DEV_RUNTIME_RACE_SLEEP` as a substring, so one test for the shorter name
# would pass on a file that had lost it and kept only the longer one.
for hook in DEV_RUNTIME_RACE_SLEEP_BREAK 'DEV_RUNTIME_RACE_SLEEP:-0'; do
  if [[ "$runtime_before" != *"$hook"* ]]; then
    echo "FATAL: dev-runtime.sh has no $hook hook; one of the two windows" >&2
    echo "       cannot be widened and its arms would be inconclusive" >&2
    exit 1
  fi
done

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
  # from every arm. Each canonicalizes its argument and eleven production roots
  # through a separate `node -e`, which is about twelve seconds per run on this
  # host and would dominate every window this harness has to hit. They run
  # before anything the lock touches and no arm differs in them.
  # The cut is made on the dispatch LINE ITSELF and not on a line number taken
  # from the shipped file: a reverted arm is shorter, and a fixed number would
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

# --- the reverts --------------------------------------------------------------
# Each is a mechanical edit of the shipped text, anchored ON the shipped text,
# and each anchor must occur exactly once or the arm it feeds is testing
# something nobody chose.
occurrences_of() { # needle, haystack
  local needle="$1" rest="$2" n=0
  while [[ "$rest" == *"$needle"* ]]; do
    n=$(( n + 1 ))
    rest="${rest#*"$needle"}"
  done
  printf '%s\n' "$n"
}
require_once() { # label, needle, haystack
  local n
  n="$(occurrences_of "$2" "$3")"
  if [[ "$n" != "1" ]]; then
    echo "FATAL: the $1 anchor matched $n times in dev-runtime.sh, wanted" >&2
    echo "       exactly 1; the arm it feeds would test nothing" >&2
    exit 1
  fi
}

# The owner re-read. `if (( 0 ))` keeps the block's shape -- its `return 1` and
# its messages -- and removes only the QUESTION, so the arm differs from the
# shipped file in the measurement and in nothing else.
REREAD_ANCHOR='        recheck=0
        owner_again="$(sed -n '"'"'1p'"'"' "$LOCK_OWNER_FILE")" || recheck=$?
        if (( recheck != 0 )) || [[ "$owner_again" != "$owner" ]]; then'
REREAD_REVERTED='        if (( 0 )); then'

# The atomic rename, put back to the two destructive steps it replaced. The span
# runs from the aside-name to the removal of what was moved; everything in it is
# the rename mechanism and nothing else is. What goes in its place is the
# ORIGINAL pair, in the original order: the owner file deleted in place, then the
# directory, with the recreating `mkdir` left where the shipped file has it.
RENAME_START='        breaking="$LOCK_DIR.breaking.$$.$RANDOM$RANDOM"'
RENAME_END='          echo "      $breaking" >&2
        fi'
RENAME_REVERTED='        rm -f "$LOCK_OWNER_FILE"
        rmdir "$LOCK_DIR"'

require_once "owner re-read" "$REREAD_ANCHOR" "$runtime_before"
require_once "rename-break opening" "$RENAME_START" "$runtime_before"
require_once "rename-break closing" "$RENAME_END" "$runtime_before"

# The rename alone, reverted.
REV_RENAME="${runtime_before%%"$RENAME_START"*}$RENAME_REVERTED${runtime_before#*"$RENAME_END"}"
# The original defect: no re-read AND no rename. Built from the arm above, so
# the two edits cannot drift apart.
require_once "owner re-read (rename-reverted copy)" "$REREAD_ANCHOR" "$REV_RENAME"
REV_BOTH="${REV_RENAME%%"$REREAD_ANCHOR"*}$REREAD_REVERTED${REV_RENAME#*"$REREAD_ANCHOR"}"

build_driver "$runtime_before" "$work/driver-shipped.sh"
build_driver "$REV_RENAME" "$work/driver-rev-rename.sh"
build_driver "$REV_BOTH" "$work/driver-rev-both.sh"
for d in shipped rev-rename rev-both; do
  if ! bash -n "$work/driver-$d.sh"; then
    echo "FATAL: the $d driver does not parse" >&2
    exit 1
  fi
done

# --- one round ----------------------------------------------------------------
# Prints one word: double, single, lockout, or void.
run_round() { # driver, round-number, hook-variable
  local driver="$1" n="$2" hook="$3" base witness pa pb ev depth=0 max=0 enters=0
  base="$work/round-$n-$RANDOM"
  witness="$base.witness"
  mkdir -p "$base/runtime.lock" || { printf 'void\n'; return; }
  printf '%s\n' "$DEAD_PID" >"$base/runtime.lock/pid" || { printf 'void\n'; return; }
  : >"$witness" || { printf 'void\n'; return; }

  # `env`, because which hook is widened is a value here and an assignment
  # prefix cannot come from an expansion.
  env WENLAN_DEV_STATE_DIR="$base" RACE_NAME=breaker RACE_WITNESS="$witness" \
    RACE_HOLD_S="$BREAKER_HOLD_S" "$hook=$WIDEN_S" \
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

run_arm() { # driver, hook-variable, label, expected-word
  local driver="$1" hook="$2" label="$3" want="$4" n verdict
  local double=0 single=0 lockout=0 void=0
  for (( n = 1; n <= ROUNDS; n++ )); do
    verdict="$(run_round "$driver" "$n" "$hook")"
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

HOOK_W1=DEV_RUNTIME_RACE_SLEEP
HOOK_W2=DEV_RUNTIME_RACE_SLEEP_BREAK

echo "two real runs against one stale dev runtime lock:"
echo "  window 1: the delay sits BEFORE the owner re-read"
run_arm "$work/driver-shipped.sh" "$HOOK_W1" "shipped/window-1" single
echo "  window 2: the delay sits BETWEEN the re-read and the break"
run_arm "$work/driver-shipped.sh" "$HOOK_W2" "shipped/window-2" single

echo "lock controls:"
echo "  nc-lock-stale-break-without-re-read-or-rename  (window 1: the owner is not"
echo "    read again and the lock is destroyed with rm+rmdir instead of renamed)"
run_arm "$work/driver-rev-both.sh" "$HOOK_W1" "reverted/window-1" double
echo "  nc-lock-stale-break-without-the-atomic-rename  (window 2: the re-read is"
echo "    kept and only the rename is put back to rm+rmdir)"
run_arm "$work/driver-rev-rename.sh" "$HOOK_W2" "reverted/window-2" double

if [[ "$(cat "$runtime")" != "$runtime_before" ]]; then
  echo "FATAL: scripts/dev-runtime.sh changed during the run"; exit 1
fi

echo "CONTROL FAILURES: $failures"
(( failures == 0 )) || exit 1
