#!/usr/bin/env bash
# Cases and controls for `read_owned_pid`, the dev daemon's ownership record,
# and for `list_dir_tristate`, the directory discovery it stands on.
#
# Round 13e. `read_owned_pid` is the front door to `stop_runtime` and
# `start_runtime`, and both branch on THREE answers: 0 "here is the record",
# 1 "nothing is recorded", 2 "something is recorded and I could not read it".
# Only the third keeps `clear_owned_state` away from a record whose daemon may
# still be running -- `start_runtime`'s own comment says "Falling through here
# is what deleted the record". So the failure that matters is not a wrong
# field; it is a 2 that comes back as a 1, after which the caller prints
# "No worktree-owned Wenlan dev daemon is recorded", deletes what it should
# have kept, and treats a held port as free.
#
# Round 13f found the same defect one level in, in the round-13e remedy. The
# `sed` reads were made tri-state; the EXISTENCE test above them was not.
# `[[ -e "$f" ]]` has two answers, so an unreadable STATE_DIR made all three
# lookups false and `present == 0` returned 1 -- the exact collapse the `sed`
# work had just closed, sitting one line above it. And this harness could not
# have seen it: its "unreadable" fixtures are directories, chosen precisely
# BECAUSE `[[ -e ]]` succeeds on them, so every case it had exercised the
# reads and none of them the discovery.
#
# Both functions are EXTRACTED from the shipped script and sourced, because
# dev-runtime.sh dispatches on "$1" at top level and cannot be sourced.
#
# NOTHING is stubbed. The record files are real files in a real temp directory,
# and every "cannot be read" fixture is a real unreadable path. Two shapes are
# used, and the difference matters:
#
#   * a DIRECTORY where a file is expected -- `[[ -e ]]` accepts it, `sed`
#     refuses it. This drives the read half.
#   * a plain FILE where a directory is expected -- `ls -A -- x` succeeds on it
#     and prints the path, `ls -A -- x/.` refuses it. This drives the discovery
#     half, and it is the shape that shows why the trailing `/.` is not
#     decoration.
#
# Both facts are measured on this host before any case runs; if either stops
# holding, the unreadable cases would silently become readable ones.
#
# NOT COVERED, and named rather than skipped: a directory that exists and
# denies listing on PERMISSIONS. `chmod 000` does not work on this filesystem
# (measured: `sed` still reads the file, `ls` still lists the directory), and
# `icacls /deny` could not be applied to a temp path here. The file-shaped
# fixtures drive the same branches of `list_dir_tristate` -- "exists but will
# not list as a directory" and "an ancestor will not list" -- but they do it
# through ENOTDIR rather than EACCES. If those two errnos are ever handled
# differently, this harness would not see it.
#
# What this proves: the three answers are distinguishable, which fields are
# populated when each is returned, and that the discovery climbs far enough to
# call a never-used machine "nothing recorded" rather than "unmeasured". What
# it does NOT prove: that `stop_runtime` and `start_runtime` branch correctly
# on them -- that is their own code, read but not executed here.
#
# Run: bash scripts/negative-controls/dev-runtime-record-controls.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="$root/scripts/dev-runtime.sh"
logs="$root/target/negative-control-logs"
mkdir -p "$logs"

runtime_before="$(cat "$runtime")"

work="$(mktemp -d "${TMPDIR:-/tmp}/dev-runtime-record-XXXXXX")" || exit 1
trap 'rm -rf "$work"' EXIT
failures=0

# --- the host facts every "cannot be read" fixture stands on -----------------
# A control whose unreadable fixture is quietly readable tests nothing, and
# would test nothing SILENTLY: the cases would still land on their expected
# answers by another route and the suite would be green. Measure all four,
# refuse otherwise.
probe="$work/probe"
mkdir -p "$probe/adir"
printf 'not a directory\n' >"$probe/afile"
if sed -n '1p' "$probe/adir" >/dev/null 2>&1; then
  echo "FATAL: sed reads a directory on this host; the read cases would test nothing" >&2
  exit 1
fi
if ! [[ -e "$probe/adir" ]]; then
  echo "FATAL: [[ -e ]] rejects a directory here; the read fixtures are not present-but-unreadable" >&2
  exit 1
fi
if sed -n '1p' "$probe/nosuchfile" >/dev/null 2>&1; then
  echo "FATAL: sed succeeds on a missing file here; nc-record-data-dir-absence-is-unreadable tests nothing" >&2
  exit 1
fi
if ! ls -A -- "$probe/afile" >/dev/null 2>&1; then
  echo "FATAL: \`ls -A -- <file>\` fails on this host, so nc-record-listing-without-the-dot tests nothing" >&2
  exit 1
fi
if ls -A -- "$probe/afile/." >/dev/null 2>&1; then
  echo "FATAL: \`ls -A -- <file>/.\` succeeds on this host; the discovery cases would test nothing" >&2
  exit 1
fi

# --- extract the functions and the record paths from the shipped script ------
extract() { # source-text, name -> that function, verbatim, by brace matching
  awk -v fn="$2" '
    $0 == fn "() {" { on = 1 }
    on {
      print
      n += gsub(/\{/, "{")
      n -= gsub(/\}/, "}")
      if (n == 0) exit
    }
  ' <<<"$1"
}

# The record paths are the shipped spelling, not this harness's guess. Four
# assignments, no more and no fewer: a fifth member added upstream without a
# case here should stop the harness, not ride along unmeasured.
record_paths="$(grep -E '^(PID_FILE|SERVER_PATH_FILE|PORT_FILE|DATA_DIR_FILE)=' "$runtime")"
path_count="$(printf '%s\n' "$record_paths" | grep -c .)"
if [[ "$path_count" != "4" ]]; then
  echo "FATAL: expected 4 record-path assignments in dev-runtime.sh, found $path_count" >&2
  exit 1
fi

# How many times a literal string occurs in another. `${a%%"$b"*}` answers "at
# least once", which is not what an anchor needs to be true: one that matches
# TWICE mutates the first occurrence and leaves the second, and the subject is
# then neither the shipped code nor the reverted code. Several anchors here are
# short case arms (`0) return 2 ;;`), which is exactly the shape that acquires a
# twin without anyone noticing.
count_occurrences() { # haystack, needle -> the count on stdout
  local rest="$1" needle="$2" n=0
  [[ -n "$needle" ]] || { printf '0'; return; }
  while [[ "$rest" == *"$needle"* ]]; do
    n=$((n + 1))
    rest="${rest#*"$needle"}"
  done
  printf '%s' "$n"
}

build_subject() { # read_owned_pid, list_dir_tristate, listing_has_name -> driver
  if [[ -z "$1" || "$1" != *"OWNED_PID"* || "${1: -1}" != "}" ]]; then
    echo "FATAL: read_owned_pid could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  if [[ -z "$2" || "$2" != *"ls -A"* || "${2: -1}" != "}" ]]; then
    echo "FATAL: list_dir_tristate could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  # Round 13h split the name search out of both callers, because `grep`'s status
  # was being read as a boolean: 0 is "present", 1 is "absent", and 2 -- a
  # broken pipe, a missing binary, an input it could not read -- was being
  # counted as "absent" alongside 1. Absent is a MEASUREMENT; 2 is the absence
  # of one, and it is the answer that must never turn into permission to delete
  # an ownership record.
  if [[ -z "$3" || "$3" != *"grep -qxF"* || "${3: -1}" != "}" ]]; then
    echo "FATAL: listing_has_name could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'STATE_DIR="$1"\n'
    printf '%s\n' "$record_paths"
    printf '%s\n' "$3"
    printf '%s\n' "$2"
    printf '%s\n' "$1"
    # Every field, not just the status. A mutation that reads past the first
    # unreadable member returns the same 2 and populates different fields;
    # the status alone cannot see it, so the status alone is not the assertion.
    printf 'set +e\n'
    printf 'rc=0\n'
    printf 'read_owned_pid || rc=$?\n'
    printf 'printf "%%s|%%s|%%s|%%s|%%s\\n" "$rc" "${OWNED_PID-<unset>}" \\\n'
    printf '  "${OWNED_SERVER-<unset>}" "${OWNED_PORT-<unset>}" "${OWNED_DATA_DIR-<unset>}"\n'
  } >"$work/driver.sh"
}

# --- fixtures ----------------------------------------------------------------
SERVER_VALUE="C:/wenlan/staged/wenlan-server.exe"
PORT_VALUE="17878"
DATA_VALUE="C:/wenlan/dev-data"

write_core() { # dir
  printf '4242\n' >"$1/wenlan-server.pid"
  printf '%s\n' "$SERVER_VALUE" >"$1/wenlan-server.path"
  printf '%s\n' "$PORT_VALUE" >"$1/wenlan-server.port"
}

fixture() { # kind, base -> prints the STATE_DIR to hand the subject
  local kind="$1" b="$2" d="$2/state"
  case "$kind" in
    empty-dir)          mkdir -p "$d" ;;
    complete)           mkdir -p "$d"; write_core "$d" ;;
    complete-with-data) mkdir -p "$d"; write_core "$d"
                        printf '%s\n' "$DATA_VALUE" >"$d/wenlan-server.data-dir" ;;
    pid-only)           mkdir -p "$d"; printf '4242\n' >"$d/wenlan-server.pid" ;;
    no-port)            mkdir -p "$d"; printf '4242\n' >"$d/wenlan-server.pid"
                        printf '%s\n' "$SERVER_VALUE" >"$d/wenlan-server.path" ;;
    pid-unreadable)     mkdir -p "$d/wenlan-server.pid"
                        printf '%s\n' "$SERVER_VALUE" >"$d/wenlan-server.path"
                        printf '%s\n' "$PORT_VALUE" >"$d/wenlan-server.port" ;;
    data-unreadable)    mkdir -p "$d"; write_core "$d"
                        mkdir -p "$d/wenlan-server.data-dir" ;;
    pid-not-a-number)   mkdir -p "$d"; write_core "$d"
                        printf 'abc\n' >"$d/wenlan-server.pid" ;;
    server-empty)       mkdir -p "$d"; write_core "$d"; : >"$d/wenlan-server.path" ;;
    port-not-a-number)  mkdir -p "$d"; write_core "$d"
                        printf 'eighty\n' >"$d/wenlan-server.port" ;;
    # --- discovery: the state dir itself
    state-dir-absent)   : ;;
    state-dir-is-a-file) printf 'not a directory\n' >"$d" ;;
    # --- discovery: an ancestor. The chain below `notadir` is two components
    # deep, so these also prove the walk climbs rather than looking once.
    parent-is-a-file)   printf 'not a directory\n' >"$b/notadir"
                        d="$b/notadir/generation/state" ;;
    # A never-used machine: STATE_DIR is `$TMPDIR/wenlan-app-dev/$ID` and
    # NEITHER of the last two components exists. This must be 1 -- if it is 2,
    # `wenlan-dev stop` fails on every clean checkout.
    never-used-machine) d="$b/wenlan-app-dev/1234567890" ;;
    # --- the name search itself, which is the third tool in this chain and the
    # last one whose status was being read as a boolean. A complete, readable
    # record in a listable directory, and a `grep` that cannot answer. The
    # record is THERE; nothing here knows it, and "I could not look" must not
    # come back as "it is not there".
    grep-cannot-answer)
      mkdir -p "$d"; write_core "$d"
      mkdir -p "$b/bin"
      {
        printf '#!/usr/bin/env bash\n'
        printf 'echo "grep: the listing could not be searched" >&2\n'
        printf 'exit 2\n'
      } >"$b/bin/grep"
      chmod 0755 "$b/bin/grep"
      ;;
    *) echo "FATAL: unknown fixture $kind" >&2; exit 1 ;;
  esac
  printf '%s\n' "$d"
}

run_case() { # name, driver, fixture-kind, want-signature
  local name="$1" driver="$2" kind="$3" want="$4"
  local b="$work/base-$RANDOM-$RANDOM" d got=""
  mkdir -p "$b"
  d="$(fixture "$kind" "$b")" || return 1
  # A fixture may put a shim of its own on PATH; the driver is the only thing
  # that sees it, so the harness's own `grep` below is untouched.
  local path="$PATH"
  if [[ -d "$b/bin" ]]; then
    local shim="$b/bin"
    if command -v cygpath >/dev/null 2>&1; then shim="$(cygpath -u "$b/bin")"; fi
    path="$shim:$PATH"
  fi
  got="$(PATH="$path" bash "$driver" "$d" 2>/dev/null | tail -n 1)"
  rm -rf "$b"
  if [[ "$got" != "$want" ]]; then
    printf '  FAIL %-26s got  [%s]\n' "$name" "${got:-<no output>}"
    printf '       %-26s want [%s]\n' "" "$want"
    return 1
  fi
  printf '  ok   %-26s [%s]\n' "$name" "$got"
  return 0
}

# rc | OWNED_PID | OWNED_SERVER | OWNED_PORT | OWNED_DATA_DIR
S="$SERVER_VALUE"
U='<unset>'
NOTHING="$U|$U|$U|$U"
CASES=(
  "empty-state-dir|empty-dir|1|$NOTHING"
  "complete-record|complete|0|4242|$S|$PORT_VALUE|"
  "record-with-data-dir|complete-with-data|0|4242|$S|$PORT_VALUE|$DATA_VALUE"
  "torn-pid-only|pid-only|2|$NOTHING"
  "torn-missing-port|no-port|2|$NOTHING"
  "pid-file-unreadable|pid-unreadable|2||$U|$U|$U"
  "data-dir-file-unreadable|data-unreadable|2|4242|$S|$PORT_VALUE|"
  "pid-not-a-number|pid-not-a-number|2|abc|$S|$PORT_VALUE|"
  "server-path-empty|server-empty|2|4242||$PORT_VALUE|"
  "port-not-a-number|port-not-a-number|2|4242|$S|eighty|"
  "state-dir-absent|state-dir-absent|1|$NOTHING"
  "state-dir-is-a-file|state-dir-is-a-file|2|$NOTHING"
  "ancestor-is-a-file|parent-is-a-file|2|$NOTHING"
  "never-used-machine|never-used-machine|1|$NOTHING"
  "name-search-cannot-answer|grep-cannot-answer|2|$NOTHING"
)

run_all() { # driver
  PASSED_CASES=(); FAILED_CASES=()
  local spec name kind want rest
  for spec in "${CASES[@]}"; do
    name="${spec%%|*}"
    rest="${spec#*|}"
    kind="${rest%%|*}"
    want="${rest#*|}"
    if run_case "$name" "$1" "$kind" "$want"; then
      PASSED_CASES+=("$name")
    else
      FAILED_CASES+=("$name")
    fi
  done
}

echo "dev-runtime-record-controls"
echo "cases against the shipped read_owned_pid + list_dir_tristate:"
OWNED="$(extract "$runtime_before" read_owned_pid)"
LISTER="$(extract "$runtime_before" list_dir_tristate)"
NAMER="$(extract "$runtime_before" listing_has_name)"
build_subject "$OWNED" "$LISTER" "$NAMER"
run_all "$work/driver.sh"
failures=$((failures + ${#FAILED_CASES[@]}))

echo "controls:"
control() { # name, why, which(owned|lister|namer), old, new, must_fail...
  local name="$1" why="$2" which="$3" old="$4" new="$5"; shift 5
  local -a must_fail=("$@")
  printf '  %s  (%s)\n' "$name" "$why"
  local text head tail hits
  case "$which" in
    owned) text="$OWNED" ;;
    lister) text="$LISTER" ;;
    namer) text="$NAMER" ;;
    *) printf '    FAIL unknown control target %s\n' "$which"; failures=$((failures + 1)); return ;;
  esac
  # EXACTLY once, and a stale anchor is a hard error. Several of these are two
  # short case arms, and `0) return 2 ;;` is one refactor away from having a
  # twin in the same function -- at which point "at least once" mutates whichever
  # comes first and reports on a subject nobody wrote.
  hits="$(count_occurrences "$text" "$old")"
  if [[ "$hits" != 1 ]]; then
    printf '    FAIL anchor matched %s times in %s, wanted exactly 1; this control tests nothing\n' \
      "$hits" "$which"
    failures=$((failures + 1))
    return
  fi
  head="${text%%"$old"*}"
  tail="${text#*"$old"}"
  case "$which" in
    owned) build_subject "$head$new$tail" "$LISTER" "$NAMER" ;;
    lister) build_subject "$OWNED" "$head$new$tail" "$NAMER" ;;
    namer) build_subject "$OWNED" "$LISTER" "$head$new$tail" ;;
  esac
  run_all "$work/driver.sh" >"$logs/$name.log" 2>&1
  local want
  for want in "${must_fail[@]}"; do
    if printf '%s\n' "${FAILED_CASES[@]:-}" | grep -qx -- "$want"; then
      printf '    ok   caught:   %s\n' "$want"
    else
      printf '    FAIL survived: %s -- the case does not defend this fix\n' "$want"
      failures=$((failures + 1))
    fi
  done
  # ROUND 4, NEW FINDING 7 (raised by the negative-control agent). The survivor
  # list used to be HAND-WRITTEN beside the must_fail name, so a mutation that
  # reddened cases nobody had listed was scored as though it had reddened only
  # the one it names. Undeclared collateral damage is exactly what tells a
  # control pinned to its fix from one that broke the subject in general, and it
  # was being credited silently. The survivor set is CASES minus this control's
  # own must_fail names, so it cannot fall behind the case list either.
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
}

# --- the record's own 1/2 boundary -------------------------------------------
# Collapsing either way is silent in a green suite: one makes a torn record read
# as no record and invites `clear_owned_state`; the other makes a clean slate
# read as a torn record and makes `wenlan-dev stop` fail with nothing to stop.
control nc-record-partial-is-absent \
  'a record missing one of its three files means "nothing is recorded"' owned \
  '  (( present == 3 )) || return 2' \
  '  (( present == 3 )) || return 1' \
  torn-pid-only torn-missing-port

control nc-record-absent-is-torn \
  'an empty state directory is reported as a record that could not be read' owned \
  '  (( present == 0 )) && return 1' \
  '  (( present == 0 )) && return 2' \
  empty-state-dir

control nc-record-malformed-is-no-record \
  'a record whose fields do not validate means "nothing is recorded"' owned \
  '  [[ "$OWNED_PID" =~ ^[0-9]+$ && -n "$OWNED_SERVER" &&
    "$OWNED_PORT" =~ ^[0-9]+$ ]] || return 2' \
  '  [[ "$OWNED_PID" =~ ^[0-9]+$ && -n "$OWNED_SERVER" &&
    "$OWNED_PORT" =~ ^[0-9]+$ ]] || return 1' \
  pid-not-a-number server-path-empty port-not-a-number

# The one round 13e asked for by name. The data dir is the record's only
# optional member, so "absent" is legitimately the empty string -- and that is
# exactly what a failed read produces if its status is dropped. The other three
# reads cannot hide this way: an empty value fails the regex below them and
# still returns 2. This one returns 0, and the caller then compares
# OWNED_DATA_DIR against DEV_DATA_DIR and reports a mismatch it never measured.
control nc-record-optional-read-status-dropped \
  'an unreadable data-dir file is indistinguishable from a record written without one' owned \
  '    0) OWNED_DATA_DIR="$(sed -n '"'"'1p'"'"' "$DATA_DIR_FILE")" || return 2 ;;' \
  '    0) OWNED_DATA_DIR="$(sed -n '"'"'1p'"'"' "$DATA_DIR_FILE" 2>/dev/null)" ;;' \
  data-dir-file-unreadable

control nc-record-data-dir-absence-is-unreadable \
  'a record legitimately written without a data dir reads as unreadable' owned \
  '  name_rc=0
  listing_has_name "$listing" "${DATA_DIR_FILE##*/}" || name_rc=$?
  case "$name_rc" in' \
  '  # INJECTED: the optional member is assumed present, so a record written
  # without one is read anyway and the failed read is reported as the record.
  name_rc=0
  case "$name_rc" in' \
  complete-record

# Dropping `|| return 2` from a REQUIRED read does not change the status -- the
# regex below catches the empty value and returns 2 either way. What changes is
# that the function reads on past a member it could not read, so the record it
# reports is part measurement and part guess. The status alone cannot see that;
# the field assertion can, which is why the cases assert fields and not status.
control nc-record-required-read-status-dropped \
  'reading past an unreadable member reports fields that were never measured' owned \
  '  OWNED_PID="$(sed -n '"'"'1p'"'"' "$PID_FILE")" || return 2' \
  '  OWNED_PID="$(sed -n '"'"'1p'"'"' "$PID_FILE" 2>/dev/null)"' \
  pid-file-unreadable

# --- the discovery, which round 13f found had no third answer at all ---------
control nc-record-existence-is-a-two-answer-test \
  'the round-13e form: three `[[ -e ]]` tests, and no way to say "could not ask"' owned \
  '  listing="$(list_dir_tristate "$STATE_DIR")" || list_rc=$?
  case "$list_rc" in
    0) ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
  # `${f##*/}` and not `$(basename "$f")`: a command substitution inside an
  # argument list has nowhere to put a failure, so a `basename` that could not
  # run would hand the search an empty name, match nothing, and be counted as a
  # member that is not there. Parameter expansion cannot fail.
  for f in "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE"; do
    name_rc=0
    listing_has_name "$listing" "${f##*/}" || name_rc=$?
    case "$name_rc" in
      0) present=$((present + 1)) ;;
      1) ;;
      # Not "this member is absent". The listing was read and the search over it
      # was not, so `present` would be a count of measurements that did not
      # happen -- and a low count reads as "no record" and then as permission to
      # delete one.
      *) return 2 ;;
    esac
  done' \
  '  listing=""
  for f in "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE"; do
    if [[ -e "$f" ]]; then present=$((present + 1)); fi
  done' \
  state-dir-is-a-file record-with-data-dir data-dir-file-unreadable \
  ancestor-is-a-file name-search-cannot-answer

control nc-record-listing-without-the-dot \
  'listing the state dir without the trailing `/.`, so a plain file "lists"' lister \
  '    if listing="$(ls -A -- "$dir/." 2>/dev/null)"; then' \
  '    if listing="$(ls -A -- "$dir" 2>/dev/null)"; then' \
  state-dir-is-a-file ancestor-is-a-file

control nc-record-unlistable-ancestor-is-absence \
  'an ancestor that will not list is reported as "nothing is recorded"' lister \
  '        0) return 2 ;;
        1) return 1 ;;' \
  '        0) return 1 ;;
        1) return 1 ;;' \
  state-dir-is-a-file ancestor-is-a-file

control nc-record-discovery-does-not-climb \
  'looking at one parent only, so a never-used machine is "could not measure"' lister \
  '    parent="$(dirname -- "$dir")" || return 2
    [[ "$parent" == "$dir" ]] && return 2
    child="$(basename -- "$dir")" || return 2
    dir="$parent"' \
  '    [[ -n "$child" ]] && return 2
    parent="$(dirname -- "$dir")" || return 2
    [[ "$parent" == "$dir" ]] && return 2
    child="$(basename -- "$dir")" || return 2
    dir="$parent"' \
  never-used-machine

# --- the name search, which round 13h found had no third answer either -------
#
# The chain is three tools deep now -- `ls` for the directory, `grep` for the
# name, `sed` for the value -- and this was the one whose status was still being
# read as a boolean. `grep` exits 0 for a match, 1 for no match, and 2 for an
# error, and `&& present=...` puts 1 and 2 on the same side. So a search that
# could not run counted every member absent, `present == 0` returned 1, and the
# caller deleted the ownership record of a daemon that is in the listing it just
# read. Two controls, because the collapse can be reinstated in either the
# helper or the caller that reads it.
control nc-record-name-search-error-is-absence \
  'grep exiting 2 is counted as "the name is not in the listing"' namer \
  '    1) return 1 ;;
    *) return 2 ;;' \
  '    *) return 1 ;;' \
  name-search-cannot-answer

control nc-record-name-search-status-collapsed \
  'the caller folds "could not search" into "not present" one line further out' owned \
  '      0) present=$((present + 1)) ;;
      1) ;;' \
  '      0) present=$((present + 1)) ;;
      1|2) ;;' \
  name-search-cannot-answer

# =============================================================================
# THE LOCK RELEASE -- round 6.
# =============================================================================
#
# `release_runtime_lock` stands on the same two primitives as `read_owned_pid`
# -- `list_dir_tristate` and `listing_has_name`, both already extracted above --
# which is why this lives here rather than in a fourth harness.
#
# THE DEFECT. `list_dir_tristate` answering 1, the lock directory MEASURED
# absent, used to `return 0` with the comment "a recovery that found this run
# dead is entitled to have done that". Nothing on that path establishes any
# such recovery, and the run reaching it is by construction not dead:
# `RUNTIME_LOCK_HELD` is set in exactly one place, last in
# `acquire_runtime_lock` after every step of taking the lock succeeded, and the
# release only runs when it is 1. So the state is "this run took the lock and
# the lock is gone" -- the exclusive claim broken while the run was still using
# it -- and it exited 0 with `DEV_RUNTIME_RESULT: ok`. `attest.sh` reports the
# same condition through `LOCK_STOLEN`; the two files must not disagree about
# what a vanished lock means.
#
# WHAT IS DRIVEN, and it is the whole outcome path rather than the function:
# the trap is installed, the lock is declared held, the driver exits 0 with
# RESULT_KIND=ok, and the signature is what the marker, the exit status and the
# two human lines say afterwards. A status-only assertion would not see a
# stolen lock described as one that failed to come off, which sends a reader
# looking for a leftover directory that is not there.
#
# NOT COVERED, for the same reason as the record cases above: EACCES. Every
# "cannot be examined" fixture here reaches the arm through a shell `ls` that
# returns 2, which is the same status `list_dir_tristate` reads but not the
# same syscall.
echo "cases against the shipped release_runtime_lock + on_runtime_exit:"

RELEASE="$(extract "$runtime_before" release_runtime_lock)"
EMITTER="$(extract "$runtime_before" emit_result)"
EXITFN="$(extract "$runtime_before" on_runtime_exit)"

build_lock_subject() { # release_runtime_lock, on_runtime_exit -> driver
  if [[ -z "$1" || "$1" != *"RUNTIME_LOCK_HELD=0"* || "${1: -1}" != "}" ]]; then
    echo "FATAL: release_runtime_lock could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  if [[ -z "$2" || "$2" != *"emit_result"* || "${2: -1}" != "}" ]]; then
    echo "FATAL: on_runtime_exit could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'STATE_DIR="$1"\n'
    # The shipped spelling of both paths, read out of the script rather than
    # guessed, for the same reason the record paths are.
    printf '%s\n' "$lock_paths"
    # Mirrors the file-scope initialisation. `on_runtime_exit` reads all five
    # and the driver runs under `set -u`, so a missing one would abort inside
    # the trap -- the one place this contract cannot express a failure.
    printf 'RESULT_KIND=ok\nRESULT_EMITTED=0\n'
    printf 'RUNTIME_LOCK_HELD=0\nRUNTIME_LOCK_STOLEN=0\nRUNTIME_EXIT_RAN=0\n'
    printf '%s\n' "$NAMER"
    printf '%s\n' "$LISTER"
    printf '%s\n' "$EMITTER"
    printf '%s\n' "$1"
    printf '%s\n' "$2"
    # The fixture is eval'd INSIDE the driver so that an owner file written as
    # "this run" carries the driver's own `$$`, which only the driver knows.
    printf 'eval "${WENLAN_NC_LOCK_FIXTURE:-}"\n'
    printf 'trap on_runtime_exit EXIT\n'
    printf 'RUNTIME_LOCK_HELD=1\n'
    printf 'exit 0\n'
  } >"$work/lock-driver.sh"
}

lock_paths="$(grep -E '^(LOCK_DIR|LOCK_OWNER_FILE)=' "$runtime")"
lock_path_count="$(printf '%s\n' "$lock_paths" | grep -c .)"
if [[ "$lock_path_count" != "2" ]]; then
  echo "FATAL: expected 2 lock-path assignments in dev-runtime.sh, found $lock_path_count" >&2
  exit 1
fi

# rc | marker kind | the caller's WORDING | the release's own diagnostic | what
# happened to the LOCK DIRECTORY ON DISK.
#
# The first four fields are text and status, and ROUND 6 is that four text-and-
# status fields cannot see the defect that matters most here. `release_runtime_lock`
# used to `rmdir` a lock directory it could not prove was its own -- in the
# world where a peer has just `mkdir`'d and not yet written its pid, that
# deletes a LIVE peer's lock and two runs then share this worktree's port and
# data directory. A release that resumed doing that while still printing the
# refusal would satisfy every text assertion in this file. The fifth field is
# the act itself, measured on the filesystem after the run: `kept` the directory
# is still there, `gone` it is not.
#
# Both text fields are derived from exact shipped substrings: a control that
# only moved a message would otherwise be scored as though it had changed
# nothing.
lock_signature() { # stderr-text, status, dir-state
  local err="$1" wording=none diag=none kind status="$2" dir="$3"
  case "$err" in
    *"broken while it still held it"*)      wording=stolen ;;
    *"its dev runtime lock did not come"*)  wording=leftover ;;
  esac
  case "$err" in
    *"lock this run took is gone"*)         diag=gone ;;
    *"is recorded to PID"*)                 diag=retaken ;;
    *"outlived its owner file"*)            diag=outlived ;;
    *"owner file could not be read at release"*) diag=unread ;;
    *"could not be examined at release"*)   diag=unexam ;;
  esac
  kind="$(printf '%s\n' "$err" | grep -o 'DEV_RUNTIME_RESULT: .*' | tail -n 1)"
  kind="${kind#DEV_RUNTIME_RESULT: }"
  printf '%s|%s|%s|%s|%s' "$status" "${kind:-<no marker>}" "$wording" "$diag" "$dir"
}

LOCK_FIX_OWNED='mkdir -p "$LOCK_DIR"; printf "%s\n" "$$" >"$LOCK_OWNER_FILE"'
LOCK_FIX_DIR_ONLY='mkdir -p "$LOCK_DIR"'
LOCK_FIX_OTHER='mkdir -p "$LOCK_DIR"; printf "999999\n" >"$LOCK_OWNER_FILE"'
LOCK_FIX_UNREMOVABLE='mkdir -p "$LOCK_DIR/squatter"'
# A directory where the owner file belongs: present in the listing, refused by
# `sed`. The host fact that makes this an unreadable fixture rather than an
# absent one is measured at the top of this harness.
LOCK_FIX_UNREADABLE='mkdir -p "$LOCK_OWNER_FILE"'
# THE ROUND-6 CASE. No lock directory at all, under a STATE_DIR that exists, so
# the absence is one `list_dir_tristate` can measure -- it walks to the parent
# and finds the name missing from a listing it really read -- rather than one
# it merely failed to examine, which is a different arm and a different answer.
LOCK_FIX_STOLEN='mkdir -p "$STATE_DIR"'
LOCK_FIX_UNEXAMINABLE='mkdir -p "$LOCK_DIR"; printf "%s\n" "$$" >"$LOCK_OWNER_FILE"; ls() { return 2; }'

# ROUND 6. `lock-owner-removed-under-holder` was `lock-empty-dir-tidied`, and it
# wanted `0|ok|none|none`. It is the same fixture; what changed is that the
# release no longer decides this arm's verdict by whether a `rmdir` it was not
# entitled to perform happened to succeed. Note that it and
# `lock-outlived-the-release` now carry the IDENTICAL signature -- that is the
# claim, not an oversight. An empty ownerless lock directory and one with a
# squatter inside are one state, "this run's owner file is not in it", and only
# the destructive act ever told them apart.
LOCK_CASES=(
  "lock-owned-by-this-run|$LOCK_FIX_OWNED|0|ok|none|none|gone"
  "lock-owner-removed-under-holder|$LOCK_FIX_DIR_ONLY|1|unknown|stolen|outlived|kept"
  "lock-retaken-by-another|$LOCK_FIX_OTHER|1|unknown|stolen|retaken|kept"
  "lock-outlived-the-release|$LOCK_FIX_UNREMOVABLE|1|unknown|stolen|outlived|kept"
  "lock-owner-unreadable|$LOCK_FIX_UNREADABLE|1|unknown|leftover|unread|kept"
  "lock-vanished-under-holder|$LOCK_FIX_STOLEN|1|unknown|stolen|gone|gone"
  "lock-cannot-be-examined|$LOCK_FIX_UNEXAMINABLE|1|unknown|leftover|unexam|kept"
)

run_lock_case() { # name, fixture, want-signature
  local name="$1" fixture="$2" want="$3"
  local b="$work/lock-$RANDOM-$RANDOM" err status=0 got dir=unmeasured
  mkdir -p "$b"
  err="$(WENLAN_NC_LOCK_FIXTURE="$fixture" bash "$work/lock-driver.sh" "$b" 2>&1 >/dev/null)" || status=$?
  # The act, read before the teardown wipes the evidence. `-d` is a two-answer
  # test, so it is only asked once the parent has been shown to LIST -- a base
  # directory this harness itself just created and cannot examine would
  # otherwise answer "gone" and score a destroyed lock as an intact one. It
  # stays `unmeasured` in that case, which matches no `want` and fails loudly.
  if ls -a "$b/." >/dev/null 2>&1; then
    if [[ -d "$b/runtime.lock" ]]; then dir=kept; else dir=gone; fi
  fi
  rm -rf "$b"
  got="$(lock_signature "$err" "$status" "$dir")"
  if [[ "$got" != "$want" ]]; then
    printf '  FAIL %-31s got  [%s]\n' "$name" "$got"
    printf '       %-31s want [%s]\n' "" "$want"
    return 1
  fi
  printf '  ok   %-31s [%s]\n' "$name" "$got"
  return 0
}

run_lock_all() {
  LOCK_PASSED=(); LOCK_FAILED=()
  local spec name fixture want rest
  for spec in "${LOCK_CASES[@]}"; do
    name="${spec%%|*}"; rest="${spec#*|}"
    fixture="${rest%%|*}"; rest="${rest#*|}"
    want="$rest"
    if run_lock_case "$name" "$fixture" "$want"; then
      LOCK_PASSED+=("$name")
    else
      LOCK_FAILED+=("$name")
    fi
  done
}

build_lock_subject "$RELEASE" "$EXITFN"
run_lock_all
failures=$((failures + ${#LOCK_FAILED[@]}))

echo "lock controls:"
lock_control() { # name, why, which(release|exitfn), old, new, must_fail...
  local name="$1" why="$2" which="$3" old="$4" new="$5"; shift 5
  local -a must_fail=("$@")
  printf '  %s  (%s)\n' "$name" "$why"
  local text head tail hits
  case "$which" in
    release) text="$RELEASE" ;;
    exitfn)  text="$EXITFN" ;;
    *) printf '    FAIL unknown control target %s\n' "$which"; failures=$((failures + 1)); return ;;
  esac
  hits="$(count_occurrences "$text" "$old")"
  if [[ "$hits" != 1 ]]; then
    printf '    FAIL anchor matched %s times in %s, wanted exactly 1; this control tests nothing\n' \
      "$hits" "$which"
    failures=$((failures + 1))
    return
  fi
  head="${text%%"$old"*}"
  tail="${text#*"$old"}"
  case "$which" in
    release) build_lock_subject "$head$new$tail" "$EXITFN" ;;
    exitfn)  build_lock_subject "$RELEASE" "$head$new$tail" ;;
  esac
  run_lock_all >"$logs/$name.log" 2>&1
  local want
  for want in "${must_fail[@]}"; do
    if printf '%s\n' "${LOCK_FAILED[@]:-}" | grep -qx -- "$want"; then
      printf '    ok   caught:   %s\n' "$want"
    else
      printf '    FAIL survived: %s -- the case does not defend this fix\n' "$want"
      failures=$((failures + 1))
    fi
  done
  # The same undeclared-collateral check the record controls make: a mutation
  # that reddens cases nobody listed is not pinned to its fix.
  local case_spec case_name
  for case_spec in "${LOCK_CASES[@]}"; do
    case_name="${case_spec%%|*}"
    if printf '%s\n' "${must_fail[@]:-}" | grep -qx -- "$case_name"; then continue; fi
    if printf '%s\n' "${LOCK_PASSED[@]:-}" | grep -qx -- "$case_name"; then
      printf '    ok   survived: %s\n' "$case_name"
    else
      printf '    FAIL also failed: %s -- the control is not pinned to the fix\n' "$case_name"
      failures=$((failures + 1))
    fi
  done
}

# THE ROUND-6 REVERT, exactly as the arm stood before it: a measured-absent lock
# reported as a release. `lock-owned-by-this-run` is the boundary beside it and
# must stay green, or the control would be indistinguishable from "any anomaly
# is now unknown".
lock_control nc-lock-vanished-is-a-clean-release \
  'a lock that vanished under its holder is reported as a clean release' release \
  '    1)
      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock this run took is gone from $LOCK_DIR" >&2
      echo "       it was removed while this run still held it, so this is not" >&2
      echo "       a release: the isolation it was providing ended early and" >&2
      echo "       nothing here can say what ran inside that window" >&2
      return 1
      ;;' \
  '    1) return 0 ;;' \
  lock-vanished-under-holder

# The status was never wrong for a retaken lock; the WORDS were. Dropping the
# flag leaves the exit code and the marker untouched and turns "the lock was
# retaken" into "the lock did not come off", which is a leftover directory this
# run should have tidied -- a different event, and one a reader would go and
# fail to find.
lock_control nc-lock-retaken-is-not-marked-stolen \
  'a lock retaken by another run is described as one that failed to come off' release \
  '    RUNTIME_LOCK_STOLEN=1
    echo "error: the dev runtime lock is recorded to PID $owner, not to this run ($$)" >&2' \
  '    echo "error: the dev runtime lock is recorded to PID $owner, not to this run ($$)" >&2' \
  lock-retaken-by-another

# And the branch at the caller, which is where both shapes are put into words.
# Collapsing it back to one message keeps every status and every marker.
lock_control nc-lock-theft-wording-collapsed \
  'both shapes of a broken claim share the leftover-directory message' exitfn \
  '      if (( RUNTIME_LOCK_STOLEN == 1 )); then
        echo "error: this command finished, but the dev runtime lock it held was" >&2
        echo "       broken while it still held it, so the isolation the work" >&2
        echo "       above assumed was not in force for some part of it" >&2
        echo "       another command may have been running against this worktree'"'"'s" >&2
        echo "       port and data directory at the same time" >&2
      else
        echo "error: this command finished but its dev runtime lock did not come" >&2
        echo "       off; the next command against this worktree will refuse" >&2
      fi' \
  '      echo "error: this command finished but its dev runtime lock did not come" >&2
      echo "       off; the next command against this worktree will refuse" >&2' \
  lock-retaken-by-another lock-vanished-under-holder \
  lock-owner-removed-under-holder lock-outlived-the-release

# ROUND 6, AND THIS PAIR IS ABOUT THE ACT, NOT THE VERDICT.
#
# The arm for "the directory stands and this run's owner file is not in it" used
# to `rmdir` it and, if that worked, report a clean release. Both halves are
# reverted here, separately, because a fix that repaired one and quietly kept
# the other would otherwise pass:
#
#   A. the destruction alone -- the refusal is still printed, the marker still
#      says `unknown`, and the directory is deleted anyway. Every text-and-
#      status assertion in this harness and in dev-runtime.test.ts stays green
#      under it. Only the fifth signature field sees it, and it must, because in
#      the world where that directory belongs to a peer that has just `mkdir`'d
#      and not yet written its pid, this is two runs sharing one isolated port.
#
#   B. the whole arm as it shipped before, `rmdir`-then-`return 0`, which is the
#      destruction AND the `ok` that was derived from it succeeding.
#
# `lock-owned-by-this-run` must survive both, and that is the boundary that
# matters here: the fix is "do not destroy what this run cannot prove is its
# own", not "never remove a lock". A release that stopped removing its OWN lock
# would leave one behind for every subsequent command to refuse, and that case
# is the one that would catch it.
lock_control nc-lock-unprovable-dir-still-destroyed \
  'a lock directory this run cannot prove is its own is removed anyway' release \
  '      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock directory outlived its owner file" >&2' \
  '      rmdir "$LOCK_DIR" 2>/dev/null || true
      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock directory outlived its owner file" >&2' \
  lock-owner-removed-under-holder

lock_control nc-lock-ownerless-dir-is-a-clean-release \
  'an ownerless lock directory is destroyed and the run reports ok' release \
  '      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock directory outlived its owner file" >&2' \
  '      if rmdir "$LOCK_DIR" 2>/dev/null; then return 0; fi
      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock directory outlived its owner file" >&2' \
  lock-owner-removed-under-holder

if [[ "$(cat "$runtime")" != "$runtime_before" ]]; then
  echo "FATAL: scripts/dev-runtime.sh changed during the run"; exit 1
fi

echo "CONTROL FAILURES: $failures"
(( failures == 0 )) || exit 1
