#!/usr/bin/env bash
# Run every negative-control harness and produce ONE receipt.
#
# Ten hand-typed command lines produce ten separate results and no aggregate:
# run eight, skip the two that would have gone red, and the summary says the
# controls pass. A harness killed by a watchdog fails from the other side --
# it prints its last scored control and stops, and the transcript tail is
# indistinguishable from one that scored everything.
#
# So this runner does not trust an exit status and does not trust a transcript
# that merely looks finished. Each harness declares a COMPLETION CONTRACT: a
# terminal line it prints last, and only after every one of its controls has
# been scored. A harness whose last non-empty line is not that line DID NOT
# COMPLETE, whatever it exited with, and this runner says so and fails.
#
# The four verdicts are kept apart on purpose:
#
#   ok                 completed, and every control fired
#   CONTROLS-FAILED    completed, and at least one control did not fire
#   DID-NOT-COMPLETE   no terminal marker: killed, refused, crashed, or ran
#                      partially. This is "unchecked", never a pass.
#   CONTRADICTORY      the marker claims failures=0 and the process still
#                      exited non-zero. Believing either one is a guess.
#
# A precondition this host cannot satisfy (no PowerShell, no Authenticode-signed
# fixture, no pnpm) lands in DID-NOT-COMPLETE, which is the honest place for it.
#
# Usage:
#   bash scripts/negative-controls/run-all.sh              # all ten
#   bash scripts/negative-controls/run-all.sh --list
#   bash scripts/negative-controls/run-all.sh --only replica,inventory
#
# A --only run is stamped partial=1 and is NOT a result about the suite.
set -uo pipefail

MARKER="NEGATIVE-CONTROL COMPLETE"
HARNESS="run-all.sh"
started=$SECONDS

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
logs="$root/target/negative-control-logs/run-all"
mkdir -p "$logs" || exit 1

reached_end=0
finish() {
  local rc=$?
  if (( reached_end == 0 )); then
    # It must not exit 0 while saying so. Some ways out arrive here with rc=0
    # -- a `return` at top level, a subshell that ended cleanly, a SIGTERM
    # landing after a successful command -- and a supervisor reading only the
    # status would call that a clean sweep.
    #
    # The number is settled BEFORE the line prints it, so the status the abort
    # line reports is the status the process exits with.
    local inherited=$rc
    (( rc )) || rc=1
    echo "NEGATIVE-CONTROL ABORTED $HARNESS rc=$rc elapsed=$((SECONDS - started))s"
    echo "  This runner did not reach its own summary. Nothing above it is a"
    echo "  result about the negative controls."
    if (( inherited != rc )); then
      echo "  (the run ended with status $inherited; a sweep that did not reach"
      echo "   its summary must not exit 0, so this exits $rc)"
    fi
    exit "$rc"
  fi
}
trap finish EXIT

# --- the registry ------------------------------------------------------------
# id | interpreter | file | contract | one-line subject
#
# contract=marker   last line is "NEGATIVE-CONTROL COMPLETE <file> failures=N ..."
#                   -- the harnesses on this track, which print it from an exit
#                   path that runs only after the final control is scored.
# contract=summary  last line is "CONTROL FAILURES: N" -- the pre-existing
#                   contract of the four harnesses owned by other lanes. It is
#                   weaker (it is a normal print, not an exit-path invariant),
#                   and it is recorded here as THEIR contract, not as one this
#                   runner imposes. If one of those files grows a marker, move
#                   its row to contract=marker.
REGISTRY=(
  "posix-probes|python3|posix-probes-negative-controls.py|marker|lib/host-process.sh tri-state probes + the suite's override lock"
  "lib-ps1|bash|lib-ps1-negative-controls.sh|marker|first-run/lib.ps1 declared rows, ledger paths, Check-Helper"
  "authenticode|python3|authenticode-step-receipt.py|marker|release.yml SignPath verification step, real signature states"
  "inventory|python3|a-drift-guard-inventory.py|marker|drift_guard.rs teeth reachable from a release.yml mention"
  "replica|python3|a-drift-guard-replica.py|marker|the drift_guard release.yml teeth, replicated in Python"
  "port-precheck|bash|port-precheck-controls.sh|marker|first-run/port-precheck.sh shared-port measurement + ledger row"
  "scan|bash|dev-runtime-scan-controls.sh|summary|dev-runtime.sh reap_staged_daemon scan/kill"
  "record|bash|dev-runtime-record-controls.sh|summary|dev-runtime.sh read_owned_pid tri-state ownership record"
  "lock-race|bash|dev-runtime-lock-race-controls.sh|summary|dev-runtime.sh acquire_runtime_lock stale-break ABA, two real runs"
  "stage|bash|dev-runtime-stage-controls.sh|summary|dev-runtime.sh stage_windows_daemon copies + call site"
  "windows-probes|python3|windows-probes-negative-controls.py|summary|windows-zip.ps1 / windows-nsis.ps1 probes, extracted"
)

registry_field() { printf '%s' "$1" | cut -d'|' -f"$2"; }

# --- the runner's own meta-control -------------------------------------------
# A registry maintained by hand goes stale the first time someone adds a
# harness, and a stale registry fails silently in the direction that flatters
# us: the new harness is simply never run, and this prints a clean sweep of the
# nine it knows about. So the directory is the authority for WHICH files are
# harnesses, and the registry only says how to run them.
registry_is_complete() {
  local problems=0 f base found row
  for f in "$here"/*.sh "$here"/*.py; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    [ "$base" = "$HARNESS" ] && continue
    found=0
    for row in "${REGISTRY[@]}"; do
      [ "$(registry_field "$row" 3)" = "$base" ] && found=1 && break
    done
    if (( found == 0 )); then
      echo "REGISTRY GAP: $base is a harness in this directory and is not in the registry."
      echo "  It would never have been run. Add a row to REGISTRY in $HARNESS."
      problems=$((problems + 1))
    fi
  done
  for row in "${REGISTRY[@]}"; do
    base="$(registry_field "$row" 3)"
    if [ ! -s "$here/$base" ]; then
      echo "REGISTRY GAP: $base is registered and is missing or empty at $here/$base."
      problems=$((problems + 1))
    fi
    case "$(registry_field "$row" 4)" in
      marker|summary) ;;
      *) echo "REGISTRY GAP: $base declares an unknown contract '$(registry_field "$row" 4)'."
         problems=$((problems + 1)) ;;
    esac
  done
  return "$problems"
}

# --- the receipt is bound to the revisions it ran ----------------------------
# A sweep is about fifty minutes and six lanes work in this tree at once, so any
# harness file can be edited while it is in flight. Then the clean line at the
# bottom is a statement about a set of files that never existed together at any
# instant -- harness three as it was at 05:20, harness nine as it was at 06:05 --
# and nothing in the transcript says so. The snapshot is taken at registry
# validation, re-checked at the end, and PRINTED, so the receipt names the
# revisions it covers. The runner hashes itself too: it is the thing doing the
# reporting, and bash reads a script by byte offset, so an edit under a running
# sweep corrupts control flow silently.
#
# THE RESIDUAL, conceded rather than hidden: this bounds the claim to "these
# bytes at the start and the same bytes at the end". An edit made and reverted
# between the two hashings is invisible, and closing that means watching the
# filesystem for fifty minutes.
#
# A digest must LOOK like one -- 64 lowercase hex characters -- at capture and
# at comparison alike, or a `sha256sum` shim, alias or broken PATH entry that
# consistently prints `UNAVAILABLE` compares equal to itself and the sweep
# certifies a receipt bound to nothing. The shape must be checked BEFORE any
# `cut -c1-64` manufactures 64 characters, and the shape alone does not ask
# whether the tool hashes; see the block above parse_sha256_line.
is_digest() {
  case "$1" in
    ""|*[!0-9a-f]*) return 1 ;;
  esac
  [ "${#1}" -eq 64 ]
}

# A SHAPE CHECK AFTER A TRUNCATION VALIDATES THE TRUNCATION, NOT A HASH -- and
# this is the guard that binds every other verdict in this file to a set of
# bytes. Under `sha256sum "$here/$base" | cut -c1-64`, `cut` runs FIRST, so a
# `sha256sum` on PATH that prints
#
#     0000000000000000000000000000000000000000000000000000000000000000 BROKEN
#
# becomes 64 perfectly valid hex characters, the same constant is captured at
# the start and compared at the end, and the receipt certifies every file
# unchanged. (64 zeros is also why `is_digest_can_fail` must not use them as its
# POSITIVE example: that is exactly what a stubbed hasher prints.)
#
# Two things are needed and only one of them is obvious.
#
#   1. VALIDATE THE WHOLE LINE, BEFORE TRUNCATING. sha256sum's contract is
#      `<64 hex><space><space-or-asterisk><path>`; the asterisk is binary mode
#      and is what GNU coreutils 8.32 prints on this host. A line that is not
#      that shape is not a digest, and one that names a DIFFERENT path is an
#      answer about a different file -- could-not-measure, never a match.
#   2. THE PROVENANCE IS STILL UNMEASURED even then. The threat model this
#      guard was written for explicitly includes a shim, an alias or a broken
#      PATH entry, and every one of those can print a well-formed line naming
#      the right file. So the tool is asked a question whose answer is known
#      independently, and a tool that gets it wrong certifies nothing.
#
# THE KNOWN ANSWER, and where it came from: the digest below is the SHA-256 of
# the 53 bytes `KAT_TEXT` plus a newline, computed with Python's hashlib --
# a different implementation, not this host's sha256sum, so embedding it is not
# the tool ratifying itself. GNU coreutils 8.32 on this host agrees with it.
NL=$'\n'
CR=$'\r'
KAT_TEXT='wenlan negative-controls sha256 known-answer test v1'
KAT_SHA256='c2be5347acfeaeb0ed5783801226b54c6590bcb814ff77fddbdb7823df480624'

# One sha256sum output LINE, checked against the path it was asked about.
# Prints the digest on stdout, and ONLY on success.
#
#   0  a `<64 hex><sp><sp|*><path>` line naming exactly $2
#   1  not that shape at all -- no digest was produced
#   2  that shape, but naming a DIFFERENT file
#
# Three statuses, not two, and every caller branches on all three: "the tool
# answered about something else" is a different problem from "the tool did not
# answer", and printing one under the other's headline sends the next reader at
# the wrong thing.
parse_sha256_line() {
  local line="$1" want="$2" digest sep name
  case "$line" in *"$NL"*) return 1 ;; esac
  [ "${#line}" -gt 66 ] || return 1
  digest="${line:0:64}"
  sep="${line:64:2}"
  name="${line:66}"
  is_digest "$digest" || return 1
  case "$sep" in "  "|" *") ;; *) return 1 ;; esac
  [ "$name" = "$want" ] || return 2
  printf '%s' "$digest"
}

# Hash one file. Same three statuses as parse_sha256_line; a tool that could
# not be run at all is 1, because "no digest was produced" is what happened.
sha256_of() {
  local path="$1" line rc
  line="$(sha256sum "$path" 2>/dev/null)"
  rc=$?
  (( rc == 0 )) || return 1
  parse_sha256_line "$line" "$path"
}

# THE KNOWN-ANSWER TEST. Everything above establishes that the tool answered in
# the right SHAPE about the right FILE. Nothing above establishes that it
# hashed. This does, and it goes through the same sha256_of the real captures
# use, so it measures the path that is actually used rather than a parallel one.
#
# WHAT IT DOES NOT CATCH, conceded rather than papered over: a hasher that
# answers this one question correctly and lies about everything else -- a shim
# that delegates for the fixture path and returns a constant for the harnesses.
# Nothing short of a second independent implementation would, and this runner
# does not have one. What it closes is the whole class the threat model actually
# names: a hasher that is broken, missing, stubbed, or printing a constant.
hasher_answers_a_known_question() {
  local f="$logs/known-answer.txt" got rc
  if ! printf '%s\n' "$KAT_TEXT" > "$f"; then
    echo "FATAL: could not write the known-answer fixture at $f, so the hasher was"
    echo "  never asked a question whose answer is known independently."
    return 1
  fi
  got="$(sha256_of "$f")"
  rc=$?
  case "$rc" in
    0) ;;
    2) echo "FATAL: sha256sum answered about a different file than the known-answer"
       echo "  fixture it was handed. Nothing it says about the harnesses is about them."
       return 1 ;;
    *) echo "FATAL: sha256sum produced no valid digest line for the known-answer"
       echo "  fixture, so it cannot be shown to hash anything at all."
       return 1 ;;
  esac
  if [ "$got" != "$KAT_SHA256" ]; then
    echo "FATAL: sha256sum does not reproduce a known SHA-256. Asked for the digest"
    echo "  of $((${#KAT_TEXT} + 1)) known bytes it answered"
    echo "    $got"
    echo "  and the answer, computed independently, is"
    echo "    $KAT_SHA256"
    echo "  A shim, an alias or a broken PATH entry that prints a well-formed line"
    echo "  naming the right file still binds this receipt to nothing. A hasher that"
    echo "  fails this is could-not-measure, and could-not-measure never certifies."
    return 1
  fi
  echo "  hasher: sha256sum reproduces a known SHA-256 over $((${#KAT_TEXT} + 1))"
  echo "    fixture bytes (${KAT_SHA256:0:16}...), so it is hashing and not"
  echo "    printing a constant"
  return 0
}

# Two parallel INDEXED arrays, not an associative one. macOS ships bash 3.2,
# which has no associative arrays at all: the declaration is a syntax error, and
# a subscript such as `SNAPSHOT[a-drift-guard.py]` is then parsed as arithmetic.
# The same construct in dev-runtime.sh broke the macOS app-check lane rather
# than degrading, so it is not used anywhere in this directory.
#
# `snapshot_get` keeps exact-key semantics: a hit exits 0 and prints the stored
# value, a miss exits 1 and prints nothing, so a file that was never hashed
# stays distinguishable from one whose stored digest is the empty string. The
# match is `=` on the whole key, never a pattern or a substring. `${#arr[@]}` is
# safe on an empty array under `set -u` where `"${arr[@]}"` is not, so every
# walk here is by index.
SNAPSHOT_KEYS=()
SNAPSHOT_VALS=()

# exit: 0 the key is present (its value on stdout), 1 no such key.
snapshot_get() {
  local i n="${#SNAPSHOT_KEYS[@]}"
  for (( i = 0; i < n; i++ )); do
    if [ "${SNAPSHOT_KEYS[$i]}" = "$1" ]; then
      printf '%s' "${SNAPSHOT_VALS[$i]}"
      return 0
    fi
  done
  return 1
}

snapshot_put() {
  local i n="${#SNAPSHOT_KEYS[@]}"
  for (( i = 0; i < n; i++ )); do
    if [ "${SNAPSHOT_KEYS[$i]}" = "$1" ]; then
      SNAPSHOT_VALS[$i]="$2"
      return 0
    fi
  done
  SNAPSHOT_KEYS[$n]="$1"
  SNAPSHOT_VALS[$n]="$2"
}

# Every recorded key, one per line, in insertion order; callers sort.
snapshot_names() {
  local i n="${#SNAPSHOT_KEYS[@]}"
  for (( i = 0; i < n; i++ )); do printf '%s\n' "${SNAPSHOT_KEYS[$i]}"; done
}

snapshot_take() {
  local row base d rc bases=()
  for row in "${REGISTRY[@]}"; do
    bases+=("$(registry_field "$row" 3)")
  done
  bases+=("$HARNESS")
  # A snapshot that cannot be TAKEN is not a snapshot that matched. Refuse at
  # the capture rather than let a non-digest travel to the compare, where equal
  # garbage reads as an unchanged file.
  for base in "${bases[@]}"; do
    d="$(sha256_of "$here/$base")"
    rc=$?
    case "$rc" in
      0) snapshot_put "$base" "$d" ;;
      2) echo "FATAL: sha256sum answered about a DIFFERENT file than $base. The line"
         echo "  it printed does not name $here/$base, so whatever it hashed, the"
         echo "  receipt cannot be bound to the file this runner is about to execute."
         return 1 ;;
      *) echo "FATAL: could not hash $base -- sha256sum did not print a"
         echo "  '<64 hex><space><space or asterisk><path>' line for $here/$base. The"
         echo "  receipt cannot be bound to what it ran, and a digest-shaped nothing"
         echo "  compared against itself at the end would have read as a file that"
         echo "  never moved."
         return 1 ;;
    esac
  done
  return 0
}

snapshot_print() {
  local base was
  echo "harness revisions this receipt covers (sha256, first 16):"
  for base in $(snapshot_names | sort); do
    was="$(snapshot_get "$base")" || was=""
    printf '  %-40s %s\n' "$base" "${was:0:16}"
  done
}

# The digest-shape guard, made to fire, in the transcript rather than in a
# comment: a guard that is only ever shown accepting is a guard nobody has
# watched refuse. The POSITIVE example is a real SHA-256, not 64 zeros --
# `printf '%064d' 0` is what a fixed-output hasher prints, so using it as the
# proof of correctness would be using the counterexample. is_digest is a SHAPE
# test and must still ACCEPT 64 zeros; they are a legal digest.
is_digest_can_fail() {
  local rejected=0 v
  for v in "UNAVAILABLE" "" "sha256sum: command not found" "abc"; do
    is_digest "$v" || rejected=$((rejected + 1))
  done
  if is_digest "$KAT_SHA256" && [ "$rejected" -eq 4 ]; then
    echo "  digest shape: 4 non-digests refused (UNAVAILABLE, empty, an error line,"
    echo "    a short string), a real SHA-256 accepted"
    return 0
  fi
  echo "FATAL: the digest-shape check does not discriminate (refused $rejected/4"
  echo "  non-digests); a could-not-hash would be certified as an unchanged file."
  return 1
}

# THE LINE PARSER, made to fire. The first fixture is the one that matters:
# under `sha256sum | cut -c1-64` that line becomes 64 valid hex characters and
# the same constant at both ends certifies every harness unchanged. It is
# refused here BEFORE anything is truncated.
sha256_line_can_fail() {
  local bad=0 got rc want fx rest
  # (expected status | line | path asked about)
  local -a fixtures=(
    "1|0000000000000000000000000000000000000000000000000000000000000000 BROKEN|/x/f"
    "1|UNAVAILABLE|/x/f"
    "1||/x/f"
    "1|sha256sum: /x/f: No such file or directory|/x/f"
    "1|000000000000000000000000000000000000000000000000000000000000000z  /x/f|/x/f"
    "2|$KAT_SHA256  /some/other/file|/x/f"
    "0|$KAT_SHA256  /x/f|/x/f"
    "0|$KAT_SHA256 */x/f|/x/f"
  )
  for fx in "${fixtures[@]}"; do
    want="${fx%%|*}"
    rest="${fx#*|}"
    got="$(parse_sha256_line "${rest%|*}" "${rest##*|}")"
    rc=$?
    if [ "$rc" != "$want" ]; then
      echo "FATAL: parse_sha256_line read '${rest%|*}' as status $rc, wanted $want."
      bad=$((bad + 1))
    elif [ "$rc" = 0 ] && [ "$got" != "$KAT_SHA256" ]; then
      echo "FATAL: parse_sha256_line accepted a line and returned '$got' rather than"
      echo "  the digest it contained."
      bad=$((bad + 1))
    fi
  done
  if (( bad )); then
    echo "  A line the parser cannot judge is a receipt bound to nothing; refusing."
    return 1
  fi
  echo "  digest line: a fixed-output hasher's '<64 zeros> BROKEN', an error line,"
  echo "    empty output and a near-miss digest all refused; a well-formed line"
  echo "    naming ANOTHER file reported separately as could-not-measure; the two"
  echo "    real spellings (text and binary mode) accepted"
  return 0
}

# --- why there is NO UNCONDITIONAL CRLF guard here ---------------------------
# The premise is that this bash strips CR from a CRLF script and leaks none into
# what the script writes, so harness-file line endings are not a measurement.
# That is a claim about a host, so it is measured every run by
# `crlf_premise_holds` below; when the premise is false or unmeasured, the guard
# applies instead and a CRLF harness is refused. Where line endings ARE the
# measurement is the SUBJECT side, handled at each site that needs it:
# posix-probes reads its subjects with newline='' and diagnoses a CRLF-only
# anchor miss; authenticode-step-receipt.py compares with read_bytes;
# windows-probes keeps a CRLF fixture in its own self-check.
#
# If you come here to add one anyway, the DETECTION is the trap. Measured on
# this host against files of known bytes:
#
#   grep -c '\r' file      -- WRONG. In a BRE, \r is an escaped literal `r`: 0
#                             on a 100%-CRLF file, and every line containing the
#                             letter r on a 100%-LF one.
#   awk '/\r/{n++}' file   -- WRONG. Reports 0 on a 100%-CRLF file; the CR is
#                             stripped before the pattern sees it.
#   sed -n p file | cat -A -- WRONG. Clean `$` on every line of a 100%-CRLF
#                             file. (`cat -A` alone is right: `^M$`.)
#   grep -c $'\r' file     -- RIGHT STANDALONE, WRONG INSIDE "$( ... )" ON THIS
#                             HOST (MSYS2 bash 4.4.23(1)-release). Over a 3-line
#                             pure-LF fixture: standalone -> 0; inside "$( )" ->
#                             3, the LINE COUNT, because xtrace shows the pattern
#                             arriving EMPTY (`++ grep -c '' pure-lf.txt`) and an
#                             empty BRE matches every line. Backticks do NOT show
#                             it, so it is $( ) specifically. This is a
#                             measurement on this host, not a rule about Bash;
#                             re-measure before repeating it elsewhere. Assigning
#                             first is right in both contexts and is what to
#                             write: CR=$'\r'; grep -c "$CR" file
#   git ls-files --eol     -- RIGHT, and the one to reach for by hand, but says
#                             nothing about a file git is not tracking, and four
#                             of the files below are untracked today.
#   python read_bytes()    -- RIGHT everywhere, context-free, and what every
#                             .count(b'\r\n') guard in this directory uses.

# The premise the section above rests on, measured rather than asserted.
#   0  a fully-CRLF script runs under this bash AND leaks no CR into its output
#   1  it does not -- the reasoning above is void on this host
#   2  the probe could not be run, which is not the premise holding
#
# Two questions in one probe, because the comment above makes two claims: that
# a CRLF script RUNS (`set -o pipefail\r` must not be an invalid option), and
# that it does not leak CR into what it WRITES.
crlf_premise_holds() {
  local f="$logs/crlf-premise-probe.sh" out rc
  {
    printf 'set -o pipefail\r\n'
    printf '%s\r\n' "printf 'CRLF-PROBE-OK'"
  } > "$f" || return 2
  [ -s "$f" ] || return 2
  out="$(bash "$f" 2>&1)"
  rc=$?
  (( rc == 0 )) || return 1
  case "$out" in *"$CR"*) return 1 ;; esac
  [ "$out" = "CRLF-PROBE-OK" ] || return 1
  return 0
}

# The guard for the two branches where the premise does NOT hold. Names every
# registered harness whose bytes are CRLF. `CR=$'\r'` is assigned first and the
# VARIABLE is passed to grep: measured on this host, `grep -c $'\r' file` inside
# "$( )" reports the LINE COUNT of a pure-LF file (see the trap table above).
crlf_harness_files() {
  local row base n hits=0
  for row in "${REGISTRY[@]}"; do
    base="$(registry_field "$row" 3)"
    n="$(grep -c "$CR" "$here/$base" 2>/dev/null)"
    if [ -n "$n" ] && [ "$n" != "0" ]; then
      echo "  $base -- $n line(s) end CRLF"
      hits=$((hits + 1))
    fi
  done
  n="$(grep -c "$CR" "$here/$HARNESS" 2>/dev/null)"
  if [ -n "$n" ] && [ "$n" != "0" ]; then
    echo "  $HARNESS -- $n line(s) end CRLF"
    hits=$((hits + 1))
  fi
  return "$hits"
}

# Returns non-zero when the sweep must not proceed.
crlf_position_is_earned() {
  local rc lines crlf=0
  crlf_premise_holds
  rc=$?
  if (( rc == 0 )); then
    echo "  line endings: a fully-CRLF script runs under this bash (${BASH_VERSION})"
    echo "    and leaks no CR into its output, so harness-file line endings are not"
    echo "    a measurement here and there is no guard on them"
    return 0
  fi
  if (( rc == 1 )); then
    echo "  line endings: A FULLY-CRLF SCRIPT DOES NOT RUN CLEANLY under this bash"
    echo "    (${BASH_VERSION}). The stated reason this runner has no CRLF guard does"
    echo "    not hold here, so the guard applies instead:"
  else
    echo "  line endings: the CRLF premise COULD NOT BE MEASURED (the probe did not"
    echo "    run). That is not the premise holding, so the guard applies instead:"
  fi
  lines="$(crlf_harness_files)" || crlf=$?
  if (( crlf == 0 )); then
    echo "    ...and no registered harness is CRLF, so nothing is blocked."
    return 0
  fi
  printf '%s\n' "$lines"
  echo "    $crlf harness file(s) are CRLF on a host where that changes what they do."
  return 1
}

# Returns the number of files this receipt CANNOT certify, and names them, each
# with which of the two reasons it was: it moved, or its end state could not be
# read. Both block a clean verdict; only one of them is "the harnesses were
# edited mid-sweep", and printing the other under that headline would be a
# misdiagnosis pointing the next reader at an editor rather than at their PATH.
snapshot_drift() {
  local base now was rc moved=0
  for base in $(snapshot_names | sort); do
    now="$(sha256_of "$here/$base")"
    rc=$?
    was="$(snapshot_get "$base")" || was=""
    if (( rc == 2 )); then
      echo "  $base -- COULD NOT MEASURE: sha256sum printed a well-formed digest line"
      echo "      naming a DIFFERENT file, so this is an answer about something else."
      echo "      Not a match and not a mismatch; the receipt cannot cover it either way."
      moved=$((moved + 1))
    elif (( rc != 0 )); then
      echo "  $base -- COULD NOT MEASURE: sha256sum did not print a"
      echo "      '<64 hex><space><space or asterisk><path>' line, so this file's end"
      echo "      state is unknown. Not a match and not a mismatch; the receipt cannot"
      echo "      cover it either way."
      moved=$((moved + 1))
    elif [ "$now" != "$was" ]; then
      echo "  $base -- MOVED: ${was:0:16} at the start, ${now:0:16} now"
      moved=$((moved + 1))
    fi
  done
  return "$moved"
}

# --- argument handling -------------------------------------------------------
only=""
while (( $# )); do
  case "$1" in
    --list)
      for row in "${REGISTRY[@]}"; do
        printf '%-16s %-8s %-38s %s\n' \
          "$(registry_field "$row" 1)" "$(registry_field "$row" 4)" \
          "$(registry_field "$row" 3)" "$(registry_field "$row" 5)"
      done
      reached_end=1
      exit 0
      ;;
    --only)
      shift
      only="${1:-}"
      [ -n "$only" ] || { echo "--only needs a comma-separated list of ids"; exit 2; }
      ;;
    --only=*) only="${1#--only=}" ;;
    -h|--help)
      sed -n '2,33p' "$here/$HARNESS"
      reached_end=1
      exit 0
      ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
  shift
done

selected() {
  [ -z "$only" ] && return 0
  case ",$only," in *",$1,"*) return 0 ;; esac
  return 1
}

if [ -n "$only" ]; then
  for id in ${only//,/ }; do
    hit=0
    for row in "${REGISTRY[@]}"; do
      [ "$(registry_field "$row" 1)" = "$id" ] && hit=1 && break
    done
    (( hit )) || { echo "--only names '$id', which is not a registered harness id."; exit 2; }
  done
fi

# --- run ---------------------------------------------------------------------
echo "== negative controls =="
echo "repo:   $root"
echo "logs:   $logs"
echo "python: $(python3 --version 2>&1 | tr -d '\r')"
echo "bash:   $BASH_VERSION"
echo

if ! registry_is_complete; then
  echo
  echo "SUITE VERDICT: REGISTRY INCOMPLETE -- refusing to report on a partial set."
  reached_end=1
  exit 1
fi

if ! is_digest_can_fail; then
  echo
  echo "SUITE VERDICT: UNBOUND -- the receipt's own digest check cannot tell a"
  echo "  hash from a non-hash, so nothing it says about drift is a measurement."
  reached_end=1
  exit 1
fi

if ! sha256_line_can_fail; then
  echo
  echo "SUITE VERDICT: UNBOUND -- the receipt's own digest-LINE check cannot tell"
  echo "  a hash line from a truncation, so a fixed-output hasher would certify"
  echo "  every harness unchanged."
  reached_end=1
  exit 1
fi

if ! hasher_answers_a_known_question; then
  echo
  echo "SUITE VERDICT: UNBOUND -- the hasher cannot be shown to hash, so the"
  echo "  digests below would be 64 characters of something and the receipt would"
  echo "  be bound to nothing."
  reached_end=1
  exit 1
fi

if ! crlf_position_is_earned; then
  echo
  echo "SUITE VERDICT: UNBOUND -- this runner's stated reason for having no CRLF"
  echo "  guard does not hold on this host, and a harness file IS CRLF, so what"
  echo "  those files do here is not what their bytes say."
  reached_end=1
  exit 1
fi

if ! snapshot_take; then
  echo
  echo "SUITE VERDICT: UNBOUND -- refusing to report a result it cannot tie to"
  echo "  a set of harness revisions."
  reached_end=1
  exit 1
fi
snapshot_print
echo

ran=0; ok=0; failed=0; incomplete=0; skipped=0
declare -a ROWS=()

for row in "${REGISTRY[@]}"; do
  id="$(registry_field "$row" 1)"
  interp="$(registry_field "$row" 2)"
  file="$(registry_field "$row" 3)"
  contract="$(registry_field "$row" 4)"
  note=""

  if ! selected "$id"; then
    skipped=$((skipped + 1))
    ROWS+=("$id|skipped|-|-|not selected by --only")
    continue
  fi

  log="$logs/$id.log"
  echo "-- $id ($file) ..."
  t0=$SECONDS
  "$interp" "$here/$file" >"$log" 2>&1
  rc=$?
  el=$((SECONDS - t0))
  ran=$((ran + 1))

  last="$(awk 'NF{last=$0} END{if (last != "") print last}' "$log" | tr -d '\r')"
  complete=0
  fails="-"
  case "$contract" in
    marker)
      if [[ "$last" == "$MARKER $file failures="* ]]; then
        complete=1
        fails="$(printf '%s' "$last" | sed -n 's/.*failures=\([0-9][0-9]*\).*/\1/p')"
        # A harness that ran only part of itself stamps partial=1. That is a
        # completed PROCESS and an incomplete MEASUREMENT; only the second one
        # matters here.
        if [[ "$last" == *"partial=1"* ]]; then
          complete=0
          fails="-"
          note="ran with --only; partial, not a result"
        fi
      fi
      ;;
    summary)
      if [[ "$last" =~ ^CONTROL\ FAILURES:\ [0-9]+$ ]]; then
        complete=1
        fails="${last##*: }"
      fi
      ;;
  esac

  if (( complete == 0 )); then
    incomplete=$((incomplete + 1))
    verdict="DID-NOT-COMPLETE"
    ROWS+=("$id|$verdict|$rc|${el}s|${note:-last line was: ${last:-<no output>}}")
    note=""
  elif [ "$fails" != "0" ]; then
    failed=$((failed + 1))
    verdict="CONTROLS-FAILED"
    ROWS+=("$id|$verdict|$rc|${el}s|$fails control(s) did not fire; see $id.log")
  elif (( rc != 0 )); then
    # It claims every control fired and still exited non-zero. Do not pick one
    # of the two to believe.
    failed=$((failed + 1))
    verdict="CONTRADICTORY"
    ROWS+=("$id|$verdict|$rc|${el}s|marker says failures=0 but exit status is $rc")
  else
    ok=$((ok + 1))
    verdict="ok"
    ROWS+=("$id|$verdict|$rc|${el}s|every control fired")
  fi
  echo "   $verdict (exit $rc, ${el}s)"
done

echo
printf '%-16s %-17s %-5s %-7s %s\n' HARNESS VERDICT EXIT TIME NOTE
printf '%-16s %-17s %-5s %-7s %s\n' ---------------- ----------------- ----- ------- ----
for r in "${ROWS[@]}"; do
  printf '%-16s %-17s %-5s %-7s %s\n' \
    "$(printf '%s' "$r" | cut -d'|' -f1)" \
    "$(printf '%s' "$r" | cut -d'|' -f2)" \
    "$(printf '%s' "$r" | cut -d'|' -f3)" \
    "$(printf '%s' "$r" | cut -d'|' -f4)" \
    "$(printf '%s' "$r" | cut -d'|' -f5-)"
done
echo

total=${#REGISTRY[@]}
echo "registered=$total ran=$ran ok=$ok controls-failed=$failed did-not-complete=$incomplete skipped=$skipped"

# Re-hash before any verdict is printed, and let drift override a clean one: if
# the files moved under the sweep, the rows above are results about different
# revisions. Checked BEFORE the verdict so there is no window in which a clean
# line has already been printed.
#
# The known-answer test at the top proved the hasher was hashing AT THE START.
# The threat model is a shim, an alias or a PATH entry, all of which can appear
# during a fifty-minute sweep, and a hasher that starts printing a constant
# halfway through makes every remaining compare come out equal. So it is asked
# the known question AGAIN before the drift compare, and one that has stopped
# answering leaves every file's end state UNKNOWN, not unchanged.
drifted=0
drift_lines=""
if hasher_answers_a_known_question; then
  drift_lines="$(snapshot_drift)" || drifted=$?
else
  drift_lines="$(for base in $(snapshot_names | sort); do
      echo "  $base -- COULD NOT MEASURE: the hasher stopped reproducing a known"
      echo "      SHA-256 (see above), so this file's end state is unknown. Not a"
      echo "      match and not a mismatch."
    done)"
  drifted=${#SNAPSHOT_KEYS[@]}
fi
if (( drifted )); then
  echo
  if printf '%s\n' "$drift_lines" | grep -q 'COULD NOT MEASURE'; then
    echo "SNAPSHOT NOT CERTIFIED: $drifted harness file(s) either changed while the"
    echo "  sweep ran or could not be re-hashed at the end. Which is which is on the"
    echo "  lines below; they are not the same problem."
  else
    echo "SNAPSHOT MOVED: $drifted harness file(s) changed while the sweep ran."
  fi
  printf '%s\n' "$drift_lines"
fi

rc=0
if (( failed > 0 || incomplete > 0 )); then rc=1; fi
if (( drifted > 0 )); then rc=1; fi
if (( drifted > 0 )); then
  echo "SUITE VERDICT: UNBOUND -- the harness bytes could not be shown constant"
  echo "  across the sweep (edited mid-run, or not re-measurable at the end), so"
  echo "  the rows above may be about different revisions of the suite and cannot"
  echo "  be added up. Re-run against a still tree, with a working sha256sum."
elif (( skipped > 0 )); then
  echo "SUITE VERDICT: PARTIAL -- $skipped harness(es) were not run. This is not a"
  echo "  result about the negative-control suite, only about the ones selected."
  rc=1
elif (( rc == 0 )); then
  echo "SUITE VERDICT: every registered harness completed and every control fired."
else
  echo "SUITE VERDICT: NOT CLEAN -- see the table above."
fi

reached_end=1
# `drifted` is counted into failures on purpose. This runner refuses a child
# whose marker says failures=0 beside a non-zero exit and calls it
# CONTRADICTORY; printing that same shape itself, because the drift lived in a
# verdict line and not in the number, would be the identical defect one level
# up.
echo "$MARKER $HARNESS failures=$((failed + incomplete + drifted)) elapsed=$((SECONDS - started))s partial=$(( skipped > 0 ? 1 : 0 ))"
exit "$rc"
