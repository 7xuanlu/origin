#!/usr/bin/env bash
# Cases and controls for `reap_staged_daemon`'s process-table scan.
#
# Round 13d. The function's own comment block explains, at length, that a
# snapshot must be taken whole and awk's status must be checked, because "a
# parser that dies leaves an empty candidate list, the loop body never runs, and
# fifty rounds later this returns 1 -- nothing to reap". The parse immediately
# below that comment then guarded every row with `col > 0`, so a `ps -W` header
# without a COMMAND column skipped every row, printed nothing, exited 0, and
# returned exactly the 1 the comment forbids. The defect was inside the remedy
# written against it, which is where this workstream keeps finding it.
#
# Round 13h moved the PARSE. It was a third hand-maintained copy of the `ps -W`
# table walk -- the same four witnesses as scripts/lib/host-process.sh, its own
# place to be edited, and a demonstrated history of the hardening landing on
# only one copy -- so `reap_staged_daemon` now asks the library for all matching
# rows and the parse exists once. Every mutation below therefore targets a
# COPY of the library rather than the extracted function: the subject is written
# to the work directory and sourced from there, and the shipped library is never
# touched (the run refuses if either file changes under it).
#
# The function is EXTRACTED from the shipped script and sourced, because
# dev-runtime.sh dispatches on "$1" at top level and cannot be sourced. Its two
# real dependencies come from scripts/lib/host-process.sh; only
# `force_terminate_process` and the process table itself are stubbed, and the
# stub for each is named in the case.
#
# What this harness proves, stated so no reader has to infer it from a case
# name: the shipped SCAN reads the table, refuses a header it cannot parse, and
# hands the WINPID and the recorded image path -- not the MSYS pid, not the
# table's spelling of the path -- to the kill helper. What it does NOT prove:
# that anything is reaped. `force_terminate_process` is a recording no-op and
# `tasklist` is a stub, so the real kill and the real liveness probe never meet
# here. Round 13e caught the positive case claiming otherwise in its name.
#
# Run: bash scripts/negative-controls/dev-runtime-scan-controls.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="$root/scripts/dev-runtime.sh"
lib="$root/scripts/lib/host-process.sh"
logs="$root/target/negative-control-logs"
mkdir -p "$logs"

runtime_before="$(cat "$runtime")"
lib_before="$(cat "$lib")"

work="$(mktemp -d "${TMPDIR:-/tmp}/dev-runtime-scan-XXXXXX")" || exit 1
trap 'rm -rf "$work"' EXIT
failures=0

# --- extract the function, by brace matching, from the shipped script --------
extract() { # source-text -> the reap_staged_daemon function, verbatim
  awk '
    /^reap_staged_daemon\(\) \{$/ { on = 1 }
    on {
      print
      n += gsub(/\{/, "{")
      n -= gsub(/\}/, "}")
      if (n == 0) exit
    }
  ' <<<"$1"
}

# The image the fake daemon claims to be. A real path on this host so that
# normalize_program_path has something to canonicalise.
server="$work/staged/wenlan-server.exe"
mkdir -p "$work/staged"
printf 'not a real binary\n' >"$server"
server_win="$server"
if command -v cygpath >/dev/null 2>&1; then server_win="$(cygpath -w "$server")"; fi

ps_header='      PID    PPID    PGID     WINPID   TTY         UID    STIME COMMAND'
row() { # winpid, command
  printf '%9s%8s%8s%11s  ?%14s0   Aug 27 %s\n' 1000 0 0 "$1" "" "$2"
}

# How many times a literal string occurs in another. `${a%%"$b"*}` answers "at
# least once" and nothing more, which is not what an anchor needs to be true: an
# anchor that matches TWICE mutates the first occurrence and leaves the second,
# so the subject is neither the shipped code nor the reverted code, and every
# verdict taken from it is about a third thing nobody wrote.
count_occurrences() { # haystack, needle -> the count on stdout
  local rest="$1" needle="$2" n=0
  [[ -n "$needle" ]] || { printf '0'; return; }
  while [[ "$rest" == *"$needle"* ]]; do
    n=$((n + 1))
    rest="${rest#*"$needle"}"
  done
  printf '%s' "$n"
}

build_subject() { # library-text, function-text -> a driver at $work/driver.sh
  # An extraction that silently produced nothing would make every case fail with
  # "command not found" -- loud, but for the wrong reason, and the failure would
  # look like the subject's. Refuse instead.
  if [[ -z "$1" || "$1" != *"_ps_w_read()"* || "$1" != *"ps_w_rows_matching()"* ]]; then
    echo "FATAL: scripts/lib/host-process.sh is not the library this harness parses" >&2
    exit 1
  fi
  # A mutated subject is still a whole function; what a failed extraction gives
  # back is nothing, or something that does not close.
  if [[ -z "$2" || "$2" != *"reap_staged_daemon() {"* || "${2: -1}" != "}" ]]; then
    echo "FATAL: reap_staged_daemon could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  printf '%s\n' "$1" >"$work/host-process.sh"
  {
    printf '#!/usr/bin/env bash\nset -uo pipefail\n'
    printf '. "%s"\n' "$work/host-process.sh"
    # Stubbed, and named as such: this harness is about the SCAN, not the kill.
    # It records what it was asked to kill, so a scan that finds the wrong row
    # cannot pass by killing nothing in particular.
    printf 'force_terminate_process() { printf "%%s %%s\\n" "$1" "$2" >"%s"; return 0; }\n' \
      "$work/killed"
    printf '%s\n' "$2"
    printf 'reap_staged_daemon "$1"; printf "rc=%%s\\n" "$?"\n'
  } >"$work/driver.sh"
}

run_case() { # name, subject-driver, ps-body, tasklist-body, want_rc, want_killed
  local name="$1" driver="$2" ps_body="$3" tasklist_body="$4" want="$5"
  local want_killed="${6:-}"
  local bin="$work/bin-$RANDOM" rc=0 out=""
  mkdir -p "$bin"
  printf '#!/usr/bin/env bash\n%s\n' "$ps_body" >"$bin/ps"
  printf '#!/usr/bin/env bash\n%s\n' "$tasklist_body" >"$bin/tasklist"
  chmod 0755 "$bin/ps" "$bin/tasklist"
  local shim="$bin"
  if command -v cygpath >/dev/null 2>&1; then shim="$(cygpath -u "$bin")"; fi
  rm -f "$work/killed"
  out="$(PATH="$shim:$PATH" WENLAN_HOST_PROCESS_PLATFORM=windows \
    bash "$driver" "$server" 2>&1)" || rc=$?
  local got="${out##*rc=}"
  got="${got%%$'\n'*}"
  if [[ "$got" != "$want" ]]; then
    printf '  FAIL %-28s rc=%s, wanted %s\n' "$name" "${got:-<none>}" "$want"
    return 1
  fi
  # rc alone would let a scan that picked the wrong row, or no row, still pass:
  # the stub returns 0 whatever it is handed. What it was handed is the answer.
  local killed=""
  [[ -f "$work/killed" ]] && killed="$(cat "$work/killed")"
  if [[ "$killed" != "$want_killed" ]]; then
    printf '  FAIL %-28s killed [%s], wanted [%s]\n' "$name" "$killed" "$want_killed"
    return 1
  fi
  printf '  ok   %-28s\n' "$name"
  return 0
}

# Round 13f: these tables were a header and ONE row. `reap_staged_daemon` now
# demands of `ps -W` the same two completeness witnesses `process_is_alive`
# demands of `tasklist` -- WINPID 4, the System process, and a floor of ten rows
# -- so a one-row fragment is "could not measure", which is what it always
# should have been. The fixture was the fragment the fix forbids, again, and the
# harness only found out because the shipped code changed under it.
ps_background="$( {
  row 4 'System'
  row 140 'C:\Windows\System32\smss.exe'
  row 680 'C:\Windows\System32\csrss.exe'
  row 708 'C:\Windows\System32\wininit.exe'
  row 796 'C:\Windows\System32\services.exe'
  row 872 'C:\Windows\System32\lsass.exe'
  row 920 'C:\Windows\System32\svchost.exe'
  row 1704 'C:\Windows\System32\svchost.exe'
  row 5320 'C:\Windows\explorer.exe'
  row 9012 'C:\Program Files\Git\usr\bin\bash.exe'
}; )"
ps_table() { # extra row -> a `cat <<TABLE` stub body
  printf 'cat <<TABLE\n%s\n%s\n%s\nTABLE\n' "$ps_header" "$ps_background" "$1"
}
# A table in which our staged daemon is running, and one in which it is not.
table_with_daemon="$(ps_table "$(row 4242 "$server_win")")"
table_without_daemon="$(ps_table "$(row 4242 'C:\Windows\System32\spoolsv.exe')")"
# The same table, under a header this parser cannot read.
table_bad_header="${table_with_daemon/COMMAND/CMDLINE}"

# Round 13f, the three shapes the completeness witnesses exist for. Each is a
# table the parser can read, containing no wenlan-server row, that must come
# back "could not measure" rather than "nothing to reap" -- because absence
# from a table that is not the whole table is not absence.
#
#   no System process: ten rows, none of them WINPID 4
table_no_system="$(printf 'cat <<TABLE\n%s\n%s\n%s\nTABLE\n' "$ps_header" \
  "$(printf '%s\n' "$ps_background" | tail -n +2)" \
  "$(row 4242 'C:\Windows\System32\spoolsv.exe')")"
#   too short to be a machine: three rows, WINPID 4 among them
table_too_short="$(printf 'cat <<TABLE\n%s\n%s\n%s\n%s\nTABLE\n' "$ps_header" \
  "$(row 4 'System')" "$(row 920 'C:\Windows\System32\svchost.exe')" \
  "$(row 4242 'C:\Windows\System32\spoolsv.exe')")"
#   a merged diagnostic riding alongside a complete-looking table. NINE fields,
#   so the old `NF < 8` floor counted it as a row: this is the case that shows
#   the field count was a word count, and the stderr merge above it decorative
#   for exactly the lines it was added to catch.
table_contaminated="$(printf 'cat <<TABLE\n%s\n%s\n%s\n%s\nTABLE\n' "$ps_header" \
  "$ps_background" 'ps: could not read the process table for you' \
  "$(row 4242 "$server_win")")"

# tasklist answers the liveness poll after the (stubbed) kill. Round 13e made
# `process_is_alive` demand three independent things of a CSV table before it
# will read a missing pid as "gone" -- every line a row, pid 4 present, and at
# least ten rows -- because a short fragment is what a half-answered tasklist
# looks like. The fixture here WAS a two-row fragment, so after that fix it
# measured as "could not measure" (2), not "gone" (1), and this harness's
# positive case failed on a stub that was never a plausible table. It is now
# the full background table scripts/host-process.test.ts uses, minus our pid:
# the staged daemon is the one process that is not in it.
gone_rows=(
  '"System","4","Console","1","9,000 K"'
  '"Registry","132","Console","1","9,000 K"'
  '"smss.exe","608","Console","1","9,000 K"'
  '"csrss.exe","888","Console","1","9,000 K"'
  '"wininit.exe","964","Console","1","9,000 K"'
  '"services.exe","1048","Console","1","9,000 K"'
  '"lsass.exe","1072","Console","1","9,000 K"'
  '"svchost.exe","1576","Console","1","9,000 K"'
  '"svchost.exe","1704","Console","1","9,000 K"'
  '"explorer.exe","5320","Console","1","9,000 K"'
  '"bash.exe","9012","Console","1","9,000 K"'
)
gone_table="$({ printf 'cat <<TABLE\n'; printf '%s\n' "${gone_rows[@]}"; printf 'TABLE\n'; })"

# The stub records "<winpid> <image>", and the shipped call passes the recorded
# server path, not the table's spelling of it. Only the hand-off case reaches
# the kill helper at all, and the row it hands over must be OUR row: pid 4242,
# not the svchost row beside it.
killed_ours="4242 $server"

# `finds-our-row-hands-winpid` is named for what it proves. It does NOT prove
# anything was reaped: `force_terminate_process` is replaced by a no-op that
# records its arguments and returns 0, and `tasklist` is a stub that reports
# 4242 absent whatever happened. What the case measures is the scan -- that the
# shipped parse picks our row out of the table and hands the WINPID and the
# recorded image path to the kill helper -- plus the shipped liveness poll's
# reading of a table in which that pid is gone. The integration of the real
# kill with the real liveness probe is out of this harness's reach and is not
# claimed anywhere in it.
CASES=(
  "finds-our-row-hands-winpid|table_with_daemon|0|ours"
  "nothing-matching-to-reap|table_without_daemon|1|none"
  "header-it-cannot-read|table_bad_header|2|none"
  "table-with-no-system-proc|table_no_system|2|none"
  "table-too-short-to-be-one|table_too_short|2|none"
  "table-with-a-diagnostic-in|table_contaminated|2|none"
)

run_all() { # driver
  PASSED_CASES=(); FAILED_CASES=()
  local spec name body want killed
  for spec in "${CASES[@]}"; do
    IFS='|' read -r name body want killed <<<"$spec"
    body="${!body}"
    if [[ "$killed" == "ours" ]]; then killed="$killed_ours"; else killed=""; fi
    if run_case "$name" "$1" "$body" "$gone_table" "$want" "$killed"; then
      PASSED_CASES+=("$name")
    else
      FAILED_CASES+=("$name")
    fi
  done
}

echo "dev-runtime-scan-controls"
echo "cases against the shipped scan:"
REAP="$(extract "$runtime_before")"
# The scan is asked for by NAME now. An extraction that lost the call would
# leave a function that reaps nothing, and every case would fail for that
# instead of for what it names.
if [[ "$REAP" != *"ps_w_rows_matching"* ]]; then
  echo "FATAL: reap_staged_daemon no longer asks the library for its scan" >&2
  exit 1
fi
build_subject "$lib_before" "$REAP"
run_all "$work/driver.sh"
failures=$((failures + ${#FAILED_CASES[@]}))

echo "controls:"
control() { # name, why, which(lib|reap), old, new, must_fail...
  local name="$1" why="$2" which="$3" old="$4" new="$5"; shift 5
  local -a must_fail=("$@")
  printf '  %s  (%s)\n' "$name" "$why"
  local text head tail hits
  case "$which" in
    lib) text="$lib_before" ;;
    reap) text="$REAP" ;;
    *) printf '    FAIL unknown control target %s\n' "$which"; failures=$((failures + 1)); return ;;
  esac
  # EXACTLY once, and a stale anchor is a hard error rather than a quiet
  # no-test. Two of the five anchors below have a near-twin in the same file --
  # `process_is_alive` demands the same completeness witnesses of `tasklist`,
  # spelled the same and indented differently -- so "at least once" was one
  # reindentation away from mutating the wrong parser and reporting on it.
  hits="$(count_occurrences "$text" "$old")"
  if [[ "$hits" != 1 ]]; then
    printf '    FAIL anchor matched %s times in %s, wanted exactly 1; this control tests nothing\n' \
      "$hits" "$which"
    failures=$((failures + 1))
    return
  fi
  # Matching once is not the same as CHANGING something. A control written by
  # copying the one above it and editing neither string replaces a span of the
  # file with exactly what was already there, runs green, and is reported as
  # "the suite passed with the fix reverted".
  if [[ "$old" == "$new" ]]; then
    printf '    FAIL the replacement is identical to the anchor; this control reverts nothing\n'
    failures=$((failures + 1))
    return
  fi
  head="${text%%"$old"*}"
  tail="${text#*"$old"}"
  case "$which" in
    lib) build_subject "$head$new$tail" "$REAP" ;;
    reap) build_subject "$lib_before" "$head$new$tail" ;;
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
  # list used to be HAND-WRITTEN beside the must_fail name, and the comment on
  # `nc-scan-parses-the-table-itself` says in as many words that its mutation
  # reddens FOUR of the six cases while naming one failure and two survivors --
  # so three real failures were credited silently, and a mutation with
  # collateral damage was indistinguishable from one pinned to its fix. The
  # survivor set is the case list minus this control's own must_fail names, so
  # it cannot fall behind CASES and undeclared collateral is a control failure.
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

# The mutation reverts exactly the header rule and nothing else: the `notarow`
# guard and the END print loop, both added in later rounds, stay as shipped, so
# a failure here can only be the header rule. Anchored on the awk as it stands
# today -- when that awk changes, the anchor stops matching and this harness
# says so out loud rather than testing a program that no longer exists.
#
# Round 13h: that awk is now the library's, and only its `scan` rule is touched
# here. `ps_w_row_for`'s `row` rule is left alone, so the four probes in
# scripts/host-process.test.ts are not what these cases are measuring.
#
# A header that says CMDLINE zeroes `col` and nothing else -- PID and WINPID are
# still there -- so the witnesses below still pass and the ONLY thing standing
# between this table and a confident "nothing to reap" is the `col` test. Drop
# it and `substr($0, 0)` hands back the whole line, which no recorded path can
# equal, and the poll runs out and returns 1.
control nc-scan-header-unchecked \
  'a ps -W header without a COMMAND column means "nothing to reap"' lib \
  '    mode == "scan" && wp && col && tolower(substr($0, col)) ~ want {
      row[++seen] = $wp " " substr($0, col)
    }
    END {
      if (!kc || !wp || !col) exit 3' \
  '    mode == "scan" && wp && tolower($0) ~ want {
      row[++seen] = $wp " " substr($0, col)
    }
    END {
      if (!kc || !wp) exit 3' \
  header-it-cannot-read

# The point of recording what the stub was handed. `ps -W` reports two identities
# per process and only the Windows one can be killed; taking the MSYS pid instead
# still returns 0 here, because the pid it then polls is genuinely not running.
# Without the killed-argument assertion this mutation is invisible.
control nc-scan-kills-the-msys-pid \
  'handing the kill helper the MSYS pid instead of the WINPID still exits 0' lib \
  '      row[++seen] = $wp " " substr($0, col)' \
  '      row[++seen] = $1 " " substr($0, col)' \
  finds-our-row-hands-winpid

# Round 13f. One control per completeness witness, each pinned to the one case
# that exists for it, so a mutation that reddens the whole table is reported as
# unpinned rather than as three successes.
control nc-scan-row-shape-is-a-word-count \
  'the round-13e form: eight FIELDS is a row, so a nine-word diagnostic is one' lib \
  '    !($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ &&
      $4 ~ /^[0-9]+$/) { notarow = 1; next }' \
  '    NF < 8 { notarow = 1 }' \
  table-with-a-diagnostic-in

# These two anchors have a near-twin eight columns to the left, in the tasklist
# parse of `process_is_alive`, which demands the same witnesses for the same
# reason. That is what the exact-count check above is for: matched on the
# spelling alone, either of them would mutate whichever came first.
control nc-scan-system-process-unchecked \
  'a ps -W table with no System process is still a whole process table' lib \
  '      if (notarow) exit 3
      if (!system_process) exit 3' \
  '      if (notarow) exit 3
      if (0) exit 3' \
  table-with-no-system-proc

control nc-scan-row-floor-unchecked \
  'three rows is a machine, and absence from three rows is "nothing to reap"' lib \
  '      if (rows < 10) exit 3
      if (!seen) exit 1' \
  '      if (0) exit 3
      if (!seen) exit 1' \
  table-too-short-to-be-one

# Round 13h, and the reason the parse moved: the scan used to be a hand-written
# copy of the library's table walk, and the copy is where the hardening was
# missed -- three rounds of witnesses went into one copy while the other still
# counted words. If `reap_staged_daemon` ever grows its own parse again, every
# library mutation above stops reaching it, and each of them would report
# "survived" for a subject that no longer contains the code under test. This
# control puts a private parse back and requires the cases to notice.
#
# It reddens FOUR of the six cases, because a private parse has none of the
# witnesses, and all four are named -- which is round 4's finding 7. This
# comment used to say the same sentence beside a scorer that required ONE
# failure and two hand-listed survivors, so three of the four were credited
# silently and a mutation with collateral damage looked exactly like one pinned
# to its fix. The survivor set is computed now, so the four have to be declared.
control nc-scan-parses-the-table-itself \
  'the scan re-implements the parse, so the library hardening no longer reaches it' reap \
  '    scan_rc=0
    rows="$(ps_w_rows_matching wenlan-server)" || scan_rc=$?' \
  '    # INJECTED: a fourth hand-maintained copy of the table walk. It reads the
    # COMMAND column out of the header, as the third one did -- and, as the
    # third one did before it was hardened, treats a header it cannot read as
    # an ordinary table rather than as a failed measurement.
    scan_rc=0
    rows="$(ps -W 2>/dev/null | awk '"'"'
      NR == 1 { col = index($0, "COMMAND"); next }
      tolower($0) ~ /wenlan-server/ { print $4 " " substr($0, col) }
    '"'"')" || scan_rc=$?' \
  header-it-cannot-read table-with-no-system-proc table-too-short-to-be-one \
  table-with-a-diagnostic-in
# --- the listener table's completeness ---------------------------------------
#
# ROUND 4 (Codex Sol), NEW FINDING 1, and it is the same shape as everything
# above it: the defect was INSIDE the remedy written for the defect class.
#
# `listener_pid_for_port` merges stderr (`netstat -ano 2>&1`) and its comment
# said, in as many words, that a merged warning line "is then not a row, and
# 'every non-empty line is a row' turns it into a refusal". The awk under that
# comment did not do it. Its only malformed-line rule fired on lines whose
# FIRST TOKEN was already `TCP` or `UDP`, so a status-0 `WARNING: partial
# results` or `Access denied` -- which begins with neither -- was matched by no
# rule at all. The valid rows satisfied the guard, the port query found nothing
# for our port in the truncated remainder, and the function returned 1:
# MEASURED FREE, off a table netstat itself had complained about. That answer
# is what deletes an ownership record and starts a second daemon on a port that
# is already held.
#
# This section drives the library's real `listener_pid_for_port` over stubbed
# netstat output, exactly as the scan cases above drive it over stubbed `ps -W`,
# and the control reverts the rule to its round-3 form and requires the case to
# turn from "could not measure" back into "free".
listener_driver() { # library-text -> a driver at $work/listener.sh
  if [[ -z "$1" || "$1" != *"listener_pid_for_port() {"* ]]; then
    echo "FATAL: listener_pid_for_port is not in the library this harness parses" >&2
    exit 1
  fi
  printf '%s\n' "$1" >"$work/host-process-listener.sh"
  {
    printf '#!/usr/bin/env bash\nset -uo pipefail\n'
    printf '. "%s"\n' "$work/host-process-listener.sh"
    printf 'rc=0\n'
    printf 'out="$(listener_pid_for_port "$1")" || rc=$?\n'
    printf 'printf "rc=%%s out=%%s\\n" "$rc" "$out"\n'
  } >"$work/listener.sh"
}

listener_case() { # name, netstat-body, want ("rc=N out=X")
  local name="$1" body="$2" want="$3"
  local bin="$work/lbin-$RANDOM$RANDOM" out="" rc=0 got=""
  mkdir -p "$bin"
  printf '#!/usr/bin/env bash\n%s\n' "$body" >"$bin/netstat"
  chmod 0755 "$bin/netstat"
  local shim="$bin"
  if command -v cygpath >/dev/null 2>&1; then shim="$(cygpath -u "$bin")"; fi
  out="$(PATH="$shim:$PATH" WENLAN_HOST_PROCESS_PLATFORM=windows \
    bash "$work/listener.sh" "$LPORT" 2>&1)" || rc=$?
  # The driver prints one line; stderr is merged, so take the LAST one.
  got="${out##*$'
'}"
  if [[ "$got" != "$want" ]]; then
    printf '  FAIL %-28s [%s], wanted [%s]\n' "$name" "${got:-<none>}" "$want"
    return 1
  fi
  printf '  ok   %-28s\n' "$name"
  return 0
}

LPORT=17931
# A netstat header and rows in the shape Windows really prints. The banner and
# the column header are the entire preamble on this host -- measured: 183 lines
# of `netstat -ano`, 2 blank, 179 protocol rows, exactly 2 non-row non-blank
# lines -- and both are LOCALISED, so the rule can only count them.
lheader=$'\nActive Connections\n\n  Proto  Local Address          Foreign Address        State           PID'
lrow() { printf '  TCP    127.0.0.1:%s         0.0.0.0:0              LISTENING       %s' "$1" "$2"; }
# `netstat -ano` prints the whole TCP table and then the whole UDP table, so a
# UDP row is the witness that the TCP section ENDED rather than stopped early.
# Every table built by `ltable` is a WHOLE one and carries it; the truncated
# fixture below is the one that deliberately does not.
ludp='  UDP    0.0.0.0:5353           *:*                                    9528'
ltable() { printf 'cat <<'"'"'TABLE'"'"'\n%s\n%s\n%s\nTABLE\n' "$lheader" "$1" "$ludp"; }
ltable_cut() { printf 'cat <<'"'"'TABLE'"'"'\n%s\n%s\nTABLE\n' "$lheader" "$1"; }

# The port asked for is ABSENT from every row below, which is the whole point:
# this is the case that used to come back "free".
lrows_clean="$(lrow 17999 4242)"$'\n'"$(lrow 18000 4243)"
lrows_dirty="$lrows_clean"$'\n''WARNING: partial results'
ltable_clean="$(ltable "$lrows_clean")"
ltable_dirty="$(ltable "$lrows_dirty")"
# The same diagnostic landing in the PREAMBLE instead. `2>&1` does not order
# the two streams, so a rule that only watches the rows watches half the file.
ltable_dirty_pre="$(printf 'cat <<'"'"'TABLE'"'"'\n%s\n%s\n%s\n%s\nTABLE\n' \
  "$lheader" 'Access denied' "$lrows_clean" "$ludp")"
# And the port present, so the "still measures something" control below is not
# satisfied by a rule that refuses everything.
ltable_hit="$(ltable "$(lrow "$LPORT" 4242)")"
# ROUND 5. The same well-formed rows with the table CUT before the UDP section:
# every line that arrived is a valid row, our port is not among them, and a
# grammar alone reads that as "free". It is a table that stopped, and the only
# honest answer to a port question over it is "could not measure".
ltable_truncated="$(ltable_cut "$lrows_clean")"

LISTENER_CASES=(
  "listener-found|ltable_hit|rc=0 out=4242"
  "listener-free-clean-table|ltable_clean|rc=1 out="
  "listener-diagnostic-after-rows|ltable_dirty|rc=2 out="
  "listener-diagnostic-in-preamble|ltable_dirty_pre|rc=2 out="
  "listener-truncated-tcp-section|ltable_truncated|rc=2 out="
)

run_listener_all() {
  LPASSED=(); LFAILED=()
  local spec name body want
  for spec in "${LISTENER_CASES[@]}"; do
    IFS='|' read -r name body want <<<"$spec"
    body="${!body}"
    if listener_case "$name" "$body" "$want"; then LPASSED+=("$name"); else LFAILED+=("$name"); fi
  done
}

echo "cases against the shipped listener table rule:"
listener_driver "$lib_before"
run_listener_all
failures=$((failures + ${#LFAILED[@]}))

listener_control() { # name, why, old, new, must_fail...
  local name="$1" why="$2" old="$3" new="$4"; shift 4
  local -a must_fail=("$@")
  printf '  %s  (%s)\n' "$name" "$why"
  local hits head tail
  hits="$(count_occurrences "$lib_before" "$old")"
  if [[ "$hits" != 1 ]]; then
    printf '    FAIL anchor matched %s times, wanted exactly 1; this control tests nothing\n' "$hits"
    failures=$((failures + 1))
    return
  fi
  if [[ "$old" == "$new" ]]; then
    printf '    FAIL the replacement is identical to the anchor; this control reverts nothing\n'
    failures=$((failures + 1))
    return
  fi
  head="${lib_before%%"$old"*}"
  tail="${lib_before#*"$old"}"
  listener_driver "$head$new$tail"
  run_listener_all >"$logs/$name.log" 2>&1
  local want
  for want in "${must_fail[@]}"; do
    if printf '%s\n' "${LFAILED[@]:-}" | grep -qx -- "$want"; then
      printf '    ok   caught:   %s\n' "$want"
    else
      printf '    FAIL survived: %s -- the case does not defend this fix\n' "$want"
      failures=$((failures + 1))
    fi
  done
  # The survivor set is computed, for the reason given on `control` above: a
  # hand-written list cannot fall behind LISTENER_CASES, and undeclared
  # collateral damage must be a control failure rather than a silent credit.
  local case_spec case_name
  for case_spec in "${LISTENER_CASES[@]}"; do
    case_name="${case_spec%%|*}"
    if printf '%s\n' "${must_fail[@]:-}" | grep -qx -- "$case_name"; then continue; fi
    if printf '%s\n' "${LPASSED[@]:-}" | grep -qx -- "$case_name"; then
      printf '    ok   survived: %s\n' "$case_name"
    else
      printf '    FAIL also failed: %s -- the control is not pinned to the fix\n' "$case_name"
      failures=$((failures + 1))
    fi
  done
  # Restore the subject for anything that runs after this control.
  listener_driver "$lib_before"
}

# THE control the round-4 finding asks for. The mutation is the awk exactly as
# it stood in round 3 -- a malformed-line rule that only ever looks at lines
# beginning `TCP` or `UDP` -- and nothing else changes. Under it, a table with a
# diagnostic merged into it and our port absent goes back to answering rc=1:
# the measured "free" that starts a second daemon on a held port.
#
# `listener-found` and `listener-free-clean-table` must both survive, because a
# rule that answered 2 to everything would catch the diagnostic cases while
# destroying the two answers the caller actually acts on.
#
# ROUND 6 re-anchored this: the anchor is the shipped grammar VERBATIM, so it
# now carries the blank-line/`sep` bookkeeping the preamble-shape rule added,
# and it must be updated in lockstep with the library or `count_occurrences`
# returns 0 and the control silently tests nothing -- which is exactly how this
# was found. The shape rule itself (banner, blank, header) is pinned by its own
# control in scripts/negative-controls/posix-probes-negative-controls.py, over
# fixtures this harness does not carry; here it is only along for the ride.
listener_control nc-listener-diagnostic-is-not-a-row \
  'a merged netstat warning beside valid rows reads as "port free"' \
  '      # A blank line is not a preamble line, but WHERE it falls is evidence:
      # `blank` records that one was seen since the last non-blank pre-row
      # line, which is how the banner is told from a diagnostic sitting
      # directly on top of the header. After the first row it is irrelevant --
      # the real table has blank lines among its rows.
      /^[[:space:]]*$/ { if (!rows) blank = 1; next }
      ($1 == "TCP" && NF == 5 && $5 ~ /^[0-9]+$/) ||
        ($1 == "UDP" && NF == 4 && $4 ~ /^[0-9]+$/) {
          rows = 1
          # The ordering the UDP witness depends on, checked rather than
          # assumed: a TCP row after a UDP row means this stream is not "all of
          # TCP, then all of UDP", so a UDP row in it says nothing about whether
          # the TCP section finished. One line and braced, because a comment
          # between an `if` body and its `else` is not portable awk.
          if ($1 == "UDP") { udp = 1 } else if (udp) { tcp_after_udp = 1 }
          if ($1 == "TCP" && ($3 == "0.0.0.0:0" || $3 == "[::]:0" || $3 == "*:*")) found = 1
          next
        }
      { if (rows) { after = 1; next }
        preamble++
        if (preamble == 2) sep = blank
        blank = 0
      }
      END {
        if (after) exit 2
        if (preamble < 1 || preamble > 2) exit 2
        if (preamble == 2 && !sep) exit 2
        if (tcp_after_udp) exit 4
        if (!udp) exit 3
        exit(found ? 0 : 1)
      }' \
  '      ($1 == "TCP" || $1 == "UDP") &&
        !(($1 == "TCP" && NF == 5 && $5 ~ /^[0-9]+$/) ||
          ($1 == "UDP" && NF == 4 && $4 ~ /^[0-9]+$/)) { notarow = 1 }
      $1 == "TCP" && NF == 5 && $5 ~ /^[0-9]+$/ &&
        ($3 == "0.0.0.0:0" || $3 == "[::]:0" || $3 == "*:*") { found = 1 }
      END {
        if (notarow) exit 2
        exit(found ? 0 : 1)
      }' \
  listener-diagnostic-after-rows listener-diagnostic-in-preamble \
  listener-truncated-tcp-section

# And the preamble half on its own, so the two rules are pinned separately. A
# rule that only watched what comes AFTER the first row would pass the control
# above and still let a warning printed BEFORE the table ride along. ROUND 6
# split `if (after || preamble > 2)` into two statements; this control keeps the
# BUDGET half -- dropping the upper bound alone is enough to let a diagnostic
# stand in the preamble unchallenged, and the `after` half is already pinned by
# the whole-grammar control above.
listener_control nc-listener-preamble-uncounted \
  'a diagnostic ahead of the first row is not counted against the preamble' \
  '        if (preamble < 1 || preamble > 2) exit 2' \
  '        if (preamble < 1) exit 2' \
  listener-diagnostic-in-preamble

# ROUND 5, and the case the two rules above cannot reach. They are a GRAMMAR:
# a table cut after a well-formed row is still all well-formed rows, so the
# validator passes, the port query finds nothing, and the answer is a MEASURED
# FREE off a stream that stopped early -- the same consequence as the merged
# diagnostic, arrived at by silence instead of by text. The witness is the UDP
# section, because netstat prints all of TCP and then all of UDP; the mutation
# takes it away and nothing else changes. `listener-found` and
# `listener-free-clean-table` must both survive, so this is pinned to the
# truncation and not to "refuse more".
listener_control nc-listener-truncation-unwitnessed \
  'a table cut before the UDP section reads as "port free"' \
  '        if (!udp) exit 3' \
  '        if (0) exit 3' \
  listener-truncated-tcp-section


if [[ "$(cat "$runtime")" != "$runtime_before" ]]; then
  echo "FATAL: scripts/dev-runtime.sh changed during the run"; exit 1
fi
if [[ "$(cat "$lib")" != "$lib_before" ]]; then
  echo "FATAL: scripts/lib/host-process.sh changed during the run"; exit 1
fi

echo "CONTROL FAILURES: $failures"
(( failures == 0 )) || exit 1
