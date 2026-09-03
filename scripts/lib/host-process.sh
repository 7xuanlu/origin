#!/usr/bin/env bash
# Host process primitives, shared by the dev runtime and the surface smokes.
# Source this file; never run it. It defines functions and one platform
# constant and touches nothing else -- no directories, no traps, no state.
#
# THE INVARIANT THIS FILE EXISTS TO CARRY:
#
#   A FAILED MEASUREMENT MUST NEVER BE INDISTINGUISHABLE FROM A NEGATIVE
#   MEASUREMENT.
#
# Every probe here is TRI-STATE -- measured / negative / could not measure --
# and returns a distinct status for the third case so that callers can branch
# on it. A missing or broken tool produces empty output, and empty output must
# never read as "port free", "process dead" or "not our image".
#
# Return conventions used throughout:
#   0  measured, affirmative (found / alive / matched)
#   1  measured, negative    (no listener / not running / no such process)
#   2  COULD NOT MEASURE     (tool absent, tool failed, output unparseable)
#
# Capturing those statuses under `set -e` needs care: `out="$(f)"; rc=$?`
# ABORTS at the assignment, because the assignment itself is the failing
# command and `rc=$?` never runs. Use `if out="$(f)"; then ... else rc=$?; fi`,
# or the `probe_*` wrappers below, which set a named state variable instead.
#
# SIGPIPE: adding `|| return N` to a pipeline whose consumer `exit`s on its
# first match CAN make the producer take SIGPIPE, and under `pipefail` that 141
# is indistinguishable from a real parser failure -- a correctness fix turned
# into a spurious failure. So no awk program here exits early: they set a flag
# and print at `END`. For the same reason `sed ... | head -1` is never used;
# where a first line is wanted it is a single `sed -n '...{...;p;q;}'`.

# --- platform ----------------------------------------------------------------
# Windows needs its own branch for all of this because none of the POSIX pieces
# exist there: no `lsof`, and an MSYS `ps` with neither `-p` nor `-o`. A process
# launched from bash also carries two identities, the MSYS pid that `$!` yields
# and the Windows pid every Windows tool reports, so callers record the Windows
# pid and compare like with like.
#
# A bare `case "$(uname -s)"` has nowhere to put a failure: a `uname` that is
# absent, exits non-zero or prints nothing yields the empty string, which
# matches no Windows pattern and falls into `*)` -- an unmeasured platform read
# as a confident POSIX, which on Windows reaches for `lsof` (absent), a POSIX
# `kill` (cannot reach a native process by its Windows pid) and the unstaged
# server path. So the platform is a measurement with three answers and the
# third REFUSES: sourcing fails, which `dev-runtime.sh`'s EXIT trap turns into
# `DEV_RUNTIME_RESULT: unknown`.
HOST_IS_WINDOWS=""
_host_uname=""
_host_uname_rc=0
_host_uname="$(uname -s 2>/dev/null)" || _host_uname_rc=$?
if (( _host_uname_rc == 0 )) && [[ -n "$_host_uname" ]]; then
  case "$_host_uname" in
    MINGW* | MSYS* | CYGWIN*) HOST_IS_WINDOWS=1 ;;
    *) HOST_IS_WINDOWS=0 ;;
  esac
fi

# Tests only, and deliberately one-way: a POSIX host may opt INTO the Windows
# branch so the Windows-only measurement paths can be exercised by the suite on
# every platform (`app-check` runs on macOS, so nothing else would ever run
# them). The reverse is impossible on purpose -- a real Windows host can never
# be talked out of its identity-checked kill path. On a POSIX host the forced
# branch finds no `netstat`/`tasklist`/`ps -W`, so every probe answers "could
# not measure", which is the safe direction.
#
# It is also the one thing that can answer the platform question when `uname`
# could not: an explicit declaration is a measurement somebody made, which is
# exactly what the empty string is not.
if [[ "${WENLAN_HOST_PROCESS_PLATFORM:-}" == "windows" ]]; then
  HOST_IS_WINDOWS=1
fi

if [[ "$HOST_IS_WINDOWS" != "0" && "$HOST_IS_WINDOWS" != "1" ]]; then
  echo "error: could not measure which platform this is" >&2
  echo "       uname -s exited $_host_uname_rc and said [$_host_uname]" >&2
  echo "       every probe below branches on it, and guessing POSIX on a" >&2
  echo "       Windows host reaches for lsof, a POSIX kill and an unstaged" >&2
  echo "       daemon path -- three wrong answers from one unread status" >&2
  echo "       set WENLAN_HOST_PROCESS_PLATFORM=windows to declare it" >&2
  # Sourced, which is the only supported use; the `exit` is for the case where
  # somebody runs this file anyway, where `return` is an error rather than a way
  # out.
  return 1 2>/dev/null || exit 1
fi
unset _host_uname _host_uname_rc

# --- path spelling -----------------------------------------------------------

# Values printed for a consumer outside this shell. The desktop app and the
# native daemon call `std::fs::canonicalize` on the directories they are handed,
# and a Windows build cannot resolve an MSYS path like /tmp/wenlan-app-dev/<id>.
# Only the boundary converts; the shell keeps working in its own form, because
# its own `rm -rf` still wants the MSYS spelling. Two variables, never one
# rewritten variable.
#
# exit: 0 converted, 2 the conversion could not be made (never a silent empty
# string, which would send a daemon to the filesystem root).
native_path() {
  local out
  if (( HOST_IS_WINDOWS == 1 )); then
    out="$(cygpath -m "$1" 2>/dev/null)" || return 2
    [[ -n "$out" ]] || return 2
    printf '%s' "$out"
  else
    printf '%s' "$1"
  fi
}

# `ps -W` reports the MSYS spelling for a process it launched and the Windows
# spelling for anything else, drops the .exe from the first, and does not
# promise a case. The recorded server path is whatever the caller wrote. Both
# sides go through this so the identity check compares one spelling.
#
# exit: 0 normalized, 2 the normalization could not be made. Never fall back to
# the raw string: an unmeasurable spelling is not a mismatched one.
normalize_program_path() {
  local path="${1%.[eE][xX][eE]}" out
  if (( HOST_IS_WINDOWS == 1 )); then
    out="$(cygpath -m "$path" 2>/dev/null)" || return 2
    [[ -n "$out" ]] || return 2
    printf '%s' "$out" | tr '[:upper:]' '[:lower:]'
  else
    printf '%s' "$path"
  fi
}

# Does the kernel say this pid does not exist?
# exit: 0 ESRCH -- there is no such process. 1 anything else, INCLUDING EPERM,
#       which is not a negative at all: EPERM is the kernel confirming the
#       process EXISTS and is simply not ours to signal.
#
# `kill -0` reports ESRCH and EPERM through the same shell status, so the status
# alone cannot tell "gone" from "there, and not yours". The message can, and in
# the C locale it is the string the C library defines for ESRCH -- so `LC_ALL=C`
# here is not tidiness, it is what turns reading the message into a measurement
# instead of a guess about the operator's language.
#
# Everything this cannot classify answers 1, the safe direction: the caller
# treats the pid as possibly alive rather than deleting its ownership record.
#
# THE RESIDUAL, recorded rather than fixed: this depends on the exact text
# bash's builtin prints in the C locale. If that text changes the helper
# degrades to "always cautious" rather than "always confident" -- a lock that is
# never recovered rather than one that is stolen.
errno_says_no_such_process() {
  local err
  # Tests only, and ONE-WAY on purpose, exactly like the platform override
  # above. EPERM cannot be produced on demand from a test -- it needs a process
  # owned by somebody else -- and `kill` is a shell builtin, so it cannot be
  # shimmed onto PATH either. This forces the CAUTIOUS answer so that callers'
  # handling of it can be measured. Nothing can force the negative: that one
  # has to come from the kernel, which is the direction that matters.
  [[ "${WENLAN_HOST_PROCESS_FORCE_EPERM:-}" == "1" ]] && return 1
  err="$(LC_ALL=C LANGUAGE=C kill -0 "$1" 2>&1)" && return 1
  # Measured on this host: bash refuses a pid past its own integer range with
  # "arguments must be process or job IDs", which is not ESRCH and must not be
  # read as one -- the question was rejected, not answered.
  [[ "$err" == *"No such process"* ]]
}

# --- tri-state measurements --------------------------------------------------

# Is a process still running?
# exit: 0 alive, 1 not alive, 2 could not measure.
#
# A failed `tasklist` prints nothing, and "nothing" must not read as dead: the
# caller's next move is deleting the ownership record of a daemon that is still
# running, after which nothing can identity-check it again.
process_is_alive() {
  local out rc
  # A pid that is not a pid is not a dead process. `tasklist //FI "PID eq abc"`
  # prints the no-such-task notice and `kill -0 abc` fails, so both branches
  # would otherwise answer "not alive" to a question they never asked.
  [[ "$1" =~ ^[0-9]+$ ]] || return 2
  if (( HOST_IS_WINDOWS == 1 )); then
    command -v tasklist >/dev/null 2>&1 || return 2
    # NOT a filtered query. Measured: `tasklist //FI "PID eq N"` exits 0 and
    # prints one line of prose both for an absent pid and for anything else
    # that goes wrong, with no CSV row either way -- so "no row" cannot tell a
    # real miss from a broken tool. The WHOLE table is self-validating instead.
    # stderr is merged rather than dropped, so a warning line is not a row.
    #
    # THE COST, accepted deliberately: ~0.35s per call against a filtered
    # query's ~0.05s, so `dev-runtime.sh`'s bounded fifty-round polls take ~22s
    # instead of ~5s in their worst case.
    #
    # Three independent conditions, because none alone means "complete":
    #
    #   every non-empty line is a CSV row -- a leading quote and at least five
    #       quote-comma-quote fields, pid in field 2. Nothing else got into the
    #       stream: no diagnostic, no merged stderr, no half-written record.
    #   pid 4 is present -- the System process exists on every Windows NT
    #       kernel from boot to shutdown.
    #   at least 10 rows -- the weakest of the three; it rejects a fragment that
    #       is well formed and contains pid 4. Measured on this host: 267 rows.
    #       smss, csrss, wininit, services, lsass and several svchosts run
    #       before anything user-visible starts.
    out="$(tasklist //NH //FO CSV 2>&1)" || return 2
    [[ -n "$out" ]] || return 2
    rc=0
    printf '%s\n' "$out" | tr -d '\r' | awk -v pid="$1" '
      BEGIN { FS = "\",\"" }
      /^[[:space:]]*$/ { next }
      { lines++ }
      /^"/ && NF >= 5 {
        rows++
        if ($2 == pid) hit = 1
        if ($2 == "4") system_process = 1
      }
      END {
        if (rows != lines) exit 3
        if (!system_process) exit 3
        if (rows < 10) exit 3
        exit(hit ? 0 : 1)
      }
    ' || rc=$?
    case "$rc" in
      0) return 0 ;;
      1) return 1 ;;
      *) return 2 ;;
    esac
  else
    # `kill -0` cannot answer this alone: ESRCH and EPERM share status 1, so a
    # daemon running as another user would read as dead. Kill answers only the
    # affirmative case; `ps -p` settles the rest because it lists a pid whoever
    # owns it. stderr is the discriminator for its failure -- silence means no
    # such pid, any text (a busybox `ps` without -p, a denied /proc) means the
    # question was not answered.
    kill -0 "$1" 2>/dev/null && return 0
    command -v ps >/dev/null 2>&1 || return 2
    rc=0
    out="$(ps -p "$1" -o pid= 2>&1)" || rc=$?
    if (( rc == 0 )); then
      [[ "$out" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 2
      return 0
    fi
    # Only status 1 can mean "no such pid": a signal death (128+N) and a usage
    # error (2) are both silent and both nonzero.
    #
    # And silence from `ps` is not by itself absence. Under Linux
    # `hidepid=invisible` another user's process is not in /proc at all, so
    # `ps -p` is silent with status 1 for a process that is running -- while
    # `kill -0` fails with EPERM, which is the kernel saying it exists. Only
    # ESRCH may close this out.
    (( rc == 1 )) || return 2
    [[ -z "$out" ]] || return 2
    errno_says_no_such_process "$1" || return 2
    return 1
  fi
}

# --- the `ps -W` table -------------------------------------------------------

# `ps -W` is read, validated and parsed in EXACTLY ONE PLACE: `_ps_w_read`.
# A different QUESTION is not a reason for a different parse, so the shape of
# the answer is what varies:
#
#   ps_w_row_for KEY WANT      -- the first row whose KEY column is WANT.
#   ps_w_rows_matching PATTERN -- every row whose COMMAND matches PATTERN
#                                 (an ERE, matched against the lowercased
#                                 command), in table order, one per line.
#
# Both are one line long, both call `_ps_w_read`, and the witnesses below have
# exactly one place to be edited. A second CALL SITE is fine; a second PARSE is
# the defect, which is why `scripts/host-process.test.ts` pins the number of
# places that run the command at all -- across this file AND the scripts that
# source it, since a library-local check cannot see a copy outside the library.
#
# args:   mode -- "row" or "scan"; the key column -- "PID", the MSYS pid `$!`
#         yields, or "WINPID", the Windows pid every Windows tool reports; and
#         the value wanted in that column ("row") or the command pattern
#         ("scan", which always keys on WINPID because that is the identity a
#         caller can act on).
# stdout: "<WINPID> <COMMAND>" per matching row.
# exit:   0 matched, 1 the table was whole and contained no such row, 2 could
#         not measure.
_ps_w_read() {
  local mode="$1" key="$2" want="$3" out rc=0 line
  # A mode this function does not know would fall through every match rule and
  # print nothing -- a question that was never asked, answered "no".
  [[ "$mode" == "row" || "$mode" == "scan" ]] || return 2
  # A key this parser does not know would resolve to column zero and match
  # nothing -- the same question never asked, answered the same way.
  [[ "$key" == "PID" || "$key" == "WINPID" ]] || return 2
  command -v ps >/dev/null 2>&1 || return 2
  # stderr merged, not dropped, and the parse below rejects any line that is
  # not a row. `2>/dev/null` let a status-0 warning sit beside a table that
  # might be partial, and absence from a partial table is not absence.
  out="$(ps -W 2>&1)" || return 2
  [[ -n "$out" ]] || return 2
  # The command is taken from the COMMAND column's own offset in the header
  # rather than from field 8 onward, because `ps -W` prints STIME as "Aug 27"
  # for a process started on an earlier day and as "10:23:45" for one started
  # today -- one field or two. Measured on this host: 214 of 246 rows carried a
  # stray leading day number under the field-index form, so a daemon that
  # outlived midnight stopped matching its own recorded path.
  #
  # Every column index is read from the header too, and a header this parser
  # cannot read is a FAILED MEASUREMENT, not an absent process. It exits 3 from
  # the last block -- after all input is read, so the producer cannot take
  # SIGPIPE.
  out="$(printf '%s\n' "$out" | awk -v mode="$mode" -v key="$key" -v want="$want" '
    NR == 1 {
      col = index($0, "COMMAND")
      for (i = 1; i <= NF; i++) {
        # Exact equality, so PPID and PGID are not mistaken for this column.
        if ($i == "PID" && !pc) pc = i
        if ($i == "WINPID") wp = i
      }
      if (key == "PID") kc = pc; else kc = wp
      next
    }
    NF == 0 { next }
    # Structural, not a word count: an eight-word diagnostic merged in from
    # stderr satisfies `NF < 8` and would be counted as a row. Every data line
    # begins PID PPID PGID WINPID, four numbers, and nothing else on this host
    # does. Measured: 277 data rows, 277 with four leading numbers, 0 without;
    # field counts run 8 to 15. Localisation cannot move it -- no column here
    # is a word.
    !($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ &&
      $4 ~ /^[0-9]+$/) { notarow = 1; next }
    { rows++ }
    # The same two witnesses `process_is_alive` demands of `tasklist`: WINPID 4
    # is the System process, present on every Windows NT kernel and in this
    # table (measured), and fewer than ten rows is not a machine anyone runs a
    # daemon on. Neither proves the table COMPLETE; see the residual below.
    wp && $wp == 4 { system_process = 1 }
    # The two questions, and the only two lines that differ between them. Each
    # collects into the same array so that everything after this point -- the
    # witnesses, the END, the per-row validation in the shell below -- is one
    # piece of code and not two that drift.
    mode == "row" && kc && wp && col && $kc == want && !seen {
      seen = 1; row[1] = $wp " " substr($0, col)
    }
    mode == "scan" && wp && col && tolower(substr($0, col)) ~ want {
      row[++seen] = $wp " " substr($0, col)
    }
    END {
      if (!kc || !wp || !col) exit 3
      if (notarow) exit 3
      if (!system_process) exit 3
      if (rows < 10) exit 3
      if (!seen) exit 1
      for (i = 1; i <= seen; i++) print row[i]
    }
  ')" || rc=$?
  # Three answers out of one program, and the middle one is the only one a
  # caller may act on as an absence: 0 printed a row, 1 read a whole table and
  # found no such row, anything else could not read the table at all.
  case "$rc" in
    0) ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
  # THE RESIDUAL, stated because it is not fixed: a `ps -W` (or `tasklist`)
  # that exits 0 having printed a well-formed table SHORT of the process looked
  # for is not detectable from here. Contamination is caught; silent truncation
  # is not, for any of the three Windows tables.
  out="$(printf '%s' "$out" | tr -d '\r')" || return 2
  while IFS= read -r line; do
    # Trailing padding is not part of the row.
    line="${line%"${line##*[![:space:]]}"}"
    # A row that matched and carries no command is not a measurement -- every
    # process in this table has a WINPID and an image -- and reporting it as an
    # absence would be this file's defect one level in. Applied per row rather
    # than to the block, so one unusable row in a scan is a refusal instead of
    # a candidate list that quietly lost a member.
    [[ "$line" == *" "* ]] || return 2
    printf '%s\n' "$line"
  done <<<"$out"
}

# The first row whose key column holds a given value.
# args: "PID" or "WINPID", and the value wanted in it.
ps_w_row_for() {
  _ps_w_read row "$1" "$2"
}

# Every row whose COMMAND matches an ERE, lowercased before matching.
# `dev-runtime.sh`'s reap scan is the caller: it has no pid to key on -- that is
# the entire reason it exists -- so the image is the identity and every row that
# could be it has to come back.
ps_w_rows_matching() {
  _ps_w_read scan WINPID "$1"
}

# The executable behind a pid.
# stdout: the image path when there is one.
# exit:   0 measured and found, 1 measured and no such process, 2 could not
#         measure.
process_image_path() {
  local out rc
  if (( HOST_IS_WINDOWS == 1 )); then
    # WINPID, because a Windows pid is what every caller of this holds: the one
    # recorded at start, or the one a listener table reported.
    rc=0
    out="$(ps_w_row_for WINPID "$1")" || rc=$?
    case "$rc" in
      0) ;;
      1) return 1 ;;
      *) return 2 ;;
    esac
    # "<WINPID> <COMMAND>", and the pid half is the one that was asked for.
    out="${out#* }"
  else
    command -v ps >/dev/null 2>&1 || return 2
    rc=0
    # stderr is merged in for the same reason `process_is_alive` merges it:
    # with `-o command=` the only thing `ps` writes to stdout is the command,
    # so any text alongside status 1 is the tool reporting a problem rather
    # than an absence.
    out="$(ps -p "$1" -o command= 2>&1)" || rc=$?
    if (( rc == 0 )); then
      # And on the success path the merge has to be undone by a shape check:
      # `ps -p N -o command=` prints exactly one line, so a second line is
      # stderr text that would otherwise be compared as part of the image path.
      # An identity this cannot prove must never be reported as one.
      [[ "$out" != *$'\n'* ]] || return 2
      # "Exactly one line" means one, not "at most one": a pid `ps` listed
      # always has a command, so status 0 with silence is state three, not the
      # `return 1` at the bottom of this function.
      [[ -n "$out" ]] || return 2
    fi
    if (( rc != 0 )); then
      # `ps -p` exits non-zero both when the pid is gone and when ps itself
      # failed. Four witnesses, in order, because none is enough alone:
      #
      #   1. THE STATUS. Only 1 is `ps`'s "no match"; a signal death (128+N) or
      #      a usage error (2) is a failed measurement wearing a nonzero status.
      #   2. `kill -0`. If the kernel still knows the pid, ps's answer was a
      #      failure whatever it said. Kill can only ADD "still there": EPERM
      #      and ESRCH share a status, so it can never confirm absence.
      #   3. SILENCE. `ps` failing with status 1 AND a message, beside a
      #      `kill -0` that failed with EPERM, is two failed measurements
      #      agreeing on a negative neither of them made.
      #   4. WHICH errno. Under Linux `hidepid=invisible` another user's
      #      process is absent from /proc, so `ps -p` is silent with status 1 --
      #      the exact shape of a real absence -- while `kill -0` fails with
      #      EPERM, which is proof the process EXISTS.
      #      `errno_says_no_such_process` reads it in the C locale, and only
      #      ESRCH may mean absence.
      (( rc == 1 )) || return 2
      [[ -z "$out" ]] || return 2
      errno_says_no_such_process "$1" || return 2
      return 1
    fi
  fi
  [[ -n "$out" ]] || return 1
  printf '%s' "$out"
}

# Did lsof just enumerate the listening TCP set without detecting an error?
#
# This is the witness the POSIX branch's negative rests on, and it is written to
# CO-VARY with the claim it ratifies: it is the same selection the targeted
# query makes -- `-iTCP -sTCP:LISTEN` -- minus the port filter, so it exercises
# the same scan over the same descriptors. A witness that asked an easier
# question (is lsof on PATH, can it read this process's own fds) would answer
# yes for exactly the host whose /proc, mount or device made the real scan fail.
#
# lsof's own contract is what makes status 0 mean something: it "returns a one
# (1) if any error was detected", so 0 is the tool stating that it found what it
# was asked for AND hit nothing it wanted to complain about. `-t` puts pids and
# nothing else on stdout, so a line that is not a pid is merged stderr riding
# alongside, and a scan with something to say is not a scan that ratifies an
# absence.
#
# THE STATED COST: a host with NO listening TCP socket at all cannot ratify
# anything, so every port there answers "could not measure" rather than "free".
# That refuses to start a daemon instead of colliding with one.
#
# exit: 0 lsof enumerated at least one listener with no error detected, 1 it
# could not be shown to have enumerated anything.
lsof_enumerated_listeners() {
  local out rc=0 parse_rc=0
  out="$(lsof -w -nP -tiTCP -sTCP:LISTEN 2>&1)" || rc=$?
  (( rc == 0 )) || return 1
  [[ -n "$out" ]] || return 1
  # No early `exit` in the program and no `head` on the pipe: a producer that
  # takes SIGPIPE reports 141 under `pipefail`, which is indistinguishable from
  # a parser failure. Flag, count, decide at END.
  printf '%s\n' "$out" | awk '
    /^[[:space:]]*$/ { next }
    $0 !~ /^[0-9]+$/ { notapid = 1 }
    { n++ }
    END { exit (notapid || n == 0) ? 1 : 0 }
  ' || parse_rc=$?
  (( parse_rc == 0 )) || return 1
}

# Which process is listening on a port?
# stdout: the listener pid when there is one.
# exit:   0 found, 1 no listener, 2 could not measure.
#
# The third state is the whole point: a probe that cannot run must never be
# indistinguishable from "port free".
listener_pid_for_port() {
  local table hit out rc
  # An argument that is not a port is not a question this probe can answer, so
  # it is 2 and never "no row matched". Guarded here rather than in the
  # callers because the smokes, dev-runtime.sh and first-run/port-precheck.sh
  # all pass ports through and only one of the four validates its own.
  #
  # ONE SPELLING, because the lookup uses the raw argument (`awk -v
  # port=":$1\$"`) against a table that prints `:135`. Measured, same port, two
  # spellings, opposite answers:
  #     135         -> rc=0 found, pid 1576
  #     0000000135  -> rc=1 measured free
  # So no spelling that needs normalising is admitted: no leading zero, at most
  # five digits (which also settles what `$(( 10#$1 ))` would do with a
  # 23-digit argument, namely overflow).
  [[ "${1-}" =~ ^[1-9][0-9]{0,4}$ ]] || return 2
  (( $1 <= 65535 )) || return 2
  if (( HOST_IS_WINDOWS == 1 )); then
    # netstat is the only listener table Windows offers here. Anchoring the
    # port keeps :17895 from matching :178950 or a foreign address.
    command -v netstat >/dev/null 2>&1 || return 2
    table="$(netstat -ano 2>&1)" || return 2
    # netstat always prints a header; silence means it did not run properly.
    [[ -n "$table" ]] || return 2
    # Non-empty is not the same as parseable. A diagnostic, a localised banner,
    # or a netstat whose columns we do not know exits 0 with plenty of text,
    # the awk below matches nothing, and "no row for this port" becomes "no
    # listener" -- a failed parse reported as a measured negative. So require
    # evidence that this really IS the table. Flag and print at END, never
    # `exit` on first match, or a producer taking SIGPIPE reports 141 and looks
    # like a parser failure.
    #
    # NOTHING HERE KEYS ON A WORD. netstat's State column is LOCALISED (German
    # Windows prints ABHOEREN) while `TCP` is not, so `LISTENING` is not a key.
    # The listening SHAPE is: a TCP row with five fields, a numeric pid last,
    # and a WILDCARD foreign address -- which only a listening socket has,
    # because a connected one names its peer. Measured against this host's real
    # table: 34 rows have that shape, all 34 are the LISTENING rows, and no
    # LISTENING row lacks it. The shape must exist SOMEWHERE in the table, so a
    # table whose columns have moved is unmeasurable rather than empty.
    #
    # stderr is merged in above rather than dropped, and EVERY non-empty line
    # must be accounted for: a status-0 `WARNING: partial results` merged beside
    # real rows begins with neither `TCP` nor `UDP`, so a rule keyed on those
    # tokens never looks at it and the branch returns MEASURED FREE off a table
    # netstat itself complained about.
    #
    # Four rules, all measured against this host's real `netstat -ano` (183
    # lines: 2 blank, 179 protocol rows, exactly 2 non-row lines):
    #
    #   * AFTER the first protocol row, every non-empty line must be a row.
    #     The real table has zero exceptions.
    #   * BEFORE it, at most 2 non-blank lines. The real preamble is exactly
    #     two -- the `Active Connections` banner and the column header -- and
    #     both are LOCALISED, so they can only be counted, not matched. One
    #     merged diagnostic makes it three.
    #   * At least one UDP row. `netstat -ano` prints the whole TCP table then
    #     the whole UDP table (measured: 198 lines, TCP rows 5..97, UDP rows
    #     98..198, no interleaving, no UDP header), so a UDP row WITNESSES THAT
    #     THE TCP SECTION ENDED -- the stream got past every TCP row there was.
    #     This function only asks about TCP listeners, so a table with a UDP row
    #     cannot be missing the TCP row being asked about.
    #   * Every TCP row precedes every UDP row. "A UDP row appeared" licenses
    #     nothing on its own; the ordering is what makes it proof the TCP
    #     section is complete, and a one-host observation of it is not a proof
    #     for every Windows version and locale. `tcp_after_udp` exits 4.
    #
    # THE COSTS, stated: a netstat whose banner runs to three lines, a host with
    # no listening TCP socket, and a host with no UDP endpoint each answer
    # "could not measure" for every port rather than "free". All three refuse to
    # start a daemon rather than collide with one, and on Windows none is
    # reachable in practice -- the RPC endpoint mapper and svchost listen, and
    # the DNS and DHCP clients hold UDP sockets (101 UDP rows here), before a
    # login shell exists.
    #
    # THE RESIDUAL: this detects a table that stopped EARLY -- the shape a
    # closed pipe, a killed writer or a truncated buffer produces. It cannot
    # detect a hole in the MIDDLE of the TCP section, because the UDP rows after
    # it still arrive. It also cannot detect a SHORT TCP section: one
    # listening-shaped TCP row plus one UDP row satisfies every rule above.
    # Unlike `tasklist` and `ps -W` this parser has NO row floor, because
    # netstat's fixtures in scripts/host-process.test.ts stand for whole tables
    # with two rows.
    #
    # The four exits are distinct for the reader; the caller turns all of them
    # into 2.
    printf '%s\n' "$table" | awk '
      /^[[:space:]]*$/ { next }
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
      { if (rows) after = 1; else preamble++ }
      END {
        if (after || preamble > 2) exit 2
        if (tcp_after_udp) exit 4
        if (!udp) exit 3
        exit(found ? 0 : 1)
      }
    ' || return 2
    hit="$(printf '%s\n' "$table" | awk -v port=":$1\$" '
      $1 == "TCP" && $2 ~ port && NF == 5 && $5 ~ /^[0-9]+$/ &&
        ($3 == "0.0.0.0:0" || $3 == "[::]:0" || $3 == "*:*") && !seen {
          hit = $5; seen = 1
        }
      END { if (seen) print hit }
    ')" || return 2
    hit="$(printf '%s' "$hit" | tr -d '\r')" || return 2
  else
    command -v lsof >/dev/null 2>&1 || return 2
    # lsof exits 1 for "nothing matched" AND for "something went wrong" -- its
    # manual says it returns 1 "if any error was detected" -- so the status
    # alone cannot tell a negative measurement from a failed one, and ruling
    # out its absence one line up only covers 127.
    #
    # With -t the only thing lsof writes to stdout is pids, so stderr merged in
    # is the discriminator: silence with status 1 means nothing matched, any
    # text means the probe had a problem. EXCEPT that `-w` is here to SUPPRESS
    # that text -- it is needed because the benign "can't stat()" warnings on a
    # host with a gvfs or docker mount would otherwise answer "could not
    # measure" on every call, and it silences the loud warnings with the same
    # hand. So `rc == 1` with both streams silent is also the shape of an lsof
    # that hit an unreadable /proc, mount or device.
    #
    # The flag stays and the NEGATIVE is ratified instead: silence with status 1
    # may mean "free" only once `lsof_enumerated_listeners` has shown lsof can
    # still enumerate the listening TCP set at all. Order matters for cost --
    # the targeted query runs first, so "found" and "hard failure" still take
    # one call and only the negative pays for its ratification.
    rc=0
    out="$(lsof -w -nP -tiTCP:"$1" -sTCP:LISTEN 2>&1)" || rc=$?
    if (( rc == 1 )) && [[ -z "$out" ]]; then
      lsof_enumerated_listeners || return 2
      return 1
    fi
    (( rc == 0 )) || return 2
    hit="$(printf '%s' "$out" | sed -n '1p')" || return 2
    # Status 0 with nothing on stdout is not "port free": a -t query that
    # matched exits 0 and prints a pid, and one that did not exits 1. Silence
    # here means this is not the lsof we think it is.
    [[ -n "$hit" ]] || return 2
  fi
  [[ -n "$hit" ]] || return 1
  # A non-numeric answer means the table was not the table we think it is.
  [[ "$hit" =~ ^[0-9]+$ ]] || return 2
  printf '%s' "$hit"
}

# --- the delay inside a polling window ---------------------------------------

# One slice of a bounded poll.
# exit: 0 the delay was performed, 2 it could not be.
#
# Every polling loop here is called from the left of a `||` or as an `if`/
# `while` condition, which disables errexit through the whole body -- so a bare
# `sleep` whose status nothing reads does not stop anything. All hundred rounds
# run back to back in microseconds and the function returns its terminal
# NEGATIVE for a window that never elapsed. A ten-second wait that took ten
# milliseconds is not a wait, and its negative is not a measurement.
#
# So the delay answers, and every caller turns a 2 into its own "could not
# measure". `sleep 0.1` is not universal -- POSIX requires only whole seconds --
# so a rejected fractional argument falls back to one second rather than to no
# delay: a poll slower than intended still measures its window. Only a `sleep`
# that cannot delay AT ALL is unmeasurable.
poll_delay() {
  sleep "$1" 2>/dev/null && return 0
  sleep 1 2>/dev/null && return 0
  return 2
}

# The Windows pid of a job just backgrounded from this shell.
#
# `nohup env ... server &` is a chain of MSYS processes that ends in an exec of
# a native binary, and MSYS spawns a fresh Windows process for that last step.
# Until it does, the MSYS pid still maps to the WINPID of `env`, which then
# exits. Taking the first WINPID that appears therefore records a pid that is
# dead moments later, so this waits for the row to name the program that was
# actually launched.
#
# stdout: the Windows pid. exit: 0 found, 1 never appeared within the window,
# 2 could not measure (no `ps`, a failed snapshot, a table that is not one, or
# an unspellable path).
windows_pid_for_job() {
  local job_pid="$1" program="$2" want row winpid command got rc
  local _attempt
  want="$(normalize_program_path "$program")" || return 2
  for (( _attempt = 0; _attempt < 100; _attempt++ )); do
    # One snapshot yields both fields, so the pid and the image it names cannot
    # come from either side of an exec.
    #
    # The parse is `ps_w_row_for`, the same validated one `process_image_path`
    # uses. The only difference between the two questions is the column keyed
    # on, so that is the only thing passed in.
    rc=0
    row="$(ps_w_row_for PID "$job_pid")" || rc=$?
    # 2 is FINAL. A table that could not be read will not become readable by
    # asking again, and ten more seconds of asking would end at the `return 1`
    # below -- the state this helper must never confuse with it.
    (( rc == 2 )) && return 2
    # 1 is the MEASURED negative and it is not final: the exec has not landed
    # yet, which is the entire reason this polls. Only a run of measured
    # negatives that exhausts the window may be reported as "never appeared".
    if (( rc == 0 )); then
      winpid="${row%% *}"
      command="${row#* }"
      if [[ "$winpid" =~ ^[0-9]+$ ]] && (( winpid > 0 )); then
        if ! got="$(normalize_program_path "$command")"; then
          return 2
        fi
        if [[ "$got" == "$want" ]]; then
          printf '%s\n' "$winpid"
          return 0
        fi
      fi
    fi
    # 2, not "keep going" and not "the window closed": see `poll_delay`. The
    # `return 1` below is a claim about ten seconds of readable tables, and a
    # delay that did not happen means those ten seconds did not either.
    poll_delay 0.1 || return 2
  done
  return 1
}

# --- stopping a process ------------------------------------------------------

# Kill a Windows process only if it is still the recorded executable.
#
# `taskkill` can filter on a pid and on an image *name*, and a production
# install and every other worktree's dev daemon are all called
# wenlan-server.exe, so that filter cannot tell them apart from this one.
# Checking the full path in a separate call would leave the reuse window open:
# the daemon can exit and Windows can hand its pid to one of those neighbours
# before the kill lands.
#
# Opening the process first closes that window. Windows keeps a pid reserved for
# as long as any handle to it is open, so once `$p.Handle` materializes, the
# path this reads and the process this kills are the same one. `MainModule` and
# the kill both address it by pid, which is now pinned. A path that cannot be
# read is not a match, so an identity this cannot prove kills nothing.
#
# The outcome is reported, never swallowed by `|| true`. The caller still takes
# its verdict from the liveness poll that follows -- a stop is proven by the
# process being gone, never by this receipt -- but a refusal or a thrown kill
# must stay distinguishable from a success in the diagnostics.
#
# exit: 0 killed, 3 already gone, 4 refused (not our image), 5 kill threw,
#       6 identity unreadable, 2 the helper itself could not run.
kill_by_image_path() {
  command -v powershell.exe >/dev/null 2>&1 || return 2
  MSYS2_ARG_CONV_EXCL='*' WENLAN_KILL_PID="$1" WENLAN_KILL_PATH="$2" \
    powershell.exe -NoProfile -NonInteractive -Command '
      $want = ($env:WENLAN_KILL_PATH -replace "/", "\")
      try { $p = [System.Diagnostics.Process]::GetProcessById([int]$env:WENLAN_KILL_PID) }
      catch { exit 3 }
      try { $null = $p.Handle } catch { exit 6 }
      try { $got = $p.MainModule.FileName } catch { exit 6 }
      if ($got -ine $want) { exit 4 }
      try { $p.Kill() } catch { exit 5 }
      exit 0
    ' >/dev/null 2>&1
}

# Windows cannot deliver SIGTERM to a console process that owns no console, so
# there is no graceful stop to try first; a forced kill is the only stop that
# works. The daemon's durability is SQLite WAL, which is crash-safe, and the
# POSIX path already force-kills anything that ignores SIGTERM for 5s.
#
# Both take the recorded server path as a second argument. Neither kills a
# process tree: `kill -KILL` does not on POSIX, so the Windows side does not
# either, and a tree kill is exactly what should not follow an identity check
# that only covers the root.
#
# Both return the stop helper's status. Callers must capture it (`|| rc=$?`):
# under `set -e` an unguarded call would now abort the script, and on POSIX
# `kill` legitimately fails when the process has already exited.
terminate_process() {
  if (( HOST_IS_WINDOWS == 1 )); then
    kill_by_image_path "$1" "$2"
  else
    kill "$1"
  fi
}

force_terminate_process() {
  if (( HOST_IS_WINDOWS == 1 )); then
    kill_by_image_path "$1" "$2"
  else
    kill -KILL "$1"
  fi
}

# --- state-setting wrappers --------------------------------------------------
#
# `set -e` makes `out="$(f)"; rc=$?` abort at the assignment, so these set named
# globals instead and every caller branches on all three values. Nothing here
# may be called inside a command substitution: a subshell's globals never reach
# the caller.

LISTENER_PROBE_STATE=""
LISTENER_PROBE_PID=""
# Sets LISTENER_PROBE_STATE to found | none | unmeasured.
#
# `out="$(f)" || rc=$?` and not `if out="$(f)"; then ... fi; rc=$?`: after an
# `if` whose condition is false with no else, `$?` is the compound's own status,
# which is 0, and every negative measurement becomes an unmeasured one.
probe_listener_port() {
  local out="" rc=0
  LISTENER_PROBE_PID=""
  out="$(listener_pid_for_port "$1")" || rc=$?
  case "$rc" in
    0) LISTENER_PROBE_PID="$out"; LISTENER_PROBE_STATE=found ;;
    1) LISTENER_PROBE_STATE=none ;;
    *) LISTENER_PROBE_STATE=unmeasured ;;
  esac
}

PROCESS_ALIVE_STATE=""
# Sets PROCESS_ALIVE_STATE to alive | gone | unmeasured.
probe_process_alive() {
  local rc=0
  process_is_alive "$1" || rc=$?
  case "$rc" in
    0) PROCESS_ALIVE_STATE=alive ;;
    1) PROCESS_ALIVE_STATE=gone ;;
    *) PROCESS_ALIVE_STATE=unmeasured ;;
  esac
}

PROCESS_IMAGE_STATE=""
PROCESS_IMAGE_VALUE=""
# Sets PROCESS_IMAGE_STATE to found | none | unmeasured.
probe_process_image() {
  local out="" rc=0
  PROCESS_IMAGE_VALUE=""
  out="$(process_image_path "$1")" || rc=$?
  case "$rc" in
    0) PROCESS_IMAGE_VALUE="$out"; PROCESS_IMAGE_STATE=found ;;
    1) PROCESS_IMAGE_STATE=none ;;
    *) PROCESS_IMAGE_STATE=unmeasured ;;
  esac
}
