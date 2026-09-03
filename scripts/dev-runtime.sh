#!/usr/bin/env bash
#
# THE OUTCOME CONTRACT
#
# The LAST line this script writes to stderr is always
#
#     DEV_RUNTIME_RESULT: <kind>
#
# and <kind> is exactly one of:
#
#     ok              the command did what it was asked to do
#     safety-refusal  a guard refused -- a production port, identity, path or
#                     socket, or a process this runtime is not allowed to kill
#     build-failure   the daemon could not be BUILT
#     staging-failure the daemon built and could not be put beside its runtime
#                     libraries, or what was put there could not be verified
#     health-failure  a daemon was started and never became healthy on its port
#     port-conflict   the isolated dev port is held by another process
#     interrupted     a signal ended the run; nothing here failed
#     unknown         anything else, and every "could not measure"
#
# `build-failure` and `staging-failure` used to be one kind, and the remediation
# they call for is opposite. A build failure is a COMPILER answer: fix the
# source, retry the build. A staging failure is almost always Windows error 32
# -- a daemon running out of the stage directory holding onnxruntime.dll open --
# where retrying the build changes nothing and the fix is to stop that process.
# A caller that retries a `build-failure` therefore loops forever on a held
# stage, and the condition that is actually blocking it is not in the marker at
# all.
#
# `interrupted` is separate from `unknown` for the mirror-image reason. `unknown`
# is a REFUSAL a consumer must not act through; `interrupted` says the run was
# stopped from outside and measured nothing wrong, which is the one failing kind
# a supervisor may re-run unchanged. Before it existed a signal reported `$?`
# from whatever ran last, so an interrupted run could exit ZERO and print `ok`.
#
# The marker is ADDITIVE. Every human line above it is unchanged, and a consumer
# that ignores it sees exactly what it saw before. It exists because that prose
# was the ONLY thing distinguishing a safety event from an ordinary failure --
# every top-level guard exits 2 and everything `start_runtime` returns exits 1 --
# so a downstream harness had to classify by string-matching the messages.
# Rewording one, combining two, or translating either would silently reclassify
# a SAFETY REFUSAL as a build failure.
#
# `unknown` is a REFUSAL and not a pass. It is what this prints when it cannot
# tell which kind applies, and a consumer must treat it the way it treats
# `safety-refusal`. That is this file's one rule -- a failed measurement is never
# a negative measurement -- carried out to the process boundary: guessing a kind
# would be exactly the collapse every probe below exists to prevent.
#
# It is printed from an EXIT trap, so every way out carries one: the file-scope
# guards that `exit 2` before `start` ever reaches its body, a `set -e` abort in
# a command whose status nothing captured, a signal, and every `return` in
# `start_runtime` and `stop_runtime`.
set -euo pipefail

# --- the outcome marker (see THE OUTCOME CONTRACT at the top) -----------------
#
# FIRST, and before this script has located itself or sourced anything. What
# stood between here and the marker was `SCRIPT_DIR`, `REPO_ROOT` and the
# library `source` -- three ways out that carried no marker at all, because
# under `set -e` a failing command substitution or an unreadable library aborts
# on the spot. A consumer reading the last stderr line would then see the
# script's own diagnostic and have to classify it by string, which is the whole
# defect this marker exists to remove. Nothing here depends on the library:
# `release_runtime_lock` is the only thing that does, and it is reached only
# through `RUNTIME_LOCK_HELD`, which nothing can have set yet.
#
# `unknown` until something measures otherwise, because the default has to be
# the answer that means "do not act on this".
RESULT_KIND=unknown
RESULT_EMITTED=0
RUNTIME_LOCK_HELD=0
# ROUND 6. A LOCK THAT VANISHED UNDER ITS HOLDER IS NOT A LOCK THAT WAS
# RELEASED. Set when `release_runtime_lock` finds that this run's exclusive
# claim was broken while the run still held it. It is the same state
# `attest.sh` carries as `LOCK_STOLEN`, and it is a global for the same reason
# `RUNTIME_LOCK_HELD` is: the outcome handler has to be able to read it, and it
# must have a value before this file has finished being read, because a
# file-scope guard can `exit 2` on the line after this one.
RUNTIME_LOCK_STOLEN=0
# THE OWNER RECORD NAMES AN ACQUISITION, NOT A PROCESS. A bare pid can repeat
# inside one stale window, so "the owner I measured is still the owner" was a
# question a DIFFERENT run could satisfy; the nonce makes two generations of the
# lock directory impossible to confuse. It is `attest.sh`'s `LOCK_TOKEN`, and
# the two files must not disagree about what owns a lock. Declared here for the
# same reason `RUNTIME_LOCK_STOLEN` is: the release reads it, and a file-scope
# guard can `exit 2` before this file has finished being read.
RUNTIME_LOCK_TOKEN=""
RUNTIME_LOCK_GEN=0
# The outcome handler runs at most once. On a signal it used to run TWICE: the
# signal trap called it, it released the lock, printed the marker and then
# `exit`ed -- which fires the EXIT trap, which calls it again. `RESULT_EMITTED`
# stopped the second MARKER, so the duplication was invisible in the contract
# and not in the behaviour: `RUNTIME_LOCK_HELD` was never cleared, so the second
# run released the lock a second time, AFTER the marker, and any `rm`/`rmdir`
# diagnostic from it landed below the line this file promises is last.
RUNTIME_EXIT_RAN=0

emit_result() {
  # Once. A second marker would let a consumer that reads the last line and a
  # consumer that reads the first line disagree about the same run.
  (( RESULT_EMITTED == 1 )) && return 0
  RESULT_EMITTED=1
  printf 'DEV_RUNTIME_RESULT: %s\n' "$1" >&2
}

# The single exit path. The lock release used to be its own EXIT trap installed
# by `acquire_runtime_lock`; two traps on EXIT is one trap, the second wins, so
# they are one function now. `RUNTIME_LOCK_HELD` is what lets this run before
# `release_runtime_lock` is even defined -- a guard below can `exit 2` while
# this file is still being read.
on_runtime_exit() {
  local status=$?
  local signal="${1:-}"
  # `$?` INSIDE A SIGNAL TRAP IS THE PREVIOUS COMMAND'S STATUS, not `128+n`.
  # One trap on `EXIT HUP INT TERM` therefore reported, for an interrupted run,
  # whatever the last command before the signal happened to return -- so a
  # Ctrl-C delivered after a successful `curl` exited ZERO, and the marker was
  # whatever `RESULT_KIND` had reached, which on the health path can already be
  # `ok`. An interrupted run reporting success is the same defect this whole
  # file is about, at the process boundary: nothing measured that outcome.
  #
  # So the signal names itself, the status is derived from it, and the kind is
  # its own. `interrupted` is not `unknown`: `unknown` says "this could not be
  # measured, refuse"; `interrupted` says "nothing is wrong with this runtime,
  # somebody stopped it", which is the one failing kind a supervisor may retry
  # unchanged rather than refuse on.
  case "$signal" in
    HUP) status=129; RESULT_KIND=interrupted ;;
    INT) status=130; RESULT_KIND=interrupted ;;
    TERM) status=143; RESULT_KIND=interrupted ;;
  esac
  # Idempotent, because `exit` below re-enters this through the EXIT trap.
  if (( RUNTIME_EXIT_RAN == 1 )); then
    exit "$status"
  fi
  RUNTIME_EXIT_RAN=1
  if (( RUNTIME_LOCK_HELD == 1 )); then
    # NOT `|| true`, which is what stood here. The reasoning for it was right
    # as far as it went -- errexit is not suspended inside a trap, so a bare
    # non-zero release would abort the handler BEFORE the marker was printed,
    # and a missing marker is the one thing this contract cannot express -- but
    # it discarded the status instead of reading it. An `if !` condition
    # suspends errexit for the same reason `|| true` did AND hands over the
    # answer, so the handler is still guaranteed to reach `emit_result` below.
    if ! release_runtime_lock; then
      # TWO different things go wrong here and they call for opposite words.
      # A lock still on disk is one this run failed to remove; a lock that is
      # GONE, or that now names somebody else, is one this run failed to KEEP.
      # Saying "did not come off" about the second is false in the direction
      # that matters -- it sends a reader to look for a leftover directory
      # while the actual event was that another command was inside this one's
      # isolated port and data directory.
      if (( RUNTIME_LOCK_STOLEN == 1 )); then
        echo "error: this command finished, but the dev runtime lock it held was" >&2
        echo "       broken while it still held it, so the isolation the work" >&2
        echo "       above assumed was not in force for some part of it" >&2
        echo "       another command may have been running against this worktree's" >&2
        echo "       port and data directory at the same time" >&2
      else
        echo "error: this command finished but its dev runtime lock did not come" >&2
        echo "       off; the next command against this worktree will refuse" >&2
      fi
      # Only from `ok`. A run that already measured a specific failure keeps
      # that kind: `unknown` is a refusal, and downgrading `health-failure` to
      # it would throw away the diagnosis a consumer acts on. What must not
      # survive is `ok` -- success is the one claim a lingering lock refutes,
      # and a 0 exit status beside it is the same claim in the other channel.
      # A STOLEN lock refutes something different and reaches the same place: it
      # does not say the work failed, it says the work was not isolated while it
      # ran, and `ok` is a claim about a run whose premises held.
      if [[ "$RESULT_KIND" == ok ]]; then
        RESULT_KIND=unknown
        # Spelled out rather than `(( status == 0 )) && status=1`. That form
        # returns 1 when the test is false, and it sits in a trap handler where
        # errexit is NOT suspended; it survives only on the `&&`-list exemption.
        # A guard whose correctness rests on a set -e exemption is a guard the
        # next edit breaks silently.
        if (( status == 0 )); then
          status=1
        fi
      fi
    fi
  fi
  emit_result "$RESULT_KIND"
  exit "$status"
}
# Four traps and not one, because the handler has to know WHICH signal arrived:
# `$?` cannot tell it, and `128+n` is the only honest exit status for a run that
# a signal ended. The EXIT line is first so that a guard which `exit 2`s while
# this file is still being read is already covered.
trap on_runtime_exit EXIT
trap 'on_runtime_exit HUP' HUP
trap 'on_runtime_exit INT' INT
trap 'on_runtime_exit TERM' TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Host process primitives: HOST_IS_WINDOWS, path spelling, and the tri-state
# measurements (`listener_pid_for_port`, `process_is_alive`,
# `process_image_path` and their `probe_*` wrappers). Shared with the surface
# smokes so that one hardening applies to both. Every probe answers measured /
# negative / could not measure, and every caller below branches on all three.
# shellcheck source=scripts/lib/host-process.sh
. "$SCRIPT_DIR/lib/host-process.sh"

# The worktree identity every isolated port and path is derived from. Read as a
# measurement rather than assumed: an `awk` that is not there, or a `cksum` that
# fails, leaves this empty, and an empty WORKTREE_ID makes every arithmetic
# expansion below fail one line at a time with nothing naming the cause.
WORKTREE_ID="$(printf '%s' "$REPO_ROOT" | cksum | awk '{ print $1 }')" || WORKTREE_ID=""
if [[ ! "$WORKTREE_ID" =~ ^[0-9]+$ ]]; then
  echo "error: could not compute this worktree's identity from $REPO_ROOT" >&2
  echo "       cksum/awk answered [$WORKTREE_ID]; every isolated port and path" >&2
  echo "       below is derived from it, so nothing here can be isolated" >&2
  RESULT_KIND=unknown
  exit 2
fi
TMP_BASE="${TMPDIR:-/tmp/}"
STATE_DIR="${WENLAN_DEV_STATE_DIR:-${TMP_BASE%/}/wenlan-app-dev/$WORKTREE_ID}"
DEV_PORT="${WENLAN_DEV_PORT:-$((17000 + WORKTREE_ID % 1000))}"
DEV_UI_PORT="${WENLAN_DEV_UI_PORT:-$((18000 + WORKTREE_ID % 1000))}"
DEV_REMOTE_PORT_START="${WENLAN_DEV_REMOTE_PORT_START:-$((20000 + (WORKTREE_ID % 1000) * 4))}"
DEV_APP_ID="${WENLAN_DEV_APP_ID:-com.wenlan.desktop.dev.$WORKTREE_ID}"
DEV_DATA_DIR="${WENLAN_DEV_DATA_DIR:-$STATE_DIR/data}"
DEV_TAURI_MCP_SOCKET="${WENLAN_DEV_TAURI_MCP_SOCKET:-$STATE_DIR/tauri-mcp.sock}"
PID_FILE="$STATE_DIR/wenlan-server.pid"
SERVER_PATH_FILE="$STATE_DIR/wenlan-server.path"
PORT_FILE="$STATE_DIR/wenlan-server.port"
DATA_DIR_FILE="$STATE_DIR/wenlan-server.data-dir"
SERVER_LOG="$STATE_DIR/wenlan-server.log"
LOCK_DIR="$STATE_DIR/runtime.lock"
LOCK_OWNER_FILE="$LOCK_DIR/pid"
STARTED_RUNTIME=0
DAEMON_STAGE_DIR="$STATE_DIR/daemon"

# Stop a daemon that was launched but never resolved to a pid.
#
# `windows_pid_for_job` gives up after ten seconds, and a daemon that is merely
# slow is still alive when it does. Nothing recorded it, and `dev-all.sh` only
# registers the runtime once `start` succeeds, so no later `stop` will find it
# either -- it goes on to bind the dev port and stays there unmanaged.
#
# Looking for the port is not enough: the process may not have bound it yet at
# the moment the snapshot is taken. The staging directory is private to this
# worktree, so the executable itself is the identity. Anything running that
# exact image is this runtime's own daemon, bound or not, and this keeps looking
# for as long as the health probe downstream would have waited. The `ps` scan is
# prefiltered by name so only the handful of candidate rows are canonicalized.
#
# exit: 0 reaped, 1 nothing matching was reaped within the window, 2 the
# question could not be measured. A reap receipt is only ever printed for 0 --
# an unmeasurable liveness result used to `return 0` from the poll below and
# claim the daemon was reaped when nothing knew whether it was.
reap_staged_daemon() {
  local server="$1" want winpid image got rows scan_rc stop_rc
  local _attempt _wait
  want="$(normalize_program_path "$server")" || return 2
  for (( _attempt = 0; _attempt < 50; _attempt++ )); do
    # THE PARSE IS NOT HERE. It was, and that was the finding: a third
    # hand-maintained copy of the `ps -W` table walk, with the same four
    # witnesses as `lib/host-process.sh` and its own place to be edited. Copy
    # two is exactly how the second one came to still be counting words three
    # roundings after the first stopped. This one scans ALL rows instead of
    # looking one row up by key, which is why it did not drop into
    # `ps_w_row_for` unchanged -- so the LIBRARY grew the second entry point and
    # this call site shrank to one line. `scripts/host-process.test.ts` now pins
    # the number of places that run `ps -W` across this file as well as the
    # library, because "exactly one parse" is a claim about the repository.
    #
    # Three answers, and the middle one is the reason this polls: 1 is a whole
    # table with no candidate row in it, which is the ordinary state while the
    # MSYS chain is still exec'ing, so it keeps looking. 2 is a table nothing
    # could read, and no number of retries turns that into knowledge.
    scan_rc=0
    rows="$(ps_w_rows_matching wenlan-server)" || scan_rc=$?
    (( scan_rc == 2 )) && return 2
    if (( scan_rc == 0 )); then
      while read -r winpid image; do
        [[ "$winpid" =~ ^[0-9]+$ ]] || continue
        got="$(normalize_program_path "$image")" || return 2
        [[ "$got" == "$want" ]] || continue
        stop_rc=0
        force_terminate_process "$winpid" "$server" || stop_rc=$?
        for (( _wait = 0; _wait < 50; _wait++ )); do
          probe_process_alive "$winpid"
          case "$PROCESS_ALIVE_STATE" in
            gone) return 0 ;;
            unmeasured)
              echo "error: liveness of staged daemon PID $winpid could not be measured" >&2
              echo "       not claiming it was reaped (stop helper status $stop_rc)" >&2
              return 2
              ;;
          esac
          # Both delays in this function answer, for the reason `poll_delay`
          # gives: "the daemon survived the stop" and "no row ever appeared"
          # are claims about elapsed time, and a delay that did not happen
          # makes them claims about nothing.
          if ! poll_delay 0.1; then
            echo "error: the wait after stopping staged daemon PID $winpid could not" >&2
            echo "       be performed, so it was not waited for" >&2
            return 2
          fi
        done
        echo "error: staged daemon PID $winpid survived the stop (helper status $stop_rc)" >&2
        return 1
      done <<< "$rows"
    fi
    poll_delay 0.2 || return 2
  done
  return 1
}

# The sha256 of a file's contents.
# stdout: the 64-hex digest. exit: 0 measured, 2 could not measure -- no hasher
# on PATH, the hasher failed, the file is not there or will not be read, or what
# came back is not a digest.
#
# TWO answers and not three, on purpose. The missing-file case needs no separate
# state because the only caller treats every unmeasured digest as "not the same
# bytes", which stages again -- the safe direction -- and never as "the same",
# which would keep whatever is already on disk. A digest is only ever allowed to
# conclude that two files MATCH, and that needs both sides measured.
file_sha256() {
  local out
  # Git Bash and Linux ship `sha256sum`; macOS ships `shasum` and not the other,
  # and this file's tests run on the macOS app-check lane.
  if command -v sha256sum >/dev/null 2>&1; then
    out="$(sha256sum -- "$1" 2>/dev/null)" || return 2
  elif command -v shasum >/dev/null 2>&1; then
    out="$(shasum -a 256 -- "$1" 2>/dev/null)" || return 2
  else
    return 2
  fi
  out="${out%% *}"
  # Status 0 with something that is not a digest means this is not the tool we
  # think it is. Comparing it would compare two things neither of which is a
  # hash, and two empty strings are equal.
  [[ "$out" =~ ^[0-9a-f]{64}$ ]] || return 2
  printf '%s' "$out"
}

# Put `source` at `dest`, and prove it is there by CONTENT.
#
# `cp -u` compares TIMESTAMPS. Provenance is a claim about BYTES, and the two
# come apart in ordinary ways: a staged executable restored from a backup, a
# clock that went backwards, an artifact copied rather than built. Any of them
# gives the stale file the newer mtime, `cp -u` keeps it, cargo reports success,
# and `/api/health` is then answered by a daemon built from other source while
# the run reports current-source provenance. Nothing downstream can notice --
# the recorded path, the pid and the port are all correct.
#
# So the digest decides, and it decides twice. Before: the copy is skipped only
# when BOTH sides measured and matched, so an unmeasurable digest stages again.
# After: the staged file is re-hashed -- what is on disk, not what `cp` said
# about it -- and anything but the source's own digest is a hard error. A copy
# that reports success and leaves different bytes (a partial write, a
# concurrent writer, a filesystem that lied) is caught here and nowhere else.
#
# exit: 0 staged and verified, 1 it could not be staged or could not be proven.
# Never 0 on an unmeasured half: a staging that cannot be proven is a staging
# that did not happen, because the whole point of it is the claim.
#
# On success it also publishes the verified digest in STAGED_FILE_DIGEST, which
# is what lets the caller reason about the staged SET rather than about one file
# at a time.
STAGED_FILE_DIGEST=""
stage_file_by_identity() {
  local source="$1" dest="$2" label="$3" want got rc
  rc=0
  want="$(file_sha256 "$source")" || rc=$?
  if (( rc != 0 )); then
    echo "error: could not hash the built $label: $source" >&2
    echo "       refusing to stage by timestamp instead: an unverified copy is" >&2
    echo "       how a stale daemon comes to answer /api/health as this build" >&2
    return 1
  fi
  rc=0
  got="$(file_sha256 "$dest")" || rc=$?
  (( rc == 0 )) || got=""
  if [[ "$got" != "$want" ]]; then
    if ! cp -f -- "$source" "$dest"; then
      echo "error: could not stage $label into $DAEMON_STAGE_DIR" >&2
      echo "       $source -> $dest" >&2
      echo "       a daemon may still be running from it; try dev-runtime.sh stop" >&2
      return 1
    fi
  fi
  rc=0
  got="$(file_sha256 "$dest")" || rc=$?
  if (( rc != 0 )); then
    echo "error: the staged $label could not be hashed after the copy: $dest" >&2
    echo "       a staging that cannot be verified has not been staged" >&2
    return 1
  fi
  if [[ "$got" != "$want" ]]; then
    echo "error: the staged $label is not the one that was just built" >&2
    echo "       built:  $want ($source)" >&2
    echo "       staged: $got ($dest)" >&2
    return 1
  fi
  # Published for the caller, which needs it to answer a question this function
  # cannot see: two source directories can hold the same DLL NAME with different
  # bytes, and each copy verifies perfectly on its own while the second silently
  # replaces the first. A per-file check cannot notice that; a set can.
  STAGED_FILE_DIGEST="$want"
}

# Tauri's build script copies the staged sidecars and their runtime DLLs into
# the cargo target directory. A daemon running out of that directory holds
# onnxruntime.dll open, and Windows fails the copy with os error 32, so
# `pnpm dev:all` would deadlock against its own daemon. Running from a private
# copy under the dev state directory keeps the two out of each other's way, and
# incidentally makes the recorded server path unique to this worktree.
#
# THIS FUNCTION IS CALLED ON THE LEFT OF `||`, so errexit is DISABLED
# throughout its body and its return value is the status of its LAST command,
# not of the command that failed. `mkdir -p` used to sit here bare on that
# basis; a `set -e` that is inert cannot propagate it, and everything after it
# would have run against a stage directory that does not exist. Every status
# below is therefore read explicitly.
stage_windows_daemon() {
  local source="$1" source_dir dir listing name want rc
  local staged=0 prepared=0 staged_names=" "
  local line previous previous_rc
  local -a missing=()
  local -a stray=()
  # name -> the digest that was verified into the stage for it. Two source
  # directories are walked, and the same DLL name can be in both.
  #
  # A NEWLINE-DELIMITED "name<TAB>digest" STRING AND NOT AN ASSOCIATIVE ARRAY.
  # `local -A` is bash 4; macOS ships 3.2, where it is a `local: -A: invalid
  # option` and every `staged_digest[$name]` read after it is parsed as an
  # ARITHMETIC index, so `onnxruntime.dll` is a syntax error on the dot. The
  # lookup below compares the WHOLE first field, so no name can be found inside
  # another, and a miss stays distinguishable from a recorded empty digest.
  local staged_digest=""
  # The runtime libraries the daemon resolves from its own directory.
  #
  # Not a guess and not a glob's opinion. `scripts/prepare-sidecars.sh` refuses
  # to finish for a Windows triple unless exactly these are in `app/binaries`,
  # and `app/tauri.windows.conf.json` bundles exactly these out of it. They are
  # needed because the daemon loads ONNX Runtime DYNAMICALLY (fastembed's
  # `ort-load-dynamic`, selected in `crates/wenlan-core/Cargo.toml`) and links
  # the Vulkan loader for llama.cpp (`llama-cpp-2` with the `vulkan` feature) --
  # neither is inside the executable and both are looked for beside it.
  # `scripts/dev-runtime.test.ts` reads those three files and fails if this line
  # stops agreeing with them, so the repository stays the authority for it.
  local -a expected_libraries=(onnxruntime.dll vulkan-1.dll)
  # `dirname` is a measurement too. Under `set -e` a bare assignment from a
  # failing command substitution aborts the whole script from inside a function
  # whose caller was going to branch on its status.
  source_dir="$(dirname -- "$source")" || {
    echo "error: could not take the directory of the built dev daemon: $source" >&2
    return 1
  }
  if ! mkdir -p "$DAEMON_STAGE_DIR"; then
    echo "error: could not create the dev daemon stage directory" >&2
    echo "       $DAEMON_STAGE_DIR" >&2
    echo "       nothing below can be staged, and every check below would be" >&2
    echo "       asking about a directory that is not there" >&2
    return 1
  fi
  stage_file_by_identity "$source" "$DAEMON_STAGE_DIR/wenlan-server.exe" \
    "dev daemon" || return 1
  # The loader resolves the runtime libraries from the executable's own
  # directory, so they have to travel with it. app/binaries is the staging area
  # prepare-sidecars.sh fills, and is the only source on a checkout that has not
  # built the app yet.
  #
  # `|| true` used to sit on the copy below, which is os error 32's own defect:
  # the one failure that actually happens here is a DLL held open by a process
  # running out of the stage directory, and swallowing it left the OLD library
  # beside a NEW daemon -- a mismatch no probe downstream looks for, reported as
  # a clean start. Every copy failure is named and fatal now.
  #
  # And then the same defect one level OUT, which is what this loop is now
  # shaped against. It used to be `for dll in "$dir"/*.dll; do [[ -f "$dll" ]] ||
  # continue`, and a glob has two answers where the question has three: a
  # directory that cannot be listed leaves the pattern unexpanded, `-f` rejects
  # the literal `*.dll`, the body never runs, and staging returns SUCCESS having
  # copied NO libraries. That is indistinguishable from the fresh checkout where
  # there are genuinely none. So the DIRECTORY is measured, with the same
  # tri-state listing `read_owned_pid` stands on, and the names are read out of
  # the listing instead of from a pattern that cannot fail.
  for dir in "$REPO_ROOT/app/binaries" "$source_dir"; do
    rc=0
    listing="$(list_dir_tristate "$dir")" || rc=$?
    case "$rc" in
      0) ;;
      # Genuinely not there. A checkout that has never run prepare-sidecars.sh
      # has no app/binaries at all, and that is not an error -- it is the
      # measured negative, and it is the one this may act on.
      1) continue ;;
      *)
        echo "error: could not list the runtime libraries in $dir" >&2
        echo "       an unreadable directory is not an empty one: the glob this" >&2
        echo "       replaced would simply not expand, every copy would be" >&2
        echo "       skipped, and staging would report success having copied" >&2
        echo "       nothing beside the daemon" >&2
        return 1
        ;;
    esac
    while IFS= read -r name; do
      [[ -n "$name" ]] || continue
      # Evidence that prepare-sidecars.sh has run into this directory: it
      # installs the sidecars under their triple-qualified names, and on a
      # Windows triple it refuses to finish unless the runtime libraries are
      # beside them. So a binaries directory holding a prepared sidecar and NOT
      # holding them is a broken layout -- which is a different thing from the
      # empty directory of a checkout that has not prepared anything, and the
      # difference is exactly what a zero-library stage has to be able to say.
      if [[ "$dir" == "$REPO_ROOT/app/binaries" && "$name" == wenlan-server-* ]]; then
        prepared=1
      fi
      [[ "$name" == *.dll ]] || continue
      stage_file_by_identity "$dir/$name" "$DAEMON_STAGE_DIR/$name" \
        "runtime library $name" || return 1
      # TWO SOURCES, ONE NAME. `app/binaries` and the build directory are both
      # walked, and nothing stopped the second from staging a DIFFERENT
      # onnxruntime.dll over the first. Each copy verified: the first was
      # written and hashed and matched, and then the second was written and
      # hashed and matched, and the file the daemon loads is whichever came
      # last. Every per-file check in this function is blind to that, because
      # the question is not about a file, it is about the SET.
      #
      # A collision on identical bytes is nothing at all -- the same library
      # prepared into both places -- and is allowed silently. A collision on
      # different bytes is a layout nobody can call correct, and picking one is
      # exactly the guess this file refuses to make.
      previous=""
      previous_rc=1
      while IFS= read -r line; do
        [[ "${line%%$'\t'*}" == "$name" ]] || continue
        previous="${line#*$'\t'}"
        previous_rc=0
        break
      done <<<"$staged_digest"
      if (( previous_rc == 0 )) && [[ "$previous" != "$STAGED_FILE_DIGEST" ]]; then
        echo "error: two different runtime libraries are both called $name" >&2
        echo "       $REPO_ROOT/app/binaries and $source_dir disagree about it" >&2
        echo "       first:  $previous" >&2
        echo "       second: $STAGED_FILE_DIGEST ($dir/$name)" >&2
        echo "       staging the second over the first would decide by walk" >&2
        echo "       order which library the daemon loads; re-run" >&2
        echo "       pnpm prepare:sidecars so the two agree" >&2
        return 1
      fi
      if (( previous_rc != 0 )); then
        staged=$((staged + 1))
        staged_names+="$name "
        staged_digest+="$name"$'\t'"$STAGED_FILE_DIGEST"$'\n'
      fi
    done <<<"$listing"
  done
  # THE STAGED SET, verified as a set and not as a series of copies.
  #
  # Everything above answers "did this file get there", and the daemon does not
  # load a series of files -- it loads whatever is in its own directory. A stage
  # left over from an earlier run can hold an onnxruntime.dll that NEITHER
  # source directory has any more (the build moved, prepare-sidecars was
  # re-run into a different layout, someone dropped a library in by hand).
  # Nothing above removes it and nothing above looks at it, so this function
  # returned success and printed "no runtime libraries beside the daemon" while
  # a stale one sat there ready to be loaded. The two claims were made about
  # different directories.
  #
  # So the destination is measured, with the same tri-state listing the sources
  # use, and any `.dll` in it that this run did not put there is named and
  # fatal. Not deleted: a library this runtime cannot account for is not a
  # library this runtime should be quietly destroying, and the fix -- which of
  # the two it wanted -- is not one this can pick.
  rc=0
  listing="$(list_dir_tristate "$DAEMON_STAGE_DIR")" || rc=$?
  case "$rc" in
    0)
      while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        [[ "$name" == *.dll ]] || continue
        [[ "$staged_names" == *" $name "* ]] || stray+=("$name")
      done <<<"$listing"
      ;;
    *)
      # Including 1. The stage directory was created at the top of this
      # function and the daemon was staged into it, so "it is not there" is not
      # a negative anybody can act on -- it is the same failed measurement as
      # "it would not list", arriving by a different route.
      echo "error: the staged daemon directory could not be listed after staging" >&2
      echo "       $DAEMON_STAGE_DIR (list status $rc)" >&2
      echo "       an unverified stage is not a stage: what is beside the daemon" >&2
      echo "       is the thing it will load, and nothing here has looked at it" >&2
      return 1
      ;;
  esac
  if (( ${#stray[@]} > 0 )); then
    echo "error: the dev daemon stage holds ${stray[*]}, which this run did not put there" >&2
    echo "       $DAEMON_STAGE_DIR" >&2
    echo "       neither $REPO_ROOT/app/binaries nor $source_dir has it now, so" >&2
    echo "       nothing verified the bytes the daemon would load from it" >&2
    echo "       remove the stage and re-run: rm -rf $DAEMON_STAGE_DIR" >&2
    return 1
  fi
  for want in "${expected_libraries[@]}"; do
    [[ "$staged_names" == *" $want "* ]] || missing+=("$want")
  done
  if (( prepared == 1 )) && (( ${#missing[@]} > 0 )); then
    echo "error: the dev daemon was staged without ${missing[*]}" >&2
    echo "       $REPO_ROOT/app/binaries holds prepared sidecars, so" >&2
    echo "       prepare-sidecars.sh put ${expected_libraries[*]} there and" >&2
    echo "       something has taken them away since" >&2
    echo "       a daemon staged without the libraries it loads from its own" >&2
    echo "       directory is not a staged daemon; re-run pnpm prepare:sidecars" >&2
    return 1
  fi
  if (( staged == 0 )); then
    # NAMED, and this is the whole point of the rewrite: a zero-library stage is
    # a real and ordinary outcome on a checkout that has not built the app, and
    # it is ALSO what a directory nobody could read used to look like. One of
    # them is now an error above; this is the other, and it says so rather than
    # being a silent success that could have been either.
    echo "note: staged the dev daemon with no runtime libraries beside it" >&2
    echo "      neither $REPO_ROOT/app/binaries nor $source_dir holds any" >&2
    echo "      run pnpm prepare:sidecars if the daemon cannot load onnxruntime" >&2
  fi
}

# One directory has many spellings on Windows -- a different case, a DOS 8.3
# alias, a `\\?\` prefix, a trailing dot, a junction -- and the production-root
# guard below compares these as strings, so every spelling it cannot reduce is a
# way past it. Normalizing them by hand is a losing game, so this asks the OS
# instead: `fs.realpathSync.native` goes through GetFinalPathNameByHandle on
# Windows and answers with the real on-disk path, and is realpath(3) on POSIX,
# which is what this always used. Only something that exists can be resolved, so
# the non-existent tail is peeled off first and re-attached afterwards.
canonicalize_path() {
  node -e '
    const fs = require("node:fs");
    const path = require("node:path");
    let resolved = path.resolve(process.argv[1]);
    // Win32 drops trailing dots and spaces from every component, so
    // `...\wenlan.` opens `...\wenlan`. Node resolves through the extended
    // `\\?\` form, where they are literal instead, so it would call that a
    // different and missing directory and hand the guard a path that never
    // matches -- while the daemon, going through Win32, writes to the real one.
    // Drop them here so both see the same directory. The drive is left alone,
    // and `.` and `..` are already gone by now.
    //
    // Only ordinary Win32 paths reach this: a caller that writes `\\?\` or
    // `\\.\` is refused upstream, because nothing here can carry that prefix
    // through -- `realpathSync.native` answers without it -- so the literal
    // trailing characters it promises would stop being literal right here.
    if (process.platform === "win32") {
      resolved = resolved
        .split(path.sep)
        .map((part, index) => (index === 0 ? part : part.replace(/[. ]+$/, "")))
        .join(path.sep);
    }
    let suffix = "";
    for (;;) {
      try {
        resolved = fs.realpathSync.native(resolved);
        break;
      } catch {
        const parent = path.dirname(resolved);
        if (parent === resolved) break;
        suffix = "/" + path.basename(resolved) + suffix;
        resolved = parent;
      }
    }
    process.stdout.write(resolved.split(path.sep).join("/") + suffix);
  ' "$1"
}

# Windows resolves paths case-insensitively, and `realpath` hands back whatever
# casing the caller wrote, so %LOCALAPPDATA%\WENLAN and %LOCALAPPDATA%\wenlan
# reach here as two different strings naming one directory. Comparing them
# literally lets the second spelling walk past the production-root guard and
# point the dev daemon at the real data directory. Fold case there, and only
# there: on Linux those are genuinely two directories, and folding would refuse
# a path that is not production at all.
#
# exit: 0 within, 1 measured outside, 2 the comparison could not be made.
#
# The third answer is here because this is called as an `if` CONDITION, which
# disables errexit for the whole body, and the two `tr` captures underneath had
# no status check at all. A `tr` that is not on PATH, or a pipeline that failed,
# yields the empty string -- and an empty child compared against a non-empty
# parent is `false`, which this function's caller reads as "this path is not
# inside production" and lets through. A case-fold that did not happen is not
# evidence about a path; it is the absence of evidence, and the guard it feeds
# is one of the ones keeping a dev daemon out of the real data directory.
path_is_within() {
  local child="$1" parent="$2" rc=0
  if (( HOST_IS_WINDOWS == 1 )); then
    child="$(printf '%s' "$child" | tr '[:upper:]' '[:lower:]')" || rc=$?
    (( rc == 0 )) || return 2
    [[ -n "$child" ]] || return 2
    parent="$(printf '%s' "$parent" | tr '[:upper:]' '[:lower:]')" || rc=$?
    (( rc == 0 )) || return 2
    [[ -n "$parent" ]] || return 2
  fi
  [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
}

# Refuse an extended-length or device path instead of quietly rewriting it.
#
# `\\?\` and `\\.\` mean "pass this to the filesystem unchanged": trailing dots
# and spaces are part of the name rather than something Win32 discards. Nothing
# downstream can honour that. `realpathSync.native` answers without the prefix,
# MSYS `mkdir` and `realpath` go through Win32, and so does the daemon -- so a
# verbatim path arrives at the guard as an ordinary one whose components no
# longer mean what the caller wrote. `\\?\%LOCALAPPDATA%\wenlan.\dev` would pass
# the production check as a sibling directory and then be opened as
# `...\wenlan\dev`, inside production.
#
# Stripping the components instead redirects the path, which is the opposite
# error: it would send a genuine `\\?\C:\scratch\dev.\data` to `...\dev\data`.
# Neither is acceptable silently, so this says no and names the fix. Windows
# only: on POSIX a leading `\\` is an ordinary relative filename.
#
# The prefix is matched by shape rather than by spelling. Win32 reads either
# separator in any of the three positions, and `path.resolve` folds all sixteen
# combinations -- `\\?/`, `//?\`, `/\./` and the rest -- into the same two
# canonical prefixes, so listing the tidy ones would leave fourteen ways in.
# `\\server\share` is not one of them: the third character has to be `?` or `.`.
# Every guard below refuses, and every one of those refusals is a SAFETY event:
# these are the gates that keep a dev runtime off production ports, identities,
# paths and sockets. They all exited 2 and nothing else, and 2 is also what a
# usage error exits, so the only thing separating the two downstream was the
# wording of the message. The kind is stated here instead. See THE OUTCOME
# CONTRACT at the top of this file; the prose is unchanged.
refuse_unsafe() {
  local line
  for line in "$@"; do
    echo "$line" >&2
  done
  RESULT_KIND=safety-refusal
  exit 2
}

reject_verbatim_path() {
  local label="$1" value="$2"
  # Written `[\\/]` rather than `[\/]` so the same text is a valid ERE here and
  # a valid regex in the test that reads it back out: inside a POSIX bracket
  # expression a backslash is literal, so the doubled one is the same set.
  local verbatim='^[\\/][\\/][?.][\\/]'
  (( HOST_IS_WINDOWS == 1 )) || return 0
  if [[ "$value" =~ $verbatim ]]; then
    refuse_unsafe \
      "error: refusing extended-length or device path for $label: $value" \
      "       write it without the \\\\?\\ or \\\\.\\ prefix"
  fi
}

refuse_production_path() {
  local label="$1" value="$2" canonical root resolved within_rc
  local -a roots=(
    "$HOME/Library/Application Support/wenlan"
    "$HOME/Library/Application Support/origin"
    "$HOME/Library/LaunchAgents"
    "$HOME/Library/Logs/com.wenlan.desktop"
    "$HOME/Library/Logs/com.origin.desktop"
    "$HOME/.config/wenlan-mcp"
    "$HOME/.config/origin-mcp"
    "$HOME/.wenlan"
    "$HOME/.origin"
  )
  # The Windows half of the same list the app enforces in
  # `production_runtime_roots`. Only appended when the variable is actually set:
  # an empty root would canonicalize to the working directory and refuse the
  # whole checkout.
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    roots+=("$LOCALAPPDATA/wenlan" "$LOCALAPPDATA/origin")
  fi
  # A path that could not be canonicalized cannot be compared against the roots,
  # and "the comparison did not run" is not "it is not production". Under
  # `set -e` this aborted the script with node's own status and no message,
  # which is the right DIRECTION and the wrong report: the run refused for a
  # reason nothing named, and the marker would have said `unknown` for what is
  # a guard that could not be applied.
  canonical="$(canonicalize_path "$value")" || refuse_unsafe \
    "error: could not resolve $label for the production-root check: $value" \
    "       an unresolvable path cannot be shown to be outside production"
  for root in "${roots[@]}"; do
    resolved="$(canonicalize_path "$root")" || refuse_unsafe \
      "error: could not resolve the production root $root" \
      "       refusing to check $label against a root that would not resolve"
    within_rc=0
    path_is_within "$canonical" "$resolved" || within_rc=$?
    case "$within_rc" in
      0) refuse_unsafe "error: refusing production path for $label: $value" ;;
      1) ;;
      # A comparison that did not run is not a path outside production, and this
      # is the guard that keeps a dev daemon off the real data directory. It
      # refuses, which is the direction every other unmeasured answer in this
      # file takes.
      *)
        refuse_unsafe \
          "error: could not compare $label against the production root $root" \
          "       $value" \
          "       a comparison that did not run is not a path outside production"
        ;;
    esac
  done
}

if [[ ! "$DEV_PORT" =~ ^[0-9]+$ ]] || (( DEV_PORT < 1 || DEV_PORT > 65535 )); then
  refuse_unsafe "error: invalid WENLAN_DEV_PORT: $DEV_PORT"
fi
if [[ ! "$DEV_UI_PORT" =~ ^[0-9]+$ ]] || (( DEV_UI_PORT < 1 || DEV_UI_PORT > 65535 )); then
  refuse_unsafe "error: invalid WENLAN_DEV_UI_PORT: $DEV_UI_PORT"
fi
if [[ ! "$DEV_REMOTE_PORT_START" =~ ^[0-9]+$ ]] ||
  (( DEV_REMOTE_PORT_START < 1 || DEV_REMOTE_PORT_START > 65532 )); then
  refuse_unsafe "error: invalid WENLAN_DEV_REMOTE_PORT_START: $DEV_REMOTE_PORT_START"
fi
if (( DEV_PORT == 7878 )); then
  refuse_unsafe "error: refusing production daemon port 7878"
fi
if (( DEV_UI_PORT == 1420 )); then
  refuse_unsafe "error: refusing production UI identity on port 1420"
fi
if (( DEV_REMOTE_PORT_START <= 18083 && DEV_REMOTE_PORT_START + 3 >= 18080 )); then
  refuse_unsafe "error: refusing production remote-access port range 18080-18083"
fi
if [[ "$DEV_APP_ID" == "com.wenlan.desktop" || "$DEV_APP_ID" == "com.origin.desktop" ]]; then
  refuse_unsafe "error: refusing production app identifier: $DEV_APP_ID"
fi
reject_verbatim_path "WENLAN_DEV_STATE_DIR" "$STATE_DIR"
reject_verbatim_path "WENLAN_DEV_DATA_DIR" "$DEV_DATA_DIR"
reject_verbatim_path "WENLAN_DEV_TAURI_MCP_SOCKET" "$DEV_TAURI_MCP_SOCKET"
# Two canonicalizations, each with a status: `[[ "$(f)" == "$(g)" ]]` compares
# the OUTPUT of two commands and discards both statuses, so two failures compare
# equal and the production socket guard fires on a comparison neither side made.
DEV_SOCKET_CANONICAL="$(canonicalize_path "$DEV_TAURI_MCP_SOCKET")" || refuse_unsafe \
  "error: could not resolve WENLAN_DEV_TAURI_MCP_SOCKET: $DEV_TAURI_MCP_SOCKET"
PROD_SOCKET_CANONICAL="$(canonicalize_path "/tmp/tauri-mcp.sock")" || refuse_unsafe \
  "error: could not resolve the production Tauri MCP socket path"
if [[ "$DEV_SOCKET_CANONICAL" == "$PROD_SOCKET_CANONICAL" ]]; then
  refuse_unsafe "error: refusing production Tauri MCP socket: $DEV_TAURI_MCP_SOCKET"
fi
refuse_production_path "WENLAN_DEV_STATE_DIR" "$STATE_DIR"
refuse_production_path "WENLAN_DEV_DATA_DIR" "$DEV_DATA_DIR"
refuse_production_path "WENLAN_DEV_TAURI_MCP_SOCKET" "$DEV_TAURI_MCP_SOCKET"

# The three native spellings are resolved before anything is printed, so a
# `cygpath` that fails cannot leave a consumer holding an empty path -- which
# `$(native_path …)` inline in a printf argument would have done silently.
#
# The printf statuses are read for the same reason the `native_path` ones are,
# and it is not tidiness. `dev-all.sh` EVALS these lines, so a run that emits
# six of the seven hands the app a configuration with one variable missing --
# and the missing one is as likely as not `WENLAN_DATA_DIR`, whose default is
# the PRODUCTION data root. This function is called on the left of `||`, where
# errexit is disabled and the return value is the last printf's, so a failure in
# any earlier line was invisible. First failure wins, and it is fatal.
print_config() {
  local socket data state rc=0
  socket="$(native_path "$DEV_TAURI_MCP_SOCKET")" ||
    { echo "error: could not spell $DEV_TAURI_MCP_SOCKET for a native consumer" >&2; return 1; }
  data="$(native_path "$DEV_DATA_DIR")" ||
    { echo "error: could not spell $DEV_DATA_DIR for a native consumer" >&2; return 1; }
  state="$(native_path "$STATE_DIR")" ||
    { echo "error: could not spell $STATE_DIR for a native consumer" >&2; return 1; }
  printf 'WENLAN_PORT=%s\n' "$DEV_PORT" || rc=1
  printf 'WENLAN_DEV_UI_PORT=%s\n' "$DEV_UI_PORT" || rc=1
  printf 'WENLAN_DEV_REMOTE_PORT_START=%s\n' "$DEV_REMOTE_PORT_START" || rc=1
  printf 'WENLAN_DEV_APP_ID=%s\n' "$DEV_APP_ID" || rc=1
  printf 'WENLAN_DEV_TAURI_MCP_SOCKET=%s\n' "$socket" || rc=1
  printf 'WENLAN_DATA_DIR=%s\n' "$data" || rc=1
  printf 'WENLAN_DEV_STATE_DIR=%s\n' "$state" || rc=1
  if (( rc != 0 )); then
    echo "error: the dev runtime configuration could not be written in full" >&2
    echo "       a consumer that evals a partial config gets production" >&2
    echo "       defaults for whatever line did not arrive" >&2
    return 1
  fi
}

# Read the ownership record.
# exit: 0 a complete, well-formed record; 1 there is NO record; 2 could not
# measure -- a torn or unreadable record, which is not the same thing.
#
# This was two-state, and it was the same defect the rest of this file is
# about, one level further back. "No record" and "a record I could not read"
# both returned 1: `stop_runtime` printed "No worktree-owned Wenlan dev daemon
# is recorded" and exited 0, and `start_runtime` fell through to
# `clear_owned_state` and DELETED all four files. An unreadable pid file --
# a torn write, a locked file, an I/O error -- therefore made a live daemon
# permanently unattributable, which is the worst outcome available here and
# exactly what the tri-state probes downstream exist to prevent.
#
# Three files define the record. All absent is the measured negative. Any other
# combination is a partial record: a torn write from a daemon that IS running,
# not evidence that none is.
# Tri-state directory listing, for `read_owned_pid` below.
#   0 -- listed; the listing is on stdout
#   1 -- the path is genuinely absent: an ancestor listed and does not contain
#        the component below it
#   2 -- could not measure: something on the path exists but would not list, or
#        nothing on the path would list at all
#
# It CLIMBS, because STATE_DIR is `$TMPDIR/wenlan-app-dev/$WORKTREE_ID` and on a
# machine that has never run this, neither of the last two components exists. A
# single "did the parent list?" test would answer 2 there and make `stop` fail
# on a clean checkout -- a correctness fix that breaks the ordinary case is not
# a fix. Climbing distinguishes the two properly: the first ancestor that lists
# either names the component below it (so that component exists and is the thing
# that would not list -- unmeasured) or does not (so the whole chain below is
# really absent).
#
# The trailing `/.` is load-bearing, and it is this workstream's defect again:
# `ls -A -- somefile` SUCCEEDS and prints the path it was given, so a path
# component that is a plain file would have "listed" and the name search would
# have found nothing -- a failed measurement reading as an empty directory.
# `/.` makes the same call mean "list this AS A directory", which fails.
#
# Is NAME one of the lines in LISTING?
# exit: 0 present, 1 measured absent, 2 the question could not be asked.
#
# `grep -qxF` answers 0 for found and 1 for absent -- and 2 for an error it hit
# on the way, and 127 when there is no `grep` on PATH at all. `... && present=…`
# and `... && return 2` read all four as the same "not found", which is this
# file's one defect in its smallest form: `ls` LISTED the state directory, the
# name search then failed to run, and the caller concluded that no ownership
# record exists -- after which `start_runtime` falls through to
# `clear_owned_state` and deletes the record of a daemon that may still be
# running. So the status is read as three answers, and the third is propagated
# by every caller below.
#
# A here-string and not a pipe: `grep -q` exits at its first match, so a
# producer on the left of a pipe can take SIGPIPE and report 141 under
# `pipefail` -- a real parser failure and a successful match, spelled the same.
# With `<<<` the status is grep's own and nothing else's.
listing_has_name() {
  local listing="$1" name="$2" rc=0
  grep -qxF -- "$name" <<<"$listing" || rc=$?
  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
}

list_dir_tristate() {
  local dir="$1" child="" listing parent has_rc
  while :; do
    if listing="$(ls -A -- "$dir/." 2>/dev/null)"; then
      if [[ -z "$child" ]]; then
        printf '%s\n' "$listing"
        return 0
      fi
      has_rc=0
      listing_has_name "$listing" "$child" || has_rc=$?
      case "$has_rc" in
        # The component below this one exists, so it is the thing that would
        # not list.
        0) return 2 ;;
        1) return 1 ;;
        # And a name search that could not run says nothing about whether the
        # component is there. Answering 1 here would report a chain nobody
        # looked at as a chain that is not there.
        *) return 2 ;;
      esac
    fi
    # `dirname` and `basename` are measurements too, and under `set -e` a bare
    # assignment from a failing one aborts the enclosing shell -- which, because
    # this function is always called inside a command substitution, surfaces at
    # the caller as status 1: "there is no record". Named and answered 2.
    parent="$(dirname -- "$dir")" || return 2
    [[ "$parent" == "$dir" ]] && return 2
    child="$(basename -- "$dir")" || return 2
    dir="$parent"
  done
}

read_owned_pid() {
  local present=0 f listing list_rc=0 name_rc
  # Round 13f reopened this one level in. The `sed` reads below were made
  # tri-state; the EXISTENCE test above them was not. `[[ -e "$f" ]]` has two
  # answers and the question has three -- a file that is not there and a
  # directory that cannot be asked are the same `false`. An unreadable
  # STATE_DIR (a permission change, a parent that has gone away, a stat that
  # fails) made all three lookups false, `present == 0` returned 1, and the
  # caller read "nothing is recorded": `stop_runtime` reports success having
  # stopped nothing, and `start_runtime` falls through to `clear_owned_state`
  # and deletes the ownership record of a daemon that may still be running.
  #
  # A fourth `-e` cannot fix that, because `-e` is the thing without a third
  # answer. Ask the DIRECTORY for its contents instead, with something that has
  # a status per outcome, and read the names out of the listing.
  listing="$(list_dir_tristate "$STATE_DIR")" || list_rc=$?
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
  done
  (( present == 0 )) && return 1
  (( present == 3 )) || return 2
  # `sed`'s status is the measurement. `OWNED_PID="$(sed …)"` discards it, and
  # an I/O error then yields an empty string that the regex below rejects --
  # reporting a failed read as a malformed record, and previously as no record.
  OWNED_PID="$(sed -n '1p' "$PID_FILE")" || return 2
  OWNED_SERVER="$(sed -n '1p' "$SERVER_PATH_FILE")" || return 2
  OWNED_PORT="$(sed -n '1p' "$PORT_FILE")" || return 2
  # The data dir is the one optional member: records written before it existed
  # legitimately lack it. Absent is empty; unreadable is not.
  OWNED_DATA_DIR=""
  # Read off the same listing, for the same reason: `-e` here would reintroduce
  # the two-answer test the block above exists to remove -- and so would a
  # search whose own failure is spelled the same as "the name is not there",
  # which is what an `if grep …; then` reads as.
  name_rc=0
  listing_has_name "$listing" "${DATA_DIR_FILE##*/}" || name_rc=$?
  case "$name_rc" in
    0) OWNED_DATA_DIR="$(sed -n '1p' "$DATA_DIR_FILE")" || return 2 ;;
    1) ;;
    *) return 2 ;;
  esac
  [[ "$OWNED_PID" =~ ^[0-9]+$ && -n "$OWNED_SERVER" &&
    "$OWNED_PORT" =~ ^[0-9]+$ ]] || return 2
}

# Is the recorded pid still running the recorded executable?
# exit: 0 yes, 1 no (dead, or a different image), 2 could not measure.
#
# The old two-state form collapsed a failed liveness probe into "no" before the
# image lookup ever ran, so ownership was decided on an unmeasured premise, and
# it captured the image lookup's text while discarding its status -- a helper
# that printed the right path and then failed still counted as a match.
has_owned_command_identity() {
  local command want got
  probe_process_alive "$OWNED_PID"
  case "$PROCESS_ALIVE_STATE" in
    gone) return 1 ;;
    unmeasured) return 2 ;;
  esac
  probe_process_image "$OWNED_PID"
  case "$PROCESS_IMAGE_STATE" in
    none) return 1 ;;
    unmeasured) return 2 ;;
  esac
  command="$PROCESS_IMAGE_VALUE"
  if (( HOST_IS_WINDOWS == 1 )); then
    got="$(normalize_program_path "$command")" || return 2
    want="$(normalize_program_path "$OWNED_SERVER")" || return 2
    [[ "$got" == "$want" ]]
  else
    [[ "$command" == "$OWNED_SERVER" || "$command" == "$OWNED_SERVER "* ]]
  fi
}

# Is the recorded pid ours AND the process serving the recorded port?
#
# exit: 0 yes.
#       1 no, and the record is stale: the recorded pid is GONE, or it is a
#         different image. Nothing of this runtime's is running under it.
#       2 could not measure.
#       3 the recorded pid IS alive and IS running the recorded executable, and
#         it is not the listener on the recorded port.
#
# ROUND 5 SPLIT 3 OUT OF 1, and 1 was the answer that licensed a deletion.
# `has_owned_command_identity` can succeed -- the pid is alive, the image is the
# staged executable this worktree launched -- and the port probe still answer
# `none`, because binding the socket happens some milliseconds AFTER the process
# exists. A launcher that died in that window leaves exactly this state: a live
# daemon, a valid record, and no listener yet. Folded into 1 it read as "the
# record is stale", and `start_runtime` fell through to `clear_owned_state` --
# deleting the ownership record of a RUNNING daemon, which is the state this
# file calls its worst outcome, and then launching a second daemon against the
# same data directory. The `found`-but-another-pid case is the same shape: our
# process is alive and something else holds the port, which is not evidence that
# our record is stale either.
#
# BOTH consumers branch on it -- `stop_runtime` and `start_runtime`, and there
# are exactly two. The count said "three" while there were two, which is worse
# than saying nothing: a number in a comment about how many callers handle a
# tri-state is the thing a reader checks against instead of grepping, and a
# reader who trusts it has already been told that one caller they cannot find
# is somewhere handling state 3. LISTENER_PROBE_STATE and LISTENER_PROBE_PID
# are left where the caller can read them, so the diagnostics can say which of
# the two shapes produced the 3 without measuring anything twice.
is_owned_process() {
  local rc=0
  has_owned_command_identity || rc=$?
  (( rc == 0 )) || return "$rc"
  probe_listener_port "$OWNED_PORT"
  case "$LISTENER_PROBE_STATE" in
    none) return 3 ;;
    unmeasured) return 2 ;;
  esac
  [[ "$LISTENER_PROBE_PID" == "$OWNED_PID" ]] || return 3
}

# One acquisition's name for the lock directory it created. The pid stays first
# so the recovery path can still ask the kernel about it; the nonce is what two
# generations cannot share.
lock_new_token() {
  RUNTIME_LOCK_GEN=$(( RUNTIME_LOCK_GEN + 1 ))
  printf '%s %s.%s.%s' "$$" "$RUNTIME_LOCK_GEN" "$RANDOM" "$RANDOM"
}

# Only this process's own lock is ever removed, and only when the owner file
# says so IN A READ THAT SUCCEEDED. A `sed` whose status was dropped yields the
# empty string, which is not `$$`, so an unreadable owner file leaves the lock
# in place -- the safe direction here, and the status is read anyway so the two
# cases stop being spelled the same.
#
# Every path through this clears RUNTIME_LOCK_HELD, including the ones that
# release nothing. It is what makes the outcome handler idempotent from the
# other side: a second release would otherwise run after the marker and put its
# own `rm` diagnostic below the line this file promises is last.
release_runtime_lock() {
  local owner rc=0 failed=0 list_rc=0 name_rc=0 listing
  # ROUND 4. These three tests used to `return 0` -- "released" -- and the
  # collapse was INSIDE this function, where the `|| true` at the trap never
  # saw a status to discard. Each of them is a case where the lock CANNOT BE
  # SHOWN to be gone, and this only runs when RUNTIME_LOCK_HELD was 1, so each
  # of them is anomalous rather than ordinary. Reporting them as a release is
  # how a command exits 0 with `DEV_RUNTIME_RESULT: ok` while its lock is still
  # on disk, and the next user-visible command refuses on a lock the previous
  # successful one said it had released.
  RUNTIME_LOCK_HELD=0
  # ROUND 5. This began `if [[ ! -f "$LOCK_OWNER_FILE" ]]; then [[ -d
  # "$LOCK_DIR" ]] || return 0`, which is two TWO-answer tests standing in for
  # one THREE-answer question, and both of them fail in the same direction.
  # Remove search permission from any ancestor of the lock -- a permission
  # change, a parent that went away, a mount that dropped out -- and neither
  # path can be examined: `-f` is false because the owner file cannot be
  # stat'ed, `-d` is false for the same reason one level up, and the function
  # returns 0, "released", for a lock it could not look at. `on_runtime_exit`
  # then keeps RESULT_KIND=ok and the command exits 0 with its lock still on
  # disk -- the exact outcome the round-4 note above says this function exists
  # to stop, reached through the one test round 4 did not convert.
  #
  # `list_dir_tristate` is the three-answer form of the same question and this
  # file's own idiom for it: 0 the directory listed, 1 it is MEASURED absent
  # (it walks the parents to establish that), 2 nothing could be established.
  # `-e`/`-f`/`-d` cannot express the third, which is why none of them is used
  # here any more.
  listing="$(list_dir_tristate "$LOCK_DIR")" || list_rc=$?
  case "$list_rc" in
    0) ;;
    # ROUND 6. THIS ARM USED TO `return 0`, AND ITS REASON WAS ASSERTED RATHER
    # THAN MEASURED: "a recovery that found this run dead is entitled to have
    # done that". Nothing on this path establishes that any such recovery
    # happened, let alone that it was entitled to. What IS established is
    # narrow and it is enough: `RUNTIME_LOCK_HELD` is set in exactly one place
    # -- last in `acquire_runtime_lock`, after every step of taking the lock
    # succeeded -- and this function only runs when it is 1. So reaching here
    # means this run demonstrably TOOK the lock and the lock directory is now
    # MEASURED absent. That is not a release. It is this run's exclusive claim
    # being broken while the run was still using it, and for the length of that
    # window the isolated port and data directory this lock exists to fence off
    # were not fenced off from anybody.
    #
    # Reported as `unknown`, and the choice is not arbitrary. It is not `ok`:
    # the premise the work above rests on did not hold. It is not
    # `safety-refusal` either, and that is the one worth being careful about --
    # a safety refusal says a guard STOPPED the command, so a supervisor reads
    # it as "nothing was done". Here the command DID its work; what failed was
    # the guard around it. Claiming a refusal would be the more dangerous of
    # the two wrong answers, because it describes an untouched system. What is
    # genuinely unmeasured is whether this run's work interleaved with another
    # run's on the same port and data directory, which is precisely what the
    # contract at the top of this file means by `unknown`: a refusal, treated
    # the way `safety-refusal` is treated, over something nobody established.
    #
    # `attest.sh` reports the same condition through `LOCK_STOLEN`. The two
    # files must not disagree about what a vanished lock means.
    1)
      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock this run took is gone from $LOCK_DIR" >&2
      echo "       it was removed while this run still held it, so this is not" >&2
      echo "       a release: the isolation it was providing ended early and" >&2
      echo "       nothing here can say what ran inside that window" >&2
      return 1
      ;;
    *)
      echo "error: the dev runtime lock could not be examined at release" >&2
      echo "       $LOCK_DIR" >&2
      echo "       a lock that cannot be looked at has not been shown to be" >&2
      echo "       gone, and reporting a release on that is how a command exits" >&2
      echo "       0 with its lock still on disk; the next command will refuse" >&2
      return 1
      ;;
  esac
  listing_has_name "$listing" "${LOCK_OWNER_FILE##*/}" || name_rc=$?
  case "$name_rc" in
    0) ;;
    1)
      # ROUND 6, AND IT IS THE `rmdir` THAT WAS THE DEFECT, NOT THE VERDICT.
      #
      # This arm used to `rmdir "$LOCK_DIR"` and report a clean release, on the
      # reasoning that "a directory still standing without an owner file is the
      # exact state `acquire_runtime_lock` refuses on, so leaving it silently
      # arms the next command's refusal". The tidying is well meant and it is
      # not this run's to do, because the state it acts on is consistent with
      # two different worlds and destroying is only licensed in one of them:
      #
      #   1. a stale-breaker is mid-break on US -- it has removed our owner file
      #      and has not yet removed the directory; or
      #   2. a breaker already finished, a NEW holder has `mkdir`'d, and it is
      #      between its `mkdir` and its owner write.
      #
      # World 2 is not a corner: an EMPTY lock directory with no owner file is
      # precisely what a fresh holder looks like for the width of that window,
      # so the case this arm was most confident about -- `rmdir` succeeds, so it
      # must have been an abandoned shell -- is the one most likely to be a live
      # peer. Removing it there deletes a lock this run does not hold, and two
      # runs then share an isolated port and data directory. That is not a
      # misreport, it is a destructive act against a live peer, and it is the
      # same ABA `attest.sh` closes from the victim's side with
      # `lock_generation_state`.
      #
      # `scripts/AGENTS.md`, verbatim: "Ending, deleting or switching something
      # off needs free-before AND measured-present-after, so the thing being
      # destroyed is provably the thing this run made." Our owner file is the
      # only evidence that would make this directory provably ours, and it is
      # the thing that is missing. So nothing is removed.
      #
      # The directory is left for `acquire_runtime_lock` to arbitrate, which is
      # where that decision belongs and where the machinery for it already is:
      # `lock_owner_file_appeared` has a FOURTH state for exactly this shape
      # ("still standing and still naming nobody") and the next command refuses
      # on it. Leaving it does arm that refusal -- correctly. The alternative
      # was disarming it by force on a guess.
      #
      # THE VERDICT MOVES WITH IT, and not by choice. What separated the old
      # `ok` from the old `unknown` here was nothing but whether `rmdir`
      # SUCCEEDED -- an empty directory returned 0 and a non-empty one returned
      # 1. Take the destructive act away and that distinction has no other
      # witness: both inputs are one indistinguishable state, "the directory
      # stands and this run's owner file is not in it". One verdict has to cover
      # both, and it is a refusal, because this run wrote that owner file last
      # in `acquire_runtime_lock` and something removed it while the run was
      # still using the port and data directory it fenced. That is the same
      # event as a vanished directory and as an owner file naming somebody else,
      # and `attest.sh:lock_generation_state` answers 1 -- stolen -- for a
      # missing owner file and a foreign one alike, for the same reason.
      RUNTIME_LOCK_STOLEN=1
      echo "error: the dev runtime lock directory outlived its owner file" >&2
      echo "       $LOCK_DIR" >&2
      echo "       this run wrote that owner file and it is gone, so the claim" >&2
      echo "       was broken while this run still held it" >&2
      echo "       the directory standing there now cannot be shown to be the" >&2
      echo "       one this run made -- a new holder between its mkdir and its" >&2
      echo "       owner write looks exactly like this -- so nothing is removed" >&2
      echo "       the next command will refuse it; once nothing is running," >&2
      echo "       clear it by hand: rm -rf $LOCK_DIR" >&2
      return 1
      ;;
    *)
      # The listing was read and the search over it was not. "The owner file is
      # not there" is a conclusion nobody reached.
      echo "error: the dev runtime lock listing could not be searched at release" >&2
      echo "       $LOCK_DIR" >&2
      echo "       not removing a lock on an unread listing; the next command" >&2
      echo "       will refuse it" >&2
      return 1
      ;;
  esac
  owner="$(sed -n '1p' "$LOCK_OWNER_FILE")" || rc=$?
  if (( rc != 0 )); then
    echo "error: the dev runtime lock owner file could not be read at release" >&2
    echo "       $LOCK_OWNER_FILE" >&2
    echo "       an owner that cannot be read cannot be shown to be this run, so" >&2
    echo "       the lock is being left exactly as it is rather than removed on" >&2
    echo "       a guess; the next command will refuse it" >&2
    return 1
  fi
  # The TOKEN, not `$$`. A pid compares equal across a release and a retake by
  # the same process, so `$$` could ratify a lock this run no longer holds.
  if [[ "$owner" != "$RUNTIME_LOCK_TOKEN" ]]; then
    # ROUND 6. The SECOND shape of the arm above, detected one level in: the
    # directory is still standing but the claim inside it is somebody else's,
    # so this run's claim was broken here too. It already returned 1, so the
    # exit status was never the problem -- the caller's wording was, because
    # "its lock did not come off" describes a leftover this run should have
    # tidied, and what actually happened is that the lock came off and was
    # retaken. `attest.sh:lock_generation_state` answers 1 for gone AND for
    # somebody else's for the same reason: to a holder they are one event.
    RUNTIME_LOCK_STOLEN=1
    echo "error: the dev runtime lock is recorded to another acquisition, not" >&2
    echo "       to this run: it names [$owner] and this run took it as" >&2
    echo "       [$RUNTIME_LOCK_TOKEN]" >&2
    echo "       $LOCK_OWNER_FILE" >&2
    echo "       this run believed it held the lock, so something recovered it" >&2
    echo "       underneath; not removing another command's lock" >&2
    return 1
  fi
  # Statuses read rather than assumed, and ROUND 4 corrected what happens to
  # them afterwards: this used to say the trap's `|| true` swallowed this
  # function's return value on purpose. It no longer does -- the trap reads it
  # and downgrades an otherwise-`ok` run -- so the messages below are the
  # DETAIL beside a verdict, not the only trace a failure leaves.
  if ! rm -f "$LOCK_OWNER_FILE"; then
    echo "error: could not remove the dev runtime lock owner file" >&2
    echo "       $LOCK_OWNER_FILE" >&2
    echo "       the next command against this worktree will refuse the lock" >&2
    failed=1
  fi
  if ! rmdir "$LOCK_DIR" 2>/dev/null; then
    # Not fatal on its own: another process may legitimately have recovered a
    # lock this run no longer owns, and an empty directory left behind is
    # recoverable. Named, because "the lock could not be tidied" and "the lock
    # was released" were the same silent 0.
    echo "note: the dev runtime lock directory was not removed: $LOCK_DIR" >&2
    failed=1
  fi
  (( failed == 0 ))
}

# Look for the lock's owner file, waiting a bounded time for it to appear.
#
# THE RACE this exists for. `mkdir` is the atomic step and the owner record is
# written one line later, so there is a window in which the lock EXISTS and
# names nobody. A second process whose `mkdir` failed used to list the directory
# inside that window, see no owner filename, conclude STALE, remove and recreate
# the lock and write its own pid -- and then the first process, resuming,
# overwrote that pid with its own. Two runs, one lock directory, both convinced
# they held it, and the isolated port they are both about to bind is the thing
# the lock exists to keep them off.
#
# The window is microseconds wide and it is not closeable with `mkdir` as the
# primitive, so it is WAITED OUT instead: five seconds is four orders of
# magnitude more than the writer needs, and a lock that still names nobody after
# it is not a lock anybody can attribute.
#
# exit: 0 the owner file is there, 1 the lock DIRECTORY is gone (so there is
# nothing to attribute and `mkdir` decides), 2 the lock directory could not be
# read well enough to say either, 3 the lock is there and still names nobody
# after the wait. 1 and 3 were one status until round 4, and the caller does
# two different things with them.
#
# Bounded by TIME and not by a round count, which is the difference between a
# five-second wait and a fifty-second one: every round here spawns `ls`, `grep`
# and `sleep`, and process creation on Windows costs about a second, so
# `for _ in $(seq 1 50)` was a five-second wait on Linux and most of a minute on
# the platform this file exists for. The quantity that matters is wall clock --
# the writer needs microseconds -- so wall clock is what is measured.
lock_owner_file_appeared() {
  local list_rc name_rc listing deadline
  deadline=$((SECONDS + 5))
  while :; do
    list_rc=0
    listing="$(list_dir_tristate "$LOCK_DIR")" || list_rc=$?
    # The lock went away entirely while this was looking: whoever held it
    # released it. That is not "no owner", it is "no lock", and the caller
    # retries the `mkdir` rather than recovering anything.
    (( list_rc == 1 )) && return 1
    (( list_rc == 0 )) || return 2
    name_rc=0
    listing_has_name "$listing" "${LOCK_OWNER_FILE##*/}" || name_rc=$?
    case "$name_rc" in
      0) return 0 ;;
      1) ;;
      *) return 2 ;;
    esac
    (( SECONDS < deadline )) || break
    # This window is WALL CLOCK, so a delay that could not be performed would
    # not shorten it -- the loop would spin until SECONDS reached the deadline
    # and the five seconds would still elapse. It answers 2 anyway: a poll that
    # cannot pause spawns `ls` and `grep` thousands of times against a lock
    # directory instead of five dozen, which is a different measurement of the
    # same thing, and "the wait could not be performed" is already one of this
    # function's three refusals. See `poll_delay` in lib/host-process.sh.
    poll_delay 0.1 || return 2
  done
  # ROUND 4. This used to be `return 1` -- the same status the vanished-lock
  # case above returns -- so the caller could not tell "there is no lock any
  # more" from "there is a lock and it still names nobody", and refused on
  # both. The comment above promised the first would go round again and let
  # `mkdir` arbitrate; the code never did. They are separate statuses now
  # because the caller does two different things with them.
  return 3
}

# Take the worktree's runtime lock, or say why not.
# exit: 0 held (and RUNTIME_LOCK_HELD is 1), 1 not held.
#
# EVERY STATUS IN HERE IS READ, and the reason is a bash rule that made `set -e`
# decorative in this function: a function invoked on the left of `||` -- which is
# how all three call sites at the bottom of this file invoke it -- runs with
# ERREXIT DISABLED THROUGHOUT ITS BODY, and its return value is the status of
# its LAST command. So the shape this used to have,
#
#     mkdir -p "$STATE_DIR"
#     ...
#     printf '%s\n' "$$" >"$LOCK_OWNER_FILE"
#     RUNTIME_LOCK_HELD=1
#
# returned 0 when the owner write FAILED -- a full disk, an ACL, an antivirus
# holding the path -- because the assignment after it succeeded. The runtime
# then proceeded believing it held a lock whose owner record does not exist,
# which is the same window the race above opens, held open for the whole run.
# The recovery `mkdir "$LOCK_DIR"` at the end had the identical shape: a
# competing process could win it, this one carried on regardless and overwrote
# the winner's owner file.
#
# `first_failure` is the discipline that replaces errexit here: the function
# returns the status of the FIRST thing that went wrong, and nothing later can
# turn it back into a success.
acquire_runtime_lock() {
  local owner owner_pid owner_again recheck read_rc list_rc name_rc listing
  local appeared_rc token other other_rc retook=0
  if ! mkdir -p "$STATE_DIR"; then
    echo "error: could not create the dev runtime state directory: $STATE_DIR" >&2
    echo "       nothing below can be isolated without it" >&2
    return 1
  fi
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    # `owner="$(sed … 2>/dev/null || true)"` used to stand here, and it is this
    # file's defect pointed at the one piece of state that exists to keep two
    # runs apart: an owner file that could not be READ produced the same empty
    # string as an owner file that is not THERE, and the recovery below then
    # deleted a live lock and took it. So the lock directory is listed first --
    # tri-state, the same listing `read_owned_pid` stands on -- and a pid file
    # that is present but unreadable is a refusal instead of a stale lock.
    list_rc=0
    listing="$(list_dir_tristate "$LOCK_DIR")" || list_rc=$?
    if (( list_rc == 2 )); then
      echo "error: the dev runtime lock exists and could not be read: $LOCK_DIR" >&2
      echo "       not taking a lock whose owner is unknown: another command" >&2
      echo "       may be running against this worktree right now" >&2
      return 1
    fi
    name_rc=1
    if (( list_rc == 0 )); then
      name_rc=0
      listing_has_name "$listing" "${LOCK_OWNER_FILE##*/}" || name_rc=$?
    fi
    if (( name_rc == 2 )); then
      echo "error: could not tell whether the dev runtime lock names an owner" >&2
      echo "       $LOCK_DIR" >&2
      return 1
    fi
    if (( name_rc != 0 )); then
      # A lock with no owner file in it. That USED to be recovered on the spot,
      # which is the race described above: the holder may simply not have
      # written the record yet. Wait for it, and only then decide.
      appeared_rc=0
      lock_owner_file_appeared || appeared_rc=$?
      case "$appeared_rc" in
        0) name_rc=0 ;;
        # The lock itself went away while this was looking, so there is nothing
        # to recover and nothing to refuse. `mkdir` is the arbiter here, exactly
        # as it is at the top of this function -- and ROUND 4 is that this arm
        # used to fall through to the refusal below instead, which is the
        # comment describing behaviour the code did not have. One attempt, not
        # a loop: two runs releasing and retaking in turn must not be able to
        # spin this.
        1)
          if ! mkdir "$LOCK_DIR" 2>/dev/null; then
            echo "error: the dev runtime lock was released and immediately" >&2
            echo "       retaken by another command: $LOCK_DIR" >&2
            return 1
          fi
          retook=1
          ;;
        # Waited out, and still nobody. An owner that cannot be established is a
        # MEASUREMENT THAT DID NOT HAPPEN, not a measurement that said "nobody".
        # Recovering it is how two runs come to share a lock directory; refusing
        # costs a manual `rm -rf` after the one thing that produces this state,
        # which is a process killed between `mkdir` and its very next line.
        3)
          echo "error: the dev runtime lock names no owner: $LOCK_DIR" >&2
          echo "       waited for one and none appeared. An owner that could not be" >&2
          echo "       established is not an absent owner, and a lock recovered on" >&2
          echo "       that basis can be held by two commands at once" >&2
          echo "       remove it by hand once nothing is running: rm -rf $LOCK_DIR" >&2
          return 1
          ;;
        *)
          echo "error: could not read the dev runtime lock while waiting for an" >&2
          echo "       owner to appear in it: $LOCK_DIR" >&2
          echo "       not recovering a lock nobody could look at" >&2
          return 1
          ;;
      esac
    fi
    if (( name_rc == 0 )); then
      read_rc=0
      owner="$(sed -n '1p' "$LOCK_OWNER_FILE")" || read_rc=$?
      if (( read_rc != 0 )); then
        echo "error: the dev runtime lock owner file could not be read" >&2
        echo "       $LOCK_OWNER_FILE" >&2
        echo "       an unreadable owner is not an absent owner; refusing to" >&2
        echo "       recover a lock that may belong to a running command" >&2
        return 1
      fi
      # AN OWNER THAT IS NOT A PID IS NOT AN ABSENT OWNER EITHER. The liveness
      # test below is guarded by `[[ "$owner" =~ ^[0-9]+$ ]]`, so an EMPTY or
      # MALFORMED owner file used to skip it entirely and fall straight into the
      # recovery -- unconditionally, for the one file whose whole purpose is to
      # say who holds this. That contradicted the paragraph above it, which
      # claims an unparsable pid keeps the lock. The claim is right and the code
      # was wrong; this is the code.
      #
      # A torn owner file is exactly what a run killed mid-write leaves, and it
      # is also what a run that is very much alive looks like for the instant
      # between `open` and `write`.
      # The pid is the token's first field, and an owner file written before the
      # token existed is one bare pid, which parses the same way.
      owner_pid="${owner%% *}"
      if [[ ! "$owner_pid" =~ ^[0-9]+$ ]]; then
        echo "error: the dev runtime lock owner is not a pid: [$owner]" >&2
        echo "       $LOCK_OWNER_FILE" >&2
        echo "       an owner that cannot be parsed cannot be shown to be dead," >&2
        echo "       so this lock is not being recovered" >&2
        echo "       remove it by hand once nothing is running: rm -rf $LOCK_DIR" >&2
        return 1
      fi
      # `kill -0 "$owner" 2>/dev/null` stood here, and it is this file's defect
      # in the one test that decides whether a lock may be TAKEN. `kill -0`
      # exits 1 for ESRCH and for EPERM alike, and with stderr discarded there
      # is nothing left to tell them apart -- so a lock held by a live process
      # this user may not signal read as "stale", and the recovery below
      # deleted it and took the lock out from under a running command.
      #
      # The library already draws that line, from the errno text, for exactly
      # this reason; it is `kill -0` underneath, so it still answers about the
      # MSYS pid `$$` records rather than about a Windows pid. It returns 0 only
      # when the kernel said "No such process", so anything else -- alive,
      # EPERM, a pid bash refused to parse -- keeps the lock where it is. Which
      # of those it was cannot be recovered from a builtin that reports one
      # status for all three, so the message names both rather than picking.
      if ! errno_says_no_such_process "$owner_pid"; then
        echo "error: another dev runtime command is active (PID $owner_pid)" >&2
        echo "       or its liveness could not be measured; either way this" >&2
        echo "       lock is not being recovered" >&2
        return 1
      fi
      # TEST HOOK, 0 in every real run. It widens the window between the
      # measurement above and the removals below so that
      # `negative-controls/dev-runtime-lock-race-controls.sh` can drive the race
      # deterministically instead of sampling it. A widening that could not be
      # performed is not a widening, so its status is read like every other.
      if [[ "${DEV_RUNTIME_RACE_SLEEP:-0}" != "0" ]]; then
        if ! sleep "${DEV_RUNTIME_RACE_SLEEP}"; then
          echo "error: DEV_RUNTIME_RACE_SLEEP is set and the delay could not be" >&2
          echo "       performed; not proceeding into the window it widens" >&2
          return 1
        fi
      fi
      # THE OWNER IS READ AGAIN, IMMEDIATELY BEFORE ANYTHING IS REMOVED, and it
      # must still be the exact value whose liveness was measured. Between that
      # measurement and these removals the dead holder's lock can be released
      # and a LIVE run can take it -- `mkdir` succeeds the moment the directory
      # is gone -- and the removals below would then destroy that run's lock
      # while this one's `mkdir` succeeds, leaving two runs on one isolated port
      # and data directory. Measured 6/6 before this re-read, 0/6 after; see
      # negative-controls/dev-runtime-lock-race-controls.sh.
      #
      # RESIDUAL, stated: the window between this comparison and the two
      # removals is NOT closed -- `mkdir` offers nothing that tests and removes
      # in one step. `release_runtime_lock` is the other half, from the victim's
      # side: a run whose lock was broken there reports it instead of finishing
      # green.
      recheck=0
      owner_again="$(sed -n '1p' "$LOCK_OWNER_FILE")" || recheck=$?
      if (( recheck != 0 )) || [[ "$owner_again" != "$owner" ]]; then
        echo "error: the dev runtime lock changed hands while this run was" >&2
        echo "       measuring it stale: $LOCK_DIR" >&2
        echo "       it named [$owner] and now names [$owner_again], so it is" >&2
        echo "       no longer the lock that was shown to be abandoned and is" >&2
        echo "       not this run's to break" >&2
        return 1
      fi
      if ! rm -f "$LOCK_OWNER_FILE"; then
        echo "error: the stale dev runtime lock owner file could not be removed" >&2
        echo "       $LOCK_OWNER_FILE" >&2
        return 1
      fi
    fi
    # ROUND 4, and the other half of the retry the comment above promised. When
    # the wait ended because the lock DIRECTORY was gone, this run has already
    # created it new, a few lines up, and holds it. There is nothing stale left
    # to recover: the rmdir below would hand back the directory this run is
    # holding, and the mkdir after it would race every other command on the
    # worktree to take back what it had just given away.
    if (( retook == 0 )); then
      if ! rmdir "$LOCK_DIR" 2>/dev/null; then
        echo "error: stale dev runtime lock could not be recovered: $LOCK_DIR" >&2
        return 1
      fi
      # CHECKED, and this is the second half of the same finding as the owner
      # write below. A recovery `mkdir` that FAILS means another process won the
      # race to recreate the lock in the instant after the `rmdir` above -- and
      # under a suspended errexit this carried straight on, wrote its own pid over
      # the winner's owner file and returned success.
      if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "error: the dev runtime lock was retaken while it was being" >&2
        echo "       recovered: $LOCK_DIR" >&2
        echo "       another command against this worktree got there first" >&2
        return 1
      fi
    fi
  fi
  # THE WRITE THIS FUNCTION EXISTS FOR, and the status that used to be dropped.
  # Everything downstream -- `release_runtime_lock`, and the next command's
  # decision about whether this lock is live -- reads this file. A lock held
  # with no owner record in it is the state the whole recovery path above cannot
  # attribute, so failing to write it must not be a lock that is reported as
  # held. The directory is given back, because a lock this run does not own is
  # not a lock this run should leave standing.
  #
  # NOCLOBBER. `set -C` makes `>` a CREATE-OR-FAIL act, so this run cannot write
  # its name over an owner file that is already there -- which is exactly the
  # state left behind when the directory this run created was broken by a waiter
  # and recreated and claimed by a third run. Under a plain `>` that write
  # succeeded, over the new holder's record, and both runs went on believing
  # they held the lock. `set -C` is scoped to the subshell, whose status is the
  # redirection's, so nothing else in this file acquires noclobber semantics.
  token="$(lock_new_token)"
  if ! ( set -C; printf '%s\n' "$token" >"$LOCK_OWNER_FILE" ) 2>/dev/null; then
    other_rc=0
    other="$(sed -n '1p' "$LOCK_OWNER_FILE" 2>/dev/null)" || other_rc=$?
    if (( other_rc == 0 )) && [[ -n "$other" ]]; then
      # An owner that is not ours: the directory this run created is gone and
      # this one belongs to another command. Removing it here is precisely the
      # destruction the re-read above refuses to commit.
      echo "error: the dev runtime lock was taken by another command between" >&2
      echo "       this run's mkdir and its owner write: $LOCK_DIR" >&2
      echo "       it is recorded to [$other]; not touching another command's" >&2
      echo "       lock" >&2
      return 1
    fi
    echo "error: could not record this command as the dev runtime lock owner" >&2
    echo "       $LOCK_OWNER_FILE" >&2
    echo "       a lock whose owner was never written cannot be attributed, and" >&2
    echo "       the next command would have to refuse it or steal it" >&2
    rm -f "$LOCK_OWNER_FILE" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
    return 1
  fi
  RUNTIME_LOCK_TOKEN="$token"
  # The EXIT trap is installed once, at the top of the file, so that a guard
  # that refuses before this is ever called still emits its outcome marker.
  # This only tells it there is now a lock to release. LAST, and only once
  # everything above has been shown to have worked: it is the flag that makes
  # the handler act, and setting it early is how a run comes to release a lock
  # it never took.
  RUNTIME_LOCK_HELD=1
}

# The ownership record is what makes a daemon attributable. Deleting it while
# the daemon may still be running is the worst outcome available here: nothing
# can identity-check that process again, so a survivor becomes permanently
# unattributable. It is therefore removed only on a MEASURED exit, never on an
# unmeasured one.
#
# exit: 0 the record is gone, 1 some of it could not be removed. READ by every
# caller, because `stop_runtime` runs with errexit suspended -- it is invoked as
# `stop_runtime || true` from `start_runtime` -- so a failing `rm` here used to
# be followed by a stop receipt and a `return 0`. The record then still names a
# daemon, and the receipt said it had been cleaned up.
clear_owned_state() {
  rm -f "$PID_FILE" "$SERVER_PATH_FILE" "$PORT_FILE" "$DATA_DIR_FILE"
}

# The three `stop` receipts all follow a successful `clear_owned_state`, and
# each of them used to follow it whether or not it worked. One helper, so the
# report and the removal cannot come apart again.
# exit: 0 cleared, 1 not.
report_unremovable_record() {
  echo "error: the dev daemon was stopped and its ownership record was not removed" >&2
  echo "       $STATE_DIR" >&2
  echo "       the next start will read a record naming a process that is gone" >&2
  echo "       remove it by hand: rm -f $PID_FILE $SERVER_PATH_FILE $PORT_FILE $DATA_DIR_FILE" >&2
}

# Poll for the recorded pid to exit.
# exit: 0 measured gone, 1 still alive when the window closed, 2 could not
# measure -- and the caller must not print a stop receipt for 1 or 2.
wait_for_owned_exit() {
  local _attempt
  for (( _attempt = 0; _attempt < 50; _attempt++ )); do
    probe_process_alive "$OWNED_PID"
    case "$PROCESS_ALIVE_STATE" in
      gone) return 0 ;;
      unmeasured) return 2 ;;
    esac
    # ROUND 5. A bare `sleep 0.1` stood here, and this function is invoked from
    # the left of a `||` at all four of its call sites -- errexit off through
    # the whole body -- so a `sleep` that failed left the loop running: fifty
    # liveness probes back to back in microseconds, then `return 1`, "still
    # alive when the window closed". The caller's response to that 1 is to
    # escalate to a FORCE KILL of the daemon, five seconds early, on a window
    # nobody waited through. See `poll_delay` in lib/host-process.sh.
    poll_delay 0.1 || return 2
  done
  return 1
}

report_unmeasured_liveness() {
  echo "error: could not measure whether PID $OWNED_PID is still running" >&2
  echo "       keeping the ownership record so the process stays attributable" >&2
  echo "       nothing was removed; re-run once tasklist/ps can be reached" >&2
}

report_unreadable_record() {
  echo "error: a dev daemon ownership record exists but could not be read" >&2
  echo "       $STATE_DIR" >&2
  echo "       not reporting 'nothing is recorded' and not removing anything:" >&2
  echo "       a torn or unreadable record is not evidence that no daemon runs" >&2
}

# `stop` has three distinguishable outcomes and they map onto the same kinds as
# `start`: it stopped what it was asked to (`ok`), it refused to kill something
# it could not prove was its own (`safety-refusal` -- the guard, wearing its
# other face), or it could not measure enough to say either (`unknown`).
#
# RESULT_KIND is set LAST at every return, immediately before it, and never
# earlier: `start_runtime` calls this function on its own failure path, and a
# kind written on the way through would be the one the run reported.
stop_runtime() {
  local owned_rc=0 wait_rc=0 stop_rc=0 read_rc=0
  read_owned_pid || read_rc=$?
  if (( read_rc == 2 )); then
    report_unreadable_record
    RESULT_KIND=unknown
    return 1
  fi
  if (( read_rc != 0 )); then
    echo "No worktree-owned Wenlan dev daemon is recorded."
    RESULT_KIND=ok
    return 0
  fi
  probe_process_alive "$OWNED_PID"
  case "$PROCESS_ALIVE_STATE" in
    gone)
      if ! clear_owned_state; then
        report_unremovable_record
        RESULT_KIND=unknown
        return 1
      fi
      echo "Removed stale Wenlan dev daemon state."
      RESULT_KIND=ok
      return 0
      ;;
    unmeasured)
      report_unmeasured_liveness
      RESULT_KIND=unknown
      return 1
      ;;
  esac

  is_owned_process || owned_rc=$?
  case "$owned_rc" in
    0) ;;
    2)
      echo "error: ownership of PID $OWNED_PID could not be measured; refusing to stop it" >&2
      echo "       an indeterminate identity is not permission to kill a process" >&2
      RESULT_KIND=unknown
      return 1
      ;;
    3)
      # The same refusal as before -- a stop kills only a process proven to be
      # ours AND serving the recorded port -- but no longer described as
      # something it is not. This branch used to share the message below, which
      # says the process is not $OWNED_SERVER; here it demonstrably IS, and it
      # is the PORT that does not answer for it. A daemon still opening its
      # socket looks exactly like this, and so does one whose port was taken.
      echo "error: refusing to stop PID $OWNED_PID: it is running $OWNED_SERVER, but" >&2
      if [[ "$LISTENER_PROBE_STATE" == found ]]; then
        echo "       port $OWNED_PORT is held by PID $LISTENER_PROBE_PID, not by it" >&2
      else
        echo "       nothing is listening on port $OWNED_PORT yet" >&2
      fi
      echo "       a process that cannot be shown to be serving the recorded port" >&2
      echo "       is not one this runtime may kill; re-run once it has bound it" >&2
      RESULT_KIND=safety-refusal
      return 1
      ;;
    *)
      echo "error: refusing to stop PID $OWNED_PID because it is not $OWNED_SERVER" >&2
      RESULT_KIND=safety-refusal
      return 1
      ;;
  esac

  # The stop helper's status is a diagnostic; the verdict below is the liveness
  # poll, because a stop is proven by the process being gone and never by a
  # receipt. Captured rather than run bare: on POSIX `kill` legitimately fails
  # when the process exits first, and under `set -e` that would abort silently.
  terminate_process "$OWNED_PID" "$OWNED_SERVER" || stop_rc=$?
  wait_for_owned_exit || wait_rc=$?
  case "$wait_rc" in
    0)
      if ! clear_owned_state; then
        report_unremovable_record
        RESULT_KIND=unknown
        return 1
      fi
      echo "Stopped worktree-owned Wenlan dev daemon (PID $OWNED_PID)."
      RESULT_KIND=ok
      return 0
      ;;
    2)
      report_unmeasured_liveness
      RESULT_KIND=unknown
      return 1
      ;;
  esac

  owned_rc=0
  has_owned_command_identity || owned_rc=$?
  case "$owned_rc" in
    0) force_terminate_process "$OWNED_PID" "$OWNED_SERVER" || stop_rc=$? ;;
    2)
      echo "error: identity of PID $OWNED_PID could not be measured after the stop" >&2
      echo "       not force-killing a process whose identity is indeterminate" >&2
      RESULT_KIND=unknown
      return 1
      ;;
  esac
  wait_rc=0
  wait_for_owned_exit || wait_rc=$?
  case "$wait_rc" in
    0)
      if ! clear_owned_state; then
        report_unremovable_record
        RESULT_KIND=unknown
        return 1
      fi
      echo "Force-stopped unresponsive worktree-owned Wenlan dev daemon (PID $OWNED_PID)."
      RESULT_KIND=ok
      return 0
      ;;
    2)
      report_unmeasured_liveness
      RESULT_KIND=unknown
      return 1
      ;;
  esac
  echo "error: worktree-owned Wenlan dev daemon PID $OWNED_PID did not exit" >&2
  echo "       last stop helper status: $stop_rc" >&2
  RESULT_KIND=unknown
  return 1
}

# Every return below sets RESULT_KIND immediately before it, and nothing sets it
# earlier: `stop_runtime` is called from the health path and writes its own kind,
# so a kind written on the way in would be overwritten by the cleanup rather than
# by the outcome. See THE OUTCOME CONTRACT at the top of this file.
start_runtime() {
  local backend build_dir server pid job_pid stage owned_rc resolve_rc reap_rc
  local health_detail health_kind read_rc cleanup_rc cleanup_kind record_rc
  STARTED_RUNTIME=0
  # Both of these used to abort the script through `set -e`, with the marker
  # left at its default -- which is `unknown`, which is the honest answer, but
  # with no line naming what failed.
  DEV_DATA_DIR="$(canonicalize_path "$DEV_DATA_DIR")" || {
    echo "error: could not resolve WENLAN_DEV_DATA_DIR: $DEV_DATA_DIR" >&2
    RESULT_KIND=unknown
    return 1
  }
  backend="$(bash "$SCRIPT_DIR/resolve-backend-dir.sh" "$REPO_ROOT")" || {
    echo "error: could not resolve the backend checkout to build from" >&2
    RESULT_KIND=build-failure
    return 1
  }
  # The same target-root resolution prepare-sidecars.sh uses. A short
  # CARGO_TARGET_DIR is how a Windows checkout stays under MAX_PATH, and reading
  # the built binary from the wrong root either fails to launch or, worse,
  # launches a stale one.
  build_dir="${CARGO_TARGET_DIR:-$backend/target}/debug"
  server="$build_dir/wenlan-server"
  if (( HOST_IS_WINDOWS == 1 )); then
    # Spelled natively so the recorded identity is one string no matter whether
    # this run inherited WENLAN_DEV_STATE_DIR from `print-config` or derived it.
    stage="$(native_path "$DAEMON_STAGE_DIR")" || {
      echo "error: could not spell $DAEMON_STAGE_DIR natively; refusing to start" >&2
      RESULT_KIND=unknown
      return 1
    }
    server="$stage/wenlan-server.exe"
  fi

  read_rc=0
  read_owned_pid || read_rc=$?
  if (( read_rc == 2 )); then
    # Falling through here is what deleted the record. See read_owned_pid.
    report_unreadable_record
    echo "       refusing to start a second daemon on an unread record" >&2
    RESULT_KIND=unknown
    return 1
  fi
  if (( read_rc == 0 )); then
    owned_rc=0
    is_owned_process || owned_rc=$?
    if (( owned_rc == 2 )); then
      # The old form was `read_owned_pid && is_owned_process`, which folded an
      # indeterminate answer into "not ours" and fell straight through to the
      # `rm -f` below -- deleting the ownership record of a possibly-running
      # daemon, and then treating the port it holds as free.
      echo "error: could not measure whether the recorded dev daemon (PID $OWNED_PID) is still ours" >&2
      echo "       refusing to start a second daemon or to delete the ownership record" >&2
      RESULT_KIND=unknown
      return 1
    fi
    if (( owned_rc == 3 )); then
      # ROUND 5, and it is the round-4 fix above one state further in. 2 was
      # handled and 1 fell through to `clear_owned_state`; 3 -- the recorded pid
      # ALIVE, running the recorded executable, and not yet the listener -- was
      # part of that 1. A launcher killed between recording the pid and the
      # daemon binding its socket leaves exactly this, and the fall-through
      # deleted a live daemon's ownership record and launched a second daemon
      # against the same data directory. Two writers, one SQLite file, and the
      # `stop` that could have ended either of them can no longer find the first.
      echo "error: the recorded dev daemon (PID $OWNED_PID) is running $OWNED_SERVER," >&2
      if [[ "$LISTENER_PROBE_STATE" == found ]]; then
        echo "       but port $OWNED_PORT is held by PID $LISTENER_PROBE_PID" >&2
      else
        echo "       but nothing is listening on port $OWNED_PORT yet" >&2
      fi
      echo "       a live daemon's record is not a stale one: deleting it and" >&2
      echo "       starting again would put a second daemon on the same data" >&2
      echo "       directory with nothing able to stop the first" >&2
      echo "       if it is still starting, retry; otherwise stop it by hand:" >&2
      echo "       kill $OWNED_PID" >&2
      RESULT_KIND=unknown
      return 1
    fi
    if (( owned_rc == 0 )); then
      if [[ "$OWNED_SERVER" != "$server" || "$OWNED_PORT" != "$DEV_PORT" ||
        "$OWNED_DATA_DIR" != "$DEV_DATA_DIR" ]]; then
        echo "error: recorded dev daemon identity does not match this runtime configuration" >&2
        echo "recorded: server=$OWNED_SERVER port=$OWNED_PORT data=$OWNED_DATA_DIR" >&2
        echo "selected: server=$server port=$DEV_PORT data=$DEV_DATA_DIR" >&2
        RESULT_KIND=unknown
        return 1
      fi
      print_config
      echo "Wenlan dev daemon is already running (PID $OWNED_PID)."
      RESULT_KIND=ok
      return 0
    fi
  fi

  # Named rather than left to errexit. `start_runtime` IS called plainly, so an
  # abort here would at least be an abort -- but it would be one with no line
  # saying which directory could not be made and a marker of `unknown` that
  # nothing explains.
  if ! mkdir -p "$STATE_DIR" "$DEV_DATA_DIR"; then
    echo "error: could not create the dev runtime directories" >&2
    echo "       $STATE_DIR" >&2
    echo "       $DEV_DATA_DIR" >&2
    RESULT_KIND=unknown
    return 1
  fi
  if ! clear_owned_state; then
    echo "error: the previous dev daemon ownership record could not be removed" >&2
    echo "       $STATE_DIR" >&2
    echo "       starting now would leave a record naming a daemon this run did" >&2
    echo "       not launch, which is the record every stop reads" >&2
    RESULT_KIND=unknown
    return 1
  fi

  # The single most load-bearing check in this file: it is what stops a
  # worktree daemon from colliding with another one. It used to fail OPEN --
  # `[[ -n "$(listener_pid_for_port …)" ]]` read an unmeasurable port as free
  # and started a daemon on a port nobody had looked at.
  probe_listener_port "$DEV_PORT"
  case "$LISTENER_PROBE_STATE" in
    found)
      echo "error: isolated dev port $DEV_PORT is already in use (PID $LISTENER_PROBE_PID); set WENLAN_DEV_PORT" >&2
      RESULT_KIND=port-conflict
      return 1
      ;;
    unmeasured)
      echo "error: could not measure whether isolated dev port $DEV_PORT is free" >&2
      echo "       refusing to start: an unmeasured port is not a free port" >&2
      echo "       needs netstat on Windows, or lsof on macOS/Linux" >&2
      # NOT `port-conflict`. A conflict is something that was measured; this is
      # the refusal that happens because nothing was. The consumer treats
      # `unknown` as a refusal, which is the same handling and the true reason.
      RESULT_KIND=unknown
      return 1
      ;;
  esac

  cargo build --manifest-path "$backend/Cargo.toml" -p wenlan-server || {
    echo "error: the dev daemon did not build" >&2
    RESULT_KIND=build-failure
    return 1
  }
  # THE CALL SITE. Everything `stage_windows_daemon` promises about what the
  # recorded server path points at is worth nothing if this line is not here or
  # its failure is not propagated -- and no extracted-function test can see
  # that, because they all call the function themselves. So it has a case of its
  # own in scripts/negative-controls/dev-runtime-stage-controls.sh and an
  # assertion in scripts/dev-runtime.test.ts, both of which read THIS text.
  #
  # `staging-failure` and not `build-failure`: cargo has already SUCCEEDED by
  # this line. The failure that actually happens here is Windows error 32 -- a
  # daemon running out of the stage directory holding onnxruntime.dll open --
  # and a caller that reads `build-failure` retries the build, which changes
  # nothing, or reports a compiler problem, which there is not one of. The
  # condition that is blocking it, a held stage, was not in the marker at all.
  if (( HOST_IS_WINDOWS == 1 )); then
    stage_windows_daemon "$build_dir/wenlan-server.exe" || {
      RESULT_KIND=staging-failure
      return 1
    }
  fi
  nohup env WENLAN_PORT="$DEV_PORT" WENLAN_DATA_DIR="$DEV_DATA_DIR" \
    "$server" </dev/null >"$SERVER_LOG" 2>&1 &
  pid=$!
  job_pid=$pid
  if (( HOST_IS_WINDOWS == 1 )); then
    # `$!` is the MSYS pid. Every Windows listener and process table reports the
    # Windows one, so that is the identity worth recording and comparing.
    resolve_rc=0
    pid="$(windows_pid_for_job "$job_pid" "$server")" || resolve_rc=$?
    if (( resolve_rc != 0 )); then
      # Nothing downstream can identify this daemon, so it cannot be left
      # running. The MSYS pid is not the handle to reach for: the resolution
      # above waits up to ten seconds, MSYS recycles pids, and by this branch
      # the job may already be gone, so signalling it can land on an unrelated
      # process. Reap by image identity instead, which needs neither the pid nor
      # the port to have settled.
      reap_rc=0
      reap_staged_daemon "$server" || reap_rc=$?
      if (( reap_rc != 0 )); then
        # Nothing was recorded, so `stop` has no pid to read and cannot clean
        # this up. Name the executable so the port can be freed by hand.
        echo "warning: a daemon may still be starting from $server" >&2
        if (( reap_rc == 2 )); then
          echo "         the reap could not be MEASURED, so it is not a reap" >&2
        fi
        echo "         nothing recorded it; end that process before retrying" >&2
      fi
      if (( resolve_rc == 2 )); then
        echo "error: the Windows pid of the dev daemon could not be measured" >&2
      else
        echo "error: could not resolve the Windows pid of the dev daemon" >&2
      fi
      tail -n 40 "$SERVER_LOG" >&2 || true
      RESULT_KIND=unknown
      return 1
    fi
  fi
  # THE OWNERSHIP RECORD, and the four writes that make the daemon above
  # attributable. Read as measurements, because a daemon that is running and
  # unrecorded is the worst outcome this file has a name for: `stop` has no pid
  # to look up, the next `start` sees a free record and launches a second one,
  # and the port is held by something nothing can identity-check. Under errexit
  # a failing write aborted the script with no line naming which one -- the
  # right direction, reported as nothing at all.
  #
  # A partial record is not left behind. `read_owned_pid` would call it torn and
  # answer 2, which is the refusal that keeps a later run from touching it, but
  # this run knows more than that: it knows the pid, so it says it.
  record_rc=0
  printf '%s\n' "$pid" >"$PID_FILE" || record_rc=1
  printf '%s\n' "$server" >"$SERVER_PATH_FILE" || record_rc=1
  printf '%s\n' "$DEV_PORT" >"$PORT_FILE" || record_rc=1
  printf '%s\n' "$DEV_DATA_DIR" >"$DATA_DIR_FILE" || record_rc=1
  if (( record_rc != 0 )); then
    echo "error: the dev daemon started and its ownership record could not be written" >&2
    echo "       $STATE_DIR" >&2
    echo "       PID $pid is running $server on port $DEV_PORT and nothing" >&2
    echo "       records it, so no later stop can find it; end that process" >&2
    echo "       before retrying" >&2
    if ! clear_owned_state; then
      echo "       and the half-written record could not be removed either" >&2
    fi
    RESULT_KIND=unknown
    return 1
  fi

  health_detail="the daemon never answered /api/health"
  # The kind travels with the detail, set at the same four places, because two
  # of those four are "could not measure" and the other two are a daemon that
  # measurably did not come up. Collapsing them here would put the workstream's
  # own defect into the marker that exists to carry the distinction outward.
  health_kind=health-failure
  for (( _attempt = 0; _attempt < 50; _attempt++ )); do
    if curl --fail --silent --max-time 1 \
      "http://127.0.0.1:$DEV_PORT/api/health" >/dev/null 2>&1; then
      # Health alone proves only that SOMETHING answers on the port. The claim
      # being made is that the daemon this run spawned owns it, so both halves
      # are measured, and an unmeasurable half is a refusal to claim ownership
      # rather than a quiet mismatch.
      probe_process_alive "$pid"
      probe_listener_port "$DEV_PORT"
      if [[ "$PROCESS_ALIVE_STATE" == alive && "$LISTENER_PROBE_STATE" == found ]] &&
        [[ "$LISTENER_PROBE_PID" == "$pid" ]]; then
        print_config
        echo "Started worktree-owned Wenlan dev daemon (PID $pid)."
        STARTED_RUNTIME=1
        RESULT_KIND=ok
        return 0
      fi
      if [[ "$PROCESS_ALIVE_STATE" == unmeasured || "$LISTENER_PROBE_STATE" == unmeasured ]]; then
        health_detail="ownership of port $DEV_PORT could not be measured (liveness=$PROCESS_ALIVE_STATE listener=$LISTENER_PROBE_STATE)"
        health_kind=unknown
      else
        health_detail="port $DEV_PORT answered health but is held by ${LISTENER_PROBE_PID:-no listener}, not PID $pid"
        health_kind=port-conflict
      fi
      break
    fi
    probe_process_alive "$pid"
    if [[ "$PROCESS_ALIVE_STATE" == gone ]]; then
      health_detail="the daemon process exited before becoming healthy"
      health_kind=health-failure
      break
    fi
    if [[ "$PROCESS_ALIVE_STATE" == unmeasured ]]; then
      health_detail="liveness of PID $pid could not be measured while waiting for health"
      health_kind=unknown
      break
    fi
    # ROUND 5, and this one decides a KIND. `health-failure` says "a daemon was
    # started and never became healthy on its port", which is a claim about ten
    # seconds of asking; with a bare `sleep 0.2` whose status nothing read, a
    # failed delay ran all fifty curls back to back and produced that verdict in
    # milliseconds, for a daemon that had not finished opening its socket. The
    # kind a consumer may retry, off a window that did not happen.
    if ! poll_delay 0.2; then
      health_detail="the wait between health probes could not be performed, so the daemon was never given the window"
      health_kind=unknown
      break
    fi
  done

  tail -n 40 "$SERVER_LOG" >&2 || true
  # THE CLEANUP, and its status is the outcome when it is the worse of the two.
  #
  # `stop_runtime || true` stood here, and it threw away the one thing this
  # branch cannot recover from. `stop_runtime` runs with errexit suspended --
  # that is what `|| true` does to it -- so its own return value is already the
  # weaker signal; `|| true` then discarded even that, and `RESULT_KIND` was
  # overwritten with `$health_kind` one line later, erasing the `unknown` or
  # `safety-refusal` the stop had just written. A daemon that could not be
  # cleaned up and may STILL BE RUNNING on the isolated port was reported to the
  # outer caller as an ordinary `health-failure`, which is a retryable
  # condition -- so the retry starts a second daemon behind the first.
  #
  # The health verdict is the SECOND question here. A cleanup that failed
  # outranks it: `unknown` is a refusal a consumer must not act through, and
  # `safety-refusal` says this runtime declined to kill something it could not
  # prove was its own. Either is the outcome; `health-failure` is only the
  # outcome when there is nothing left running to report.
  cleanup_rc=0
  stop_runtime || cleanup_rc=$?
  cleanup_kind="$RESULT_KIND"
  echo "error: Wenlan dev daemon did not become healthy on port $DEV_PORT" >&2
  echo "       $health_detail" >&2
  if (( cleanup_rc != 0 )); then
    echo "error: and the daemon could not be cleaned up: it may still be running" >&2
    echo "       on port $DEV_PORT (cleanup outcome: $cleanup_kind)" >&2
    echo "       reporting the cleanup rather than the health check: a second" >&2
    echo "       start against this port would collide with what is still there" >&2
    RESULT_KIND="$cleanup_kind"
    return 1
  fi
  # After `stop_runtime`, never before: that call writes its own kind, and the
  # outcome being reported is this one.
  RESULT_KIND="$health_kind"
  return 1
}

# Each arm ends by naming its outcome, and the EXIT trap prints it. The lock
# acquisition is captured rather than run bare so that "another command is
# active" is a refusal with a kind of its own instead of a `set -e` abort with
# whatever the default happened to be.
case "${1:-}" in
  print-config)
    print_config || { RESULT_KIND=unknown; exit 1; }
    RESULT_KIND=ok
    ;;
  start)
    acquire_runtime_lock || { RESULT_KIND=unknown; exit 1; }
    start_runtime
    ;;
  start-for-session)
    acquire_runtime_lock || { RESULT_KIND=unknown; exit 1; }
    start_runtime
    if (( STARTED_RUNTIME == 0 )); then
      # Not a failure: a healthy worktree daemon was already running and this
      # run reuses it. `dev-all.sh` reads the 10 and leaves it alone, so the
      # kind stays whatever `start_runtime` measured, which is `ok`.
      exit 10
    fi
    ;;
  stop)
    acquire_runtime_lock || { RESULT_KIND=unknown; exit 1; }
    stop_runtime
    ;;
  *)
    # A usage error is not a safety event and not a failed measurement of
    # anything. `unknown` is the honest kind, and the consumer refuses on it.
    echo "usage: $0 {print-config|start|start-for-session|stop}" >&2
    RESULT_KIND=unknown
    exit 2
    ;;
esac
