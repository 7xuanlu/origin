#!/usr/bin/env bash
# Cases and controls for `stage_windows_daemon`, the dev daemon's private copy.
#
# Round 13g. Staging was `cp -u`, and `cp -u` compares TIMESTAMPS while the
# claim being made about the staged file is about BYTES. The two come apart in
# ordinary ways -- an executable restored from a backup, a clock that went
# backwards, an artifact copied rather than built -- and when they do, cargo
# succeeds, `cp -u` keeps the older bytes, `/api/health` is answered by a daemon
# built from other source, and the run reports current-source provenance.
# Nothing downstream can notice: the recorded path, the pid and the port are all
# correct, and every probe in lib/host-process.sh agrees. The DLL loop carried
# the same comparison with `|| true` on top, which is os error 32's own failure
# mode -- a library held open by a process running out of the stage directory,
# left at its old version beside a new daemon, reported as a clean start.
#
# The three functions are EXTRACTED from the shipped script and sourced, because
# dev-runtime.sh dispatches on "$1" at top level and cannot be sourced.
#
# NOTHING about the staging is stubbed: real files, real digests, a real `cp`.
# Two cases replace `cp` with a shim, and each says which failure it is
# reproducing -- a copy that exits 0 having written other bytes, and a DLL that
# cannot be written because something holds it open. Those two cannot be
# produced on demand any other way.
#
# Round 13h added two things this could not previously see.
#
# The first is the DIRECTORY. `for dll in "$dir"/*.dll; do [[ -f "$dll" ]] ||
# continue` has two answers and the question has three: a directory that cannot
# be listed leaves the pattern unexpanded, `-f` rejects the literal `*.dll`, the
# loop body never runs, and staging returns SUCCESS having copied no libraries
# -- indistinguishable from the fresh checkout where there genuinely are none.
# So the listing is measured tri-state, a zero-library stage is a NAMED outcome,
# and a prepared `app/binaries` that has lost the libraries it must carry is an
# error. `nc-stage-globs-the-library-directory` puts the two-state shape back.
#
# The second is the CALL SITE. Every case here drives `stage_windows_daemon`
# directly, which is how a fix can be complete and unreached at the same time:
# none of them would notice the call being deleted from `start_runtime`, or its
# status swallowed with `|| true`. `start-calls-staging` reads the shipped call
# site out of the script, and two controls break it.
#
# What this proves: that the copy is decided and then VERIFIED by content, that
# no copy failure is swallowed, that the library directory is measured rather
# than globbed, and that `start_runtime` calls this and stops when it fails.
# What it does NOT prove: that a Tauri build then finds what it staged -- that
# is its own code, read but not executed here.
#
# Run: bash scripts/negative-controls/dev-runtime-stage-controls.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="$root/scripts/dev-runtime.sh"
logs="$root/target/negative-control-logs"
mkdir -p "$logs"

runtime_before="$(cat "$runtime")"

work="$(mktemp -d "${TMPDIR:-/tmp}/dev-runtime-stage-XXXXXX")" || exit 1
trap 'rm -rf "$work"' EXIT
failures=0

# --- the host facts every fixture stands on ----------------------------------
# A control whose "stale" fixture is not actually stale tests nothing, and would
# test nothing SILENTLY: the timestamp control would report "caught" for the
# wrong reason, or "survived" for one. Measure, refuse otherwise.
probe="$work/probe"
mkdir -p "$probe"
printf 'old\n' >"$probe/older"
printf 'new\n' >"$probe/newer"
touch -d '2030-01-01' "$probe/newer" 2>/dev/null ||
  touch -t 203001010000 "$probe/newer" 2>/dev/null || true
if ! [[ "$probe/newer" -nt "$probe/older" ]]; then
  echo "FATAL: mtimes cannot be moved on this filesystem; the cp -u control tests nothing" >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  echo "FATAL: no sha256 tool here, so every case would answer 'could not hash'" >&2
  exit 1
fi

# --- extract the functions, by brace matching, from the shipped script -------
extract() { # source-text, name -> that function, verbatim
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

# How many times a literal string occurs in another. `${a%%"$b"*}` answers "at
# least once" and nothing more, and "at least once" is not what an anchor needs
# to be true: an anchor that matches TWICE mutates the first occurrence and
# leaves the second, so the control runs against a subject that is neither the
# shipped code nor the reverted code, and reports on it either way.
count_occurrences() { # haystack, needle -> the count on stdout
  local rest="$1" needle="$2" n=0
  [[ -n "$needle" ]] || { printf '0'; return; }
  while [[ "$rest" == *"$needle"* ]]; do
    n=$((n + 1))
    rest="${rest#*"$needle"}"
  done
  printf '%s' "$n"
}

# The shipped call site, read out of `start_runtime`. Not driven -- reaching it
# needs a cargo build, a port and a daemon -- but READ, because the thing that
# has to be true of it is structural: it is called, only on the Windows path,
# and its failure ends the start rather than being logged past.
assert_call_site() { # start_runtime text -> 0 intact, 1 with a reason on stdout
  local text="$1" body calls near
  # Comments stripped: the prose above the call names the function too, and a
  # control that deletes the call must not be excused by the paragraph that
  # explains it.
  body="$(grep -v '^[[:space:]]*#' <<<"$text")"
  calls="$(grep -c '^[[:space:]]*stage_windows_daemon[[:space:]]' <<<"$body")"
  if [[ "$calls" != 1 ]]; then
    printf 'start_runtime calls stage_windows_daemon %s times, wanted exactly 1' "$calls"
    return 1
  fi
  near="$(grep -B2 '^[[:space:]]*stage_windows_daemon[[:space:]]' <<<"$body")"
  if ! grep -q 'HOST_IS_WINDOWS == 1' <<<"$near"; then
    printf 'the staging call is not guarded by the Windows branch'
    return 1
  fi
  near="$(grep -A3 '^[[:space:]]*stage_windows_daemon[[:space:]]' <<<"$body")"
  if grep -q '||[[:space:]]*true' <<<"$near"; then
    printf 'the staging call swallows its own status with || true'
    return 1
  fi
  if ! grep -q 'return 1' <<<"$near"; then
    printf 'a staging failure does not stop the start'
    return 1
  fi
  if ! grep -q 'RESULT_KIND=staging-failure' <<<"$near"; then
    printf 'a staging failure is not classified for the outcome marker'
    return 1
  fi
  return 0
}

build_subject() { # hasher-text, stager-text, daemon-text, start-text -> driver
  # An extraction that silently produced nothing would make every case fail with
  # "command not found" -- loud, but for the wrong reason, and the failure would
  # look like the subject's. Refuse instead.
  if [[ -z "$1" || "$1" != *"sha256"* || "${1: -1}" != "}" ]]; then
    echo "FATAL: file_sha256 could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  if [[ -z "$2" || "$2" != *"cp "* || "${2: -1}" != "}" ]]; then
    echo "FATAL: stage_file_by_identity could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  if [[ -z "$3" || "$3" != *"DAEMON_STAGE_DIR"* || "${3: -1}" != "}" ]]; then
    echo "FATAL: stage_windows_daemon could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  if [[ -z "$4" || "$4" != *"cargo build"* || "${4: -1}" != "}" ]]; then
    echo "FATAL: start_runtime could not be extracted from dev-runtime.sh" >&2
    exit 1
  fi
  # The call site is not driven, so it travels beside the driver rather than in
  # it, and `start-calls-staging` reads THIS -- which a control has mutated
  # exactly as it mutates the three functions below.
  SUBJECT_START="$4"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    # Git for Windows puts /usr/bin at the front of PATH before this script
    # runs, so a shim directory handed in from outside can lose to /usr/bin/cp.
    printf 'if [ -n "${WENLAN_TEST_SHIM_DIR:-}" ]; then\n'
    printf '  shim_dir="$WENLAN_TEST_SHIM_DIR"\n'
    printf '  if command -v cygpath >/dev/null 2>&1; then shim_dir="$(cygpath -u "$shim_dir")"; fi\n'
    printf '  PATH="$shim_dir:$PATH"; export PATH\n'
    printf 'fi\n'
    printf 'REPO_ROOT="$WENLAN_TEST_REPO_ROOT"\n'
    printf 'DAEMON_STAGE_DIR="$WENLAN_TEST_STAGE_DIR"\n'
    printf '%s\n' "$1"
    printf '%s\n' "$2"
    # The directory measurement stage_windows_daemon now stands on, extracted
    # unmutated: its own controls live in dev-runtime-record-controls.sh, and
    # here it is part of the subject rather than part of the test.
    printf '%s\n' "$NAMER"
    printf '%s\n' "$LISTER"
    printf '%s\n' "$3"
    printf 'rc=0\n'
    printf 'stage_windows_daemon "$WENLAN_TEST_SOURCE" || rc=$?\n'
    printf 'printf "rc=%%s\\n" "$rc"\n'
  } >"$work/driver.sh"
}

# --- fixtures ----------------------------------------------------------------
BUILT='the daemon this run built'
OTHER='a daemon from some other build'
LIB_BUILT='the library it was built against'
LIB_OTHER='the library from some other build'

# A `cp` that hands the real one everything it is not asked to sabotage, so a
# case breaks exactly one copy and the rest of the staging runs.
real_cp='
for real in /usr/bin/cp /bin/cp; do
  if [ -x "$real" ]; then exec "$real" "$@"; fi
done
echo "no real cp on this host" >&2
exit 127'
last_arg='
last=""
for a in "$@"; do last="$a"; done'

fixture() { # kind, base -> builds the tree, prints the cp shim body (or empty)
  local kind="$1" b="$2"
  mkdir -p "$b/repo/app/binaries" "$b/stage" "$b/build"
  case "$kind" in
    stale-daemon)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$OTHER" >"$b/stage/wenlan-server.exe"
      touch -d '2030-01-01' "$b/stage/wenlan-server.exe" 2>/dev/null ||
        touch -t 203001010000 "$b/stage/wenlan-server.exe"
      ;;
    stale-library)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$LIB_BUILT" >"$b/build/onnxruntime.dll"
      printf '%s\n' "$LIB_OTHER" >"$b/stage/onnxruntime.dll"
      touch -d '2030-01-01' "$b/stage/onnxruntime.dll" 2>/dev/null ||
        touch -t 203001010000 "$b/stage/onnxruntime.dll"
      ;;
    already-staged)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$BUILT" >"$b/stage/wenlan-server.exe"
      touch -d '2030-01-01' "$b/stage/wenlan-server.exe" 2>/dev/null ||
        touch -t 203001010000 "$b/stage/wenlan-server.exe"
      # Records every destination it was asked to write. The case asserts the
      # log is EMPTY: a staging that copies whatever the digests say is not
      # deciding by identity, and every other case here would still pass.
      printf '%s\nprintf "%%s\\n" "$last" >>"%s"\n%s\n' \
        "$last_arg" "$b/cp.log" "$real_cp"
      ;;
    copy-that-lied)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$OTHER" >"$b/stage/wenlan-server.exe"
      # Exits 0 having written other bytes: a partial write, a concurrent
      # writer, a filesystem that lied. cp's own status cannot see it and
      # neither can the comparison made before the copy.
      printf '%s\nprintf "%%s\\n" "not what was asked for" >"$last"\nexit 0\n' "$last_arg"
      ;;
    locked-library)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$LIB_BUILT" >"$b/build/onnxruntime.dll"
      printf '%s\n' "$LIB_OTHER" >"$b/stage/onnxruntime.dll"
      # os error 32, the failure that actually happens here.
      printf '%s\ncase $last in\n  *.dll) echo "cp: cannot create regular file: Device or resource busy" >&2; exit 1 ;;\nesac\n%s\n' \
        "$last_arg" "$real_cp"
      ;;
    bare-daemon)
      # A checkout that has never run prepare-sidecars.sh: app/binaries exists
      # and is empty, and there is nothing beside the built daemon either. Zero
      # libraries, and it is not an error -- but it must be SAID, because it is
      # also what an unreadable directory used to look like.
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      ;;
    unreadable-binaries)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      # Not a missing directory -- an unreadable one, which is the state a glob
      # cannot tell apart from empty. `chmod 000` is unenforceable on the Git
      # Bash lane this runs on (and root ignores it on POSIX), so the LISTING
      # is what fails, which is precisely the measurement under test.
      # list_dir_tristate then climbs to the parent, finds `binaries` in its
      # listing, and so answers 2: it is there and could not be read.
      mkdir -p "$b/bin"
      {
        printf '#!/usr/bin/env bash\n'
        printf 'case "$*" in\n'
        printf '  *app/binaries*) echo "ls: cannot open directory: Permission denied" >&2; exit 2 ;;\n'
        printf 'esac\n'
        printf 'for real in /usr/bin/ls /bin/ls; do\n'
        printf '  if [ -x "$real" ]; then exec "$real" "$@"; fi\n'
        printf 'done\n'
        printf 'echo "no real ls on this host" >&2\n'
        printf 'exit 127\n'
      } >"$b/bin/ls"
      chmod 0755 "$b/bin/ls"
      ;;
    prepared-without-libraries)
      # prepare-sidecars.sh installs the sidecars under triple-qualified names
      # and refuses to finish on a Windows triple unless onnxruntime.dll and
      # vulkan-1.dll are beside them. So a binaries directory holding a prepared
      # sidecar and NOT holding those is a broken layout, and staging out of it
      # with zero libraries is a failure -- the distinction the glob could not
      # make, because to it both were "no matches".
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf 'a prepared sidecar\n' \
        >"$b/repo/app/binaries/wenlan-server-x86_64-pc-windows-msvc.exe"
      ;;
    prepared-with-libraries)
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf 'a prepared sidecar\n' \
        >"$b/repo/app/binaries/wenlan-server-x86_64-pc-windows-msvc.exe"
      printf '%s\n' "$LIB_BUILT" >"$b/repo/app/binaries/onnxruntime.dll"
      printf '%s\n' "$LIB_BUILT" >"$b/repo/app/binaries/vulkan-1.dll"
      ;;
    stale-in-stage)
      # A library in the STAGE that neither source directory has. Every
      # per-file check above is blind to it: no copy touches it, so no copy
      # fails, and the staging reported success while printing "no runtime
      # libraries beside the daemon" -- a claim about a directory it had not
      # read. The daemon loads whatever is in its own directory, so the two
      # statements were about different directories and the stale one wins.
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$LIB_OTHER" >"$b/stage/onnxruntime.dll"
      ;;
    two-sources-disagree)
      # ONE NAME, TWO SOURCES. app/binaries and the build directory are both
      # walked and each copy is verified on its own, so a DIFFERENT
      # onnxruntime.dll in the second was written over the verified first with
      # every per-file check still green. Which library the daemon loads was
      # decided by walk order.
      printf '%s\n' "$BUILT" >"$b/build/wenlan-server.exe"
      printf '%s\n' "$LIB_BUILT" >"$b/repo/app/binaries/onnxruntime.dll"
      printf '%s\n' "$LIB_OTHER" >"$b/build/onnxruntime.dll"
      ;;
    no-built-daemon) : ;;
    *) echo "FATAL: unknown fixture $kind" >&2; exit 1 ;;
  esac
}

run_case() { # name, driver, fixture-kind, want_rc, want_needle, want_staged
  local name="$1" driver="$2" kind="$3" want_rc="$4" needle="$5" want_staged="${6:-}"
  local b="$work/base-$RANDOM-$RANDOM" shim_body out rc=0 got
  # The one case that is READ rather than run: the shipped call site in
  # `start_runtime`, which no driver here can reach.
  if [[ "$kind" == "call-site" ]]; then
    local why
    if why="$(assert_call_site "$SUBJECT_START")"; then
      printf '  ok   %-24s\n' "$name"
      return 0
    fi
    printf '  FAIL %-24s %s\n' "$name" "$why"
    return 1
  fi
  mkdir -p "$b"
  shim_body="$(fixture "$kind" "$b")" || return 1
  local shim_env=()
  if [[ -n "$shim_body" ]]; then
    mkdir -p "$b/bin"
    printf '#!/usr/bin/env bash\n%s\n' "$shim_body" >"$b/bin/cp"
    chmod 0755 "$b/bin/cp"
  fi
  # A fixture may also have written a shim of its own (the unreadable listing),
  # so the directory decides, not the `cp` body.
  if [[ -d "$b/bin" ]]; then
    local shim="$b/bin"
    if command -v cygpath >/dev/null 2>&1; then shim="$(cygpath -w "$b/bin")"; fi
    shim_env=("WENLAN_TEST_SHIM_DIR=$shim")
  fi
  out="$(env ${shim_env[@]+"${shim_env[@]}"} \
    WENLAN_TEST_REPO_ROOT="$b/repo" \
    WENLAN_TEST_STAGE_DIR="$b/stage" \
    WENLAN_TEST_SOURCE="$b/build/wenlan-server.exe" \
    bash "$driver" 2>&1)" || rc=$?
  got="${out##*rc=}"
  got="${got%%$'\n'*}"
  if [[ "$got" != "$want_rc" ]]; then
    printf '  FAIL %-24s rc=%s, wanted %s\n' "$name" "${got:-<none>}" "$want_rc"
    rm -rf "$b"
    return 1
  fi
  # The status alone cannot tell a refusal from a refusal for another reason,
  # and three of the controls below change only which refusal is reached. The
  # message is the discriminator, so the message is asserted.
  if [[ -n "$needle" && "$out" != *"$needle"* ]]; then
    printf '  FAIL %-24s did not say [%s]\n' "$name" "$needle"
    rm -rf "$b"
    return 1
  fi
  # And what is ON DISK, because rc=0 is a claim about the staged bytes.
  if [[ -n "$want_staged" ]]; then
    case "$want_staged" in
      no-copies)
        if [[ -s "$b/cp.log" ]]; then
          printf '  FAIL %-24s copied [%s] over identical bytes\n' \
            "$name" "$(tr '\n' ' ' <"$b/cp.log")"
          rm -rf "$b"
          return 1
        fi
        ;;
      *)
        local file="${want_staged%%=*}" want_text="${want_staged#*=}"
        local staged=""
        [[ -f "$b/stage/$file" ]] && staged="$(cat "$b/stage/$file")"
        if [[ "$staged" != "$want_text" ]]; then
          printf '  FAIL %-24s staged [%s], wanted [%s]\n' "$name" "$staged" "$want_text"
          rm -rf "$b"
          return 1
        fi
        ;;
    esac
  fi
  printf '  ok   %-24s\n' "$name"
  rm -rf "$b"
  return 0
}

# name | fixture | rc | message needle | what must be on disk
CASES=(
  "stale-bytes-replaced|stale-daemon|0||wenlan-server.exe=$BUILT"
  "stale-library-replaced|stale-library|0||onnxruntime.dll=$LIB_BUILT"
  "identical-not-copied|already-staged|0||no-copies"
  "copy-that-lied-refused|copy-that-lied|1|is not the one that was just built|"
  "locked-library-is-fatal|locked-library|1|could not stage runtime library onnxruntime.dll|"
  "unhashable-source-refused|no-built-daemon|1|could not hash the built dev daemon|"
  "unreadable-binaries-refused|unreadable-binaries|1|could not list the runtime libraries in|"
  "prepared-without-libs-refused|prepared-without-libraries|1|the dev daemon was staged without onnxruntime.dll vulkan-1.dll|"
  "prepared-libraries-staged|prepared-with-libraries|0||vulkan-1.dll=$LIB_BUILT"
  "zero-libraries-is-named|bare-daemon|0|staged the dev daemon with no runtime libraries beside it|"
  "stale-in-stage-refused|stale-in-stage|1|which this run did not put there|"
  "two-sources-disagree-refused|two-sources-disagree|1|two different runtime libraries are both called onnxruntime.dll|"
  "start-calls-staging|call-site|0||"
)

run_all() { # driver
  PASSED_CASES=(); FAILED_CASES=()
  local spec name kind want needle staged rest
  for spec in "${CASES[@]}"; do
    name="${spec%%|*}"; rest="${spec#*|}"
    kind="${rest%%|*}"; rest="${rest#*|}"
    want="${rest%%|*}"; rest="${rest#*|}"
    needle="${rest%%|*}"; staged="${rest#*|}"
    if run_case "$name" "$1" "$kind" "$want" "$needle" "$staged"; then
      PASSED_CASES+=("$name")
    else
      FAILED_CASES+=("$name")
    fi
  done
}

echo "dev-runtime-stage-controls"
echo "cases against the shipped staging:"
HASHER="$(extract "$runtime_before" file_sha256)"
STAGER="$(extract "$runtime_before" stage_file_by_identity)"
DAEMON="$(extract "$runtime_before" stage_windows_daemon)"
START="$(extract "$runtime_before" start_runtime)"
# The tri-state directory listing the staging now stands on. Extracted so the
# subject is the shipped code and not a stub of it; never mutated here.
NAMER="$(extract "$runtime_before" listing_has_name)"
LISTER="$(extract "$runtime_before" list_dir_tristate)"
if [[ "$NAMER" != *"grep -qxF"* || "$LISTER" != *"ls -A"* ]]; then
  echo "FATAL: the directory listing helpers could not be extracted from dev-runtime.sh" >&2
  exit 1
fi
build_subject "$HASHER" "$STAGER" "$DAEMON" "$START"
run_all "$work/driver.sh"
failures=$((failures + ${#FAILED_CASES[@]}))

echo "controls:"
control() { # name, why, which(hasher|stager|daemon|start), old, new, must_fail...
  local name="$1" why="$2" which="$3" old="$4" new="$5"; shift 5
  local -a must_fail=("$@")
  printf '  %s  (%s)\n' "$name" "$why"
  local text head tail hits
  case "$which" in
    hasher) text="$HASHER" ;;
    stager) text="$STAGER" ;;
    daemon) text="$DAEMON" ;;
    start) text="$START" ;;
    *) printf '    FAIL unknown control target %s\n' "$which"; failures=$((failures + 1)); return ;;
  esac
  # EXACTLY once. What stood here was `head="${text%%"$old"*}"` plus a check
  # that the halves recombine and that the head is not the whole text -- which
  # establishes "at least once" and cannot see a second occurrence at all. An
  # anchor that matches twice mutates the first and leaves the second, so the
  # subject is neither the shipped code nor the reverted code, and every verdict
  # taken from it is about a third thing nobody wrote. Anything but one is a
  # hard error, including zero: a stale anchor is how a control quietly stops
  # testing what it names.
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
    hasher) build_subject "$head$new$tail" "$STAGER" "$DAEMON" "$START" ;;
    stager) build_subject "$HASHER" "$head$new$tail" "$DAEMON" "$START" ;;
    daemon) build_subject "$HASHER" "$STAGER" "$head$new$tail" "$START" ;;
    start) build_subject "$HASHER" "$STAGER" "$DAEMON" "$head$new$tail" ;;
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

# One control per property, because the three are INDEPENDENT: the comparison
# that decides the copy, the re-read that verifies it, and the status of the
# copy itself. A single revert of all three would be satisfied by whichever one
# happened to still be there.

# `cp -u` and nothing else. The digest still decides whether to copy, so the
# already-identical case is untouched; what changes is that the copy it then
# asks for is refused by mtime. Caught by the verification, which is the point:
# a timestamp cannot make the bytes match, so this cannot end anywhere but a
# refusal, and the case wanted a staged daemon.
control nc-stage-copies-by-timestamp \
  'the pre-fix shape: cp -u keeps a stale file that carries a newer mtime' stager \
  '    if ! cp -f -- "$source" "$dest"; then' \
  '    if ! cp -u -- "$source" "$dest"; then' \
  stale-bytes-replaced stale-library-replaced two-sources-disagree-refused

# The re-read. Without it the only witness to what was staged is `cp`s own exit
# status, and a copy that exits 0 having written other bytes is exactly the
# thing a status cannot see.
control nc-stage-not-verified-after-copy \
  'nothing re-reads the staged file, so cp exiting 0 is the whole proof' stager \
  '  rc=0
  got="$(file_sha256 "$dest")" || rc=$?
  if (( rc != 0 )); then
    echo "error: the staged $label could not be hashed after the copy: $dest" >&2
    echo "       a staging that cannot be verified has not been staged" >&2
    return 1
  fi
  if [[ "$got" != "$want" ]]; then' \
  '  # INJECTED: cp said it worked, so it worked.
  if false; then' \
  copy-that-lied-refused

# `|| true` on the DLL copy. os error 32 is the failure that actually happens
# here -- a library held open by a process running out of the stage directory --
# and swallowing it leaves the OLD library beside a NEW daemon.
control nc-stage-library-failure-swallowed \
  'a runtime library that cannot be written is skipped instead of fatal' daemon \
  '      stage_file_by_identity "$dir/$name" "$DAEMON_STAGE_DIR/$name" \
        "runtime library $name" || return 1' \
  '      stage_file_by_identity "$dir/$name" "$DAEMON_STAGE_DIR/$name" \
        "runtime library $name" || true' \
  locked-library-is-fatal

# The directory itself. This is the round-13h finding: the loop used to be a
# glob, and a glob answers "no matches" for a directory that is empty AND for
# one that could not be read. The injected shape keeps everything else -- the
# prepared-sidecar detection, the expected-library count, the named
# zero-library outcome -- so what is reverted is only the MEASUREMENT, and only
# the case that turns on an unreadable directory may notice.
control nc-stage-globs-the-library-directory \
  'the library directory is globbed again, so unreadable and empty are one answer' daemon \
  '    rc=0
    listing="$(list_dir_tristate "$dir")" || rc=$?' \
  '    # INJECTED: the two-state shape. An unreadable directory leaves the
    # pattern unexpanded, -f rejects the literal, and the result is exactly
    # the empty listing of a directory that has nothing in it.
    rc=0
    listing=""
    for name in "$dir"/*; do
      if [[ -f "$name" ]]; then listing+="${name##*/}"$'"'"'\n'"'"'; fi
    done' \
  unreadable-binaries-refused

# The workstream signature, in the new code: a digest whose status is dropped is
# the empty string, and two empty strings are EQUAL -- so an unhashable source
# and an unhashable destination agree that the staging is current. The refusal
# still arrives, from the verification below, which is why this control is
# pinned to the MESSAGE and not to the status.
control nc-stage-source-hash-status-dropped \
  'an unmeasured source digest compares equal to an unmeasured staged one' stager \
  '  rc=0
  want="$(file_sha256 "$source")" || rc=$?
  if (( rc != 0 )); then
    echo "error: could not hash the built $label: $source" >&2
    echo "       refusing to stage by timestamp instead: an unverified copy is" >&2
    echo "       how a stale daemon comes to answer /api/health as this build" >&2
    return 1
  fi' \
  '  # INJECTED: the status of the source digest is discarded.
  want="$(file_sha256 "$source" 2>/dev/null)" || want=""' \
  unhashable-source-refused

# --- the staged SET ----------------------------------------------------------
#
# Round 4. Every control above is about one FILE: did this copy happen, was it
# verified, was its failure fatal. The daemon does not load a series of files --
# it loads whatever is in its own directory -- and nothing had ever measured
# that directory as a whole. These two break the two ways a verified series of
# copies still leaves the wrong set behind.

control nc-stage-destination-not-measured \
  'the staged directory is never read back, so a library nobody staged is invisible' daemon \
  '  rc=0
  listing="$(list_dir_tristate "$DAEMON_STAGE_DIR")" || rc=$?' \
  '  # INJECTED: the destination is not listed; `rc` and `listing` keep whatever
  # the source walk left in them, and no name can be found stray.
  rc=0
  listing=""' \
  stale-in-stage-refused

control nc-stage-name-collision-decided-by-walk-order \
  'two sources with one name are staged in sequence, so the last one wins' daemon \
  '      if (( previous_rc == 0 )) && [[ "$previous" != "$STAGED_FILE_DIGEST" ]]; then' \
  '      # INJECTED: each copy was verified on its own, so the set is fine.
      if false; then' \
  two-sources-disagree-refused

# --- the call site -----------------------------------------------------------
#
# Everything above drives `stage_windows_daemon` directly, and every one of
# those cases stays green when nothing calls it. That is the shape of the
# finding: the remedy was complete and the code path was unreached, and the
# suite could not tell the difference. So the call site is read, and these two
# break it in the two ways it can be broken.

control nc-stage-call-site-removed \
  'nothing in start_runtime stages the daemon at all' start \
  '  if (( HOST_IS_WINDOWS == 1 )); then
    stage_windows_daemon "$build_dir/wenlan-server.exe" || {
      RESULT_KIND=staging-failure
      return 1
    }
  fi' \
  '  # INJECTED: the call is gone, and every case above is still green.' \
  start-calls-staging

control nc-stage-call-site-failure-swallowed \
  'the staging runs and its failure is logged past, which is || true one level out' start \
  '    stage_windows_daemon "$build_dir/wenlan-server.exe" || {
      RESULT_KIND=staging-failure
      return 1
    }' \
  '    stage_windows_daemon "$build_dir/wenlan-server.exe" || true' \
  start-calls-staging

if [[ "$(cat "$runtime")" != "$runtime_before" ]]; then
  echo "FATAL: scripts/dev-runtime.sh changed during the run"; exit 1
fi

echo "CONTROL FAILURES: $failures"
(( failures == 0 )) || exit 1
