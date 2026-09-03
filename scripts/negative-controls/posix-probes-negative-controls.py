"""Negative controls for the POSIX branches of scripts/lib/host-process.sh.

Both defects were found by Codex Sol in the Phase 4 review, and both are the
workstream's signature shape: a probe that could not measure answering as if it
had. New green tests prove nothing on their own, so each control below reverts
exactly one fix, runs the shipped suite against it, and requires the cases that
defend that fix to FAIL -- and the cases either side of it to SURVIVE, which is
what pins the failure to the discrimination rather than to collateral damage.

The mutated library is written to a TEMP FILE and the suite is pointed at it
with WENLAN_HOST_PROCESS_LIB; the shipped scripts/lib/host-process.sh is never
touched, and the run aborts if it changed anyway.

It used to patch the shipped file in place and restore it in a `finally`. That
works only if nothing else reads the file during the window. Something did: a
`pnpm vitest run` of host-process/attest/dev-runtime launched alongside this
harness came back "2 failed" against a library that was byte-identical before
and after. The failure was real, the code was fine, and the same race with the
timing inverted would have produced a green run over a reverted fix -- a
measurement harness that can corrupt an unrelated measurement.

ROUND 3 (Codex Sol), FINDING N1. Three defects, all of them this workstream's
signature shape sitting in the harness that exists to find it:

  1. THERE WAS NO BASELINE. Nineteen mutations, each credited when a named
     `must_fail` test comes back `failed` -- which is also what a test that was
     ALREADY failing on the shipped library comes back as. Every one of the
     nineteen could have been credited for a pre-existing red while its
     survivors passed exactly as they would have. There is now one unmutated
     run of the SHIPPED library, with no override at all, in this same
     invocation, and every name any control mentions -- must_fail and
     must_survive alike -- has to PASS on it before a single mutation is
     scored. The same argument applies to the nine override-lock cases: they
     all assert a REFUSAL, and a lock wired shut would satisfy all nine, so the
     mutation runs supply the counterexample -- the lock ADMITTED a legitimate
     triple, loaded, and passed the identity row.

  2. THERE WAS NO EXECUTION RECEIPT. `cmd /c npx vitest ...` is not the known
     bad `cmd /c "pnpm vitest run ..."` form and does not print a banner and
     exit 0 -- but nothing here established that the local suite RAN. A run is
     now scored only if its report names tests (a count, not a claim), names
     `scripts/host-process.test.ts` as the file they came from, and lists the
     SAME case set the baseline did. The runner is `pnpm exec`, which is this
     repository's spelling.

  3. A RUN THAT WAS KILLED LOOKED LIKE A RUN THAT FINISHED. This file has been
     run in the background after a ten-minute watchdog killed it, and starting a
     background job proves only that process creation succeeded. The last line
     is now a terminal completion marker printed after every control has been
     scored, every print is flushed, and the aggregate runner treats a missing
     marker as a harness that did not run to completion.

Run: python3 scripts/negative-controls/posix-probes-negative-controls.py
     python3 scripts/negative-controls/posix-probes-negative-controls.py --only nc-lsof
"""
import atexit
import hashlib
import io
import json
import os
import subprocess
import tempfile
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LIB = os.path.join(ROOT, 'scripts', 'lib', 'host-process.sh')
SUITE = 'scripts/host-process.test.ts'
SUITE_PATH = os.path.join(ROOT, 'scripts', 'host-process.test.ts')

# ROUND 4 (Codex Sol), FINDING C5.6. Round 3 guarded the IMPLEMENTATION against
# a mid-run edit and left the file that produces the evidence unguarded. Every
# verdict in this harness is a per-case status read out of this suite, twenty
# times over twenty-five minutes, and the lane that owns `host-process.sh` owns
# `host-process.test.ts` too -- so the likeliest edit of all was the one nothing
# was watching. A same-title edit to a test BODY is the bad case: the case-set
# receipt still matches the baseline name for name, the `host-process.sh` guard
# is still green, and the assertion behind the name is a different assertion.
#
# Checked immediately before and immediately after EVERY suite invocation, not
# once at each end of the run. Start/end equality answers "did the file finish
# where it started", which an edit that is later reverted also satisfies, and
# which says nothing about which side of the edit any individual control was
# scored on.
#
# The residual, stated rather than papered over: an edit that lands after the
# before-check and is reverted before the after-check is invisible to this, and
# to any polling scheme. The window is the width of one `read`, not the width
# of the run, and that is the whole of the improvement -- it is not a proof
# that the file was constant.
SUITE_BEFORE = io.open(SUITE_PATH, encoding='utf-8', newline='').read()
SUITE_DIGEST = hashlib.sha256(SUITE_BEFORE.encode('utf-8')).hexdigest()


def assert_suite_unchanged(when):
    """Refuse if the evidence-producing suite moved. `when` names the moment."""
    try:
        now = io.open(SUITE_PATH, encoding='utf-8', newline='').read()
    except OSError as exc:
        # Unreadable is not unchanged. This harness exists because a failed
        # measurement must never read as a negative one.
        sys.exit('FATAL: %s could not be read %s: %s' % (SUITE, when, exc))
    if now != SUITE_BEFORE:
        sys.exit(
            'FATAL: %s changed %s.\n'
            '  Every result in this harness is a per-case status read out of '
            'that file. Cases scored before the edit and after it are about '
            'different assertions, and nothing in a transcript says which side '
            'a given case fell on. Nothing above this line is a result.\n'
            '  at start: sha256 %s\n'
            '  now:      sha256 %s'
            % (SUITE, when, SUITE_DIGEST[:16],
               hashlib.sha256(now.encode('utf-8')).hexdigest()[:16]))

#: ROUND 3 (Codex Sol), FINDING N1, third part. This harness runs the suite
#: about thirty times and has been killed by a ten-minute watchdog. A run that
#: was killed and a run that finished clean produce the same tail unless the
#: finish says so out loud, and "the harness was started in the background"
#: proves only that process creation succeeded. So: this line is printed LAST,
#: unconditionally, only after every control has been scored, and a consumer
#: that does not see it has not seen a finished run. Every print is flushed for
#: the same reason -- a block-buffered pipe loses the transcript of a run that
#: is killed, which is precisely the run whose transcript matters.
MARKER = 'NEGATIVE-CONTROL COMPLETE'
HARNESS = 'posix-probes-negative-controls.py'
STARTED = time.time()

#: The suite's own row for "the library under test is the one I was handed".
#: Named up here because both the mutation loop and the override-lock section
#: read it: the loop uses it as the evidence that the lock ADMITS a legitimate
#: control, which is the counterexample the lock section needs before "every
#: case was refused" says anything at all.
IDENTITY = 'tests scripts/lib/host-process.sh unless a control explicitly says otherwise'


def say(text=''):
    print(text)
    sys.stdout.flush()


#: `--only <substring>` runs the baseline and just the mutations whose name
#: matches. It exists for fault injection -- breaking one control and watching
#: it fire should not cost thirty suite runs -- and it is deliberately loud:
#: the completion marker carries `partial=1`, and the aggregate runner refuses
#: a partial run rather than counting it. A convenience that can be mistaken
#: for a full run is the defect this directory is about.
ONLY = None
if '--only' in sys.argv:
    ONLY = sys.argv[sys.argv.index('--only') + 1]


#: Set by `finish` and read by the atexit handler below. Every OTHER way out of
#: this file -- a `sys.exit("FATAL: ...")`, an unhandled exception, a signal, a
#: watchdog kill -- leaves it False and gets an ABORTED line instead. There are
#: real ways out above: the subject changing on disk mid-run is one, and it
#: fired for real while this harness was being repaired, because another lane
#: was editing scripts/lib/host-process.sh at the time.
_COMPLETED = False


def finish(failures):
    global _COMPLETED
    say('CONTROL FAILURES: %d' % failures)
    if ONLY:
        say('PARTIAL RUN: --only %r; this is not a result about the harness' % ONLY)
    _COMPLETED = True
    say('%s %s failures=%d elapsed=%.1fs partial=%d'
        % (MARKER, HARNESS, failures, time.time() - STARTED, 1 if ONLY else 0))
    sys.exit(1 if failures else 0)


@atexit.register
def _abort_marker():
    if _COMPLETED:
        return
    # stderr first: `sys.exit("FATAL: ...")` writes there, and the aggregate
    # runner reads the LAST non-empty line of the two merged.
    sys.stderr.flush()
    print('NEGATIVE-CONTROL ABORTED %s elapsed=%.1fs' % (HARNESS,
                                                         time.time() - STARTED))
    print('  This run did not score every control. Nothing above it is a '
          'result about the harness.')
    sys.stdout.flush()
# Transcripts go under the gitignored target/, not next to this file: the
# harness now lives in the tracked tree (round 13c, new finding 4) and a control
# that litters the checkout it controls is one people delete rather than run.
HERE = os.path.join(ROOT, 'target', 'negative-control-logs')
os.makedirs(HERE, exist_ok=True)


#: The authorship stamp. Round 13e reopened finding 6: every rule this lock has
#: had was an argument from TIME -- a fifteen-minute window, then a two-second
#: one -- and each of them accepted a leftover younger than the window. A nonce
#: has no window. Only a writer that knew this run's value could have produced
#: bytes containing it, which is what "this run wrote the file" means.
NONCE_LINE = '# wenlan-control-nonce: %s\n'


def stamp(text, nonce=None):
    """`text` carrying an authorship nonce, and the nonce. A comment line, so
    the stamped copy is still a runnable library; the suite strips it before
    comparing the override against the shipped file."""
    nonce = nonce or hashlib.sha256(os.urandom(32)).hexdigest()
    return text.rstrip('\n') + '\n' + NONCE_LINE % nonce, nonce


def _bash():
    """A bash that can read a Windows path, CHOSEN BY TRYING IT.

    `bash` on PATH here is `C:\\Windows\\System32\\bash.exe`, the WSL launcher,
    which cannot open `C:/Users/...` at all -- it answers 127, "No such file or
    directory", for a file that is plainly there. That is indistinguishable
    from a syntax check that ran and found nothing wrong unless somebody looks
    at the status, and the first version of the gate below reported all
    fourteen mutants as unparseable because of it. So the choice is measured:
    write a file that must parse, and take the first bash that agrees it does.
    """
    import shutil
    probe = os.path.join(tempfile.gettempdir(), 'wenlan-bash-probe.sh')
    io.open(probe, 'w', encoding='utf-8', newline='').write('true\n')
    tried = []
    for candidate in (
        os.environ.get('WENLAN_BASH'),
        r'C:\Program Files\Git\bin\bash.exe',
        r'C:\Program Files\Git\usr\bin\bash.exe',
        r'C:\Program Files (x86)\Git\bin\bash.exe',
        shutil.which('bash'),
    ):
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            rc = subprocess.run([candidate, '-n', probe.replace('\\', '/')],
                                capture_output=True, text=True).returncode
        except OSError as exc:
            tried.append('%s: %s' % (candidate, exc))
            continue
        if rc == 0:
            return candidate
        tried.append('%s: exit %d' % (candidate, rc))
    sys.exit('FATAL: no bash here can read a Windows path; tried %s' % tried)


BASH = _bash()


def collect_states(data):
    """Per-test states keyed by title, REFUSING a title that is used twice.

    Every must_fail/must_survive check below looks a test up by name. Two tests
    sharing a name collapse to one entry, last write wins, and the check then
    scores a probe it was never pointed at while reporting a clean result --
    this file's own signature defect, one level out. There were two such names
    in scripts/host-process.test.ts when this guard was written.
    """
    states = {}
    for f in data.get('testResults', []):
        for a in f.get('assertionResults', []):
            title = a.get('title')
            if title in states:
                raise ValueError(
                    'two tests are named %r; the name-keyed checks in this '
                    'harness cannot tell them apart' % title)
            states[title] = a.get('status')
    return states


def collect_files(data):
    """The test FILES the report says ran.

    ROUND 3, FINDING N1, second part. "Only an execution receipt can establish
    that local Vitest actually ran." A runner that starts, prints a banner and
    exits 0 without running anything leaves either no report or an empty one;
    a runner pointed at the wrong file leaves a full report about something
    else. Both read downstream as a suite that ran. The file list and the test
    COUNT are the receipt, and both are asserted rather than assumed.
    """
    return [f.get('name', '') for f in data.get('testResults', [])]


def read_report(path):
    """(states, files, raw) from a vitest json report; (None, [], raw) if it
    cannot be read at all."""
    if not os.path.exists(path):
        return None, [], ''
    raw = io.open(path, encoding='utf-8').read()
    try:
        data = json.loads(raw)
        return collect_states(data), collect_files(data), raw
    except (ValueError, KeyError, TypeError) as exc:
        say('    (could not read the vitest report: %s)' % exc)
        return None, [], raw


def receipt_problems(states, files, label):
    """Complaints about a run that did not actually execute the local suite."""
    problems = []
    if not states:
        problems.append('%s: the report names no test at all; a runner that '
                        'silently ran nothing produces exactly this' % label)
        return problems
    if not any(os.path.basename(SUITE) in (f or '') for f in files):
        problems.append('%s: the report is about %s, not %s; the run executed '
                        'some other suite' % (label, files or '<nothing>', SUITE))
    return problems

# An earlier version of the lsof control stopped one line short and left the
# `[[ -n "$hit" ]] || return 2` behind, which turned the genuine negative into a
# 2 as well: the suite went red, but on the wrong test, and the four cases the
# control exists to defend all still passed. A control that reverts part of a
# fix measures the part it left alone. MUST_SURVIVE is the guard against that.
MUTATIONS = [
    {
        'name': 'nc-lsof-status-collapsed',
        'why': 'every nonzero lsof status becomes "port free"',
        'old': '''    rc=0
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
    [[ -n "$hit" ]] || return 2''',
        'new': '''    # INJECTED: the pre-fix shape
    hit="$(lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null)" || hit=""
    hit="$(printf '%s' "$hit" | sed -n '1p')" || return 2''',
        'must_fail': [
            'reports COULD NOT MEASURE when lsof exits 1 with an error on stderr',
            'reports COULD NOT MEASURE for any status other than 0 or 1',
            'reports COULD NOT MEASURE when lsof exits 0 saying nothing',
            'passes the failure through probe_listener_port as unmeasured, not none',
            # The pre-fix shape this control restores also loses the
            # ratification of the silent negative, which landed in the library
            # while this harness was being repaired. Under all-survivor scoring
            # a case a mutation legitimately reddens has to be DECLARED; the
            # alternative is a control quietly credited for collateral.
            'reports COULD NOT MEASURE when a silent lsof cannot ratify its own '
            'negative',
            'reports COULD NOT MEASURE when the enumeration exits 0 saying nothing',
            'reports COULD NOT MEASURE when the enumeration prints something '
            'that is not a pid',
            # And the STRUCTURAL row: the injected `|| hit=""` is the exact
            # shape that test pins by reading the function's source, so the
            # revert reddens it too. Declared rather than tolerated -- under
            # the hand-listed survivor scoring this file used to do, all four
            # of these reddened invisibly.
            'checks the status of every stage of the Windows listener pipeline',
        ],
        'must_survive': [
            'reports a NEGATIVE when lsof exits 1 with nothing to say and can '
            'still enumerate',
            'reports the listener pid when lsof matches',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-kill-0-is-the-only-witness',
        'why': 'EPERM and ESRCH both read as "process is dead"',
        'old': '''    kill -0 "$1" 2>/dev/null && return 0
    command -v ps >/dev/null 2>&1 || return 2''',
        'new': '''    kill -0 "$1" 2>/dev/null && return 0
    # INJECTED: the pre-fix shape -- `kill -0` is the only witness, so
    # everything below is unreachable and every one of its failures, EPERM
    # included, reads as "the process is dead".
    return 1
    command -v ps >/dev/null 2>&1 || return 2''',
        'must_fail': [
            'reports ALIVE when kill says no but the process table lists the pid',
            'reports COULD NOT MEASURE when the process table errors',
            'reports COULD NOT MEASURE when the process table prints something else',
            'reports COULD NOT MEASURE for a silent ps when kill says EPERM, not ESRCH',
            "reports COULD NOT MEASURE for a pid past the shell's own integer range",
            'passes the failure through probe_process_alive as unmeasured, not dead',
        ],
        'must_survive': [
            'reports a NEGATIVE when the process table quietly has no such pid',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ---- round 13d ---------------------------------------------------------
    # Codex Sol's round-13d new finding 1 and 13c-new findings 2 and 3: the same
    # defect, three more levels out. Every one of these reverts is a shape that
    # SHIPPED, and every one of them turns a failed parse into a measured
    # negative that a caller acts on by deleting a live daemon's ownership
    # record or by binding onto an occupied port.
    {
        'name': 'nc-tasklist-prose-is-absence',
        'why': 'a status-0 diagnostic from tasklist becomes "the process is dead"',
        'old': '''    out="$(tasklist //NH //FO CSV 2>&1)" || return 2
    [[ -n "$out" ]] || return 2
    rc=0
    printf '%s\\n' "$out" | tr -d '\\r' | awk -v pid="$1" '
      BEGIN { FS = "\\",\\"" }
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
    esac''',
        'new': '''    # INJECTED: the pre-fix shape -- a filtered query, and "does not start
    # with a quote" standing in for "absent".
    out="$(tasklist //FI "PID eq $1" //NH //FO CSV 2>/dev/null)" || return 2
    [[ -n "$out" ]] || return 2
    case "$out" in
      '"'*) return 0 ;;
      *) return 1 ;;
    esac''',
        'must_fail': [
            'reports a NEGATIVE for a table that lists other processes but not this pid',
            'reports COULD NOT MEASURE for a status-0 line of prose, not a negative',
            'reports COULD NOT MEASURE for a table too short to be a process table',
            'reports COULD NOT MEASURE for a well-formed fragment that contains pid 4',
            'reports COULD NOT MEASURE for a full table with no System process in it',
            'reports COULD NOT MEASURE for a table with a warning merged into it',
            'probe_process_alive reports gone',
        ],
        'must_survive': [
            'reports alive for a pid the process table lists',
            'reports COULD NOT MEASURE when the process table command fails',
            'reports COULD NOT MEASURE when the process table says nothing at all',
            'reports COULD NOT MEASURE for an argument that is not a pid',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ---- round 13e ---------------------------------------------------------
    # Codex Sol's round-13e new finding 2: "the Windows whole-table remedies
    # validate row fragments, not table completeness." The row floor, the System
    # process and the every-line-is-a-row rule are three INDEPENDENT conditions,
    # so each one needs its own revert -- a single control over all three would
    # be satisfied by whichever of them happened to still be there.
    {
        'name': 'nc-tasklist-fragment-is-a-table',
        'why': 'a two-row fragment is accepted as this machine\'s process list',
        # Anchored on the tasklist END block as a whole, not on the floor line
        # alone. Round 13f gave `process_image_path` the same two witnesses, so
        # `        if (rows < 10) exit 3` now appears TWICE in this library and
        # a one-line anchor matches both -- which the pre-check below reports
        # as "the control would test nothing", correctly and unhelpfully.
        'old': '''        if (!system_process) exit 3
        if (rows < 10) exit 3
        exit(hit ? 0 : 1)''',
        'new': '''        if (!system_process) exit 3
        # INJECTED: no floor, so a fragment answers for the whole machine.
        exit(hit ? 0 : 1)''',
        'must_fail': [
            'reports COULD NOT MEASURE for a well-formed fragment that contains pid 4',
            'reports COULD NOT MEASURE for a table too short to be a process table',
        ],
        'must_survive': [
            'reports alive for a pid the process table lists',
            'reports a NEGATIVE for a table that lists other processes but not this pid',
            'reports COULD NOT MEASURE for a full table with no System process in it',
            'reports COULD NOT MEASURE for a table with a warning merged into it',
            'reports COULD NOT MEASURE for a status-0 line of prose, not a negative',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-tasklist-system-process-unchecked',
        'why': 'a full-length CSV table that is not a process table is accepted',
        # Anchored on the CSV-specific line above it, for the same reason as
        # the control before this one: the System-process witness is now in two
        # awk programs in this file, and only one of them is tasklist's.
        'old': '''        if (rows != lines) exit 3
        if (!system_process) exit 3''',
        'new': '''        if (rows != lines) exit 3
        # INJECTED: nothing has to be present for this to be a process table.''',
        'must_fail': [
            'reports COULD NOT MEASURE for a full table with no System process in it',
        ],
        'must_survive': [
            'reports alive for a pid the process table lists',
            'reports a NEGATIVE for a table that lists other processes but not this pid',
            'reports COULD NOT MEASURE for a well-formed fragment that contains pid 4',
            'reports COULD NOT MEASURE for a table with a warning merged into it',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-tasklist-stderr-dropped',
        'why': 'a status-0 warning beside a short table is invisible again',
        'old': '''    out="$(tasklist //NH //FO CSV 2>&1)" || return 2''',
        'new': '''    # INJECTED: stderr discarded, so the every-line-is-a-row rule has nothing
    # to catch and a warning about a table this probe cannot trust is lost.
    out="$(tasklist //NH //FO CSV 2>/dev/null)" || return 2''',
        'must_fail': [
            'reports COULD NOT MEASURE for a table with a warning merged into it',
        ],
        'must_survive': [
            'reports alive for a pid the process table lists',
            'reports a NEGATIVE for a table that lists other processes but not this pid',
            'reports COULD NOT MEASURE for a status-0 line of prose, not a negative',
            'reports COULD NOT MEASURE for a well-formed fragment that contains pid 4',
            'reports COULD NOT MEASURE when the process table command fails',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # Codex Sol's round-13e reopened 13c-new#2, the POSIX half: `kill -0`'s
    # status is not an answer on its own, and the errno text is what separates
    # "there is no such process" from "there it is, and it is not yours".
    {
        'name': 'nc-eperm-is-absence',
        'why': 'a silent ps plus an EPERM kill agree on a negative neither made',
        'old': '''    errno_says_no_such_process "$1" || return 2
    return 1''',
        'new': '''    # INJECTED: kill's STATUS again, with its errno thrown away, so EPERM --
    # the kernel confirming the process exists -- reads as absence.
    return 1''',
        'must_fail': [
            'reports COULD NOT MEASURE for a silent ps when kill says EPERM, not ESRCH',
            "reports COULD NOT MEASURE for a pid past the shell's own integer range",
        ],
        'must_survive': [
            'reports a NEGATIVE when the process table quietly has no such pid',
            'reports ALIVE when kill says no but the process table lists the pid',
            'reports COULD NOT MEASURE when the process table errors',
            'reports COULD NOT MEASURE when the process table prints something else',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-image-eperm-is-absence',
        'why': 'the image probe reads EPERM as "no such process" as well',
        'old': '''      errno_says_no_such_process "$1" || return 2
      return 1''',
        'new': '''      # INJECTED: kill's status without its errno, on the image path.
      return 1''',
        'must_fail': [
            'reports COULD NOT MEASURE for a silent image ps when kill says EPERM, not ESRCH',
        ],
        'must_survive': [
            'reports a NEGATIVE only for status 1 with nothing on either stream',
            'reports the image path for a pid ps lists',
            "reports COULD NOT MEASURE for a status that is not ps's own no-match",
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-image-status-0-silence-is-absence',
        'why': 'a ps that exits 0 saying nothing falls through to "no such process"',
        'old': '''      [[ -n "$out" ]] || return 2
    fi
    if (( rc != 0 )); then''',
        'new': '''    fi
    if (( rc != 0 )); then''',
        'must_fail': [
            'reports COULD NOT MEASURE when ps succeeds and prints nothing',
        ],
        'must_survive': [
            'reports the image path for a pid ps lists',
            'reports COULD NOT MEASURE when a warning is merged into the image path',
            'reports a NEGATIVE only for status 1 with nothing on either stream',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # Round 13g moved this parse out of `process_image_path` and into
    # `ps_w_row_for`, because `windows_pid_for_job` had a second copy of it that
    # no hardening had ever reached. So each of the five reverts below now
    # reddens BOTH sets of cases -- the image ones and the job ones -- and that
    # is the property the factoring was for: a witness cannot be removed from
    # one caller and left in the other, because there is only one of it.
    {
        'name': 'nc-ps-w-header-unchecked',
        'why': 'a `ps -W` header this parser cannot read becomes "no such pid"',
        'old': '''      if (!kc || !wp || !col) exit 3
      if (notarow) exit 3''',
        'new': '''      # INJECTED: the pre-fix shape -- no header check. `kc && wp && col`
      # then skips every row, awk exits 1 with nothing printed, and the empty
      # result reads as "no such pid" rather than "unreadable header".
      if (notarow) exit 3''',
        'must_fail': [
            'reports COULD NOT MEASURE when the snapshot header is not the one it parses',
            'reports COULD NOT MEASURE when the job snapshot header is not the one it parses',
        ],
        'must_survive': [
            'reports the image path for a pid in the snapshot',
            'reads the image column by offset, so a two-token STIME cannot corrupt it',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports COULD NOT MEASURE when the snapshot command fails',
            'refuses a ps -W snapshot with a warning merged into it',
            'refuses a ps -W snapshot with a nine-word diagnostic in the table',
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'reports the Windows pid when the row names the program that was launched',
            'keeps polling while the job row is measurably absent, then reports the pid',
            'probe_process_image reports found',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-ps-w-contamination-tolerated',
        'why': 'a stderr line merged into the snapshot is parsed as part of it',
        'old': '''      if (!kc || !wp || !col) exit 3
      if (notarow) exit 3
      if (!system_process) exit 3''',
        'new': '''      if (!kc || !wp || !col) exit 3
      # INJECTED: every line with a readable header is a data row, however
      # few fields it has, so a merged `ps: ...` message is silently ignored.
      if (!system_process) exit 3''',
        'must_fail': [
            'refuses a ps -W snapshot with a warning merged into it',
            # The nine-word case defends the same line: the row SHAPE decides
            # what `notarow` is set on, and this decides whether that verdict is
            # acted on. Both must go red, or only one half of the pair is real.
            'refuses a ps -W snapshot with a nine-word diagnostic in the table',
            # And the same line, reached through the other caller: eight words
            # this time, which is the count round 13g found `windows_pid_for_job`
            # still accepting.
            'reports COULD NOT MEASURE for a job snapshot with an eight-word warning in it',
        ],
        'must_survive': [
            'reports the image path for a pid in the snapshot',
            'reads the image column by offset, so a two-token STIME cannot corrupt it',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports COULD NOT MEASURE when the snapshot header is not the one it parses',
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'reports the Windows pid when the row names the program that was launched',
            'reports COULD NOT MEASURE when the job snapshot header is not the one it parses',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ---- round 13f ---------------------------------------------------------
    # `process_image_path` now asks `ps -W` for the two completeness witnesses
    # `process_is_alive` has asked of `tasklist` since round 13e, and reads the
    # row shape structurally instead of counting words. Three INDEPENDENT
    # conditions, so three reverts: a single control over all of them would be
    # satisfied by whichever one happened to still be there, which is the
    # mistake round 13e wrote the tasklist trio to avoid.
    #
    # Each is pinned to the one case that exists for it, and lists the other two
    # under must_survive, so a mutation that reddens the whole table is reported
    # as unpinned rather than as three successes.
    {
        'name': 'nc-ps-w-system-process-unchecked',
        'why': 'a ps -W table with no System process is still a whole process table',
        # `if (!seen) exit 1` is what makes this anchor `ps_w_row_for`s block and
        # not tasklist's; the witness line itself is in both, and at a shallower
        # indent here, which a substring count cannot tell apart on its own.
        'old': '''      if (!system_process) exit 3
      if (rows < 10) exit 3
      if (!seen) exit 1''',
        'new': '''      # INJECTED: nothing has to be present for this to be the whole
      # process table, so ten rows without WINPID 4 answer for the machine
      # and absence from them reads as "no such pid".
      if (rows < 10) exit 3
      if (!seen) exit 1''',
        'must_fail': [
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'reports COULD NOT MEASURE for a job snapshot with no System process in it',
        ],
        'must_survive': [
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'refuses a ps -W snapshot with a nine-word diagnostic in the table',
            'reports COULD NOT MEASURE when the snapshot header is not the one it parses',
            'reports the image path for a pid in the snapshot',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports COULD NOT MEASURE for a job snapshot too short to be a process table',
            'reports the Windows pid when the row names the program that was launched',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-ps-w-row-floor-unchecked',
        'why': 'three rows is a machine, and absence from three rows is "no such pid"',
        'old': '''      if (rows < 10) exit 3
      if (!seen) exit 1''',
        'new': '''      # INJECTED: no floor, so a fragment answers for the whole machine.
      if (!seen) exit 1''',
        'must_fail': [
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'reports COULD NOT MEASURE for a job snapshot too short to be a process table',
        ],
        'must_survive': [
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'refuses a ps -W snapshot with a nine-word diagnostic in the table',
            'reports COULD NOT MEASURE when the snapshot header is not the one it parses',
            'reports the image path for a pid in the snapshot',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports COULD NOT MEASURE for a job snapshot with no System process in it',
            'reports the Windows pid when the row names the program that was launched',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-ps-w-row-shape-is-a-word-count',
        'why': 'the round-13e form: eight FIELDS is a row, so a nine-word diagnostic is one',
        'old': '''    !($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ &&
      $4 ~ /^[0-9]+$/) { notarow = 1; next }''',
        'new': '''    # INJECTED: the pre-fix shape -- a WORD COUNT wearing a row shape, so
    # the stderr merge above is decorative for exactly the lines it was
    # added to catch, and a nine-word diagnostic is counted as a row.
    NF < 8 { notarow = 1 }''',
        'must_fail': [
            'refuses a ps -W snapshot with a nine-word diagnostic in the table',
            # The eight-word one, which is the count `windows_pid_for_job`s own
            # copy of this rule was still accepting when round 13g found it.
            'reports COULD NOT MEASURE for a job snapshot with an eight-word warning in it',
        ],
        'must_survive': [
            # The four-word warning is still shorter than eight fields, so this
            # one stays green -- which is the whole point: the old rule caught
            # the diagnostic it was written against and nothing wider.
            'refuses a ps -W snapshot with a warning merged into it',
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'reports COULD NOT MEASURE when the snapshot header is not the one it parses',
            'reports the image path for a pid in the snapshot',
            'reads the image column by offset, so a two-token STIME cannot corrupt it',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports the Windows pid when the row names the program that was launched',
            'reports COULD NOT MEASURE for a job snapshot too short to be a process table',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ---- round 13g ---------------------------------------------------------
    # The finding itself: `windows_pid_for_job` carried its own copy of the
    # parse, and every hardening above landed on the other one. These two put
    # the copy back -- the whole parse, and then just the escalation rule -- so
    # that the factoring and the poll semantics are each measured on their own.
    {
        'name': 'nc-jobpid-parses-the-table-itself',
        'why': 'the second copy of the parse, with NF < 8 and neither witness',
        'old': '''    rc=0
    row="$(ps_w_row_for PID "$job_pid")" || rc=$?''',
        'new': '''    # INJECTED: the pre-fix shape -- this helper's own parse, which tested
    # `NF < 8` (a word count) and asked for neither completeness witness. A
    # warning line or a short table then yields no row and exits 0, and the
    # poll below reads that as "the process has not appeared YET".
    rc=0
    row="$(ps -W 2>&1 | awk -v p="$job_pid" '
      NR == 1 {
        col = index($0, "COMMAND")
        for (i = 1; i <= NF; i++) {
          if ($i == "PID" && !pc) pc = i
          if ($i == "WINPID") wp = i
        }
        next
      }
      NF == 0 { next }
      NF < 8 { notarow = 1 }
      pc && wp && col && $pc == p && !seen { seen = 1; row = $wp " " substr($0, col) }
      END {
        if (!pc || !wp || !col) exit 3
        if (notarow) exit 3
        if (seen) print row
      }
    ')" || rc=2
    if (( rc == 0 )); then
      row="$(printf '%s' "$row" | tr -d '\\r')" || rc=2
    fi
    if (( rc == 0 )) && [[ -z "$row" ]]; then rc=1; fi''',
        'must_fail': [
            'reports COULD NOT MEASURE for a job snapshot with an eight-word warning in it',
            'reports COULD NOT MEASURE for a job snapshot too short to be a process table',
            'reports COULD NOT MEASURE for a job snapshot with no System process in it',
            # The shape contract, which is the only thing that can see a SECOND
            # parse rather than a weaker one. The three cases above would all be
            # answered by hardening the copy in place; this one would not.
            'reads the `ps -W` table in exactly one place',
        ],
        'must_survive': [
            'reports the Windows pid when the row names the program that was launched',
            'keeps polling while the job row is measurably absent, then reports the pid',
            # The old copy did check its header, which is why the header rule is
            # not what this control is about.
            'reports COULD NOT MEASURE when the job snapshot header is not the one it parses',
            'reports the image path for a pid in the snapshot',
            'reports a NEGATIVE when the snapshot has no such pid',
            'reports COULD NOT MEASURE for a ps -W table too short to be a process table',
            'reports COULD NOT MEASURE for a ps -W table with no System process in it',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-jobpid-negative-is-final',
        'why': 'a row that has not appeared YET is reported as an unreadable table',
        'old': '''    (( rc == 2 )) && return 2''',
        'new': '''    # INJECTED: the third state swallows the second, so the first snapshot
    # taken before the exec lands ends the poll -- and a daemon that was
    # merely slow is reported as a table that could not be read.
    (( rc != 0 )) && return 2''',
        'must_fail': [
            'keeps polling while the job row is measurably absent, then reports the pid',
            # Legitimate collateral rather than a missed pin, and the reason is
            # readable off the mutation above rather than resting on how it was
            # found: this mutation folds `rc == 2` into `rc != 0`, so the poll
            # ends on the FIRST snapshot whatever it said. A run that must see a
            # full window of readable tables before it may answer "not there"
            # cannot see one either. Declared, because a case a control's own
            # mutation reddens has to be named or the control is credited for
            # collateral it never claimed.
            'reports the MEASURED NEGATIVE after a full window of readable tables',
        ],
        'must_survive': [
            'reports the Windows pid when the row names the program that was launched',
            'reports COULD NOT MEASURE for a job snapshot with an eight-word warning in it',
            'reports COULD NOT MEASURE for a job snapshot too short to be a process table',
            'reports COULD NOT MEASURE for a job snapshot with no System process in it',
            'reports COULD NOT MEASURE when the job snapshot header is not the one it parses',
            'reports the image path for a pid in the snapshot',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-image-path-swallows-ps-stderr',
        'why': 'a failing `ps` plus an EPERM `kill -0` agree on a negative neither made',
        'old': '''    out="$(ps -p "$1" -o command= 2>&1)" || rc=$?''',
        'new': '''    # INJECTED: the pre-fix shape -- stderr discarded, so the message that
    # distinguishes a broken ps from an absent pid never arrives.
    out="$(ps -p "$1" -o command= 2>/dev/null)" || rc=$?''',
        'must_fail': [
            'reports COULD NOT MEASURE when ps fails with status 1 AND a message',
            'reports COULD NOT MEASURE when a warning is merged into the image path',
        ],
        'must_survive': [
            'reports the image path for a pid ps lists',
            'reports a NEGATIVE only for status 1 with nothing on either stream',
            "reports COULD NOT MEASURE for a status that is not ps's own no-match",
            'reports COULD NOT MEASURE for a silent image ps when kill says EPERM, not ESRCH',
            'reports COULD NOT MEASURE when ps succeeds and prints nothing',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ROUND 4. These two replace `nc-netstat-schema-is-any-tcp-line` and
    # `nc-netstat-torn-row-tolerated`, whose anchors named the `notarow`
    # program that the listener parse was built around until it was rewritten.
    # Both dropped to zero matches the moment that landed, and the harness
    # refused rather than scoring them -- which is the anchor count doing its
    # job, and also two controls silently reverting nothing until it spoke up.
    #
    # The rewrite split the old single rule in two, so there are two controls
    # rather than one renamed one. The parse now says: after the first
    # protocol-shaped row, every non-blank line must also be a row (`after`);
    # before it, at most two non-blank lines may not be (`preamble > 2`, the
    # localised banner and the column header, which can be counted but never
    # matched by text). The second rule is not a refinement of the first --
    # `2>&1` does not order the two streams, so a warning printed BEFORE the
    # banner is not covered by `after` at all.
    #
    # One property of the call site shapes what these can assert: the awk is
    # invoked as `... || return 2`, so awk exit 1 (no listening-shaped row
    # anywhere) and awk exit 2 (contaminated) BOTH become return 2. A control
    # here therefore cannot distinguish which exit the awk took, and must be
    # pinned to case outcomes instead. That is also why neither of these
    # reddens the two "not a table at all" cases: with the rule reverted those
    # tables produce exit 1 rather than exit 2, and `|| return 2` maps both to
    # the same answer, so the cases still pass. Checked, not assumed -- the
    # computed survivor set below would have reported it either way.
    {
        'name': 'nc-netstat-preamble-uncounted',
        'why': 'a diagnostic printed ahead of the banner is counted as banner',
        'old': '        if (after || preamble > 2) exit 2',
        'new': '        if (after) exit 2',
        'must_fail': [
            'reports COULD NOT MEASURE for a diagnostic merged into the preamble',
        ],
        'must_survive': [
            'reports the listener pid when the port is held',
            'reports a NEGATIVE, not a failure, when nothing is listening',
            'reports COULD NOT MEASURE for a diagnostic merged after valid rows',
            'still parses the real table shape, blank lines and all',
            'finds the listener in a table whose State column is translated',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    {
        'name': 'nc-netstat-trailing-diagnostic-skipped',
        'why': 'a warning merged in after the rows is skipped, not refused',
        # No apostrophe anywhere in this replacement: the awk program it lands
        # in is a single-quoted shell string, so one `'` ends the program and
        # the library stops parsing. That produced four unrelated test failures
        # and one "caught" -- a control that broke the file reading exactly like
        # a control that reverted a fix. The `bash -n` gate below now separates
        # those two, and this comment is why it exists.
        'old': '      { if (rows) after = 1; else preamble++ }',
        'new': '      { if (rows) after = 0; else preamble++ }',
        'must_fail': [
            'reports COULD NOT MEASURE for a diagnostic merged after valid rows',
            # Declared, not collateral: this case puts the SAME trailing
            # warning after a table that does contain the port, so reverting
            # the `after` rule takes it from "refused" to a confidently
            # reported pid. It is the more dangerous half of the property --
            # the other case merely loses a refusal, this one answers.
            'refuses a contaminated table even when the port IS in it',
            # Found by the computed survivor set, not by reading the code --
            # which is the point of computing it. I traced this control's blast
            # radius by hand before running it, got two of the three, and would
            # have shipped a control credited for a red it never declared.
            # Legitimate collateral and the same property: this case is a valid
            # row followed by `  TCP    0.0.0.0:445`, a row torn mid-write. A
            # torn row is not a diagnostic, but it arrives AFTER a good row and
            # so is caught by the same `after` rule, and reverting that rule
            # stops catching it too.
            'refuses a netstat table with a torn protocol row in it',
        ],
        'must_survive': [
            'reports the listener pid when the port is held',
            'reports a NEGATIVE, not a failure, when nothing is listening',
            'reports COULD NOT MEASURE for a diagnostic merged into the preamble',
            'still parses the real table shape, blank lines and all',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
    # ROUND 5, and it is the third rule of the same parse. The two above are a
    # GRAMMAR -- they refuse a table with something in it that is not a row --
    # and a table cut short after a well-formed row is all well-formed rows, so
    # neither of them can see it. Every surviving row validates, the port query
    # finds no row for our port in what arrived, and the branch returns MEASURED
    # FREE off a table that stopped early. The remedy is the one witness netstat
    # does offer: `netstat -ano` prints the whole TCP table and then the whole
    # UDP one, so a UDP row proves the stream got past every TCP row there was.
    #
    # This control exists in two places on purpose and they are not duplicates.
    # `nc-listener-truncation-unwitnessed` in dev-runtime-scan-controls.sh
    # proves the BASH harness would notice the witness going away; this one
    # proves host-process.test.ts would, and host-process.test.ts is the suite
    # this library's changes are actually run against. A remedy defended only
    # from the other lane is a remedy the vitest file may quietly stop covering.
    #
    # Adding the witness is also what broke `nc-netstat-preamble-uncounted` the
    # first time this harness ran after it landed: that control's fixture was
    # hand-built and carried no UDP row, so the truncation rule refused it no
    # matter what the preamble rule did, and the control reported that it was
    # measuring nothing. The fixture now carries the witness; ONE fixture, ONE
    # defect is the rule a three-rule parse makes load-bearing.
    {
        'name': 'nc-netstat-truncation-unwitnessed',
        'why': 'a table cut off before the UDP section is treated as a whole table',
        'old': '        if (!udp) exit 3',
        'new': '        if (0) exit 3',
        'must_fail': [
            'reports COULD NOT MEASURE for a table that stopped inside the TCP section',
        ],
        'must_survive': [
            # The three that decide the other two rules, plus the two ordinary
            # answers: reverting this must not change what a COMPLETE table
            # says, in either direction.
            'reports the listener pid when the port is held',
            'reports a NEGATIVE, not a failure, when nothing is listening',
            'reports COULD NOT MEASURE for a diagnostic merged into the preamble',
            'reports COULD NOT MEASURE for a diagnostic merged after valid rows',
            'still parses the real table shape, blank lines and all',
            'tests scripts/lib/host-process.sh unless a control explicitly says otherwise',
        ],
    },
]

original = io.open(LIB, encoding='utf-8', newline='').read()
# EXACTLY once, or this stops before it measures anything. Zero means the code
# moved out from under the anchor and the control reverts nothing; more than
# one means the mutation lands somewhere it was never aimed, and both of those
# read downstream as a control that ran. Round 13f produced both at once: the
# two new witnesses in `process_image_path` are spelled the same as tasklist's,
# which doubled two anchors, and they were inserted inside the END block two
# more anchors quoted, which zeroed those.
#
# Every anchor is reported, not just the first. The suite under each control
# takes about a minute, so a harness that exits at the first stale anchor makes
# fixing four of them four full runs -- and the state it exits in looks exactly
# like the state where only one is wrong.
stale = [(m['name'], original.count(m['old'])) for m in MUTATIONS
         if original.count(m['old']) != 1]
if stale:
    # A bare "matched 0 times" sends the reader looking for a code change that
    # may not have happened. The commonest cause on this platform is not a code
    # change at all: a whole-file rewrite that flips the line endings. Every
    # multi-line anchor drops to zero at once, because the anchor carries \n and
    # the file now carries \r\n, while every single-line anchor keeps matching
    # -- which makes it look like a handful of unrelated controls rotted rather
    # than one cosmetic edit landing. Measured on this repo the same hour this
    # was written: `scripts/first-run/lib.ps1` went to 556 CRLF against 556 LF
    # and took every multi-line anchor in the lib-ps1 harness with it.
    #
    # So say which it is. This does NOT normalise and retry -- a mutant written
    # with different line endings than its subject is a different program, and
    # for a shell library CRLF is a broken one. It reports the cause and still
    # refuses.
    _flipped = original.replace('\r\n', '\n')
    for name, count in stale:
        note = ''
        if count == 0:
            for m in MUTATIONS:
                if m['name'] == name and _flipped.count(m['old']) == 1:
                    note = ('; it matches once after CRLF->LF normalisation, so '
                            'the line endings moved, not the code')
                    break
        print('%s: anchor matched %d times; the control would test nothing%s'
              % (name, count, note))
    if original != _flipped:
        print('note: %s contains %d CRLF line ending(s); the rest of scripts/ is LF'
              % (os.path.relpath(LIB, ROOT), original.count('\r\n')))
    sys.exit('%d of %d anchors no longer name one place in %s'
             % (len(stale), len(MUTATIONS), os.path.relpath(LIB, ROOT)))

# ROUND 4 (Codex Sol), FINDING C5.1. An anchor that occurs exactly once still
# reverts nothing if the replacement is the text it replaces. `old == new` is
# not a hypothetical typo: these entries are written by copying the neighbour
# above and editing one of the two strings, and editing neither leaves a
# control that mutates a real place in the file to exactly what was already
# there. It then parses, the suite passes, and the control is reported as "the
# suite PASSED with the fix reverted" -- a control failure, but for a reason
# that names the suite instead of the control. Caught here, by name, before any
# of it runs.
inert = [m['name'] for m in MUTATIONS
         if original.replace(m['old'], m['new']) == original]
if inert:
    for name in inert:
        print('%s: the mutant is byte-identical to the shipped library; this '
              'control reverts nothing' % name)
    sys.exit('%d of %d mutations do not change %s'
             % (len(inert), len(MUTATIONS), os.path.relpath(LIB, ROOT)))

say('posix-probes-negative-controls')
failures = 0

# --- THE BASELINE ----------------------------------------------------------
#
# ROUND 3 (Codex Sol), FINDING N1. The nineteen mutation runs below had NO
# UNMUTATED BASELINE. Each one credits itself for a named test that came back
# `failed`, and a test that was ALREADY failing on the shipped library comes
# back `failed` too -- so every one of the nineteen could be credited for a
# pre-existing red, with the survivors passing exactly as they would have. That
# is this file's own signature defect, in the file itself, for the second time.
#
# The baseline runs the suite against the SHIPPED library with no override at
# all, in this same invocation, and every name any control names -- must_fail
# and must_survive alike -- has to PASS on it. A must_fail name that is already
# red is not evidence about the mutation; a must_survive name that is already
# red makes its control's "survived" check unmeetable for the wrong reason.
#
# One extra suite run for nineteen controls, which is the cheapest thing in
# this file.
say('baseline: the shipped library, no override')
_named = sorted({n for m in MUTATIONS for n in m['must_fail'] + m['must_survive']})
with tempfile.TemporaryDirectory() as work:
    _report = os.path.join(work, 'vitest.json')
    _env = dict(os.environ)
    for _key in ('WENLAN_HOST_PROCESS_LIB', 'WENLAN_HOST_PROCESS_LIB_CONTROL',
                 'WENLAN_HOST_PROCESS_LIB_NONCE'):
        _env.pop(_key, None)
    assert_suite_unchanged('before the baseline run')
    _proc = subprocess.run(
        ['cmd', '/c', 'pnpm', 'exec', 'vitest', 'run', SUITE,
         '--reporter=json', '--outputFile=' + _report],
        cwd=ROOT, capture_output=True, text=True, env=_env)
    assert_suite_unchanged('during the baseline run')
    BASELINE, _files, _raw = read_report(_report)
    _out = _proc.stdout + _proc.stderr + _raw

_problems = receipt_problems(BASELINE, _files, 'baseline')
if _proc.returncode != 0:
    _problems.append('baseline: the shipped library does not pass its own suite '
                     '(exit %d); no mutation below could be told apart from the '
                     'red that is already there' % _proc.returncode)
if BASELINE:
    for _name in _named:
        _state = BASELINE.get(_name)
        if _state != 'passed':
            _problems.append(
                'baseline: %r is %s on the SHIPPED library; a control that '
                'names it would be credited for a red it did not cause'
                % (_name, _state or 'absent'))
if _problems:
    for _problem in _problems:
        say('  FAIL %s' % _problem)
    io.open(os.path.join(HERE, 'baseline.log'), 'w', encoding='utf-8').write(_out)
    say('  full suite output in target/negative-control-logs/baseline.log')
    say('  refusing to score any control against a baseline that is not green')
    finish(len(_problems))
say('  ok   %d test(s) ran from %s; all %d named case(s) pass unmutated'
    % (len(BASELINE), ', '.join(os.path.basename(f) for f in _files), len(_named)))
# ROUND 4 (Codex Sol), FINDING C5.1. Round 3 answered "can this control fail?"
# with one fault-injection receipt on the shared scorer and then reported all
# nineteen as demonstrated. A receipt on shared code shows the rejection branch
# is reachable; it says nothing about whether control #12's own anchor, its own
# replacement and its own expected-outcome set are non-vacuous. Those are now
# preconditions, checked per control in every run, and this line is the claim
# they license -- deliberately narrower than "every control was shown to fail".
say('  per-control preconditions, established above for all %d mutation(s):'
    % len(MUTATIONS))
say('    anchor occurs exactly once, so the mutation lands in one known place')
say('    mutant differs from the shipped bytes, so it reverts something')
say('    every case it names exists and PASSES unmutated, so the outcome set '
    'is non-vacuous')
say('  each control below adds: the mutant parses, and the suite goes red on '
    'exactly the cases named')

if ONLY:
    MUTATIONS = [m for m in MUTATIONS if ONLY in m['name']]
    say('  (--only %r: %d of the mutations will run; this is a PARTIAL run)'
        % (ONLY, len(MUTATIONS)))

# Set by the mutation loop; read by the override-lock section. The lock has to
# be shown to ADMIT a legitimate control as well as to refuse everything else,
# or "every case was refused" is also what a lock that refuses everything says.
lock_admitted = None

for m in MUTATIONS:
    proc = None
    out = ''
    states = None
    broken = None
    with tempfile.TemporaryDirectory() as work:
        mutated = os.path.join(work, 'host-process.sh')
        # Stamped with a nonce generated for THIS write and named in the
        # environment: the suite refuses an override that does not carry it, so
        # a leftover at this path cannot be mistaken for what this run meant to
        # test, whatever its timestamp says (round 13e, reopened finding 6).
        text, nonce = stamp(original.replace(m['old'], m['new']))
        io.open(mutated, 'w', encoding='utf-8', newline='').write(text)
        # A mutation must revert a fix, not break the file. Those two produce
        # the same shape of result -- a red suite with the defending case among
        # the failures -- and one of them means the control measured nothing.
        # It has happened here: an apostrophe inside an injected awk comment
        # ended the single-quoted program, and four unrelated cases went red
        # beside a "caught" that had caught nothing.
        #
        # Forward slashes, and `BASH` rather than whatever `bash` resolves to:
        # MSYS bash eats the backslashes out of a `C:\Users\...` argument, and
        # the bash first on PATH here is the WSL launcher, which cannot open
        # the file at all. Between them the first version of this gate reported
        # all fourteen mutants as unparseable -- a guard against a control that
        # measures nothing, itself measuring nothing.
        syntax = subprocess.run([BASH, '-n', mutated.replace('\\', '/')],
                                capture_output=True, text=True)
        # 0 is clean and 2 is a syntax error; ANY other status means the gate
        # did not run -- 127 for a path bash could not open is how the slash
        # bug above announced itself, wearing "the mutant does not parse".
        # A guard that cannot run is not a guard that passed.
        if syntax.returncode not in (0, 2):
            sys.exit('FATAL: bash -n could not check the mutant for %s: exit %d, %r'
                     % (m['name'], syntax.returncode,
                        (syntax.stdout + syntax.stderr).strip()[:200]))
        if syntax.returncode == 2:
            broken = (syntax.stdout + syntax.stderr).strip()
        report = os.path.join(work, 'vitest.json')
        env = dict(os.environ)
        env['WENLAN_HOST_PROCESS_LIB'] = mutated
        env['WENLAN_HOST_PROCESS_LIB_NONCE'] = nonce
        # The suite refuses an override whose flag is not the sha256 of the
        # file's own bytes, so that a pair of values left in a shell cannot
        # silently redirect an ordinary run at a stale copy -- and so that the
        # suite's identity test has something to ASSERT rather than an early
        # return. Only a harness that just wrote the file can name the digest.
        env['WENLAN_HOST_PROCESS_LIB_CONTROL'] = hashlib.sha256(
            text.encode('utf-8')).hexdigest()
        if broken is None:
            assert_suite_unchanged('before the %r run' % m['name'])
            proc = subprocess.run(
                ['cmd', '/c', 'pnpm', 'exec', 'vitest', 'run', SUITE,
                 '--reporter=json', '--outputFile=' + report],
                cwd=ROOT, capture_output=True, text=True, env=env)
            assert_suite_unchanged('during the %r run' % m['name'])
            out = proc.stdout + proc.stderr
        # Per-test STATES, not a scrape of the console. "The name is absent from
        # the failure lines" is satisfied by a test that never ran at all -- a
        # renamed case, a file that failed to load, a filter that matched
        # nothing -- so the old survival check could be answered by silence.
        # That is the defect this whole file is about, in the file itself.
        states, files, raw = read_report(report)
    # The shipped library was never opened for writing. Asserted anyway: this
    # harness exists because "it should be fine" is not a measurement.
    if io.open(LIB, encoding='utf-8', newline='').read() != original:
        sys.exit('FATAL: scripts/lib/host-process.sh changed during the run')

    say('  %s  (%s)' % (m['name'], m['why']))
    mine = 0
    if broken is not None:
        why = broken.splitlines()[0] if broken.strip() else 'bash -n exited nonzero'
        say('    FAIL the mutated library does not parse, so this control '
            'reverted nothing: %s' % why)
        failures += 1
        continue
    if proc is None:
        say('    FAIL the suite never ran; this control measured nothing')
        failures += 1
        continue
    # THE EXECUTION RECEIPT. A report with no tests in it, or one about some
    # other file, is what a runner that silently ran nothing leaves behind.
    for problem in receipt_problems(states, files, 'mutant'):
        say('    FAIL %s' % problem)
        mine += 1
    if mine:
        failures += mine
        io.open(os.path.join(HERE, '%s.log' % m['name']), 'w',
                encoding='utf-8').write(out)
        continue
    # The SAME suite as the baseline, case for case. A mutation of the shell
    # library cannot add or remove a TypeScript test, so a set that differs
    # means the run was not the run the baseline licensed -- a partial load, a
    # filter, a different file -- and every name-keyed check below would be
    # scored against it anyway.
    if set(states) != set(BASELINE):
        gone = sorted(set(BASELINE) - set(states))
        extra = sorted(set(states) - set(BASELINE))
        say('    FAIL the mutant ran a different set of tests than the '
            'baseline; missing %s, unexpected %s' % (gone[:3], extra[:3]))
        failures += 1
        io.open(os.path.join(HERE, '%s.log' % m['name']), 'w',
                encoding='utf-8').write(out)
        continue
    if proc.returncode == 0:
        say('    FAIL the suite PASSED with the fix reverted - it is not '
            'measuring the fix')
        failures += 1
        continue

    for name in m['must_fail']:
        state = states.get(name)
        if state == 'failed':
            say('    ok   caught:   %s (baseline: passed)' % name)
        elif state is None:
            say('    FAIL %r never ran; absence is not a caught defect' % name)
            mine += 1
        else:
            say('    FAIL expected %r to fail, it %s' % (name, state))
            mine += 1
    # ROUND 3 (Codex Sol), FINDING N2. This used to walk `must_survive` -- a
    # hand-picked handful -- and nothing else. A mutation that reddened the
    # whole file satisfied every one of these controls: the must_fail names
    # were red, the four or five hand-listed survivors happened to be among
    # the ones that stayed green, and the eighty cases in between were never
    # looked at. "The suite caught it" then says nothing about WHAT it caught.
    #
    # The survivor set is now computed -- every case the baseline ran, minus
    # this control's own must_fail list -- so it cannot fall behind the suite,
    # and a control is pinned to its discrimination rather than to a sample of
    # it. `must_survive` stays as the control's own statement of intent and is
    # checked for consistency against the computed set.
    expected_fail = set(m['must_fail'])
    collateral = [(n, states.get(n)) for n in sorted(BASELINE)
                  if n not in expected_fail and states.get(n) != 'passed']
    if collateral:
        say('    FAIL %d case(s) outside must_fail did not survive; this '
            'control is not pinned to the property it reverts:'
            % len(collateral))
        for name, state in collateral[:8]:
            say('         %s -> %s' % (name, state or 'absent'))
        if len(collateral) > 8:
            say('         ... and %d more, in full in the log below'
                % (len(collateral) - 8))
        # In full in the log, not just the first eight on screen: the answer to
        # "which cases does this mutation legitimately redden" is the list, and
        # a truncated one costs another ninety-second suite run to recover.
        out += ('\n\nUNDECLARED COLLATERAL (%d), in must_fail order:\n' % len(collateral)
                + ''.join("            %r,  # -> %s\n" % (n, s or 'absent')
                          for n, s in collateral))
        mine += 1
    else:
        say('    ok   survived: every one of the %d other case(s) the '
            'baseline ran' % (len(BASELINE) - len(expected_fail)))
    for name in m['must_survive']:
        if name in expected_fail:
            say('    FAIL %r is named in both must_fail and must_survive; the '
                'control cannot be scored either way' % name)
            mine += 1

    # The lock ADMITTED this override: the suite loaded against a copy, ran the
    # baseline's whole case list, and the identity row -- the one that asserts
    # the library under test is a control and not the shipped file -- passed.
    # Read by the override-lock section below, where "everything was refused"
    # needs a counterexample to mean anything.
    if states.get(IDENTITY) == 'passed':
        lock_admitted = m['name']

    failures += mine
    if mine:
        log = os.path.join(HERE, '%s.log' % m['name'])
        io.open(log, 'w', encoding='utf-8').write(out)
        say('    full suite output in target/negative-control-logs/%s'
            % os.path.basename(log))


# --- controls for the override lock itself ----------------------------------
#
# Round 13b, finding 6. Everything above depends on WENLAN_HOST_PROCESS_LIB
# pointing at the file this harness just wrote. The lock that guarantees it is
# code like any other, and until it is attacked it is only a claim. A stale
# pair of environment variables inherited from a shell is the realistic way it
# breaks -- and under the previous boolean flag the suite's own identity test
# returned early exactly when the flag was set, so the row that watches for
# this was switched off by the thing it was watching for.
def run_suite(lib, flag, work, nonce=None):
    """Run the suite with a given override triple. Returns (proc, states)."""
    report = os.path.join(work, 'vitest.json')
    env = dict(os.environ)
    if lib is None:
        env.pop('WENLAN_HOST_PROCESS_LIB', None)
    else:
        env['WENLAN_HOST_PROCESS_LIB'] = lib
    if flag is None:
        env.pop('WENLAN_HOST_PROCESS_LIB_CONTROL', None)
    else:
        env['WENLAN_HOST_PROCESS_LIB_CONTROL'] = flag
    # Round 13e, reopened finding 6. The digest proves the bytes; only the nonce
    # proves the harness put them there in THIS invocation, which no window over
    # mtime ever could -- the last one accepted a leftover under two seconds old.
    if nonce is None:
        env.pop('WENLAN_HOST_PROCESS_LIB_NONCE', None)
    else:
        env['WENLAN_HOST_PROCESS_LIB_NONCE'] = nonce
    assert_suite_unchanged('before an override-lock run')
    proc = subprocess.run(
        ['cmd', '/c', 'pnpm', 'exec', 'vitest', 'run', SUITE,
         '--reporter=json', '--outputFile=' + report],
        cwd=ROOT, capture_output=True, text=True, env=env)
    assert_suite_unchanged('during an override-lock run')
    states, _files, raw = read_report(report)
    # The report, not just the console: under --reporter=json the reason a suite
    # refused to load lives in testResults[].message and nowhere else, so a
    # control that only read stdout would score "it failed" without ever
    # checking that it failed for the reason under test.
    return proc, states, proc.stdout + proc.stderr + raw


def write_copy(work, text, then=None):
    """A copy of the library in `work`, at the usual path.

    With `then`, the bytes move on AFTER the caller has hashed `text` -- the
    stale-flag case, exactly."""
    path = os.path.join(work, 'host-process.sh')
    io.open(path, 'w', encoding='utf-8', newline='').write(text)
    if then is not None:
        io.open(path, 'w', encoding='utf-8', newline='').write(then)
    return path


def aged_copy(work, text, delta_seconds):
    """A copy whose mtime is moved by `delta_seconds`, contents untouched.

    Here to prove the rule no longer depends on the timestamp at ALL. Round 13c
    answered the authorship question with a fifteen-minute age window and round
    13d with a two-second one; round 13e walked through the second, because a
    leftover younger than the window is inside every window. Under the nonce
    rule an hour-old file and a one-second-old file are refused for the same
    reason and a file of any age carrying this run's nonce is accepted.
    """
    path = write_copy(work, text)
    when = time.time() + delta_seconds
    os.utime(path, (when, when))
    return path


say('override lock (the harness\'s own aim)')

# ROUND 3, FINDING N1, the override-lock half: "do the same for the
# override-lock cases". Every case below asserts a REFUSAL, and a lock that
# refused everything -- one wired shut by a typo in an environment variable
# name, say -- would satisfy all nine of them. The counterexample is the one
# thing that separates a lock from a wall, and it is free: the mutation runs
# above handed the suite a legitimate override triple, and the suite loaded,
# ran the baseline's whole case list and passed the identity row.
if lock_admitted:
    say('  ok   lock-admits-a-legitimate-control (%s loaded and its identity '
        'row passed)' % lock_admitted)
else:
    say('  FAIL no mutation above got past the lock with a valid triple; the '
        'refusals below are what a lock that refuses EVERYTHING also produces')
    failures += 1

if ONLY:
    say('  (--only: the nine refusal cases are skipped in a partial run)')
    finish(failures)

digest = hashlib.sha256(original.encode('utf-8')).hexdigest()
stamped_original, original_nonce = stamp(original)
stamped_original_digest = hashlib.sha256(
    stamped_original.encode('utf-8')).hexdigest()
sample, sample_nonce = stamp(
    original.replace(MUTATIONS[0]['old'], MUTATIONS[0]['new']))
sample_digest = hashlib.sha256(sample.encode('utf-8')).hexdigest()
# A nonce from some OTHER run. Never handed to the suite; the leftover cases
# below are files stamped with it while this run declares a different one.
stale_sample, stale_nonce = stamp(
    original.replace(MUTATIONS[0]['old'], MUTATIONS[0]['new']))
stale_sample_digest = hashlib.sha256(stale_sample.encode('utf-8')).hexdigest()

for name, make, expect_load, needle in (
    ('flag-absent',
     lambda w: (write_copy(w, sample), None, sample_nonce),
     False, 'not the sha256 of its contents'),
    ('flag-stale-after-the-file-moved-on',
     lambda w: (write_copy(w, sample, then=stamped_original), sample_digest,
                sample_nonce),
     False, 'not the sha256 of its contents'),
    ('override-is-the-shipped-file',
     lambda w: (LIB, digest, None),
     True, 'a control must be a copy, never the shipped file'),
    # The shipped library with a nonce line and nothing else changed. It hashes
    # to its flag, it carries this run's nonce, and it still mutates nothing --
    # which is why the identity row strips the nonce before comparing, and why
    # this control exists to prove the strip did not turn that row off.
    ('override-is-an-identical-copy',
     lambda w: (write_copy(w, stamped_original), stamped_original_digest,
                original_nonce),
     True, 'an override identical to the shipped library is not a control'),
    # Round 13c finding 6, reopened through 13d and 13e: the ones the digest
    # cannot see. Every file below hashes to exactly what the flag says. Only
    # the nonce says it is not this run's.
    #
    # THE ROUND-13e CASE. This file's mtime is NOW -- zero seconds old, well
    # inside the two-second slack the previous rule had to allow for coarse
    # filesystem timestamps, and therefore accepted by it. It is still a
    # leftover: it carries a nonce this run never generated.
    ('override-is-a-leftover-written-this-instant',
     lambda w: (write_copy(w, stale_sample), stale_sample_digest, sample_nonce),
     False, 'copy left behind by an earlier write'),
    # And the same refusal for a file an hour old, to show the rule stopped
    # reasoning from time rather than merely tightening its window.
    ('override-is-a-leftover-from-an-hour-ago',
     lambda w: (aged_copy(w, stale_sample, -3600), stale_sample_digest,
                sample_nonce),
     False, 'copy left behind by an earlier write'),
    # A harness that cannot name a nonce has not measured anything either:
    # without it there is nothing for the file's contents to have to match.
    ('nonce-absent',
     lambda w: (write_copy(w, sample), sample_digest, None),
     False, 'is not a nonce this run generated'),
    ('nonce-is-not-a-nonce',
     lambda w: (write_copy(w, sample), sample_digest, 'yesterday'),
     False, 'is not a nonce this run generated'),
    # An unstamped copy: the flag matches its bytes exactly, and there is
    # nothing in it that only this run could have written.
    ('override-carries-no-nonce-at-all',
     lambda w: (write_copy(w, original.replace(MUTATIONS[0]['old'],
                                               MUTATIONS[0]['new'])),
                hashlib.sha256(original.replace(
                    MUTATIONS[0]['old'], MUTATIONS[0]['new']
                ).encode('utf-8')).hexdigest(), sample_nonce),
     False, 'the file carries no nonce'),
):
    with tempfile.TemporaryDirectory() as work:
        lib, flag, nonce = make(work)
        proc, states, out = run_suite(lib, flag, work, nonce)
    ok = True
    if proc.returncode == 0:
        say('  FAIL %s: the suite passed; the lock did not hold' % name)
        ok = False
    elif expect_load:
        # The suite must LOAD and the identity row must be the thing that fails
        # -- a load-time throw here would mean the row is untested. And it must
        # be the SAME case list the baseline ran: a partial load that happens to
        # include the identity row would satisfy the row check while proving
        # nothing about the lock.
        if not states or states.get(IDENTITY) != 'failed':
            say('  FAIL %s: expected the identity row to fail, got %r'
                % (name, (states or {}).get(IDENTITY)))
            ok = False
        elif set(states) != set(BASELINE):
            say('  FAIL %s: the suite loaded a different case list than the '
                'baseline (%d vs %d); this is not the run the case describes'
                % (name, len(states), len(BASELINE)))
            ok = False
        else:
            # ROUND 3 (Codex Sol), FINDING N2, applied here too. The identity
            # row failing is necessary and not sufficient: these two cases hand
            # the suite a library that BEHAVES correctly (the shipped file, or
            # a byte-identical copy of it), so the identity row is the only
            # case that has any business being red. If the rest went red as
            # well, the run is red for some other reason and the lock was
            # credited for it.
            others = [(n, states.get(n)) for n in sorted(BASELINE)
                      if n != IDENTITY and states.get(n) != 'passed']
            if others:
                say('  FAIL %s: the identity row failed and so did %d other '
                    'case(s); this run is red for some other reason'
                    % (name, len(others)))
                for _n, _s in others[:5]:
                    say('         %s -> %s' % (_n, _s or 'absent'))
                ok = False
    else:
        # A load-time refusal: no report at all, or one with nothing in it.
        if states:
            say('  FAIL %s: the suite ran %d test(s); it should have refused '
                'to load' % (name, len(states)))
            ok = False
    if needle not in out:
        say('  FAIL %s: the refusal never said why (%r absent)' % (name, needle))
        ok = False
    if ok:
        say('  ok   %s (exit %d)' % (name, proc.returncode))
    else:
        failures += 1
        io.open(os.path.join(HERE, '%s.log' % name), 'w', encoding='utf-8').write(out)

finish(failures)
