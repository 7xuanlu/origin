# Negative controls

A test that passes proves the test passes. It does not prove the test would
notice the bug. Every harness here removes exactly one half of a shipped
remedy, re-runs the suite that defends it, and **fails when the suite stays
green** — so the assertion it guards is measured rather than assumed.

They live in the tracked tree on purpose. They used to sit under
`target/windows-track-evidence/`, which is gitignored: the fixes shipped, the
controls did not, and a fresh checkout could not re-run a single one. A control
nobody else can run is a claim, not a control.

There are **ten executable harnesses** and one runner. This file is not a
harness; running the harnesses is not something you do by reading a table and
typing ten command lines, which is why `run-all.sh` exists.

| Harness | Defends | Subject |
| --- | --- | --- |
| `posix-probes-negative-controls.py` | `scripts/host-process.test.ts` | `scripts/lib/host-process.sh` — the tri-state port/liveness/image probes, and the suite's own override lock |
| `lib-ps1-negative-controls.sh` | `scripts/first-run/lib.test.ps1` | `scripts/first-run/lib.ps1` — the declared-row contract, the ledger read and write paths, `Check-Helper` |
| `authenticode-step-receipt.py` | `scripts/release-workflow-contract.test.py` | `release.yml`'s `Verify the installer carries a valid SignPath signature`, run against real signature states this host produces — see the caveat below |
| `a-drift-guard-inventory.py` | itself | `crates/wenlan-core/src/drift_guard.rs` — pins a digest per function reachable from a `release.yml` site, so a tooth cannot be edited without a re-read |
| `a-drift-guard-replica.py` | itself | the `drift_guard` teeth that read `release.yml`, re-implemented in Python because `cargo test -p wenlan-core` needs a Vulkan SDK |
| `port-precheck-controls.sh` | itself | `scripts/first-run/port-precheck.sh` — the shared-port measurement and the ledger row that carries its verdict. It supplies the cases as well as the controls, because the script had none |
| `dev-runtime-scan-controls.sh` | itself | `scripts/dev-runtime.sh`'s `reap_staged_daemon` — the `ps -W` scan that decides what to kill, and the WINPID and image path it hands the kill helper. Also case-less before this; the function is extracted by brace matching because the script dispatches on `$1` and cannot be sourced. The parse itself now lives in `scripts/lib/host-process.sh`, so the mutations target a written copy of the library and one targets the call that reaches it |
| `dev-runtime-record-controls.sh` | itself | `scripts/dev-runtime.sh`'s `read_owned_pid`, `list_dir_tristate` and `listing_has_name` — the three answers (`0` recorded / `1` nothing recorded / `2` recorded but unreadable) that `stop_runtime` and `start_runtime` branch on before anything is deleted or started, and the `ls`/`grep`/`sed` chain each of those answers is assembled from |
| `dev-runtime-stage-controls.sh` | itself | `scripts/dev-runtime.sh`'s `stage_windows_daemon` — the daemon and DLL copies that decide, by CONTENT and not by mtime, what the recorded server path points at, the re-read that proves what landed, the tri-state listing of the directory they come from, and the call site in `start_runtime` that reaches all of it. Extracted by brace matching for the same reason as the two above |
| `windows-probes-negative-controls.py` | itself | `scripts/first-run/windows-zip.ps1` and `scripts/first-run/windows-nsis.ps1` — the port, health and process-liveness probes, and the `Check` blocks that branch on them. Case-less before this; the probes are *extracted* rather than run, because neither channel script can be executed on a developer machine (each deletes `%LOCALAPPDATA%\wenlan`, the real memorydb and config) |

## Running them

```bash
bash scripts/negative-controls/run-all.sh          # all ten, one receipt
bash scripts/negative-controls/run-all.sh --list
bash scripts/negative-controls/run-all.sh --only replica,inventory
```

Or one at a time:

```bash
python3 scripts/negative-controls/posix-probes-negative-controls.py
bash    scripts/negative-controls/lib-ps1-negative-controls.sh
python3 scripts/negative-controls/authenticode-step-receipt.py
python3 scripts/negative-controls/a-drift-guard-inventory.py
python3 scripts/negative-controls/a-drift-guard-replica.py
bash    scripts/negative-controls/port-precheck-controls.sh
bash    scripts/negative-controls/dev-runtime-scan-controls.sh
bash    scripts/negative-controls/dev-runtime-record-controls.sh
bash    scripts/negative-controls/dev-runtime-stage-controls.sh
python3 scripts/negative-controls/windows-probes-negative-controls.py
```

A full sweep is an hour. Measured, from one clean bound run of `run-all.sh` on
this host (Windows 11, MSYS2 bash 4.4.23, Python 3.10.0) — `elapsed=3588s`,
`registered=10 ran=10 ok=10`:

```
posix-probes 1685s   authenticode 547s   windows-probes 405s   scan 327s
lib-ps1       397s   stage         92s   record          85s
port-precheck  21s   inventory     11s   replica          4s
```

Read those as a floor, not a spec: the same suite on a cold host earlier the
same morning took 1999 s for `posix-probes` and 1470 s for `authenticode`, so
the first sweep after a reboot is meaningfully slower than the second.

`posix-probes-negative-controls.py` dominates because it pays for one vitest
invocation per mutation: 19 mutations, one baseline, and the single
override-lock case that loads a real control, so 21 invocations at ≈80 s.
`authenticode-step-receipt.py` runs the release contract suite once per arm.
`windows-probes-negative-controls.py` runs 72 cases against the shipped source
and then the affected subject's cases once per control, each in its own
PowerShell process.

That harness also prints the count breakdown the rest of this directory should
copy, because collapsing these three into one number is how a suite flatters
itself:

```
CASES AGAINST THE SHIPPED SOURCE: 72 (every one must pass; a red case here is a defect in the shipped script, not in a control)
MUTANTS APPLIED: 47 (one source mutation per control)
RED CASE OUTCOMES REQUIRED AND OBSERVED: 112 (sum across those 47 mutants, not a mutation count; any unlisted red outcome is a failure)
CONTROL FAILURES: 0
```

The third line is the one worth reading twice. 112 is not 112 mutations — it is
the total number of case-level red outcomes those 47 mutants were required to
produce and did, and the word "unlisted" is load-bearing: a red outcome nobody
declared is a control failure, not a bonus. A harness that reported only
"47 controls, 0 failures" would look identical whether it required one red
apiece or a hundred and twelve.

`dev-runtime-scan-controls.sh`'s five and a half minutes are almost all sleep —
several of its cases exhaust `reap_staged_daemon`'s bounded fifty-round poll and
every round waits. The bottom three in the table are seconds because their
subjects are read, not run.

Transcripts go to `target/negative-control-logs/`, which is gitignored;
`run-all.sh` writes one log per harness under `run-all/`.

### What the aggregate result means, and what it does not

`run-all.sh` does not trust an exit status and does not trust a transcript that
merely looks finished. Each harness declares a **completion contract**: a
terminal line printed last, and only after every one of its controls has been
scored.

- The six harnesses on this track print
  `NEGATIVE-CONTROL COMPLETE <file> failures=N elapsed=...` from an exit path —
  a Python `atexit` handler guarded by a `_COMPLETED` flag, or a bash `EXIT`
  trap guarded by `reached_end` — so a `set -u` abort, a refusal, a signal and
  a watchdog kill all land on `NEGATIVE-CONTROL ABORTED` instead, and none of
  them can be mistaken for a clean run. This is reported to have happened while
  these were being written — another lane editing `scripts/lib/host-process.sh`
  twice inside a twenty-five-minute POSIX run, the harness refusing with
  `FATAL: scripts/lib/host-process.sh changed during the run`, the runner
  reporting `DID-NOT-COMPLETE` over five controls that had already passed. No
  transcript of those runs was kept, so read that as what the mechanism is built
  to do and not as a receipt; the receipt is that you can reproduce it by
  touching the file mid-run.
- The other four — `dev-runtime-scan-controls.sh`,
  `dev-runtime-record-controls.sh`, `dev-runtime-stage-controls.sh` and
  `windows-probes-negative-controls.py` — end with `CONTROL FAILURES: N`
  instead. That is *their* contract, recorded in the runner's registry rather
  than imposed by it. It is weaker: an ordinary print rather than an exit-path
  invariant, so it proves the harness reached its last statement and nothing
  about how it would have exited elsewhere. They are not upgraded here because
  they belong to other lanes and are edited concurrently; upgrading one is four
  lines (a `MARKER`/`HARNESS` pair, an `EXIT` trap, a `reached_end` flag) and
  moving its registry row from `summary` to `marker`, and it should be done by
  whoever owns the file. Until then the runner reports what each file actually
  guarantees rather than what would be tidier.

The runner keeps four verdicts apart, and only the first is a pass:

| Verdict | Means |
| --- | --- |
| `ok` | completed, and every control fired |
| `CONTROLS-FAILED` | completed, and at least one control did not fire |
| `DID-NOT-COMPLETE` | no terminal marker — killed, refused, crashed, or run with `--only`. **Unchecked, never a pass.** |
| `CONTRADICTORY` | the marker says `failures=0` and the process exited non-zero. Do not pick one to believe |

A precondition this host cannot satisfy — no PowerShell, no Authenticode-signed
fixture on the machine, no `pnpm` — produces `DID-NOT-COMPLETE`, which is the
honest place for it. A `--only` run and a run where any harness was skipped is
stamped `PARTIAL` and exits non-zero, because a subset is not a result about
the suite.

The runner's registry is checked against the directory before anything runs: a
`.sh` or `.py` file here that is not registered is a `REGISTRY GAP` and the
sweep refuses. That is the one failure mode a hand-maintained list has that
flatters us — an eleventh harness that is simply never run, under a clean
report of the ten that were.

**What a clean sweep does not mean.** It is a statement about ten harnesses,
not about the repository. Every harness here reverts a *named* property; a
defect nobody thought of has no control and is not measured by one.

**Seven of the ten defend *themselves*** — count the `itself` rows in the
`Defends` column: `a-drift-guard-inventory.py`, `a-drift-guard-replica.py`,
`port-precheck-controls.sh`, `dev-runtime-scan-controls.sh`,
`dev-runtime-record-controls.sh`, `dev-runtime-stage-controls.sh` and
`windows-probes-negative-controls.py`. They supply the cases as well as the
controls, because the subject had no suite at all, so what they establish is
that their own extracted cases would notice the reversion, not that any CI lane
would. Only three point at a suite that exists independently of the harness —
written for the subject, not extracted by the control that grades it:
`posix-probes-negative-controls.py` at `scripts/host-process.test.ts`,
`lib-ps1-negative-controls.sh` at `first-run/lib.test.ps1`, and
`authenticode-step-receipt.py` at `scripts/release-workflow-contract.test.py`.
That ratio is the single most important thing on this page, which is why it is
a number here and not the word "several" — seven tenths of this suite is a
closed loop marking its own homework.

And none of this runs in CI: no PR lane
on this repo runs on Windows, so all ten are a pre-merge step by hand for
anyone touching the files in the table above.

### What the Authenticode receipt does and does not measure

Four of its six rows run the shipped step's body, unmodified, against a
signature state this host actually produced: no file, a signed binary with a
byte appended (`NotSigned`), one with a byte patched in the middle
(`HashMismatch`), and an untouched signed binary whose publisher is not
SignPath. Windows answers each one; nothing is stubbed.

The other two rows are the ones named in the suite's
`AUTHENTICODE_SUBSTITUTED_KINDS`, and they are **not** the shipped body
verbatim — the receipt's closing line derives its verbatim/substituted counts
from that same constant rather than restating them, so adding a third
substituted row cannot leave this claim behind. There is no SignPath-signed
file on any host that runs this, so both rows rewrite the step's expected
publisher from `SignPath Foundation` to whatever publisher this host's fixture
actually carries: `accepted` uses it as-is and is the one row that reaches
`PASS`, and `casefolded` swaps its case, which the case-sensitive comparison
must reject — the only row in the table that can tell `-cne` from the `-ne`
that shipped. What they prove is that a *valid signature from the expected
publisher* reaches PASS while one differing only in case does not — the `Valid`
check, the null-certificate check, the `SimpleName` parse and the PASS are all
shipped code over a real signature. What it does not prove is that
the literal string `SignPath Foundation` is the one being compared; a separate
static contract pins that literal, and the receipt's own mutation of the
publisher check is what shows the comparison is load-bearing. Only a real
SignPath-signed installer closes the gap, and that arrives with the Foundation
application, not before.

### How the harnesses keep themselves honest

Each of these was added because the harness itself could have failed in a way
that looked like a pass, or like an unrelated red.

- **The override is locked by a nonce, not by a timestamp.**
  `posix-probes-negative-controls.py` hands `scripts/host-process.test.ts` a
  mutated copy of the library through `WENLAN_HOST_PROCESS_LIB`, and the suite
  refuses to run against a copy it cannot prove this run wrote. It used to
  prove that with an mtime slack window, which a leftover copy written inside
  the window satisfies. The harness now generates a nonce per mutation, stamps
  it into the copy as a `# wenlan-control-nonce:` line, and passes it in
  `WENLAN_HOST_PROCESS_LIB_NONCE`; the suite requires exactly one carried nonce
  and requires it to be that one. Nine cases in the harness drive that lock,
  including the leftover-written-this-instant case the mtime rule accepted.
- **Every mutant is parsed before it is scored.** A mutation that breaks the
  file's syntax reverts nothing, and its "the suite went red" reads as a caught
  defect. Each mutant now goes through `bash -n` first, and a mutant that does
  not parse is reported as a control failure naming the syntax error — not as a
  success. The gate itself must be a measurement, so a `bash -n` that cannot
  run at all (status other than 0 or 2) is fatal.

  Round 5 found the PowerShell half of that gate weaker than it read.
  `windows-probes`'s `parse_ok` returns `parses` / `syntax` / `unmeasured`, but
  the syntax arm was `returncode == 1 and out.strip()` — three-valued in shape,
  two-valued in meaning, because *any* noise on exit 1 counted. A checker that
  printed `PowerShell host initialization failed` and exited would have been
  called a syntax verdict, and every control in the run failed as "the mutant
  does not parse" over a mutant that was fine. The checker prints `line <n>:
  <message>` for each syntax error it finds, so that shape is now the criterion
  and a noisy off-contract exit 1 is `unmeasured`. A fifth probe in the
  self-check *is* that checker, so the criterion is watched refusing something
  every run rather than asserted in a docstring; writing the mutant to disk can
  no longer escape as an exception either.
- **A driver that died is not a control that fired.** Round 5's top finding, and
  the same defect one level in from where the harness was looking for it.
  `windows-probes`'s `run_case` already separated "the driver produced no row,
  it died rather than answering" from "the case failed" — and then `run_all`
  wrote `(passed if ok else failed).append(...)`, and the control scorer read
  `want in mfailed` as *caught*. So a mutant that killed the driver scored as a
  control that caught the reverted defect, with a green line in the transcript.
  `run_all` returns three lists now; a case that could not be measured is a
  control failure whether or not the control was hoping to see it go red, and it
  cannot slip past the unpinned-collateral check either. The scoring moved into
  `score_control` for one reason: it can be driven with synthetic buckets in the
  self-check, so the refusal is demonstrated in every run. `RED CASE OUTCOMES
  REQUIRED AND OBSERVED` also split into two numbers — one number under that
  label asserted the equality it was meant to report.

  That refusal was correct and it was not sufficient. It left
  `nc-stop-daemon-native-stderr-into-the-error-stream` permanently unmeasurable:
  the mutation puts back a native call that throws under an inherited
  `$ErrorActionPreference = 'Stop'`, and it throws at the SETUP statement
  (`$stopped = Stop-Daemon`), which runs outside `Check` because that is where
  the shipped script runs it. The mutation broke something real; the harness
  simply had no way to say what. A control stuck at "could not measure" forever
  is not a control. So the setup is part of the measured surface now: it is
  wrapped in a `try`/`catch` that prints one tab-separated `SETUP-THREW` record
  — exception type, message, and the driver line that raised — before exiting.
  A driver that dies *with* that record is a case that FAILED for a named
  reason; a driver that dies without one is still `unmeasured`. The split is
  general, not a special case for that control: any mutation that kills a setup
  statement is a legitimate way for a defect to manifest, and all 41
  setup-driven cases carry the arm. `try`/`catch` is not a new scope in
  PowerShell, so nothing changes when the setup does not throw. Four
  hand-written drivers push all four outcomes through the real subprocess path
  in the self-check — including a check that the tab survives PowerShell, since
  the branch turns on an escaping contract that could silently stop working.
- **A failure message must point at evidence that agrees with it.** Every
  control re-runs its channel's whole case list, and each run wrote
  `case-<name>.log`, so by the end of a sweep that file held the *last* run of
  that case — typically a later control's, typically green. A control failure
  therefore pointed the reader at a transcript showing `ROW … PASS`, which is
  worse than no pointer at all: the file looks authoritative and contradicts the
  message. Control runs write `<control>--case-<name>.log` now and the messages
  name that path. Only non-passing cases are kept under a control (58 controls
  times a channel's case list is thousands of files, and the green ones are the
  copies nobody opens); the shipped-source sweep still keeps all of them.
- **A receipt's digest has to look like a digest.** `run-all.sh` bound its
  receipt to `sha256sum` output validated only by `[ -z … ]`. A shim, alias or
  broken PATH entry printing `UNAVAILABLE` at both hashings compared equal to
  itself, and the sweep certified a receipt bound to nothing — a failed
  measurement wearing a negative one's clothes, inside the apparatus built to
  tell those apart. Both the capture and the end-of-sweep comparison now require
  64 lowercase hex characters, anything else is could-not-measure and never
  certifies, and the end-of-run headline distinguishes "these files moved" from
  "this file's end state could not be read" because they send the next reader to
  different places. The shape check is shown refusing four non-digests at the
  top of every sweep. What is still conceded, and stated in the file: an edit
  made *and reverted* between the two hashings is invisible. Closing that means
  watching the filesystem for the whole hour, and this runner does not.
- **An exit path that succeeded must not print the abort marker.**
  `a-drift-guard-inventory.py --print-digests` printed its digests and returned
  without setting `_COMPLETED`, so the `atexit` handler stamped
  `NEGATIVE-CONTROL ABORTED` and "Nothing above it is a result" over a listing
  while the process exited `0` — a transcript contradicting its own status, in
  the file that audits exactly that. It sets the flag and says what the mode is
  now. It deliberately does *not* print the completion MARKER: `run-all.sh`
  reads the last line, and a `failures=0` marker from a mode that runs no
  controls would be a clean result minted out of a listing.
- **The bash that runs the gate is measured, not assumed.** On Windows the
  first `bash` on `PATH` is `C:\Windows\System32\bash.exe`, the WSL launcher,
  which exits 127 on a `C:/Users/...` path; and MSYS bash eats the backslashes
  out of a `C:\Users\...` argument. `_bash()` writes a probe script and tries
  candidates until one reads it, and paths are handed over with forward
  slashes.
- **Name-keyed state refuses duplicate names.** The harnesses key each case's
  result by its title. Two vitest tests sharing a title collapsed to one entry,
  so a mutation could redden the shadowed one invisibly. `collect_states()`
  raises on a duplicate title rather than merging.
- **A check that needs an optional dependency says so when it does not have
  it.** The `continue-on-error` rule reads the decoded YAML, which needs PyYAML,
  and the CI lane that runs this suite installs nothing. Without a parser it
  falls back to a token scan and reports "unmeasured, not clean" for the two
  constructs that scan is structurally blind to — a YAML alias, and a
  double-quoted key carrying a backslash escape — and reports nothing for a file
  containing neither. `authenticode-step-receipt.py` drives all three of those
  states with `sys.modules["yaml"] = None`, so the fallback is measured rather
  than trusted.
- **A subject that must never run is extracted, and its anchors are proven
  before anything is scored.** `windows-probes-negative-controls.py` cannot
  execute either Windows channel: both call `Remove-Item -Recurse -Force` on
  `%LOCALAPPDATA%\wenlan`, which on a developer machine is real user data. So it
  lifts the probe functions and the `Check` blocks out of the shipped text by
  brace matching, the way `dev-runtime-scan-controls.sh` lifts
  `reap_staged_daemon`. That makes every anchor a place the source could drift
  away from, so a pre-check validates all of them — each extraction, each
  mutation anchor's exactly-once match, and each case name a control claims to
  pin — against the real source, and a stale one is a hard error before a single
  case runs. A driver that produces no verdict line is reported as having died
  rather than counted as a case that failed, because inside a control a death
  would read as a defect caught.
- **An anchor must match EXACTLY once, and the count is taken.** Four of the
  shell harnesses located their mutation with `head="${text%%"$old"*}"` and then
  checked that the halves recombine and that the head is not the whole text.
  That establishes *at least* once: `%%` stops at the first occurrence, so a
  doubled anchor recombines just as well, the first copy is mutated, the second
  is left alone, and the subject is a third program nobody wrote — reported on
  as if it were the reverted one. `count_occurrences` now counts, and anything
  but one is a hard error, zero included. Anchor rot is the ordinary case, not
  an exotic one — these anchors are literal spans of a file six lanes edit. It
  is reported to have caught a stale anchor in `dev-runtime-record-controls.sh`
  when it first landed, whose data-dir read had moved into a `case` arm; no
  transcript of that was kept, so take it as illustration. What *is* on the
  record: running `posix-probes-negative-controls.py` on 2026-08-31 at 06:52,
  minutes after another lane edited `scripts/lib/host-process.sh`, printed
  `nc-netstat-schema-is-any-tcp-line: anchor matched 0 times; the control would
  test nothing` and `2 of 19 anchors no longer name one place in
  scripts/lib/host-process.sh`, and refused to run. Two controls had silently
  stopped reverting anything, in the time it took to write this paragraph.

  A zero count also has to say *which* zero it is. The commonest cause here is
  not a code change: a whole-file rewrite that flips the line endings drops
  every multi-line anchor to zero at once, while every single-line anchor keeps
  matching — so it reads as a handful of unrelated controls rotting rather than
  one cosmetic edit landing. `posix-probes` re-tests each zero-count anchor
  against a CRLF→LF normalisation of the subject and, when that matches once,
  says `the line endings moved, not the code`, with the file's CRLF count
  beside it. It does **not** normalise and carry on: a mutant written with
  different line endings than its subject is a different program, and for a
  shell library a CRLF one is broken. It names the cause and still refuses.
- **A control that drives a function proves something calls it.** Every case in
  `dev-runtime-stage-controls.sh` extracts `stage_windows_daemon` and runs it
  directly, so all of them stay green when nothing invokes it. It now also reads
  the shipped call site out of `start_runtime` — called, guarded by the Windows
  branch, failure propagated — and two controls break exactly that: one deletes
  the call, one swallows its status with `|| true`.
- **An unreadable fixture is proven unreadable.**
  `dev-runtime-record-controls.sh` builds its "present but unreadable" record
  files as directories, and measures on the host that `[[ -e ]]` accepts one
  and `sed` refuses it before any case runs. `chmod 000` does not work on this
  filesystem — measured: `sed` still reads the file — so a permissions-based
  fixture would have been silently readable and every case would still have
  been green.

- **A mutation is credited only against a baseline taken in the same run.**
  `posix-probes-negative-controls.py` ran nineteen mutations and never once ran
  the suite unmutated. A `must_fail` test that was already red — for any reason,
  including a half-saved edit in another worktree — made every mutation that
  named it look like a control that fired. It now runs the suite with no
  override first, requires every `must_fail` and `must_survive` case named
  anywhere in the harness to be *passing* there, and refuses to score a single
  mutation otherwise. `port-precheck-controls.sh`, `lib-ps1-negative-controls.sh`
  and `authenticode-step-receipt.py` take the same baseline; `a-drift-guard-replica.py`
  additionally refuses a control whose check id is already violated by the
  shipped file, which is the same property for a harness with no test runner.
- **The suite has to prove it ran.** "The suite went red" and "the suite never
  started" reach the harness as the same non-zero status. Each vitest run is
  read back from its JSON reporter: a run that reported zero tests, or that ran
  a file other than the one the control is about, is a control failure naming
  the discrepancy rather than a mutation that was caught.
- **Every control is pinned to ALL survivors, not to selected ones.** A control
  that names three cases it expects to fail and checks nothing else is green
  when the mutation reddens the whole file. Each control now declares the cases
  that must fail *and* the cases that must survive, and the survivor list is the
  rest of the suite rather than a hand-picked few — so a mutation with
  collateral damage is reported as unpinned, not as a success.
- **A check with no control is a failure, and the list cannot go stale.**
  `a-drift-guard-replica.py` implemented sixteen checks and six controls; the
  other ten were never driven. Every check now tags its violation with an id,
  the ids are read back out of the function's own source with `inspect.getsource`,
  and a meta-control fails if any id has no mutation that provokes it — or if a
  mutation provokes an id it did not declare. Adding a check without a control
  fails the harness on the next run, which is the only version of this rule that
  survives contact with a codebase being edited.
- **An anchor must match exactly once — in the Python harnesses too.**
  `a-drift-guard-replica.py` located its mutations with
  `text.replace(old, new, 1)`. A duplicated anchor mutates the first occurrence,
  leaves the second, and the control reports on a workflow nobody wrote. `once`,
  `once_re` and `once_in_job` count first and raise `StaleAnchor` on anything
  but one, zero included; `job_span` refuses a duplicate job key for the same
  reason. `authenticode-step-receipt.py` stopped hand-listing the steps it
  mutates and enumerates them out of the job instead, because a hand-list goes
  stale in the direction that flatters us: a renamed step is loud, an *added*
  step is silent.
- **What cannot be measured is refused, not reported as a short answer.**
  `a-drift-guard-inventory.py` seeded its closure on the literal
  `workflows/release.yml`, walked one file, and keyed its rows by bare function
  name. So a path assembled at runtime, a helper in a `#[path]` module, an
  `include!`, and a function whose name is also used elsewhere were each
  invisible — and invisibility read as "no such tooth". It now seeds on
  `release.yml`, loads the module tree, qualifies every name by its enclosing
  `mod`/`impl`/`trait`/`fn`, reports a duplicate definition as a `DUPLICATE` row
  rather than merging it, and raises `Unmeasurable` — a refusal, not a short
  inventory — for an `include!`, an assembled workflow path, or an
  attribute macro on a function inside the closure. Brace matching runs over a
  masked copy of the source with strings and comments blanked and offsets
  preserved, so a brace inside a string literal cannot end a function early.
- **A refusal arm is tied to the refusal, not to any non-zero exit.** The four
  arms of `authenticode-step-receipt.py`'s classification test used to accept
  "exited non-zero and the phrase appears somewhere in the output". A copy that
  died on an unrelated import error satisfies both. Each arm now reads the
  *terminating* `AssertionError` and requires the phrase to be in that; the one
  arm that must succeed requires exit 0, no `AssertionError` anywhere, and the
  terminal `PASS` line. The number of mutations an arm must name as unchecked is
  measured from the degraded-host section in the same run rather than compared
  against a literal, which is the point whether or not the literal it replaced
  had already drifted — a hand-written count in an assertion about counts is a
  second copy of the fact, and second copies are what this directory exists to
  catch.
- **The subject may not move under the run.** A sweep is an hour and a harness
  is minutes, which is long enough for another lane to land an edit halfway
  through — and then the controls scored before it and the ones scored after it
  are about different files, with nothing in the transcript to say which. Every
  harness whose subject is read more than once now snapshots it at the start and
  refuses at the end: `posix-probes` and `port-precheck` on
  `scripts/lib/host-process.sh`, `lib-ps1` on `first-run/lib.ps1` and its suite,
  `authenticode` on `release.yml` and the contract test, the three
  `dev-runtime-*` harnesses on `dev-runtime.sh`, `windows-probes` on both
  channel scripts. `FATAL: <file> changed during the run` is the whole result,
  not a warning beside one. (`posix-probes` and `lib-ps1` re-check after every
  control rather than only at the end, so the refusal names the control it was
  about to score. That is not a refinement — it is the difference between the
  guard running and not running. A check placed only at the bottom of the file
  never executes in a run that does not reach the bottom, and the runs that do
  not reach it are the long ones, which is to say the ones with the most time
  for another lane to land an edit. Found the plain way: a watchdog killed the
  `lib-ps1` harness at its ninth control while another lane was rewriting
  `lib.ps1` underneath it, and the end-of-file check never ran.)
  `a-drift-guard-inventory.py` and
  `a-drift-guard-replica.py` are the two exceptions and are exempt for a reason
  rather than by omission: each reads its subject exactly once and works from
  that text, so there is no window to guard.

  Round 4 found the hole in that: `posix-probes` guarded the IMPLEMENTATION and
  left the file that produces its EVIDENCE — `scripts/host-process.test.ts`,
  read twenty-one times over twenty-eight minutes — unguarded, and the lane that
  owns the library owns the suite too. A same-title edit to a test body defeats
  every other check at once: the case-set receipt still matches the baseline
  name for name, the library guard is still green, and the assertion behind the
  name is a different assertion. It is guarded now, and checked **immediately
  before and immediately after every suite invocation** rather than once at each
  end of the run — because start/end equality answers "did the file finish where
  it started", which an edit that is later reverted also satisfies, and which
  says nothing about which side of an edit any individual control was scored on.

  The residual, since a bounded claim is the point of this whole directory: an
  edit that lands after the before-check and is reverted before the after-check
  is invisible to this and to any polling scheme. What the guard establishes is
  that the file was these bytes on both sides of each measurement — not that it
  was constant throughout.
- **The aggregate receipt names the revisions it covers.** A sweep is an hour
  and the registry used to be validated once, at the top; everything
  after that was added up as one result about one suite. But six lanes work in
  this tree at once, and a harness edited at minute thirty makes the bottom line
  a claim about a set of files that never existed together at any instant.
  `run-all.sh` now hashes every registered harness — and itself — at registry
  validation, prints the manifest under `harness revisions this receipt covers`,
  and re-hashes before it prints any verdict. Drift is `SNAPSHOT MOVED`, the
  verdict becomes `UNBOUND`, and the drift is counted into the marker's
  `failures=` rather than living only in a prose line: this runner refuses a
  child whose marker says `failures=0` beside a non-zero exit, and printing that
  same shape itself would be the identical defect one level up.

  This one is not a capability claim. It fired on its first full sweep, against
  an edit nobody arranged: `windows-probes-negative-controls.py` was hashed
  `968bd9f58a810d4a` when the sweep validated the registry and
  `7c286e0c2b17d3cd` when it finished, having been rewritten by another lane
  about eight minutes in. Nine harnesses had already come back `ok`. The
  transcript is:

  ```
  registered=10 ran=10 ok=9 controls-failed=1 did-not-complete=0 skipped=0

  SNAPSHOT MOVED: 1 harness file(s) changed while the sweep ran.
    windows-probes-negative-controls.py -- 968bd9f58a810d4a at the start, 7c286e0c2b17d3cd now
  SUITE VERDICT: UNBOUND -- the harnesses were edited mid-sweep, so the
    rows above are about different revisions of the suite and cannot be
    added up. Re-run against a still tree.
  NEGATIVE-CONTROL COMPLETE run-all.sh failures=2 elapsed=3610s partial=0
  ```

  Without the snapshot that hour would have been added up as nine passes and one
  harness whose 113 controls all failed — a bottom line about a set of files
  that never existed together. The 113 turned out to be a separate matter and
  not the edit: every one of them reported `the mutant does not parse` with an
  *empty* reason, and the same file at the same revision run on its own a few
  minutes later produced `MUTANTS APPLIED: 47`, `RED CASE OUTCOMES REQUIRED AND
  OBSERVED: 112`, `CONTROL FAILURES: 0`. `parse_ok` there reads its verdict off
  a subprocess exit status alone, so a PowerShell that failed to start —
  plausible in the seconds after three process-heavy bash harnesses — is
  indistinguishable from a mutant with a syntax error. Which is this
  workstream's own rule, one level up: a measurement that could not be taken
  arriving as a negative one. Both facts are worth having, and the sweep would
  have handed over neither.

  Re-run against a still tree, the same eleven revisions in and out:

  ```
  registered=10 ran=10 ok=10 controls-failed=0 did-not-complete=0 skipped=0
  SUITE VERDICT: every registered harness completed and every control fired.
  NEGATIVE-CONTROL COMPLETE run-all.sh failures=0 elapsed=3588s partial=0
  ```

  That line is worth exactly what the manifest above it says and no more: these
  eleven files, at these hashes, on this host, at that hour.
- **Each control's own falsifiability is a precondition, not an inherited
  claim.** Round 3 answered "can this control fail?" with one fault-injection
  receipt per harness and reported every control in that harness as
  demonstrated. That is a real overclaim: a receipt on shared scoring code shows
  the shared rejection branch is reachable, and says nothing about whether
  control #12's own anchor, its own replacement text, or its own expected
  outcomes are non-vacuous. So the per-control facts are checked in every run,
  before anything is scored, and the harness prints the claim they license:

  1. the anchor occurs **exactly once** in the subject, so the mutation lands in
     one known place — zero means it reverts nothing, more than one means it
     lands somewhere it was never aimed, and both read downstream as a control
     that ran;
  2. the mutant **differs from the shipped bytes**, which a once-matching anchor
     does not imply — these entries are written by copying the neighbour above
     and editing one of two strings, and editing neither leaves a control that
     replaces a real place in the file with what was already there;
  3. every case the control **names exists and PASSES unmutated**, so its
     expected-outcome set is non-vacuous and no name can be credited for a red
     that was already there;
  4. and at score time the mutant must parse, and the suite must go red on
     exactly the named cases and nothing else.

  Together those make each control individually non-vacuous on every run, which
  is a stronger and narrower thing than a receipt. All four hold in the four
  harnesses built on anchored text replacement — `posix-probes`, `lib-ps1`,
  `port-precheck` and `a-drift-guard-replica`. `a-drift-guard-inventory.py` is
  scenario-based rather than anchored: it builds a synthetic source tree per
  scenario, so "the anchor occurs once" has no meaning there and the equivalent
  guarantee is that each scenario asserts against a tree it wrote itself. The
  remaining four harnesses belong to other lanes and are not covered by this
  claim.

  What none of it establishes is that a control's named outcome is the *right*
  one — a case can redden for a reason unrelated to the property being reverted,
  and nothing here can tell those apart. That judgement is still a human's.

  Precondition 1 is not a formality, and the cost of not having it is measured
  rather than argued. Two lanes rewrote subjects during the writing of this
  paragraph. `scripts/lib/host-process.sh` had its `netstat` parse rebuilt
  around new variables at 06:35, and two POSIX controls whose anchors named the
  old `notarow` design dropped to zero matches. `scripts/first-run/lib.ps1` was
  rewritten to CRLF at 07:05 — measured, `556` CRLF against `556` LF, every line
  — and every `lib-ps1` anchor spanning more than one line dropped to zero
  matches at once, because the anchors hold `\n` and the file now holds `\r\n`.
  In both cases the harness refused instead of scoring, which is the whole
  point: without the count, four controls would have quietly reverted nothing
  and reported that they caught their cases.
- **Every harness ends with a terminal completion marker.** A harness killed by
  a watchdog prints its last scored control and stops, and the tail of that
  transcript is indistinguishable from a harness that scored everything. The six
  on this track print `NEGATIVE-CONTROL COMPLETE …` from an exit path and
  `NEGATIVE-CONTROL ABORTED …` from every other way out; `run-all.sh` requires
  the marker to be the last non-empty line and reports its absence as
  `DID-NOT-COMPLETE`. A partial run (`--only`) stamps `partial=1` and is refused
  as a result. The abort path also carries the status: it exits with the code
  that killed the run — `137` for a `SIGKILL`, the same number the `rc=` in the
  abort line prints — and rewrites only a zero, because a run that stopped
  before it scored every control must not exit 0 for a supervisor that reads
  nothing but the status.

  Round 4 found that claim was half true. The rewrite happened *after* the line
  was printed, so on the single path where it fires the transcript said `rc=0`
  and the process exited `1` — two numbers to reconcile, in the one place the
  line exists to stop a reader having to. It was not theoretical: `timeout`
  SIGTERMs a harness and bash enters the `EXIT` trap with `$?` still `0`, which
  is how a real killed run printed `NEGATIVE-CONTROL ABORTED
  lib-ps1-negative-controls.sh rc=0` on its way to exiting non-zero. The status
  is settled before it is printed now, and when it was rewritten the abort path
  says so on its own line instead of leaving the discrepancy for the reader.
- **A control that requires a refusal has to name which refusal.**
  `a-drift-guard-inventory.py` raises `Unmeasurable` at eight distinct sites,
  and seven of its controls were written `try: reachable(fixture) except
  Unmeasurable: pass` — the exception *is* the pass condition. A witness
  reachable only from the exception path ratifies "something here declined",
  never "the construct under test was declined": any of the other seven raises
  would have satisfied it, including one from a fixture that failed to
  substitute. Each of those controls now checks that the refusal text names its
  own rule. The receipt is two-sided, because a guard is only interesting if the
  old spelling would have missed it: adding `concat_idents!(tooth, _a)()` to the
  `include!` fixture — so it is refused by the identifier-pasting rule instead
  of the `include!` rule — scores `0 control failures, exit 0` under the old
  `is None` test and `1 control failure, exit 1` under the new one, naming the
  rule that actually fired.
- **There is no CRLF guard on the harnesses, and that is a measurement, not an
  oversight.** One was written and demonstrated firing before it was deleted.
  Converted end to end to CRLF, `a-drift-guard-replica.py` (937 CRs) still runs
  17 controls to `failures=0`, and `port-precheck-controls.sh` (384 CRs) still
  reaches `CONTROL FAILURES: 0` — Python normalises source line endings, and
  this bash (MSYS2 4.4.23) strips them before the script runs, including out of
  quoted heredocs and `$'...'` literals. So a refusal there would have been one
  nobody earned. Where line endings *are* the measurement is the subject side,
  and each harness that compares subject bytes already handles it: `posix-probes`
  reads with `newline=''` and, when an anchor is missing but matches after
  normalisation, says the line endings moved rather than the code;
  `authenticode-step-receipt.py` compares with `read_bytes` and reports a
  CRLF-only difference as such; `windows-probes` keeps a fully-CRLF fixture in
  its self-check to prove its own guard reads bytes and not text. `run-all.sh`
  carries the reasoning and, more usefully, the detection traps — `grep -c
  '\r'`, `awk '/\r/'` and `sed | cat -A` all report a 100%-CRLF file as clean on
  this host, and `grep -c $'\r'` is right standalone but returns the *line
  count* from inside `"$( … )"`. That last one is **a measurement on this host,
  not a rule about Bash.** Round 5 refuted the mechanism this entry used to
  assert — that `$'…'` is expanded before a command substitution's text is
  re-parsed — and was right to; the sentence is gone. The observation reproduces
  without it, from a plain script file with no `eval` and no outer `bash -c`, on
  MSYS2 bash 4.4.23(1)-release: over a 3-line pure-LF fixture, standalone gives
  `0` and `n="$(grep -c $'\r' pure-lf.txt)"` gives `3`. Bash's own xtrace is the
  evidence — `+ grep -c $'\r' pure-lf.txt` standalone against `++ grep -c ''
  pure-lf.txt` inside `$( )`, the pattern arriving empty — and backticks do
  *not* show it, so it is `$( )` specifically rather than capture in general.
  Why is not claimed; re-measure before repeating it on another bash. Assign the
  pattern first (`CR=$'\r'`), or use `git ls-files --eol` for a tracked file and
  `read_bytes().count(b'\r\n')` for anything else.


## The rule they all encode

> A failed measurement must never be indistinguishable from a negative
> measurement.

Every probe is tri-state — measured / negative / **could not measure** — and
every caller branches on all three. Most of the controls here exist because
some remedy for that defect reintroduced it one level further out: a guard that
could not run, a digest that could not be read, a gate whose failure was
discarded, a fixture that was last run's. When adding a control, revert exactly
one property and pin the expectation to the cases that defend it, so a mutation
that reddens everything is reported as unpinned rather than as a success.
