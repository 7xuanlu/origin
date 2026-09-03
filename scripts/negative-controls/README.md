# Negative controls

A test that passes proves the test passes. It does not prove the test would
notice the bug. Every harness here removes exactly one half of a shipped
remedy, re-runs the suite that defends it, and **fails when the suite stays
green** — so the assertion it guards is measured rather than assumed.

They live in the tracked tree on purpose: a control nobody else can run is a
claim, not a control.

There are **eleven executable harnesses** and one runner. Run them through
`run-all.sh`; eleven hand-typed command lines produce eleven results and no
aggregate.

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
| `dev-runtime-lock-race-controls.sh` | itself | `scripts/dev-runtime.sh`'s `acquire_runtime_lock` — the stale-lock break, driven as two REAL runs of the whole shipped file against one state directory. The ABA it defends has two windows: the owner re-read closes the one before it, and the ATOMIC RENAME closes the one after it, which no re-read can. Each shipped arm is read against the revert that defends its own window; `DEV_RUNTIME_RACE_SLEEP` and `DEV_RUNTIME_RACE_SLEEP_BREAK` (0 in every real run) widen one window each, so the interleaving is arranged rather than sampled |
| `dev-runtime-stage-controls.sh` | itself | `scripts/dev-runtime.sh`'s `stage_windows_daemon` — the daemon and DLL copies that decide, by CONTENT and not by mtime, what the recorded server path points at, the re-read that proves what landed, the tri-state listing of the directory they come from, and the call site in `start_runtime` that reaches all of it. Extracted by brace matching for the same reason as the two above |
| `windows-probes-negative-controls.py` | itself | `scripts/first-run/windows-zip.ps1` and `scripts/first-run/windows-nsis.ps1` — the port, health and process-liveness probes, and the `Check` blocks that branch on them. Case-less before this; the probes are *extracted* rather than run, because neither channel script can be executed on a developer machine (each deletes `%LOCALAPPDATA%\wenlan`, the real memorydb and config) |

## Running them

```bash
bash scripts/negative-controls/run-all.sh          # all eleven, one receipt
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
bash    scripts/negative-controls/dev-runtime-lock-race-controls.sh
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
  them can be mistaken for a clean run. Reproduce it by touching a subject file
  mid-run.
- The other four — `dev-runtime-scan-controls.sh`,
  `dev-runtime-record-controls.sh`, `dev-runtime-stage-controls.sh` and
  `windows-probes-negative-controls.py` — end with `CONTROL FAILURES: N`
  instead. That is *their* contract, recorded in the runner's registry rather
  than imposed by it, and it is weaker: an ordinary print proves the harness
  reached its last statement and nothing about how it would have exited
  elsewhere. Upgrading one is four lines (a `MARKER`/`HARNESS` pair, an `EXIT`
  trap, a `reached_end` flag) plus moving its registry row from `summary` to
  `marker`; they belong to other lanes, so it is for whoever owns the file.

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
sweep refuses — otherwise a twelfth harness is simply never run, under a clean
report of the eleven that were.

**What a clean sweep does not mean.** It is a statement about eleven harnesses,
not about the repository. Every harness here reverts a *named* property; a
defect nobody thought of has no control and is not measured by one.

**Eight of the eleven defend *themselves*** — count the `itself` rows in the
`Defends` column: `a-drift-guard-inventory.py`, `a-drift-guard-replica.py`,
`port-precheck-controls.sh`, `dev-runtime-scan-controls.sh`,
`dev-runtime-record-controls.sh`, `dev-runtime-lock-race-controls.sh`,
`dev-runtime-stage-controls.sh` and `windows-probes-negative-controls.py`. They supply the cases as well as the
controls, because the subject had no suite at all, so what they establish is
that their own extracted cases would notice the reversion, not that any CI lane
would. Only three point at a suite that exists independently of the harness —
written for the subject, not extracted by the control that grades it:
`posix-probes-negative-controls.py` at `scripts/host-process.test.ts`,
`lib-ps1-negative-controls.sh` at `first-run/lib.test.ps1`, and
`authenticode-step-receipt.py` at `scripts/release-workflow-contract.test.py`.
That ratio is the single most important thing on this page, which is why it is
a number here and not the word "several" — eight elevenths of this suite is a
closed loop marking its own homework.

And none of this runs in CI: no PR lane runs
these negative-control harnesses, `lib.test.ps1` or the first-run channels, so
all eleven are a pre-merge step by hand for anyone touching the files in the
table above.

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

### What every harness here holds itself to

A harness can fail in a way that looks like a pass, or like an unrelated red.
These are the properties that stop it, and the ones a new control must satisfy.

**Per control, checked on every run before anything is scored.** These four make
each control individually non-vacuous, which is stronger and narrower than a
one-off fault-injection receipt on shared scoring code:

1. the anchor occurs **exactly once** in the subject — zero reverts nothing,
   more than one lands somewhere it was never aimed, and both read downstream as
   a control that ran;
2. the mutant **differs from the shipped bytes**, which a once-matching anchor
   does not imply;
3. every case the control **names exists and PASSES unmutated**, so no name can
   be credited for a red that was already there;
4. at score time the mutant must parse, and the suite must go red on exactly the
   named cases and nothing else — a mutation with collateral damage is reported
   as unpinned, not as a success.

All four hold in the four anchored-text harnesses (`posix-probes`, `lib-ps1`,
`port-precheck`, `a-drift-guard-replica`). `a-drift-guard-inventory.py` is
scenario-based — it builds a synthetic tree per scenario, so "the anchor occurs
once" has no meaning there. The remaining four belong to other lanes.

What none of it establishes is that a control's named outcome is the *right*
one: a case can redden for a reason unrelated to the reverted property. That
judgement is still a human's.

**Per run.**

- **A baseline in the same run.** Every `must_fail` and `must_survive` case named
  anywhere in a harness must be passing with no mutation applied, or no mutation
  is scored. `a-drift-guard-replica.py` has no test runner, so it refuses a
  control whose check id is already violated by the shipped file.
- **The suite must prove it ran.** "The suite went red" and "the suite never
  started" arrive as the same non-zero status, so each vitest run is read back
  from its JSON reporter: zero tests, or a file other than the one the control is
  about, is a control failure naming the discrepancy.
- **The override is locked by a nonce, not an mtime.**
  `posix-probes-negative-controls.py` hands the suite a mutated library through
  `WENLAN_HOST_PROCESS_LIB`; it stamps a per-mutation nonce into the copy as a
  `# wenlan-control-nonce:` line and passes it in
  `WENLAN_HOST_PROCESS_LIB_NONCE`, and the suite requires exactly one carried
  nonce and requires it to be that one. An mtime slack window accepts a leftover
  copy written inside it.
- **The subject may not move under the run.** A sweep is an hour, which is long
  enough for another lane to land an edit. Every harness whose subject is read
  more than once snapshots it and refuses with `FATAL: <file> changed during the
  run`; `posix-probes` and `lib-ps1` re-check after every control, and
  `posix-probes` guards `scripts/host-process.test.ts` — its evidence — as well
  as the library, immediately before and after every suite invocation. A check
  only at the bottom of the file never runs in the runs that do not reach it,
  which are the long ones. `a-drift-guard-inventory.py` and
  `a-drift-guard-replica.py` are exempt: each reads its subject exactly once.
  **Residual:** an edit landing after the before-check and reverted before the
  after-check is invisible to this and to any polling scheme.
- **The aggregate receipt names the revisions it covers.** `run-all.sh` hashes
  every registered harness — and itself — at registry validation, prints the
  manifest under `harness revisions this receipt covers`, re-hashes before any
  verdict, and reports drift as `SNAPSHOT MOVED` with verdict `UNBOUND`, counted
  into the marker's `failures=`. Same residual as above.
- **Every harness ends with a terminal completion marker.** The six on this track
  print `NEGATIVE-CONTROL COMPLETE …` from an exit path and `NEGATIVE-CONTROL
  ABORTED …` from every other way out; `run-all.sh` requires the marker to be the
  last non-empty line. The abort path exits with the status that killed the run
  and rewrites only a zero — `timeout` SIGTERMs a harness and bash enters the
  `EXIT` trap with `$?` still `0`. The status is settled before it is printed, so
  the line and the exit code agree. A mode that runs no controls (for example
  `--print-digests`) sets the completed flag but deliberately prints no marker: a
  `failures=0` marker minted out of a listing is a clean result from nothing.
- **A refusal arm must name which refusal.** Where a control's pass condition is
  an exception, `try: probe(x) except Refused: pass` passes on any refusal from
  any raise site. Each such control matches text only its own rule emits, and the
  one arm that must succeed requires exit 0, no `AssertionError` anywhere, and
  the terminal `PASS` line.
- **A driver that died is not a control that fired.** `windows-probes`'s
  `run_all` returns three lists — passed, failed, unmeasured — and a case that
  could not be measured is a control failure whether or not the control hoped to
  see it go red. A mutation that kills a *setup* statement is a legitimate way for
  a defect to manifest, so all 41 setup-driven cases wrap setup in a `try`/`catch`
  that prints one tab-separated `SETUP-THREW` record (exception type, message,
  raising line) before exiting: dying *with* that record is a case that FAILED for
  a named reason, dying without one is still unmeasured.
- **A parse gate must be a measurement.** Every mutant goes through `bash -n`
  first, and one that does not parse is a control failure naming the syntax error.
  A `bash -n` that cannot run at all (status other than 0 or 2) is fatal. On the
  PowerShell side the criterion is the checker's `line <n>: <message>` shape, not
  "exit 1 with any output" — a host that printed `PowerShell host initialization
  failed` and exited would otherwise be read as a syntax verdict.
- **Name-keyed state refuses duplicate names.** Case results are keyed by title,
  so `collect_states()` raises on a duplicate rather than merging: two tests
  sharing a title collapse to one entry and a mutation can redden the shadowed
  one invisibly.
- **A failure message must point at evidence that agrees with it.** Control runs
  write `<control>--case-<name>.log` and the messages name that path; a shared
  `case-<name>.log` holds the *last* run of that case, typically a later
  control's and typically green. Only non-passing cases are kept under a control;
  the shipped-source sweep keeps all of them.
- **A receipt's digest must look like a digest.** `run-all.sh` requires 64
  lowercase hex characters at capture and at comparison, validates the whole
  `sha256sum` line before truncating it, and asks the hasher a question whose
  answer is known independently. Anything else is could-not-measure and never
  certifies. **Residual:** an edit made *and reverted* between the two hashings
  is invisible.
- **An optional dependency's absence is stated.** The `continue-on-error` rule
  needs PyYAML and the CI lane installs nothing; without a parser it falls back
  to a token scan and reports "unmeasured, not clean" for the two constructs that
  scan is structurally blind to (a YAML alias, and a double-quoted key carrying a
  backslash escape). `authenticode-step-receipt.py` drives all three states with
  `sys.modules["yaml"] = None`.
- **A subject that must never run is extracted, and its anchors proven first.**
  `windows-probes-negative-controls.py` lifts the probe functions and `Check`
  blocks out of both Windows channel scripts by brace matching, because each
  channel `Remove-Item -Recurse -Force`s `%LOCALAPPDATA%\wenlan` — the real
  memorydb and config. A pre-check validates every extraction, every mutation
  anchor's exactly-once match, and every case name a control claims to pin
  against the real source before a single case runs.
- **A control that drives a function proves something calls it.**
  `dev-runtime-stage-controls.sh` extracts `stage_windows_daemon` and runs it
  directly, so it also reads the shipped call site out of `start_runtime` —
  called, guarded by the Windows branch, failure propagated — and two controls
  break exactly that.

### Host facts these harnesses were built against

- **`bash` on PATH is not this bash.** On Windows the first `bash` on `PATH` is
  `C:\Windows\System32\bash.exe`, the WSL launcher, which exits 127 on a
  `C:/Users/...` path; MSYS bash eats the backslashes out of a `C:\Users\...`
  argument. `_bash()` writes a probe script and tries candidates until one reads
  it, and paths are handed over with forward slashes.
- **`chmod 000` does not make a file unreadable here** — measured: `sed` still
  reads it. `dev-runtime-record-controls.sh` builds its "present but unreadable"
  fixtures as directories, and measures before any case runs that `[[ -e ]]`
  accepts one and `sed` refuses it.
- **The harnesses need no CRLF guard, and that is measured, not assumed.**
  Converted end to end to CRLF, `a-drift-guard-replica.py` (937 CRs) still runs
  17 controls to `failures=0` and `port-precheck-controls.sh` (384 CRs) still
  reaches `CONTROL FAILURES: 0`: Python normalises source line endings and this
  bash (MSYS2 4.4.23) strips them before the script runs. `run-all.sh` re-measures
  that premise every sweep and applies the guard when it does not hold.
- **Line endings *are* the measurement on the subject side.** `posix-probes`
  reads with `newline=''` and, when an anchor is missing but matches after
  normalisation, says `the line endings moved, not the code` — it does not
  normalise and carry on, because a mutant with different line endings than its
  subject is a different program. `authenticode-step-receipt.py` compares with
  `read_bytes`; `windows-probes` keeps a fully-CRLF fixture in its self-check.
  The CRLF *detection* traps on this host (`grep -c '\r'`, `awk '/\r/'`,
  `sed | cat -A` and `grep -c $'\r'` inside `"$( … )"` all report a 100%-CRLF
  file as clean, or a pure-LF one as dirty) are documented in `run-all.sh` above
  `crlf_premise_holds`. Assign the pattern first (`CR=$'\r'`), or use
  `git ls-files --eol` for a tracked file and `read_bytes().count(b'\r\n')` for
  anything else.

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
