# AGENTS.md - scripts/

## OVERVIEW

Release, sidecar, and repo-inventory contracts. These scripts are
part of packaging behavior, not generic local helpers.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Stage sidecars | `prepare-sidecars.sh` | tree-build only; compiles from the checked-out backend |
| Tauri build hook | `prepare-tauri-build-sidecars.sh` | picks debug vs release based on `TAURI_ENV_DEBUG` |
| Resolve backend checkout | `resolve-backend-dir.sh` | validates the current checkout, or `WENLAN_BACKEND_DIR`; a sibling checkout is only a legacy fallback |
| Isolated dev runtime | `dev-runtime.sh`, `dev-all.sh` | worktree-owned daemon/UI ports, data dir, debug MCP socket, PID, and teardown |
| Host process primitives | `lib/host-process.sh` | sourced by `dev-runtime.sh` and both smokes; tri-state port/liveness/image probes, path spelling, identity-checked kill |
| Evidence wrapper | `attest.sh` | portable replacement for the personal `~/.claude/bin/attest.sh`; appends to `.claude/attest.jsonl` and fails when it cannot |
| Negative controls | `negative-controls/` | each reverts one half of a shipped remedy and FAILS if the defending suite stays green; see its README |
| Surface smokes | `smoke-cli.sh`, `smoke-mcp.sh` | isolated port + data dir + pages dir; asserted teardown; exact ledger multiset |
| Version lockstep | `release-version-sync.test.ts` | app, Cargo, Tauri versions must match |
| Sidecar tests | `prepare-sidecars.test.ts` | locks path and cloudflared behavior |
| API route inventory | `refactor/api-route-diff.mjs` | route coverage signal, not a product requirement |

## CONVENTIONS

- Sidecars always come from a backend checkout, found by
  `resolve-backend-dir.sh` (the current checkout by default, or
  `WENLAN_BACKEND_DIR`; a sibling checkout is only a legacy fallback). See
  `HISTORY.md` for the retired pinned-download mode.
- `prepare-tauri-build-sidecars.sh` is the Tauri hook; keep it aligned with
  `app/tauri.conf.json` `beforeBuildCommand`.
- `cloudflared` is required for a full Tauri bundle:
  `binaries/cloudflared-$TRIPLE`.
- Update scripts, tests, and workflows together when release or sidecar behavior
  changes. The workflow comments are part of the operational contract.

- Every probe in `lib/host-process.sh` is TRI-STATE — measured / negative /
  **could not measure** — and every caller branches on all three. A port check
  that cannot run must FAIL, exactly as `lsof`'s absence used to; it must never
  read as "port free". Same for liveness ("could not measure" is not "dead", and
  must never delete an ownership record) and for the image lookup ("could not
  measure" is not "a different binary"). Capture the status as
  `out="$(f)" || rc=$?`: under `set -e`, `out="$(f)"; rc=$?` aborts at the
  assignment, and `if out="$(f)"; then …; fi; rc=$?` reads the compound's own
  status, which is 0.

- An exit status is only a measurement when the tool has a status per outcome.
  `lsof` returns 1 both for "nothing matched" and for "an error was detected",
  so `lsof … || hit=""` reports a broken probe as a free port. `-t` puts only
  pids on stdout, so the POSIX branch merges stderr in (`-w` first, to drop the
  benign `can't stat()` warnings) and reads silence-with-1 as the negative and
  any text as unmeasured. Check the same way before trusting any other tool's
  nonzero status. `kill -0` is the same shape and reads as if it were not:
  status 1 is ESRCH *and* EPERM, so `kill -0 "$pid" 2>/dev/null` calls a live
  process this user may not signal "gone". Only the errno TEXT separates them,
  which is what `errno_says_no_such_process` reads; it is still `kill -0`
  underneath, so unlike `process_is_alive` it keeps answering about the MSYS pid
  `$$` records rather than about a WINPID.

- `dev-runtime.sh`'s `read_owned_pid` is tri-state for the same reason and is
  read the same way: `0` a record, `1` no record, `2` a record that could not be
  read. Only `2` keeps `clear_owned_state` away from a record whose daemon may
  still be running, so folding it into `1` deletes the ownership of a live
  process and then reads its port as free. The record's data-dir member is the
  only optional one, which makes it the one place a dropped read status is
  invisible: absent and unreadable are both the empty string unless the `sed`
  status is checked.

- `lock_owner_file_appeared` is the one probe here with FOUR statuses, and the
  fourth is not decoration: `0` an owner appeared, `1` the lock DIRECTORY went
  away while waiting, `2` the directory could not be read, `3` the lock is still
  there and still names nobody. 1 and 3 were one status, and the caller refused
  on both — while the source comment promised 1 would go round again and let
  `mkdir` arbitrate. A lock that was released mid-wait is not an unattributable
  lock; a lock that is still standing with no owner in it is, and recovering
  THAT is how two commands come to share a lock directory. When a fourth answer
  exists, spell it; folding it into a neighbour is the same defect as folding
  "could not measure" into "no".

- `dev-runtime.sh`'s LAST line on stderr is `DEV_RUNTIME_RESULT: <kind>`, with
  `<kind>` one of `ok`, `safety-refusal`, `build-failure`, `staging-failure`,
  `health-failure`, `port-conflict`, `interrupted`, `unknown`. `build-failure`
  is cargo; `staging-failure` is everything after it succeeded (the held DLL,
  the unverifiable stage) and calls for a different remedy, so a caller that
  retries the build must not see one for the other. `interrupted` is a signal
  and is the only failing kind a supervisor may re-run unchanged.
  It is a consumed contract: another lane classifies
  by that spelling, so do not reword, translate, add or drop a kind without
  changing the consumer. It is additive — every human line above it is
  unchanged — and it is printed from an EXIT trap so the file-scope guards, a
  `set -e` abort and a signal all carry one. The trap is installed FIRST, before
  `SCRIPT_DIR`, `REPO_ROOT` and the library `source`: each of those is a way out
  under `set -e`, and a way out that precedes the trap prints no marker at all.
  For the same reason a failed release must not abort the trap before the marker
  is printed — but ROUND 4 is that it must not be DISCARDED either, and
  `release_runtime_lock || true` was the wrong remedy for a reason that has
  nothing to do with the trap: the collapse was INSIDE the function, whose
  `[[ -f ]]` and `sed` failures each returned 0, so the `|| true` never saw a
  status to throw away. The release now reports, and the trap reads it with a
  checked `if !` — which suspends errexit exactly as `|| true` did AND keeps the
  answer. A run that finished but left its lock standing makes the NEXT command
  refuse on a lock this one called released, so an otherwise-`ok` run is
  downgraded and exits non-zero. `unknown` is a
  REFUSAL, not a pass:
  it is what this prints when it cannot tell which kind applies, and a consumer
  must treat it the way it treats `safety-refusal`. Never guess a kind.

- The same rule governs `attest.sh`: an unrecorded run must never be
  indistinguishable from a recorded one. The weekly sweep reads a missing ledger
  row as "the smoke never ran", so a passing command whose ledger write failed
  exits non-zero. Command status wins when non-zero; the ledger's status
  otherwise.

- A tri-state is only tri-state as far as its result travels, and a witness only
  counts if it CO-VARIES with the claim it ratifies. Two rules for the first-run
  Windows probes, and both were found by review after the tri-state landed:
  `Stop-Daemon` computed three states, printed them and returned success, so
  every caller saw one; it returns the state now and both call sites record a
  row. And `Get-ProcessTableWitness` checked "is this a process table" (pid 4,
  ten rows) to ratify "wenlan-server is not in it" — so a targeted read could
  throw its absence error while the whole-table read CONTAINED the process, and
  the negative was ratified anyway. The witness is told what it is covering; the
  listener probe cross-checks its table-derived negative with a targeted read
  for the same reason, which closes the truncated-table residual on the
  PowerShell side that `netstat` still has on the POSIX side. Disagreement
  between two reads is "could not measure", never the negative.

- Where a probe's negative depends on a timing constant, the constant is a
  measurement and the control must observe it. A refused loopback connect takes
  ~2.05 s on Windows (the SYN is retried), so `-TimeoutSec 2` turned every
  genuine refusal into a Timeout and the negative became unreachable in
  principle. `GAUNTLET_HEALTH_TIMEOUT_SEC` raises the 5 s default for a slower
  runner and is floored at it; below the floor the probe cannot return its own
  negative, which is the defect, not a tuning choice.

## ANTI-PATTERNS

- Do not let CI placeholder binaries become a release substitute.
- Do not make `resolve-backend-dir.sh` silently accept a directory that lacks
  `crates/wenlan-server`, `crates/wenlan-mcp`, and `crates/wenlan-cli`.
- Do not let an awk program in a measurement pipeline `exit` on its first
  match, and do not write `… | head -1`. Under `pipefail` a producer that takes
  SIGPIPE reports 141, which is indistinguishable from a real parser failure.
  Set a flag and print at `END`; collapse the head into one
  `sed -n '/pat/{s///;p;q;}'`.

- Do not assume a column index when parsing `ps -W`, `netstat -ano`, or
  `tasklist`. Read each index from the header, and `exit 3` at `END` when the
  header is not the one the program parses, so the caller's `|| return 2` turns
  an unreadable table into "could not measure". Guarding rows on `col > 0`
  instead does the opposite: every row is skipped, awk prints nothing and exits
  0, and the empty result reads as a negative — no listener, no such process,
  nothing to reap. That substitution has now been found and fixed six times in
  this directory, twice inside the comment written to prevent it and once inside
  the negative control written to prove it could not happen. `netstat`'s State
  column is localised on a non-English Windows, so `LISTENING` is not a key
  either; the structural rule (a wildcard foreign address) is.

- Do not write a second parse of `ps -W`. `lib/host-process.sh` has one, behind
  two entry points — `ps_w_row_for <PID|WINPID> <value>` for one row and
  `ps_w_rows_matching <pattern>` for every matching row — and a caller that
  needs a third question adds a mode there rather than a table walk of its own.
  There have been three copies; each time, the hardening landed on one of them
  and the others went on counting words. A second CALL SITE is fine, a second
  PARSE is the defect, and `scripts/host-process.test.ts` counts the places that
  run the command at all across the library and every script that sources it.

- A parsed table is not a complete table. `tasklist` and `ps -W` are checked for
  completeness, not just shape: every line must be a row, `tasklist` must carry
  pid 4 (the System process exists on every Windows NT kernel) and at least ten
  rows, and stderr is merged into the snapshot so a warning riding alongside a
  truncated table is a refusal rather than an unnoticed gap. `netstat` has no
  must-appear socket, so its completeness rule is structural instead: both
  parses — the bash one in `lib/host-process.sh` and the PowerShell one in
  `first-run/windows-zip.ps1`'s `Get-PortListenerWitness` — require every
  non-blank line to BE a row, and require a UDP row as the END WITNESS, because
  `netstat -ano` prints the whole TCP table and then the whole UDP table, so a
  UDP row is evidence the stream got past the end of TCP. The ordering that
  rests on is checked rather than assumed (a TCP row after a UDP row is a
  refusal), and a table neither parse can account for is refused rather than
  read as "no listener". What remains is a hole in the MIDDLE of the TCP
  section, which the end witness cannot see; it is stated in both parses.

## COMMANDS

```bash
bash -n scripts/prepare-sidecars.sh
bash -n scripts/prepare-tauri-build-sidecars.sh
bash -n scripts/resolve-backend-dir.sh
bash -n scripts/lib/host-process.sh
bash -n scripts/dev-runtime.sh
bash -n scripts/dev-all.sh
bash -n scripts/smoke-cli.sh
bash -n scripts/smoke-mcp.sh
bash -n scripts/attest.sh
bash scripts/prepare-sidecars.sh --print-paths
pnpm vitest run scripts/prepare-sidecars.test.ts scripts/release-version-sync.test.ts scripts/dev-runtime.test.ts scripts/host-process.test.ts scripts/attest.test.ts

# A green suite does not prove the suite would notice the bug. Before changing
# host-process.sh, dev-runtime.sh's process scan or ownership record,
# first-run/lib.ps1, first-run/port-precheck.sh, first-run/windows-zip.ps1 or
# first-run/windows-nsis.ps1 (their port, health and process-liveness probes),
# the Authenticode step in release.yml, or a drift_guard tooth, run the control
# that defends it -- no PR lane runs these negative-control harnesses,
# lib.test.ps1 or the first-run channels, so these are a pre-merge step by hand.
# Run the sweep, not the individual harnesses. Eleven separate command lines
# produce eleven separate results and no aggregate: run eight of them, read eight
# greens, and conclude the suite swept -- a partial run that looks exactly like
# a complete one. run-all.sh holds a registry, refuses a harness that exited 0
# without reaching its completion marker, refuses a marker that contradicts the
# exit status, and prints one verdict. ~50 min; the posix harness is ~26 of it.
bash scripts/negative-controls/run-all.sh

# Individual harnesses, for iterating on one control. A green here is evidence
# about that harness only -- it is not a sweep, and must not be reported as one.
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

- Creating and destroying need different evidence. A name being free beforehand
  licenses *registering* it; it licenses nothing else. Ending, deleting or
  switching something off needs free-before **and** measured-present-after, so
  the thing being destroyed is provably the thing this run made. The scheduled
  task, the daemon process and the data directory have all been reached by a
  gate that only proved the first half.
- A witness reached only from the exception path ratifies total absence, never
  the specific case. `try: probe(x) except Refused: pass` passes on any refusal
  from any of the raise sites, so a control written that way tests that
  something declined, not that this construct did — seven controls in one file
  were spelled that way and none of them measured what its name claimed. Where a
  control's pass condition is an exception, the control must check *which*
  exception, by matching text that only its own rule emits.
- The two Windows first-run channels must never be RUN to check a change to
  them: `windows-zip.ps1` and `windows-nsis.ps1` each `Remove-Item -Recurse
  -Force` `%LOCALAPPDATA%\wenlan`, which on a developer machine is the real
  memorydb, config and logs. `windows-probes-negative-controls.py` extracts
  their probes and the `Check` blocks that call them and drives those in
  isolation; verification of these two files is static analysis plus that
  harness, never execution.
