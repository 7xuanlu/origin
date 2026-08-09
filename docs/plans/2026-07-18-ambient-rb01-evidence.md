# Ambient RB-01 target-Mac evidence

Date: 2026-07-21
Branch: `feature/ambient-enrichment-scheduler`
Status: COMPLETE — RB-01 closed by representative daemon and deterministic proofs

## Current conclusion

- Do not use Wenlan write recency as a foreground-idle proxy. Ambient and
  automatic admission have no fixed ten-minute delay; the existing ten-minute
  value remains only on the explicit Idle recap trigger.
- Keep the production CPU admission ceiling at 20%, with two consecutive
  30-second samples. A target-Mac calibration that measured foreground timer
  jitter before and during the same one-request Document inference passed at a
  20.70% starting load, but the 23.82% and 28.89% candidates both exceeded the
  predeclared `p99 <= baseline p99 + 5 ms` budget. All three runs remained at
  nominal thermal state, so 20% is a responsiveness reserve rather than a
  claimed thermal cliff.
- On-device background routes additionally require 2 GiB above the ordinary
  `max(2 GiB, 15%)` memory floor. The calibration observed at most 1.59 GiB of
  additional process RSS during one inference; the rounded reserve prevents a
  model call from consuming the foreground floor. Remote routes keep the
  ordinary memory policy.
- Replace the provisional ten-minute inter-turn floor with
  `max(120 seconds, 19 × measured turn time)`. Across the five isolated lanes
  and two persistent-daemon turns below, the longest measured turn was
  5.528 seconds, so the multiplier requires about 105 seconds and the
  two-minute floor keeps the observed envelope below roughly 4.4% duty. The
  production runtime's CPU, memory, and two-sample admission gates remain
  independent and can defer a turn longer. On macOS, production also reads the
  public `NSProcessInfo.thermalState` and CoreGraphics seconds-since-input
  signals before model startup and every new heavy turn. Any non-nominal
  thermal state, unavailable supported telemetry, or physical input in the
  previous 60 seconds vetoes new work. Other platforms retain the portable
  CPU/RAM gate until they gain an equivalent supported host probe.
- The current-source representative persistent-daemon proof loaded the model
  once, completed Classification and StructuredExtract with one provider call
  and current-version receipt each, kept all 14 thermal samples nominal, and
  exited cleanly. CPU samples at 25.15%, 20.71%, and 22.83% deferred work;
  admitted turns began at 19.64% or below. The turns were separated by
  120.018 seconds. The earlier `std::process::exit`/Metal teardown crash remains
  useful regression history; the `_exit` repair is covered by process tests and
  the successful live exit.

## Evidence boundary

- The live Wenlan database was opened read-only for aggregate counts only. No content was selected, copied, or migrated.
- The user's config has no everyday/synthesis pin and no on-device model choice. The profiler therefore does not start the daemon or alter config. A manual ignored test uses the already-cached registry default `qwen3-4b`, synthetic content, and a temp database only.
- Deterministic slice/convergence tests and real-model thermal measurements are reported separately. A fast mock is not thermal proof; one real inference is not backlog-convergence proof.
- Five isolated lanes completed against source bundle
  `52740615cfcc6e0c58b0e3f18907328cba9eca2a55c71d677977b5b97e7e7027`.
  Those rows establish lane functionality and the measured work-time/memory
  envelope used for calibration. The later `_exit` fix and two-minute policy
  did not change lane algorithms. Their current behavior remains covered by
  deterministic scheduler/integration gates; only the representative
  persistent-daemon soak must be repeated.
- The profile refuses to start unless the model is already cached, macOS thermal state is nominal, free memory is at least 15%, and two consecutive 30-second aggregate CPU samples are at or below 20%. The Rust runner additionally enforces the scheduler's ordinary memory floor plus the 2 GiB on-device inference reserve.
- Real profiles take an exclusive atomic lock before the first admission window
  and retain it through build, the second admission window, and the measured
  child. A failed child cannot replace the pre-armed 5,700-second cooldown with
  a shorter value.
- The profiler fingerprints a canonical byte stream containing the complete
  tracked diff plus every untracked path, type, size, and raw byte before the
  build, then refuses the run if the fingerprint changes after compilation.
  Build outputs are copied into owner-read/execute-only, run-specific paths
  before the second admission window; their hashes, the lockfile, source
  bundle, and individual source hashes are rechecked after that window and
  again before execution. The profiler executes the frozen test copy. Daemon
  mode passes the frozen daemon path/hash, and the Rust harness rehashes it
  immediately before spawn. The manifest records those exact binaries,
  lockfile, model blobs, harness, audit, and compact backlog result.
- Daemon evidence uses a pre-created external run directory; its isolated
  config, database, Pages root, raw log, and manifests survive both assertion
  and timeout failures. The daemon inherits only an explicit
  `HOME`/`PATH`/`TMPDIR` plus local RB-01 variables, not provider credentials,
  endpoints, or proxy configuration.
- The live daemon proof samples macOS thermal state and `sysinfo` total and
  available memory every 30 seconds from model load through the second turn.
  Missing telemetry, non-nominal thermal state, or memory below
  the configured floor requests cooperative shutdown and fails the proof. The
  historical representative artifact below used `max(2 GiB, 15%)`; the current
  harness adds the new 2 GiB on-device inference reserve and compiles locally,
  while the reserve behavior itself is covered by deterministic scheduler
  tests. CPU remains owned by the production two-sample admission gate.
- The profiler requires exact `refs/main` snapshots for both the Qwen GGUF and
  FastEmbed's `Qdrant/bge-base-en-v1.5-onnx-Q` model plus all four tokenizer
  metadata files. Real children remove inherited `HF_HOME`, pin Wenlan's
  test-only FastEmbed cache, and set `HF_ENDPOINT` to an unavailable loopback
  address. Because pinned `hf-hub 0.4.3` checks the exact cache pointer before
  requesting metadata, a complete cache performs no network call and any
  unexpected miss fails closed without external download.
- The locally available `origin/main` is schema 79; a live SSH fetch failed
  authentication, so the ref is not claimed current beyond this machine. The
  read-only user database observed on 2026-07-20 is also schema 79. Released main used
  versions 78/79 for Page history and operation receipts, while the unshipped
  feature lineage used 78/79 for ambient receipt/provenance columns and reached
  schema 83. The integrated terminal is schema 85: main's two tables are
  idempotent migrations 84/85, and migration 82 repairs the ambient columns
  for an already-main-79 database. Its origin rows still cannot be
  reconstructed into service classes, so migration 83 does not invent a
  bulk/interactive backfill.

## Current live aggregate

Read-only query against the current schema-79 database on 2026-07-20. The
database has neither `enrichment_origin` nor `enrichment_steps.input_version`;
the query did not run migrations, select content, or expose row ids. The
repeatable audit and compact result are
`scripts/audit-ambient-rb01-backlog.sh` and
`docs/eval/results/ambient-rb01-backlog-2026-07-20.json`.

| Metric | Count |
|---|---:|
| Memory rows, including chunks | 5,807 |
| Memory heads (`chunk_index=0`) | 2,885 |
| Active Pages | 133 |
| Pending/in-progress/paused document queue rows | 0 |
| Pending-revision memory heads | 1 |
| Legacy enrichment receipts | 9,548 |
| Active Pages missing citation processing | 1 |
| Missing-citation Pages with source evidence | 0 |
| Classification population projection | 2,443 |
| Eventual Page-growth population projection | 2,397 |
| Title-enrichment population upper bound | 2,443 |
| Entity candidates | 2,787 |
| Entity candidates with an authoritative link | 1,523 |
| Entity candidates without an authoritative link | 1,264 |

Legacy entity-related receipt counts:

| Step/status | Count |
|---|---:|
| `entity_extract / ok` | 71 |
| `entity_extract / skipped` | 1,468 |
| `entity_extract / failed` | 1 |
| `entity_link / ok` | 768 |
| `entity_link / skipped` | 772 |

These are upper bounds, not 1:1 inference counts. A missing primary entity can auto-link or terminate empty without inference, and new fixed-stage eligibility also applies origin, pending, version, and receipt filters.

## Backlog convergence risk

The current live schema predates the fixed-lane migration. Because it lacks
both selector columns, exact schema-85 current eligibility is deliberately
reported as `null`; running the production SQL directly would be false
precision. Applying only the stable head/filter predicates yields population
projections of 2,443 classification heads, 2,397 eventual Page-growth heads,
and 2,787 entity candidates. The privacy boundary does not evaluate memory
content or title, so the audit uses all 2,443 classification heads as a
conservative title-enrichment upper bound rather than claiming the exact
production title predicate. Page growth is
entity-dependent, so its immediate post-migration eligibility begins only as
current-version entity receipts are repaired. Of the entity candidates, 1,523 already have an
authoritative entity link and can repair their receipt without inference;
1,264 have no authoritative link and may still auto-link before inference.
There are also 94 accepted revisions inside the classification count.

Classification always spends one provider call for a selected row. At the
measured two-minute minimum recovery, its 2,443-row one-pass envelope is 3.39
days. The conservative title upper bound adds up to 3.39 days and entity
extraction adds up to 1.76 days before auto-link savings. Conservatively
charging every one of the 2,397 Page-growth candidates adds 3.33 days. Those
four counted populations total 8,547 turns, or 11.87 days before retries or
long requests, but that is a partial counted envelope rather than a complete
upper bound because the privacy-preserving audit does not know how many
StructuredExtract rows will make a nonzero provider call. Charging an
additional classification-sized 2,443-turn allowance for StructuredExtract
produces a conservative 10,990-turn, 15.26-day envelope at the two-minute
floor (76.32 days at the superseded ten-minute floor). Production exempts only
a committed terminal Page-growth no-match with zero provider calls; the exact
no-match and zero-call StructuredExtract shares are unknown, so the
conservative envelope does not subtract them. The one citation Page without
evidence does not add a thermal turn. Fresh and accepted-revision service
classes remain ahead of this legacy catch-up, so a new user write does not
wait behind the multi-day historical envelope.

The full scheduler initially made this worse: an empty Idle ReDistill or empty
maintenance stage consumed the same ten-minute cooldown before useful ambient
work. A focused RED proved the empty outcome was charged; GREEN now charges an
automatic turn only when it selected work, forwarded an LLM call, or panicked.
Empty scans still yield one poll to ambient fairness but no longer manufacture
thermal debt. This removes false delay; it does not lower the real-work
cooldown or the one-inference limit.

The earlier 40.2-day classification estimate was wrong because it counted
every chunk as a separate scheduler candidate. Fixed memory lanes select only
`chunk_index=0`; this document now keeps row, head, and inference envelopes
separate.

The existing oldest-first selectors would put a newly stored memory behind that
legacy catch-up. That violates the UX target even if the thermal duty cycle is
safe: background work could remain invisible while fresh data stays
unenriched for weeks.

The release-candidate convergence contract is therefore:

1. Every fixed memory lane uses explicit FIFO service classes:
   accepted revision, interactive/current work, raw bulk import, then legacy
   catch-up. It never relies on NULL collation.
2. A large raw import cannot enter the interactive class, and FIFO ordering
   avoids newest-first starvation inside a class.
3. A post-migration user mutation promotes a bulk or legacy memory into the
   interactive FIFO. For a missing legacy origin, it materializes the existing
   all-protected fallback without guessing which fields were originally
   explicit; an existing import retains its stored provenance flags.
4. Accepted revisions retain their existing highest priority.
5. Bulk rows and then legacy rows remain selectable after current interactive
   work advances; the change
   must not hide or discard catch-up.
6. Each slice retains its existing one-durable-item and at-most-one-provider-call
   bound. This is prioritization, not semantic batching.

A cross-lane RED regression uses distinct `ACCEPTED_REVISION`,
`FRESH_AUTOMATIC`, `KNOWN_BULK_IMPORT`, and `LEGACY_BACKFILL` markers across
classification, structured extraction, title, entity, and Page growth. The
service-class group passes 4/4; migrations 80–83 pass 4/4; both collision-lineage
migrations pass 2/2. The four existing deterministic slice bounds each pass
1/1 for Document, Entity, Reconcile, and Citation.

## Source-grounded controller facts

- Admission is checked before a scheduler turn. It requires two 30-second healthy samples, aggregate CPU at or below 20%, and available memory at or above both 2 GiB and 15%.
- One ambient turn selects at most one durable item and `AmbientBudgetProvider` forwards at most one LLM request.
- A selected thermal turn receives `max(2 minutes, 19 × elapsed)` recovery.
- A committed Page-growth terminal no-match forwards zero provider calls and
  does not consume the global thermal turn. Selected matches, panics, and any
  provider call still consume it.
- An already-started on-device request is not preempted when foreground activity begins. The pinned `llama.cpp` C API contains an abort callback, but `llama-cpp-2` 0.1.143 does not expose a safe context-level cancellation method.
- The existing continuous-batch engine only batches requests that are submitted concurrently. The serialized ambient controller currently submits one request, so its one-request invariant is not secretly a multi-item physical batch.

## RED/GREEN profiler gate

- RED: `rtk cargo test -j 2 -p wenlan-server rb01_profile_admission --lib -- --nocapture`
  failed at compile time with 13 expected missing-symbol errors for the profile admission type, block reasons, and explicit opt-in gate.
- GREEN: the same command passed the two admission/opt-in tests.
- Current head, after adding the gated real-model fixture and Page-growth lane,
  passes `3 passed, 1 ignored, 282 filtered out` for `rb01_profile_`.
- `rtk bash -n scripts/profile-ambient-rb01.sh` passed.
- RED: `rtk bash scripts/test-profile-ambient-rb01.sh` failed with
  `expected exactly two real-profile preflight calls, found 1`; the script
  would compile before checking whether the Mac was already busy.
- GREEN: the same shell regression passed after the real path became
  `preflight -> bounded -j 2 build -> preflight -> inference`. The second
  admission window ensures build heat cannot flow directly into model load.
- RED: the shell regression then failed with
  `expected preflight to enforce model working-set headroom exactly once`.
  On this 16 GiB Mac, the cached GGUF is 2,497,281,120 bytes, so a bare 15%
  pre-load check could pass and immediately enter compression after model load.
- GREEN: preflight now also requires estimated free bytes to cover the cached
  model file plus `max(2 GiB, 15% total memory)`. The static safety regression
  and syntax check both pass.
- RED: the shell regression then failed with
  `expected cooldown to use 19x measured job time with a timeout-sized fail-safe`.
  The script had always written 600 seconds even though production uses
  `max(600 seconds, 19 × job elapsed)`.
- GREEN: before the child can load a model, the script now arms a conservative
  `19 × 300-second timeout` recovery window. A complete JSON report replaces it
  with the then-current `max(600 seconds, 19 × report_elapsed_ms)`; an interrupt or missing
  report leaves the fail-safe in place. The manifest records which source won.
- CALIBRATION RED/GREEN: after all five isolated rows bounded observed turn
  time at 5.528 seconds, the policy test failed at `left: 600s, right: 120s`.
  Production and the profiler now use
  `max(120 seconds, 19 × report_elapsed_ms)` while retaining the unchanged
  5,700-second nonzero fail-safe.
- RED: the static harness rejected manifests that identified only `git_head`
  while the measured source tree could still be dirty.
- GREEN: every real-profile manifest now binds the exact scheduler, DB selector,
  and profiler script bytes with SHA-256 values.
- RED: both the Rust lane parser and shell surface rejected `page-growth`.
- GREEN: `page-growth` now profiles one synthetic memory with no Pages. It must
  create a durable `page_growth` receipt with zero provider calls, and
  measures the CPU-only embedding/search path. A later policy regression now
  exempts only a committed terminal no-match from the thermal turn.
- RED after Opus review: the generic batch-import path used the atomic origin
  writer but fresh rows entered service class 0. The review's explanation that
  every folder row was selector-excluded was source-inaccurate; the production
  caller is `importer.rs`, not directory sync.
- GREEN: fresh generic imports now enter class 1. Conflict updates preserve the
  existing class, so re-import cannot demote a user-promoted class-0 memory.
- RED after the first admitted run: `raw.log` contained only RTK's one-line
  Cargo summary, so the JSON measurement and measured cooldown could not be
  recovered. The run remained successful partial evidence, but not a complete
  profile row.
- GREEN: Cargo now runs only the bounded `--no-run` build with JSON metadata.
  The script extracts the exact `wenlan_server` test executable, hashes it, and
  invokes that binary directly through `rtk proxy`, so `--nocapture` JSON is not
  filtered by Cargo/RTK. The failed first parse correctly left the already-armed
  5,700-second fail-safe cooldown intact.
- RED: the profiler promised peak/current RSS but emitted only point-in-time
  baseline/model-loaded/before/after values.
- GREEN: a 50 ms test-only sampler now refreshes only the current PID through
  the exact `sysinfo` version resolved by `Cargo.lock` and emits
  `rss_peak_during_slice_bytes`; it does not
  call the system-wide `refresh_all()` loop while the slice runs.
- RED after independent review: two concurrent scripts could both pass the
  cooldown check before either armed it.
- GREEN: an atomic `/private/tmp/wenlan-rb01-profile.lock` is acquired before
  the first preflight and held until the child and manifest are finalized.
- RED after independent review: a nonzero child could print a JSON row before
  its assertions failed, allowing the shell to shorten the fail-safe from that
  invalid report.
- GREEN: `cooldown_from_report` is reachable only when the exact child exits
  zero. Nonzero, interrupted, timed-out, or unparsable runs retain the
  timeout-sized recovery window.
- RED after independent review: source hashes did not identify the actual test
  executable, dependency lock, or cached model blob.
- GREEN: the manifest now adds SHA-256 for `Cargo.lock`, the exact test binary,
  the tracked worktree diff, and the scheduler/DB/profiler sources, plus the
  resolved content-addressed Hugging Face model blob hash. A bounded no-run
  build proved Cargo selects one `wenlan_server` test executable, and invoking
  that exact binary passed 3 profiler harness tests with the real-model test
  ignored.
- RED on the current Mac: the hardened script used `/usr/bin/jq`, but Homebrew
  provides `jq` at `/opt/homebrew/bin/jq`.
- GREEN: `jq` is resolved from `PATH` and absence fails before compilation. The
  shell syntax/safety regression passes with the installed path.
- RED after tracing pinned `fastembed 5.13.1`: the profiler proved only the
  Qwen cache hit. DB startup could still fall through to FastEmbed's downloader,
  and an inherited `HF_HOME` overrides Wenlan's resolved cache directory.
- GREEN: preflight now validates the exact FastEmbed `refs/main` snapshot and
  all five files read by `TextEmbedding::try_new`. Both real child lanes clear
  `HF_HOME`, pass the checked cache explicitly, and fail closed against an
  unavailable loopback endpoint. The live daemon harness refuses to spawn
  without that explicit cache. Shell syntax/safety tests pass and the ignored
  integration target compiles with locked offline dependencies; no live daemon
  or inference was started for this proof.
- RED during pre-live process review: the timeout alarm killed only the test
  process. A signal does not run its `ChildGuard` destructor, while shell
  cleanup skipped termination once that PID was gone, so the daemon grandchild
  could remain alive and keep the model resident.
- GREEN: each measured child now creates a dedicated process group before
  `exec`. Cleanup targets that group even if the leader already died, gives it
  six seconds to exit cooperatively, then force-kills only the same isolated
  group and verifies it disappeared. A surviving group makes the profiler
  fail rather than emitting only a warning. The shell regression statically
  requires both lanes and both cleanup paths, and its macOS positive control
  starts a real grandchild and proves the group is empty after `TERM`.
- A later independent pre-live review rejected the first daemon harness
  because selected-turn-only timing could false-pass a 590-second restart,
  automatic work could invalidate the fixed timeout, there was no in-run
  thermal watchdog, provider residency was checked only at turn endpoints,
  failure data/provenance/environment isolation were incomplete, and the
  aggregate audit used multiple read snapshots while evaluating content.
  Each observation received a focused guard before any daemon run:
  stderr lines are timestamped in the reader; every ambient start is checked
  against the exact prior completion plus its published `next_eligible_ms`;
  any heavy automatic outcome fails the isolated fixture immediately;
  provider residency is sampled every 60 seconds between selected turns with
  at least one check inside the measured two-minute floor; the 30-second
  fail-closed watchdog covers
  model load and run; failure DBs and exact binaries/source bytes persist; the
  daemon uses an environment allowlist; and all backlog counts now share one
  read transaction without reading memory content or titles.
- The first follow-up review closed five of those six groups but found one
  remaining provenance race: another Cargo process could replace a hashed
  target binary during the second 60-second admission window. The binaries are
  now frozen before that window and executed only from the run-specific
  SHA-keyed artifact. A shell positive control first accepts the frozen test
  and daemon copies, mutates the daemon copy, and proves the same final
  verifier rejects it. The second follow-up verdict remains pending.
- RED/GREEN evidence for that hardening: the 590-second event regression
  panics as required; heavy automatic work and below-floor start telemetry
  each panic; thermal, memory-pressure, and unavailable-telemetry cases all
  return a fatal watchdog reason; the cheap live-harness target passes
  4 tests with the real model test ignored. Profiler and audit fixtures pass,
  shell syntax passes, scheduler passes 72 with one real-model test ignored,
  all-target `wenlan-core`/`wenlan-server` clippy is clean, rust-analyzer
  reports no warnings in the live harness, and formatting passes. The
  independent follow-up verdict is still pending; no live model run is
  authorized by these local greens alone.
- RED after final Opus review:
  `drift_guard::behavioral_flags_are_documented` failed on
  `WENLAN_RB01_PROFILE`, `WENLAN_RB01_LANE`, and the existing
  `WENLAN_TEST_STARTUP_SIGNAL_BARRIER`. The guard scans source text and does not
  exempt `cfg(test)` reads.
- GREEN: all three process-only test controls are explicitly documented in
  `FLAG_ALLOWLIST` with test-only reasons. The same fail-closed drift-guard test
  passes. Opus reported no other blocker in the service-order, one-call,
  lock/cooldown, SQL, or profiler provenance changes.
- RED after the independent current-tree review: manual content edits could
  replace a source set read outside PageWrite, successful no-op and gated
  outcomes lacked retry receipts, deleting/recreating a deterministic Page id
  retained the deleted generation's history, and daemon startup loaded a
  selected cached model before any CPU/memory admission.
- GREEN: manual edits now derive preserved sources from the exact Page
  generation inside the CAS, and their retry digest excludes that mutable
  server-owned state. Mutating, unchanged, acknowledged, and gated success
  responses have durable receipts; revision-card rows share the receipt
  transaction. Page deletion clears that id's history in the same transaction.
  Startup model load now waits for two quiet samples and requires the registry
  working set above the ordinary 2 GiB/15% reserve; a sticky reservation blocks
  automatic heavy turns from racing that load.
- RED after the follow-up review: two concurrent gated writes could both miss
  the initial receipt lookup. The winner atomically committed its revision card
  and receipt, while the loser's card rolled back but leaked the receipt primary
  key conflict to the caller.
- GREEN: the losing transaction now reloads and replays the winning response
  only when the stored digest matches; a different request still conflicts.
  The deterministic concurrent regression passes, and leaves exactly one
  pending card. Focused PageWrite passes 28/28,
  scheduler passes 70 with the manual real-model test ignored, manual HTTP
  edit passes 5/5, schema lineages pass 2/2 with real history/receipt bytes
  preserved, and startup admission/reservation passes 2/2 plus the daemon
  working-set test 1/1. The bounded follow-up review returned `SHIP` with no
  release blockers. Live startup/residency proof is still pending.

## Target-Mac preflight refusals

`rtk scripts/profile-ambient-rb01.sh --preflight-only` correctly refused to start
twice:

```text
refusing profile: aggregate CPU sample 1/2 was 36.79% (>20%)
refusing profile: aggregate CPU sample 1/2 was 41.42% (>20%)
refusing profile: aggregate CPU sample 1/2 was 41.5% (>20%)
refusing profile: aggregate CPU sample 1/2 was 43.68% (>20%)
refusing profile: aggregate CPU sample 1/2 was 23.4% (>20%)
refusing profile: aggregate CPU sample 1/2 was 23.3% (>20%)
refusing profile: aggregate CPU sample 1/2 was 33.3% (>20%)
refusing profile: aggregate CPU sample 1/2 was 67.26% (>20%)
refusing profile: aggregate CPU sample 1/2 was 33.02% (>20%)
refusing profile: aggregate CPU sample 2/2 was 54.29% (>20%)
refusing profile: aggregate CPU sample 2/2 was 37.41% (>20%)
```

A later current-source Entity attempt passed its first admission window at
thermal state `0`, `61%` free memory, and `16.58%`/`16.62%` aggregate CPU.
After the exact profiler binary was compiled, the mandatory second window
refused before model load because its first CPU sample was `20.8%`. No inference
ran and no measurement row was produced. A single cached-build retry then
refused in its first admission window at `22.12%`, again before compilation or
model load. After a further quiet interval, the next attempt passed its first
CPU sample but refused on the second at `23.1%`; it also ran no inference.
The next guarded attempt saw an `89.73%` aggregate sample and stopped
immediately. A read-only process snapshot attributed the spike to a separate
Claude worktree's `wenlan_core` test binary at roughly `609–773%` process CPU,
with another `rustc` active; the ambient branch did not start its model.

The first admitted current-source Entity run then exposed a release blocker
before inference. Both shell gates passed at thermal state `0`, `48%` free
memory, and CPU samples `11.28%`/`14.23%` then `8.74%`/`12.64%`. The cached
Qwen3-4B provider loaded a roughly `2.38 GiB` Metal model buffer, but the
production `SystemResourceProbe` returned
`available_memory_bytes: 0` with `total_memory_bytes: 17179869184` and refused
the slice as `MemoryPressure`; the surrounding macOS probe still reported
thermal state `0`, and the profiler manifest reported `44%` free memory after
the model was released. No Entity request ran.

This is grounded in the dependency source rather than inferred from the failed
profile. The failed binary pinned `sysinfo 0.33.1`; its macOS implementation
computes available memory as
`free + inactive + purgeable - compressor_page_count` with saturating
subtraction. Upstream `sysinfo 0.38.3` explicitly corrected macOS
available/used-memory accuracy to the Apple XNU accounting and removes that
zero-saturation path. That release requires Rust 1.88; this repository pins Rust
1.95.0. The approved fix now pins exact `sysinfo = "=0.38.3"`. A deterministic
lockfile regression failed on `(0, 33, 1)` and passes on `(0, 38, 3)`; a macOS
production-probe smoke test also reports nonzero total and available memory.
`cargo tree -i sysinfo` resolves only `sysinfo v0.38.3`, the current bounded
scheduler filter passes 70 tests with the manual real-model profile ignored,
and current clippy, formatting, and diff checks pass. These are dependency and
probe gates; the later five isolated lane rows supersede the failed Entity
attempt, so no additional isolated rerun remains.
The failed run is preserved at
`/private/tmp/wenlan-rb01-profile-20260720T053812Z-entity` and its 5,700-second
failsafe cooldown was honored before the dependency change.

The first admitted exact-`0.38.3` rerun passed both shell admission windows at
thermal state `0`, `72%` free memory, and CPU samples
`16.63%`/`19.2%` then `18.3%`/`18.66%`. After the roughly `2.38 GiB` model
loaded, the production probe reported `available_memory_bytes: 9730736128`
with `total_memory_bytes: 17179869184`, proving the false-zero blocker is
removed under the real model residency. The slice still did not run:
model-load CPU was `34.22874%`, so the production policy correctly returned
`CpuBusy` before Entity inference.

That refusal exposed a profiler-only timing mismatch. Production retains the
provider and its resource probe across scheduler polls, resets admission on the
load spike, and later requires two consecutive quiet samples. The manual
profile helper instead tried exactly two post-load samples and failed on the
first busy sample. A RED regression first failed because no bounded-retry seam
existed. GREEN now permits only `CpuBusy` and `Warming` to retry for at most
four samples; memory pressure, thermal pressure, unavailable telemetry, and a
busy final sample remain immediately or terminally fail-closed. The focused
profile logic group passes 4 tests with the real-model test ignored. The failed
run is preserved at
`/private/tmp/wenlan-rb01-profile-20260720T152941Z-entity`; its 5,700-second
failsafe cooldown remains authoritative before the next real-model attempt.

Independent review then found that simultaneous high CPU and low memory was
reported as retryable `CpuBusy` because CPU was checked first. The overlap RED
returned `Retry` instead of `Fail(MemoryPressure)`. GREEN gives memory pressure
reason priority—both conditions still deny admission, but the profiler now
releases the resident model immediately. The focused overlap regression passes
1/1 and the bounded scheduler filter remains 70 passed with the real-model test
ignored.

The earlier preflight refusals loaded no model and ran no inference; the
admitted dependency failure did load and release the model but ran no Entity
inference. These are positive-control results for the foreground-protection
gate, not thermal measurements. A later
metadata-only snapshot also found unrelated `rustc` processes consuming
approximately 78–87% CPU each, so neither the focused RED test nor the real
profile was layered on top. A bounded five-attempt waiter subsequently
reacquired the complete two-sample gate on every attempt and stopped with exit
75 after all five refusals; it never reached its single focused Cargo command.

The current quiet-host control passed at 8.44%/7.33% aggregate CPU, thermal
state 0, and 59% memory free. The Page-growth run then passed its full
pre-build window at 8.31%/5.87% and its post-build window at 4.49%/4.41%;
thermal state remained 0 and memory free was 66% then 73%.

A later Entity attempt refused before model load when its second aggregate CPU
sample reached 37.41%; the process snapshot was led by macOS `fseventsd`, not
Wenlan inference. A subsequent preflight-only check passed at 17.9%/16.4%,
thermal state 0, and 52% free memory. The inference was deliberately not
started because the independent review had already invalidated the current
source tree for profiling.

## Measurements

| Lane | Real-model wall time | LLM calls | RSS before/after | CPU before/after | Thermal before/after | Durable progress |
|---|---:|---:|---:|---:|---:|---|
| Document | 4,302 ms | 1 | 1,854,029,824 → 2,427,699,200 bytes; peak 3,090,432,000 | system 16.03% → 34.60% | 0 → 0 | yes |
| Entity | 5,528 ms | 1 | 1,581,645,824 → 1,940,357,120 bytes; peak 2,266,497,024 | system 15.51% → 25.86% | 0 → 0 | yes |
| Page Growth (terminal no match) | 154 ms | 0 | 2,544,271,360 → 3,058,827,264 bytes; sampled peak 3,058,450,432 | system 5.54% → 5.46% | 0 → 0 | yes |
| Reconcile | 1,833 ms | 1 | 3,070,492,672 → 3,048,161,280 bytes; peak 3,818,110,976 | system 13.42% → 19.10% | 0 → 0 | yes |
| Citation | 1,885 ms | 1 | 3,359,490,048 → 3,889,840,128 bytes; peak 4,554,539,008 | system 15.84% → 28.57% | 0 → 0 | yes |

### CPU admission and foreground-latency calibration

The 2026-07-21 target-Mac calibration used the same cached `qwen3-4b`, one
Document slice, a bounded synthetic foreground load, 250 ms CPU/RAM samples,
and a 5 ms timer probe. The acceptance rule was fixed before the runs: during
inference, timer p99 must stay at or below 10 ms and no more than 5 ms above
the loaded baseline. The load was admitted only when its measured starting CPU
fell inside the requested band; an out-of-band 15.65% attempt was recorded as
skipped and ran no inference.

| Starting CPU | Inference | CPU peak | Timer p99 baseline → during | Delta | Thermal | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 20.70% | 4,447 ms | 91.26% | 1.241 → 3.014 ms | +1.773 ms | 0 → 0 | pass |
| 23.82% | 3,374 ms | 91.44% | 2.275 → 8.036 ms | +5.761 ms | 0 → 0 | reject 25% candidate |
| 28.89% | 3,493 ms | 93.10% | 2.336 → 7.436 ms | +5.100 ms | 0 → 0 | reject 30% candidate |

The production ceiling therefore remains 20%. The result does not claim that
24% causes thermal pressure; it shows that the short inference peak consumes
enough scheduler headroom to miss the foreground-latency budget before thermal
state changes.

Across these runs, the largest process-RSS increase from immediately before
the slice to its sampled peak was 1,706,475,520 bytes (1.59 GiB). Production
rounds this to a 2 GiB on-device inference reserve above the existing
foreground floor. The three accepted calibration artifacts are:

- `c5e1a9813d6754e5959fffcbf4c6c28988b46bdf-e5fe4d9cf54c/20260722T023122Z-document`
- `c5e1a9813d6754e5959fffcbf4c6c28988b46bdf-f822e5cc1c00/20260722T025557Z-document`
- `c5e1a9813d6754e5959fffcbf4c6c28988b46bdf-da36e1939ae6/20260722T024617Z-document`

They are stored under
`/Users/lucian/.local/share/repo-data/wenlan/benchmarks/`. Each records the
exact source bundle, frozen test binary, model blob, runtime samples, and
foreground timer summary. No additional candidate or live lane matrix is
required.

These five isolated runs are stored under
`/Users/lucian/.local/share/repo-data/wenlan/benchmarks/c8a87bfee2661b4c474b2df63b2d259563252d57-52740615cfcc/`
at `20260721T000621Z-page-growth`, `20260721T021529Z-document`,
`20260721T023021Z-entity`, `20260721T024924Z-reconcile`, and
`20260721T030309Z-citation`. Each manifest exited zero and records the same
source bundle, scheduler, profiler, Cargo lock, model blob, and frozen test
binary provenance. The isolated process intentionally reloads the roughly
2.32 GiB GGUF for each lane, so its total test duration is not the production
per-turn cost; the table reports the measured scheduler slice.

The persistent-daemon attempt is preserved at
`/Users/lucian/.local/share/repo-data/wenlan/benchmarks/c8a87bfee2661b4c474b2df63b2d259563252d57-52740615cfcc/20260721T031717Z-daemon`.
It loaded the model once at `03:20:30Z`, stored
the fixture at `03:20:31Z`, completed Classification in 3,222 ms at
`03:32:32Z`, and completed StructuredExtract in 2,331 ms at `03:43:04Z`.
The second turn began 600.07 seconds after the first completion and no second
model-load event appeared. The HTTP server and scheduler drained and logged
`graceful shutdown complete`, after which the old `std::process::exit` path
crashed in `ggml_metal_device_free`; the manifest therefore correctly records
exit 101 and retains the nonzero fail-safe. The macOS crash report is
`~/Library/Logs/DiagnosticReports/wenlan-server-2026-07-20-204311.ips`.

The teardown RED registered a C exit handler that terminates with status 71;
the old `exit_daemon(0)` produced 71. Unix `_exit` skips that handler and the
same test now exits zero. The complete binary/startup group passes 20 tests,
graceful shutdown passes 4, the cheap live harness passes 8 with the real-model
case ignored, the profiler shell safety regression passes, and the scheduler
module passes 72 with the real profile ignored.

The representative proof is stored at
`/Users/lucian/.local/share/repo-data/wenlan/benchmarks/c8a87bfee2661b4c474b2df63b2d259563252d57-4549c000bdf7/20260721T055753Z-daemon`.
Preflight was thermal state 0 with 72% free memory and 12.37%/12.26% aggregate
CPU; post-build was thermal state 0 with 72% free memory and 13.87%/11.41%
CPU. The daemon loaded the model once. Classification used one provider call
and completed in 2.629 seconds; the immediate 21.09% CPU sample denied another
turn, and the next 13.54% sample remained in the required two-sample warming
state. StructuredExtract then used one provider call and completed in 1.935
seconds, beginning 150.019 seconds after the first completion. Both durable
receipts target version 1. All 32 thermal samples remained 0, peak daemon RSS
was 3,462,561,792 bytes, graceful shutdown completed, and the process exited 0.

RB-01 is closed. The five isolated lane profiles cover lane functionality; the
single representative daemon proof covers persistent model residency, resource
admission, dependency order, cooldown, thermal state, and clean exit. No
additional live matrix is required for this change. The later CPU calibration
keeps the 20% ceiling and adds the deterministic 2 GiB route-memory reserve;
it does not require repeating the completed lane matrix.

The final production host-gate proof is stored at
`/Users/lucian/.local/share/repo-data/wenlan/benchmarks/b199d9301fdc008dcc7eb2b50ed8737173fce137-aa136449abb5/20260722T045828Z-daemon`.
Preflight passed at thermal state 0, 74% free memory, and 10.64%/9.62% CPU;
post-build admission passed at thermal state 0, 74% free memory, and
10.19%/10.15% CPU. After model load, production deferred new turns for three
CPU samples above the 20% ceiling, then completed Classification in 2.651
seconds and StructuredExtract in 1.503 seconds with one call each. The measured
completion-to-start gap was 120.018 seconds, all 14 watchdog thermal samples
were 0, peak daemon RSS was 3,474,391,040 bytes, both version-1 receipts were
durable, graceful shutdown completed, exit status was 0, and no daemon remained.

The preceding interrupted current-source run is retained at
`20260722T044647Z-daemon` as a positive control for the physical-input veto. It
made no inference: after CPU recovery it logged `ForegroundActive` on four
successive scheduler polls and did not begin another heavy turn. The operator
then terminated the harness; that nonzero run is not completion evidence.

The final Fable review changed only native build packaging and a non-macOS lint
annotation after these artifacts: the same Objective-C object is now placed in
a deterministic static archive with normal Cargo link metadata, so downstream
crates receive it, and non-macOS builds explicitly allow macOS-only enum
variants to remain unconstructed. Scheduler behavior and the macOS probe code
did not change. Fresh local builds linked both the `wenlan-server` binary and
the downstream `wenlan-mcp` `real_router` test executable without warnings;
the real-model soak is not repeated for these build-only corrections.
