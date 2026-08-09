# Ambient Enrichment Scheduler Implementation Plan

Status: Integrated with the locally available `origin/main` at `9cb6c6ac`; the
live remote could not be refreshed because SSH authentication failed, so this
is not a claim that the remote ref is still current. Bounded automatic
implementation, foreground resource protection, fixed-lane service ordering,
both schema-collision upgrade lineages, and cross-surface source consent are
locally verified. No local product release blocker remains open; ordinary
required-CI runner execution still applies. RB-01 is closed by five isolated
lane profiles plus one representative persistent-daemon soak: the daemon
loaded the model once, completed the Classification → StructuredExtract
dependency chain, stayed at nominal thermal state for all 14 current-source
samples, and
exited cleanly. A later target-Mac foreground-latency calibration retained the
20% CPU ceiling, rejected 25% and 30% candidates, and added a 2 GiB
on-device inference reserve above the ordinary memory floor. RB-05's workflow contract is corrected locally; its
Linux/Windows runner execution remains a normal required-CI gate.

> Scope: The conservative scheduler plus approved fixed memory stages. Store and imports perform durable handoff only; document, classification, structured extraction, entity, title, Page growth, reconcile, and citation share ambient admission. Automatic maintenance uses bounded stage slices. Automatic steep is fail-closed to the one bounded ReDistill Page slice; legacy global phases remain available only through explicit foreground steep. No semantic batching, generalized dependency graph, or non-macOS host-activity probe.

## Contract

- The scheduler owns document, classification, structured extraction, entity, title, Page growth, reconcile, and citation lanes.
- One ambient turn selects at most one durable work item and performs at most one LLM request.
- One admitted automatic turn selects at most one bounded item; every provider role used by that turn shares one hard one-inference budget.
- Ambient turns never overlap, never detach, and do not start during cooldown, elevated system CPU, low available memory, recent macOS physical input, unavailable supported host telemetry, or non-nominal macOS thermal state.
- Heavy automatic work requires two consecutive 30-second resource samples. Target-Mac calibration keeps aggregate CPU at or below 20%. Available RAM must exceed `max(2 GiB, 15%)`; on-device background routes additionally reserve 2 GiB for inference, while remote routes retain the ordinary floor.
- Every selected item either advances a durable checkpoint or records a terminal/retry outcome.
- Ambient admission has no fixed Wenlan-write delay. `WriteSignal` is used only for trigger-specific recap batching, not as an OS foreground-idle proxy. Inter-turn recovery is `max(120 seconds, 19 × elapsed)`; production CPU, RAM, and two-sample admission can extend that delay but never shorten it. On macOS, public OS signals also veto model startup and new heavy turns for 60 seconds after physical input and throughout any non-nominal thermal state.

The automatic store/import enrichment path is inside the bounded controller. Automatic maintenance runs one cursor-backed bounded stage slice. Automatic steep cannot enter Decay, Promote, Recaps, Reweave, Reembed, EntityExtraction, CommunityDetection, Detect, Emergence, SummaryRollup, Overview, RefinementQueue, DecisionLogs, PruneRejections, Evict, or KgRethink; those legacy whole-corpus algorithms are explicit foreground work until they independently earn a bound and convergence proof. ReDistill admits one Page, rejects more than 64 sources before provider invocation, caps each source to 800 characters, caps topic/title-hint text, and advances its durable cursor on an oversized Page.

## Decisions, compressed

Everything below is an engineering correction and does **not** need a product decision: bounded graceful shutdown, panic isolation around provider availability, invalidating stale entity links on memory edits, scheduling maintenance as its own durable turn, citation retry accounting, reconcile backoff, and updating the ingest-parity rule.

The source-policy decision is closed: automatic work uses a hard pin. No pin means paused; a present but unavailable pin also means paused; neither state falls back to Anthropic, external, or on-device. A healthy explicit pin authorizes deferred work but never an inference in the store request path. RB-06 now gives App, CLI, Claude plugin, and Codex plugin the same disclosed consent contract without weakening this runtime boundary.

The numeric thermal/convergence decision is evidence-backed: keep the 20% CPU ceiling and `max(120 seconds, 19 × elapsed)` recovery. The 20.70% calibration point met the foreground timer budget; 23.82% and 28.89% did not. All remained thermally nominal, so the lower ceiling protects responsiveness rather than pretending to be a universal thermal boundary.

## Release blocker register

Every row marked `OPEN` blocks a thermal-ready release. A blocker may move to `CLOSED` only when its exit criteria and evidence address are filled; implementation intent is not evidence.

| ID | Status | Blocker | Why it was not completed in Phase 1 | Exit criteria | Evidence |
|---|---|---|---|---|---|
| RB-01 | CLOSED | Target-hardware thermal and convergence envelope | The fixed ten-minute ambient delay was removed rather than treated as foreground protection. Five isolated lanes provide functional/timing evidence; one representative persistent daemon proves the resource/cooldown loop. A later bounded calibration measured foreground timer jitter at 20.70%, 23.82%, and 28.89% starting CPU instead of guessing a higher threshold. macOS production admission now also uses public physical-input and thermal signals. | Busy-host and recent-input refusal, clean exit, one model load, Classification → StructuredExtract dependency, mid-cooldown residency, production thermal admission, 30-second thermal/RAM watchdog, bounded RSS, durable receipts, foreground-latency calibration, and fast deterministic scheduling/retry/CAS/one-call gates. | `docs/plans/2026-07-18-ambient-rb01-evidence.md`; current-source daemon artifact `20260722T045828Z-daemon`; 14/14 thermal samples nominal; 120.018-second turn gap; 20.70% candidate passed foreground p99 budget while 23.82% and 28.89% failed; CPU ceiling remains 20%; on-device routes reserve an additional 2 GiB; process exit 0. |
| RB-02 | CLOSED (automatic path) | Legacy global steep phases were not item-sliced | A one-phase wrapper was insufficient because every non-ReDistill phase still executed its full legacy algorithm. The release seam is therefore an explicit automatic allowlist, not pretending every phase is bounded. | Automatic scheduling can invoke only phases with a source-proven per-turn bound and durable cursor; foreground trigger semantics stay unchanged; unsupported BurstEnd/Daily work cannot create an empty-batch panic or hot-loop. | `crates/wenlan-server/src/scheduler.rs`; scheduler module 56/56; ReDistill cap tests |
| RB-03 | CLOSED | Accepted reconcile revisions did not converge through fixed enrichment dependencies | Acceptance stays SQL-only and invalidates the fixed receipts for the new head version. The existing ambient lanes then classify, extract structured fields, resolve entity/title/Page work over later turns under the shared budget. | Prove accepted revision priority and eventual receipt convergence without nested acceptance inference. | focused accepted-revision convergence test; `crates/wenlan-core/src/post_write.rs`, `db.rs`, `reconcile.rs` |
| RB-04 | CLOSED | Graceful shutdown and non-destructive `background off` | Shutdown signals the scheduler, bounds process drain at 1.5 seconds, and preserves launchd/systemd/Task Scheduler registration. If a registered daemon is still starting or respawning and its port refuses connections, CLI stop now stops the supervisor instead of falsely reporting success. Signal handlers are installed before bind or durable startup work, so a SIGTERM arriving during startup cannot take the default abrupt-exit path. | Code-level and child-process tests prove cooperative stop, bounded process exit, startup-signal handling, registered-manager fallback, and preserved registration semantics. | `crates/wenlan-server/src/lifecycle.rs`; `crates/wenlan-server/tests/graceful_shutdown.rs` (4/4); `background_off_` CLI integration group (8/8) |
| RB-05 | CLOSED (runner execution pending) | Required CI asserted destructive `background off` behavior | The workflow contradicted the supported stop-but-keep-registration contract. User approval was required before changing required CI. | Linux/Windows assertions verify stopped-but-registered; a separate test-only cleanup always removes the unit/task. | `.github/workflows/ci.yml`; YAML parse and Linux shell syntax pass locally; Windows runner is the remaining execution gate |
| RB-06 | CLOSED | Cross-surface source consent/onboarding | The runtime hard-pin boundary existed before every surface could explain and configure it. The App now writes no pin until the exact Everyday/Synthesis mapping is shown and confirmed; CLI and both plugin surfaces use the same explicit flow. Unknown older routing contracts are update-required rather than misreported as off. | Every surface discloses automatic work and local heat or remote data/cost, requires consent, writes the same hard pins, supports off/disable, and reports ready/paused/off without fallback. | wenlan-app `7a47b40`; App 133/133 focused tests, production build, Rust routing DTO 2/2, and rendered EN/zh-Hans/zh-Hant consent plus off/paused/legacy/update-required states; Wenlan CLI 89 passed/1 manual distribution smoke ignored; enrichment consent 5/5; plugin contract guard passed; final Opus reviews completed. |

Verification risk, tracked but not currently classified as a product release blocker: the unbounded full-workspace verifier itself drove the `wenlan_core` test binary to roughly 535–570% CPU for more than a minute and was interrupted after about 4.5 minutes to avoid recreating the user-facing thermal failure. No assertion failed before interruption. Release evidence uses serialized, bounded module/package gates; required CI still needs its ordinary complete runner execution.

## Review integration

- Claude first proposed a per-request semaphore, then its implementation review correctly identified serialization as redundant once every ambient job was awaited inline by the single poll loop. The final provider facade has a different purpose: it hard-caps each ambient slice at one forwarded inference and fails closed on any hidden nested call; the same counter feeds telemetry and cooldown decisions.
- Fable rejected semaphore-only serialization because a large document could immediately reacquire it and sustain load. The implementation therefore returns to the scheduler after every request.
- Fable correctly identified fresh-document preparation as a separate CPU risk. Source verification narrowed the claim: embedding is one batch call, not one call per chunk, and directory ingestion already caps text files at 1 MB and PDFs at 10 MB. It is still variable-cost work, so a selected fresh document consumes the same cooldown even if it made no LLM request.
- Store-time and chat-import enrichment no longer spawn the canonical multi-stage pipeline. They persist origin/backlog state and return; the ambient lanes own subsequent inference.
- The original ten-minute inter-turn cooldown was an explicitly conservative hotfix borrowed from the Idle recap window. Target-Mac evidence now separates the two: the ten-minute value remains only on the existing Idle recap trigger; ambient and other automatic admission do not use it. Minimum inter-turn recovery is two minutes with the 19× long-turn multiplier.
- Independent correctness review found and the implementation now covers same-poll activity races, document queue starvation, reconcile frontier starvation, historical no-provider entity rows, and non-atomic document checkpoints.
- The final Opus follow-up found two additional silent-stall paths: an unavailable configured API prevented fallback to an available local provider, and a panic inside an ambient slice terminated the scheduler task. Both were fixed RED-first; provider selection now chooses the first available backend, and ambient panics are isolated while still charging a conservative thermal cooldown. Focused recovery/fallback tests cover both paths.
- A later live audit found the larger automatic path: the old installed daemon repeatedly ran whole Idle steeps for roughly 10–19 minutes, including 1,138,342 ms and 771,001 ms runs with 682 pending. Source audit then proved that a one-phase wrapper was still unsafe: every phase except ReDistill fell through to a whole legacy algorithm. The corrective design is an automatic safe-phase allowlist, not more timeout/semaphore layers. Only ReDistill is eligible; maintenance uses its separate bounded stage round.
- BurstEnd and Daily currently have no allowlisted steep phase. Mature BurstEnd timestamps are drained as unsupported bookkeeping so they cannot grow forever or repeatedly win selection; this path does not charge a thermal cooldown. Idle and Backstop may admit one ReDistill Page. A maintenance panic remains isolated at the scheduler boundary, and an actually attempted bounded turn starts cooldown.
- An audit of the combined automatic/ambient loop found that even a completely
  empty ReDistill or maintenance scan was treated as an attempted thermal turn.
  With six maintenance stages this could delay useful ambient work by tens of
  minutes without doing any work. A focused RED/GREEN now charges the shared
  cooldown only when the automatic outcome selected an item, forwarded an LLM
  call, or panicked. Empty scans still yield one poll for ambient fairness.
- The final full working-tree Opus review rejected release: provider `is_available()` calls still escape the panic boundary, edited memories can retain stale entity links, steep-first shared budgeting can starve maintenance, and citation provider errors do not advance retry state. Its reconcile "every poll" wording was overstated because round-robin advances the cursor, but reconcile backpressure still lacks the intended backoff.
- Mainline synchronization first rested at `4fd6df7c`, then
  `0c273c52`, and now integrates the locally available `origin/main` at
  `9cb6c6ac`. The attempted live SSH fetch failed authentication, so the last
  ref is local evidence only. The ambient lanes and store response use the
  same hard-pin resolver. Chat import keeps durable handoff and contains no
  detached enrichment task; generic import records automatic-owned origin
  atomically with its memory rows.
- Claude's first follow-up incorrectly claimed revision acceptance already runs canonical enrichment; source verification showed acceptance is SQL-only. The final fixed-stage design keeps acceptance fast, invalidates versioned receipts, gives accepted revisions priority, and lets later ambient classification/extraction/entity/title/Page lanes converge without nested acceptance inference.
- Fable's final adversarial review rated the branch `needs-attention`, primarily because the unmeasured ten-minute fallback can make full first-boot backlogs converge over days. That remains a release-calibration blocker, not a reason to guess a faster constant. Its valid code-level finding — a spent reconcile slice still reading the second frontier — was fixed with a RED metadata-side-effect test and an early budget return. Claims that citation repeated the same lane every 30 seconds and that synthesis/external providers bypassed `run_ambient_job` were rejected after source verification: the cursor advances, and those provider handles are not passed into the ambient function.
- The final standard Opus review found five release-blocking gaps after the earlier green suite. CI still encoded destructive stop semantics; a Document provider panic leaked the claimed row as `in_progress`; Maintenance `StalePage` bypassed the 64-source automatic cap; numeric stale-page OFFSET cursors skipped rows when a successful refresh shrank the set; and zero-call reconcile watermark progress was mislabeled as an empty lane. Each finding was reproduced or source-proven, then fixed. Stale-page scans now use a v2 composite keyset plus an additive supporting index; reconcile progress consumes the global thermal turn without receiving the 30-minute empty-lane delay.
- The same review's round-robin cleanup is deferred because only ReDistill is allowlisted and the current metadata write is not a thermal defect. Its citation-counter duplication claim was rejected: citation attempts are Page-version state, while `enrichment_steps` is Memory-version state. Follow-up shutdown audits found and fixed both the registered-but-not-yet-reachable `background off` false-success path and the startup SIGTERM registration race. Medium cleanup remains for simultaneous timer/shutdown readiness and detached/admission task accounting; none escapes the 1.5-second process-exit bound, so they remain follow-up rather than thermal release blockers.
- A post-fix lock audit found five `handle_store_memory` stages that could retain a `ServerState` read guard across a DB await: dedup, agent gating, entity resolution, activity attribution, and activity logging. A five-stage behavioral regression first named each still-blocked stage, then all five paths were changed to clone owned DB/provider Arcs inside a short read scope. The final exhaustive Opus re-audit enumerated every state read in the handler and returned `SHIP` with no blocking findings. The synchronous per-store config read remains deliberate so App/CLI/plugin hard-pin changes become visible without a stale daemon cache.
- RB-06's final Opus reviews found two cross-surface honesty defects. CLI `enrichment disable` could swallow a live daemon's HTTP rejection, rewrite only disk, and falsely claim background work was off; it now falls back only on a real connection failure, while 4xx/5xx leaves pins untouched and exits nonzero. The App could label routing modes from an older pin/fallback contract as `Off`; it now shows `Update required` and withholds the source picker. Both fixes were reproduced RED-first and pass focused GREEN tests plus the full bounded gates above.
- Latest-main integration now terminates at schema 85. Ambient input versions,
  observation provenance, fixed-stage origin, stale-Page indexing, collision
  reconciliation, and explicit service classes occupy feature migrations
  78–83. Released main used 78/79 for Page history and operation receipts; the
  integrated chain places those idempotent tables at 84/85, while migration 82
  repairs the ambient 78/79 columns for an already-main-79 database. RED/GREEN
  tests start from both main-79 and feature-83 layouts so neither existing
  lineage silently skips the other schema.
- An independent RB-01 review found three real profiler defects: concurrent
  launches could race before cooldown, a nonzero child could shorten the
  fail-safe from pre-assertion JSON, and the manifest did not identify the
  executable/dependency/model bytes. All three were reproduced by the shell
  safety regression and fixed: one atomic lock spans both admission windows and
  the child, measured cooldown requires exit zero, and Cargo metadata selects a
  directly invoked test binary whose hash is recorded with `Cargo.lock`, the
  resolved model blob, source files, and unfiltered tracked diff.
- Pre-live process review found a fourth profiler defect: `SIGALRM` terminated
  only the test process, and its spawned daemon could survive because Rust
  destructors do not run on that signal. Both real lanes now enter a dedicated
  process group. Timeout, interrupt, and nonzero cleanup terminate the entire
  group, wait up to the daemon's cooperative shutdown bound, then force-kill
  only that isolated group if necessary. A macOS positive control proves a
  grandchild does not survive the same group-termination path.
- The next independent pre-live review rejected six harness/evidence gaps
  before any model run: selected-only timestamps and a ten-second cooldown
  tolerance, an unproven empty-automatic premise behind the 27-minute bound,
  no in-run thermal/memory watchdog, incomplete cleanup/failure-data/source
  provenance, inherited network/provider environment, and a multi-connection
  aggregate audit that evaluated content. The remediation now timestamps every
  scheduler line at reader receipt, checks every ambient start against the
  exact published deadline, rejects any heavy automatic outcome in the
  isolated fixture, samples provider residency during cooldown, and runs a
  30-second fail-closed thermal/memory watchdog from model load onward.
  Process-group cleanup gets six seconds plus post-KILL verification and
  returns failure if an orphan survives; daemon data lives in the external
  artifact; the manifest hashes the actual daemon/test binaries and a
  build-stable source byte stream including untracked files; daemon launch
  clears its environment; and the audit now uses one read transaction without
  content/title evaluation. Cheap harness, profiler, audit, scheduler,
  clippy/fmt, and LSP gates are green; the independent follow-up verdict and
  all live rows remain pending.
- The same review proposed treating every pre-service-class feature row as
  interactive or bulk. Source inspection rejects that backfill as
  unreconstructable: committed schema 82 wrote ordinary stores and imports into
  the same origin table, and identical all-false flags can represent either.
  This does not affect the supported release upgrade: `origin/main` is schema
  77 and migration 80 creates an empty table with `service_class` already in
  its schema. Migration 83 exists only to make interrupted feature-lineage
  databases structurally restart-safe; it does not fabricate provenance.
- The final current-tree Opus review found one required-gate blocker rather than
  a scheduler defect: the regex drift guard sees env reads inside test modules,
  so the two profiler controls and the startup-signal test barrier were
  undocumented flags. The fail-closed test reproduced all three names; adding
  them to the explicit test-only allowlist made the same gate green. Opus found
  no other violation in service ordering, one-call enforcement, SQL, Page
  Growth charging, lock/cooldown safety, or profiler provenance.
- A later independent current-tree review rejected four Page lifecycle/startup
  gaps that earlier focused gates did not cover: manual preserve-sources was
  outside the CAS, unchanged and gated retry identities had no receipts,
  delete/recreate reused old Page history, and selected local models loaded at
  daemon startup outside resource admission. Each now has a focused regression.
  Preserve-sources is resolved inside the Page CAS and excluded from caller
  digests; revision cards share their receipt transaction, acknowledged no-ops
  share their watermark transaction, delete clears the old history generation,
  and startup model load uses the shared two-sample policy with additive model
  headroom while reserving the scheduler's automatic-heavy boundary.
- The follow-up review found one remaining concurrent idempotency edge: two
  gated attempts could both miss the receipt lookup, then the loser surfaced
  the receipt primary-key conflict after its card transaction rolled back. A
  deterministic concurrent RED now proves that the loser reloads and replays
  the winner only for the same digest, with exactly one pending revision card.
  The bounded follow-up review returned `SHIP` with no release blockers.
- The first admitted current-source Entity profile found a platform blocker
  before inference: pinned `sysinfo 0.33.1` returned zero available memory after
  the model loaded while macOS still reported 44% free. The approved fix pins
  exact `sysinfo 0.38.3`, which replaces the subtract-compressed accounting
  that saturated to zero. The next admitted run proved nonzero available memory
  under real model residency, then correctly refused a `34.22874%` post-load
  CPU sample before inference. A profiler-only RED/GREEN now models
  production's settle-and-require-two-quiet-samples behavior with a four-sample
  cap; memory, thermal, and unavailable-telemetry failures remain immediate.
  Independent review exposed combined CPU and memory pressure being labeled as
  retryable CPU pressure; its overlap RED now passes with memory-first failure
  and immediate model release.
  The later representative daemon proof supersedes the missing isolated Entity
  rerun for persistent resource/cooldown behavior; the isolated lane matrix was
  not repeated.
- Wenlan inactivity alone was insufficient foreground protection because unrelated compilation or model work is invisible to the write signal. The scheduler now samples whole-system CPU and available RAM through the pinned `sysinfo` dependency, requires two quiet samples, and fails closed for every heavy automatic/ambient admission while the machine is busy or memory-constrained. Target-Mac calibration retained the 20% CPU ceiling and added a 2 GiB route-specific inference reserve because the largest observed one-turn RSS increase was 1.59 GiB.
- Post-merge review found relation vocabulary healing committed destructive collision folding before its recovery ledger. A forced-ledger-failure regression reproduced the provenance loss; relation mutation and pre-image ledger insertion now share one transaction, matching entity folding's atomic boundary.

## Why a semaphore alone is insufficient

`Semaphore(1)` prevents concurrent requests but still permits a large document to release and immediately reacquire the permit for every chunk. That can keep CPU/GPU utilization continuous and preserve the fan/heat failure. The unit of scheduling must therefore be one request followed by a return to the scheduler, not one end-to-end job guarded by a semaphore.

## Task 1: Make entity backfill a durable one-item slice

**Files:**

- Modify: `crates/wenlan-core/src/db.rs`

**RED tests:**

1. A database with five eligible rows invokes the extractor once and returns after one selected row.
2. An empty entity result records terminal `entity_extract = ok` and is not selected again; `skipped` remains a compatibility input for rows created when no provider was available.
3. A provider/parser error records `needs_retry`; the configured attempt cap records `abandoned` and prevents another selection.
4. A valid extraction records `ok`, links the entity, and is not selected again.

**Implementation:**

- Replace the drain-until-partial-batch behavior with `run_entity_enrichment_slice`.
- Select one candidate by joining `enrichment_steps`; allow fresh rows and retryable rows below the attempt cap.
- Stop recording a terminal entity-extract step when ingestion has no provider; retain compatibility for historical `skipped` rows without charging that old skip against the actual retry cap.
- Reuse `record_enrichment_step`; add no table or migration.
- Return a small report containing whether an item was selected and whether one LLM call was attempted.

## Task 2: Add an ambient one-request document slice without breaking foreground callers

**Files:**

- Modify: `crates/wenlan-core/src/db.rs`
- Modify: `crates/wenlan-core/src/document_enrichment.rs`

**RED tests:**

1. A multi-chunk document processes exactly one chunk, checkpoints it, returns, and leaves the queue claimable.
2. Repeated slices resume at the next chunk without repeating prior calls.
3. After the final chunk, the next slice performs only entity extraction and completes the document.
4. A failed request preserves the checkpoint and follows the existing pause/backoff path.
5. Summary persistence and the queue resume point commit atomically.
6. A yielded long document moves behind other pending documents.

**Implementation:**

- Add a queue transition that yields `in_progress` back to `pending` without incrementing the retry counter or discarding `last_completed_chunk`.
- Factor the existing pipeline behind a request budget.
- Keep `run_document_enrichment` as the end-to-end compatibility wrapper.
- Add `run_document_enrichment_slice` with a budget of one request for the ambient scheduler.
- After consuming the request budget, persist/yield and return before another LLM call. The existing last-chunk checkpoint is sufficient to make entity extraction a separate next slice.
- Persist each per-chunk summary and its resume index in one transaction; persistence failure pauses instead of yielding.
- Move a yielded row to the durable queue tail so one large file cannot monopolize every thermal turn.

## Task 3: Expose one-request reconcile and citation slices

**Files:**

- Modify: `crates/wenlan-core/src/reconcile.rs`
- Modify: `crates/wenlan-core/src/citations.rs`

**RED tests:**

1. Reconcile with multiple eligible candidates performs at most one judge call and preserves its watermark for the next slice.
2. Citation backfill with multiple eligible pages performs at most one annotate call.
3. A citation page with no evidence can become terminal without an LLM call, but the slice still returns after that selected page.
4. Reconcile alternates its persisted starting frontier so a steady document backlog cannot starve captures.
5. A judge-created revision does not trigger hidden canonical-enrichment calls in the same ambient slice.

**Implementation:**

- Keep the existing batch/tick entry points for explicit callers and current tests.
- Add ambient wrappers that pass a request/page budget of one through the same implementation.
- Do not add another queue: existing reconcile watermarks and citation attempt metadata remain authoritative.
- Stage ambient revisions with no nested provider. This preserves the hard thermal bound but leaves full accepted-revision classification as an explicit Phase-2 gap rather than silently breaking the one-call contract.

## Task 4: Split cheap directory sync from ambient document work

**Files:**

- Modify: `crates/wenlan-server/src/scheduler.rs`

**RED tests:**

1. Directory mtime/hash synchronization still runs every poll without claiming document work.
2. The document slice claims at most one queue item when selected by the ambient scheduler.

**Implementation:**

- Split `run_directory_sync_tick` into cheap source synchronization and one ambient document slice.
- Keep filesystem/database freshness outside the LLM gate.
- Remove `MAX_DOC_ENRICH_PER_POLL` and the document drain.

## Task 5: Add the private round-robin ambient controller

**Files:**

- Modify: `crates/wenlan-server/src/scheduler.rs`

**RED tests:**

1. Recent activity prevents every ambient job from starting.
2. Cooldown prevents another turn after a completed request.
3. When all eight jobs are due, eight eligible turns select document, classification, structured extraction, entity, title, Page growth, reconcile, and citation once each.
4. A job with no work does not starve later jobs.
5. Core slice tests prove one request per invocation, and the scheduler source contains no detached ambient job dispatch; the single poll loop awaits each invocation inline.
6. No periodic sweep uses `tokio::spawn` after the change.
7. The ambient provider facade forwards one request and rejects a second, so future nested calls fail closed in production as well as tests.
8. A lane that selected backlog remains due after the global cooldown; only an empty lane receives the 30-minute rescan backoff.

**Implementation:**

- Add a private closed `AmbientJob` enum and state containing the round-robin cursor, due times, and `next_allowed_at`.
- After cheap sync and normal trigger handling, check resource admission and cooldown, select one due job, await one slice inline, record duration/outcome, advance the cursor, and return to the poll loop.
- Remove detached entity/reconcile/citation sweep spawns.
- Emit compact telemetry: job, selected/progressed, LLM calls, elapsed milliseconds, attempt/outcome where available, and next eligible time.
- Re-read Wenlan write activity before trigger selection so a write that arrives during filesystem sync resets only the write-derived recap triggers.
- Keep known backlog governed by the single global cooldown and round-robin cursor; use the 30-minute lane timestamps only to avoid repeatedly scanning an empty lane.
- Admit steep and maintenance through the same resource/cooldown policy, with one trigger and one shared inference budget per turn. Trigger-specific batching remains inside trigger selection.

## Task 6: Profile and freeze the first production envelope

**Files:**

- Modify: `crates/wenlan-server/src/scheduler.rs`
- Update: this plan with the evidence command and measured values

**Verification:**

1. Record a pre-change single-request wall time and system pressure/CPU sample on the current supported Mac.
2. Run the same fixture through the one-request slice.
3. Choose the initial target duty cycle and minimum cooldown from that evidence, favoring slower convergence over perceptible fan/heat.
4. Add unit tests for the duration-to-cooldown calculation with an injected policy and paused Tokio time.

Protective code is complete: the injected policy keeps a measured two-minute
inter-turn floor, extends recovery after long turns, and adds a two-sample
CPU/RAM admission gate. On macOS, public OS probes also defer model startup and
new heavy turns after recent physical input or during non-nominal thermal
state. The ten-minute value remains only on the existing Idle recap trigger.
The representative persistent-daemon soak and five isolated lanes provide the
live and functional evidence. The timer and host-signal policies are covered by
deterministic scheduler tests plus a real macOS signal smoke test.

The controller's measured thermal envelope remains bounded to the target Mac.
Production now reads macOS thermal state directly before model startup and new
heavy turns; it does not claim equivalent OS telemetry on other platforms or
that model residency has no memory cost.

## Final verification

- `cargo fmt --check`
- Focused RED/GREEN tests for each core module and scheduler.
- Bounded `wenlan-core` module/focused regressions; do not use the verifier itself to recreate the thermal incident locally.
- `cargo test -p wenlan-server --lib`
- `cargo clippy -p wenlan-core -p wenlan-server --all-targets -- -D warnings`
- Independent correctness/concurrency review.
- Opus architecture/code review; the earlier explicitly requested Fable adversarial UX/thermal review remains supporting historical evidence, not a standing review requirement.
- Live daemon proof: no overlapping ambient request, cooldown visible in logs, backlog advances across turns, and stop/start CLI behavior remains unchanged.

Latest code evidence (2026-07-19/20):

- Five isolated source-bundle `52740615cfcc...` rows passed: Document 4.302
  seconds, Entity 5.528 seconds, committed Page-growth no-match 0.154 seconds
  with zero calls, Reconcile 1.833 seconds, and Citation 1.885 seconds; every
  thermal sample was `0→0`. These rows calibrate the policy but require a
  functional and timing evidence; they are not repeated after the later
  lifecycle/policy changes.
- The same-source persistent daemon loaded the model once, then completed
  Classification in 3.222 seconds and StructuredExtract in 2.331 seconds with
  a 600.07-second gap. It logged graceful shutdown and then crashed in the C
  Metal exit handler. The macOS report pins the fault to
  `std::process::exit → __cxa_finalize_ranges → ggml_metal_device_free`.
  A child-process RED returned 71 when a C exit handler ran; Unix `_exit` is
  GREEN. The binary group passes 20/20 and graceful shutdown passes 4/4.
- Supported-Mac calibration removes the fixed write-recency gate from ambient
  and automatic admission, while the existing Idle recap trigger keeps its
  own ten-minute batching condition. Inter-turn recovery is
  `max(120 seconds, 19 × elapsed)`. Scheduler tests pass 72/72 with the real
  profile ignored; the live harness passes 8/8 with its real profile ignored;
  profiler shell safety passes. The nonzero fail-safe blocks immediate reruns.
- Local `origin/main` resolves to
  `9cb6c6acb94c04cce82e359a1c6c4e8587668d64`; it is the active
  `MERGE_HEAD`. Live fetch failed SSH authentication, and the merge is not yet
  committed.
- Main-79/feature-83 collision lineages — 2/2. Migrations 80–85 — 6/6;
  legacy m78, m79, and main-72 upgrade tests — 1/1 each. Service-class
  FIFO/promotion/generic-import/migration group — 4/4.
- Current profile harness group — 3/3 with the gated real-model test ignored.
  Cargo metadata selected one exact `wenlan_server` test executable, and
  invoking that binary directly reproduced 3 passed/1 ignored. Shell
  syntax/order/exclusive-lock/failure-cooldown/provenance/Page-growth/jq-path
  regression passes.
- Behavioral-flag drift guard — RED on three test-only controls, then GREEN
  after explicit allowlist reasons (1/1).
- Deterministic one-slice bounds — Document 1/1, Entity 1/1, Reconcile 1/1, Citation 1/1; Page-growth no-match is terminal with zero inference — 1/1.
- The corrected read-only live aggregate uses one transaction and found 2,885
  memory heads, including 2,443 classification and 2,397 eventual Page-growth
  population projections plus 1,264 entity candidates without an authoritative
  link. It evaluates no memory content/title predicate, returns no raw
  identifiers, and therefore reports the classification population as a
  conservative title upper bound instead of claiming an exact title count.
- Target-Mac admission has both controls: earlier attempts rejected busy CPU
  before model load; the current preflight passed at 8.44%/7.33%, and the
  Page-growth run passed its pre/post-build windows at 8.31%/5.87% and
  4.49%/4.41%. The complete Page-growth no-match profile exited 0 with 0
  provider calls, durable progress, 190 ms slice wall time, and thermal 0→0.
  The isolated test process took 66.46 seconds, including a 5.033-second model
  boot; model RSS delta was 2,687,746,048 bytes and the end samples were
  429.40% process CPU / 57.39% system CPU. Its then-current measured-job
  cooldown was 600 seconds; that provisional policy is superseded. Evidence:
  `/private/tmp/wenlan-rb01-profile-20260720T040843Z-page-growth`.
- Foreground resource-policy RED then GREEN: high CPU/low memory admission and two-sample recovery 2/2; duration-aware cooldown 1/1; current scheduler filter 68/68 with one manual profile ignored.
- Relation vocabulary atomicity RED then GREEN: a forced ledger failure first left only the mutated canonical edge, then preserved both original relation rows after the transaction fix; full `kg_quality` regression group 18/18.
- RED then GREEN: generic import writes automatic-owned enrichment origin atomically — 1/1.
- RED then GREEN: automatic batch contains only ReDistill and its cursor never leaves the allowlist; full scheduler module — 56/56 in 5.15 seconds.
- ReDistill regression group — 6/6, including one-Page slicing, 64-source join/legacy-JSON caps, zero provider calls when oversized, retryable cursor reachability, and mutation-safe stale-page keyset traversal.
- Automatic maintenance module — 17/17. The StalePage path now shares the automatic 64-source cap and mutation-safe composite keyset cursor with ReDistill.
- Accepted-revision fixed-dependency convergence — 1/1.
- Everyday and synthesis hard-pin resolver groups — 5/5 each. Healthy external store route reports `pending` but performs zero inline inference; unconfigured/unavailable reports `paused`.
- RED then GREEN: Document provider panic pauses the exact claimed generation instead of leaking `in_progress`; stale-page success cannot skip a shifted row; zero-call reconcile watermark progress consumes the shared thermal turn without receiving empty-lane backoff.
- RED then GREEN: five-stage store-route lock regression identifies each `ServerState` guard retained across a DB await; all stages now release the guard before waiting, and panic/timeout hook cleanup is RAII-safe — 1/1.
- Current bounded integration gates: PageWrite 28/28; Page Growth 8/8;
  citation/backfill 27 passed with two GPU tests ignored; stale-citation CAS
  2/2; manual-edit HTTP gate 5/5; route convergence 14/14. A historical full
  `wenlan-server --lib` run remains evidence for the pre-`9cb6c6ac`
  integration, not a fresh current-tree claim.
- `wenlan` CLI — 83 passed, 1 ignored; `background off` preserves service registration and stops a registered-but-unreachable supervisor.
- `wenlan-types` paused response round-trip — 1/1.
- Current scheduler group: 72 passed with the one manual real-model profile
  ignored. Startup model admission/reservation is 2/2; registry working-set
  conversion in the daemon binary is 1/1. Schema-85 collision lineages are 2/2
  and now assert real Page-history and operation-receipt bytes survive.
- `cargo clippy -j 2 -p wenlan-core -p wenlan-server --all-targets -- -D warnings` — current review-fix tree passes with no issues.
- `cargo fmt --all -- --check` and `git diff --check` — passed.
- CI lifecycle assertions now keep the Linux unit and Windows task registered after `background off`; YAML parsing and extracted Linux shell syntax pass locally. Windows runner execution remains required CI.
- Lean Dev Tools/rust-analyzer at warning severity — no diagnostics in scheduler, refinery, DB, store route, distill, or importer.
- The full-workspace local test was intentionally interrupted when its verifier sustained roughly 535–570% CPU; no assertion failure had occurred. Required CI remains the complete-suite and Linux/Windows lifecycle execution gate.
- A standard Opus working-tree review found the startup SIGTERM registration race; it was reproduced RED and fixed GREEN. One additional `ReflectionDebouncer` observation is baseline dead code outside this feature diff, not a thermal behavior introduced by this branch. The final file-backed Opus gate returned `SHIP`: no blocking findings in the completed store/scheduler/lifecycle seam.

Representative persistent-daemon proof completed at
`20260721T055753Z-daemon`: one model load, Classification in 2.629 seconds,
StructuredExtract in 1.935 seconds, a 150.019-second completion-to-start gap,
32 nominal thermal samples, peak daemon RSS 3,462,561,792 bytes, durable
current-version receipts, graceful shutdown, and process exit 0.

The production policy keeps two 30-second CPU samples, a 20% CPU ceiling,
2 GiB/15% RAM floor plus startup working-set reserve, and `19 × elapsed`
long-turn recovery with a 120-second minimum. Ambient admission has no fixed
write-recency delay. On macOS, production additionally requires 60 seconds
without physical input and nominal public OS thermal state before model startup
or a new heavy turn; supported telemetry failure denies admission. The live
profiler/watchdog independently samples thermal state and RAM every 30 seconds.
A committed Page-growth terminal no-match with zero calls no longer consumes a
thermal turn; matches, provider calls, selected Document/Reconcile CPU work,
and panics remain charged.

The final current-source daemon proof is
`20260722T045828Z-daemon`: one model load, Classification in 2.651 seconds,
StructuredExtract in 1.503 seconds, exactly one call per turn, a 120.018-second
gap, 14/14 nominal thermal samples, peak RSS 3,474,391,040 bytes, durable
version-1 receipts, graceful shutdown, and exit 0. Production denied three
post-load samples above 20% before admitting work at 19.64% or below. The
preceding interrupted `20260722T044647Z-daemon` run is a positive control for
the new physical-input gate: it logged `ForegroundActive` repeatedly and began
no inference before operator termination.

Fable's final platform-glue review found that macOS-only enum variants would be
unconstructed in a non-macOS production build and that raw object
`rustc-link-arg` metadata was not a reliable downstream native-link contract.
The enum now carries a target-specific dead-code annotation, and the bridge is
packaged with cc's target-aware archiver/ranlib commands into a deterministic
static library published through normal Cargo link metadata. Fresh
`wenlan-server` and downstream `wenlan-mcp` `real_router` builds both link
without warnings; scheduler tests pass 85/85 with two manual profiles ignored,
and the cheap daemon harness passes 8/8 with its real-model proof ignored.
Fable's follow-up verdict is `sound` with no confirmed correctness bugs. The
macOS link smoke accepts fail-closed `Unavailable` on CI hosts without a usable
WindowServer; the target-Mac live artifact separately proves real telemetry.
Surfacing permanent host-telemetry deferral in `/api/status` remains a
non-blocking cross-surface observability follow-up; current daemon logs expose
the transition at the default log level.
