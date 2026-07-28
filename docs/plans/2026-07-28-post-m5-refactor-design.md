# Post-M5 refactor design and execution plan

Date: 2026-07-28
Baseline: `origin/main@e4790ce857056050a90a4adeef391375e8ce5f19`
Status: **Fable gate 1 approved; R0 implemented and verified**

## Authority and change control

This file is the single source of truth for the post-M5 refactor. Chat summaries,
PR descriptions, reviewer messages, and generated inventories may point here but
must not become independent copies of the design.

Any change to a locked decision, priority, protected surface, PR boundary, or
review gate must update this file first and add an entry to the decision-change
log at the end. A material design change invalidates the current Fable design
verdict and requires a new one before implementation continues.

Snapshot metrics are evidence about the named baseline, not evergreen truths.
Rebaseline them after any merge that touches a target surface. Executable
inventories must fail on drift; prose counts are never the enforcement
mechanism.

## Goal

Reduce the amount of unrelated code an agent must understand and edit for a
bounded Wenlan change, while preserving:

- one daemon-owned database and one `MemoryDB` facade;
- existing SQL, transaction, async-lock, API, and wire behavior;
- the canonical write and enrichment paths;
- M5 claim identity, truth-state, edge-rebuild, and reader-cutover contracts;
- executable completeness checks rather than review promises.

This is a structure and change-safety refactor. It is not permission to redesign
the database, introduce repository traits, change public APIs, rewrite SQL, or
clean up adjacent code.

## Baseline evidence

Measured on the baseline above:

| Surface | Current evidence |
|---|---:|
| all tracked Rust | 367,929 lines |
| `crates/wenlan-core/src/db.rs` | 94,744 lines, 25.7% of tracked Rust |
| main `db.rs` test module | begins at line 49,059; about 45,686 lines |
| `MemoryDB::run_migrations` | 5,017 lines |
| `db/claim_identity.rs` | 946 production lines |
| `db/edges_rebuild.rs` | 556 production lines |
| M5 external DB tests | 2,219 lines across two files |
| server `.route(` text occurrences in Rust source | 156; size signal, not the 162-route census |
| `memory_routes.rs` public handlers | 95 |
| MCP tool declarations | 29 |
| public request/response wire declarations | 129 matching `^pub (struct|enum|type) ` in `requests.rs` + `responses.rs` |
| root + core `AGENTS.md` | 71,679 bytes |

M5 established the preferred DB seam: child modules contain inherent
`impl MemoryDB` blocks, tests live in sibling files, and `db.rs` retains module
wiring plus migration orchestration. M5 added 1,502 lines of production DB code
and 2,219 lines of DB tests outside `db.rs`, while `db.rs` itself grew by only
131 net lines.

Rust LSP on this baseline can enumerate `db.rs` document symbols and resolve a
child-module `MemoryDB` definition back to `db.rs`. A whole-workspace
`MemoryDB` reference query still times out. File decomposition has therefore
improved local navigation but has not removed the supernode.

Before R0, the M5 internal-reader generator reported:

- 191 total internal page-prose readers;
- 22 exposure paths;
- depth partition `54 / 50 / 87`.

Its committed inventory still says `190 / 22`. The extra live row is
`wenlan-server::memory_routes::handle_create_page`, which reads the persisted
page after creation. The generator self-test passes, so this is an artifact
freshness gap, not a self-test failure. Until set equality is executable, the
inventory is evidence but not a refactor gate.

R0 then found that the passing self-test was insufficient: the brace matcher
counted braces inside Rust strings/comments, so `run_migrations` swallowed
later sibling methods despite staying below `MAX_FN_LINES`. An LSP
document-symbol comparison exposed the boundary mismatch. After adding a
structural lexer and positive controls, the same baseline resolves to `192`
readers with partition `55 / 50 / 87`; the second previously hidden row is
`MemoryDB::migrate_89_page_kind_fold`.

`crate::db::tests::test_db` and DB test hooks are used widely outside `db.rs`.
Moving the main test module must preserve the exact
`crate::db::tests::*` module path and visibility.

## Locked design decisions

### D1 — one facade, child inherent impl modules

Keep one public `MemoryDB` facade. Move cohesive implementations into
`db/<domain>.rs` as inherent `impl MemoryDB` blocks, following
`claim_identity.rs`, `edges_rebuild.rs`, `page_map.rs`, and the existing scoped
modules.

Do not create a repository trait per table or domain. Trait extraction is
allowed only when a real second implementation or an existing boundary requires
one.

### D2 — movement and behavior never share a PR

A movement PR may change paths, module declarations, and visibility only as
required to preserve the same symbols. It may not rewrite SQL, change query
shape, alter transaction boundaries, rename public methods, or clean up nearby
code.

A later behavior PR must start from the moved, verified baseline and carry its
own RED test.

### D3 — tests leave `db.rs` before production domains

First move the main `pub(crate) mod tests` body out of `db.rs` while preserving:

- the module path `crate::db::tests`;
- `test_db`, shared locks, and test-hook visibility;
- ignored-test names used by runbooks;
- the exact discovered test inventory.

The first move may produce one large external test file. Domain-by-domain test
splitting is a later mechanical series; combining both transformations would
make missing tests harder to detect.

The external file must remain invisible to the reader census. Either name it
with the existing `_test.rs` / `_tests.rs` suffix that
`scripts/m5-reader-sweep.py` excludes, or extend the generator first so a
`#[cfg(test)]` module declared through `#[path]` is excluded and mutation-test
that exact shape. A natural but unsafe `db/tests.rs` move is forbidden while
the current filename filter is authoritative.

### D4 — connection encapsulation is the last DB step

Do not make `MemoryDB::conn` private at the start. First introduce or move the
domain methods that replace direct access, one caller group at a time.

During the refactor:

- no new direct `.conn.lock().await` outside the DB subtree;
- the existing external set must decrease monotonically;
- final target: no direct connection access outside the DB implementation
  boundary, except explicitly justified test support.

R0 makes this executable with a tracked external-access allowlist keyed by
source file and occurrence count. The allowlist may shrink but a new file or an
increased count fails the same `drift_guard` library-test layer used by the
other source contracts. Existing test-support entries remain explicit rather
than disappearing through a broad test exemption.

### D5 — M5 surfaces are protected until their cutover sequence clears

Until M5 PR-B/PR-C are merged, or the M5 owner explicitly coordinates the
change, do not structurally move:

- `db/claim_identity.rs`;
- `db/edges_rebuild.rs`;
- migrations 98 and 99;
- page truth-state readers or adapters;
- human-edit-delta and support-edge write paths;
- reader addresses consumed by the M5 manifest;
- the canonical page-write seam needed by the exact-base save path.

While PR-B/PR-C remain active, only R0 and R1 may proceed:

- R0 strengthens the interim inventory under the ownership handoff defined in
  its scope;
- R1 is allowed only because it moves the trailing test module, preserves every
  production reader address, and obeys D3's generator-exclusion constraint.

R2 through R7 remain blocked until PR-B/PR-C merge or the M5 owner explicitly
coordinates a narrower exception. R8 remains last by design.

### D6 — executable change maps precede broad movement

Any inventory used to assert completeness must be regenerated from the current
tree and compared by set equality in a test or CI-visible verifier. Line-number
lists and hand-maintained prose tables are not gates.

For reader movement, the guard must cover the symbol/address set and its
partition. For test movement, it must cover test names before and after. For
public APIs, it must cover exported symbol or typed-contract shape.

### D7 — one writer per hot file

Only one worker edits a hot file or module family in a PR. Parallel agents may
audit, review, measure, or prepare a non-overlapping later slice; they do not
concurrently rewrite `db.rs`, `post_write.rs`, `memory_routes.rs`, or
`scheduler.rs`.

## Impact order

### 1. `db.rs` modularization

Highest impact because it is simultaneously the largest file, the most
frequently touched normalized path, the DB facade, the migration registry, and
the shared test-fixture home.

Agent effect:

- smaller local symbol and context set;
- fewer unrelated SQL blocks in a normal edit;
- more reliable document-symbol and definition navigation;
- narrower diffs and reviewer blast radius.

### 2. HTTP, wire, and MCP vertical slices

Split server handlers and route wiring by product domain while keeping
`wenlan-types` as the typed boundary and MCP typed deserialization intact.

Agent effect:

- one feature change has a discoverable route/handler/type/tool path;
- fewer cross-file searches to find the complete public contract;
- contract tests can bind a vertical slice instead of a monolithic router.

This waits behind the M5 reader-adapter work because both touch the same
surfaces; it is part of the explicit R2-through-R7 block in D5.

### 3. `post_write` internal phases

Keep the canonical page-write facade. Separate validation, planning, storage,
and post-commit work behind it without creating another writer.

Agent effect:

- mutations continue to have one obvious gate;
- a validator or planner change does not require loading all write-path tests;
- agents are less likely to bypass CAS, evidence, or exact-base rules.

### 4. daemon startup and scheduler orchestration

Extract named startup phases and scheduler-lane registration while preserving
ordering and lifecycle semantics.

Agent effect:

- startup and one ambient lane become independently navigable and testable;
- adding a job no longer requires editing one large control-flow body;
- lifecycle regressions remain visible at the registry/orchestrator boundary.

### 5. agent instruction surface

Move drift-prone facts into executable checks or narrowly loaded references.
Keep always-loaded files focused on rules that change agent behavior.

Agent effect:

- less fixed context before core work;
- higher salience for crate-boundary, SQL, async-lock, and verification rules;
- fewer stale prose claims competing with live code.

### 6. repair, eval, and LLM long functions

Refactor only when the corresponding domain is active. Their complexity is real
but more local, so they do not outrank the cross-cutting DB/API boundaries.

## PR sequence and gates

### R0 — make the reader census a real gate

Scope:

- explain and reconcile `191` generated vs `190` committed;
- regenerate the inventory from the merge baseline;
- add a `drift_guard` library test, executed by the existing
  `cargo test --workspace --lib` PR path, that invokes the generator's check
  mode and enforces current-tree set equality plus the depth/exposure
  partition;
- add the external direct-connection ratchet required by D4;
- correct the stale `run_migrations` size comment in the generator while this
  script is already in scope;
- keep production behavior unchanged.

Ownership handoff: R0 owns the interim name-resolved
`m5-reader-sweep.py`/inventory gate. M5 PR-B owns the LSP-resolved successor
already promised by the M5 manifest. Only one owner edits these artifacts at a
time; PR-B must replace or extend the interim gate rather than install a second
competing census.

Required evidence:

- generator self-test;
- generated/current set equality;
- the gate is exercised through `cargo test --workspace --lib` on every
  applicable PR, not only by a manual Python invocation;
- deliberate add/remove reader mutations make the census guard fail;
- deliberate new/increased external `.conn.lock()` mutations make the
  connection ratchet fail;
- Opus reviews both predicates and their positive controls.

Execution record (2026-07-28):

- RED 1: the first Rust guard failed because the old script ignored `--check`
  and emitted no success receipt;
- RED 2: the connection-ratchet positive control failed while its helper
  returned an empty violation set;
- RED 3: an LSP document-symbol comparison showed that the first exact reader
  gate still let `run_migrations` swallow sibling methods; a synthetic
  string/raw-string/comment fixture reproduced the truncation;
- GREEN: the generated inventory now has `192` rows, partition
  `55 / 50 / 87`, with `22` exposure paths, and is exact-compared through
  `drift_guard`;
- GREEN: direct `.conn.lock().await` access outside `db.rs`/`db/**` is frozen
  at `333` occurrences across `56` tracked files, including explicit test
  support per D4, with new files and per-file increases rejected;
- Opus re-reviewed the structural lexer and both ratchets and found the design
  sound. Its Windows `python3` concern does not apply to the current CI
  contract because workspace lib tests run on Linux/macOS and are explicitly
  skipped on Windows; its proposed broad test exclusion contradicts D4's
  locked explicit-test-support rule.

### R1 — externalize the main `db.rs` test module

Scope:

- move only the main test-module body;
- use a filename that satisfies D3's census-exclusion rule;
- preserve `crate::db::tests::*`;
- preserve all test names and ignored-test entrypoints;
- no production movement.

Required evidence:

- before/after test-name inventory equality;
- focused `wenlan-core` library tests;
- workspace library tests;
- formatting and Clippy at the repository-standard layer;
- Opus diff review focused on lost fixtures, visibility, and cfg boundaries.

### R2 — migration dispatcher and historical migration modules

Scope:

- leave `run_migrations` as an ordered dispatcher;
- move immutable historical migration bodies in bounded version/domain groups;
- preserve ordering, backup timing, transaction boundaries, and
  `user_version` stamping;
- do not move migrations 98/99 until D5 clears.

Each PR moves one group only and includes upgrade/idempotence tests relevant to
that group.

### R3 — DB domain modules

Select the next domain by evidence:

1. outside the M5 protected surface;
2. cohesive public method set;
3. existing focused tests;
4. low cross-domain private-helper dependence.

One domain per PR. Keep method names and `MemoryDB` call sites unchanged.

### R4 — close direct connection access

Replace remaining external connection locks with bounded domain APIs. Make the
field private only after the caller census reaches its accepted floor.

### R5 — server vertical slices

Move route registration and handlers domain by domain. Preserve route identity,
typed request/response contracts, and `TrackedRouter` classification.

### R6 — `post_write` phase decomposition

Begin only after the M5 exact-base and truth-state reader/write paths have
settled. Preserve the one canonical facade.

### R7 — daemon startup and scheduler lanes

Separate orchestration from phase/lane implementations without changing startup
or scheduling order.

### R8 — instruction and verification-surface cleanup

Run after the code boundaries have stabilized so the documentation describes
the resulting architecture rather than predicting it.

## Review and team protocol

### Fable gate 1 — frozen design

Timing: now, after this document is written and before R0 implementation.

Fable reviews system shape only:

- whether the impact ordering follows the evidence;
- whether D5 contains the M5 PR-B/PR-C blast radius;
- whether R0 has real teeth and a positive control;
- whether the facade/child-module strategy avoids needless abstraction;
- whether the sequence separates movement from behavior;
- whether the end state materially improves agent code-writing conditions.

Expected verdict: `APPROVE`, `APPROVE-WITH-FIXES`, or `BLOCK`. All fixes land in
this document. Implementation starts only after the design verdict is clear.

Gate result, 2026-07-28: **APPROVE-WITH-FIXES → APPROVE after five required
document corrections.** Fable required an explicit D5 concurrency rule, an R1
reader-census exclusion rule, a named CI execution layer for R0, an R0/PR-B
artifact-ownership handoff, and an executable direct-connection ratchet. Those
corrections are now incorporated above. Fable explicitly classified them as
clarifications within the frozen intent and said no second full review was
needed before R0.

### Intermediate PRs — Opus, not Fable

Every PR receives a focused Opus opinion or diff review against:

- the exact PR scope;
- the locked decisions above;
- the relevant SQL/API/transaction/test invariant;
- verifier evidence and positive controls.

Opus may identify a design-level contradiction. If resolving it would change a
locked decision or PR sequence, work stops and returns to Fable gate 1. Otherwise
Fable is not used for intermediate PRs.

### Fable gate 2 — final whole-refactor review

Timing: after the final planned refactor PR is implemented and all aggregate
gates pass, before declaring the refactor complete or merging the final closure
PR.

Fable compares the delivered system with this frozen design:

- original goals vs actual boundaries;
- cumulative behavior/API/SQL drift;
- whether temporary compatibility seams were removed or explicitly retained;
- whether agent navigation and bounded-change conditions materially improved;
- whether any PR-local compromise undermined the whole.

This is a system acceptance review, not another line-by-line code review.

### Team shape

- **Root/architect:** owns this document, PR boundaries, sequencing, and stop
  decisions.
- **One implementation worker:** writes the current bounded PR.
- **Opus reviewer:** provides the intermediate independent opinion/diff review.
- **Auditor:** independently runs inventories, positive controls, LSP/AST
  comparisons, and repository gates.

Reviewers and auditors do not edit the implementation while reviewing it.

## Aggregate completion criteria

The refactor is complete only when:

- R0's current-tree inventories pass by set equality;
- `db.rs` contains no giant inline test module, no unrelated domain
  implementation bodies, and only ordered dispatch rather than inline bodies
  for migrations covered by R2;
- movement PRs have no intentional SQL, API, transaction, or behavior change;
- no new external direct DB connection access was introduced and the accepted
  end-state boundary is enforced;
- M5 claim/truth/reader contracts remain green;
- route, wire, and MCP typed contracts remain complete;
- repository-standard formatting, Clippy, workspace library tests, and any
  affected higher-layer tests pass after the final edit;
- Fable gate 2 approves the whole result.

Agent impact is recorded before and after on the same machine with the same
tool versions:

- document-symbol enumeration for `db.rs`;
- child-module `MemoryDB` definition navigation;
- whole-workspace `MemoryDB` reference lookup, including timeout/result and
  wall time;
- the number of source files and functions an agent must inspect for one
  representative bounded DB change and one representative vertical API change.

These probes are supporting evidence, not correctness gates, because
language-server latency also depends on tooling and machine state. The
structural facade definition and executable contracts above remain the gates.

## Decision-change log

### 2026-07-28 — initial freeze candidate

- Baseline set to M5 PR-A merge `e4790ce8`.
- Reader-inventory freshness promoted to R0 after live output reported
  `191 / 22` while the committed inventory reported `190 / 22`.
- M5 PR-B/PR-C surfaces protected before broad movement.
- Review policy locked: Fable at frozen design and final system acceptance only;
  Opus for intermediate PR opinions and diff reviews.

### 2026-07-28 — Fable gate 1 corrections

- Clarified that R0 and R1 may proceed during active M5 PR-B/PR-C work while
  R2 through R7 remain blocked.
- Bound the R1 test-file move to the reader generator's exclusion contract.
- Named `drift_guard` plus `cargo test --workspace --lib` as R0's PR execution
  layer.
- Assigned interim name-resolved census ownership to R0 and the LSP-resolved
  successor to M5 PR-B, with one writer at a time.
- Added an executable, shrink-only direct-connection access ratchet to R0.
- Relabeled two snapshot metrics with their exact textual predicates and added
  concrete before/after agent-impact probes.
