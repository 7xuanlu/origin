# Post-M5 refactor design and execution plan

Date: 2026-07-28
Baseline: `origin/main@e4790ce857056050a90a4adeef391375e8ce5f19`
Status: **R4 complete; R5 in progress; registration slice 6 APPROVED**

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

Execution record, 2026-07-28:

- RED: the new `drift_guard::db_main_tests_live_outside_db_rs` contract failed
  on the inline body, missing external declaration, and missing external file;
  its positive control also rejects the census-visible `db/tests.rs` shape.
- GREEN: the body now lives at `db/main_tests.rs`, while `db.rs` declares it
  through `#[path]`; `crate::db::tests::*`, shared hooks, and `test_db` retain
  their original module path and visibility.
- GREEN: compiler-discovered before/after inventories are byte-identical:
  `3,232` total `wenlan-core` library tests, `950` under `db::tests::*`,
  `33` ignored tests overall, and `6` ignored `db::tests::*` entrypoints.
- GREEN: focused `db::tests::*` execution passed `944`, ignored `6`, failed
  `0`; rust-analyzer reported no errors in the three changed Rust files and an
  external `crate::db::tests::test_db` definition lookup resolved to the new
  file.
- GREEN: the reader-census check stayed at `192` rows, partition
  `55 / 50 / 87`, with `22` exposure paths.
- NOTE: rustfmt's module-level outdent collapsed `93` physical lines after the
  byte-exact body extraction; compiler-discovered test-name equality is the
  semantic movement proof rather than raw line-count equality.
- REVIEW: Opus/xhigh returned `APPROVE` with no material findings and confirmed
  preserved names, ignored entrypoints, cfg scope, census exclusion, and zero
  production movement. A permanent `950 / 6` count lock was not added: the
  before/after compiler inventory is the R1 movement gate, while a static count
  would also reject legitimate future test additions.
- GREEN: repository-standard `cargo fmt --all --check` and workspace
  all-targets Clippy with `-D warnings` passed. Final
  `cargo test --workspace --lib` passed: CLI `31 / 31`; core `3,199` passed,
  `33` ignored; MCP `177 / 177`; server `311` passed, `2` ignored; types
  `180 / 180`.

### R2 — migration dispatcher and historical migration modules

Scope:

- leave `run_migrations` as an ordered dispatcher;
- move immutable historical migration bodies in bounded version/domain groups;
- preserve ordering, backup timing, transaction boundaries, and
  `user_version` stamping;
- do not move migrations 98/99 until D5 clears.

Each PR moves one group only and includes upgrade/idempotence tests relevant to
that group.

Execution record, 2026-07-28 — migrations 4 through 9:

- Baseline refreshed to `origin/main` `07afba7d`, which includes PR-B
  `f29c2c54`, PR-C `3932e3d5`, release `544965a1`, and the unrelated
  test-config isolation fix #411; the D5 dependency is therefore clear for
  this bounded group. The separate PR-D ceremony remains the sole owner of the
  cutover fence, generation advance, truth manifest changes, and the
  page-write/cutover files listed in its handoff.
- RED 1: the structural guard rejected the missing child module, six missing
  ordered dispatcher calls, six missing `user_version` stamps, and the SQL
  bodies still embedded in `run_migrations`; its positive control separately
  rejects inline SQL and M5 scope creep.
- RED 2: the direct replay test failed to compile on exactly the six migration
  methods that did not yet exist.
- GREEN: migrations 4 through 9 now live in
  `db/migrations_v004_v009.rs`; `run_migrations` retains the six ordered
  `if version < N` guards and delegates one-for-one.
- GREEN: regeneration from the pre-move `db.rs`, followed by rustfmt, was
  byte-identical to the current dispatcher and child module
  (`sha256=5d584ce2aab38f9027709cf5b2eaa13984298a56b6d8a0b3e8fa3b14e583357d`).
- GREEN: direct method replay ran the group twice, asserted every
  `user_version` stamp from 4 through 9, and retained the exact eight tables
  and eight indexes owned by the group.
- GREEN: `db.rs` fell from `49,160` to `48,877` lines; the bounded child module
  is `323` lines after rustfmt. The M5 reader set stayed `191` rows with partition
  `55 / 50 / 86` and `22` exposure paths after regenerating only its
  line-address projection.
- GREEN: rust-analyzer reported zero errors in all four changed Rust files and
  definition navigation from the migration-4 dispatcher call resolved to the
  child module. The first definition request timed out while caches primed;
  the immediate warm retry resolved in six seconds.
- GREEN: PR-C's three new, explicit test-support connection accesses were
  reconciled into R0's shrink-only baseline. The current floor is `336`
  occurrences across `59` tracked files; broad test exclusions remain
  forbidden.
- GREEN: on refreshed main `07afba7d`, repository-standard
  `cargo fmt --all -- --check`, workspace all-targets Clippy with
  `-D warnings`, and one uninterrupted `cargo test --workspace --lib` passed:
  CLI `32 / 32`; core `3,299` passed with `33` ignored; MCP `178 / 178`;
  server `339` passed with `2` ignored; types `183 / 183`.
- NOTE: an earlier pre-#411 workspace run passed CLI, core, and MCP before one
  server fixture hit a non-deterministic `online_backup integrity` SQLite
  API-misuse error. The unchanged test passed alone (`1 / 1`) and the complete
  server suite passed at normal parallelism before the clean current-main run
  above. The interruption is retained here rather than erased by the retry.
- REVIEW, 2026-07-29: Opus/xhigh returned **APPROVE / ship-ready** with no
  blocking findings. It confirmed the six methods are verbatim moves with
  unchanged SQL, transaction order, version stamps, errors, and logs; the
  visibility, structural guard, positive control, and replay test are
  appropriately bounded. It classified the pre-existing lock gap and
  no-rollback-on-error migration pattern as follow-up concerns rather than R2
  regressions. Fable was intentionally not used for this intermediate PR.

### R3 — DB domain modules

Select the next domain by evidence:

1. outside the M5 protected surface;
2. cohesive public method set;
3. existing focused tests;
4. low cross-domain private-helper dependence.

One domain per PR. Keep method names and `MemoryDB` call sites unchanged.

#### R3-1 selection — source-sync CRUD method boundary

Selected on 2026-07-29 at `29803914`:

- Move only `upsert_sync_state`, `get_sync_state`,
  `list_sync_state_paths`, `delete_sync_state`, and
  `delete_all_sync_state` into `db/source_sync.rs`.
- Keep `FileSyncState` at the public `db` facade and keep every caller
  unchanged.
- The five methods are one contiguous CRUD block, use only `self.conn` plus
  `FileSyncState`, and have a focused `test_source_sync_state_crud` test.
  LSP references also reach the real source routes, document-enrichment path,
  lint endpoint tests, and folder-ingest e2e.
- This is a method boundary, not exclusive ownership of the
  `source_sync_state` table. Two transaction-scoped statements intentionally
  remain in `db.rs`: source deletion removes the receipt alongside every other
  owned dependency, and document-enrichment completion writes its receipt in
  the same transaction as the queue transition. Calling the public CRUD
  methods from either site would re-lock `MemoryDB::conn`; moving those
  statements alone would break the transaction boundary. Any redesign of
  those transactions belongs to a separate behavior PR, not this movement.
- The five method names have no M5 reader-manifest or truth-manifest match.
  `import_state` was rejected as the first domain because
  `list_pending_imports` is an M5 page-bearing reader.
- rust-analyzer resolved workspace symbols and method references. Its document
  outline request for the 48k-line `db.rs` failed in the client, so the bounded
  block census used literal discovery followed by LSP references and direct
  source inspection.
- A structural test must require the module declaration and file, require
  exactly those five public async methods without imposing meaningless source
  order, reject copies left in `db.rs`, and carry a positive control that
  proves the guard catches missing files or methods, duplicate declarations or
  definitions, inline bodies, and any extra visible method including
  `pub fn`, `pub(crate)`, and `pub(super)`.

Execution evidence:

- RED: the structural test failed on the pre-move tree for the missing module,
  missing declaration, and all five inline method bodies. GREEN: the boundary
  test and its positive control each pass `1 / 1`.
- The normalized pre-move block and the new module implementation have the
  same SHA-256,
  `db13a794539bceb2eb408073a5eaa78fa5e3078e43ed93bf1c14105217993197`.
  SQL, error text, ordering, visibility, and callers are unchanged.
- Focused behavior passed: source-sync CRUD `1 / 1`; sync-receipt failure
  atomicity `1 / 1`; folder-ingest e2e `2 / 2` with `1` fixture generator
  ignored; source-route tests `13 / 13`.
- The direct-connection ratchet passes. The M5 inventory remains exactly
  `191` rows, depth `55 / 50 / 86`, exposure `22`; only the generated
  `db.rs` line addresses changed by the new module declaration.
- rust-analyzer reports no diagnostics in `db.rs` or `db/source_sync.rs`;
  definition navigation from a server route resolves to the new module, and
  references span server routes, lint tests, DB tests, and folder-ingest e2e.
- `db.rs` fell from `48,877` to `48,758` lines; the bounded child module is
  `130` lines. Repository format and diff checks pass.
- PR-level verification passes: all-target Clippy with `-D warnings` for
  `wenlan-core` plus `wenlan-server`, and one uninterrupted
  `cargo test --workspace --lib`: CLI `32 / 32`; core `3,301` passed with
  `33` ignored; MCP `178 / 178`; server `339` passed with `2` ignored; types
  `183 / 183`.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **FIX-FIRST**. It found the
  extraction faithful but required the boundary to stop implying exclusive
  table ownership, the guard to detect visible methods beyond only
  `pub async fn`, and its positive control to cover the rules it claimed.
  The section now names the two intentionally retained transactional
  statements and why moving them would change locking or atomicity. The guard
  compares an unordered exact visible-method set, while its positive control
  exercises missing file/method, duplicate module/implementation/method,
  inline body, and unexpected `pub`, `pub(crate)`, and `pub(super)` methods.
  The untracked child-file observation is handled by exact staging at commit;
  temporary working-tree tracking state is not a source-layout invariant.
- REVIEW 2, 2026-07-29: after those changes, Opus/xhigh returned
  **APPROVE**, with exact staging of the child file as the commit-time hard
  gate. It independently confirmed the faithful five-method extraction,
  generated inventory shift, direct-connection ratchet, and non-vacuous
  positive control. Cross-reference comments now mark both intentionally
  retained transaction-local table writers. Its remaining parser-grammar and
  one-implementation observations are non-blocking for this deliberately
  bounded contract: one domain module owns exactly the intended visible async
  CRUD methods, while transactional inline statements remain explicit.

#### R3-2 selection — onboarding-milestone CRUD boundary

Selected on 2026-07-29 at `4e308a82`:

- Move only `record_milestone`, `list_milestones`,
  `acknowledge_milestone`, `increment_milestone_shown_count`, and
  `reset_onboarding_milestones` into `db/onboarding_milestones.rs`.
- Keep the `MilestoneId` and `MilestoneRecord` public types in
  `crate::onboarding`, and keep every caller and method signature unchanged.
- Stop before the adjacent `MilestoneEvaluator` count helpers. Those queries
  span page, memory, and graph domains; `oldest_active_page` also appears in
  the frozen M5 reader inventory. Moving them with the milestone CRUD block
  would widen this movement-only slice.
- The three HTTP routes are explicitly `page_bearing: No`,
  `TruthClass::NotApplicable`, with `no prose fields`; this extraction changes
  no M5 adapter, writer permit, or cutover surface.
- Extend the generic R3 structural guard with this module's exact unordered
  five-method set. Reuse the guard's existing positive control rather than
  creating a weaker domain-specific duplicate.

Execution evidence:

- RED: the new structural test failed on the pre-move tree for the missing
  module, missing declaration, missing child definitions, and all five inline
  bodies. GREEN: the domain boundary and generic positive control each pass
  `1 / 1`.
- The normalized pre-move block and child implementation are byte-identical
  with SHA-256
  `02e494222031da817128166fdff20d1ec795c1a34d536594c6c802a390fe9bf9`.
  SQL, log and error text, ordering, visibility, and callers are unchanged.
- Focused behavior passes: milestone storage plus schema `7 / 7`; onboarding
  evaluator and wire-id behavior `7 / 7`; direct-connection ratchet `1 / 1`.
- rust-analyzer reports no diagnostics in the child module and no errors in
  `db.rs`; navigation from the evaluator resolves to the child module, and
  references span core onboarding, DB tests, and server onboarding routes.
- `db.rs` fell from `48,758` to `48,560` lines; the bounded child module is
  `203` lines. The generated M5 inventory remains exactly `191` rows; only
  line addresses move.
- PR-level verification passes: all-target Clippy with `-D warnings` for
  `wenlan-core` plus `wenlan-server`, and one uninterrupted
  `cargo test --workspace --lib --quiet`: CLI `32 / 32`; core `3,302` passed
  with `33` ignored; MCP `178 / 178`; server `339` passed with `2` ignored;
  types `183 / 183`.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **APPROVE**. It independently
  confirmed the five methods are byte-identical, no method body remains in
  `db.rs`, removing its `FromStr` import is safe because the remaining
  `Vendor` and `ImportStage` parsers are inherent methods, and five sampled
  M5 inventory addresses shifted by exactly `-198`. It also confirmed
  milestone SQL cannot enter the reader inventory because it never reads
  `pages`. Exact staging of the new child file is the only commit-time hard
  gate; the generic guard's narrow literal matching and future public test
  helper restriction remain accepted low-risk constraints.

### R4 — close direct connection access

Replace remaining external connection locks with bounded domain APIs. Make the
field private only after the caller census reaches its accepted floor.

#### R4 execution ledger

The current shrink-only guard is a migration ratchet, not the end-state
contract. The corrected brace-aware census at R4 entry is 336 literal
`.conn.lock().await` occurrences across 59 external files: 47 production
occurrences in 15 files and 289 test-scoped occurrences in 50 files. One
additional production escape is not counted by that regex:
`CommunityGroupingLeaseCleanup` retains an `Arc<Mutex<Connection>>` and locks
it from `Drop`. R4-6 has now closed that retained escape behind an opaque
DB-owned cleanup type. R4 closes the remaining boundary in bounded,
independently reviewed slices:

1. Move the existing Space-context `MemoryDB` implementation into a DB child
   module without changing its public methods, SQL, transaction boundaries, or
   public type paths. This removes the six raw-lock occurrences from
   `crate::space_context` without inventing another API.
2. Replace callers that already have an adequate domain method, then add narrow
   typed read methods for the remaining point reads, maintenance scans,
   knowledge-quality scans, derived-state sweep, community-grouping lease and
   cursor work, and eval integrity reads. Code is moved under `db/**` only when
   it is genuinely a `MemoryDB` implementation; moving orchestration merely to
   disappear from the matcher is forbidden.
3. Close the repair and post-write CAS/recovery tail as a sequence of
   independently reviewed high-risk sub-slices, not one combined move. The
   remaining operations have three different atomicity shapes: DB-only
   `BEGIN IMMEDIATE` transactions; projection-lock-before-DB transactions; and
   DB-lock-before-projection compensation flows. Each named operation must
   preserve its own existing lock acquisition order, lock lifetime, receipt
   checks, commit/rollback point, filesystem rollback, journal publication,
   and hook position. A generic `with_conn`, renamed connection accessor,
   generic SQL/transaction facade, raw-handle lease, or public transaction
   escape hatch is not an acceptable substitute.
4. Replace the remaining test fixture access with one `#[cfg(test)]` DB-owned
   support seam and an exact, location-aware manifest. Then make `conn` private
   in every build. A broad filename or `#[cfg(test)]` exemption is forbidden
   because production and test code coexist in several source files.

The test-support choice is locked to the fourth approach above: both `conn` and
the alternate `_db.connect()` capability remain private in every build, and the
existing test-only callers move mechanically to the named helper. Conditional
`pub(crate)` field visibility was rejected because `cargo test` would expose
the raw fields to every production module compiled in that build; the AST
manifest would then be policy rather than a type-system boundary. Opus
independently preferred the always-private field for the same reason. Its
suggestion that the community cleanup could remain an audited exception is
rejected: the zero-production-capability contract includes that retained
`MemoryDB` handle.

Here “raw capability” means a connection extracted from, cloned from, retained
from, or generically yielded by `MemoryDB`, including its primary mutex and
alternate database handle. A separately created private observer connection
behind a narrow domain API, such as `LintFreshnessClock`, is not a `MemoryDB`
capability escape and is outside R4; banning every internal use of
`libsql::Connection` would conflate DB implementation with DB boundary.

The current repair families also contain a pre-existing cross-family lock-order
inversion: rename takes projection-session then DB, while regenerate and stale
projection paths take DB then projection. Per-manifest artifact locking does
not globally serialize different manifests. These movement slices preserve and
test each current order without endorsing the inversion or silently attempting
to repair it. Any topology change requires a separate behavior/concurrency
design with deadlock and crash-recovery controls; changing only one family
inside R4 is forbidden.

Every slice removes its old allowlist entries immediately, keeps the external
set monotonically decreasing, runs the M5 inventories plus affected behavior
tests, receives an independent Sol review by default, and commits before the
next slice. Opus is reserved for an exceptional judgment escalation rather
than every intermediate slice. R4 is complete only when production raw
connection access is zero, retained or renamed raw handles are also rejected,
the named test-support set is exact, and a RED mutation control proves the
final guard catches a production bypass.

#### R4-1 — Space-context DB implementation

- RED: removing the six-occurrence `space_context.rs` allowance from the
  shrink-only ratchet failed with exactly
  `crates/wenlan-core/src/space_context.rs: 6 new direct .conn.lock() accesses`.
- Move the complete existing `impl MemoryDB`, legacy watermark constant, and
  private TOML helper into `db/space_context.rs`. Keep
  `ResolvedWriteSpace`, `LegacyDefaultImport`, their public paths, and the
  existing tests in `crate::space_context`.
- The normalized old and new moved blocks are byte-identical with SHA-256
  `2c695b82ad063355ee9cd14c67eaaecec79666932ddd64f42841edd5233897ae`.
  Public types and tests are also byte-identical. SQL, error/log text,
  signatures, locking, transactions, ordering, and callers are unchanged.
- GREEN: the direct-access ratchet passes `1 / 1`; Space-context behavior
  passes `4 / 4`; rust-analyzer reports no diagnostics in either Space module
  and no errors in `db.rs`.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only `db.rs` line addresses shift by
  `+1` for the new module declaration. No truth adapter, manifest row, writer
  permit, or cutover generation changes.
- PR-level verification passes: all-target Clippy with `-D warnings` for
  `wenlan-core` plus `wenlan-server`, and one uninterrupted
  `cargo test --workspace --lib --quiet`: CLI `32 / 32`; core `3,302` passed
  with `33` ignored; MCP `178 / 178`; server `339` passed with `2` ignored;
  types `183 / 183`.
- REVIEW 1, 2026-07-29: Opus/xhigh judged the refactor sound and independently
  confirmed movement fidelity, private-method callers, descendant-module
  visibility, import wiring, honest ratchet shrinkage, and the uniform `+1`
  generated inventory shift. Its one **FIX-FIRST** blocker was mechanical:
  `db/space_context.rs` was still untracked, so worktree tests could pass while
  the committed tree omitted the declared module. Exact staging of the new
  child plus the five tracked files is therefore the commit gate. The four
  behavior tests intentionally stay with the public Space-context types: they
  exercise the stable public lifecycle rather than private SQL mechanics.
- REVIEW 2, 2026-07-29: after exact staging and the staged-tree drift/M5
  rerun, Opus/xhigh returned **clean, mergeable**. It independently confirmed
  the child is present in the index, all 65 live `db.rs` inventory addresses,
  private-method locality, descendant visibility, and that this is a genuine
  DB implementation rather than path-based matcher evasion. Two non-blocking
  guard gaps remain explicit R4 closure work: dead baseline rows must not
  remain standing grants, and the final boundary must reject external
  connection retention or renamed access rather than trust the `db/**` path
  exclusion alone.

#### R4-2 — typed memory point reads

Selected after tracing callers, route responses, and M5 write semantics:

- Add one bounded `db/memory_point_reads.rs` child with exactly three typed
  read methods:
  `has_active_non_episode_memory_id`,
  `pending_memory_revision_payload`, and
  `memory_content_hashes_for_source_ids`.
- Keep `PageRevisionCard` and all `structured_fields` interpretation in
  `post_write`. The DB method returns only the selected pending-memory row
  projection, so deciding whether it is a `page_write` / `page` card remains
  post-write business logic.
- Preserve caller ordering: `page_source_reference_exists` first performs its
  logical `get_memory_detail` lookup and only uses the physical-id fallback for
  `creation_kind == "source"`; revision accept/dismiss still resolve the card
  before their existing atomic write/delete path; distillation still sorts,
  deduplicates, and takes the empty fast path before querying.
- Preserve exact query semantics: active, non-episode physical ids; exact
  revision `source_id` before legacy `supersedes`, then newest
  `last_modified`; and only `source = 'memory'`, `chunk_index = 0` content
  hashes with SQL `NULL` retained.
- This slice removes two production locks from `post_write.rs` and one from
  `synthesis/distill.rs`. RED is the ratchet reduction from `20 → 18` and
  `13 → 12`; it must fail before the caller migration and pass afterward.
- Strengthen the migration ratchet to exact per-path counts: an absent baseline
  row or a current count below its recorded allowance fails until the same diff
  lowers/removes the baseline. This prevents a successful cleanup from leaving
  a standing future grant. Globally sort diagnostics and extend the existing
  positive control with stale-path and reduced-count cases.
- M5 boundary: revision accept/dismiss HTTP responses remain classified
  `page_bearing: No` with `no prose fields`; the page revision payload is
  internal input to the unchanged `try_accept_page_revision` or delete path.
  No page query, truth adapter, `PagePermit`, manifest row, or cutover
  generation changes.

Execution evidence:

- RED: after lowering the two exact baselines before caller migration, the
  guard failed with `post_write.rs` increased `18 → 20` and
  `synthesis/distill.rs` increased `12 → 13`; its two mutation controls still
  passed. GREEN: the three caller migrations plus exact baseline counts pass
  the ratchet and controls `3 / 3`.
- `db/memory_point_reads.rs` contains exactly the three selected reads. Direct
  SQL-contract tests pass `3 / 3`: physical-id active/non-episode behavior;
  exact revision `source_id` precedence plus the newest eligible legacy
  `supersedes` fallback with raw JSON retained; and chunk-zero content hashes
  with SQL `NULL` and missing-row semantics.
- The production child is `112` lines. Its `358` lines of SQL-contract
  fixtures/tests live in `db/memory_point_reads_test.rs`, declared only under
  `#[cfg(test)]` and excluded from the M5 sweep by the established `_test.rs`
  suffix convention. The row projection is re-exported as a nameable
  `pub(crate)` DB type and explicitly named by its one production caller.
- The external literal total falls `330 → 327`; production literals fall
  `41 → 38`, while the explicit test-scoped set remains `289`. The ratchet now
  rejects increases, decreases without a same-diff baseline update, new files,
  and stale rows in globally sorted diagnostics.
- Existing behavior passes: revision accept `11 / 11`; dismiss `6 / 6`;
  atomic page-revision failure/retry `2 / 2`; document-majority distillation
  `1 / 1`. Caller order, page-card JSON interpretation, accept/delete writes,
  and transaction boundaries are unchanged.
- rust-analyzer reports no errors in the new child, `post_write.rs`,
  `synthesis/distill.rs`, or the ratchet. Reference lookup resolves every new
  method to exactly one production caller plus its local contract tests.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; generated line addresses shift in
  `db.rs`, `post_write.rs`, and `synthesis/distill.rs`, with no row,
  depth, or exposure change.
- PR-level verification passes: all-target Clippy with `-D warnings` for
  `wenlan-core` plus `wenlan-server`, and one uninterrupted
  `cargo test --workspace --lib --quiet`: CLI `32 / 32`; core `3,305` passed
  with `33` ignored; MCP `178 / 178`; server `339` passed with `2` ignored;
  types `183 / 183`.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **sound, ship-able** and found no
  semantic drift. Its useful non-blocking findings led to exact-baseline
  wording and the legacy-`supersedes` fallback assertions. Caller-specific
  error prefixes were retained deliberately because this movement slice
  preserves observable error text.
- REVIEW 2, 2026-07-29: Opus/xhigh caught one **FIX-FIRST** diagnostic bug:
  an increased count incorrectly told contributors to update the baseline.
  Increase and decrease remedies are now distinct and mutation-tested. The DB
  method also handles an empty hash-id slice, and the M5 evidence now names
  every file whose generated addresses shifted.
- REVIEW 3, 2026-07-29: Opus/xhigh found the extraction faithful with no
  correctness finding. Its consistency follow-ups externalized the inline test
  module, made the row projection nameable, and pinned missing-row plus SQL
  `NULL` structured-field behavior. The caller's sort/dedup/empty fast path
  remains intentionally explicit, and original error prefixes remain stable.
- REVIEW 4, 2026-07-29: Opus/xhigh returned **faithful, ship-able** after
  independently checking exact SQL text and ordering, error prefixes, caller
  semantics, lock scope, ratchet counts, test reachability, and inventory
  addresses. Its remaining findings were non-blocking documentation and
  positive-control nits: the ratchet heading now names exact matching, an
  equal-count fixture proves the accept branch, and the three bounded reads
  document physical-id, legacy precedence, and chunk-zero/`NULL` semantics.

#### R4-3 — derived-artifact sweep DB ownership

- Move the complete `MemoryDB` sweep implementation and its population helper
  from `derived_artifact_state/sweep.rs` into
  `db/derived_artifact_sweep.rs`. Keep only liveness/runtime state and the
  shared summary-eligibility predicate in `derived_artifact_state`.
- This is a genuine DB-owned transaction body, not orchestration moved to evade
  the matcher. Preserve the public scheduler method, test-only timestamp
  method, feature capture timing, exact eligibility SQL, 30-minute receipt
  cadence, one held lock, `BEGIN`/`COMMIT`/`ROLLBACK`, and every error prefix.
- The implementation bodies are byte-identical after the necessary import
  lines. Git recognizes the main file as a `98%` rename and its population
  child as `100%`.
- The exact ratchet removes the old external path. External literals fall
  `327 → 326`; production falls `38 → 37`; tests remain `289`. This is
  ownership relocation into the sanctioned DB layer, not deletion of the
  transaction's internal lock.
- Existing behavior gates pass:
  `durable_sweeps_drive_runner_readiness_and_active_backfill_suppresses_findings`
  `1 / 1` and
  `source_text_controls_episode_eligibility_in_sweep_and_runner` `1 / 1`.
  The ratchet passes `3 / 3`; core lib Clippy with `-D warnings` and formatting
  pass.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only the 65 generated `db.rs` addresses
  shift by `+1` for the new module declaration. The sweep reads memory and
  receipt tables, not Pages, and changes no truth adapter, manifest row,
  permit, or cutover generation.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **sound, ship it** with no
  correctness defects. It independently verified method resolution,
  visibility, imports, the required ratchet-row removal, all 65 inventory
  address shifts, and absence from the Page-reader set. Its useful
  non-blocking note fixed the M4 plan pointer to
  `summary_eligible_predicate`; the existing shared predicate dependency is
  retained rather than widening this movement-only slice.

#### R4-4 — maintenance queue presence probes

- Add `db/maintenance_queue.rs` with exactly two bounded read methods:
  `has_pending_retro_review` for `page_merge` /
  `page_keep_or_archive`, and `has_pending_cross_space_discovery` for the
  exact `cross_space_discovery` action.
- Preserve `SELECT 1`, `LIMIT 1`, only `pending` / `awaiting_review` statuses,
  exact error prefixes, and caller short-circuit order. The RetroReview,
  NearDuplicate, and CrossSpaceDiscovery stages still probe before metadata,
  Page scans, ANN work, cursor writes, or card emission.
- Do not fold in `pending_retro_review_count`: the full-tick retro-pause
  contract counts the already-filtered public refinement list and remains a
  separate path.
- RED is the exact `maintenance.rs` baseline reduction `7 → 5` before caller
  migration. GREEN moves the two production locks into the DB layer: external
  literals `326 → 324`, production `37 → 35`, tests `289` unchanged.
- Direct SQL-contract tests pass `2 / 2`, covering both retro actions and the
  cross-space action across both open statuses, with dismissed, resolved,
  auto-applied, and open wrong-action controls. A caller-level regression
  proves all three stages pause with zero scan/ANN work and no cursor advance;
  only RetroReview sets `retro_paused`.
- The existing maintenance suite passes `18 / 18`; the exact ratchet passes
  `3 / 3`; formatting and core all-target Clippy with `-D warnings` pass.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`. The new `_test.rs` module is excluded;
  generated addresses shift mechanically in `db.rs` and `maintenance.rs`, but
  the probes read only `refinement_queue` and change no truth or Page contract.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **clean movement-only refactor**
  with no correctness findings after independently checking byte-identical SQL
  and errors, all caller positions, the exact `7 → 5` ratchet, test-module
  reachability, schema defaults, and generated addresses. Its useful
  non-blocking notes added the table-level module description and strengthened
  the RetroReview early-return control to prove the empty scan cannot mark the
  lane complete. The duplicated two-action vocabulary remains byte-identical
  in this movement slice; consolidating it would change query construction and
  is intentionally deferred.

#### R4-5 — bounded automatic-retro Page scan

- Add `db/maintenance_retro_scan.rs` with one typed
  `scan_automatic_retro_stub_slice` method. It receives the existing cursor and
  fixed caller-owned source-read limit, holds one DB lock across the Page and
  normalized-source queries, and returns only `next_cursor`,
  `eligible_source_count`, and `more`.
- Preserve exact mechanism: keyset `id > cursor`, `ORDER BY id LIMIT 2`,
  active/distilled/not-user-edited/non-Overview eligibility,
  `page_sources ORDER BY memory_source_id LIMIT 3`, normalized-first semantics,
  legacy `source_memory_ids` fallback only when no normalized row exists, and
  all error text.
- Keep candidate and stage policy in the stage branch:
  `< STUB_PAGE_SOURCE_FLOOR`, `StubPageCandidate`, work counters, card
  emission, cursor persistence, and completion remain in `maintenance.rs`.
  The existing eligibility predicate remains with its SQL in the DB method.
- The first draft was rejected **FIX-FIRST** by the independent auditor. It
  returned a DB-owned `candidate_source_count` and left a wrapper, which moved
  policy into DB and changed the M5 topology to `192 / 55-51-86 / 22`.
  Removing the wrapper, returning the raw capped count, and retaining the
  original function name restores the exact reader set.
- RED is the `maintenance.rs` exact baseline reduction `5 → 4` before caller
  migration. GREEN moves one production lock: external literals `324 → 323`,
  production `35 → 34`, tests `289` unchanged.
- Direct contract tests pass `5 / 5`: empty and ineligible rows; zero, one,
  and four normalized sources with a three-row cap; legacy fallback and
  normalized precedence; malformed legacy JSON plus the reachable
  `user_edited = NULL` compatibility case; and two-row `more` with cursor
  advance. The
  maintenance suite passes `19 / 19`, including an ineligible-first-row
  control that advances the cursor without creating a card.
- The exact ratchet passes `3 / 3`; formatting and core all-target Clippy with
  `-D warnings` pass. The generated M5 inventory is again exactly `191` rows
  with depth `55 / 50 / 86` and exposure `22`: the sole depth-zero reader moves
  from `maintenance.rs` to the DB child with the same function identity. No
  truth adapter, manifest row, permit, or cutover generation changes.
- REVIEW 1, 2026-07-29: Opus/xhigh returned **behavior-preserving extraction**
  after tracing every empty, ineligible, and eligible caller output, lock
  ordering, exact ratchet counts, test wiring, and generated-reader movement.
  Its non-blocking findings made the result type explicitly nameable, aligned
  comments with the caller-supplied cap, removed an unreachable
  `writable_schema` test in favor of nullable `user_edited`, restored the
  movement-only blank line, and narrowed the plan's policy claim.

#### R4-6 — community cursor and opaque lease cleanup

- Add `db/community_grouping_state.rs` for the two DB-owned mechanics that
  escaped the boundary: the dirty-space cursor operation and the attempt-drop
  lease cleanup.
- Move the entire cursor `SELECT` plus `app_metadata` upsert under the same one
  lock into
  `claim_next_dirty_community_space`. Keep
  `run_next_community_grouping_cycle`, runtime ownership, prepare/compute/
  finalize order, held/not-dirty handling, and Page-route refresh in
  `community_grouping.rs`.
- Move `CommunityGroupingLeaseCleanup`, its Debug implementation, armed flag,
  `disarm`, and Drop into the DB child. Its constructor is private; a
  `pub(super)` `MemoryDB` factory is the only mint path and the opaque type
  exposes no raw field or connection constructor outside `db/**`.
- Preserve Drop semantics exactly: return without panic when no Tokio runtime
  exists; otherwise spawn the exact token/generation-gated lease delete and
  swallow cleanup errors. Preserve prepare failure's awaited release then
  disarm, and finalize's transactional delete/commit then disarm.
- RED removes the one `community_grouping.rs` ratchet row. GREEN moves the
  external literal total `323 → 322` and production `34 → 33`, with tests
  `289` unchanged. The previously uncounted retained
  `Arc<Mutex<Connection>>` capability also disappears from external production
  code; its narrowly typed implementation now exists only inside `db/**`.
- Behavior gates pass: no-runtime Drop `1 / 1`; aborted-finalize cleanup
  `1 / 1`; durable round-robin/restart `1 / 1`; M4 real-job/source-shape
  `1 / 1`; exact ratchet `3 / 3`. Removing the old file-tail test exposed a
  pre-existing Clippy layout constraint, so the existing
  `identity_event_tests` module moved mechanically to file end and its control
  passes `1 / 1`.
- Core plus server all-target Clippy with `-D warnings`, formatting, and staged
  diff checks pass. The generated M5 inventory remains exactly `191` rows with
  depth `55 / 50 / 86` and exposure `22`; only addresses move. No truth
  adapter, manifest row, permit, or cutover generation changes.
- AUDIT, 2026-07-29: the independent R4 auditor returned **APPROVE**, confirming
  opaque capability visibility, same-lock cursor atomicity, Drop/no-runtime
  behavior, prepare/finalize disarm order, M4 gate strength, and exact
  accounting. Its only commit gate was exact staging of both new files, now
  satisfied.
- REVIEW 1, 2026-07-29: Opus/xhigh found no correctness defect and judged the
  extraction faithful. It independently confirmed same-guard cursor mutation,
  verbatim Drop cleanup, unchanged prepare/finalize ordering, required ratchet
  removal, and the mechanical test relocation. Its two naming nits were adopted:
  the child now describes community grouping state and the mutating cursor
  operation is named `claim_next_dirty_community_space`. The M4 phase-slot
  clone gate intentionally remains scoped to `run_next_community_grouping_cycle`
  because it guards volatile runtime-state ownership, not DB capability access.

#### R4-7 — duplicate-maintenance typed readers

- Add `db/maintenance_duplicate_reads.rs` with an opaque
  `NearDuplicateSliceReader<'db>` that privately owns the DB mutex guard. The
  caller acquires it through `begin_near_duplicate_slice_reader`; it exposes no
  raw field, constructor, generic SQL callback, or `Deref`.
- Move the exact bounded pair-window SQL and lossy row decoding into the
  reader's `scan_near_duplicate_slice`, and the exact per-Page normalized
  source query into `load_bounded_page_source_ids`. The pure duplicate-policy
  evaluator in `maintenance/duplicates.rs` still owns eligible-page fanout,
  normalized-first legacy fallback, cap/truncation accounting, cosine and
  source overlap, thresholds, candidate selection, cursor, and `more`.
- Preserve the original lock lifetime explicitly: `maintenance.rs` begins the
  reader, loads the ordered pair rows, awaits the typed source reads through
  the evaluator, and calls `drop(reader)` only after policy evaluation. A
  direct concurrency test proves another DB operation remains pending until
  that drop.
- Move the exact embedding-distance query and decode into
  `MemoryDB::embedding_near_duplicate_pairs`. Its caller still computes the
  distance threshold, maps distance to similarity, merges source-overlap
  candidates, sorts them, and applies the final limit. The query deliberately
  retains its existing `a.space = b.space` predicate rather than changing to
  effective-workspace semantics.
- RED removes the complete `maintenance/duplicates.rs` baseline row. GREEN
  moves both production locks under `db/**`: external literals `322 → 320`,
  production `33 → 31`, tests `289` unchanged, and raw-access files `56 → 55`.
- Direct DB seam tests pass `3 / 3`, covering ordered pair limits and keyset
  resume, raw fallback and normalized-source projections, ineligible rows,
  scoped lock lifetime, embedding distance order, Overview exclusion, and SQL
  limit. Affected maintenance controls pass `6 / 6`: the two bounded automatic
  duplicate tests, dismissed and retro card behavior, and both deterministic
  survivor-order tests. The exact ratchet passes `3 / 3`.
- Formatting and core plus server all-target Clippy with `-D warnings` pass.
  The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`: the two depth-zero reader identities move
  from `maintenance/duplicates.rs` to the DB child, with no wrapper row added.
  No truth adapter, manifest row, permit, or cutover generation changes.
- AUDIT, 2026-07-29: the independent R4 auditor returned **APPROVE**. It
  confirmed the opaque capability, pair/source split, policy ownership,
  explicit post-evaluator drop, unchanged SQL/binds/errors/lossy decode,
  exact M5 topology, ratchet accounting, and positive-control strength.
- REVIEW 1, 2026-07-29: Opus 4.6 Thinking returned **APPROVE** with no material
  finding. The normal companion transport twice reached Anthropic but ended in
  provider `529 Overloaded`; the authorized direct `agy` transport then ran the
  same read-only staged-diff contract against the same Opus model. It
  independently verified the opaque reader, intentional lock lifetime,
  policy split, character-identical SQL/binds/error strings, threshold and
  result mapping, `191 / 55-50-86 / 22` M5 topology, ratchet, and substantive
  DB seam controls. Its only notes were non-blocking: the cross-await mutex is
  intentional movement-only behavior, SQL branch duplication is pre-existing,
  and the transported field names are clearer.

#### R4-8 — knowledge-quality diagnostic readers

- Add `db/kg_quality_diagnostics.rs` with two narrow typed reads:
  `count_stale_relation_sources` and
  `list_contradiction_observation_counts`. The exact SQL, row decoding, and
  existing error prefixes move under `MemoryDB`, including the thresholds that
  are part of those queries. Threshold interpretation, warning and info logs,
  `RethinkReport`, and orchestration remain in `kg_quality.rs`.
- Preserve the stale-source query's matched/missing/SQL-NULL behavior and the
  contradiction query's `HAVING obs_count >= 10`, descending count order, and
  `LIMIT 20` without adding another ordering contract.
- RED lowers the exact `kg_quality.rs` baseline from `26` to `24`; before
  extraction the guard reports a direct-access increase `24 → 26`. GREEN moves
  both production locks under `db/**`: external literals `320 → 318`,
  production `31 → 29`, tests `289` unchanged, and the per-file baseline is
  `26 → 24`.
- Direct DB controls cover a matched, missing, and SQL-NULL relation source,
  plus the contradiction boundary (`9` excluded, `10` included), descending
  counts, and the twenty-row cap. The two existing `test_run_rethink` controls
  pass, including the empty-DB report.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only generated source addresses move. No
  truth adapter, manifest row, permit, or cutover generation changes.
- AUDIT 1, 2026-07-29: the independent R4 auditor returned **FIX-FIRST** because
  the original combined threshold/order/cap fixture let `LIMIT 20` hide both
  the nine- and ten-observation rows. It also questioned the inner
  `memories.source_id IS NOT NULL` branch. The threshold now has an uncapped
  `9` excluded / `10` included control. AUDIT 2 returned **APPROVE** after
  confirming `memories.source_id` is `TEXT NOT NULL` in both production schema
  paths, so manufacturing an inner SQL NULL would require an invalid schema;
  the runtime control instead covers the genuinely nullable relation source.
- REVIEW 1, 2026-07-29: Opus/xhigh independently returned **FIX-FIRST** on the
  same vacuous threshold control. REVIEW 2 found the corrected extraction
  clean and movement-only with no blocking finding, confirming byte-preserved
  SQL/errors/returns, exact ratchet arithmetic, mechanical inventory addresses,
  and unchanged M5 topology. Its low-severity notes were adopted: the ledger
  now distinguishes query thresholds from quality interpretation, the
  multi-row observation fixture uses one transaction, and the typed row derives
  `Debug` consistently with sibling DB transports.

#### R4-9 — knowledge-quality vocabulary inputs

- Add `db/kg_quality_vocabulary.rs` with the purpose-bounded typed reads
  `distinct_relation_types_for_vocabulary_heal` and
  `distinct_entity_types_for_vocabulary_heal`. They move only the exact
  `SELECT DISTINCT` statements, lossy empty-string decoding, and existing
  error prefixes under `MemoryDB`; neither query gains an `ORDER BY`.
- Keep every canonical, alias, safe-transform, fold, proposal, log, count, and
  orchestration decision in `kg_quality.rs`. In particular, the caller still
  skips the empty string returned by the DB seam.
- RED lowers the exact `kg_quality.rs` baseline from `24` to `22`; before
  extraction the guard reports a direct-access increase `22 → 24`. GREEN moves
  both production locks under `db/**`: external literals `318 → 316`,
  production `29 → 27`, tests `289` unchanged, and the per-file baseline is
  `24 → 22`.
- Direct DB controls pass `2 / 2`, proving duplicate relation and entity types
  collapse, empty strings remain present, and novel values survive. The two
  existing vocabulary-heal behavior tests and three `run_rethink` controls
  pass `5 / 5`; the exact ratchet passes `3 / 3`.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only generated source addresses move. No
  truth adapter, manifest row, permit, or cutover generation changes.
- AUDIT, 2026-07-29: the independent R4 auditor returned **APPROVE**. It
  confirmed the two-lock move, exact SQL and diagnostics, unchanged policy
  ownership, meaningful duplicate/empty/novel-value controls, the
  `318 → 316` external and `29 → 27` production decreases, and unchanged M5
  topology.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding. It independently checked the staged diff against the
  frozen R4-9 contract and accepted the direct controls, focused behavior
  tests, ratchet evidence, M5 inventory, and reviewer-policy correction.

#### R4-10 — knowledge-quality duplicate-candidate inputs

- Add `db/kg_quality_duplicate_candidates.rs` with two purpose-bounded typed
  inputs: case-folded duplicate-name groups for merge-candidate counting and
  complete entity rows for the MinHash candidate pass. The exact SQL, lossy
  decoding, and existing error prefixes move under `MemoryDB`; neither query
  gains an `ORDER BY`.
- Keep warning logs, N−1 counting, the feature flag, high-entropy filtering,
  MinHash/LSH buckets, similarity thresholds, proposal identity/payload, and
  enqueueing in `kg_quality.rs`. The raw MinHash input deliberately includes
  low-entropy rows so the DB seam cannot become a hidden policy filter.
- The DB mutex now releases after each complete result is materialized rather
  than after caller-side logging/counting or entropy filtering. This bounded
  read-only lock-scope shrink is accepted: both queries still produce one
  materialized snapshot, while no policy or second DB operation moves under
  the guard.
- RED lowers the exact `kg_quality.rs` baseline from `22` to `20`; before
  extraction the guard reports a direct-access increase `20 → 22`. GREEN moves
  both production locks under `db/**`: external literals `316 → 314`,
  production `27 → 25`, tests `289` unchanged, and the per-file baseline is
  `22 → 20`.
- Direct DB controls pass `2 / 2`: three case variants form one count-three
  group, a second duplicate group remains distinct, singletons are excluded,
  and the raw MinHash read returns low- and high-entropy rows with every
  id/name/type field. Caller behavior controls pass `4 / 4`: three case-folded
  duplicates yield exactly N−1=`2` with MinHash off; MinHash on surfaces a
  true borderline pair but rejects low-entropy collision-like rows; the flag
  off path creates no band proposal. The exact ratchet passes `3 / 3`.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only generated source addresses move. No
  truth adapter, manifest row, permit, or cutover generation changes.
- AUDIT, 2026-07-29: the independent R4 auditor returned **APPROVE**, confirming
  the two-lock boundary, exact SQL/errors/decoding, policy ownership, and
  direct controls. Its requested caller N−1 and entropy-negative controls are
  included above.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE** after reconciling the staged implementation with the contract.
  It confirmed the low-entropy negative is substantive: the cross-type
  `aaaaaa` / `aaaaaaa` pair shares its sole trigram and would enqueue without
  the caller-owned entropy guard.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding. It independently confirmed exact SQL, diagnostics and
  decode defaults; policy placement; non-vacuous direct and caller controls;
  ratchet and M5 accounting; and exact staging of both new files.

#### R4-11 — knowledge-quality embedding-refresh inputs

- Add `db/kg_quality_embedding_refresh.rs` with
  `stale_entity_embedding_candidates_for_refresh` and
  `recent_entity_observation_contents_for_embedding_refresh`. The exact SQL,
  bind value, lossy decoding, and existing error prefixes move under
  `MemoryDB`; the typed candidate carries only entity id and name.
- Preserve the candidate query's strict
  `o.created_at > COALESCE(e.embedding_updated_at, 0)`, grouping,
  `HAVING COUNT(o.id) >= 5`, and absence of ordering. Preserve the observation
  query's `ORDER BY created_at DESC LIMIT 10` without adding a tie-breaker.
- Keep the candidate loop, name-first `parts`, empty-content filtering, `. `
  join, embedding generation/write, refreshed count, and success/failure logs
  in `kg_quality.rs`. Each bounded read returns before
  `refresh_entity_embedding`; no DB read lock spans embedding generation or
  the write transaction.
- Preserve the two-read race rather than combining candidate and observation
  reads into a transaction. The observation guard now releases after the raw
  contents are materialized, before caller-side empty filtering; this accepted
  materialized-snapshot lock shrink changes no selected row or ordering.
- RED lowers the exact `kg_quality.rs` baseline from `20` to `18`; before
  extraction the guard reports a direct-access increase `18 → 20`. GREEN moves
  both production locks under `db/**`: external literals `314 → 312`,
  production `25 → 23`, tests `289` unchanged, and the per-file baseline is
  `20 → 18`.
- Direct and caller controls pass `3 / 3`: SQL-NULL timestamp plus exactly five
  observations qualifies; four, and five equal to the refresh timestamp, do
  not; five newer rows qualify. A twelve-row fixture returns the exact newest
  ten in descending timestamp order, retains raw empty content, and excludes
  the two oldest. The caller refreshes one qualifying entity and the existing
  vector-search distance proves its stored embedding matches exactly
  `name + nonempty newest-ten` joined with `. `; the same rows do not trigger a
  second refresh. Existing `test_run_rethink` controls pass `2 / 2`; the exact
  ratchet passes `3 / 3`.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only generated source addresses move. No
  truth adapter, manifest row, permit, or cutover generation changes.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: after the missing ledger was added,
  the independent auditor returned **APPROVE**. It confirmed exact movement,
  test-enclave placement, preserved race and lock release, and accepted the
  vector-distance oracle as substantive when combined with the separate
  strict-timestamp boundary control.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding. It independently verified the two narrow methods, exact
  SQL/decode/error/order semantics, caller policy, non-vacuous controls,
  `20 → 18` ratchet, mechanical inventory shifts, and exact staging.

#### R4-12 — eval substrate guards

- Add `db/eval_substrate_guard.rs` with
  `assert_eval_feature_substrate_live` and
  `assert_eval_migration_substrates_live`. Each method acquires the DB mutex
  once and delegates policy, SQL, feature matching, and diagnostics unchanged
  to `eval::seed_contract::assert_feature_substrate_live`; no raw handle or
  callback escapes the DB seam.
- Keep the LoCoMo and LongMemEval per-feature gates in their existing position
  before fixture loading. Keep the migrate-stale gate after the fully-enriched
  check and before the shared Phase-1 backfill, as one mutex-held ordered
  sequence with the exact `temporal → graph → pages` refusal order. It opens no
  transaction and promises no cross-statement database snapshot.
- RED lowers the exact baselines before extraction: remove the sole
  `eval/locomo.rs` entry, lower `eval/longmemeval.rs` from `3` to `2`, and
  lower `eval/shared.rs` from `3` to `2`. The pre-extraction guard reports
  exactly those three direct-access violations. GREEN moves the three
  production locks under `db/**`: external literals `312 → 309`, production
  `23 → 20`, and tests remain `289`.
- Direct DB controls pass `2 / 2`: the feature wrapper preserves graph,
  temporal, and page substring aliases plus the unmodeled-feature pass; the
  migration wrapper refuses staged fixtures first for temporal, then graph,
  then pages, and passes only when all three substrates are live. Existing
  seed-contract controls pass `16 / 16` with the cached-scenario integration
  control still ignored.
- The exact external-access ratchet passes `3 / 3`. The generated M5 inventory
  remains exactly `191` rows with depth `55 / 50 / 86` and exposure `22`;
  only generated source addresses move. No truth adapter, manifest row,
  permit, or cutover generation changes.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE**, confirming exact forwarding, gate positions and refusal order,
  non-vacuous staged-substrate controls, ratchet accounting, exact staging,
  and unchanged M5 topology.
- REVIEW 1, 2026-07-29: the independent Sol reviewer returned **FIX-FIRST**
  because “materialized snapshot” overstated a mutex-held sequence of three
  queries. The ledger now explicitly promises neither a transaction nor a
  cross-statement snapshot. REVIEW 2 returned **APPROVE** with no remaining
  material finding.

#### R4-13 — eval lifecycle integrity reads

- Add `db/eval_lifecycle_integrity.rs` with four purpose-bounded reads:
  supersedes inputs, lifecycle state counts, merged source ids, and archived
  search inputs. Exact SQL, binds, error prefixes, lossy field decoding, and
  silent row-iteration termination move under `MemoryDB`; the methods return
  typed rows/state or a deduplicated id set, never a raw handle or callback.
- Keep `merged_before` filtering, cluster matching, tracker-map construction,
  relevance grades, and relevant-set policy in `eval/lifecycle.rs`. Keep
  archive-id construction, leakage calculation, and the search loop there.
  The supersedes, merged-id, and archive guards release after complete rows
  materialize and before caller policy or search.
- Preserve `count_db_state` as one mutex-held ordered sequence of its four
  count queries. It opens no transaction and promises no cross-statement
  database snapshot. The DB method returns only the four typed counts;
  `PhaseMetrics` construction and tuple interpretation remain in lifecycle.
  The legacy `concepts WHERE status = 'active'` query is unchanged even though
  current production schema no longer owns that table.
- RED removes the `eval/lifecycle.rs: 4` baseline row first; the exact guard
  reports four new direct accesses. GREEN moves those four production locks
  under `db/**`: external literals `309 → 305`, production `20 → 16`, tests
  remain `289`, and lifecycle disappears from the per-file baseline.
- Direct DB controls pass `3 / 3`: supersedes inputs retain preexisting rows,
  `DISTINCT` collapse, and NULL; merged ids exclude nonmerged/non-memory rows
  and deduplicate; the four counts preserve memory/archive/entity/legacy-active
  concept selection with negative controls; archived inputs retain empty
  content and include only archived memory heads. The exact legacy concepts
  surface exists only in the test fixture.
- Lifecycle controls pass `16 / 16`, including a caller-level control seeded
  through `upsert_documents`: `SupersedesTracker::build` filters the
  preexisting merged id, matches only the new row to its cluster, and extends
  grades/relevance only for that row. The exact external-access ratchet passes
  `3 / 3`.
- The generated M5 inventory remains exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only generated source addresses move. No
  truth adapter, manifest row, permit, or cutover generation changes.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE**, confirming exact lossy read behavior, legacy-count semantics,
  caller-owned tracker and leakage policy, substantive direct and caller
  controls, ratchet accounting, and unchanged M5 topology.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding. It independently verified all four DB methods, the
  same-guard count sequence, accepted materialization lock shrink, exact
  staging, and truth isolation.

#### R4-14 — eval pipeline reads

- Add `db/eval_pipeline_reads.rs` with two purpose-bounded reads. The corpus
  method returns only raw `Vec<String>` contents; `eval/pipeline.rs` remains
  the sole owner of BPE `count_tokens` aggregation and the row count. Preserve
  the exact
  `SELECT content FROM memories WHERE chunk_index = 0 AND supersede_mode <> 'archive'`
  SQL, `()` bind, `count_corpus: {e}` query error, and `Generic(e.to_string())`
  propagation for row iteration and field decoding. Do not add a source
  predicate, `DISTINCT`, ordering, empty-content filtering, or token policy.
  Materialize every row under the DB mutex and release it before BPE work.
- The evidence method returns only raw `Vec<(String, String)>` lineage pairs,
  including duplicates. Preserve the exact legacy
  `concept_sources` / `concepts` / `memories` join, `()` bind,
  `expand_evidence: {e}` query error, per-row `Generic(e.to_string())`
  propagation, absence of `DISTINCT` and ordering, and the literal
  `LIKE 'merged_%'` predicate. The caller retains the
  `HashMap<memory_sid, Vec<concept_sid>>`, `original_evidence.to_vec()`,
  additions loop, and `HashSet` insertion policy. Materialize every pair under
  the DB mutex and release it before reverse-map construction or expansion.
- Direct controls prove corpus head/non-archive selection, chunk and archive
  exclusion, retention of document and empty rows, and duplicate contents.
  Evidence controls prove active-only selection plus the merged-prefix,
  chunk-zero, and same-entity join boundaries; two distinct matching memory
  rows with the same source id prove that duplicate raw pairs survive the DB
  seam. A caller control proves that only matched evidence expands, original
  ids remain, and `HashSet` insertion collapses duplicate additions.
- RED removes only the `eval/pipeline.rs: 2` baseline row; before extraction
  the exact guard reports two new direct accesses. GREEN must move exactly
  those two production locks under `db/**`: external literals `305 → 303`,
  production `16 → 14`, and tests remain `289`. `eval/paired.rs: 1` retains
  its fail-soft `Err(_) => 0` fairness contract, and both
  `eval/longmemeval.rs: 2` event-date write loops remain unchanged. The DB test
  uses an `_test.rs` suffix.
- The generated M5 inventory must remain exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only mechanically shifted source addresses
  may change. No truth adapter, manifest row, permit, or cutover generation
  changes. This is a movement-only slice: run focused unit and structural
  controls, not a real LoCoMo/LongMemEval quality benchmark, and make no
  eval-effect claim.
- PRE-IMPLEMENTATION PREFLIGHT, 2026-07-29: the independent auditor identified
  the missing ledger as a **BLOCK** before implementation. The section above
  records its exact SQL, error, duplicate, lock-lifetime, caller-policy,
  ratchet, exclusion, and M5 conditions. The worker had made no implementation
  edit; its only change was the reversible RED baseline removal, which failed
  with exactly two new direct accesses.
- IMPLEMENTATION, 2026-07-29: `eval_pipeline_corpus_contents` materializes only
  raw contents and `eval_pipeline_evidence_pairs` materializes only raw
  lineage pairs. `eval/pipeline.rs` retains BPE aggregation, row counting,
  reverse-map construction, evidence copying, additions, and `HashSet`
  insertion. The two production guards release before those caller-owned
  operations.
- Direct DB controls pass `2 / 2`: corpus selection retains document, empty,
  and duplicate head rows while excluding archives and non-head chunks; the
  evidence fixture exercises status, prefix, chunk, and entity negatives,
  duplicate pairs from distinct matching rows, and a `mergedXwild` positive
  proving the underscore in `LIKE 'merged_%'` remains a SQL wildcard. Caller
  controls pass `2 / 2`: BPE totals and raw-row counts include empty and
  duplicate contents; evidence union preserves originals, ignores unmatched
  pairs, and collapses duplicate additions.
- The exact external-access ratchet passes `3 / 3`: external literals
  `305 → 303`, production `16 → 14`, tests remain `289`, and pipeline
  disappears from the per-file baseline. The generated M5 inventory and drift
  control remain exactly `191` rows with depth `55 / 50 / 86` and exposure
  `22`; only generated source addresses move. Focused tests, formatting,
  diff checks, and core/server all-target Clippy with `-D warnings` pass. No
  quality benchmark was run and no eval-effect claim is made.
- ROOT GATE, 2026-07-29: the relevant pipeline controls pass `4 / 4`, the
  exact ratchet passes `3 / 3`, and the M5 drift control passes `1 / 1`.
  Rust-analyzer reports no error in the new DB module or its direct test; each
  new method has exactly one production caller and one direct test reference.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor initially
  blocked a false `db.rs`-only inventory-address claim. After the ledger was
  corrected to permit all and only mechanical source-address shifts, it
  returned **APPROVE** for the code, controls, ratchet, inventory, exclusions,
  and exact seven-file staging.
- REVIEW 1, 2026-07-29: the independent Sol reviewer returned **FIX-FIRST** on
  the same ledger wording and found no code-level issue. REVIEW 2 returned
  **APPROVE** after the correction, with the staged scope still exact and
  clean.

#### R4-15 — paired-eval summary-node guard

- Add `db/eval_paired_guard.rs` with one narrow
  `MemoryDB::eval_paired_summary_node_count() -> Result<i64, WenlanError>`
  mechanic. It owns only the DB mutex, literal
  `SELECT COUNT(*) FROM summary_nodes` query with `libsql::params![]`, row
  iteration, and `i64` field decode. Query, `next`, and `get` failures surface
  as `Err`; an `Ok` query with no row returns `Ok(0)`. The method must not
  default an error to zero, assert, expose a raw handle, or accept a callback.
- Keep the fail-soft fairness policy in
  `eval/paired.rs::assert_summary_nodes_empty`: only the caller applies
  `.unwrap_or(0)`, so a missing table or any query/row/decode failure still
  behaves as zero. Keep the existing `assert_eq!` and its complete diagnostic
  text unchanged; only a positive `i64` count refuses the paired baseline.
  The DB mutex releases after read/decode and before caller fallback/assert.
- Keep both existing calls in their exact pre-fixture position: before
  `load_locomo` in the LoCoMo cross-rerank collector and before
  `load_longmemeval` in the LongMemEval collector. No scoring, attribution,
  fixture, or statistical policy moves under `MemoryDB`.
- Direct controls use an isolated DB to prove a missing table yields `Err`, an
  empty table yields `0`, and one inserted row yields `1`. Caller controls
  prove the missing-table error still passes and one row panics with the
  existing `summary_nodes is non-empty (1 rows): the global-prelude prepend`
  diagnostic prefix. The DB test module uses an `_test.rs` suffix.
- RED removes only the `eval/paired.rs: 1` baseline row; before extraction the
  exact guard reports one new direct access. GREEN moves that production lock
  under `db/**`: external literals `303 → 302`, production `14 → 13`, and
  tests remain `289`. Both `eval/longmemeval.rs: 2` event-date write loops
  remain unchanged.
- The generated M5 inventory must remain exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only mechanically shifted source addresses
  may change. No truth adapter, manifest row, permit, or cutover generation
  changes. This is a movement-only evaluator-integrity slice: run focused
  unit and structural controls, not a real paired quality benchmark, and make
  no eval-effect claim.
- PRE-IMPLEMENTATION PREFLIGHT, 2026-07-29: independent Sol and auditor reviews
  both returned **APPROVE** for this boundary. Both rejected an `*_or_zero` DB
  method because error-to-zero is evaluator fairness policy, not a storage
  mechanic; the `Result<i64, WenlanError>` seam and caller fallback above are
  the frozen resolution.
- IMPLEMENTATION, 2026-07-29:
  `MemoryDB::eval_paired_summary_node_count` owns only the literal count query,
  row iteration, and `i64` decode. Missing-table and other query/row/decode
  failures remain `Err`, while `assert_summary_nodes_empty` alone applies the
  unchanged `.unwrap_or(0)` fairness fallback and exact assertion diagnostic.
  Both LoCoMo and LongMemEval call sites remain in their original pre-fixture
  positions.
- Direct and caller controls pass `2 / 2`: one isolated fixture proves missing
  table `Err` plus caller pass, empty table `0`, and one row `1`; a second
  proves the one-row caller panic retains the required
  `summary_nodes is non-empty (1 rows): the global-prelude prepend` prefix.
  Existing paired serialization control passes `1 / 1`.
- RED reports exactly one new `eval/paired.rs` direct access after removing
  only its baseline row. The exact ratchet then passes `3 / 3`: external
  literals `303 → 302`, production `14 → 13`, tests remain `289`, and paired
  disappears from the per-file baseline. The generated M5 inventory and drift
  control remain exactly `191` rows with depth `55 / 50 / 86` and exposure
  `22`; only mechanical source addresses move. Focused tests, formatting, diff
  checks, and core/server all-target Clippy with `-D warnings` pass. No quality
  benchmark was run and no eval-effect claim is made.
- ROOT GATE, 2026-07-29: paired guard controls pass `2 / 2`, the existing
  paired serialization control passes `1 / 1`, the exact ratchet passes
  `3 / 3`, and the M5 drift control passes `1 / 1`. Rust-analyzer reports no
  diagnostic in the new DB module or its direct test; the method has one
  production caller.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE** for the fail-soft boundary, controls, unchanged call positions,
  ratchet, M5 inventory, excluded LongMemEval writes and truth scope, and exact
  seven-file staging.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding. It identified pre-existing prose drift in the M5 inventory
  document: its historical “Draft 5” narrative still says
  `190 / 54-50-86`, while the executable generator and frozen ledger say
  `191 / 55-50-86`. That cleanup is assigned to R8 rather than widening this
  movement slice.

#### R4-16 — LongMemEval temporal seed writes

- Add `db/eval_temporal_seed.rs` with a crate-visible typed seed carrying
  `source_id: String` and `event_date: i64`, plus one purpose-bounded
  `MemoryDB` method returning `()`. The method acquires one DB mutex even for
  an empty slice, iterates seeds in supplied order, and executes the exact
  `UPDATE memories SET event_date = ? WHERE source_id = ?` with
  `libsql::params![seed.event_date, seed.source_id.as_str()]`.
- Preserve the existing intentionally lossy mutation semantics: apply
  `let _ = conn.execute(...).await` independently for every seed, continue
  after a failed statement, open no transaction, perform no rollback,
  prevalidation, sorting, or deduplication, and return no affected-row count or
  error. One mutex remains held across the complete ordered loop; no raw handle
  or callback crosses the DB boundary.
- Extract one private pure `t4a_event_date_seeds` helper in
  `eval/longmemeval.rs` and call it from both temporal runners. The helper owns
  `haystack_dates.get(mem.session_idx)`, `parse_lme_date`, full
  `DateTime::timestamp()` conversion, and `memory_source_id` construction from
  `mem.question_id`, `session_idx`, and `turn_idx`. It preserves input order
  and duplicate seeds and skips missing or malformed dates.
- Do not reuse `event_date_map`: that separate T11/T20 contract stores seeds in
  a `HashMap` and deliberately truncates timestamps to midnight. Both T4a
  loops currently preserve the full parsed time, such as `17:50`; changing
  them to midnight, reordered, or deduplicated seeds is behavior drift.
- Keep both calls immediately after `upsert_documents` and before relevance,
  cue, or search work. The report runner retains `total_memories`; the
  per-query collector retains its latency and touched-channel policy. No
  fixture, relevance, temporal-cue, search, scoring, or reporting policy moves
  under `MemoryDB`.
- Direct controls install a trigger that aborts one selected update and pass
  ordered seeds with successful writes before and after it. They prove earlier
  and later successes persist, the failure is swallowed, no transaction rolls
  the batch back, and an unmatched source is harmless. Two physical memory rows
  sharing one matching `source_id` must both update; an existing unrelated row
  must retain its prior `event_date`; duplicate seeds for the matching
  `source_id` must apply in input order with the last successful timestamp
  retained. Separate controls prove an empty slice completes without mutation
  and an update against a missing `memories` table returns normally after
  swallowing the SQL error.
- A caller-helper control uses a non-midnight fixture and asserts the exact
  full Unix timestamp, non-natural input order, duplicate preservation, and
  malformed plus missing-session skips. The DB test module uses an `_test.rs`
  suffix.
- RED removes only the `eval/longmemeval.rs: 2` baseline row; before
  extraction the exact guard reports two new direct accesses. GREEN moves
  both production locks under `db/**`: external literals `302 → 300`,
  production `13 → 11`, and tests remain `289`. The method/seed re-export and
  test wiring add four mechanical `db.rs` lines.
- The generated M5 inventory must remain exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only mechanically shifted source addresses
  may change. No truth adapter, manifest row, permit, or cutover generation
  changes. This is a movement-only eval-seed slice: run focused unit and
  structural controls, not a real LongMemEval quality benchmark, and make no
  eval-effect claim.
- PRE-IMPLEMENTATION PREFLIGHT, 2026-07-29: independent Sol and auditor reviews
  agree on the ordered typed-seed boundary. The auditor returned **BLOCK** on
  the initial candidate until the full-time-versus-midnight distinction and
  non-midnight control were explicit; with those additions, the section above
  is the frozen resolution.
- IMPLEMENTATION, 2026-07-29: `EvalTemporalSeed` and
  `MemoryDB::apply_eval_temporal_seeds` move only the ordered lossy update
  mechanic under one mutex. The method locks even for an empty slice, executes
  the exact update and params for every supplied seed with an independent
  ignored result, opens no transaction, and returns no result or count.
  `t4a_event_date_seeds` remains caller-owned and both temporal runners invoke
  it immediately after their existing upserts.
- Direct DB controls pass `3 / 3`: a trigger-aborted middle update is swallowed
  while earlier and later writes persist; unmatched and unrelated rows remain
  harmless; both physical rows sharing one source id update; duplicate seeds
  retain their last successful timestamp. Separate controls prove an empty
  slice leaves state untouched and a nonempty write against a missing
  `memories` table returns normally. The caller-helper control passes `1 / 1`,
  preserving a non-midnight full timestamp, non-natural input order,
  duplicates, and malformed/missing-date skips.
- RED reports exactly two new `eval/longmemeval.rs` direct accesses after
  removing only its baseline row. The exact ratchet passes `3 / 3`: external
  literals `302 → 300`, production `13 → 11`, tests remain `289`, and
  longmemeval disappears from the per-file baseline. The generated M5
  inventory and drift control remain exactly `191` rows with depth
  `55 / 50 / 86` and exposure `22`; only mechanical source addresses move.
  Focused tests, formatting, diff checks, and core/server all-target Clippy
  with `-D warnings` pass. No quality benchmark was run and no eval-effect
  claim is made.
- ROOT GATE, 2026-07-29: direct DB controls pass `3 / 3`, the caller helper
  passes `1 / 1`, the exact ratchet passes `3 / 3`, and the M5 drift control
  passes `1 / 1`. The Rust LSP is active; the new DB module, direct test, and
  LongMemEval caller have zero error diagnostics, and the method has exactly
  two production plus three direct-test references.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE** for the original staged code and again after the ordered-audit
  delta. It verified the deterministic one-row-per-statement trigger oracle,
  all lossy-write controls, ratchet, M5 inventory, truth isolation, and exact
  seven-file staging.
- REVIEW 1, 2026-07-29: the independent Sol reviewer returned **FIX-FIRST**
  because final-state assertions did not reject an implementation that sorted
  seeds before writing. The direct control now records and asserts the exact
  successful statement order with deliberately non-sortable inputs; it also
  proves failed and unmatched seeds emit no audit row. REVIEW 2 returned
  **APPROVE** with no remaining material finding.

#### R4-17 — complete-entity recovery receipt read

- Add `db/repair_receipt.rs` with one narrow
  `MemoryDB::read_repair_target_receipt(&RepairTarget)` method for the existing
  `repair_target_receipt_on_connection` mechanic. It locks, delegates, and
  returns the same `Result<(RepairDigest, u64), WenlanError>` without exposing
  a connection, guard, callback, SQL string, transaction, artifact, or
  target-filtering surface. Keep the target-specific helper in `repair.rs`
  until its four same-transaction callers move in later slices; do not
  reimplement its scope, row-count, capture-limit, error, or lossy-decode
  semantics.
- Replace only the independent lock in
  `recover_complete_entity_extraction_apply_receipt`. Do not rewrite
  `target_receipt_current`, or replace same-transaction uses in post-write or
  verification: those callers must continue to read through the connection
  that already owns their atomic operation.
- Preserve the caller's ordering: pending artifact parse, receipt read, guard
  release, then publish/remove/sync filesystem work. A concurrency control
  must prove the DB mutex is released immediately before both the publish and
  remove artifact operations. Use a private `#[cfg(test)]` assertion/hook at
  those exact sites; do not add a production callback surface.
- Add `db/repair_receipt_test.rs`. A real
  `MemoryEntityExtraction` target must return the same non-default digest and
  row count `1` as the existing on-connection helper, while a wrong scope
  preserves `repair_target_stale`. The existing entity-recovery control must
  still prove both original-state pending removal and valid committed-pending
  publication; both branches exercise the pre-artifact unlocked assertion.
- GREEN floor: external literals `300 → 299`, production `11 → 10`, tests
  remain `289`; lower only the `repair.rs` ratchet row `20 → 19`.
- PRE-IMPLEMENTATION PREFLIGHT, 2026-07-29: the independent auditor returned
  **APPROVE** and the independent Sol reviewer returned **FIX-FIRST** until
  the concrete API and non-vacuous controls above were frozen. LSP finds one
  production caller of `recover_complete_entity_extraction_apply_receipt` and
  five production uses of `repair_target_receipt_on_connection`; after the
  move the new DB child replaces only the recovery use, so the connection
  helper remains at five production callers and the new method has exactly one
  production caller plus direct tests. Both reviewers confirmed
  `300 → 299`, `11 → 10`, and tests `289`, with no M5/truth or generation
  change.
- RED, 2026-07-29: lowering only the `repair.rs` ratchet row `20 → 19`
  produced the exact expected failure:
  `repair.rs: direct .conn.lock() access increased 19 -> 20`.
- IMPLEMENTED, 2026-07-29: `db/repair_receipt.rs` owns the narrow locking
  method and delegates unchanged to
  `repair_target_receipt_on_connection`. Only
  `recover_complete_entity_extraction_apply_receipt` moved to it; the five
  production helper callers remain the DB child plus the four
  same-transaction callers. Private test-only assertions at the exact publish
  and remove sites acquire the DB mutex with `try_lock` before artifact I/O,
  and the existing recovery control observes both sites by manifest id. No
  production callback, truth, generation, transaction, or filtering surface
  changed.
- GREEN, 2026-07-29: the direct real-entity receipt control passes `1 / 1`
  with exact helper equality, row count `1`, a non-default digest, and
  `repair_target_stale` for the wrong scope. The existing recovery control
  passes `1 / 1` with committed-pending publication, original-state pending
  removal, final artifact assertions, and both unlocked-site observations.
  The exact ratchet passes `3 / 3`; external literals are `300 → 299`,
  production `11 → 10`, and tests remain `289`.
- ROOT GATE, 2026-07-29: Rust LSP resolves the recovery call to the new DB
  method, reports exactly one production and two direct-test references plus
  the declaration, and reports zero error diagnostics in all five changed
  Rust files. The M5 inventory check and drift control pass with exactly `191`
  rows, depth `55 / 50 / 86`, and exposure `22`; only mechanical source
  addresses changed. Formatting, diff checks, and core/server all-target
  Clippy with `-D warnings` pass.
- ROOT MUTATION CONTROL, 2026-07-29: the first approved test-only check proved
  the exact branch and unlocked mutex but would still pass if the whole check
  moved after artifact I/O. Both sites now also require the pending file to
  exist and the final file not to exist before `try_lock`. Moving the publish
  check after publication makes both predicates fail; moving the remove check
  after removal makes the pending predicate fail. The focused recovery control
  and all-target Clippy pass after this strengthening.
- POST-IMPLEMENTATION AUDIT, 2026-07-29: the independent auditor returned
  **APPROVE** for the original eight-file staged diff and **APPROVE** again for
  the mutation-control delta. It confirmed the typed boundary, exact helper
  reference set, two artifact branches, UUID-isolated observation log,
  ratchet, address-only M5 inventory, and unchanged truth/generation surface.
- REVIEW, 2026-07-29: the independent Sol reviewer returned **APPROVE** with no
  material finding, then **APPROVE** on the delta. It independently verified
  the one-lock movement, untouched same-transaction callers, exact direct and
  recovery controls, parallel-test safety, and that the strengthened
  pre-artifact predicates make ordering mutations fail deterministically while
  remaining entirely `#[cfg(test)]`.
- POST-COMMIT CORRECTION, 2026-07-29: the root mutation-control delta added
  fourteen test-only lines in `repair.rs` after the initial inventory
  regeneration. Focused tests and Clippy were rerun, but the M5 check was not;
  the committed ROOT GATE sentence above therefore described the pre-delta
  inventory. R4-18 preflight caught the exact address-only RED:
  six `repair.rs` reader addresses were stale by `+14`. The generated block is
  now refreshed, `python3 scripts/m5-reader-sweep.py --check` and the Rust M5
  drift test pass again at `191 / 55-50-86 / 22`, and no function identity,
  exposure, truth, or production code changed. This correction lands in a
  separate commit before R4-18 implementation rather than hiding the missed
  gate in the next slice.

#### R4-18 — memory repair CAS transactions

- Move the complete transaction bodies of `reclassify_memory_cas_inner` and
  `complete_entity_extraction_cas_inner` behind two named `MemoryDB` methods in
  `db/repair_memory_cas.rs`:
  `reclassify_memory_repair_cas` and
  `complete_entity_extraction_repair_cas`. Caller facades retain their current
  public or crate-visible names, signatures, proof callback, and paths and
  delegate exactly once.
- The deliberate rollback-failure injection remains production-excluded:
  expose one `#[cfg(test)]` companion DB method per operation through the
  existing `*_with_forced_rollback_failure` facades. Normal and forced methods
  may share one private operation-specific inner with the existing boolean;
  no production-callable failure flag/enum, generic transaction callback, or
  common CAS executor is allowed.
- Keep `RepairWriteProof` at `crate::post_write` with private fields and its
  existing public methods. Add only a crate-private
  `RepairWriteProof::from_parts` constructor so the DB child can construct the
  existing type. Move `rollback_repair_transaction` and
  `recovery_required_after_rollback_failure` into the DB child as private
  helpers; do not make the proof fields, connection, rollback helper, or
  transaction state broadly visible.
- Preserve each exact `BEGIN IMMEDIATE`, validation and receipt order,
  `total_changes` normalization, proof construction, hook-before-commit
  position, forced rollback-failure path, commit handling, and mutex lifetime.
  Sharing a private rollback/proof helper is allowed only if normalized moved
  bodies remain mechanically equivalent; the two operations do not become a
  generic CAS executor.
- Direct controls must cover stale target, successful proof, hook failure,
  ordinary rollback, forced rollback failure, and a blocked concurrent writer
  across the pre-commit hook for each operation. Put the direct method controls
  in `db/repair_memory_cas_test.rs`; existing repair and entity-extraction
  tests remain facade/integration proof rather than the sole evidence.
- The mutex control is deterministic, not timeout-based: while the existing
  synchronous proof hook runs, a scoped OS thread and two-party
  `std::sync::Barrier` must attempt `db.conn.try_lock()`, record whether it was
  blocked, drop any unexpectedly acquired guard, and always rendezvous. Only
  after the rendezvous does the hook assert that the attempt was blocked.
  Never assert on the worker before the barrier: an early-unlock mutation must
  fail cleanly rather than panic one party and strand the hook. After the
  method returns, the test proves the committed state is visible and the mutex
  can be acquired. No production async callback or new test hook is added.
- GREEN floor: external literals `299 → 297`, production `10 → 8`, tests
  remain `289`; lower only the `post_write.rs` ratchet row `18 → 16`.
- LSP must show each normal DB method with one production facade caller plus
  its direct tests, and each forced companion reachable only through the
  retained `#[cfg(test)]` facade and direct tests. The public reclassification
  facade remains declaration plus one production caller; the complete-entity
  facade remains declaration plus one production and two existing direct-test
  callers. Neither moved lock remains in `post_write.rs`.
- PRE-IMPLEMENTATION PREFLIGHT, 2026-07-29: independent Sol and auditor reviews
  returned **FIX-FIRST** until the proof-construction seam,
  production-excluded rollback injection, purpose-specific DB methods, and
  per-operation barrier controls above were explicit. Both traced the current
  bodies as one mutex plus one `BEGIN IMMEDIATE`, with every body error,
  callback error, and commit error routed through the same rollback mapping.
  Existing tests already cover stale, success, ordinary rollback, and forced
  rollback for both operations, but neither directly names the new DB methods
  or proves the mutex is held across the live proof hook; the new controls
  close those gaps. Both confirmed `299 → 297`, `10 → 8`, tests `289`, and no
  M5/truth or generation change. The independently discovered stale R4-17 M5
  addresses were corrected in commit `111d0fc7` before this slice began, so
  R4-18 starts from a green `191 / 55-50-86 / 22` inventory.
- RED, 2026-07-29: lowering only the `post_write.rs` ratchet row `18 → 16`
  produced the exact expected failure:
  `post_write.rs: direct .conn.lock() access increased 16 -> 18`.
- IMPLEMENTED, 2026-07-29: `db/repair_memory_cas.rs` now owns the complete
  reclassification and entity-extraction transaction bodies as two
  purpose-specific `MemoryDB` operations. The existing normal facades each
  delegate exactly once; their test-only forced-rollback facades delegate to
  operation-specific `#[cfg(test)]` companions. `RepairWriteProof` retains
  private fields and gains only `pub(crate) from_parts`; rollback helpers are
  private to the DB child. SQL, validation/receipt order, normalized change
  arithmetic, saturation, proof-hook-before-COMMIT order, rollback mapping,
  and mutex lifetime remain unchanged, with no generic executor or
  production-callable failure flag.
- GREEN, 2026-07-29: direct DB controls pass `10 / 10`. For each operation
  they prove stale/no mutation, successful proof and committed state, hook
  failure rollback, SQL/mutation failure rollback, exact
  `repair_apply_recovery_required` on forced rollback failure, and a
  deterministic scoped-thread `Barrier` control showing `try_lock` blocked
  during the proof hook then succeeded after return. Focused existing facade
  success and rollback-uncertainty controls pass `4 / 4`; the exact ratchet
  passes `3 / 3`. External literals are `299 → 297`, production `10 → 8`,
  and tests remain `289`.
- ROOT GATE, 2026-07-29: Rust LSP reports each normal DB method as its
  declaration, one production facade, and four direct controls; each forced
  companion is its declaration, the retained test-only facade, and one direct
  control. The public reclassification facade is declaration plus one
  production caller; the complete-entity facade is declaration plus one
  production and two existing direct-test callers. All five changed Rust
  files have zero error diagnostics. The final generated M5 inventory and
  Rust drift control pass at exactly `191` rows, depth `55 / 50 / 86`, and
  exposure `22`; only mechanical addresses changed. Formatting, diff checks,
  and core/server all-target Clippy with `-D warnings` pass. No truth or
  generation surface changed.
- REVIEW GATE, 2026-07-29: the routine Sol architecture/API review returned
  `APPROVE`; the independent contract and kill-power audit returned `APPROVE`.
  Both reviewed the exact staged seven-file slice and reported no finding or
  source conflict.

#### R4-19 — deterministic database repair transaction

- Move `apply_deterministic_repair_cas` as one named DB-owned atomic operation.
  Keep writer dispatch, tag-record validation, route-invalidation accounting,
  parity/effect guards, receipt reads, proof, callback, rollback, and commit in
  their current order under one mutex and one `BEGIN IMMEDIATE`.
- Do not split the mutation match into independently committed methods and do
  not reuse the new separately locking receipt method from R4-17 inside this
  transaction.
- Existing writer-specific behavior tests remain the policy controls. Add a
  transaction barrier proving no second DB writer crosses between the initial
  receipt and commit, plus hook-failure rollback and route-invalidation
  positive controls.
- GREEN floor: external literals `297 → 296`, production `8 → 7`, tests remain
  `289`.

Frozen slice contract:

- Add `db/repair_deterministic.rs` plus its sibling direct-test module. The
  child defines the sole canonical
  `pub async fn apply_deterministic_repair_cas(db: &MemoryDB, ...)` and owns the
  complete atomic body. `db.rs` exposes only the child module as `pub(crate)`;
  `post_write.rs` publicly re-exports the function with
  `pub use crate::db::repair_deterministic::apply_deterministic_repair_cas`.
  This is an item re-export, not a delegating wrapper or second declaration.
  The existing public path, signature, visibility, production caller, and
  policy-test callers remain unchanged. The implementation module stays
  inaccessible outside the crate, so the only external path remains
  `post_write::apply_deterministic_repair_cas`. Do not add a generic
  transaction runner, connection parameter, transaction token, or callback
  other than the existing synchronous proof hook.
- Move the complete body, not only its SQL match. Preserve the exact order:
  reject the reclassification/projection writer mismatch before locking; take
  the DB mutex; `BEGIN IMMEDIATE`; validate the tag-record set; read and compare
  the initial target receipt; conditionally capture archive-page route
  generations; capture parity and non-target guards; dispatch exactly the
  seven existing writer/mutation arms; require a nonzero affected count; read
  and compare the final target receipt; verify route invalidation; normalize
  allowed changes and parity; compare the non-target guard; capture the final
  database digest; construct the opaque proof; run the synchronous hook; then
  `COMMIT`. Do not call R4-17's separately locking receipt method from inside
  the transaction.
- Preserve every arm and predicate byte-for-byte apart from mechanical
  qualification/formatting:
  `NormalizeMemorySourceAgent`, `ClearMemorySupersedes`,
  `UnstageOrphanRevision`, `DeleteTagRow`, `DeleteMemoryEntityLink`,
  `BindPageLink`, and `ArchiveEmptySourcePage`. The archive arm must still
  require the empty, active, unconfirmed, unedited, source-kind, source-less
  page at the expected version and scope.
- Preserve route-invalidation arithmetic exactly. Only archive-page repair
  captures the pre-state; its page generation must equal
  `page_generation.saturating_add(1)`, its space and space generation must
  remain equal, and exactly one derived change is added with `checked_add`.
  That saturating compare, parity subtraction, total-change subtraction, and
  `repair_effect_counter_{overflow,underflow}` errors do not change.
- Preserve this operation's existing error mapping, which intentionally is not
  R4-18's recovery-required mapping. Body or proof-hook failure attempts
  `ROLLBACK`; rollback failure returns
  `VectorDb("{error}; repair rollback failed: {rollback_error}")`. `COMMIT`
  failure performs best-effort rollback and returns
  `VectorDb("repair commit failed: {error}")`. No new production failure
  injection is added. `RepairWriteProof` fields remain private; use only its
  existing crate-private constructor at the DB boundary.
- Direct controls live under the DB child without widening production
  visibility. A simple deterministic writer proves success, proof contents,
  committed state, and the mutex boundary. Its synchronous hook uses the
  no-hang scoped-thread protocol: attempt `try_lock`, record whether blocked,
  drop any unexpectedly acquired guard, always rendezvous at a two-party
  `Barrier`, then assert in the hook. A separate hook-failure control proves the
  target mutation rolls back and the mutex is acquirable after return.
- A positive `ArchiveEmptySourcePage` control must prove the page commits as
  archived, the page-route generation advances by exactly one, the route space
  and space generation do not change, and the proof's non-target guards match.
  It catches missing or broadened derived-change accounting. Add a separate
  negative control in the disposable test DB: seed a valid archive-page route
  row, drop `m4_page_community_page_invalidate`, invoke the DB method, and
  require the exact `repair target page route invalidation unproven` error plus
  rollback to an active page. That control must fail if route verification is
  removed or weakened. Existing writer-specific deterministic repair tests
  remain unchanged and green.
- RED lowers only the `post_write.rs` exact ratchet row `16 → 15`. GREEN is
  external `297 → 296`, production `8 → 7`, tests `289`. Expected files are
  `db.rs`, the two new DB child files, `drift_guard.rs`, `post_write.rs`, the
  generated M5 inventory, and this ledger; no existing policy-test file,
  truth/generation surface, or later R4 slice changes.
- A source/syntax census must show exactly one
  `fn apply_deterministic_repair_cas` declaration, in the DB child. The
  unchanged public-path calls are exactly one production expression at current
  `repair.rs:2108` plus the same four test expressions at current
  `deterministic_tests.rs` lines `941`, `1153`, `1241`, and `3568`; direct
  sibling controls call the canonical function. LSP `goto_definition` from
  every production/policy-test call must resolve through the re-export to the
  DB-child definition. Do not require `find_references` alone to enumerate
  line `1241`: rust-analyzer currently omits it even though definition lookup
  succeeds. The moved raw lock disappears from `post_write.rs`; no raw DB
  capability escapes the child.
- Start from the green R4-18 M5 state `191 / 55-50-86 / exposure 22`.
  Regenerate the inventory only after all code, test, and ledger changes, then
  run both the script and Rust drift gate again after any review delta. Counts
  and depth/exposure partition remain unchanged; only mechanical source
  addresses may move.
- CONTRACT GATE, 2026-07-29: routine Sol and the independent contract auditor
  both returned `APPROVE` after the dropped-trigger negative control,
  saturating route-generation semantics, and the four-call syntax/LSP split
  were made explicit. No implementation file changed during contract review.
- PRE-IMPLEMENTATION TOOLING CORRECTION, 2026-07-29: the first inherent-method
  move demonstrated two scanner-visible topology failures despite green
  runtime tests. Sharing the facade name increased M5 name ambiguity `9 → 10`
  and yielded `189 / 55-49-85 / exposure 21`; a distinct inherent name restored
  `191 / 55-50-86` but still pushed the exposed public repair entry to depth
  three, yielding exposure `21`. Bypassing the facade would retain that extra
  definition and yield `192 / 55-51-86 / exposure 22`. The approved topology
  therefore uses one canonical DB-child free-function definition plus the
  public `post_write` item re-export. This preserves the original call edges
  and external path while restoring `191 / 55-50-86 / exposure 22`. Lowering
  the inventory, adding fake edges, or editing the scanner is forbidden.
- TOPOLOGY CORRECTION GATE, 2026-07-29: routine Sol and the independent
  contract auditor both returned `APPROVE` for the canonical DB-child
  definition plus public item re-export. They rejected both the extra wrapper
  layer and the direct-dispatcher bypass because those topologies cannot
  preserve the frozen M5 row/depth/exposure contract.
- RED, 2026-07-29: lowering only the `post_write.rs` ratchet row `16 → 15`
  produced the exact expected failure:
  `post_write.rs: direct .conn.lock() access increased 15 -> 16`.
- IMPLEMENTED, 2026-07-29: `db/repair_deterministic.rs` now owns the complete
  deterministic CAS as the canonical `apply_deterministic_repair_cas`;
  `post_write` retains the public path as a direct item re-export. The mutex,
  `BEGIN IMMEDIATE`, tag-set validation, receipt and route reads, seven SQL
  arms and predicates, affected-count proof, route-invalidation accounting,
  parity/effect normalization, opaque proof construction, synchronous hook,
  rollback mapping, and commit mapping remain in their original order.
  `RepairWriteProof::from_parts` is the only construction seam; no generic
  executor, production failure seam, raw connection, or transaction
  capability was added.
- GREEN, 2026-07-29: direct DB controls pass `4 / 4`: simple-writer
  success/proof/commit with the no-hang scoped-thread `Barrier` mutex control,
  proof-hook rollback, archive-page route generation/space/proof accounting,
  and the dropped-`m4_page_community_page_invalidate` exact
  `repair target page route invalidation unproven` error with active-page
  rollback. The unchanged deterministic policy module passes `46 / 46`; the
  exact ratchet passes `3 / 3`. External literals are `297 → 296`,
  production `8 → 7`, and tests remain `289`. Formatting, diff checks, and
  core/server all-target Clippy with `-D warnings` pass.
- LSP GATE, 2026-07-29: source search shows one canonical DB-child function
  declaration, the unchanged production call, four unchanged policy-test
  calls, and four direct controls. `goto_definition` from the production and
  policy-test calls resolves through the `post_write` re-export to the
  canonical DB-child definition. As documented, rust-analyzer
  `find_references` omits policy-test line `1241` while its definition lookup
  succeeds. All five changed Rust files have zero error diagnostics.
- ROOT GATE, 2026-07-29: a whitespace-insensitive comparison of the old
  transaction with the DB-child function differs only at opaque proof
  construction (`RepairWriteProof { ... }` becomes the existing
  `from_parts(...)` boundary). Root independently reran direct controls `4 / 4`,
  unchanged policy controls `46 / 46`, ratchet controls `3 / 3`, core/server
  all-target Clippy, formatting, and diff checks. The final generated M5
  inventory and Rust drift gate pass at `191` rows, depth `55 / 50 / 86`, and
  exposure `22`. Ast-grep finds the sole function declaration plus the
  production and three parsed policy calls; the fourth policy call is inside
  an `assert!` token tree, so exact text census plus successful LSP definition
  lookup closes that parser boundary. No truth/generation surface changed.
- REVIEW GATE, 2026-07-29: the routine Sol architecture/API review returned
  `APPROVE`; the independent movement, accounting, concurrency, and kill-power
  audit returned `APPROVE`. Both reviewed the exact staged seven-file slice
  after topology D and reported no finding or source conflict.

#### R4-20 — rename-page apply and recovery transactions

- Move the rename apply CAS and pending-receipt recovery as two named
  operations in one purpose-specific DB child because they share the same
  cross-resource contract.
- Preserve the existing lock order exactly: acquire the owned repair
  projection session, take its locked projection, then acquire the DB mutex
  and begin `IMMEDIATE`. Hold both through DB receipt checks, controlled
  projection scans/write-or-restore, proof or recovery decision, and
  commit/rollback. Publish or clear pending artifact files only after commit,
  exactly where the current recovery path does so.
- Keep `page_on_connection` and the R4-20-only rename match helpers private to
  its DB child. Keep `capture_rename_page_title_on_connection` as the narrow
  `pub(crate)` helper shared by `repair_page_rename` and the later
  `repair_verification` child, as frozen below. No projection object, DB guard,
  or transaction token may escape the named operation.
- Existing crash-window and restore tests remain required. Add a barrier that
  proves a concurrent DB writer cannot enter while the operation is between
  its DB read and projection write/restore, and an order control that fails if
  DB locking moves before projection-session locking.
- GREEN floor: external literals `296 → 294`, production `7 → 5`, tests remain
  `289`.

##### R4-20 frozen slice contract

- **One child, two operations, unchanged facade graph.**
  `db/repair_page_rename.rs` owns the canonical
  `rename_page_title_cas_inner` and
  `recover_rename_page_title_apply_receipt` operations. The normal and
  test-hook `post_write` facades remain wrappers. The normal facade keeps its
  current name and signature. The `cfg(test)` facade keeps its current name
  and argument list, while its `G` hook bound deliberately gains `+ 'static`
  so the owned callback can be stored safely in the Tokio task-local test
  control. Its sole existing caller supplies a noncapturing closure; this
  changes no production surface. `repair::apply_rename_page_title` remains the
  recovery/apply orchestrator and calls the DB child for pending recovery. No
  second transaction implementation, same-name re-export, or direct
  dispatcher bypass is allowed.
- **Apply lock order and lifetime are byte-for-byte behavior.** Validate the
  manifest shape, review binding, and embedding before acquiring either
  resource. Then acquire the owned repair projection session, derive its
  locked projection, compute excluded paths, acquire `db.conn`, and execute
  `BEGIN IMMEDIATE`, in that order. Retain the projection session, projection
  handle, and DB guard through every target/collision/review read, projection
  scan and write, proof hook, restore, and `COMMIT`/`ROLLBACK`. Projection
  compensation remains before DB rollback. The existing error mapping remains
  exact: an original operation/hook error is returned only when projection
  restore and DB rollback both succeed; a commit error remains a vector DB
  error only when both compensations succeed; every compensation failure is
  `repair_apply_recovery_required`.
- **Recovery lock order and post-commit lifetime are also behavior.** Inspect
  the final/pending artifact and load the typed rollback before either lock.
  Then acquire the owned projection session and locked projection, acquire
  `db.conn`, and execute `BEGIN IMMEDIATE`. Retain both resource guards through
  capture, classification, optional restore, and `COMMIT`. Preserve the
  current lexical lifetime after `COMMIT`: publishing a committed pending
  receipt or clearing a retryable pending receipt occurs while both guards are
  still alive. The advisory projection acquisition stays fail-fast through
  `try_lock_exclusive`; this slice must not replace it with a blocking lock.
- **Artifact I/O crosses the boundary only as a typed domain seam.**
  `RepairArtifactStore` gains a narrow rename-recovery inspection result with
  exactly three states: completed final receipt, no recovery artifact, or
  pending with an optional verified receipt plus the typed rename rollback.
  The inspection method preserves final-receipt cleanup and bounded pending
  read/parse/verification. Two narrow store operations publish the rename
  pending receipt or clear it after commit. The DB child does not receive raw
  artifact paths, and generic `read_bounded_file`, `publish_no_replace`, or
  `sync_dir` do not become cross-module APIs.
- **Classifier behavior is unchanged.** Exact committed post publishes;
  database-pre plus projection-pre clears and retries; database-pre plus
  projection-post restores, clears, and retries; every other state rolls back,
  retains pending, and returns `repair_apply_recovery_required`. Both
  split-projection states—target-post/state-pre and
  state-post/target-pre—are explicitly unknown. This movement records but
  does not silently repair the existing recovery paths where receipt or
  post-match `?`, or a failed `COMMIT`, can return without an explicit
  rollback; that behavioral correction requires a separate reviewed slice.
- **Helper ownership has one narrow shared exception.** Move
  `page_on_connection`, embedding decode, apply projection restore, recovery
  rollback, and recovery-only database/projection match helpers private to the
  DB child. Keep shared receipt, non-target receipt, excluded-path, effect
  guard, and DB-digest helpers in `repair.rs`. R4-20 moves its three
  apply/recovery call sites for
  `capture_rename_page_title_on_connection` into
  `db/repair_page_rename.rs`; it does not eliminate them. R4-24 moves the
  remaining verification call site into `db/repair_verification.rs`.
  Therefore the capture helper remains a narrow `pub(crate)` seam shared by
  those two DB children. The earlier promise to make it private at R4-24 was
  stale; LSP reference closure corrected the ownership contract rather than
  widening R4-24 to duplicate or relocate the shared capture machinery.
- **RED controls carry the concurrency proof.**
  1. Hold `db.conn` and invoke apply and pending recovery against an invalid
     regular-file projection root under a bounded timeout. Correct
     projection-first order returns the projection-root error without waiting
     for DB; a DB-first mutation times out.
  2. A narrowly named `cfg(test)` checkpoint immediately before the apply
     projection write releases a prestarted contender on a multi-thread Tokio
     runtime. The checkpoint waits with a finite bound until that same
     contender has polled `db.conn.lock()` and observed it Pending; it must not
     merely spawn or schedule the contender. The existing after-target-write
     hook proves the contender still has not entered, then fails so the
     unchanged projection and DB compensation path is also exercised. After
     the operation returns and releases the guard, the test awaits the same
     contender and proves that it then enters. This paired interval kills a
     release-before-write and reacquire-after-write mutant without joining a
     blocked thread inside a synchronous hook.
  3. Restore has two narrowly named `cfg(test)` checkpoints: one immediately
     before the first target restore and one after the target has been restored
     but before state restoration. Apply compensation and recovery
     `RestoreRetry` tests use the same pending-future handshake at entry, prove
     the contender still has not entered at the midpoint, then prove it enters
     only after the operation returns. The production restore method remains a
     no-hook wrapper.
  4. A narrowly named `cfg(test)` recovery hook fires after `COMMIT` and before
     artifact publish/clear on both Publish and Clear paths. At that point the
     test first confirms the pending artifact still exists and the final
     artifact does not, then proves `db.conn.try_lock()` is blocked and a
     second owned repair projection session fails fast with
     `page_projection_locked`.
  5. Target-post/state-pre and state-post/target-pre fixtures both return
     `repair_apply_recovery_required` and retain the pending artifact.
  Existing title-rename controls remain unchanged and green.
- **Mechanical and topology gates.** Lower only the exact ratchet rows
  `post_write.rs: 15 → 14` and `repair.rs: 19 → 18`; no production raw lock
  appears outside `db/**`. The expected external census is `296 → 294`,
  production `7 → 5`, tests `289`. Preserve the M5 inventory at `191` rows,
  depth `55 / 50 / 86`, exposure `22`, and ambiguity `9`. Ast-grep closes the
  declaration/call shape; exact text census covers macro token trees; LSP
  definition/references and zero-error diagnostics close semantic routing.
- RED, 2026-07-29: the seven direct controls first failed `0 / 7` against the
  two explicit DB-child stubs. Lowering only the two ratchet rows produced the
  exact expected failures:
  `post_write.rs: direct .conn.lock() access increased 14 -> 15` and
  `repair.rs: direct .conn.lock() access increased 18 -> 19`.
- IMPLEMENTED, 2026-07-29: `db/repair_page_rename.rs` now owns the canonical
  apply and pending-recovery transactions. The unchanged `post_write` facades
  call the child, while `repair::apply_rename_page_title` retains orchestration
  and crosses the boundary through the three-state typed artifact inspection
  plus narrow publish/clear methods. Projection-session-before-DB ordering,
  compensation-before-rollback, post-commit guard lifetimes, classifier
  branches, error mapping, and the known no-explicit-rollback recovery caveat
  remain unchanged. Only test builds expose the write, restore-entry,
  restore-midpoint, and post-commit artifact checkpoints.
- GREEN, 2026-07-29: direct lock/order, compensation, artifact-lifetime, and
  split-state controls pass `7 / 7`; the unchanged title-rename policy suite
  passes `10 / 10`; and the exact external-access ratchet passes `3 / 3`.
  External literals are `296 → 294`, production `7 → 5`, and tests remain
  `289`. Core/server all-target Clippy with `-D warnings`, formatting, and the
  generated inventory check pass.
- LSP AND TOPOLOGY GATE, 2026-07-29: ast-grep finds exactly one declaration of
  each canonical operation in the DB child. Exact source census shows only the
  two unchanged apply facades and one recovery orchestrator production call;
  direct sibling tests call the canonical operations. LSP definition lookup
  from both production paths resolves to the DB-child definitions, references
  enumerate those paths and the direct controls, and all seven changed Rust
  files have zero error diagnostics. The final M5 inventory remains exactly
  `191 / 55-50-86 / exposure 22 / ambiguity 9`.
- REVIEW DELTA, 2026-07-29: Sol and the concurrency auditor correctly blocked
  the first GREEN because the canonical DB operations still accepted arbitrary
  checkpoint callbacks in production. The corrected canonical apply API now
  has only `db`, `manifest`, `rollback`, `page_root`, and `before_commit`; the
  recovery API has only `db`, `store`, `manifest`, and `page_root`. Whole
  `#[cfg(test)]` wrappers install owned checkpoints in a Tokio task-local
  around exactly one canonical future. Statement-level test lookups drive the
  direct controls, while production uses the normal write and unconditional
  restore methods. The FTS maintenance rationale remains beside the moved
  target-change accounting. This delta adds no production callback,
  same-name implementation, scanner exception, or inventory row.
- CHECKPOINT CONSUMPTION RED, 2026-07-29: the concurrency auditor found that a
  supplied checkpoint could remain installed without failing its control.
  The test-only scope now inspects the task-local `RefCell` after the canonical
  future returns and before scope exit; any still-`Some` hook panics with its
  stable field name. Two temporary deletion mutants proved the assertion is
  load-bearing: removing the restore-midpoint invocation failed the apply
  compensation control with
  `unconsumed rename page title test checkpoints: after_target_restore`, and
  removing the post-commit invocation failed the recovery publish control with
  `unconsumed rename page title test checkpoints:
  after_commit_before_artifact`. Each hook is taken before invocation, so an
  intentional hook error still counts as consumed. Both mutants were restored
  before the final GREEN rerun.
- TEST-FACADE ADJUDICATION, 2026-07-29: Sol approved the safe owned-callback
  option. The frozen contract now records the sole deliberate signature-bound
  delta: the `cfg(test)` facade keeps its name and argument list, while `G`
  gains `+ 'static` for Tokio task-local storage; production is unchanged.
  The existing midpoint-write test remains one test and preserves its original
  noncapturing invocation. A second independent fixture in that same test uses
  an owned `move` closure over `Arc<AtomicBool>`, proves the hook fired, and
  independently proves both the Page row/version and raw projection files
  were restored. The title-rename suite therefore remains `10 / 10` while
  exercising both accepted callback shapes.
- FINAL REVIEW GATE, 2026-07-29: Sol and the independent concurrency auditor
  both returned `APPROVE` after inspecting the amended contract and current
  diff. Root independently reran the title suite `10 / 10`, DB-child controls
  `7 / 7`, external-access ratchet `3 / 3`, Rust M5 gate `1 / 1`, and reader
  inventory at `191 / 55-50-86 / exposure 22`; LSP resolves both production
  callers to the DB child and reports zero errors in the child, facade, and
  amended title test.

#### R4-21 — regenerate-page projection compensation

##### R4-21 frozen slice contract

- **One declaration and a path-preserving re-export.** Move the complete
  `post_write::regenerate_page_projection_cas` body to
  `db/repair_page_regenerate.rs` as the sole `pub(crate)` declaration of that
  exact name. Replace the old declaration with a `pub(crate) use` re-export,
  so `repair::apply_repair_with_pages_inner` keeps its sole production call
  through `crate::post_write::regenerate_page_projection_cas`. The
  task-specific module name is deliberate: it may not become a generic home
  for R4-22's stale-projection protocol. No forwarding wrapper, second
  same-name declaration, inherent `MemoryDB` method, generic
  connection/transaction API, or dispatcher bypass.
- **Exact precheck and split-snapshot behavior.** Preserve this order:
  manifest target/writer/mutation validation; `db.get_page` precheck and
  version/scope check; rollback paths and target-path validation; outer
  `db.conn.lock`; `projection_page_row_on_connection` and
  `database_content_digest`; then
  `KnowledgeProjectionWrite::with_repair_lock` while retaining the DB guard.
  The Page rendered by `write_page` remains the value read before the outer
  lock, while rollback capture remains based on the later locked page row.
  This pre-existing split snapshot and the gap between the two reads are
  preserved, not repaired or revalidated in this movement slice.
- **Exact lock topology and lifetime.** `get_page` must remain before the outer
  guard because moving it inside would re-enter the non-reentrant DB mutex.
  Keep the synchronous `with_repair_lock` closure under the async DB guard;
  do not use `spawn_blocking`, introduce an await in the closure, replace it
  with an owned session, or drop/reacquire either guard. The DB guard and
  process/file-backed repair projection lock remain live through pre-scan,
  target exclusivity, write, post-scan/capture, both compensation branches,
  proof construction, `before_commit`, and the operation return. The
  operation deliberately remains non-transactional: no `BEGIN` or `COMMIT`.
- **Two distinct compensation branches and exact errors.** Preserve the first
  restore branch for write/post-scan/capture/non-target errors and the second
  for after-target-receipt, write-unproven, or `before_commit` errors. A
  successful restore returns the original error unchanged. A failed restore
  returns
  `WenlanError::VectorDb("{original}; repair projection rollback failed: {restore}")`
  in both branches. Errors before `write_page` do not restore.
  `before_commit` remains a receipt-preparation hook, runs only after the
  complete proof exists, and runs under both guards; it is not a database
  commit. Preserve the caller consequence: only
  `repair_apply_recovery_required` retains a pending apply receipt, so this
  operation's ordinary and composed errors continue to abort it. Changing
  that behavior requires a separately approved semantic slice.
- **Narrow seams only.** Leave rollback-path, capture, receipt, digest, and
  restore helpers in `repair`; this slice moves the owning critical section,
  not those transitional helpers. Import `post_write::RepairWriteProof` and
  construct it only with `RepairWriteProof::from_parts`. Any direct-test
  controls are whole-item `#[cfg(test)]`, task-local, owned, and consumption
  checked. They may not add a production callback parameter or expose the
  connection, either guard, a generic projection session, or filesystem
  capabilities.
- **RED controls that bite.** Add direct child controls with bounded timeouts
  that prove: the normal path cannot self-deadlock; the DB row/digest snapshot
  is taken before projection work; DB-before-projection lock order; one DB
  contender remains `Pending` after the snapshot, through target write and
  proof preparation, and enters only after operation return; and the proof's
  `post_apply_db_digest` is the pinned pre-projection digest. Independently
  force a post-write result-branch error and a `before_commit`
  completion-branch error, asserting byte-exact target/state/non-target
  restoration and the unchanged original error. Force restore failure in
  each branch and assert the exact composed `VectorDb` mapping, plus one
  apply-level control proving the pending receipt is aborted. Temporary
  mutations must remove each restore call, shorten the DB-guard lifetime,
  move the proof callback, and remove a checkpoint invocation; each matching
  control must fail, including fail-loud unconsumed-checkpoint names.
- **Behavior and topology gates.** Keep the existing four
  `page_projection_*` integration tests green. Lower only the
  `post_write.rs` raw-lock ratchet `14 → 13`; require external literals
  `294 → 293`, production `5 → 4`, and tests `289`. Regenerate only M5 source
  addresses and retain exactly `191` rows, depth `55 / 50 / 86`, exposure
  `22`, and ambiguity `9`. Ast-grep must find one function declaration; LSP
  definition/references must resolve the sole production path to the DB child
  with zero error diagnostics.
- RED, 2026-07-29: the five direct controls first failed `0 / 5` against the
  explicit DB-child stub. Four stopped at `R4-21 RED stub`; the apply-level
  control failed loud with the unconsumed write/restore checkpoint names.
  Lowering only the ratchet row produced the exact expected
  `post_write.rs: direct .conn.lock() access increased 13 -> 14` failure.
- IMPLEMENTED, 2026-07-29: `db/repair_page_regenerate.rs` now owns the sole
  canonical declaration and complete critical section; `post_write` preserves
  the old path with a `pub(crate) use`. The split Page/row snapshot, DB-before-
  projection lock topology, both compensation branches, exact error mapping,
  and apply receipt behavior are unchanged. `RepairWriteProof::from_parts`
  replaces field construction, and only test builds expose owned,
  consumption-checked task-local checkpoints.
- GREEN, 2026-07-29: the direct child suite passes `5 / 5`; the four named
  `page_projection_*` integration controls pass; and the exact external-access
  ratchet passes. External literals are `294 → 293`, production `5 → 4`, and
  tests remain `289`. Core/server all-target Clippy with `-D warnings`,
  formatting, and the generated inventory checks pass.
- MUTATION GATE, 2026-07-29: removing either restore call failed its matching
  branch with unconsumed restore checkpoints; shortening the DB-guard lifetime
  let the contender enter before operation return; moving `before_commit`
  before the target write failed `target_write_seen`; and removing the
  DB-snapshot checkpoint failed with its exact unconsumed name. Every mutant
  was restored before the final GREEN rerun. The apply-level composed
  restore-failure `VectorDb` path independently proves both pending and final
  receipts remain absent.
- LSP AND TOPOLOGY GATE, 2026-07-29: ast-grep finds exactly one declaration in
  the DB child. LSP references enumerate the unchanged sole production call
  through the `post_write` re-export plus the direct tests; the child, facade,
  and direct-test files report zero error diagnostics. The regenerated M5
  inventory remains exactly `191 / 55-50-86 / exposure 22 / ambiguity 9`;
  only source addresses moved.
- FINAL REVIEW GATE, 2026-07-29: Sol and the independent concurrency auditor
  both returned `APPROVE` after inspecting the current diff against the frozen
  contract. Root independently reran the direct suite `5 / 5`, each of the
  four named projection integration tests, the external-access ratchet
  `3 / 3`, Rust M5 gate `1 / 1`, and reader inventory at
  `191 / 55-50-86 / exposure 22`.

#### R4-22 — stale-projection quarantine and recovery

##### R4-22 frozen slice contract

- **One protocol module and two production declarations.** Add
  `db/repair_stale_projection.rs`; the stale-page apply journal, quarantine,
  and recovery state machine remain one protocol. It owns exactly these
  production operations:
  `quarantine_stale_page_projection_cas_with_apply_journal` and
  `recover_stale_page_projection_apply_receipt`. `post_write` retains the
  existing apply path through a `pub(crate) use` re-export only. `repair`
  imports the recovery operation directly and keeps its two existing calls
  inside generic apply-receipt recovery. No production wrapper, second
  same-name declaration, inherent `MemoryDB` method, generic connection or
  projection-session API, or dispatcher bypass.
- **Test facades move with their owner.** The child also owns the four existing
  `cfg(test)` facades
  `quarantine_stale_page_projection_cas`,
  `quarantine_stale_page_projection_cas_with_before_pin`,
  `quarantine_stale_page_projection_cas_with_after_pin`, and
  `quarantine_stale_page_projection_cas_with_before_source_stage`;
  `post_write` re-exports them only under `cfg(test)`, preserving every caller
  path and argument list. The `before_pin`, `after_pin`, and
  `before_source_stage` hook bounds deliberately gain `+ 'static` for safe
  owned Tokio task-local storage. Convert the four current borrowed callers
  to owned `PathBuf`/byte captures with `move`. Do not add `'static` to the
  real journal callback `J` or receipt callback `F`: both production closures
  intentionally borrow orchestrator state.
- **Apply validation and lock order are exact.** Preserve: validate the
  target/writer/global scope/mutation; validate captured source and quarantine
  paths against the plan rollback; acquire the DB mutex; run the owner-absence
  query with the exact `repair projection owner CAS` error mapping and return
  `repair_target_stale` if an owner exists; capture the DB digest; then enter
  synchronous `KnowledgeProjectionWrite::with_repair_lock` while retaining the
  DB guard. Under both guards capture the current projection, validate its
  expected receipt and exact stale ownership, scan non-target state, persist
  the journal, pin, quarantine, rescan/capture, prove effects, construct
  `RepairWriteProof::from_parts`, and invoke `before_commit`. Do not await,
  spawn, switch to an owned session, or drop/reacquire either guard inside the
  projection phase.
- **The journal is the locked dynamic snapshot.** The journal payload is the
  dynamically captured `before` state under both guards, not the plan-time
  rollback. Invoke `J(&before)` after capture/ownership/non-target validation
  and before pin or any quarantine mutation. The existing orchestrator closure
  must durably publish the journal through its `.pending → final` window first,
  then create the empty pending apply receipt. Only after `J` succeeds may pin
  begin. This ordering protects a prior repair from the same plan and is not
  interchangeable with persisting the original rollback.
- **Mutation-started split and compensation stay distinct.**
  `pin_stale_page_projection` sets `mutation_started = true` as soon as pin
  succeeds, even before the first hard link. Its internal compensations may
  reset the flag to false after restoring their own partial work. An outer
  error with `mutation_started == false` returns the original error without a
  second snapshot restore. An outer error with the flag still true restores
  the dynamic `before`; restore success returns the original error, while
  restore failure returns
  `WenlanError::VectorDb("{error}; stale projection rollback failed: {rollback_error}")`.
  Preserve zero-mutation stale, internally self-cleaned failure, partial
  mutation, and rollback-required failure as separate states.
- **Apply artifact boundaries are unchanged.** Journal persistence, pending
  receipt creation, and pending receipt preparation occur inside both DB and
  projection guards through `J` and `F`. After the canonical operation returns
  successfully, both guards are released before the orchestrator publishes
  the prepared pending receipt and then clears the journal. On apply error,
  retain an existing pending receipt whenever the journal still exists or the
  ordinary recovery predicate requires it; abort only otherwise. Never clear
  the journal before publishing the receipt.
- **Typed recovery-artifact seam only.** Keep paths and generic filesystem
  primitives private to `repair`. Add
  `StalePageProjectionRecoveryJournal::{Absent, Present(StoredRollbackArtifact)}`
  and
  `StalePageProjectionRecoveryArtifactUpdate::{ClearPendingOnly,
  PublishPendingAndClearJournal, ClearPendingAndJournal}` plus two narrow
  `RepairArtifactStore` methods that inspect/promote the journal and apply one
  typed update. The two enums and two methods are `pub(crate)`, while their
  fields, paths, path builders, and filesystem primitives remain private to
  `repair`; the child receives no `Path` or generic filesystem capability.
  They preserve the existing pending/final publication, cleanup ordering,
  sync behavior, and error strings. Do not expose `publish_no_replace`,
  pending/final paths, or a generic publish/delete API. The sole-use
  `stale_page_projection_post_target_receipt` helper becomes `pub(crate)`;
  `capture_stale_page_projection_current` is already `pub(crate)` and no
  broader repair helper visibility changes.
- **Recovery lock and pre-classifier behavior are exact.** The child receives
  typed `&RepairArtifactStore`, manifest, plan rollback, optional page root,
  and the already parsed `Option<RepairApplyReceipt>`; it does not inspect the
  raw pending path to decide semantics. Acquire the DB mutex, repeat the
  owner-absence CAS with `repair database` mapping and return
  `repair_apply_recovery_required` if an owner exists. While retaining the DB
  guard, inspect/promote the journal. If it is absent and the current
  projection exactly equals the plan rollback, apply `ClearPendingOnly` and
  return retry. Otherwise use the journal rollback when present, falling back
  to the plan rollback, and set
  `restore_post = pending_receipt.is_none()` exactly. Enter
  `with_repair_lock` only around `recover_stale_page_projection`; deliberately
  convert every error from that projection recovery to `Unknown`. The
  projection guard ends before artifact publication/cleanup, while the DB
  guard remains live through the typed update and function return.
- **Recovery state table is closed.**
  `Some(valid receipt) + Post + exact post-target digest` publishes pending to
  final and then clears the journal, returning that receipt.
  `Original`, with either `Some` or `None`, clears pending first and then the
  journal, returning retry. `Unknown`, a nonterminal recovery state, a Post
  digest mismatch, or every other combination returns
  `repair_apply_recovery_required` and preserves all artifacts. With
  `pending_receipt == None`, `restore_post = true` restores a recoverable Post
  state to Original before this table; with `Some`, it remains Post for exact
  receipt publication. The generic already-final receipt cleanup in
  `recover_apply_receipt` remains outside this DB critical section and is not
  moved.
- **Test controls have teeth without production callbacks.** Use one owned,
  consumption-checked `cfg(test)` Tokio task-local for
  `after_db_snapshot_before_projection_lock`,
  `before_journal_persist`, `after_journal_persist_before_pin`,
  `after_pin_before_link`, `before_source_stage`,
  `before_snapshot_restore`, `after_snapshot_restore`,
  `after_recovery_state_before_artifact`, and
  `after_recovery_artifact`. The three legacy hook facades map to their exact
  historical sites. The after-journal checkpoint can inject a test-only error
  after canonical `J` has durably created the journal and empty pending
  receipt but before pin. Canonical production parameters remain only the
  real `J` and `F` callbacks.
- **RED and mutation floor.** Direct child controls must cover: normal apply
  with the pinned DB digest and a DB contender `Pending` through journal,
  quarantine, proof, and return; DB-before-projection lock order; owner
  appearing after prepare for both apply and recovery; journal failure before
  publish versus crash after journal publish/before pin; a two-manifest
  canonical apply in which repair A completes, repair B is interrupted by the
  after-journal checkpoint after its real journal plus pending receipt exist,
  and B recovery preserves A; no-mutation and internally self-cleaned errors
  versus restore-required and restore-failure mappings; journal-absent exact
  Original retry; the complete
  `Original`/matching-`Post`/mismatched-`Post`/`Unknown` table; and the
  `restore_post` split. A recovery contender remains blocked through typed
  artifact publish/clear and enters only after return. Use a separately opened
  `.projection.lock` file for lock-order contention; never hold the process
  mutex in another thread under a Tokio timeout. Required temporary mutations
  move journal persistence after pin, replace canonical `J(&before)` with
  `J(rollback)`, remove the owner CAS, remove the `mutation_started` split or
  snapshot restore, shorten the DB-guard lifetime, move the proof callback,
  invert `restore_post`, propagate recovery error instead of `Unknown`, or
  clear artifacts from an unmatched/Unknown arm; each matching control and
  each deleted checkpoint invocation must fail.
- **Generic-final boundary has a source tooth.** A narrow source-shape guard
  requires `recover_apply_receipt` to complete the already-final receipt
  cleanup branch before either stale-recovery call, and forbids
  `repair_stale_projection.rs` from loading a final apply receipt or performing
  that generic pending/journal cleanup. This guards the deliberate outside-DB
  boundary without exposing a new production callback or raw artifact path.
- **Existing behavior and topology gates.** Keep all `33`
  `repair_plan::deterministic::tests::stale_page_projection_*` controls green,
  plus the exact lower-level export quarantine/recovery/ancestor-swap controls
  and pending-retention predicate test. Lower only the raw-lock rows
  `post_write.rs 13 → 12` and `repair.rs 18 → 17`; require external literals
  `293 → 291`, production `4 → 2`, and tests `289`. Regenerate only moved M5
  source addresses and retain exactly `191` rows, depth `55 / 50 / 86`,
  exposure `22`, and ambiguity `9`. Ast-grep must find one production
  declaration for each operation; LSP must resolve one apply production path,
  two recovery callers, all six legacy test-facade calls, and zero error
  diagnostics.
- RED, 2026-07-29: the direct child suite first produced `1 / 8` green against
  explicit apply/recovery stubs; only the source-boundary tooth passed, while
  the seven behavior controls failed at the stub. Lowering only the two
  raw-lock rows produced the exact expected `post_write.rs 12 -> 13` and
  `repair.rs 17 -> 18` failures.
- IMPLEMENTED, 2026-07-29: `db/repair_stale_projection.rs` now owns the two
  canonical production declarations plus the four test facades and one
  consumption-checked task-local checkpoint set. `post_write` retains the
  production apply path and test paths as re-exports; `repair` imports recovery
  directly and retains both generic-recovery calls. The DB-first critical
  sections, locked dynamic journal, mutation-started compensation split, and
  closed recovery table are unchanged. `RepairArtifactStore` exposes only the
  two typed stale-projection recovery operations and their closed enum inputs.
- GREEN, 2026-07-29: the direct child suite passes `8 / 8`; the existing
  stale-projection family passes `33 / 33`; the three lower-level stale
  projection race controls, duplicate-state rejection, ancestor-swap canary,
  and pending-retention predicate all pass. The external raw-lock census is
  exactly `291`, split into production `2` and tests `289`; the exact ratchet
  passes with only `post_write.rs 13 -> 12` and `repair.rs 18 -> 17`.
  Core/server all-target Clippy with `-D warnings` and formatting pass.
- MUTATION GATE, 2026-07-29: direct controls killed and then restored every
  required temporary mutation: `J(&before) -> J(rollback)`; journal persistence
  moved after pin; both owner CAS branches removed; both DB guards shortened;
  the after-journal checkpoint invocation removed; the `mutation_started`
  split removed; snapshot restore removed; the proof callback moved outside
  both guards; `restore_post` inverted; projection recovery errors propagated
  instead of becoming `Unknown`; and unmatched/Unknown arms cleared artifacts.
  The proof-location control uses a second DB contender released inside the
  callback, so mutex fairness cannot make an early unlock pass accidentally.
  No mutant remained for the final direct-suite rerun.
- LSP AND TOPOLOGY GATE, 2026-07-29: ast-grep finds exactly one production
  declaration for each operation. LSP resolves the sole apply production call,
  both recovery calls, and all six legacy facade calls to the child; the child,
  `repair`, and `post_write` report zero error diagnostics. The regenerated M5
  inventory changes source addresses only and remains exactly
  `191 / 55-50-86 / exposure 22 / ambiguity 9`.
- FINAL REVIEW GATE, 2026-07-29: Sol and the independent concurrency auditor
  both returned `APPROVE` against the frozen contract. Root found that the
  generic-final source tooth initially ordered only branch entry before stale
  dispatch, strengthened it to require
  `branch entry < return Ok(Some(receipt)) < first stale dispatch`, reran the
  exact and full direct suites, and received both reviewers' renewed
  `APPROVE`. Root independently reran direct `8 / 8`, the preflight-confirmed
  stale family `33 / 33`, the external-access ratchet `1 / 1`, Rust M5 gate
  `1 / 1`, reader inventory `191 / 55-50-86 / exposure 22 / ambiguity 9`,
  core/server all-target Clippy with `-D warnings`, formatting, and
  `git diff --check`.

#### R4-23 — current repair target dispatcher

##### R4-23 frozen slice contract

- **One new child and one production entry.** Add
  `db/repair_target_receipt.rs` with the sole public production operation
  `read_current_repair_target_receipt(db, manifest, rollback, page_root)`.
  It owns only private projection-branch logic and returns the owned
  `(RepairDigest, u64)`. `recover_apply_receipt` imports it directly at both
  existing non-stale call sites; delete `target_receipt_current` rather than
  retaining a wrapper or same-name declaration in `repair`. Do not add a read
  operation to the already-frozen R4-21 or R4-22 children.
- **Branch before locking.** The public dispatcher selects target/writer first.
  Every non-`PageProjection` target delegates directly to R4-17
  `db.read_repair_target_receipt(manifest.target()).await`; it must not acquire
  `db.conn` first, recreate `repair_target_receipt_on_connection`, inspect
  `page_root`, or parse the supplied rollback. Both projection branches enter
  private child logic and acquire the DB mutex before any branch-local root
  validation, rollback parsing, query, or filesystem/projection capture.
  Moving the DB lock above dispatch would self-deadlock the default branch
  when R4-17 locks it again.
- **Ordinary projection order is exact and intentionally has no projection
  lock.** Preserve DB mutex → require `page_root` with
  `page projection repair root unavailable` → `projection_rollback_paths` →
  existing `capture_page_projection_on_connection` (DB page row followed by
  raw filesystem capture) → `target_receipt(&current)` → count `1`. The DB
  guard remains live through digest construction and function return. Do not
  add `with_projection_lock`, `with_repair_lock`, an owned repair session, or
  split the page-row and filesystem snapshots across guards; the current raw
  filesystem read is protected only by that DB lifetime, and changing it would
  alter contention and error behavior.
- **Stale projection order is exact.** Preserve DB mutex → require the same
  root → owner-absence `SELECT 1 FROM pages WHERE id=?1 LIMIT 1` using the
  existing `repair::database_error` mapping → owner returns
  `repair_target_stale` → drop the query row → `stale_page_projection_paths` →
  existing `capture_stale_page_projection_current` while still holding DB
  (therefore DB → projection lock) → `target_receipt(&current)` → count `0`.
  Promote only `repair::database_error` to `pub(crate)`. Do not substitute an
  R4-22 apply/recovery operation, repair lock, journal, artifact store, or
  rollback bytes for the dynamic capture.
- **Error priority and outer artifact boundaries remain unchanged.** The DB
  lock precedes missing-root and malformed/unsafe rollback errors in both
  projection branches. Stale root validation precedes the owner query, and
  owner presence precedes stale rollback parsing or projection-lock
  contention. Preserve all current DB, missing-owner, root, unsafe-path,
  size, UTF-8, and rollback-shape error strings. Both DB guards end when the
  child returns; generic pending publication/removal and directory sync remain
  afterward in `recover_apply_receipt`. R4-22's earlier stale-writer dispatch
  remains before both generic call sites, so the new stale arm stays
  deliberately unreachable through the current production route and is
  protected directly rather than replacing specialized recovery.
- **No raw capability crosses the seam.** The child receives only
  `MemoryDB`, manifest, rollback, and optional page root; it returns only the
  owned digest/count pair. No `Connection`, mutex guard, projection session,
  callback, `RepairArtifactStore`, artifact path, pending receipt, or generic
  query/transaction capability enters or exits the public operation.
- **Owned, consumption-checked test checkpoints only.** One `cfg(test)` Tokio
  task-local checkpoint set may observe immediately after a projection DB
  lock and before validation, immediately before projection/filesystem
  capture, and immediately after capture before return. No production
  callback, static caller bound, global mutable hook, or sleep-based
  synchronization. A prestarted DB contender released by the first checkpoint
  must be observed `Pending` before and after capture and may enter only after
  child return.
- **Branch-complete direct controls.** The sibling
  `db/repair_target_receipt_test.rs` covers:
  default digest/count/error equivalence to R4-17, wrong-scope handling, bogus
  projection rollback/root inputs being ignored, and a bounded timeout against
  double-lock self-deadlock; ordinary dynamic digest/count `1`, missing page,
  missing root, malformed and unsafe rollback, DB-held validation ordering,
  and the DB contender across page-row plus filesystem capture; stale dynamic
  digest/count `0`, missing root, malformed rollback, DB-held validation
  ordering, and the same full-lifetime contender across owner check plus
  projection capture. Holding a separately opened
  `.wenlan/.projection.lock` file must not affect ordinary capture; for stale,
  an existing owner plus that lock must still return `repair_target_stale`
  before contention, while no owner must fail fast with
  `page_projection_locked`. Never hold the process-global projection mutex in
  another thread under a Tokio timeout.
- **Source and mutation teeth.** A narrow source/AST guard requires exactly
  one public child dispatcher; its default branch calls R4-17; ordinary and
  stale branches retain their exact existing capture helpers; the child
  contains no repair/projection session, artifact store, or generic
  publish/delete primitive; `repair.rs` retains the earlier specialized stale
  dispatch and only the two existing non-stale child calls. Temporary
  mutations must fail for: lifting the DB lock above dispatch; bypassing
  R4-17; moving either root/parser before the projection DB lock; dropping or
  reacquiring DB before either capture; deleting a checkpoint; adding an
  ordinary projection/repair lock; removing or delaying the stale owner CAS;
  substituting rollback for dynamic stale capture; substituting an R4-22
  operation; and swapping counts `0 / 1`.
- **Existing behavior and topology gates.** Keep R4-17
  `repair_receipt_matches_connection_helper_and_preserves_scope_guard`, both
  default recovery controls
  `apply_recovers_committed_receipt_after_unrelated_background_write` and
  `apply_discards_precommit_partial_receipt_and_retries`, plus
  `page_projection_invalid_pending_with_changed_target_fails_closed` green.
  Lower only `repair.rs` raw-lock baseline `17 → 16`; require external
  literals `291 → 290`, production `2 → 1`, tests `289`, while
  `post_write.rs` remains `12`. Regenerate M5 source addresses only and retain
  exactly `191` rows, depth `55 / 50 / 86`, exposure `22`, ambiguity `9`.
  Ast-grep must find one child declaration and none in `repair`; LSP must
  resolve both recovery calls to the child, the default branch to R4-17, and
  zero error diagnostics in child, test sibling, `repair`, `db.rs`, and the
  drift guard.
- RED, 2026-07-29: the explicit child stub compiled and the qualified filter
  listed exactly eight direct tests. The direct suite then failed `0 / 8` on
  the unimplemented dispatcher, unconsumed checkpoints, and retained
  `target_receipt_current`; lowering only the `repair.rs` ratchet row failed
  exactly `16 -> 17`.
- GREEN, 2026-07-29: `db/repair_target_receipt.rs` now owns the sole production
  dispatcher. Default targets branch before locking and delegate to R4-17;
  ordinary projection retains DB-only capture and count `1`; stale projection
  retains owner absence before dynamic DB-to-projection capture and count `0`.
  `recover_apply_receipt` has exactly two child calls, the old wrapper is gone,
  the specialized stale recovery dispatch remains earlier, and only
  `database_error` widened to `pub(crate)`.
- MUTATION, 2026-07-29: temporary lock-above-dispatch, R4-17 bypass,
  parser/root-before-lock, DB reacquire-before-capture, deleted checkpoint,
  ordinary projection-lock addition, stale-owner removal, stale-owner delay,
  rollback-for-dynamic-stale substitution, R4-22 substitution, and `0 / 1`
  count-swap mutants each failed its direct behavior or source tooth and were
  restored. The two contender tests remained bounded; no timeout escaped its
  stated limit.
- VERIFIED, 2026-07-29: direct controls pass `8 / 8`; the four frozen existing
  controls pass `4 / 4`; the exact ratchet plus its two positive controls pass
  `3 / 3`. External literals are exactly `290`, split into production `1` and
  tests `289`; `repair.rs` is `16`, `post_write.rs` remains `12`. The generated
  M5 inventory changed addresses only and passes at exactly
  `191 / 55-50-86 / exposure 22 / ambiguity 9`. Ast-grep finds one child
  declaration and no old wrapper; LSP resolves both callers to the child and
  its default branch to R4-17, with zero error diagnostics in the five frozen
  Rust files. Core/server all-target Clippy with `-D warnings` passes.
- FINAL REVIEW GATE, 2026-07-29: Sol and the independent concurrency auditor
  both returned `APPROVE` against the frozen contract. Root found that the
  runtime validation-order control covered malformed rollback but not the
  separately required missing-root case, added ordinary and stale
  missing-root checkpoints that assert the DB mutex is already held, and
  reran the exact control plus direct `8 / 8`; both reviewers inspected that
  final diff. Root also reran the four existing controls `4 / 4`, all three
  raw-lock teeth `3 / 3`, Rust M5 gate `1 / 1`, reader inventory
  `191 / 55-50-86 / exposure 22 / ambiguity 9`, LSP definition/reference and
  file diagnostics, ast-grep topology, formatting, and `git diff --check`.

#### R4-24a — repair verification atomic-operation movement

- Add `db/repair_verification.rs` with one `pub(crate)` free operation,
  `record_repair_verification_atomic(db, input)`, and no wrapper, re-export,
  generic executor, transaction callback, or R4-25 capability seam.
  `RepairVerificationAtomicInput<'a>` is its one typed input boundary. It
  carries references to the already-loaded `RepairArtifactStore`,
  `RepairManifest`, `RepairApplyReceipt`, and `VerifyRepairRequest`; prior
  verified tag targets; optional ordinary and rename rollback values; optional
  page root; verified-at epoch; and an optional **borrowed**
  `OwnedRepairProjectionSession`. It carries no connection, mutex guard,
  transaction, pending path, callback, or arbitrary query capability.
- Keep the public API, artifact ownership, and preflight in `repair.rs`, in
  this exact order: support/time and manifest checks; manifest-operation and
  tag-record locks; prior tag targets; apply receipt plus legacy/already
  verified branches; typed rollback loads; report and unlocked page
  prevalidation; rename owned-session acquisition; then the child call.
  Existing-receipt and legacy recovery remain caller-side and may clear
  pending without entering the child.
- The child acquires `db.conn`, executes the literal `BEGIN IMMEDIATE`, and
  preserves the existing `repair verify begin: ...` mapping. It then preserves
  the exact recheck order: current DB reports; tag-record set on the held
  connection; deterministic target; branch-specific target/non-target state;
  verification draft/digest; final current-page receipt and durable receipt
  persistence under the applicable projection lock; then the literal
  `COMMIT` with the existing `repair verify commit: ...` mapping. The child
  returns only after commit; the caller then clears pending.
- Moving pending cleanup to the caller intentionally releases the DB mutex
  before that filesystem cleanup. Manifest/tag locks and the caller-owned
  rename session remain live through cleanup because the session crosses the
  child by borrow, not by value. This is the one explicit lock-lifetime
  boundary change in the slice; SQL, transaction, validation, artifact, and
  recovery behavior remain unchanged.
- Preserve the literal best-effort `ROLLBACK` only for pre-commit body errors.
  Do not introduce `TransactionBehavior`, RAII transaction conversion,
  cancellation claims, or commit-failure cleanup in this movement slice.
  Verification receipt persistence still precedes DB commit: a
  persistence-success/commit-failure may leave the durable receipt plus the
  pending apply link, and retry loads that receipt then clears pending. A
  persistence/recheck failure retains pending and reports no completed
  receipt. This is an at-least-once terminal-receipt model, not a
  cross-filesystem transaction.
- Promote exactly the five single-use seams required by the child:
  `RepairArtifactStore::persist_verification_receipt`,
  `validate_current_db_receipts`,
  `validate_deterministic_target_resolved`,
  `validate_current_page_receipts_locked`, and
  `validate_current_page_receipts_on_repair_projection`. All other helper
  locations and visibilities remain as they stand after R4-23, including the
  shared `pub(crate)` rename capture helper corrected above.
- Preserve rename, stale-projection, ordinary-projection, and nonprojection
  branches independently. Do not route same-transaction reads through R4-17's
  separately locking method and do not generalize the operation into a
  transaction callback. Rename keeps its one caller-owned session throughout.
  Ordinary and stale each keep the current first DB-to-projection lock for
  target/non-target proof, release it, and later take the separate final
  projection lock for page-receipt validation plus receipt persistence; do not
  merge or widen those intervals.
- Preserve the existing caller-side post-session test hook unchanged. Re-run
  the existing branch and conflict controls, including rename session reuse,
  pinned-root ancestor swap, stale projection, ordinary projection,
  nonprojection, crash-window pending cleanup, existing-receipt retry, stale
  reports, changed target/tag set, and changed projection non-target state.
  R4-24a adds no fault injection, checkpoint framework, or production behavior
  beyond the explicit DB-mutex release point above.
- Source/LSP/AST teeth require one child operation, one production caller, one
  raw child DB lock, literal
  `BEGIN IMMEDIATE -> rechecks/persist -> COMMIT`, no RAII transaction,
  pending cleanup, artifact loading, outer locks, or production callback in
  the child, and caller order `child return -> clear pending` with no await or
  session drop between them. The error arm retains best-effort `ROLLBACK`;
  persistence stays inside the applicable projection lock and before
  `COMMIT`. LSP must resolve the caller and all five promoted helpers and
  report zero errors in every changed Rust file.
- GREEN floor: external literals `290 → 289`, production `1 → 0`, tests remain
  `289`; `repair.rs` is `16 → 15` and `post_write.rs` remains `12`. At this
  point the production raw-capability census is zero and its RED mutation
  control becomes a zero-baseline prohibition. Regenerate source addresses
  only; the M5 inventory remains
  `191 / 55-50-86 / exposure 22 / ambiguity 9`.
- RED, 2026-07-29: the exact new source-contract test failed at compile time
  with exit `101` because `db/repair_verification.rs` did not exist; no test
  body ran. This proved the required bounded child/module was absent before
  implementation.
- GREEN, 2026-07-29: one typed child now owns the sole production raw DB lock,
  manual transaction, four target branches, applicable final projection lock,
  receipt persistence, rollback arm, and commit. The caller retains preflight,
  artifact loading, outer locks, the borrowed rename session, the existing
  post-session hook, and immediate post-return pending cleanup. Only the five
  named helper visibilities were promoted; the shared rename capture helper
  remains `pub(crate)` for its four cross-module callers.
- Controls and gates: the R4-24a source/mutation teeth pass `3 / 3`, including
  explicit mutants for persistence outside the page-root lock and persistence
  before final locked validation in both page branches. The complete
  verification filter passes `23 / 23`; all external-access teeth pass
  `3 / 3`; the exact Rust M5 gate passes `1 / 1`; and the generated sweep
  remains `191 / 55-50-86 / exposure 22`. Core/server all-target Clippy with
  `-D warnings`, formatting, and `git diff --check` pass.
- LSP resolves the caller to the child and reports exactly one production
  call. Its reference closure caught and preserved the three rename-apply
  callers plus the moved verification caller; error diagnostics are empty for
  the child, its test, and `repair.rs`. The structural declaration check finds
  exactly one atomic child operation.
- FINAL REVIEW GATE, 2026-07-29: both independent Sol reviewers returned
  **APPROVE** against the final diff. The first confirmed strict
  `lock -> final validation -> persist -> closure end` teeth plus explicit
  mutation controls. The concurrency reviewer independently reconciled all
  three mutants with the production transaction/lock order and found no
  remaining semantic drift.

#### R4-24b — verification crash/lock teeth after movement

- Start only from the committed, fully green R4-24a baseline. Write the RED
  controls before adding their seam; do not mix these changes into the
  movement commit.
- Replace the one ad-hoc projection-session test hook with a scoped
  `#[cfg(test)]` task-local verification control. It has typed, consumed
  checkpoints after rename-session acquisition, after begin, immediately
  around receipt persistence while the applicable lock is still held, and
  after commit/before pending clear, plus one consumed commit-failure mode
  immediately after successful persistence and before literal `COMMIT`. Test
  hooks receive no DB/projection/store capability and no callback crosses the
  production child boundary. Persistence failure is not faked by a fault
  enum: at the pre-persist checkpoint the test creates a real conflicting
  final-receipt filesystem object after the caller's earlier receipt lookup,
  so the unchanged store write itself fails.
- Direct controls cover all four target branches; report, target, and
  non-target conflicts before persistence; real receipt-persistence failure
  and rollback; commit failure with the exact error plus durable
  receipt/pending retry cleanup; and the post-commit/pre-clear state. At the
  persistence checkpoint a DB contender remains pending and the applicable
  rename/stale/ordinary projection lock is contended. At the post-commit
  checkpoint DB is acquirable while final receipt and pending link both exist,
  and manifest/tag locks plus the rename session when applicable remain held.
- Keep production semantics byte-for-byte equivalent to the R4-24a baseline.
  Test-only commit failure exercises the existing at-least-once hazard; it
  does not add rollback-on-commit-error, RAII, or cancellation hardening.
  Source teeth reject test controls outside `#[cfg(test)]`, raw capability
  escape through a checkpoint, altered production ordering, or an unconsumed
  configured checkpoint/fault. Raw-capability and M5 counts remain unchanged
  from R4-24a.
- RED, 2026-07-29: the first exact runtime-control test failed to compile with
  exit `101` and `E0432` because
  `with_repair_verification_test_control` and
  `RepairVerificationTestControl` did not exist. The missing imports proved
  the old ad-hoc hook had no scoped, consumed replacement.
- GREEN, 2026-07-29: one `#[cfg(test)]` task-local control now consumes five
  typed checkpoints plus one post-persist/pre-COMMIT fault. The old generic
  hook parameter/facade is gone; the production atomic input still carries no
  callback or raw capability. The fault preserves the existing lingering
  transaction hazard deliberately: durable receipt and pending link survive,
  retry returns the existing receipt and clears pending, then a DB-owned test
  helper cleans up the injected transaction.
- Runtime controls cover rename, stale projection, ordinary projection, and
  nonprojection. The three page branches start a real DB contender after
  BEGIN, observe its lock future as `Poll::Pending`, keep it pending while the
  applicable projection lock is contended before and after receipt
  persistence, then prove it enters only after verification returns. Pending
  and completion waits are both bounded to five seconds. Post-COMMIT controls
  prove the DB mutex is free while the durable receipt and pending link still
  coexist and the applicable manifest/tag/rename-session locks remain held.
- Failure controls prove report, target, and non-target conflicts return
  before persistence, retain pending, and leave the transaction reusable. A
  real conflicting final-receipt filesystem object exercises the unchanged
  persistence failure plus rollback. The source teeth fail closed when a
  marker is absent, require every configured checkpoint/fault to be consumed,
  and keep the exact DB test helpers under `#[cfg(test)]`.
- Gates: verification passes `28 / 28`; R4-24a+b source and mutation teeth pass
  `5 / 5`; exact rename/stale/ordinary/non-target/tag controls pass `5 / 5`;
  all external-access teeth pass `3 / 3` with no baseline change; and the M5
  inventory remains `191 / 55-50-86 / exposure 22`. Root's first four guessed
  fully-qualified `--exact` paths selected zero tests and are not counted;
  `cargo test -- --list` supplied the canonical paths used for the recorded
  `5 / 5`.
- LSP reports zero errors in all six changed Rust files and resolves the
  contender as the one DB-child test struct. One bulk reference request timed
  out inside rust-analyzer; literal/source teeth and the exact runtime controls
  provide the closure evidence instead. Core/server all-target Clippy with
  `-D warnings`, formatting, `git diff --check`, and the full drift-guard
  filter pass.
- FINAL REVIEW GATE, 2026-07-29: the concurrency auditor first returned
  **BLOCK** because instantaneous `try_lock` samples did not prove a queued
  contender, conflict tests lacked error-poststate assertions, and one source
  helper passed vacuously when its marker disappeared. Root then found a
  second **BLOCK** when those assertions added five external raw connection
  locks and broke the exact ratchet. The final test-only revision uses an
  opaque DB-owned contender and exact DB-owned transaction probes without
  changing the baseline. Both independent Sol reviewers re-read that final
  diff and returned **APPROVE** with no remaining blocker.

#### R4-25 — exact test-support seam and always-private connection

- PRE-IMPLEMENTATION CENSUS, 2026-07-29: the old ratchet's `289` is only the
  primary `.conn.lock().await` subset. AST discovery plus source inspection
  finds `342` current external raw-field occurrences:
  `289` primary locks, `5` primary `try_lock()` observations, and `48`
  alternate `_db` references. The `_db` set is `46` test references plus
  exactly `2` production references in `lint/snapshot.rs`; `21` of the test
  references open independent secondary connections. There is no external
  `Arc::clone(&db.conn)` or `Arc::clone(&db._db)` at entry. LSP references for
  the `_db` field independently corroborate those two production entries,
  every test reference, and the DB-internal construction sites.
- R4-25a is a separate production micro-slice before any fixture migration.
  Move only the existing `MemoryDB::open_lint_snapshot` and
  `MemoryDB::open_unpinned_lint_snapshot` implementation from
  `lint/snapshot.rs` into a `db/**` child. Preserve signatures, visibility,
  freshness-observer cloning, returned snapshot lifetime, errors, and call
  sites byte-for-byte semantically. This makes the production external
  `_db` count `2 → 0`; it is not hidden inside a test-support commit.
- After R4-25a, the exact test-entry universe is `340` raw-field occurrences:
  `289` primary locks, `5` primary mutex-availability observations, and `46`
  alternate-handle references. Add a syntax-aware census before migration and
  ratchet each bounded group down from that exact set. Ordinary Rust
  expressions are read from the parsed AST; macro invocations receive explicit
  recursive token-tree traversal. A bare ast-grep rule is insufficient:
  current `assert!` / `assert_eq!` token trees hide four `try_lock` references
  and one `_db` handoff from its ordinary expression matches.
- Implement that CI-local parser with test-only direct dependencies already
  present in the lockfile: `syn 2.0.117` with `full,visit` and
  `proc-macro2 1.0.106` with `span-locations`. `syn::Visit` records ordinary
  field expressions, destructuring, support calls, and raw type paths exposed
  by a DB-support API signature; a recursive `TokenTree` scan covers macro
  bodies, which `syn` keeps opaque. Literal and comment lookalikes do not
  count. A standalone test-created `libsql::Database` that never originates
  from `MemoryDB` is outside this capability-boundary contract; a future one
  must have its constructor recorded exactly so origin is explicit, but its
  raw type alone is not a violation. A tracked Rust file, item, or macro body
  that the tooth cannot classify is an error rather than a silent exclusion.
- Introduce one `#[cfg(test)]`, DB-owned, opaque `TestDbSession` type with
  private state and private raw constructors. Named `MemoryDB` methods create
  either a primary session that retains the mutex guard or a genuinely
  independent secondary session. The same opaque type privately tracks the
  secondary connection's ordinary, immediate-transaction, or read-only-
  transaction state; `begin_immediate`, `begin_read_only`, `commit`, and
  `rollback` preserve the existing connection and transaction lifetimes.
  `ReadOnly` remains literal `libsql::TransactionBehavior::ReadOnly`, and the
  independent `BEGIN IMMEDIATE` contender remains independent rather than
  being routed through the primary mutex.
- The common session surface is limited to the fixture operations actually
  present at entry: `execute`, `execute_batch`, and `query`. Query results and
  rows are opaque wrappers with async iteration and typed getters; a lifetime
  tie keeps the owning session alive while rows are consumed. The wrappers
  never return, dereference, borrow, or callback-yield a
  `libsql::Connection`, `Database`, `MutexGuard`, `Rows`, `Row`, transaction,
  or another raw capability. Input SQL and `IntoParams` remain deliberately
  generic because this is a test-only fixture seam, not a production SQL
  facade.
- The operation census covers all `289` primary locks across `217` enclosing
  functions: their direct connection calls are only `execute`,
  `execute_batch`, and `query`. Four repair assertions also pass the held
  connection to `database_content_digest`, and one eval test passes it to
  `check_seed_contract`; admit exact typed
  `repair_database_content_digest` and `check_seed_contract` session
  operations rather than exposing the connection. The answer-quality feature
  assertion moves to the existing
  `MemoryDB::assert_eval_feature_substrate_live` domain method. Local fixture
  helpers such as `set_truth` accept the opaque session.
- The only other non-generic operations are primary-mutex availability,
  structural digest on an opaque secondary session, and
  `MemoryDB::open_isolated_lint_snapshot_for_test()`. The last operation
  creates a fresh observer internally on every call. The exact `46` test
  `_db` census is `21` secondary connections, `18` direct
  `LintReadSnapshot::open` calls, `3` semantic-fingerprint handoffs,
  `2` fresh-clock constructions, and `2` matching `open_with_freshness`
  calls. The isolated operation replaces the `18` direct opens, the `3`
  fingerprint handoffs after their helper accepts `&MemoryDB`, and each
  two-field old/new observer pair with one call. Those daemon-lifetime epochs
  therefore stay distinct without handing `_db` to `LintFreshnessClock` or
  `LintReadSnapshot`.
  R4-24b's existing DB-child contender and transaction probes remain separate
  exact `#[cfg(test)]` operations; they are included in the approved
  test-support manifest rather than widened into the generic session.
- Freeze two manifests by syntax identity, not a total count and not a broad
  filename / `#[cfg(test)]` allowlist. `RAW_CAPABILITY_ENTRY` records
  `(tracked path, enclosing module/test item, shape, same-shape ordinal)` for
  the `342` starting references and shrinks to empty.
  `TEST_SUPPORT_CALLS` records
  `(tracked path, enclosing test/support item, exact DB-support callee,
  same-callee ordinal)` for the reviewed replacement source. Source lines are
  diagnostic metadata only. The two manifests are deliberately independent:
  one held raw guard may become several opaque `execute` / `query` calls, so
  replacement cardinality is not inferred from `340`. Adding, duplicating,
  renaming, retaining, moving between test items, or calling a support
  operation from production fails even when an aggregate count is unchanged.
- RED controls must kill single-line and multiline primary access,
  `try_lock` inside a macro, alternate `.connect()`, raw `_db` handoff,
  renamed receivers, dereference/borrow forms, and retained/cloned fields.
  Separate manifest mutations must kill an extra call, a removed call, a
  same-count move, a changed callee, and a macro-contained call. The parser
  fails closed on an unparseable tracked Rust file or macro form. Final field
  privacy is the compiler backstop, not a substitute for the source teeth.
- Migrate without assertion changes, test splitting, fixture cleanup, or
  adjacent production refactors, in this order: (1) the production snapshot
  micro-slice; (2) the opaque seam, exact census/manifest, and RED controls;
  (3) all alternate-handle and lint-snapshot fixtures; (4) the repair family,
  whose `107` primary locks include the independent
  `BEGIN IMMEDIATE`/blocked-CAS control; (5) lint's remaining `71` primary
  locks; (6) the remaining `111` lower-risk primary locks; (7) make both
  `MemoryDB::conn` and `_db` private in every build and replace the old
  `289`-only baseline with the zero raw-capability gate. Each group commits
  only after its exact raw-set decrease and support-manifest addition agree.
- Final floor: production external raw capability `0`; test raw capability
  `340 → 0`; both fields private under normal and test builds; the exact
  support manifest matches the parsed source; all mutation controls pass; the
  M5 inventories remain unchanged semantically; and one uninterrupted
  `cargo test --workspace --lib --quiet` passes after the final fixture group.
  The two highest-risk focused controls must additionally prove that the
  independent write transaction still blocks then releases the repair CAS and
  that the read-only lint snapshot plus replacement freshness observers retain
  their original lifetime semantics.
- R4-25a RED, 2026-07-29: the new ownership tooth failed with exit `101`,
  reporting `lint/snapshot.rs` still contained `2` external `self._db`
  references instead of `0`.
- R4-25a GREEN, 2026-07-29: only the two existing snapshot entrypoints moved
  into `db/lint_snapshot.rs`; `db.rs` gained production/test module wiring and
  `lint/snapshot.rs` lost only its `MemoryDB` import and impl. The source tooth
  passes `1 / 1`, locks both complete forwarding expressions, and rejects
  changed receiver and observer mutants. Snapshot behavior passes `10 / 10`;
  the fixed-clock runner control that exercises the unpinned facade passes
  `1 / 1`; and external-access teeth pass `3 / 3`.
- The M5 sweep remains exactly `191 / 55-50-86 / exposure 22`; all `65`
  changed `core/db.rs` inventory addresses move by exactly `+3`, with names,
  visibility, classification, and exposure unchanged. LSP reports zero errors
  in the three changed production Rust files and resolves the pinned facade to
  `35` callers plus its declaration and the unpinned facade to its sole runner
  caller plus declaration. Formatting and `git diff --check` pass. Root's
  first guessed `lint::snapshot_tests` filter selected zero tests and is not
  counted; `cargo test -- --list` supplied the canonical
  `lint::snapshot::tests` path used for the recorded `10 / 10`.
- R4-25a REVIEW, 2026-07-29: the independent Sol reviewer returned
  **APPROVE** after inspecting the final diff. It confirmed exact movement,
  visibility/lifetime/observer fidelity, non-vacuous source mutants, the
  address-only inventory refresh, and absence of fixture, support-session, or
  behavior changes.
- R4-25b RED/GREEN evidence, 2026-07-29: the parser RED returned an empty raw
  set against six expected ordinary/multiline/macro shapes; the manifest RED
  accepted an empty manifest; and the API RED failed compilation on the
  missing `test_primary_session` / `test_secondary_session` methods. GREEN
  passes `26 / 26` parser, graph, exact-manifest, mutation, DB-child escape,
  and visible-signature teeth plus `5 / 5` opaque-session behavior tests. The
  review RED returned `8` parser/tooth failures out of `25`, `2` behavior
  failures out of `5`, and three `E0599` compile errors when commit/rollback
  still returned `()`. The frozen manifests carry
  `347` raw identities (`340` `MemoryDB`-derived plus `7` standalone libSQL
  Builder origins) and `31` existing R4-24b support-call identities. A live
  libSQL `0.9.30` probe found that `TransactionBehavior::ReadOnly` emits
  `BEGIN READONLY` but does not itself reject local DML, so the secondary-only
  test session now follows lint's established `PRAGMA query_only = ON`
  pattern, resets it on commit/rollback, and proves the same consumed
  secondary session is writable afterward. Managed transactions and structural
  digests are secondary-only; repair digests and seed checks remain on the
  primary session. Every tracked Rust file must resolve through the module
  graph; `eval/eval_judge.rs` is retained as a private empty module so both the
  compiler and the guard classify it. The legacy external-access teeth pass
  `3 / 3`; M5
  remains `191 / 55-50-86 / exposure 22` with address-only `db.rs` shifts.
- R4-25b second-closure RED/GREEN evidence, 2026-07-30: the focused RED passed
  `26 / 30` and failed exactly four controls: production capability aliases
  returned no violations, macro UFCS/wrapped receivers returned no calls,
  ambiguous macro targets returned no errors, and `eval_judge.rs` was absent.
  GREEN passes `30 / 30`. The DB-child scan resolves exact grouped and renamed
  imports, rejects public and `pub(crate)` `Database` / `Connection` /
  `Transaction` capabilities without confusing an unrelated domain
  `Connection`, and treats a mutex guard as a DB capability only when its
  generic target contains a libSQL `Connection`. The live scan found four
  existing exports; reference inspection closed them by making
  `parity_guard_shape_drift` private and both entity-page adapter functions
  `pub(super)`, while `lock_space_writes -> MutexGuard<()>` remained public
  because it carries no DB handle. Macro teeth now classify UFCS,
  parenthesized/dereferenced typed receivers, and aliased Builder constructors,
  while unknown target-bearing forms fail through the analysis error channel.
  A lexical-order regression first left the suite at `29 / 30`; receiver-first
  visitation restored the exact factory-before-method identity order. Final
  gates pass the `5 / 5` opaque session suite, `3 / 3` legacy access teeth,
  `16 / 16` truth manifest, `13 / 13` truth exposure, the Rust M5 guard, the
  `191 / 55-50-86 / exposure 22` Python sweep, and clippy with warnings denied.
- R4-25b third-closure RED/GREEN evidence, 2026-07-30: the focused RED passed
  `30 / 34` and failed exactly four mutants: `db.rs` `pub(super)` plus libSQL
  module/glob imports were missed, module-level macro UFCS produced no support
  identity, comma-separated wrapped receivers produced no calls, and a
  function-local Builder alias produced no standalone origin. GREEN passes
  `34 / 34`. Visibility is now path-aware: `db.rs` `pub(super)` reaches the
  crate root and is scanned, while a DB child module's `pub(super)` remains
  inside the DB boundary; `pub(in crate)` remains crate-visible and
  `pub(in crate::db)` remains DB-internal. Exact type resolution covers
  `use libsql as sql; sql::Connection` and `use libsql::*; Connection` without
  treating an unrelated imported `Connection` as libSQL. Module macro
  definitions record exact `TestDbSession` UFCS calls; unproven metavariables
  tied to a known session method fail through the analysis error channel;
  comma-separated invocations preserve typed receiver flow; and block-local
  Builder aliases feed the same standalone-origin classifier. An intermediate
  repository run exposed false positives on ordinary `$start.await` /
  `$pcn.push` / raw `$db` macros; requiring a known session method (or exact
  `$session`) as the candidate signal removed them without weakening the
  mutants. Final gates pass `5 / 5` opaque sessions, `3 / 3` legacy access,
  `16 / 16` truth manifest, `13 / 13` truth exposure, Rust M5, the
  `191 / 55-50-86 / exposure 22` Python sweep, clippy with warnings denied,
  formatting, and diff hygiene.
- R4-25b REVIEW, 2026-07-30: the final independent Sol review returned
  **APPROVE** after confirming the path-aware DB visibility rules, direct /
  module / glob libSQL alias resolution, macro UFCS and fail-closed
  metavariable handling, comma-separated wrapped receivers, function-local
  Builder aliases, and the private module-graph wiring for `eval_judge`.
  Root then re-ran the `34 / 34` parser / manifest teeth, `5 / 5` opaque
  session behavior tests, `16 / 16` truth manifest tests, `13 / 13` truth
  exposure tests, the Rust M5 guard, the `191 / 55-50-86 / exposure 22`
  Python sweep, clippy with warnings denied, formatting, diff hygiene, and LSP
  diagnostics on every changed Rust seam; all passed.
- R4-25 group 3 RED/GREEN evidence, 2026-07-30: the focused census RED failed
  exactly `46 != 0`. After migrating the frozen alternate-handle group, the
  exact raw-manifest RED reported precisely the `46` expected
  `AlternateDbField` identities removed and no raw identity added. GREEN
  leaves `301` raw identities: `289` `PrimaryConnLock`, `5`
  `PrimaryConnTryLock`, `7` standalone libSQL origins, and `0`
  `AlternateDbField`. The support manifest grows `31 -> 108`, adding `77`
  exact call identities, including `21` `test_secondary_session` and `21`
  `open_isolated_lint_snapshot_for_test` entries. The existing opaque seam
  expressed every fixture operation; group 3 added no seam method and exposed
  no raw handle. Standalone local-libSQL semantic-fingerprint coverage remains
  on its original helper, while the three `MemoryDB` handoffs use a separate
  `&MemoryDB` helper.
- The group 3 behavior closure passes snapshot `10 / 10`, lint test-support
  `7 / 7`, affected lint/pages/serving `66 / 66`, affected repair fingerprint
  `5 / 5`, and all `15 / 15` affected deterministic repair-plan controls,
  including the blocked-CAS and subprocess-contention cases. The parser and
  exact manifests pass `34 / 34`; the opaque seam passes `5 / 5`; the legacy
  external-access teeth pass `3 / 3`; truth manifest and exposure pass
  `16 / 16` and `13 / 13`; the Rust M5 gate passes `1 / 1`; and the Python
  inventory remains `191 / 55-50-86 / exposure 22`. Core/server all-target
  Clippy with warnings denied, formatting, and diff hygiene pass.
- Tracking the R4 parser fixture exposed one latent R4-25b guard conflict: the
  earlier legacy regex census had run while that file was untracked, then
  counted eight synthetic `.conn.lock().await` strings and macro controls as
  real external access once it became tracked. Its focused RED reported those
  exact eight false positives. The legacy census now excludes only
  `crates/wenlan-core/src/drift_guard/r4_test_support_test.rs` through a pure,
  exact-path predicate; a direct control proves that path is excluded and an
  ordinary new source path is still included. The syntax-aware raw/support
  manifests remain the authoritative guard for real accesses in the parser
  fixture; the eight synthetic matches were not added to the legacy baseline.
- R4-25 group 3 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no correctness, capability-boundary, concurrency,
  data-loss, or manifest-gaming finding. It independently confirmed the
  `301` raw / `108` support exact sets, all `21` secondary-session and `21`
  isolated-snapshot call sites, the genuinely independent blocked
  `BEGIN IMMEDIATE` contender, fresh replacement-observer epochs, preserved
  standalone libSQL fingerprint helper, and exact-path legacy exclusion.
  Its focused parser, seam, snapshot, legacy, blocked-CAS, fingerprint,
  formatting, diff, and LSP checks passed; one guessed zero-test filter was
  explicitly excluded from its evidence.
- R4-25 group 4 RED/GREEN evidence, 2026-07-30: the focused census RED failed
  exactly `289 != 182`. The exact raw-manifest RED then reported only the
  frozen repair family removed: `107` `PrimaryConnLock` identities
  (`15 + 17 + 8 + 29 + 38`) and all `5` repair
  `PrimaryConnTryLock` identities. The legacy baseline independently failed
  on exactly the same five now-stale repair paths. GREEN leaves `189` raw
  identities: `182` non-repair primary locks, `0` try-locks, `0` alternate
  fields, and the unchanged `7` standalone libSQL origins.
- The first support-manifest RED found `292`, not the preflight's expected
  `303`, replacement identities. Every category agreed except
  `TestDbRow::get`, where `11` calls inside `while let
  Some(row) = rows.next().await...` were missing. A focused parser RED
  resolved only `[query, next, next]` instead of
  `[query, next, get, next, get]`, while an unrelated `next/get` remained
  unclassified. The syntax-aware visitor now carries a proven Rows initializer
  into only that loop body and classifies its bound pattern as Row. GREEN
  records the exact `303`: `107` `test_primary_session`, `61` execute, `29`
  execute-batch, `27` query, `27` next, `43` get, `5`
  `primary_mutex_available`, and `4`
  `repair_database_content_digest`. The support manifest is therefore
  `108 -> 411`; no seam method or raw capability was added.
- The two disjoint implementation lanes pass all affected repair modules:
  `18 / 18` entity extraction, `10 / 10` title rename, `45 / 45`
  `repair::tests`, `34 / 34` repair-plan tests, and `46 / 46` deterministic
  repair-plan tests. Focused root controls pass the four availability
  checkpoints, owned projection-session reuse, aggregate-CAS zero mutation,
  and the independent secondary `BEGIN IMMEDIATE` blocked-CAS release. The
  parser / exact manifests pass `35 / 35`; the opaque seam passes `5 / 5`;
  legacy access passes `3 / 3` plus the exact parser-fixture exclusion control;
  truth manifest and exposure pass `16 / 16` and `13 / 13`; Rust M5 passes
  `1 / 1`; and the Python inventory remains
  `191 / 55-50-86 / exposure 22`. Eight generated inventory addresses in
  later `repair.rs` functions moved by exactly `-1`, with names, visibility,
  classification, and exposure unchanged. Core/server all-target Clippy with
  warnings denied passes. The uninterrupted workspace library floor remains
  reserved for the final R4-25 fixture group.
- R4-25 group 4 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no correctness, concurrency, data-loss, security,
  regression, or manifest-gaming blocker. It independently confirmed the
  exact `107`-lock and `5`-availability migration, `189` raw / `411` support
  manifests, bounded while-let propagation with an unrelated-get negative
  control, raw primary rollback semantics, equivalent mutex-timing
  assertions, and the independent secondary blocked-CAS transaction. Its
  parser, repair, repair-plan, legacy, M5, clippy, formatting, diff, and
  per-file LSP checks all passed.
- R4-25 group 5 RED/GREEN evidence, 2026-07-30: after correcting the frozen
  prose census from `70` to the manifest-backed `71`, the focused census RED
  failed exactly `182 != 111`. Migrating the complete lint family removed
  exactly those `71` `PrimaryConnLock` identities and no other raw identity.
  GREEN leaves `118` raw identities: `111` non-lint primary locks, `0`
  try-locks, `0` alternate fields, and the unchanged `7` standalone libSQL
  origins. The legacy baseline independently failed on exactly the `24`
  lint files whose direct-lock counts reached zero.
- The support-manifest RED added exactly `160` identities and removed none:
  `71` `test_primary_session`, `55` execute, `27` execute-batch, `2` query,
  `2` next, and `3` get. GREEN is therefore `411 -> 571`; every fixture fits
  the existing opaque seam, with no new support method or raw capability.
  Original explicit drops, isolated snapshots, fresh observers, read-only
  transactions, and SQL/assertion order remain unchanged.
- The full lint namespace passes `218 / 218` with `2` ignored. The parser /
  exact manifests pass `35 / 35`; the opaque seam passes `5 / 5`; legacy
  access passes `3 / 3`; truth manifest and exposure pass `16 / 16` and
  `13 / 13`; Rust M5 passes `1 / 1`; and the Python inventory remains
  `191 / 55-50-86 / exposure 22`. Core/server all-target Clippy with warnings
  denied, formatting, diff hygiene, and LSP error diagnostics pass.
- R4-25 group 5 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no correctness, lifetime, concurrency, capability-boundary,
  freshness, read-only, manifest-gaming, M5, or truth blocker. It independently
  confirmed all `71` migrations, zero lint references to raw `MemoryDB::conn`,
  exactly `71` LSP-resolved references to `test_primary_session`, the
  `118` raw / `571` support manifests, unchanged statement/drop ordering, and
  the `71 + 111` census correction. Its parser, seam, legacy, full-lint, truth,
  M5, Clippy, formatting, diff, and per-file LSP checks all passed.
- R4-25 group 6 RED/GREEN evidence, 2026-07-30: the final fixture-group census
  RED failed exactly `111 != 0`. Migrating the complete remaining set removed
  exactly those `111` `PrimaryConnLock` identities and no standalone origin.
  GREEN leaves only the `7` intentional standalone libSQL Builder origins,
  with `0` primary locks, `0` try-locks, `0` alternate fields, and `0` field
  escapes. The legacy external-access baseline independently failed on the
  exact `21` stale paths and is now empty.
- The support-manifest RED added exactly `397` identities and removed none:
  `110` `test_primary_session`, `79` execute, `9` execute-batch, `56` query,
  `57` next, `85` get, and `1` typed `check_seed_contract`. The preflight's
  approximate row-flow trace predicted only about `72` gets; the syntax-aware
  parser's exact `85` identities, including chained opaque-row uses but
  excluding unrelated JSON `get` calls, are the frozen evidence. GREEN is
  therefore `571 -> 968`. The one non-session migration routes answer-quality
  through the existing `MemoryDB::assert_eval_feature_substrate_live` domain
  method. The projection truth helper now takes `&TestDbSession`; no seam
  method or raw capability was added.
- The three implementation lanes pass `431` focused tests. The parser / exact
  manifests pass `35 / 35`; the opaque seam passes `5 / 5`; the blocked repair
  CAS, read-only lint snapshot, and replacement freshness-observer controls
  pass individually; and the required uninterrupted workspace library floor
  passes `4168` tests with `35` ignored and `0` failures. Truth manifest and
  exposure pass `16 / 16` and `13 / 13`; Rust M5 passes `1 / 1`; the Python
  inventory remains `191 / 55-50-86 / exposure 22`; core/server all-target
  Clippy with warnings denied, formatting, diff hygiene, and LSP error
  diagnostics across all `23` changed Rust files pass.
- R4-25 group 6 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no correctness, concurrency, data-loss, security,
  truth/M5-drift, capability-boundary, or manifest-gaming blocker. It
  independently reconciled raw `-111/+0` and support `+397/-0`, verified the
  retained mutex/row lifetimes and explicit releases, inspected trigger
  cleanup, eval domain/typed routes, truth permits and projection invariants,
  and confirmed zero LSP errors across all changed Rust files. The
  workspace-wide LSP reference display capped at `200`, so neither root nor
  reviewer treated it as the exact census; the syntax-aware manifest remains
  authoritative.
- R4-25 group 7 RED/GREEN evidence, 2026-07-30: removing the parser guard's
  exact transitional exception failed the repository gate on precisely two
  visible strong-capability fields: `MemoryDB::_db` exposing
  `libsql::Database` and `MemoryDB::conn` exposing `libsql::Connection`.
  Making both fields private turned the same gate GREEN; the synthetic
  positive control now requires both visible field violations alongside
  function/enum/trait escapes. The obsolete exception and its type-shape
  parser were deleted rather than replaced by a new allowlist.
- Normal and test builds both compile with the private fields. Parser /
  manifests pass `35 / 35`; the opaque seam passes `5 / 5`; the now-empty
  legacy external-access baseline passes `3 / 3`; truth manifest and exposure
  pass `16 / 16` and `13 / 13`; Rust M5 passes `1 / 1`; and the Python
  inventory remains `191 / 55-50-86 / exposure 22`. LSP resolves the opaque
  test-support accesses back to the private field declarations and reports
  zero errors in both changed Rust files. Core/server all-target Clippy with
  warnings denied, formatting, and diff hygiene pass.
- R4-25 group 7 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no correctness, architecture, security, concurrency,
  data-loss, regression, truth/M5, or verification blocker. It confirmed that
  Rust privacy keeps the test-only DB child seam functional while preventing
  sibling/external field access, the transitional exception is fully gone,
  both visible-field positive controls bite, manifests are byte-identical to
  group 6, and normal/test compilation plus every recorded gate is green.

### R5 — server vertical slices

Move route registration and handlers domain by domain. Preserve route identity,
typed request/response contracts, and `TrackedRouter` classification.

Current-tree rebaseline at `383a2f5d`:

- `memory_routes.rs` contains `95` public `handle_*` functions across `5,732`
  lines;
- `router.rs` contains the main composition root across `685` lines;
- the truth manifest contains `167` static HTTP rows and expands to exactly
  `171` runtime `(builder, method, path)` rows: `165` main and `6` repair;
- `TrackedRouter` proves exact truth-manifest and sensitive-read route sets, but
  it does not currently prove that a preserved method/path still points at the
  same handler.

#### R5-0 — executable route-to-handler boundary

Before moving a registration or handler, extend `TrackedMethodRouter` to retain
the concrete handler identity captured by Rust's type system. Add a separate,
current-tree manifest over exact
`(builder, method, path, handler-identity)` rows and require production
`finish()` / `finish_restricted()` to match it by set equality.

The new manifest is independent of both existing route contracts:

- `truth_manifest` continues to classify every registered method/path for M5;
- `sensitive_read_routes` continues to classify the scoped-read subset;
- the R5 handler manifest proves that movement did not silently bind the right
  route to the wrong function.

The guard must bite before any movement:

- a production main-router and repair-router positive control pass with all
  exact handler rows;
- substituting a different handler at an existing method/path fails;
- omitting or duplicating one row fails;
- a test-only unbound constructor, if needed for synthetic classifier tests,
  remains `#[cfg(test)]` and cannot be called by production builders;
- no source parser or line-number table is the runtime oracle. A parser may
  generate/reconcile the static rows, but `TrackedRouter` supplies the observed
  set at build time.

The initial manifest is a guard-only commit. It changes no route, handler,
layer order, response, truth adapter, or generation.

Handler identity is exact only within the pinned Rust toolchain. If a toolchain
change makes every handler row fail at once, refresh the manifest in a
manifest-only commit after review; never weaken equality to suffix or substring
matching.

R5-0 implementation evidence, 2026-07-30:

- The first RED failed compilation with nine missing-API errors for the new
  handler identity, exact coverage assertion, and test-only unbound
  constructor. After the minimal tracker existed but before the manifest was
  populated, the production positive control supplied the second RED:
  `finish()` observed all `165` main bindings and rejected the empty expected
  set. Building the restricted router independently observed and rejected its
  `6` bindings.
- `TrackedMethodRouter` now records the concrete function-item
  `std::any::type_name::<H>()` beside every top-level and chained method.
  `TrackedRouter` compares a count-aware observed map against a duplicate-free
  exact `(builder, method, path, handler)` manifest in both production finish
  paths. The only opt-out is a private `#[cfg(test)]` constructor used by the
  synthetic truth-manifest registration tests.
- The focused guard suite passes `17 / 17`, including chained-method identity,
  wrong-handler,
  missing-binding, duplicate-row, malformed-row, and production
  main-plus-repair controls. The complete server library passes
  `347 passed / 2 ignored`; truth manifest and exposure pass `16 / 16` and
  `23 / 23`; the Rust M5 gate passes `1 / 1`; and the Python inventory remains
  `191 / 55-50-86 / exposure 22`. Core/server all-target Clippy with warnings
  denied, formatting, diff hygiene, and LSP error diagnostics over the complete
  route-registry directory pass.
- R5-0 REVIEW, 2026-07-30: the independent Sol reviewer returned
  **APPROVE** with no critical, important, or minor finding. It independently
  verified the exact `165 + 6` manifest, count-aware actual rows, duplicate
  rejection before builder filtering, chained-method identity pairing, pinned
  toolchain failure mode, library-target type paths, unchanged truth/security
  layer order, and the private test-only opt-out's single synthetic caller. Its
  focused run passed `17 / 17`; LSP reported zero diagnostics and diff hygiene
  passed.

#### R5 movement sequence

Every following commit is movement-only under D2. One writer owns `router.rs`
and the active domain file under D7. A slice moves its registration helper,
handlers, private helpers, request/response wrappers, and directly owned tests
together; it does not rewrite handler bodies or opportunistically consolidate
types.

1. Existing handler modules gain `register(TrackedRouter<SharedState>)`
   composition helpers, in small bounded commits: general/brief/community,
   ingest/import, source/config, and
   knowledge/onboarding/refinery/page-map/websocket. `router.rs` retains only
   ordering and cross-cutting layers.
2. Extract the lowest-coupling `memory_routes` families first:
   profile/agents, entity graph, spaces, then indexed files/chunks.
3. Extract activities/tags plus the non-page capture/detail family, followed by
   decisions/briefing/profile narrative/pinned/revisions/snapshots.
4. Extract the remaining memory CRUD/search/enrichment family after its
   scheduler-handoff, rerank, attribution, and update tests move with it.
5. Move the page family last as one protected lane: page list/get/search,
   sources, export, archive/delete/create/refresh, links/orphans/revisions, and
   existing page-map registration. Its truth marker/adapters and
   `PagePermit` write gates remain unchanged.

The exact handler count and each route-to-handler partition are taken from the
R5 manifest, not the prose grouping above. Each commit may change handler
module identities only for its named rows; all `(builder, method, path)` sets
remain byte-for-byte equal, and every untouched handler identity must remain
equal.

The R5 baseline merges current main through `22b401da` in `383a2f5d`, including
PR-D `d1fb5d9f`. PR-D resolves the old
`GET /api/activities` manifest/handler mismatch with
`truth_adapter::redact_page_activity_detail`; the activity slice must move that
call byte-for-byte with the handler. The live production database is already at
generation `1` with fence `2:committed`, while a freshly migrated database
still defaults to `0`. R5 does not readjust either state: it preserves PR-D's
generation-sensitive behavior and never calls the cutover ceremony or setter.

#### Per-slice evidence

Each movement commit must show:

- a handler-movement slice makes the exact handler manifest RED before its
  intentional identity update, then GREEN with only the named rows changed; a
  registration-only slice instead leaves that manifest byte-identical and
  proves a deliberately omitted route fails before completion;
- exact `171` runtime truth rows and exact sensitive-read set equality;
- at least one typed built-router round-trip per moved HTTP endpoint, using
  existing `wenlan-types` contracts whenever available, never
  `serde_json::Value`, as the oracle. When production has no shared
  deserializable DTO, an exact test-local `Deserialize` mirror is permitted
  and must be named in the slice evidence; errors use the established
  test-only envelope. Cover typed success and typed deterministic error where
  both are hermetically reachable. Where a success path requires non-hermetic work
  (`POST /api/on-device-model/download`: real model download plus engine
  initialization), the typed deterministic unknown-model error is sufficient;
  infallible read handlers cover typed success only. Name every relaxation and
  its reason in the slice evidence. Adding a production seam solely to make a
  success path hermetic is outside R5; any such seam is a post-R5
  behavior-change slice;
- LSP definition/reference closure before movement and zero error diagnostics
  after movement; LSP's `200`-reference display cap is never treated as the
  complete census;
- affected server tests, M5 truth/reader/write gates, Clippy with warnings
  denied, formatting, and diff hygiene;
- a fresh Sol review against D2, route/wire identity, async lock lifetimes, and
  M5 drift.

The truth guard remains a route layer over the finalized router, and the CORS,
local-only, shutdown-extension, and state layers retain their current order.

R5 evidence-contract addendum, 2026-07-30:

- Fable selected risk-shaped per-endpoint evidence (**C**) over adding
  test-only production injection surface (**A**) or leaving the download route
  behind in `router.rs` (**B**). This is a scoped evidence clarification, not
  a movement-design change: the handler manifest still pins exact identity,
  D2 still forbids body edits, and every moved endpoint must traverse the
  built router with a typed oracle. Sol remains the per-slice reviewer.

R5 registration slice 1 evidence, 2026-07-30:

- Registration ownership for the `21` main-builder bindings in `routes`,
  `brief_routes`, and `community_routes` moved into one `register` helper per
  owning module. No handler body, helper, request/response type, route layer,
  or exact handler-manifest row changed. The legacy `/api/context` direct call
  into `handle_read_brief` remains intact.
- Omitting `GET /api/pages/recent-changes` during the movement made the
  production-builder control RED on the exact sensitive-route set. Restoring
  it to its original page lane made the same control GREEN. The general helper
  groups only disjoint paths; a scoped Sol check against pinned Axum `0.8.8`
  and matchit `0.8.4` accepted the textual registration regrouping as
  behavior-equivalent and rejected four artificial historical-order helpers
  as needless surface.
- The handler guard passes `17 / 17`, the server library passes
  `347 passed / 2 ignored`, server truth guard passes `12 / 12`, and focused
  Brief, context, and search HTTP suites pass `6 / 6`, `4 / 4`, and `3 / 3`.
  Truth manifest passes `16 / 16`; the Rust M5 gate passes `1 / 1`; the
  generated Python inventory remains `191 / 55-50-86 / exposure 22`.
  Generator-owned changes are eight address-only `routes.rs` shifts. LSP
  definition/reference closure and per-file error diagnostics pass, as do
  core/server all-target Clippy with warnings denied.
- R5 registration slice 1 REVIEW, 2026-07-30: the fresh Sol reviewer returned
  **APPROVE** with no correctness, architecture, security, concurrency,
  data-loss, regression, truth/M5, or verification blocker. It confirmed the
  exact `13 + 2 + 6` ownership move, unchanged Brief direct-call and
  cross-cutting layers, `pages/recent-changes` remaining in the page lane, and
  byte-identical handler manifest. Its only observed drift was corrected
  before approval: server `AGENTS.md` now teaches composition/helper ownership,
  and the inventory prose names the current
  `139 + 13 + 2 + 6 + 5 + 2 = 167` call-site distribution.

R5 registration slice 2 evidence, 2026-07-30:

- Registration ownership for all `4` `ingest_routes` bindings and all `3`
  `import_routes` bindings moved from the composition root into their owning
  modules without changing registration order. No handler body, request or
  response type, lock scope, truth classification, or handler-manifest row
  changed.
- Omitting `GET /api/import/state` made the production-builder control RED on
  the exact sensitive-route set; restoring the same
  `handle_list_pending_imports` binding made it GREEN. The handler guard passes
  `17 / 17`, server library `347 passed / 2 ignored`, truth guard `12 / 12`,
  and default-save-space import coverage `5 / 5`. Rust and Python M5 gates pass
  at `1 / 1` and `191 / 55-50-86 / exposure 22`; no generated inventory row
  or exact handler/truth/sensitive manifest changed.
- LSP resolves representative ingest/import handlers only to their colocated
  registration and definition after movement, with zero error diagnostics in
  all three changed Rust files and the new contract test. The built-router
  contract suite passes `2 / 2`: all seven moved endpoints deserialize their
  success wire into the existing `wenlan-types` response structs, and all
  seven deterministic failures deserialize into a typed test-only error
  envelope. It uses no `serde_json::Value` response oracle. Core/server
  all-target Clippy with warnings denied passes.
- R5 registration slice 2 REVIEW, 2026-07-30: after first returning **BLOCK**
  for missing typed built-router coverage, the same independent Sol reviewer
  returned **APPROVE** with no critical, important, or minor finding. It
  confirmed exact `4 + 3` route order and identities, typed success and error
  coverage for every moved endpoint, the test-only error envelope as the
  movement-only choice, a hermetic missing-file fixture, and byte-identical
  handler, truth, sensitive, and generated M5 manifests.

R5 registration slice 3 evidence, 2026-07-30:

- Registration ownership for all `4` `source_routes` bindings and all `10`
  `config_routes` bindings moved from the composition root into their owning
  modules in original order. No handler body, request/response type, lock
  scope, route layer, or handler-manifest row changed; `router.rs` is now
  `581` lines.
- Deliberately omitting `POST /api/on-device-model/download` made the
  production-builder control RED on that exact missing route. Restoring
  `handle_download_on_device_model` in the config helper made the same control
  GREEN.
- The built-router typed suite passes `1 / 1` across all `14` bindings. It
  covers the full Source add/list/sync/delete lifecycle and deterministic
  errors; typed config update, skip-apps, routing, setup/key, and model-list
  responses; and typed deterministic validation errors. Per the Fable
  addendum, the real multi-GB model-download success path is not invoked:
  unknown model id supplies its typed error round-trip. Pure read handlers
  have success coverage; the bodyless Source delete success is frozen as
  `204` plus an empty body. All other hermetically reachable success/error
  pairs are covered without a `serde_json::Value` response oracle.
  `SetupStatusResponse` and `ResolvedRoutingResponse` use exact test-local
  `Deserialize` mirrors because their production-local structs are
  `Serialize`-only and no shared response DTO exists. A test-local RAII guard
  binds `WENLAN_DATA_DIR` to a fresh temp root for the whole async test and
  restores the inherited value on drop; an inherited canary run passed while
  leaving the inherited directory file-empty.
- LSP reference closure moves representative Source and model-download
  handlers from `router.rs` into their owning helpers, with zero diagnostics
  in both modules, the composition root, and the new contract test. Server
  library passes `347 passed / 2 ignored`; Rust M5 passes `1 / 1`; Python
  inventory remains `191 / 55-50-86 / exposure 22`; and core/server
  all-target Clippy with warnings denied passes.
- R5 registration slice 3 REVIEW, 2026-07-30: Sol first caught and blocked
  false `4 + 11 = 15` evidence arithmetic plus an inherited
  `WENLAN_DATA_DIR` overwrite risk. After correction to exact `4 + 10 = 14`,
  explicit local-success-mirror wording, and the RAII isolation guard, it
  returned **APPROVE** with no remaining finding. It independently confirmed
  all identities/order, mixed-sensitive `/api/sources`, the no-download
  validation path, typed coverage, untouched canary directory, and
  byte-identical protected manifests.

R5 registration slice 4 evidence, 2026-07-30:

- Registration ownership for the `3 + 3 + 3 = 9` refinery, knowledge, and
  onboarding bindings moved into their existing modules at the same three
  composition positions. WebSocket and all ten page-map bindings remain
  deliberately out of this lower-risk slice. No handler body, request/response
  type, lock scope, route layer, or handler-manifest row changed;
  `router.rs` is now `547` lines.
- Deliberately omitting `POST /api/onboarding/reset` made the
  production-builder control RED on that exact missing route; restoring the
  same handler in `onboarding_routes::register` made it GREEN.
- The built-router typed suite passes `1 / 1` across all nine bindings:
  isolated knowledge path/count and scoped-relation wires, including the
  count route's regular-file `500` envelope; milestone list/acknowledge/reset,
  including typed validation and no-DB errors; and refinement
  list/reject/accept with real awaiting-review rows plus typed no-DB errors.
  The two bodyless onboarding successes are frozen as `204` plus empty bodies.
  No `serde_json::Value` is used as a response oracle.
- After refreshing diagnostics first, LSP reference closure drops the stale
  `router.rs` references and resolves the three representative handlers only
  inside their owning modules; all four changed Rust files and the contract
  test have zero diagnostics. Server library passes
  `347 passed / 2 ignored`; Rust M5 and Python inventory pass at `1 / 1` and
  `191 / 55-50-86 / exposure 22`; generator-owned inventory changes are two
  `knowledge_routes.rs` address shifts; and core/server all-target Clippy with
  warnings denied passes.
- R5 registration slice 4 REVIEW, 2026-07-30: Sol first blocked the missing
  hermetic `GET /api/knowledge/count` regular-file error leg. After the typed
  `500` envelope was added and the directory configuration restored, it
  returned **APPROVE** with no remaining finding. It independently confirmed
  exact nine-route identity/order and composition positions, D2/lock
  preservation, untouched WebSocket/page-map/layers, isolated page/config
  roots, address-only generated inventory changes, and byte-identical
  protected manifests.

R5 registration slice 5 evidence, 2026-07-30:

- Registration ownership for the one `GET /ws/updates` binding moved from the
  composition root into `websocket::register` at the same position after
  onboarding registration. No handler body, protocol type, lock scope, truth
  demotion, route layer, or handler-manifest row changed; `router.rs` is now
  `546` lines.
- Deliberately omitting the helper call made the production-builder control
  RED on the exact missing `GET /ws/updates` binding. Restoring
  `websocket::register` made the same control GREEN. The exact handler manifest
  still observes `165 + 6` runtime bindings.
- The built-router contract suite uses a real ephemeral loopback TCP listener
  and WebSocket upgrade rather than an HTTP `oneshot`. It passes `1 / 1`,
  deserializing the subscription response into an exact test-local
  `WsServerMessage` mirror and asserting the zero-count `index_progress`
  payload, then sending malformed text and deserializing the deterministic
  typed `error` response. `tokio-tungstenite 0.28`, already lockfile-resolved
  through Axum, is now a direct test-only dependency. No `serde_json::Value`
  response oracle or production test seam was added.
- LSP initially exposed its stale-index failure mode by retaining the removed
  `router.rs` reference. After zero-error diagnostics refreshed the three
  changed Rust files, the same reference query resolved
  `handle_ws_upgrade` only to its colocated registration and definition.
  Server library passes `347 passed / 2 ignored`; truth guard and truth
  manifest pass `12 / 12` and `16 / 16`; the Rust M5 gate passes `1 / 1`;
  Python inventory remains `191 / 55-50-86 / exposure 22`; core/server
  all-target Clippy with warnings denied, formatting, and diff hygiene pass.
  The generated reader rows remain byte-identical because they carry no source
  address. The ownership arithmetic above records the `router.rs` to
  `websocket.rs` move, and both M5 design documents advance their closed-enum
  demotion proof from `websocket.rs:34` to `websocket.rs:41`.
- R5 registration slice 5 REVIEW, 2026-07-30: the fresh Sol reviewer first
  returned **BLOCK** because both M5 design documents still pointed the
  WebSocket demotion proof at the pre-move `websocket.rs:34`; that line no
  longer contained `WsServerMessage`. After both citations moved to the exact
  enum at `websocket.rs:41` and the address-only evidence shift was recorded,
  its closure review returned **APPROVE** with no remaining finding. It also
  independently confirmed production-only exclusion of the test dependency,
  exact handler registration, unchanged layer order, real loopback typed
  success/error behavior, and the protected reader/truth gates.

R5 registration slice 6 evidence, 2026-07-30:

- Registration ownership for all ten Page Map bindings moved from the
  composition root into `page_map_routes::register` at the same position
  between ordinary Page routes and memory revisions. Seven `.route` calls
  expand to the same ten method/path/handler triples. No handler body,
  request/response type, CAS order, truth grant/adapter, `PagePermit`, route
  layer, or handler-manifest row changed; `router.rs` is now `518` lines.
- Deliberately omitting the new helper call made the production-builder
  control RED on the exact sensitive `GET /api/pages/{id}/map` binding before
  handler-manifest comparison. Restoring `page_map_routes::register` made the
  same control GREEN with all `165 + 6` runtime bindings.
- The built-router typed suite passes `1 / 1` across all ten endpoints. One
  real DB chain freezes the absent-map `0`, init-plus-first-node `2`, and
  subsequent CAS revision increments while exercising typed node, layout,
  edge, reset, and improve successes. Every endpoint also returns a typed
  deterministic missing-page `404`; the authoritative `MarkerShape::None`
  path is driven with reader intent and returns a typed `403` refusal.
  Shared `wenlan-types::page_map` DTOs are used everywhere except reset's
  exact test-local `ResetMapResponse`, because production still returns
  `Json<serde_json::Value>`. No production test seam or untyped response oracle
  was added.
- LSP reference closure covered all ten handlers. Before movement each
  resolved to `router.rs`, its definition, and any direct module tests; after
  refreshing zero-error diagnostics, every `router.rs` reference disappeared
  and the set contains only the colocated registration, definitions, and
  existing module tests. Server library passes `347 passed / 2 ignored`; the
  existing page/entity-shadow integration passes `10 / 10`; truth manifest
  and Rust M5 pass `16 / 16` and `1 / 1`; Python inventory remains
  `191 / 55-50-86 / exposure 22`; core/server all-target Clippy with warnings
  denied passes. The generator-owned inventory delta is exactly the two
  direct-connection addresses shifted by the new helper:
  `visible_page :49→:73` and `ensure_page_is_active :69→:93`. The M5 ceremony
  proof for Page Map response assembly moves from
  `page_map_routes.rs:160-185` to `page_map_routes.rs:190-214`.
- R5 registration slice 6 REVIEW, 2026-07-30: the fresh Sol reviewer first
  returned **BLOCK** because the M5 ceremony still cited the pre-helper Page
  Map response range. After that citation moved to the exact
  `build_map_response` body at `page_map_routes.rs:190-214` and the prose-only
  shift was recorded here, closure review returned **APPROVE** with no
  remaining finding. It independently confirmed the ten exact bindings and
  composition position, D2-clean bodies and CAS behavior, unchanged protected
  manifests, module-local LSP references, full typed route evidence, and that
  the fixture's `knowledge_path=None` cannot write the user's live vault.

### R6 — `post_write` phase decomposition

Begin only after the M5 exact-base and truth-state reader/write paths have
settled. Preserve the one canonical facade.

### R7 — daemon startup and scheduler lanes

Separate orchestration from phase/lane implementations without changing startup
or scheduling order.

### R8 — instruction and verification-surface cleanup

Run after the code boundaries have stabilized so the documentation describes
the resulting architecture rather than predicting it.

- Reconcile or explicitly mark historical the stale
  `m5-reader-manifest-inventory.md` “Draft 5” prose/table
  (`190 / 54-50-86`) against the executable current-tree inventory; never
  hand-edit the generated reader rows.

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

Scoped R4 re-gate, 2026-07-29:
**APPROVE-WITH-FIXES → APPROVE.** LSP/AST discovery showed that the final
eleven production locks were not one atomicity shape, so R4-17 through R4-24
now preserve eight independently testable DB/projection contracts and R4-25
owns the test seam. Fable verified `300 = 11 production + 289 tests`, every
slice delta, and the three current lock topologies. Its sole blocker was the
initially unstated R4-24 inbound capability contract: the caller-acquired
locked rename projection session and `RepairArtifactStore` must flow into the
named verification operation. That contract is now explicit above; the narrow
follow-up returned **APPROVE** with no new blocker.

Scoped R4-24 split re-gate, 2026-07-29: **APPROVE.** Sol found that the
original R4-24 combined movement with new crash/dual-lock controls, contrary
to D2, and LSP disproved R4-20's promise that the shared rename capture helper
could become private. Fable independently re-ran the helper reference lookup,
approved the R4-24a movement / R4-24b RED-first control split, and accepted the
one explicit mutex-lifetime change because the existing at-least-once receipt
state and outer repair locks contain the new window. Re-open the design only
if that post-commit/pre-clear window proves non-idempotent or R4-24b cannot add
its `#[cfg(test)]` seam without reordering production statements.

Scoped R4-25 re-gate, 2026-07-29:
**APPROVE-WITH-FIXES → APPROVE.** AST plus LSP corrected the old
`289`-literal subset to `342 = 340 test + 2 production` raw-field references.
Fable independently verified both production `_db` callers, all five external
`try_lock` observations, fresh-observer and read-only transaction semantics,
the independent R4-24b contender, and that `syn` / `proc-macro2` are already
lockfile-resolved. Its required clarifications now state that standalone
libSQL fixtures not derived from `MemoryDB` are outside this capability
boundary and reconcile the exact `46` test `_db` shapes. Fable authorized
R4-25a immediately and the remaining groups after those document fixes, with
no further re-gate required.

Scoped R5 re-gate, 2026-07-30:
**APPROVE-WITH-FIXES → APPROVE.** Fable confirmed that Rust function
item identity captured by `TrackedMethodRouter` is a real independent oracle,
that a `#[cfg(test)]` synthetic-router escape cannot weaken production, and that
the wrong-handler/missing/duplicate/main+repair controls plus the movement
order satisfy D2/D6/D7. It required exact handler identities to be declared
toolchain-scoped and the stale generation-0 premise to be corrected.

Both corrections are now in the R5 contract. Current-main verification also
showed that PR-D had already fixed the branch-local activities mismatch, so
`383a2f5d` integrates PR-D before R5 and preserves its adapter in the activity
slice. The merge resolution moved PR-D's two page permits into R4's new repair
children before the same DB mutexes, removed raw production projection helpers,
and passed `4195` workspace library tests with `35` ignored, `114` drift guards,
the projection-permit source scan, core/server warnings-denied Clippy, LSP
diagnostics, and a fresh Sol **APPROVE**.
The narrow Fable follow-up verified both corrections against `383a2f5d`,
confirmed the PR-D activity adapter in code, and authorized R5-0.

### Intermediate PRs — Sol by default, not Fable

Every PR receives a focused Sol opinion or diff review against:

- the exact PR scope;
- the locked decisions above;
- the relevant SQL/API/transaction/test invariant;
- verifier evidence and positive controls.

Sol may identify a design-level contradiction. If resolving it would change a
locked decision or PR sequence, work stops and returns to Fable gate 1. Opus
may be used for an exceptional escalation, but is not the routine intermediate
reviewer. Otherwise Fable is not used for intermediate PRs.

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
- **Sol reviewer:** provides the routine intermediate independent opinion/diff
  review; Opus is an exceptional escalation only.
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

### 2026-07-28 — PR-B/PR-C baseline refresh

- Execution baseline advanced from PR-A `e4790ce8` through release main
  `544965a1` to `07afba7d`, containing PR-B `f29c2c54`, PR-C `3932e3d5`,
  and the unrelated test-config isolation fix #411.
- D5 now permits bounded R2-through-R7 work, but the in-flight PR-D ceremony
  remains a separate hot-file owner and `truth_cutover_generation` stays 0
  until its fenced command receives explicit user approval.
- R0 was regenerated rather than hand-merged: `191` reader rows,
  `55 / 50 / 86`, `22` exposures. Its external connection floor was refreshed
  to `336 / 59` to include exactly three PR-C test-support files.
- R1 was mechanically replayed from the refreshed main body: `45,687` inline
  test lines moved to `db/main_tests.rs`, with rustfmt-normalized reconstruction
  equality before R2 began.

### 2026-07-29 — intermediate reviewer correction

- User correction supersedes the initial review policy: routine intermediate
  slices use Sol, not Opus on every slice.
- Fable remains limited to the frozen-design gate and final whole-system Gate
  2. Opus is available only for an exceptional judgment escalation.

### 2026-07-29 — R4-24 movement/control split

- LSP reference closure corrected the stale R4-20 promise that
  `capture_rename_page_title_on_connection` could become private at R4-24:
  three apply/recovery callers remain in `db/repair_page_rename.rs`, and the
  verification caller moves into a second DB child. The helper remains a
  narrow `pub(crate)` seam shared only by those children.
- Sol review found that R4-24's required new commit-failure and dual-lock
  checkpoints contradicted D2's movement-only boundary. R4-24 is split into
  R4-24a mechanical critical-section movement and R4-24b RED-first,
  `#[cfg(test)]` crash/lock teeth from the committed green movement baseline.
- R4-24a preserves the manual transaction and at-least-once terminal-receipt
  model. Its only explicit lock-lifetime change is release of the DB mutex
  after child commit/return and before caller-owned pending cleanup; outer
  manifest/tag locks and the borrowed rename session remain live.
- Fable's narrow gate-1 re-review returned **APPROVE** after independently
  confirming the LSP reference set and the movement/control boundary.

### 2026-07-30 — R4-25 lint census correction

- The exact raw manifests at `f8b791f6`, `4e0c9bf7`, and `a4edf9f4` each
  contain `71`, not `70`, `PrimaryConnLock` identities under
  `crates/wenlan-core/src/lint/**`. No lint identity was assigned to another
  fixture group.
- The final lower-risk group is consequently `111`, not `112`
  (`289 = 107 repair + 71 lint + 111 remaining`). This corrects a prose
  arithmetic error only; the lint-family boundary, migration order, protected
  contracts, and review gates are unchanged, so Fable gate 1 remains valid.

### 2026-07-30 — R5 executable handler boundary

- Rebaselined the server surface after current-main integration at `383a2f5d`:
  `95` public handlers remain in `memory_routes.rs`; the truth manifest has
  `167` static HTTP rows expanding to `171` runtime builder/method/path rows.
- Added R5-0 before movement because the existing exact route-set gates cannot
  detect binding the correct route to the wrong handler. The proposed
  `TrackedRouter` handler-identity manifest is an independent set-equality
  contract with positive and mutation controls.
- Froze the movement order from existing handler modules through low-coupling
  memory families, memory core, and finally the M5-protected page lane. One
  writer owns the hot route and handler files.
- Found that the branch-local `GET /api/activities` mismatch was stale relative
  to merged PR-D `d1fb5d9f`, which adds
  `redact_page_activity_detail`. R5 now requires PR-D integration first and
  moves that adapter call byte-for-byte with the handler.
- Corrected the generation premise from the stale plan: the live production DB
  reads generation `1` and fence `2:committed`; fresh databases still default
  to `0`. R5 mutates neither state.
- Declared exact handler names toolchain-scoped: a toolchain-wide manifest
  refresh stays exact and isolated instead of weakening identity comparison.
- This adds an executable boundary and PR sequence, so it is a material design
  change. Fable returned `APPROVE-WITH-FIXES`; both required corrections landed,
  and the narrow follow-up returned `APPROVE`, unblocking R5-0.
