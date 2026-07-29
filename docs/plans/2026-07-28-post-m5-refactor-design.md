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

- Add one narrow `MemoryDB` receipt-read method for the existing
  `repair_target_receipt_on_connection` mechanic. It returns the same
  `(RepairDigest, u64)` and preserves exact errors and lossy decoding without
  exposing a connection, guard, callback, SQL string, or transaction surface.
- Replace only the independent lock in
  `recover_complete_entity_extraction_apply_receipt`. Do not rewrite
  `target_receipt_current`, or replace same-transaction uses in post-write or
  verification: those callers must continue to read through the connection
  that already owns their atomic operation.
- Preserve the caller's ordering: pending artifact parse, receipt read, guard
  release, then publish/remove/sync filesystem work. A concurrency control
  must prove the DB mutex is released before the artifact operation.
- GREEN floor: external literals `300 → 299`, production `11 → 10`, tests
  remain `289`.

#### R4-18 — memory repair CAS transactions

- Move the complete transaction bodies of `reclassify_memory_cas_inner` and
  `complete_entity_extraction_cas_inner` behind two named `MemoryDB` methods in
  one DB child. Caller facades and test hooks retain their current public or
  crate-visible paths.
- Preserve each exact `BEGIN IMMEDIATE`, validation and receipt order,
  `total_changes` normalization, proof construction, hook-before-commit
  position, forced rollback-failure path, commit handling, and mutex lifetime.
  Sharing a private rollback/proof helper is allowed only if normalized moved
  bodies remain mechanically equivalent; the two operations do not become a
  generic CAS executor.
- Direct controls must cover stale target, successful proof, hook failure,
  ordinary rollback, forced rollback failure, and a blocked concurrent writer
  across the pre-commit hook.
- GREEN floor: external literals `299 → 297`, production `10 → 8`, tests
  remain `289`.

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
- Keep `page_on_connection` and rename capture/match helpers private to the DB
  implementation boundary after the move. No projection object, DB guard, or
  transaction token may escape the named operation.
- Existing crash-window and restore tests remain required. Add a barrier that
  proves a concurrent DB writer cannot enter while the operation is between
  its DB read and projection write/restore, and an order control that fails if
  DB locking moves before projection-session locking.
- GREEN floor: external literals `296 → 294`, production `7 → 5`, tests remain
  `289`.

#### R4-21 — regenerate-page projection compensation

- Move `regenerate_page_projection_cas` as one named cross-resource operation.
  It deliberately remains non-transactional.
- Preserve the opposite lock topology from rename: materialize the existing
  `get_page` precheck, acquire the DB mutex, read the projection page row and
  database digest, then acquire the repair projection lock while still
  holding the DB mutex. Hold it through scan, write, proof hook, and any
  snapshot restore; release only on the existing return path.
- Do not collapse this operation into the rename transaction or stale-page
  quarantine. Direct controls must prove the DB row/digest snapshot remains
  pinned across projection work, both rollback branches restore, and a
  concurrent DB writer stays blocked until the projection operation exits.
- GREEN floor: external literals `294 → 293`, production `5 → 4`, tests remain
  `289`.

#### R4-22 — stale-projection quarantine and recovery

- Move stale-projection quarantine and its apply-journal recovery together as
  two named operations. Preserve owner-absence CAS, journal load/persist/clear,
  pending receipt behavior, `restore_post`, recovery-state mapping, proof hook,
  and conditional snapshot restore.
- Preserve the current order and lifetime: acquire the DB mutex before the
  repair projection lock and retain it across owner check plus every journal
  and filesystem recovery/apply step. No effort in this movement PR may
  shorten that lifetime, even where the filesystem work is slow.
- Required controls cover owner appearing, crash windows before and after
  journal persistence, `Original`/`Post`/`Unknown` recovery, mutation-not-started
  versus restore-required failures, and a concurrent writer blocked across the
  filesystem phase.
- GREEN floor: external literals `293 → 291`, production `4 → 2`, tests remain
  `289`.

#### R4-23 — current repair target dispatcher

- Replace the one lock in `target_receipt_current` only after R4-20 through
  R4-22 provide the purpose-specific projection capture seams. The dispatcher
  retains target/writer branching, row counts, owner-absence checks, rollback
  parsing, and errors; DB and projection snapshots still occur under the same
  mutex lifetime as today.
- The default branch uses R4-17's receipt read. Projection branches call narrow
  owned-result methods; they may not receive a connection, guard, generic
  session, or callback.
- Branch-complete controls must cover default, ordinary page projection, and
  stale projection paths, including a barrier for each projection branch.
- GREEN floor: external literals `291 → 290`, production `2 → 1`, tests remain
  `289`.

#### R4-24 — repair verification atomic operation

- Move the DB/projection critical section of
  `record_repair_verification_inner` behind one named DB-owned verification
  operation. Keep manifest and tag-record locks, artifact loading, report
  prevalidation, and rename projection-session acquisition in their current
  caller order; the named operation then acquires the DB mutex, begins
  `IMMEDIATE`, rechecks DB/tag/target/projection state, persists the
  verification receipt under the same applicable projection lock, and commits
  before the caller clears the pending apply receipt.
- The caller-acquired locked rename projection session and
  `RepairArtifactStore` are sanctioned inbound parameters of this named
  operation. They must flow into the critical section so current acquisition
  order and in-transaction receipt persistence survive. R4-23's prohibition
  concerns raw or generic capabilities flowing out of `MemoryDB`; it does not
  forbid these repair-side projection/store capabilities from flowing in.
- Preserve rename, stale-projection, ordinary-projection, and nonprojection
  branches independently. Do not route same-transaction reads through R4-17's
  separately locking method and do not generalize the operation into a
  transaction callback.
- Required controls cover every target branch, report/target/non-target
  conflict before receipt persistence, persistence failure rollback, commit
  failure, pending-receipt clearing only after commit, and barriers showing
  both the DB mutex and applicable projection lock remain held across receipt
  persistence.
- GREEN floor: external literals `290 → 289`, production `1 → 0`, tests remain
  `289`. At this point the production raw-capability census is zero and its
  RED mutation control becomes a zero-baseline prohibition.

#### R4-25 — exact test-support seam and always-private connection

- First classify all remaining test raw-access shapes with AST plus LSP
  references; a literal count alone is not sufficient because multiline
  chains and alternate `_db.connect()` handles are part of the contract.
- Introduce one `#[cfg(test)]` DB-owned opaque test-support session. It may
  expose a generic test-only execute/query pair plus only the exact
  transaction-observation operations the fixtures require at R4-25 entry.
  Those calls are frozen by the exact AST call-site manifest; this is not a
  production SQL facade. The session never returns or dereferences to
  `libsql::Connection`, `MutexGuard`, `Database`, a raw transaction, or a
  callback argument. Its constructor and fields remain private, and the exact
  call sites are frozen in a location-aware AST manifest rather than admitted
  by filename or `#[cfg(test)]`.
- Move fixtures mechanically in bounded file groups. Do not combine fixture
  cleanup, assertion changes, test splitting, or production refactors with
  this migration. Each group must reduce the old raw-capability manifest and
  add only its exact named support calls.
- Make `MemoryDB::conn` and the alternate database handle private in every
  build only after all production and test callers compile through named
  seams. The final guard rejects direct, multiline, retained, renamed,
  dereferenced, or alternate-handle access outside `db/**`; positive controls
  must kill every recognized escape shape.
- Final floor: production raw capability `0`, external direct literals
  `289 → 0`, and the test-support manifest equals the call-site set measured
  at R4-25 entry exactly, plus only review-approved call sites added by
  R4-25's own positive controls. Run the uninterrupted workspace library suite
  after the last fixture group.

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
