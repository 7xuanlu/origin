# M4 gate-stage receipt — Gate 1 + Gate 2.1–2.3

**Date:** 2026-07-26
**Branch:** `kg-m4-communities-shadow`
**Stage-0 base commit:** `430ee7d2`
**Verdict:** Gate 1 is
`PROVISIONAL PASS — 1.1–1.3 and 1.5 MET; 1.4 SELECT preflight MET, real job pending PR-1`.
Gate 2 is
`PROVISIONAL PASS — 2.1–2.3 MET, 2.4 pending PR-1`.

Gate 1.4 keeps its authored cumulative mutex and transaction-structure bars. Gate 2.4 keeps
its authored `≤ 1 cycle` common / `≤ 2 cycles` worst-case bar. Per the sequencing rulings
in `2026-07-25-m4-gate-criteria.md`, the first PR-1 persistence commit must be the RED
integration test against the real lease + generation-CAS job and published
`community_members` snapshot. Neither gate is final until that test is GREEN.

## Candidate and corpus

- `leiden-rs = "=0.8.1"`, `default-features = false`
- sequential execution, fixed seed `0x4D34`, stable lexicographic node indexing
- 3 spaces
- 18,453 active grounded `relates` edges
- 6,144 participating entities (2,048 per space)
- DB scale: 100,000 memories / 5,000 pages
- participation fraction: 100% in each seeded space (reported, non-gating)

## Gate 1 receipt

Command:

```text
cargo test --release -j 1 -p wenlan-core --test m4_community_gates \
  m4_partition_scale_churn_and_poison_receipt -- --ignored --nocapture
```

Result:

| Sub-gate | Authored bar | Measured | Verdict |
|---|---:|---:|---|
| 1.1 full partition p95 / space | ≤ 10 s | 1.634 ms / 1.525 ms / 1.489 ms | MET |
| 1.1 hard fail | no run > 30 s | no run > 30 s | MET |
| 1.2 warm p95 / full p95 | ≤ 10% | 0.023 ms / 1.634 ms = 1.377% | MET |
| 1.2 hard fail | < 50% | 1.377% | MET |
| 1.2 fixed-frontier 2,048→32,768 nodes | ≤ 3× | 2.458 µs → 1.792 µs | MET |
| 1.2 dirty-set 8→64 nodes @ 32,768 | ≤ 12× | 2.209 µs → 20.208 µs (9.15×) | MET |
| 1.2 expanded-frontier cap | > 25% routes full | 100% high-degree-hub frontier rejected | MET |
| 1.3 fresh-child peak additional RSS | ≤ 256 MiB | 2,048,000 bytes | MET |
| 1.3 hard fail | ≤ 1 GiB | 2,048,000 bytes | MET |
| 1.5 modularity | Leiden ≥ label-prop | MET on every space/run | MET |
| 1.5 connectedness | 0 disconnected | 0 | MET |

The clean incremental frontier max was 0.732% of the space. Reusable incremental state
updates global modularity totals only from the dirty endpoints' old/new incident edges;
graph-wide initialization is outside steady-state cycles. The first negative
fresh-full telemetry delta was diagnosed: the frontier had zero remaining improving
single-node moves while 32 improving moves existed outside it. This is the expected
bounded-frontier/local-basin tradeoff, not an authored failure. The load-bearing
incremental invariant is monotonic global modularity versus carrying forward the prior
membership on the changed graph. Fresh-full delta remains reported telemetry; the minimum
observed clean delta was `-1.027e-3`.

DB mutex command:

```text
cargo test -j 1 -p wenlan-core --lib \
  m4_grounded_projection_select_mutex_receipt -- --ignored --nocapture
```

DB result:

| Sub-gate | Authored bar | Measured | Verdict |
|---|---:|---:|---|
| 1.4 SELECT mutex p95, 20 firings | ≤ 500 ms | 8.082 ms | PREFLIGHT MET |
| 1.4 SELECT hard fail | no firing > 2 s | max 8.387 ms | PREFLIGHT MET |
| 1.4 access path | partial grounded index | `idx_edges_active_grounded_space_type` | MET |
| 1.4 preflight structure | compute outside SELECT guard | SELECT guard dropped before projection/partition | PREFLIGHT MET |
| 1.4 real lease + SELECT + finalize CAS | ≤ 500 ms cumulative; no compute in txn | pending first PR-1 RED test | PROVISIONAL |

The original `ORDER BY edge_id` query chose the primary-key index and failed the access
path assertion. Removing SQL ordering made SQLite use the partial grounded-space index;
the projection's existing deterministic `edge_id` sort runs after the DB guard is dropped.
This is not a claim about the not-yet-written lease/finalize path.

## Gate 2.1–2.3 receipt

| Sub-gate | Authored bar | Measured | Verdict |
|---|---:|---:|---|
| 2.1 frozen determinism | 0 membership churn | byte-identical across 10 reruns | MET |
| 2.2 clean churn mean | ≤ 2% | 0.0186% | MET |
| 2.2 clean churn p95 | ≤ 10% | 0.1465% | MET |
| 2.3 poison fraction | 5% stand-in | 5.004% | MET |
| 2.3 poison churn mean | ≤ 4% | 0.0195% | MET |
| 2.3 community-count floor | ≥ 70% of clean | 36 poison / 33 clean | MET |

The maximum poisoned frontier was 8.594% because one incremental edge touched the
deliberately poisoned high-degree hub. That does not change Gate 1.2's clean `≤ 1%`
frontier timing corpus; it is retained here as poison telemetry.

## Mutation proof

Each mutation was applied temporarily, the named focused test was observed RED, and the
production code was restored before the green runs above.

| Mutation | Expected tooth | RED evidence |
|---|---|---|
| `LeidenConfig.seed: Some(config.seed) → None` | deterministic partition | strengthened ring corpus differed on rerun 1 |
| parallel cap → `f64::INFINITY` | pair cap | projected parallel weight became 4.0, expected 3.0 |
| hub factors → `1.0` | soft normalization | hub edge weight became 1.0, expected 2/3 |
| per-call global incremental rebuild | frontier locality | fixed frontier grew 15.73× from 2,048 to 32,768 nodes |
| omit expanded-frontier check | bounded frontier | one dirty hub expanded to 100% and was incorrectly accepted |
| omit retraction-touched connectivity check | connected communities | sole-bridge retraction returned `Ok` with one disconnected community |

## Fable rulings

Fable signed off on five load-bearing decisions:

1. exact-pin `leiden-rs 0.8.1` with default features disabled; the crate is authorized for
   full seeded partitions only, while Wenlan owns the true bounded frontier optimizer;
2. Gate 2.4 is RED-first on the real PR-1 publication path rather than “proven” with a
   disposable SQLite fake;
3. incremental quality is gated by monotonic modularity versus carried-forward membership.
   Fresh seeded full is a telemetry comparator, because two valid greedy starts can land in
   different local basins. Running fresh full every cycle is prohibited by Gate 1.2.
4. Gate 1.2's original PASS was blocked: the fixed-size timing did not prove the authored
   scaling sentence. The repaired gate observes fixed-frontier multi-size cost and an
   expanded-frontier cap; it gates behavior rather than mandating a particular data structure.
5. Gate 1.4 is provisional under the same RED-first sequencing exception as Gate 2.4, and
   Gate 1.3 uses a fresh child first-operation peak rather than a warmed-parent baseline.

## Independent review disposition

The first independent review returned `FIX-FIRST` on Gate 1.2 scaling, Gate 1.3 RSS
measurement, and the overclaimed Gate 1.4 receipt. The re-review closed all three and found
one new blocker: an internal bridge retraction could disconnect a prior community without
an accepted node move, skipping the postcondition. A RED bridge-retraction test reproduced
it; the state now checks communities whose internal edge was removed and routes a
disconnected result to full repartition. Gate 1.4 remains honestly provisional until the
real PR-1 job test turns GREEN.

## 2026-07-26 PR-1 closure addendum

The production composer is now implemented. The post-review closure run produced:

- complete M4 target: `18 passed; 0 failed; 4 ignored`;
- real composer p95 `244.089 ms`, max `285.747 ms`, `8,201` input rows loaded,
  `2,048` member rows written, DB-mutex p95 `57.589 ms`, foreground p95 `139.260 ms`;
- Gate 1.2 with the production rollback-journal path included: fixed frontier
  `18.750 µs` at 2,048 nodes and `9.833 µs` at 32,768 nodes; dirty set 8→64 at
  32,768 nodes measured `50.458 µs`→`322.000 µs`;
- deterministic abort after the finalize member-delete proved transaction rollback,
  connection reuse, token-guarded lease cleanup, runtime restoration, and an incremental
  retry;
- raw grounded INSERT/DELETE/topology UPDATE each advanced `graph_generation` and routed
  through `Full(DirtyFrontierMissing)` rather than reusing stale core state; concurrent
  same-space phase calls produced exactly one publisher and one clean skip;
- the supported supersede/reactivate/entity-merge collision first reproduced the erased
  generation bump (`13`, expected `14`), then passed with exact transition accounting and
  `Full(NodeOrderChanged)`;
- a controlled overlapping G/G+1 publication first reproduced the stale runtime overwrite
  (`32`, expected `34`), then passed with generation-aware shared-runtime installation;
- strict Clippy, formatting, diff whitespace, the 26-test CI planner suite, and the 6-test
  release-verifier suite all passed. The first full-core run had one transient libSQL
  `PRAGMA page_count` API-misuse during a test fixture's migration setup; the focused test
  immediately passed, and a second normally parallel full run exited 0 with core lib
  `3,074 passed; 0 failed; 33 ignored` plus every integration target green.

Gate 1.3 remains a partition-only RSS receipt. The production composer reports elapsed time
and row counts, but PR-1 does not measure or claim whole-snapshot composer peak RSS.

## 2026-07-26 PR-2 cutover addendum

**Branch:** `kg-m4-communities-cutover`
**Stack base:** `kg-m4-communities-shadow` at `a5d985bc`
**Review status:** final blocker closure implemented; CI-owned M4 gate verified;
Sol closure re-review pending

PR-2 keeps every frozen M3 response unchanged and exposes communities only through new,
versioned typed surfaces:

- `GET /api/communities` — paginated community summaries, entity-space scoped;
- `GET /api/communities/members` — independently paginated membership rows,
  entity-space scoped;
- `GET /api/communities/page-assignments` — independently paginated routing rows,
  parent-page-workspace scoped;
- `GET /api/communities/proposals` plus generation-guarded accept/reject actions.

The two daemon-side readers cut over independently. Each requires a current publication
receipt whose membership digest matches the live `community_members` snapshot, a current
published generation, and zero unexplained drift in every relevant space. Supported entity
assignment and memory-lifecycle writes delete the affected parity rows; the production
`CommunityDetection` phase republishes dirty membership/routing work and reconciles pending
reader proofs. Either consumer can be flipped back to `entities.community_id`.

### PR-2 hard-gate receipt

| Gate | Current-tree evidence | Verdict |
|---|---|---|
| 4.6 per-consumer differential cutover | `community_reader_reconciliation_flips_equivalent_consumers_and_is_reversible`: both readers flip only after current zero-drift receipts; entity and memory mutations close the gate; production reconcile restores it; same-coverage membership corruption fails closed | MET |
| 4.7 routing hysteresis | `exact_three_way_hysteresis_assigns_holds_and_drops`: exact `0.50` assign, exact `0.30` hold, below `0.30` drop; routing reads native active `mentions`/`cites` edges plus direct entity ownership | MET |
| 4.8 proposal-only relabeling | `finalizing_a_split_queues_review_without_overwriting_curated_names` and `community_rename_accept_is_generation_guarded_and_absent_from_legacy_queue`: membership applies structurally, curated names survive publication, action/payload mismatch is rejected, and rename changes only after a current-generation accept | MET |
| 4.9 hard-zero consistency | `community_consistency_detects_a_missing_connected_participant` proves the oracle detects an omitted still-connected participant; the graded receipt reports `5,000` assigned pages, `0` stale writes, and `0` consistency violations | MET |
| 4.10 indexes at scale | the graded receipt accepts `idx_edges_active_grounded_space_type` and `idx_community_members_community` over `100,000` memories / `5,000` pages | MET |

The graded command and receipt:

```text
cargo test -p wenlan-core --test m4_community_gates \
  m4_pr2_routing_consistency_and_index_scale_receipt -- --ignored --nocapture

[m4_pr2_scale] memories=100000 pages=5000 routed_pages=5000 \
  stale_writes=0 consistency_violations=0 indexes=accepted
```

The corpus now forces all `5,000` pages through real native `cites` edges into two
same-community entity votes. The earlier all-dropped corpus is no longer accepted as a
routing receipt.

### Sol preliminary review disposition

Sol's first integrated review returned `CODE READY: NO` with six findings. The current tree
addresses each before final re-review:

1. reader parity now has production phase wiring and invalidates on all supported legacy
   assignment and memory-lifecycle inputs;
2. structural differences are classified as explained only after an atomically published
   membership digest proves the live durable snapshot;
3. summaries, members, and page assignments have separate pagination and correct
   entity-space versus page-workspace scope;
4. routing consumes native `edges` `mentions`/`cites`, not legacy `page_sources`;
5. consistency independently re-derives expected participants, and the graded corpus
   exercises nontrivial assigned routes;
6. proposal listing and acceptance require the queue action to agree with the typed
   payload action.

The fresh-eye Sol gate then returned `CODE READY: NO` with five Important findings.
The current tree carries a RED-first closure for each:

1. `clean_community_cycle_retries_new_or_refreshed_page_routes` now proves that
   page source refresh/direct ownership, native routing-edge changes, and memory ownership
   changes invalidate persisted routes before the normal clean-cycle retry;
2. `community_reader_reconciliation_flips_equivalent_consumers_and_is_reversible`
   no longer invokes the reconciler directly after same-generation membership corruption:
   membership triggers remove cached parity, and the production pending-reader hook detects
   and records the invalid digest;
3. `community_member_fence_rejects_a_cross_space_community_id` rejects new cross-space
   membership, while `community_consistency_counts_a_stored_cross_space_membership`
   proves the oracle catches already-stored corruption with same-space ownership joins;
4. proposal rejection now runs typed action/payload agreement and the same clean-generation
   CAS as acceptance; stale and mismatched rows remain open in
   `community_rename_accept_is_generation_guarded_and_absent_from_legacy_queue`;
5. M96 startup inspection verifies durable table schema signatures and the scale-critical
   indexes/trigger bodies. Recomputable malformed indexes/triggers repair transactionally;
   malformed durable table shapes fail loudly. The two named migration-96 regressions prove
   both paths.

The next Sol pass found three remaining issues. The current tree disposes them as follows:

1. **Canonical centroid ownership:** entity-poor routing now derives centroid inputs only
   from active entity-shadow pages joined through canonical `entity_page_map` ownership;
   `entity_shadow_centroid_routes_and_invalidates_entity_poor_pages` proves both routing
   and invalidation after a canonical shadow embedding change.
2. **Snapshot/publish input race:** every assignment stores and checks both the page-local
   `page_community_route_inputs.generation` and the per-space
   `community_route_space_inputs.generation`, in addition to page version and published
   community generation. `route_publish_rejects_snapshot_invalidated_before_publish`
   pauses after snapshot, mutates a routing input, proves the stale publish is rejected,
   preserves the prior assignment, and proves the normal stale-route retry catches up.
3. **Broad invalidation:** page-local inputs advance only for affected pages; global
   centroid/membership inputs advance one O(1) per-space epoch. Membership replacement
   while `space_graph_state.dirty=1` does not fan out through per-page generations, while
   a same-generation direct membership change advances the space epoch exactly once.
   Unrelated memory writes and metadata-only entity-shadow synchronization leave both
   relevant generations unchanged.

The M96 upgrade backfills the two route-input substrates, adds both assignment-generation
columns to the known older shape, repairs stale input-space ownership, and stamps existing
assignments with a deliberately stale per-space generation so they are recomputed. The
startup shape guard covers the new tables, columns, index, and trigger predicates.

The deterministic source-page archive repair is the one path whose target status update
legitimately fires a derived route invalidation. Its effect guard snapshots the target
page's local and space generations, then allows exactly one extra database change only
after proving `page generation = old + 1`, unchanged space, and unchanged space generation.
`source_page_archive_cas_changes_only_status_and_clears_lint_findings` exercises the
successful guarded repair and continues to prove that only target status changes at the
domain level.

Sol closure sign-off remains a separate gate. These three dispositions are awaiting the
final Sol closure re-review; this receipt does not pre-claim approval.

The final blocker pass found three routing correctness gaps and two independent
default-parallel harness races. The focused routing regressions failed before the fix:

1. `migration_96_upgrades_the_exact_pre_generation_assignment_shape` published `held`
   instead of `dropped` for a real current `0.40` candidate because the upgraded
   `routing_space_generation=-1` assignment was accepted as a hysteresis prior.
2. `clean_route_retry_refreshes_only_the_selected_page` reported
   `pages_considered=3` instead of `1`, proving that a page-local stale retry rewrote
   every active assignment in the space.
3. `public_page_assignment_reader_hides_obsolete_route_versions` continued to expose
   the target assignment after changing only its route version to
   `obsolete-route-version`.

The routing closure keeps `refresh_page_community_routes(space, generation)` as the
whole-space publication refresh. The clean stale selector now retains the selected
`page_id` and invokes an internal target-page snapshot/score/publish path. Page,
page-entity, and prior-assignment reads are target-scoped there; membership, grounded
weights, and centroids remain space-scoped scoring inputs. A prior is eligible only when
its route version is current and both routing generations are non-negative. The public
page-assignment SQL also requires the current route version, including the corrected
parameter position for scoped reads.

Two separate normally parallel full-core runs had previously failed in independent
libSQL backup operations:

- `test_entity_enrichment_slice_abandons_after_attempt_cap` returned
  `online_backup integrity: SQLite failure: bad parameter or other API misuse`;
- `delete_entity_deletes_shadow_page` returned
  `PRAGMA page_count: SQLite failure: bad parameter or other API misuse`.

The backup implementation now takes a process-global async mutex only under `cfg(test)`
for the complete `online_backup` duration. Production backup locking and behavior are
unchanged. `repair_projection_lock_covers_the_whole_transaction` retains its load-bearing
in-closure cross-thread exclusion proof; its redundant final naked
`PROJECTION_WRITE_LOCK.try_lock()` assertion was removed because another legitimate
parallel test can own the global lock after the closure returns. The exact three formerly
failing tests each pass focused, and the required default-parallel full-core gate is now
green.

### Fresh verification

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| three new focused routing regressions, exact-name invocations | each `1 passed; 0 failed` |
| exact three formerly failing tests, exact-name invocations | each `1 passed; 0 failed` |
| `cargo test -p wenlan-core --tests --quiet` | exit 0 under default parallelism; core lib `3,100 passed; 0 failed; 33 ignored`; every core integration target green |
| `cargo test -p wenlan-server --no-fail-fast --quiet` | exit 0; unit target `311 passed; 0 failed; 2 ignored`; every integration target green |
| `cargo test -p wenlan-mcp --quiet` | exit 0; unit target `210 passed; 0 failed`; every integration target green |
| `cargo test -p wenlan-core --lib drift_guard::doc -- --nocapture` | `2 passed; 0 failed` |
| `git diff --check` | exit 0 |
| graded 100k/5k command above | `1 passed; 0 failed`; `memories=100000 pages=5000 routed_pages=5000 stale_writes=0 consistency_violations=0 indexes=accepted` |
| `cargo test -p wenlan-core --test m4_community_gates -- --nocapture` | `18 passed; 0 failed; 5 ignored`; `composer_p95_ms=369.459`, `composer_max_ms=372.995`, `mutex_p95_ms=183.057`, `foreground_p95_ms=148.083` |

During closure, the first full run exposed a stale `58`-route count in
`space_scoping_e2e`; the four new scoped community routes make the exact contract
`62 = 15 global + 47 scoped`. The focused regression and the subsequent complete
workspace run passed. The later independent backup failures were not dismissed as
transient; the test-only serialization closure and fresh default-parallel full-core result
above replace the earlier false-green claim.

The final Sol blocker found that the clean stale selector still sent every stale cause
through the target-page helper. A bump to
`community_route_space_inputs.generation` immediately hides every assignment carrying
the prior epoch from public reads, but one phase firing repaired only the first selected
page. The same whole-space meaning applies to a community published-generation mismatch
and a route-version mismatch.

The RED-first regression
`space_generation_retry_refreshes_all_active_page_routes` creates three active routed
pages in one clean space, including two named concept-page assignments. It bumps only the
space-wide route generation and proves both named assignments become non-public. Before
the production edit, the exact focused command failed with
`left: 1`, `right: 3` at the `pages_considered` assertion:

```text
cargo test -p wenlan-core --lib \
  space_generation_retry_refreshes_all_active_page_routes -- --nocapture

test db::tests::space_generation_retry_refreshes_all_active_page_routes ... FAILED
test result: FAILED. 0 passed; 1 failed
```

The selector now joins the raw assignment by `page_id` and classifies its stale cause
explicitly. Existing assignments with a space-route generation, clean published
generation, or route-version mismatch are ordered ahead of local work and dispatched to
the existing whole-space `refresh_page_community_routes`; this includes the M96
`routing_space_generation=-1` migration sentinel. Only a missing assignment or a
page-version/page-input-generation mismatch dispatches to the internal target-page helper.
The query retains active-page and clean/current-publication guards plus deterministic
space/page ordering; both publication paths retain the existing generation CAS.
`clean_route_retry_refreshes_only_the_selected_page` remains the negative control.

The macOS CI owner is the exact command at `.github/workflows/ci.yml:653-658`:
`cargo nextest run -p wenlan-core --test m4_community_gates`. Nextest runs each test in
its own process; it does not require unrelated sibling tests to share one libtest process.
Two aggregate `cargo test` diagnostics did exceed the foreground p95 bar while sibling
tests ran in the same process, at `594.512834ms` and `1.09689075s`. Those failures remain
recorded below as diagnostics. They are not the CI-owned gate. The isolated measured
Gate 1.4 control passed with `foreground_p95_ms=384.081` and
`mutex_p95_ms=187.860`, and two clean-boundary executions of the actual CI command
(root Sol plus the repeatability run below) each passed all 18 scheduled tests.

The final focused and complete evidence for this closure is:

| Command | Result |
|---|---|
| `cargo test -p wenlan-core --lib space_generation_retry_refreshes_all_active_page_routes -- --nocapture` | `1 passed; 0 failed`; `pages_considered=3` contract satisfied and both named assignments public afterward |
| `cargo test -p wenlan-core --lib clean_route_retry_refreshes_only_the_selected_page -- --nocapture` | `1 passed; 0 failed`; local retry retained `pages_considered=1` and the unrelated row byte-for-byte |
| `cargo test -p wenlan-core --lib migration_96_upgrades_the_exact_pre_generation_assignment_shape -- --nocapture` | `1 passed; 0 failed`; sentinel prior stayed ineligible for hold-band hysteresis |
| `cargo test -p wenlan-core --lib public_page_assignment_reader_hides_obsolete_route_versions -- --nocapture` | `1 passed; 0 failed`; obsolete route version hidden and restored current version public |
| `cargo fmt --all -- --check` | exit 0 |
| `git diff --check` | exit 0 |
| `cargo test -p wenlan-core --tests --quiet` | exit 0 under default parallelism; core lib `3,101 passed; 0 failed; 33 ignored`; every core integration target green |
| `cargo test -p wenlan-server --no-fail-fast --quiet` | exit 0; unit target `311 passed; 0 failed; 2 ignored`; every integration target green |
| `cargo test -p wenlan-mcp --quiet` | exit 0; unit target `210 passed; 0 failed`; every integration target green |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo test -p wenlan-core --test m4_community_gates -- --nocapture` | aggregate diagnostic **FAILED twice** at `m4_real_job_mutex_and_correction_latency_gate`: foreground p95 `594.512834ms`, then `1.09689075s`, above the `500ms` bar; each run had the other `17 passed; 5 ignored` |
| `cargo test -p wenlan-core --test m4_community_gates m4_real_job_mutex_and_correction_latency_gate -- --nocapture` | isolated Gate 1.4 control passed with `foreground_p95_ms=384.081` and `mutex_p95_ms=187.860` |
| `cargo nextest run -p wenlan-core --test m4_community_gates` (root Sol) | exit 0; `18 passed; 5 skipped`; latency test passed |
| `cargo nextest run -p wenlan-core --test m4_community_gates` (repeatability run) | exit 0; `18 tests run: 18 passed, 5 skipped`; latency test passed in `32.285s` |
| `cargo test -p wenlan-core --test m4_community_gates m4_pr2_routing_consistency_and_index_scale_receipt -- --ignored --nocapture` | `1 passed; 0 failed`; `memories=100000 pages=5000 routed_pages=5000 stale_writes=0 consistency_violations=0 indexes=accepted` |
| `cargo test -p wenlan-core --lib drift_guard::doc -- --nocapture` | `2 passed; 0 failed` |

Final Sol closure re-review: **APPROVE / CODE-READY: YES**, with no blocking
findings. The only residual risk is that clean-state direct/manual multi-row
`community_members` writes still increment the space epoch once per row; no
current production writer uses that path because production membership
replacement runs while `dirty=1`.
