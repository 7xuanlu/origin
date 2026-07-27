# M4 gate-stage receipt — Gate 1 + Gate 2

**Date:** 2026-07-26
**Branch:** `kg-m4-communities-shadow`
**Stage-0 base commit:** `430ee7d2`
**Closed:** 2026-07-27 on local live-tested candidate `597760ac`, with the subsequent
ambiguity repair still uncommitted.
**Verdict:** `PASS — Gate 1.1–1.5 and Gate 2.1–2.4 MET`.
**Publication boundary:** the live-tested candidate and subsequent ambiguity repair are
local-only. PR #395 remains at remote `a5d985bc`; the two repair commits and current
uncommitted changes are not included in that PR or its CI.

The provisional history below is preserved as recorded. Its pending Gate 1.4 and Gate 2.4
language is superseded by the 2026-07-27 live-closure addendum.

**Prior provisional verdict (2026-07-26):** Gate 1 was
`PROVISIONAL PASS — 1.1–1.3 and 1.5 MET; 1.4 SELECT preflight MET, real job pending PR-1`.
Gate 2 was
`PROVISIONAL PASS — 2.1–2.3 MET, 2.4 pending PR-1`.

Gate 1.4 kept its authored cumulative mutex and transaction-structure bars. Gate 2.4 kept
its authored `≤ 1 cycle` common / `≤ 2 cycles` worst-case bar. Per the sequencing rulings
in `2026-07-25-m4-gate-criteria.md`, the first PR-1 persistence commit had to be the RED
integration test against the real lease + generation-CAS job and published
`community_members` snapshot. Neither gate was final until that test was GREEN.

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

## 2026-07-27 live-closure addendum

### Repaired local candidate

- Worktree: `/private/tmp/wenlan-m4-pr1-live-a5d985bc`
- Branch: `kg-m4-communities-shadow`
- Live-test baseline, clean before this receipt-only edit:
  `597760ac8cf6bc60b6a064617f566e5a50dda344`
- `target/debug/wenlan-server` SHA-256:
  `d7d9ac5505039fd609b92972ee9184c49c2a1d211701bd962c90f344e2302203`
- Repair `b7c22ce6932f06577e33dcab24a54c1a8deb5ac7` holds undersized spaces
  without publishing an invalid snapshot and advances the durable selection cursor so a
  held space cannot starve a later viable space.
- Repair `597760ac8cf6bc60b6a064617f566e5a50dda344` preserves a document's
  effective space across semantic replacement.
- The subsequent uncommitted repair rejects an omitted-space replacement when the exact
  existing `(source, source_id)` rows disagree on space or contain any NULL space. The
  rejection occurs before page marking, projection invalidation, child deletion, or memory
  deletion; an explicit incoming space remains authoritative and can heal the old ambiguity.

The first positive live attempt exposed a real product bug rather than proving closure:
document replacement omitted the assigned space, reset the source to `unfiled`, and
invalidated the grounded projection. The daemon was formally shut down before repair.
The RED-first replacement repair proves that an explicit replacement space wins, a fresh
source with no space still lands in `unfiled`, derived episodes inherit the parent's
effective space, and supersession uses that same effective space.

An earlier independent review of the document-space repair through `597760ac` returned
`APPROVE`. That review preceded the later full unpublished-delta review, which found the
mixed-space replacement ambiguity. After the local fix, the independent full
unpublished-delta closure review returned `APPROVE`: all prior findings were resolved and
it found no new findings.
Fresh verification after the uncommitted ambiguity repair reported:

- focused conflicting-space regressions: `2 passed`;
- focused `replacement_` run: `15 passed`;
- first core library run: `3,079 passed; 1 failed; 33 ignored`, with the already-observed
  transient libSQL fixture failure
  `online_backup integrity: SQLite failure: bad parameter or other API misuse` in
  `reconcile_detects_map_row_whose_page_is_not_live`;
- immediate isolated rerun of that test: `1 passed`;
- fresh full core library rerun: `3,080 passed; 0 failed; 33 ignored`;
- M4 gate target: `20 passed; 4 skipped`;
- Clippy with `-D warnings`, formatting, and diff whitespace checks: green.

### Isolated live contract

- Data root: `/private/tmp/wenlan-m4-pr1-data-597760ac-v3`
- Watched source:
  `/private/tmp/wenlan-m4-pr1-source-597760ac-v3`
- Bind: `127.0.0.1:7879`
- Common environment: `WENLAN_LLM_DEVICE=auto`; entity sweep, document reconciliation,
  and citation backfill disabled; no synthetic idle or admission overrides.
- M3g leg: edge-grounding promotion enabled and Leiden grouping disabled.
- M4 leg: edge-grounding promotion disabled and Leiden grouping enabled.
- Routing config: `everyday_source=on_device`, `synthesis_source=on_device`,
  `on_device_model=qwen3-4b`.
- Health/version: `0.15.0+g597760ac`.
- Runtime: Metal on Apple M2 Pro with `gpu_layers=99`.

The M3g leg proves isolated production ambient admission for edge-grounding promotion. The
M4 leg proves production publication through the public `POST /api/steep` surface. Neither
leg proves default-on contention or fairness under a competing production workload.

### Positive M3g grounding leg

The source memory was
`directory-wenlan-m4-pr1-source-597760ac-v3::/private/tmp/wenlan-m4-pr1-source-597760ac-v3/grounded-community-source.md`.
Its final stored state was version `3`, space `m4-live`, source agent `folder`, and content
hash `1113696816ca1d41f732d9be97051035bfd3b484fe4a3e92fca3f115d9647db2`.
The document queue was `done` with `attempt_count=0`, and all five source evidence
sentences remained present.

Before grounding, the public fixture had 10 `m4-live` entities, five relations with rowids
`1..5` while the grounding cursor was `0`, five active ungrounded assertion edges, and 10
distinct endpoints. Production ambient admission then recorded five genuine
`EdgeGroundingPromote selected=true` completions:

| Completion | `llm_calls` | `panicked` |
|---|---:|---|
| `2026-07-27T07:27:38.569175Z` | 1 | false |
| `2026-07-27T07:30:10.428486Z` | 1 | false |
| `2026-07-27T07:32:42.627356Z` | 1 | false |
| `2026-07-27T07:35:14.567510Z` | 1 | false |
| `2026-07-27T07:37:46.475322Z` | 1 | false |

The final projection had five active assertion edges, all five grounded, across 10 distinct
endpoints. All shared root `2b9b2e2f-e374-4df6-8202-258889ecfe43`, whose state was active
`document_ingest`. Every grounding payload recorded model id/version `qwen3-4b`, prompt
`m3g-entailment-v3`, and entailment score `1.0`. The durable cursor was `5`, with
`stuck_rowid=null` and `failures=0`. The leg ended with a formal shutdown.

### Positive M4 publication and restart leg

On the same database, the pre-state was:

```text
graph_generation=5
grouping_generation=5
published_generation=NULL
dirty=1
communities=0
members=0
leases=0
```

`POST /api/steep` returned a `community_detection` phase with
`items_processed=15`, `duration_ms=2`, and `error=null`; the full steep returned 15 phases
and zero errors. The persisted post-state was graph/grouping/published generation
`5/5/5`, `dirty=0`, five active communities, and 10 distinct members arranged as five
pairs. Every row was generation `5`, algorithm `leiden-m4-v1`, projection
`grounded-relates-v1`; orphan members and leases were both zero. The sorted logical-row
SHA-256 was
`4c5367b3c7e08e6a02cac520c6464d002382c19f1a69c9570497f895fba5575c`.

The canonical pipeline used for each live read was:

```sh
sqlite3 /private/tmp/wenlan-m4-pr1-data-597760ac-v3/memorydb/origin_memory.db \
  "SELECT row FROM (
     SELECT 'C|'||space||'|'||community_id||'|'||coalesce(display_name,'')||'|'||
            algo_version||'|'||projection_version||'|'||
            CASE WHEN retired_at IS NULL THEN 'active' ELSE 'retired' END AS row
       FROM communities
     UNION ALL
     SELECT 'M|'||space||'|'||node_id||'|'||node_kind||'|'||community_id||'|'||
            published_generation||'|'||attachment AS row
       FROM community_members
     UNION ALL
     SELECT 'S|'||space||'|'||graph_generation||'|'||grouping_generation||'|'||
            coalesce(published_generation,'NULL')||'|'||dirty AS row
       FROM space_graph_state
   ) ORDER BY row;" |
  shasum -a 256
```

After formal shutdown, an actual second M4 process restarted against the same environment
and database. Before any mutating request, health was again `0.15.0+g597760ac`. During the
two live reads, the operator observed the pipeline above return the same hash before and
after restart; the receipt does not persist both raw command outputs, so replaying the
surviving database is not independent proof of the pre-restart read. State remained
`5/5/5`, `dirty=0`, with five communities, 10 members, and zero leases. The second process
then completed a final formal shutdown, and port `7879` was free.

### Post-main integration live rerun

The full signed live contract was rerun on 2026-07-27 at clean exact HEAD
`f367ae7485ba97c7fd18e05b50af728fdf212fdd`, after the latest `main` was integrated into
the local branch and the document-enrichment retry repair was committed locally. Neither
is yet published by PR #395. The exact `target/debug/wenlan-server` had SHA-256
`363959550a215612da999d6035c4111045de8eb1c2611f3312fa51884e668626`, size
`155468528`, schema version `95`, and health version `0.15.0+gf367ae74`. The passing
isolated roots were:

- data: `/private/tmp/wenlan-m4-pr1-data-f367ae74-v2`;
- source: `/private/tmp/wenlan-m4-pr1-source-f367ae74-v2`;
- evidence: `/private/tmp/wenlan-m4-pr1-live-f367ae74-v2`.

The signed document timeline was reproduced before any graph fixture write:

1. Fresh public source admission produced one version `1` `folder` memory in `unfiled`
   with hash
   `71f401ddd10f69f2d6d0ba2390c22395bd0c367a2d84234d3bcbdeb620d641a3`;
   the queue drained `done` at `attempt_count=0`.
2. The public official document-to-space assignment alone produced version `2` in
   `m4-live`, preserving the same hash and `folder` inventory.
3. A semantic source replacement, submitted through the public source-sync endpoint,
   produced version `3` while preserving `m4-live` and `folder`; its final hash was
   `1113696816ca1d41f732d9be97051035bfd3b484fe4a3e92fca3f115d9647db2`,
   and the queue again drained `done` at `attempt_count=0`.

There was no provider failure or retry in the passing run, so this live leg did not
dynamically exercise a same-hash retry. Repeated automatic same-hash source syncs while
each queue item was active did not create another semantic generation. The signed
pre-fixture snapshot had zero entities, zero relations, and zero `relates` edges.

The public fixture then created exactly 10 `m4-live` entities and five relations, each
linked to the exact full folder `memories.source_id`
`directory-wenlan-m4-pr1-source-f367ae74-v2::/private/tmp/wenlan-m4-pr1-source-f367ae74-v2/grounded-community-source.md`.
Before grounding, the five assertion edges were all active and ungrounded across 10
distinct endpoints, with cursor `0`. Production ambient admission recorded five genuine
`EdgeGroundingPromote selected=true` completions:

| Completion | `llm_calls` | `panicked` |
|---|---:|---|
| `2026-07-27T11:47:41.752513Z` | 1 | false |
| `2026-07-27T11:50:13.695776Z` | 1 | false |
| `2026-07-27T11:52:45.718929Z` | 1 | false |
| `2026-07-27T11:55:17.699825Z` | 1 | false |
| `2026-07-27T11:57:49.647778Z` | 1 | false |

All five assertion edges ended grounded across 10 endpoints and shared active
`document_ingest` root `8aed80a5-eb24-493d-be1a-1a7e0f907943`. Every payload recorded
model id/version `qwen3-4b`, prompt `m3g-entailment-v3`, path `entailment-only`, and
entailment score `1.0`. The durable cursor was `5`, with `stuck_rowid=null` and
`failures=0`. The source remained version `3`, `m4-live`, `folder`, at the final hash,
with all five evidence sentences present. The M3g process then shut down formally and
port `7879` was free.

On the same database, the M4 pre-state was `5/5/NULL`, `dirty=1`, with zero communities,
members, and grouping leases. Public `POST /api/steep` returned 15 phases with zero
errors; `community_detection` processed `15` items in `3 ms`. The post-state was
`5/5/5`, `dirty=0`, with five active communities, 10 distinct members arranged as five
pairs, and zero orphan members or grouping leases. Every persisted row used generation
`5`, algorithm `leiden-m4-v1`, and projection `grounded-relates-v1`.

Unlike the earlier live receipt, this rerun persisted both sorted logical-row outputs:
`55-logical-rows-pre-restart.txt` and `62-logical-rows-post-restart.txt`. An actual second
M4 process returned health `0.15.0+gf367ae74` before any mutating request; the two files
were byte-identical and each had SHA-256
`c3b29ff5544fe2cb23b6d4257681c5ae3dece634ff4be8e251bdbb6f3716529b`.
Restarted state remained `5/5/5`, `dirty=0`, with five communities, 10 members, and zero
leases. The second process completed a formal shutdown and port `7879` was free.

One earlier isolated verifier-negative run is preserved under the corresponding `-v1`
roots. Its public fixture mistakenly supplied `memories.id` instead of the required full
folder `memories.source_id`; ambient M3g therefore made a zero-LLM external-origin skip
and advanced cursor `0→5` without grounding. That database was formally shut down,
proved port-free, and abandoned without direct database correction or reuse. It is not
part of the positive result above.

### Open publication step

The local branch is 15 commits ahead of `origin/main`. Its first-parent commits after the
remote PR candidate `a5d985bc` are exactly `b7c22ce6`, `597760ac`, `7727c72b`,
`d0d070b7` (merge latest `main`), and `f367ae74`. This receipt remains uncommitted. None
of those local updates or the live evidence above are yet in remote PR #395 or covered
by remote CI. Pushing them waits for explicit user approval.
