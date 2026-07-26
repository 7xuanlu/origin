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
