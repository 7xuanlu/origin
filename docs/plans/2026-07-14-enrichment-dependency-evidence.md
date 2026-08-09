# Enrichment dependency evidence

Date: 2026-07-16

Scope: complexity evidence for choosing the minimum dependency/invalidation and automatic-sweep design. Live database and logs were read-only. Mutation behavior and bounded automatic slices were verified with temp-DB fixtures; no copy of the user's memory database was created.

## Measured facts

- Live DB: 5,749 memories, 196 pages, 1,496 source ids with enrichment receipts, and 9,284 receipts across eight fixed stage names.
- Mutation activity over the last 30 days: 718 stores, 45 refines, 24 revision accepts, 45 revision dismisses, and 128 steep runs; activity existed on all 31 dates in the query window.
- Of 26 edited sources that have receipts, 18 have at least one receipt older than the current source modification. That is 69% of edited sources; 119 receipt rows are stale across those 18 sources.
- Twelve pending-revision rows cover 11 sources. Ten pending-only sources have all eight receipts updated after the pending row was created; none of those ten has a non-pending canonical sibling.
- Of 132 active pages, zero have a linked source modified after the page compile time. One archived page does.
- Those 132 active pages have 9.64 `page_sources` rows on average and 48 at most; none exceeds the automatic 64-row source cap. Four active pages rely on JSON source fallback, and nine have no joined `page_sources` rows.
- Existing focused fixture `update_memory_preserves_metadata_and_synchronizes_episode_and_fts` confirms that a content update currently preserves `version`, `last_modified`, `changelog`, `content_hash`, and `enrichment_status`; it passed on the current branch (1 passed, 2,303 filtered out).
- The 2026-07-14 daemon log contains six Idle runs, all 622-700 seconds and all over ten minutes. It records 243 `refresh_page` inferences averaging 21.466 seconds (5,216.2 seconds total), 169 `citation_annotate` inferences averaging 20.197 seconds (3,413.2 seconds total), 47 inference timeouts, six deadline overruns, and 672-707 work items still pending at Idle completion.

## Automatic sweep design and bounded profile

Automatic work uses cooperative, durable cursors rather than running the foreground full-scan APIs:

- Retro review inspects one raw Page and at most three source rows per turn. It is a one-time backfill; a pending review card pauses later retro/duplicate work.
- Near-duplicate detection keyset-scans at most 128 raw Page pairs per turn before eligibility or similarity work. It reads at most 257 source rows per distinct Page (256 plus an overflow sentinel); overflow suppresses source-overlap matching rather than treating truncated evidence as proof. Embedding comparison remains available.
- Cross-space discovery keyset-scans at most eight raw memory seeds and asks DiskANN for at most 64 neighbors per eligible seed. It never falls back to a full-table cosine sort, emits at most one review card, and pauses while such a card is pending.
- ReDistill refreshes at most one changed/stale Page per automatic turn. A bounded `cap + 1` source query rejects Pages above 64 sources before loading their contents or invoking the provider; the legacy JSON fallback receives the same check. Each included source is capped to 800 characters, topic text to 192 characters, and each of at most 100 existing-title hints to 128 characters. The fallback reads titles only rather than materializing up to 1,000 complete Pages.
- ReDistill is the only legacy steep phase on the automatic allowlist. Every other global phase remains reachable through explicit foreground steep but cannot run from BurstEnd, Idle, Daily, or Backstop scheduling. Unsupported matured bursts are drained as bookkeeping without charging a thermal turn.
- Generic imports now write their all-false `enrichment_origin` receipt in the same transaction as chunked/embedded memory rows. Missing legacy origin remains fail-protected, while new imports remain eligible for fixed-stage background classification.

One temporary ignored profiler reproduced the live aggregate shape with synthetic content: 5,749 memories, 132 active Pages, ten sources per Page except one Page with 48. The fixture setup took 2,555 ms and is not part of a scheduler turn. The profiler was removed after capture.

| Automatic stage | Observed wall time | Work actually inspected |
|---|---:|---|
| Retro review | 12 ms | 1 Page, 3 source rows |
| Near duplicate | 10 ms | 128 raw pairs, 129 distinct Pages, 1,290 source rows |
| Cross-space discovery | 20 ms | 1 raw/eligible seed, 3 ANN rows; one card emitted and queue backpressure engaged |

All three stages completed below their independent five-second test ceiling in that run. This proves the implemented work envelope and gives a debug-build wall-time sample; it does **not** prove the target-hardware thermal, battery, or full-daemon convergence envelope.

Cross-space automatic discovery is intentionally best-effort. Exhaustively enumerating all pairs among 5,749 memories would require 16,522,626 comparisons and is incompatible with an invisible low-duty background review-card funnel. The `fully_filtered_seeds / eligible_seeds_probed` metric is therefore required to detect when fixed `K=64` neighborhoods stop surfacing cross-space candidates; effectiveness data, not intuition, decides whether K or the index strategy changes.

## Ponytail verdict

Choose the fixed-stage version/CAS design. Do not build a generic dependency DAG or workflow engine.

The minimum retained mechanisms are:

1. Reuse the head memory's existing `version`; increment it atomically on every semantic mutation. Do not add a second source-generation field.
2. Add `input_version` to the existing `(source_id, step_name)` receipt. A receipt is current only when it matches the head version.
3. Select work with `(source_id, expected_version, input)` and commit every derived write plus its receipt only if the current version still equals `expected_version` (compare-and-swap). If it changed, discard the stale result and leave the current version eligible.
4. Exclude `pending_revision = 1` from machine enrichment and machine retrieval until the human accepts it. Acceptance becomes the semantic mutation that makes the source eligible.
5. Keep stage dependencies explicit in the existing fixed pipeline. No runtime graph, artifact registry, transitive invalidation engine, or generalized scheduler state.

## Why this is the minimum

- Mutation-only reset without CAS cannot stop an old 20-second inference from committing after the reset.
- The 119 stale receipts and the 18-of-26 edited-source rate show that input identity is not speculative.
- Eight stable stage names and fixed lanes show no evidence of graph-shape churn that would justify a general DAG.
- Zero active page/source timestamp mismatches do not justify page-wide generation machinery now. Page-specific CAS already exists and should remain stage-local.
- The pending data proves that the human-gate filter is needed now; it is not an optional abstraction.

## Deferred until data says otherwise

- Generic dependency graph, artifact registry, and workflow engine.
- Page-wide source generation or transitive invalidation.
- Eager delete/reset sweeps; lazy receipt-version mismatch is enough, with prove-before-delete cleanup later.
- Any estimate that treats every receipt as one 21-second inference. The independent Fable/Ponytail review made that unsupported conversion; it is intentionally excluded here.

## Measurements required after implementation

- Ambient inference count, duration, cooldown, and thermal/battery deferrals per lane.
- CAS rejection count (`stale_result_dropped`) per lane.
- Eligible backlog age and drain rate per lane.
- Number of pending/security/scope rows rejected before provider invocation.
- Receipt-version mismatch count after each mutation path.
- Active page/source freshness mismatch count.
- Automatic maintenance work counters and `fully_filtered_seeds / eligible_seeds_probed` by completed cursor pass.

Thirty days of zero CAS rejections would justify reconsidering whether some low-risk lanes can use a cheaper timestamp/version freshness check. New dynamic stage dependencies, or repeated cross-stage invalidation bugs that cannot be expressed in the fixed matrix, would be the evidence needed to revisit a graph.
