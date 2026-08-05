# G6 Stage 1.3 — migrate `page_sources` + `page_evidence` readers onto `edges`

Parent program: `docs/plans/2026-08-04-g6-retirement-program.md` (Stage 1, store 3).
Scout: `docs/superpowers/g6-stage1-page-sources-evidence-scout.md` (gitignored working
doc; findings restated here). Prerequisites: #486/#487 (cites edges carry payload keys
`source_kind` (always), `linked_at`, `link_reason`, `title` via `cites_semantic_patch`,
refresh-on-reassert; m115 backfilled them from live legacy rows) and the Stage 1.2 PR
(NULL-guard discipline, divergence-test pattern). The Stage-0-class writer gap the
scout found (`try_update_page_content`) was fixed in PR #482 — writers are complete;
`dual_write_edge` is `#[cfg(test)]`, so no production mint can bypass the payload path
(compile-time guarantee, no writer changes in this PR).

## Structural differences from Stage 1.2 (why this store is easier)

- The identifying value (`page_sources.memory_source_id` == `page_evidence.locator`)
  is a REAL edge column (`dst_id`) at every dual-write site, not a hash-input-only
  discriminator. No semantic_type analog needed.
- No wire type exposes a legacy row id (`PageSource` = page_id, memory_source_id,
  linked_at, link_reason; `PageEvidence` adds source_kind, locator, title). NO trap-2
  id swap in this PR.
- The 4-way `source_kind` (`memory`/`external_url`/`external_file`/`authored`)
  collapses to 2-way `dst_kind` on the edge row but survives in payload
  `$.source_kind` — and `cites_semantic_patch` writes it UNCONDITIONALLY, so only
  pre-#486 edges whose legacy row vanished before m115 can lack it.

## Equivalence rationale (S1, 2026-08-05 review finding)

Parity drift 0 is often read as "edges ≡ `page_sources` ∪ `page_evidence`" (a
two-store union). That's the wrong equivalence, and reading the migrated
readers against it makes the D7 survivor look like a bug. The parity-honest
equivalence is a THREE-store union:

```
edges (cites) ≡ page_sources
              ∪ page_evidence (non-NULL locator)
              ∪ pages.citations
```

`pages.citations` independently derives `cites` edges (`backfill_edges_from_page_citations`,
db.rs:20358), same as `page_sources` (`backfill_edges_from_page_sources`, db.rs:20161)
and `page_evidence` (`backfill_edges_from_page_evidence`, db.rs:20214) — three
derivation sites feeding the same content-addressed edge_id space, all folded
into the SAME parity sweep. A `cites` edge for `(page, locator)` can therefore
stay active with NEITHER `page_sources` NOR `page_evidence` backing it, as
long as `pages.citations` still does (the D7 refcount: `cites_backed_by_page_citations`,
checked before an orphan-cleanup invalidate). The migrated readers know this
and adopt it: post-1.3, "page sources"/"page evidence" MEANS "active `cites`
edges", which is a WIDER population than "rows in the legacy table" whenever
a D7 survivor exists. This is deliberate and time-boxed — Stage 2 retires
`pages.citations`' edge-backing role, at which point the union narrows back
to two stores. See `get_page_sources_returns_d7_survivor_backed_only_by_citations`
(db/main_tests.rs) for the pinned reader-side behavior.

## The ordering trap (same class as 1.2's trap 1)

Legacy readers order by `linked_at ASC` and the numbered-source lists downstream
(citation backfill, post-ingest) depend on that stable order. `edges.created_at` on
m81-era edges is migration-day, NOT the legacy `linked_at`. Every migrated reader that
ordered by `linked_at` uses:

```sql
COALESCE(json_extract(e.payload,'$.linked_at'), e.created_at)
```

and projects that same expression wherever the wire field `linked_at: i64`
(non-optional) is populated — never a bare payload extract, which can be NULL.

## NULL-payload discipline (1.2 review finding S2/R1, applied at design time)

Any projection of a payload key into a non-optional field must be COALESCE-total:

- `linked_at` → the COALESCE above (always non-NULL: created_at is NOT NULL).
- `source_kind` → `COALESCE(json_extract(e.payload,'$.source_kind'), CASE e.dst_kind
  WHEN 'memory' THEN 'memory' ELSE 'external' END)` — dst_kind is a NOT NULL real
  column, so the fallback is total and lossless for memory rows, coarse-but-honest
  for degraded external rows.
- `link_reason`, `title` → already Option on the wire; plain extract is fine.

## Reader migration table

All migrated queries filter `edge_type='cites' AND valid_until IS NULL`, with
`src_kind='page'` where the query is page-anchored. **Ruling (2026-08-05, closure
review, corrects the original premise below):** `page_sources` was never
memory-only at the data level -- `insert_page`'s dual-write does `INSERT OR IGNORE
INTO page_sources` for every id in `source_memory_ids` with no kind check, memory
or external alike, and mints the typed `cites` edge twin regardless (`dst_kind`
resolved per source). Filtering the enumeration readers (`get_page_sources`,
`get_page_sources_scoped`, `count_page_sources_up_to`) to `dst_kind='memory'`
narrowed their output below what legacy `page_sources` actually returned -- a real
regression a production caller (`refresh_page`, distill.rs) depended on. Those
three readers carry NO `dst_kind` filter, same as `page_evidence`'s span-all-kinds
readers; the S1 union semantics ("page sources = active cites edges") has no kind
carve-out either. `dst_kind='memory'` stays correct only where a reader is keyed
BY a known memory id (`get_pages_for_memory`, `mark_pages_depending_on_memory_
sources_except`, `load_bounded_page_source_ids`, retro-scan, and the deletion-path
edge retirement added alongside this ruling) -- there `dst_id` IS the memory id
being matched, not an enumeration-scope filter.

| # | Reader | Location | Notes |
|---|---|---|---|
| 1 | `get_page_sources` | db.rs ~48099 | product route via `handle_get_page_sources`; SELECT src_id, dst_id, COALESCE-linked_at, `$.link_reason`; ORDER BY the COALESCE ASC; no dst_kind filter (ruling above) |
| 2 | `get_page_sources_scoped` | db/scoped_pages.rs:739 | same + scope handling exactly as the existing scoped variant; no dst_kind filter (ruling above) |
| 3 | `count_page_sources_up_to` | db.rs ~48134 | clean COUNT-with-LIMIT swap; no dst_kind filter (ruling above) |
| 4 | `get_page_evidence` | db.rs ~48160 | all kinds; COALESCE source_kind + linked_at per discipline; `$.title`, `$.link_reason` optional |
| 5 | `get_pages_for_memory` | db.rs ~47650 | `SELECT src_id ... WHERE dst_kind='memory' AND dst_id=?` then JOIN pages — clean |
| 6 | `mark_pages_depending_on_memory_sources_except` | db.rs ~27613 | EXISTS swap — clean |
| 7 | `load_bounded_page_source_ids` | db/maintenance_duplicate_reads.rs:177 | clean |
| 8 | retro-scan bounded source count | db/maintenance_retro_scan.rs:79 | clean |
| 9 | db_checks page-has-sources / page-has-evidence | lint/pages/db_checks.rs:110-111 | EXISTS swaps — clean |
| 10 | memories-lint reverse lookup | lint/memories/query.rs:51-52 | EXISTS swap — clean |
| 11 | semantic-candidates evidence loader | lint/semantic_candidates.rs:876 | needs page_id, source_kind, locator → src_id, COALESCE source_kind, dst_id |

**Conditional (investigate first, bounded discretion — same protocol as 1.2's #12/#13):**

| # | Reader | Location | Condition |
|---|---|---|---|
| 12 | orphan-evidence check | lint/deep.rs:155 | Scout time-boxed it; read it fully. If it is a well-formedness check over the canonical store, migrate with the discipline above. If it cross-checks the legacy stores against each other or another store, it is a carryover — dated comment. Report which and why. |

**Deliberate carryovers (stay on legacy, dated comments):**

- The provenance-consistency lints in `lint/pages/provenance_checks/source.rs` (:37,
  :78, :193-236) — they validate the two legacy tables AGAINST EACH OTHER (cross-store
  consistency), which is definitionally inexpressible over a single canonical store.
  They retire with the stores at Stage 3.
- `page_memory_provenance_state` (db.rs ~46822) — writer-internal undo-ledger
  bookkeeping, not a reader migration target.
- The D7 refcount helpers (`cites_backed_outside_page_citations`,
  `cites_backed_by_page_citations`) — writer-side machinery, untouched.
- The claim-derivation UNION bridge (db/claim_derivation.rs:1832-1845) — already
  edges-aware by design; leave as is.

Parity derivation for this pair STAYS in `reconcile_edges_parity` (the carryover lints
are still legacy readers). Program-doc status note in the same PR.

## No migration needed

No new schema, no new migration: m115 already backfilled the cites payload keys. (If
implementation uncovers active cites edges whose payload lacks `linked_at` on rows the
COALESCE ordering would visibly mis-order — i.e. legacy row still live with a
different linked_at — bounce it back; do not invent an m117 unilaterally.)

## Tests (RED-control discipline as in 1.2)

1. **Equivalence per product reader** — seeded fixture via the real writers
   (`link_page_source`, `link_page_evidence`); field-by-field including order for
   `get_page_sources(_scoped)` and `get_page_evidence`; assert `linked_at` on the wire
   equals the legacy value (payload-backed), not edge insert time (seed one edge whose
   `created_at` is forced to differ from payload `$.linked_at`).
2. **Kind fallback** — an active cites edge with NO payload (simulated pre-#486
   remnant): `get_page_evidence` returns it with the dst_kind-derived kind and
   created_at-derived linked_at rather than erroring or dropping it.
3. **Divergence test** (the 1.2 pattern, asymmetric seeding): one cites fact only in
   `edges`, two only in `page_sources`/`page_evidence`; assert readers #1, #4, #5, #6
   and the two EXISTS lints see the edge-only fact and stay blind to the legacy-only
   orphans. RED-control via a prove-then-revert mutation of one reader.
4. **Parity stays 0** — existing parity fixtures unchanged.

## Gates

Same as 1.2: fmt, clippy (both variants), focused modules (page/source/evidence
filters, the lint modules touched, the divergence test), full suites at integration
time (queued by the session lead, not the executor).

## Out of scope

`pages.citations` readers (Stage 1.4 — `row_to_page` reshape, its own PR); entities
(Stage 1.5); writer changes (none needed); dropping any parity derivation; the
provenance-consistency lints' eventual retirement design.
