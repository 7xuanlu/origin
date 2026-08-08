# G6 retirement program — retire the six legacy KG stores

Status: authored 2026-08-04, after G5 completed (all three flips live, both parity
watermarks drift 0, T1 ceremony committed at generation 1). This is the execution
plan for the close plan's G6: *"delete the five old link-bookkeeping tables and the
old entity table so there is exactly one source of truth."*

Scope basis: a full non-test reference map of `crates/wenlan-core/src` +
`crates/wenlan-server/src` (2026-08-04). Counts below are distinct functions, not
lines.

## The load-bearing finding

The G5 reader-cutover flips gate exactly **two** consumers: `detect_communities`
(`reader_uses_edges`, consumer `communities`) and the scoped-entity reads in
`db/scoped_entities.rs` (`reader_uses_entity_pages`, consumer `scoped_entities`).
**Every other reader in the product reads the legacy tables directly and ungated** —
`handle_get_entity_detail`, `handle_list_recent_relations`, `handle_get_page_sources`,
`handle_get_page`, `handle_search_pages`, the lint/kg-quality/repair tooling, and the
retrieval helpers. Dropping any table today breaks them regardless of cutover state.
G6 is therefore a reader-migration program with a drop at the end, not a drop with
prep.

## Stage 0 — close the dual-write gaps (do this FIRST, keeps parity honest)

Writers that mutate legacy stores without the canonical `edges` dual-write.
Each is the same training-serving-skew class as #473/#479, and each re-drifts
parity the moment it runs. All sites below were verified at the code level on
2026-08-04 (the original scope-sweep list contained one false positive).

**Part 1 (this PR): the three primary-path gaps.**

| Function | Gap | Fix |
|---|---|---|
| `fold_relation_type` (db.rs ~435) | UPDATE `relation_type` + collision DELETE, no edge retire/mint. `relation_type` is a structural field of the derived edge id — one vocabulary heal re-drifts parity (extra + missing per folded row). | Per folded row: mint the canonical-type edge (endpoint-space classification mirrors `create_relation_with_span`), then retire the old-type edge superseded-by the new one (mint first — `superseded_by` is an FK into `edges`). Community-graph generations bumped for any grounded assertions touched. |
| `replace_page_sources` (db.rs ~46282) | Prunes `page_sources` + memory-kind `page_evidence` rows and NULLs `pages.citations`, no edge invalidation. **Likely root cause of the 2026-07-23 live damage** (3 stale active edges the 32-row repair retired). | Snapshot the pruned sids before the DELETEs, retire each one's cites edge in the same transaction. Kept sids re-assert through `insert_resolved_page_evidence`'s dual-write. |
| `delete_page` (db.rs ~45109) | FK CASCADE drops the page's rows in all three link stores and NULLs inbound `page_links` targets (NULL targets derive no edge), no edge retirement. | Bulk-retire every active edge touching the page (`src_kind='page' AND src_id` OR `dst_kind='page' AND dst_id`) in the delete transaction — page edges are never grounded `relates` assertions, so the direct UPDATE is equivalent to per-edge invalidation (mirrors `replace_page_links`). |

`link_page_source` (flagged by the sweep) is a **false positive**: it calls
`insert_resolved_page_evidence`, which dual-writes the same content-addressed
edge id the page_sources row derives to.

**Part 2 (second PR): secondary writers, verified and fixed 2026-08-05.**

| Function | Gap | Status |
|---|---|---|
| `rebind_source_id_inner` (db.rs ~27860, incl. `rebind_source_page_in_transaction`) | Renames `page_sources.memory_source_id`, `page_evidence.locator`, and optionally the source page id — identity fields of the content-addressed edge id. The M2 PR-1 block already retired+minted memory-locator cites edges for pages listed in `page_sources`/`page_evidence`, but the page-id rename half had no edge rewrite, and citations-only cites edges were missed. | FIXED. New `rebind_edges_identity` helper re-addresses cites edges (disc = dst locator, derivable from the row) with FK-safe `superseded_by` detach/re-attach, collision-aware (an already-minted successor absorbs the old edge as retired history); links edges retire + re-assert from `page_links`. Payload `source_memory_id` provenance re-stamped. |
| `replace_source_page_inner` (db.rs ~43093) | DELETEs ALL `page_sources` + `page_evidence` rows (external kinds included) then reinserts the new set; removed rows' edges never retired. | FIXED. Removed-locator snapshot before the DELETEs; each dropped locator's cites edge retired in-transaction (external kinds included). |
| `accept_page_merge` (db.rs ~44859) | Repoints inbound `page_links.target_page_id` loser→winner with a raw UPDATE; the loser-dst links edges stay active (extra), winner-dst edges show missing. Also copies the absorbed page's external evidence rows without minting their edges. | FIXED. Per repointed row: mint the winner-dst links edge (replace_page_links derivation), retire the loser-dst edge superseded-by it. Copied external evidence rows mint their cites edges. |
| `resolve_orphan_page_links` (db.rs ~45684) | Sets `target_page_id` on a previously-orphan row (which derives no edge) without minting the now-implied links edge. | FIXED. Mints with `replace_page_links`'s derivation in the same per-row transaction. |
| `try_update_page_content` (db.rs ~43975) | Rewrites `pages.citations` wholesale without reconciling edges. Narrow: drifts only when a locator backed ONLY by citations (not by `page_sources`/`page_evidence`) drops out of the new value. | FIXED. Calls the shared `dual_write_page_citations` reconciler inside the CAS transaction (`set_page_citations_with_changelog_at_version` rewired onto the same helper). |
| `apply_deterministic_repair_cas`, `RepairWriter::BindPageLink` arm (db/repair_deterministic.rs) | The repair tool's own orphan-bind — a second writer of `page_links.target_page_id` beside `resolve_orphan_page_links` — set the target with a raw UPDATE and no edge mint. Found by the Stage 1.1 scout (`docs/superpowers/g6-stage1-page-links-scout.md`). | FIXED. The arm mints the links edge in-transaction (same derivation as `resolve_orphan_page_links`); the mint's row changes are measured and allowed by the repair effect guard. |
| `try_update_page_content` — `page_sources`/`page_evidence` half (db.rs ~44266) | Beyond the citations rewrite (row above), the same function does its OWN prune-then-reinsert of `page_sources` + memory-kind `page_evidence` against the `source_memory_ids` argument — a second instance of the `replace_page_sources` bug, with no removed-sid snapshot and no edge retirement. High traffic: manual edits, refinery rewrites, and page growth all route through it. Found by the Stage 1.3 scout (`docs/superpowers/g6-stage1-page-sources-evidence-scout.md`). | FIXED. Snapshot of the pruned locators (page_sources ∪ memory-kind page_evidence) before the DELETEs; each retired in-transaction UNLESS the new `pages.citations` value still backs it (D7 refcount via `cites_backed_by_page_citations` — this path, unlike `replace_page_sources`, sets citations to an explicit new value). Kept sids re-assert through `insert_resolved_page_evidence`. |

Residual (not Stage 0 scope, tracked): the generic repair **rollback** artifact
restores legacy-store rows byte-wise (`rollback-v1.json` row restore) without
edge reconciliation — rolling back a bind would re-orphan the row while its
minted edge stays active. Operator-driven and rare; the parity sweep catches it.
Fold into Stage 2 when repair writers go canonical-only.

Known limitation (M5 follow-up, outside Stage 0 scope): M5 claim `supports`
edges are fenced out of the parity universe, and a source-id rebind retracts
them (`retract_support_for_rebound_source`, M5 row 13) rather than re-address
them — pages citing the renamed document re-derive support. No parity impact.

Exit: all fixes merged with regression tests (parity clean after driving each
path); ambient watermarks stay drift 0 across a soak.

## Stage 1 — migrate readers onto canonical stores, cheapest first

**Decision (user, 2026-08-05): one source of truth.** The semantic-payload
schema change is authorized — `relates` edges carry `relation_type` (+
`confidence`/`explanation`/`source_agent`), `links` edges carry the display
`label`, with a backfill migration from the legacy rows. No legacy store
survives Stage 3 as a permanent side-table. The same principle extends to
`cites` edges where Stage 1.3 found unrecoverable columns (`link_reason`,
`linked_at`, the 4-way `source_kind`): carry them in edge payload/columns
rather than keeping `page_sources`/`page_evidence` alive. The payload PR
lands FIRST (before any reader migration), since every blocked reader in the
scout reports migrates only once the semantic fields exist on the edge.

Order by measured entanglement:

1. **`page_links`** (~9 fns) — one dual-write-aware writer choke point
   (`replace_page_links`), small reader fan-out (`get_page_outbound_links`,
   `get_page_inbound_links`, orphan-label lint), no shared-struct entanglement.
   **Scout correction (2026-08-05, `docs/superpowers/g6-stage1-page-links-scout.md`):
   this store cannot fully migrate onto `edges` as wired.** The `label` display
   text is not stored on edge rows (label_key is only a hash input), and orphan
   rows (`target_page_id IS NULL`) derive no edge at all — so the two product-route
   readers behind `GET /api/pages/{id}/links` and all orphan-feed readers stay on
   `page_links`. Only `load_link_counts` migrates cleanly today. Decision needed
   before this store reaches Stage 3: carry the label in a `links` edge payload
   (schema change), or keep `page_links` alive as a label+orphan side-table and
   shrink the Stage 3 drop list accordingly.
   **Status (2026-08-04):** PR #486 closed the label gap. The product-route
   readers and `load_link_counts` now read resolved links from `edges`, while
   orphan readers stay on `page_links`. The decision is recorded: at Stage 3,
   `page_links` narrows to an orphan-only store rather than being dropped
   entirely.
2. **`relations`** (~20 fns) — clear writer choke points, but readers include
   product routes (`get_entity_detail`, `list_recent_relations`), k-hop expansion,
   and lint/repair tooling. Entangled with `entities` via shared CRUD
   (`merge_entities`, `commit_entity_enrichment_at_version`,
   `delete_by_source_id_in_transaction`) — those functions change once, in this
   stage, for both stores' read sides.
   **Scout correction (2026-08-05, `docs/superpowers/g6-stage1-relations-scout.md`):
   blocked harder than page_links.** `relation_type` is a hash-input-only
   discriminator — never stored on the edge row or payload — so every reader
   that returns or filters on it (both product routes, most lint/repair tooling;
   9 of 13 readers) cannot migrate as wired. Only topology-only readers (k-hop
   ×2, aggregate count, scope subquery) migrate cleanly. All relations writers
   already dual-write (no Stage 0 gap). Same decision as page_links, but
   sharper: the semantic-payload schema change (relation_type + confidence /
   explanation / source_agent on `relates` edges, label on `links` edges, plus
   backfill migration) is realistically the only path to Stage 3 for this store.
3. **`page_sources` + `page_evidence`** (~30 fns, co-written pair) —
   `insert_resolved_page_evidence` is the evidence choke point; `page_sources` has
   no single choke point and needs one first.
   **Status (2026-08-05):** Stage 1.3 migrated all readers in the spec table
   (11 mandatory plus one conditional, `retrieval_substrate` in `lint/deep.rs`,
   a per-channel well-formedness check rather than a cross-store consistency
   check) onto `edges` (`cites` edge type). No id-swap trap here (unlike
   Stage 1.2): `memory_source_id`/`locator` are real `dst_id` columns, not
   hash-input discriminators, so both stores migrate cleanly with no schema
   change.

   Review round (S1-S7) surfaced four readers beyond the spec table and one
   equivalence-rationale correction:
   - **Equivalence (S1):** parity drift 0 proves a THREE-store union, not
     two — `edges ≡ page_sources ∪ page_evidence(non-NULL locator) ∪
     pages.citations`. A `cites` edge can stay active backed only by
     `pages.citations` after its `page_sources`/`page_evidence` row is
     pruned (the D7 refcount survivor). The migrated readers adopt this
     union knowingly, pinned by
     `get_page_sources_returns_d7_survivor_backed_only_by_citations`.
   - **`delete_non_head_memory_chunks` (db.rs)** — the same page-invalidation
     shape as reader #6, migrated to `edges` (S3a).
   - **The evidence half of `lint/pages/db_checks.rs`'s
     `pages.source_page_integrity` check** — reverted to `page_evidence OR
     edges` (S2): a NULL-locator `authored` row is real provenance with no
     edge twin, so an edges-only read undercounted it. The page_sources half
     of the same check stays edges-only.
   - **Three distill-eligibility predicates** (`query_distillation_staging_pool`,
     `query_distillation_seed_slice`, `query_distillation_ann_neighbors`, all
     db.rs) — CARRYOVER, dated in place (S3b): widening them to the
     edges/cites union changes the eligibility pool and needs its own test
     attention; folds into Stage 2.

   Four locations stay on legacy by design and are dated in place: the two
   provenance cross-store lints in `lint/pages/provenance_checks/source.rs`
   (compare the legacy tables against each other, not against edges — an
   edges-only read would make the check trivially pass), the D7 refcount
   helpers `cites_backed_by_page_citations` / `cites_backed_outside_page_citations`
   (writer-side machinery gating whether `pages.citations` still backs a
   locator before retiring its edge), and `page_memory_provenance_state`
   (writer-internal before/after snapshot). `page_sources`/`page_evidence`
   themselves stay live as the dual-write target; only the reader side moved.
4. **`pages.citations`** (4 writer fns, 10+ readers) — hardest: a column in every
   `Page` row mapper (db.rs + db/scoped_pages.rs). Retiring it is a struct/mapping
   reshape, its own PR.
   **Stage 1.4 (pages.citations) — reclassified 2026-08-05:** `pages.citations`
   is a derived render cache, not a truth store. Per-occurrence annotation
   state (occurrence/marker/score/status/scope) is render-layer by design;
   `citations IS NULL` drives the citation-backfill sweep with no edges
   analog; the column passes the delete-and-rebuild test. Stage 2 retires its
   edge-backing role (D7 refcount simplification); the column survives Stage 3
   the way the FTS index does. No reader cutover.
5. **`entities` + `entity_aliases` + `observations`** (~25+ fns) — largest surface;
   readers move to the `kind='entity'` shadow pages. Depends on stage 1.2's shared-
   CRUD rework.
   **Stage 1.5a — status (2026-08-05):** migrated the 18 CLEAN readers in the
   spec table (`docs/plans/2026-08-05-g6-stage15a-entity-clean-readers-spec.md`)
   onto `entity_page_map` JOIN `pages` (`kind='entity'`, `status='active'`); the
   `create_entity` shadow-page landmine fixed alongside. 10 readers stay CARRYOVER
   (space-sensitive under the NULL->`UNFILED_SPACE_ID` fold, or writer-coupled —
   dated in place, see the spec's carryover list). Hard-cut, no `reader_uses_entity_pages`
   gate on the migrated readers: this is the program's Stage 1 contract, the same
   unconditional cutover 1.1-1.3 used for `edges` while that gate + cutover-lever
   still existed; the gate and `entity_reader_cutover` retire together with the
   writers in Stage 2. Empirically backed on the live DB (swept 2026-08-05 07:30,
   post-migrations 113/114): `entity_page_parity_watermark` drift 0, 907/907
   expected-vs-actual shadow pages, 0 `entities` rows without a live shadow page.
   **Stage 1.5b — status (2026-08-05):** Part 1 (migration 117) extended the
   shadow-page scalar mirror to `source_agent`/`entity_created_at`/
   `entity_updated_at`. Part 2 (migration 118) folded `entities.space` NULL to
   the `UNFILED_SPACE_ID` sentinel (matching the `memories`/`pages` folds),
   making every space-sensitive reader safe to trust off the mirror. Part 3
   (spec `docs/plans/2026-08-05-g6-stage15b-entity-reader-completion-spec.md`)
   migrated the 9 reader targets 1.5a's CLEAN-reader spec had excluded as
   space-sensitive or structurally coupled: `list_entities`, `get_entity_detail`
   (entity-half), `search_entities_by_name` (same unconditional-cutover shape
   as 1.5a); `search_entities_by_vector`/`_scoped` (the row set stays on
   `entities` -- the DiskANN index lives there -- but every display field is
   unconditionally hydrated from the shadow page); `load_summary_buckets`'s
   legacy branch + `summary_eligible_predicate`; the embedding-refresh
   staleness sweep; and `list_recent_relations`/`_scoped` (structural rework,
   not a hydration overlay -- the join itself was legacy-shaped even when
   gated). This collapses the last `reader_uses_entity_pages` gated hybrid in
   `scoped_entities.rs`; the gate and `SCOPED_ENTITIES_CONSUMER` are left
   dead-but-present (retire with the writers in Stage 2, per the spec). Of
   1.5a's 10 CARRYOVER sites, 3 were already self-documented as writer-coupled
   (defer to Stage 2) or intentionally legacy (dual-metric emission); the
   remaining 7 are lint/integrity audits of the `entities` store's own data
   quality and are reclassified intentionally-legacy -- auditing the shadow
   mirror instead would validate the mirror, not the store the check exists to
   audit. Verification: `cargo test -p wenlan-core --lib` -- 4083 passed, 0
   failed, 33 ignored (GPU-gated), 0 filtered; fmt + clippy (`--lib --bins` and
   `--all-targets`) clean; M5 reader-inventory and R4 test-support census
   regenerated.

Each store's migration is one PR: readers redirected to `edges`/shadow pages,
behavior-equivalence tests, and the store's parity derivation dropped from
`reconcile_edges_parity` in the SAME PR that removes its last legacy reader.

Exit per store: zero non-test readers of the legacy store outside migrations and
the parity sweep; ambient drift stays 0.

## Stage 2 — writers go canonical-only, sweeps retire

Only after Stage 1 empties the reader side: flip writers from dual-write to
canonical-only, retire `reconcile_edges_parity` / `reconcile_entity_page_parity`
and their watermarks/flags/plist entries, drop the `backfill_edges_from_*`
machinery. The cutover-lever tables (`edges_reader_cutover`,
`entity_reader_cutover`) retire with them.

## Stage 3 — the retirement migration (the point of no return)

One migration drops `relations`, `page_sources`, `page_evidence`, `page_links`,
the `pages.citations` column, `entities`, `entity_aliases`, `observations`, and
`entity_page_map`. Contract per the close plan:

- Pre-migration SQLite online backup (the standard `backup_before_migration`), plus
  an **operator-verified restore drill receipt** (drill already rehearsed 2026-08-04,
  receipts doc §7 — re-verify against the pre-retirement backup specifically).
- A **declared downgrade barrier**: daemon versions before this migration cannot
  open the database afterward; `wenlan doctor` must say so plainly.
- Export/import is NOT the safety net (debt register).
- User confirms the point of no return before the migration ships in a release.
- **Code cleanup, same PR as the drop migration:** three legacy FK-guard
  deletes against `entity_aliases.canonical_entity_id` (a NO ACTION FK into
  `entities(id)`) were restored in Stage 2 sub-step 3 item 5 after an
  initial (wrong) attempt to retire them early — `delete_entity`,
  `merge_entities`' loser-side delete, and `delete_space`'s bulk equivalent
  (db.rs). Each is commented "Retires in Stage 3 with the `entity_aliases`
  drop, not before" — grep `Retires in Stage 3 with the` to find and
  remove all three when this migration lands, since the table they guard
  against will no longer exist to violate.

## Rollback story per stage

- Stage 0/1: normal PR reverts; legacy stores still authoritative-adjacent.
- Stage 2: revert restores dual-write; canonical stores never stopped being written.
- Stage 3: restore-from-backup only. That is the whole reason it goes last.
