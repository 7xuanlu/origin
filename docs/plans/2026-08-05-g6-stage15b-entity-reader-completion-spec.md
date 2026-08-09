# G6 Stage 1.5b — scalar mirror extension + entity reader completion

Parent program: `docs/plans/2026-08-04-g6-retirement-program.md` (Stage 1, store 5,
second half). Prerequisite: Stage 1.5a merged. Design decisions here were ruled
in-session 2026-08-05 and recorded in the 1.5a spec
(`docs/plans/2026-08-05-g6-stage15a-entity-clean-readers-spec.md`, "1.5b decisions
recorded"); this spec turns them into an executable contract. The implementer
verifies every site at the code level before editing and bounces back any site
whose read contradicts this spec rather than improvising.

## Goal

After 1.5b, every entity READER in the tree is either (a) on the shadow-page
store, or (b) a writer-coupled read that flips with the writers in Stage 2, or
(c) a cross-store consistency check (parity/reconcile machinery) that retires in
Stage 2. No reader remains blocked on unmirrored fields, and the
`reader_uses_entity_pages` gated hybrids become unconditional canonical reads
(the gate itself and `entity_reader_cutover` retire in Stage 2 with the writers).

## Part 1 — m117: scalar mirror extension

New migration (next free slot; 117 assumed — verify against the migration table)
adds entity-page columns per the M3 precedent (entity_type / confidence /
entity_confirmed are already real `pages` columns):

- `source_agent TEXT`
- `entity_created_at INTEGER`
- `entity_updated_at INTEGER`
- `community_id TEXT` (ruled 2026-08-05 after a false-start INTEGER
  correction: the source column `entities.community_id` is INTEGER, but every
  consumer already reads it as a string — `CAST(e.community_id AS TEXT)` into
  `Option<String>` durable-community fields, and the m4 `communities` table
  keys are TEXT — so the mirror matches the read side; parity compares via
  the entities-side integer's string form, commented at the compare site)
- `embedding_updated_at INTEGER`

Naming note: `pages` already has its own `created_at`/`updated_at` — the entity
timestamps get the `entity_` prefix to avoid capture. Backfill in the same
migration from `entities` via `entity_page_map`. Writer thread-through: every
shared-CRUD writer that sets one of these on `entities` (`store_entity`,
`merge_entities`, `commit_entity_enrichment_at_version`,
`refresh_entity_embedding`, `confirm_entity`, community assignment) mirrors it
onto the shadow page in the same transaction, following the existing
`insert_entity_shadow_page`/`update_entity_shadow_page` pattern. Parity:
`reconcile_entity_page_parity` extends its field comparison to the new columns
so drift is observable until Stage 2 retires it.

## Part 2 — space-sentinel canonicalization

Target state: "space is never NULL". A data migration (its own number after
m117 — ruled 2026-08-05: the fold does not ride m117, so a failed scalar
backfill and a failed fold stay distinguishable in the wild) folds NULL
`entities.space` → `UNFILED_SPACE_ID` in `entities` itself, making
`pages.space` and `entities.space` semantically identical from then on.

Audit outcome (2026-08-05, pre-fold gate): 17 load-bearing consumer sites, not
just the one named `scope_matches` caller. Governing ruling: **the fold changes
the SPELLING of "unfiled," never the semantics** — every consumer that
distinguished NULL now distinguishes the sentinel, and the external wire
contract keeps `null`. All three tiers ride this PR, reworked ahead of the
fold in edit order:

- **Tier 1 (13 sites, mechanical):** swap to the established sentinel-aware
  tools (`push_read_scope_filter_folded`, db.rs:190; the
  `!= UNFILED_SPACE_ID` Rust-side filter pattern) — semantic_candidates
  (`load_entities`/`entity_scope_clause`/`load_relations`), scoped_entities
  `get_entity_detail_scoped`:211 / `search_entities_by_vector_scoped`:676,
  the lint deep/kg/aggregate carryover sites, repair
  `validate_selected_entities_*`.
- **Tier 2 (2 sites, classification counts):** preserve classification
  behavior with the sentinel as the unfiled marker.
  `audit_legacy_cross_space_links` (db.rs:20488): sentinel endpoints classify
  into `null_space`, not same/cross. Community adjacency builder
  (db.rs:35459): sentinel endpoint stays "indeterminate, not excluded" — the
  admitted edge set is IDENTICAL pre/post fold, pinned by a test on the same
  fixture.
- **Tier 3 (wire boundary):** sentinel → `null` at serialization —
  `entity_from_row` (scoped_entities.rs:973) and `handle_create_entity`
  (entity_graph_routes.rs:103, response `space` AND the `space_source` pick,
  normalized before `.is_some()`). Reuse the memories/pages post-fold
  boundary helper if one exists. CLI/MCP see no wire change. Server test:
  `CreateEntityResponse.space == null` + correct `space_source` for an
  unfiled write.
- **Writer-side fold:** `store_entity` (and any writer accepting an Option
  space) folds NULL→sentinel at write time, else NULLs recur post-migration.
  Invariant test: 0 NULL spaces post-fold AND after a `store_entity`
  round-trip with `space: None`.

## Part 3 — migration targets

Unblocked by Part 1 (scalars) and Part 2 (space semantics). Same target shape
as 1.5a: `pages` joined via `entity_page_map` (`kind='entity'`,
`status='active'`), result shapes and ordering byte-compatible.

1. `list_entities` (db.rs:33247) and `list_entities_scoped`
   (db/scoped_entities.rs:12)
2. `get_entity_detail` entity-half (db.rs:33301) and `get_entity_detail_scoped`
   (db/scoped_entities.rs:84) — the observations/relations halves stay on their
   own stores
3. `search_entities_by_vector` (db.rs:27295) and `_scoped`
   (db/scoped_entities.rs:640) — unblocks the `resolve_entity_by_name` vector
   fallthrough left legacy in 1.5a
4. `search_entities_by_name` (db.rs:32179)
5. `load_summary_buckets` legacy branch (db.rs:27555) and the
   summary-eligibility predicate's legacy branch (locate; name drifted —
   verify against the current tree)
6. embedding-refresh sweep (db/kg_quality_embedding_refresh.rs:22) — needs
   `embedding_updated_at` from Part 1; its `observations` join stays
7. `list_recent_relations` (db.rs:37171) and `_scoped`
   (db/scoped_entities.rs:297) — structural rework: the entity join is
   structurally legacy even when gated (the gate only overlays title
   hydration); rebuild the join on shadow pages outright
8. The 1.5a dated carryovers (10 sites, grep `G6 Stage 1.5a` carryover
   comments) — each either migrates here (space-sensitivity dissolved by
   Part 2) or is reclassified writer-coupled with a dated Stage 2 comment
9. `reader_uses_entity_pages` gated hybrids in `db/scoped_entities.rs` —
   collapse to the canonical branch unconditionally (hard-cut, same program
   contract as 1.5a; the gate function itself is deleted in Stage 2, so leave
   it dead-but-present here only if deleting it would touch writer code)

NOT in 1.5b: writers and writer-coupled discovery reads (Stage 2);
`assert_entities_have_shadow_pages` / `reconcile_entity_page_parity`
(cross-store by definition, Stage 2); `observations` content reads (survive per
the observations ruling — user veto still open).

## Tests (RED-control discipline as 1.2/1.3/1.5a)

1. m117 migration test: backfill correctness on a raw-seeded pre-m117 fixture
   (columns land, values match `entities` row-for-row).
2. Writer thread-through: `store_entity`/`merge_entities`/
   `refresh_entity_embedding`/`confirm_entity` round-trip the new scalars onto
   the page row.
3. Equivalence + asymmetric-divergence tests for the newly migrated
   product-surface readers (list/detail/search-by-name/vector), seeded via real
   writers; RED via prove-then-revert on one reader.
4. Space fold: post-migration, no NULL `entities.space` remains; a
   `scope_matches`-consumer test pinning the post-fold behavior.
5. Existing gated-hybrid legacy-vs-canonical A/B tests
   (scoped_entities_test.rs) rework or retire with the gate collapse —
   whichever the implementer picks, state it in the report.

## Gates

fmt, clippy both variants, focused modules; full suites queued by the session
lead at integration time. Program-doc status note for 1.5b in the same PR.

## Why the hard cut is safe where the gate was once load-bearing

(Recorded 2026-08-05, closing the reviewer's standing question from the 1.5a
round.) The gate's "only thing standing between a premature flip and a wrong
read" framing was true when the shadow store was young: the mirror could be
incomplete, and the gate's drift check was the guard. Three things changed
since: migrations 113/114 backfilled and asserted mirror completeness for the
whole store; the live-DB sweep receipts show sustained drift 0
(entity_page_parity 907/907, 0 entities without an active shadow page); and
every writer mints or updates the shadow page in the same transaction as the
legacy row, so incompleteness cannot recur short of a bug — which the
still-running parity reconciler would surface as nonzero drift. Stage 1's
program contract (readers cut hard, gates retire with the writers in Stage 2)
therefore rests on invariants that are enforced, not assumed.

Downgrade note (release-note material, not a code concern): after the space
fold, an old binary reading `space IS NULL` for uncategorized sees zero
unfiled entities — the same rollback property migration 91 established for
memories.
