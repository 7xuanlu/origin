# G6 Stage 1.5a — migrate the CLEAN entity readers onto shadow pages

Parent program: `docs/plans/2026-08-04-g6-retirement-program.md` (Stage 1, store 5).
Grounding: the 2026-08-05 entity-reader inventory (scout, delivered in-session; key
facts restated here — the implementer verifies every site at the code level before
editing, and bounces back any site whose read contradicts this spec rather than
improvising). Prerequisites: Stages 1.1–1.3 merged. NO schema change in this PR —
every reader migrated here needs only fields the shadow pages already mirror
(title=name, entity_type, confidence, entity_confirmed, embedding, space,
workspace, aliases via `entity_page_map`).

## What this PR is NOT (deferred to 1.5b, decisions recorded below)

- NO migration of readers needing `source_agent`, `created_at`/`updated_at`,
  `community_id`, `embedding_updated_at`, or `observations` content — those wait
  for the 1.5b scalar mirror extension (m117) and the observations ruling.
- NO change to the `reader_uses_entity_pages` gated hybrid paths in
  `db/scoped_entities.rs` — they keep their residual direct reads until 1.5b.
- NO touch of the WRITER-COUPLED shared-CRUD core (`store_entity`,
  `merge_entities`, `commit_entity_enrichment_at_version`, `add_entity_alias`,
  `refresh_entity_embedding`, `confirm_entity`, `add_observation`,
  `create_relation_with_span`, `delete_entity`, `fold_entity_type`,
  `resolve_entity_by_alias`, dedup-candidate + repair-apply EXISTS guards) —
  writers and their coupled discovery reads flip together in Stage 2.

## The space-sentinel trap (this stage's Trap 1)

`insert_entity_shadow_page`/`update_entity_shadow_page` fold NULL
`entities.space` to the `UNFILED_SPACE_ID` sentinel on the page row, so
`pages.space` ≠ `entities.space` semantically. Any migrated reader that filters
or projects space MUST either (a) prove its consumers treat NULL and the
sentinel identically, or (b) keep reading `entities.space` with a dated comment
(making it a 1.5b carryover, not a clean migration). `load_entities`
(lint/semantic_candidates.rs:748) is known to distinguish via `scope_matches` —
it is NOT in this PR's migration set despite otherwise-clean fields. The
implementer verifies the space handling of every reader below before migrating
it; any that turns out space-sensitive gets the (b) treatment and a report line.

## Migration targets (verify each; migrate those that hold)

All from the inventory's CLEAN set, excluding space-sensitive and
writer-coupled entries. Target store: `pages` joined via `entity_page_map`
(`kind='entity'`, `status='active'`), fields per the mirror. Keep result
shapes and ordering byte-compatible; where legacy ordered by `entities.name`,
order by `pages.title` (same value by mirror invariant).

1. `expand_anchor_entities_khop` (db.rs:26188) — id/space existence checks
2. `expand_entities_khop_scoped` (db.rs:26746) — same
3. `filter_entity_ids_scoped` (db/scoped_entities.rs:575)
4. `get_entity_name_type` (db.rs:31766) — name/entity_type projection
5. `count_entities` (db.rs:36983)
6. `get_space_by_id` entity count (db/space_context.rs:12)
7. distillation query trio's `e.name` join (db.rs:39837/40110/40164)
8. `entity_integrity` (lint/kg/query.rs:48) — name/entity_type/confirmed/confidence/space
9. `relation_integrity` / `link_integrity` entity-existence sides (lint/kg/query.rs:83/100)
10. aggregate readers (lint/kg/query/aggregate.rs:30/54/72)
11. `alias_integrity` entity-existence side (lint/deep.rs:89)
12. `relation_vocabulary` entity side (lint/deep.rs)
13. `load_relations` entity join (lint/semantic_candidates.rs:831)
14. `load_record_inventory` entity side (repair_plan/semantic.rs:164)
15. `resolve_memory_entity_links` (repair_plan/deterministic.rs:1003)
16. `validate_selected_entities_on_{snapshot,connection}` (repair.rs:4709/4738)
17. `list_contradiction_observation_counts` entity-name side only (db/kg_quality_diagnostics.rs:41)
18. `observation_integrity` / `observation_duplicates` entity-existence sides only
    (lint/kg/query.rs:66, lint/deep.rs:205) — the observations reads stay
19. `entity_exists` (db.rs:33103) — verify no writer coupling bites; it guards
    relation writes, reading the store `store_entity` writes; if the coupling
    argument holds (write path must see its own uncommitted row), carryover with
    a dated comment instead
20. `EDGE_GROUNDING_CANDIDATE_SCAN_SQL` name hydration (db.rs:14993)
21. `eval_lifecycle_integrity` count (db/eval_lifecycle_integrity.rs:49)
22. `resolve_entity_by_name`'s own SQL (db.rs:33398) — id-by-name; its
    fallthrough to `search_entities_by_vector` stays legacy (BLOCKED, 1.5b)
23. `search_memory_with_cue`'s post-search `entity_name` hydration
    (db.rs:24754) — name-only, space-free `SELECT id, name FROM entities
    WHERE id IN (…)` over the result set's entity ids; missing from the
    original scout inventory, added by review during the fix round
    (2026-08-05)

Lint/repair readers here follow the 1.2/1.3 precedent: a well-formedness check
over the canonical store migrates; a cross-store consistency check does not.
`assert_entities_have_shadow_pages` and `reconcile_entity_page_parity` are
cross-store by definition — untouched.

## The `create_entity` landmine (fix in this PR)

`create_entity` (db.rs:31475) is a live `pub async fn` with zero production
callers that does NOT write a shadow page — any future caller silently violates
the shadow invariant. Fix: `#[cfg(test)]`-gate it AND make it call
`insert_entity_shadow_page` inside its transaction, so even test fixtures
produce invariant-complete state (several 1.5a tests will rely on that).

## Tests (RED-control discipline as 1.2/1.3)

1. Equivalence per migrated product-adjacent reader (khop pair, name_type,
   counts) — seeded via real writers (`store_entity`), field-by-field.
2. Divergence test, asymmetric seeding: one entity present only as a shadow
   page + map row (seeded by writer then legacy row deleted via raw SQL), two
   present only as raw `entities` rows without shadow pages (raw SQL, bypassing
   the invariant on purpose); assert migrated readers see 1, not 2. RED via
   prove-then-revert mutation of one reader.
3. Mirror-invariant leans: a test asserting `pages.title/entity_type/confidence`
   round-trip `store_entity` updates (pins the mirror this PR now load-bears).
4. Parity: `entity_page_parity_watermark` fixtures unchanged, still drift 0.

## Gates

fmt, clippy both variants, focused modules; full suites queued by the session
lead at integration time. Program-doc status note for 1.5a in the same PR.

## 1.5b decisions recorded (design ruled in-session 2026-08-05, execution later)

- **Scalar mirror extension (m117):** `source_agent`, `created_at`,
  `updated_at`, `community_id`, `embedding_updated_at` become entity-page
  columns per the M3 precedent (entity_type/confidence/entity_confirmed are
  already real `pages` columns), with a backfill migration and writer
  thread-through. Unblocks `list_entities(_scoped)`, `get_entity_detail`
  entity-half, `search_entities_by_vector(_scoped)`, `search_entities_by_name`,
  `load_summary_buckets` legacy branch, `summary_eligible_predicate` legacy
  branch, embedding-refresh sweep.
- **Space sentinel:** target state is "space is never NULL" — the retirement
  migration folds NULL→sentinel in `entities` before the drop, and consumers
  that distinguish must be reworked or the distinction declared dead by then.
  1.5b carries the audit of `scope_matches` consumers.
- **Observations (FLAGGED for user veto, same class as the 1.4
  reclassification):** `observations` is NOT a duplicated legacy store — it is
  the only store of observation content, with no canonical twin and no
  dual-write. Ruling: reclassify out of the Stage 3 drop list; it survives as
  the single source of truth for observation content, re-anchored if needed
  (entity ids stay meaningful post-drop — they are the edges src/dst identity
  space). The alternative (a JSON observations column on shadow pages) puts an
  unbounded per-entity collection into a row column and couples every
  observation write to a page-row rewrite; rejected. Stage 3's drop list
  shrinks accordingly: `entities`, `entity_aliases`, `entity_page_map` drop;
  `observations` stays.
- **`list_recent_relations` structural join:** the entity join is structurally
  legacy even when gated (gate only overlays title hydration) — rework rides
  1.5b with the scalar extension.
