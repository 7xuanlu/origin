# G6 Stage 2 — writers canonical-only, parity machinery retires

Status: FROZEN 2026-08-06. All three design forks ruled on the Opus-lane
investigation (findings summarized in place below), and both closing
drift-0 receipts captured from the live post-#493 daemon (binary 6e558894,
schema 119):

- `edges_parity_watermark` ts 1785985407: expected 2421 / actual 2421 /
  missing 0 / extra 0 / corrupt 0 — **drift 0**. (Expected dropped from the
  incident's 2464 because fix (a) folded the 48 phantom memory-kind
  expectations onto the evidence-implied ids, per the repair spec.)
- `entity_page_parity_watermark` ts 1785985588: expected 912 / actual 912 /
  drift 0.

These are the program's closing parity receipts; PR 2a copies them into its
body before retiring the machinery that produced them. Program contract:
`docs/plans/2026-08-04-g6-retirement-program.md` "Stage 2 — writers go
canonical-only, sweeps retire". Precondition: Stage 1 complete (five reader
cutovers shipped, 1.4 reclassified) and the edges-parity repair (PR #493,
m119) live with a fresh drift-0 watermark on BOTH oracles.

## Goal

Flip every dual-writing path to canonical-only, retire the parity oracles
and cutover machinery, and disposition every dated Stage-1 carryover — so
that Stage 3 is exactly one drop migration plus its safety ceremony, nothing
else. Rollback story (program doc): a Stage 2 revert restores dual-write;
canonical stores never stop being written.

## Ordering constraint (governs the PR split)

The parity sweeps derive their EXPECTED set from the legacy stores. The
moment any writer goes canonical-only, new canonical rows have no legacy
twin and the sweep reports "extra" drift — the oracle screams precisely
because the program succeeded. Therefore:

1. **Final receipts first.** Capture a dated drift-0 watermark payload from
   both sweeps on the live DB (edges + entity/page) under the post-#493
   binary. These are the program's closing parity receipts; they go in the
   Stage 2 PR body.
2. **Oracle retirement (PR 2a) lands before or with the first writer flip.**
   After 2a, correctness is carried by per-writer regression tests
   (drive path → assert canonical row correct AND no legacy write), not by
   the ambient diff.
3. **Writer flips follow** (PR 2b edges-side, PR 2c entity-side).

**Correction (2026-08-06, oracle-lifecycle split):** "Oracle retirement (PR
2a)" above named the ambient/production machinery only — the sweep
functions, scheduler lanes, and watermark tables listed under PR 2a below.
It did not cover the `#[cfg(test)]`-only diff functions the sweeps share
code with (`compute_edges_parity_report`/`ParityReport`,
`compute_entity_page_parity_report`/`EntityPageParityReport`), which PR 2a
left standing as TRANSITIONAL test oracles because at that point every
legacy store was still dual-written. Each test oracle's actual retirement
point is the PR that flips its last live writer to canonical-only, since
that is the point its "legacy store is ground truth" premise goes false:
`compute_edges_parity_report`/`ParityReport` retire in **PR 2b** (edges-side
writers all flip there), `compute_entity_page_parity_report`/
`EntityPageParityReport` stay live through **PR 2c** (entity-side writers
flip there instead). Not "both retire in 2a" and not "both ride to the end
of 2c" — the split follows each oracle's own writers.

## PR 2a — retire the oracle + cutover machinery

- `reconcile_edges_parity`, `reconcile_entity_page_parity` and their
  scheduler lanes; `edges_parity_watermark`, `entity_page_parity_watermark`
  tables (drop in migration m120); `WENLAN_ENABLE_EDGES_RECONCILE`,
  `WENLAN_ENABLE_ENTITY_PAGE_RECONCILE` flags (AGENTS.md doc rows removed,
  drift teeth #2 stays satisfied); the live daemon's plist entries for both
  flags (operator step, recorded in the PR body).
- Cutover levers: `set_reader_cutover`, `set_entity_reader_cutover`,
  `reader_uses_edges`, `reader_uses_entity_pages`, `SCOPED_ENTITIES_CONSUMER`;
  `edges_reader_cutover` and `entity_reader_cutover` tables drop in the same
  m120. **Correction (2026-08-06, implementer sweep):** "dead-but-present"
  holds only for the entity-side pair. `reader_uses_edges` has ONE live
  caller — `detect_communities`' adjacency branch (db.rs:36873). Ruling:
  collapse that branch to an unconditional edges-read and drop the legacy
  `relations` arm (the 1.5b `scoped_entities` hard-cut precedent). Safe
  today: writers dual-write both stores until 2b and the closing receipts
  prove agreement. Necessary, not merely safe: with the sweep retired no
  watermark can ever be current again, so the fail-closed gate would
  permanently pin any not-yet-stamped DB to the legacy path — keeping the
  gate is the broken option, not the conservative one. The
  `detect_communities_edges_path_matches_legacy` A/B pin survives 2a as a
  direct positive control iff its legacy arm can live inline in the test
  (writers still dual-write, so equivalence still holds); it retires in 2b
  with the relations writers. The three cutover-lever tests retire
  category-2.
- **Backfill machinery ruling:** `backfill_edges_from_*` (5 fns) CANNOT be
  deleted — a fresh DB replays every migration from `user_version 0`
  (`run_migrations`, db.rs:3814; each gated `if version < N`), so m81
  (db.rs:9163-9167) and m111 (db.rs:12757) call them on every new install.
  They demote to migration-internal private helpers, documented as
  m81/m111-only, removed from any live-writer surface. Deletion only ever
  happens with a migration re-baseline, which is out of program scope.
- Tests: m120 migration family (tables gone, idempotent); teeth: no
  remaining references to the retired flags/fns outside migrations.

## PR 2b — edges-side writers go canonical-only

Per store group, with the store's Stage-3 disposition beside it:

| Store | Stage 2 writer change | Stage 3 end state |
|---|---|---|
| `relations` | dual-writers stop writing `relations` rows (mint/retire edges only) | dropped |
| `page_sources` | writers stop — UNBLOCKED by Q2: the cites semantic payload (`source_kind`/`linked_at`/`link_reason`/`title`, `cites_semantic_patch` db.rs:14995, applied at all six mint sites, backfilled by m115) already carries the unrecoverable columns at 100% live coverage (1992/1992) | dropped |
| `page_evidence` | writers stop — same Q2 basis; the authored/NULL-locator shape is CLOSED at the writer instead of migrated (see below) | dropped |
| `page_links` | resolved-link rows stop; orphan rows KEEP being written (recorded 1.1 decision: narrows to an orphan-only store, not dropped) | orphan-only side-table survives |
| `pages.citations` | edge-backing role retires: D7 refcount helpers (`cites_backed_by_page_citations`, `cites_backed_outside_page_citations`) and every retire-guard consulting them go away; column becomes pure render cache (1.4 reclassification) | column survives as render cache |

**Q2 rulings (investigation confirmed the extension shipped; nothing to
build, two guards to add):**

- **Payload-coverage verification gate, not implementation.** A teeth-style
  test asserts every active `cites` edge carries `source_kind`, and
  `linked_at`/`link_reason` wherever its legacy row has them. Passes today
  (1934 memory + 58 external, 0 NULL payloads); its job is to make the
  writer flip unable to silently regress payload coverage.
- **Close the authored/NULL-locator writer; no migration machinery for an
  empty set.** The live set is zero rows and the only entry point
  (`link_page_evidence`, db.rs:49418) has exclusively test callers. It goes
  `#[cfg(test)]`, so the schema stops admitting a row with no canonical
  representation. Belt-and-braces: Stage 3's drop migration carries a
  pre-drop assertion that the authored/NULL-locator set is still empty
  (fails loud if a future writer regresses this).
- **Recorded so nobody re-derives it as a gap:** m115's page_sources pass
  hard-codes `"memory"` in its edge-id derivation (db.rs:13439-13441) — the
  PR #493 Defect-2 shape. Harmless in effect: for folder-doc rows it patched
  a phantom id and matched zero rows, and pass 3 (stored `source_kind`,
  db.rs:13365-13369) covered those rows, which is why external coverage
  reads 58/58. Migrations are history; do not retro-fix.

Also in 2b:

- **The 3 distill-eligibility predicates** (`query_distillation_staging_pool`,
  `query_distillation_seed_slice`, `query_distillation_ann_neighbors` —
  Stage 1.3 S3b carryover) widen to the edges/cites union with their own
  test attention, as dated in place.
- **Repair-rollback residual** (program doc lines 55-58) — RULED, receipt
  model, not the undo model the program doc assumed. The `rollback-v1.json`
  artifact does not restore legacy-store rows byte-wise anywhere in the
  pipeline: its deserialized bytes feed only a CAS precondition (hashed into
  a receipt via `target_receipt`, compared against
  `manifest.expected_state().canonical_receipt()` at apply/recovery time)
  and the strict per-target shape check `rollback_matches_target`. No code
  path takes a `StoredRollbackArtifact` for `relations`/`page_sources`/
  `page_evidence`/`page_links` and writes its rows back into any table —
  confirmed by tracing every consumer of `rollback.rows`/`.columns`/`.table`
  in `repair.rs`. The two functions that DO restore something
  (`restore_page_projection_snapshot`, and `RenamePageTitleRecoveryArtifact`
  via `load_rename_page_title_rollback`) operate on a structurally disjoint
  rollback shape — page-projection files and page-title-rename payloads,
  each gated by its own table/variant check that can never accept the four
  DB-table rollbacks this item concerns. So there is nothing to version or
  re-capture: the fix is a refusal guard at the single deserialization choke
  point (`load_rollback`, repair.rs:417), not an artifact-format change.
  Shipped: `rollback_targets_retired_store` (repair.rs:1336) refuses
  `relations`/`page_sources`/`page_evidence` unconditionally (their
  "current state" precondition can no longer be trusted once nothing writes
  them) and admits `page_links` only when every captured row is provably
  orphan-shaped (`target_page_id` reads back `"NULL"`) — `page_links`
  survives as item 3's orphan side-table, so a repair whose precondition is
  a resolved `page_links` row is refused the same as the three frozen
  stores, while a genuine orphan-bind repair still proceeds.
- Tests per flipped writer: drive the real path, assert the canonical
  mint/retire AND zero rows written to the legacy store (RED control:
  un-flip one writer, its no-legacy-write assertion must fail).

### Discovery-scan sweep findings (implementer sweep, 2026-08-06)

Flipping a writer's own INSERT/UPDATE side to canonical-only does not make it
safe: a function can dual-write correctly and still scan the now-frozen
`relations` table to DISCOVER which rows to act on, silently finding zero
once nothing populates that table for new facts. Seven instances of this
"discover-via-frozen-store" bug class turned up while working PR 2b, in the
order found (instances 5, 6, and 8 scan `page_sources`/`page_evidence`/
`page_links`, not `relations`, but are the same discover-via-frozen-store
shape). Found between them, instance 7 is a distinct, related finding — a
missing inverse operation, not a frozen-store scan — recorded alongside
the others in discovery order:

1. `edge_grounding_candidates` / `promote_edges_grounded` (M3g). Fixed
   pre-2b.
2. `merge_entities`'s `merge_edge_plans` scan. Fixed in this PR.
3. `invalidate_relation_edges_for_source_in_transaction`'s discovery
   `SELECT`. Fixed in this PR.
4. `fold_relation_type`'s two discovery `SELECT`s (db.rs, the fold's
   snapshot scan and its per-row survivor lookup) plus
   `distinct_relation_types_for_vocabulary_heal`
   (db/kg_quality_vocabulary.rs) — `heal_relation_vocabulary`'s writer
   (`fold_relation_type`) and its own discovery read shared the same
   `relations`-scan hazard, one item-1 gap the sweep caught. Ported
   DISCOVERY only: both `fold_relation_type` `SELECT`s now scan
   `edges` (`edge_type='relates' AND src_kind='entity' AND
   dst_kind='entity' AND valid_until IS NULL`), reading the semantic-key
   fields via `json_extract(payload, ...)` and `created_at` off the
   edges row itself — the same pattern `merge_edge_plans` (instance 2)
   established. `distinct_relation_types_for_vocabulary_heal` now reads
   `DISTINCT semantic_type FROM edges WHERE edge_type='relates' AND
   valid_until IS NULL`. The surrounding merge/ledger/community-bump
   logic is unchanged; the `vocab_heal_ledger` pre-image's `"id"` key
   is renamed to `"edge_id"` (no structural consumer depends on the old
   name, confirmed by sweep — only string-`.contains()` test checks).
   Two pre-existing tests seeded fixtures via raw `INSERT INTO
   relations`, which the port makes inert (discovery finds nothing);
   they now seed `edges` directly instead — a fixture change, not a
   `fold_relation_type` semantics change. New acceptance pin:
   `heal_relation_vocabulary_discovers_and_folds_edges_collision_keeps_stronger`
   (kg_quality.rs), exercising the full `heal_relation_vocabulary` ->
   `fold_relation_type` path against a genuine collision (a non-canonical
   type folding into an already-live canonical edge) and asserting the
   survivor keeps the stronger confidence rather than being duplicate-
   asserted. `docs/plans/2026-08-05-g6-stage12-relations-readers-spec.md`
   item 10's "heal writers still write `relations`" justification for
   deferring `distinct_relation_types_for_vocabulary_heal` is now STALE
   (superseded by this fix) — read that doc historically, not as current
   routing.
5. `rebind_source_id_inner`'s affected-pages scan (db.rs, the page-rebind
   citation-move step). Fixed in this PR. Discovered which pages cited a
   renamed memory locator by scanning `page_sources UNION page_evidence`
   before moving their `cites` edges onto the new locator — a citation
   minted straight to `edges` after the item 1/2 cutover was invisible to
   that scan, so its edge stayed stranded on the old locator forever after
   a rename. Fix: scan `edges` directly (`edge_type='cites' AND
   src_kind='page' AND dst_kind='memory' AND dst_id=?old_source_id AND
   valid_until IS NULL`), no union needed — `edges` is a superset of both
   frozen stores (drift-0 receipts + migration 111's repair). Test:
   `rebind_source_id_moves_durable_provenance_without_memory_generation_bump`
   (main_tests.rs) now asserts the old locator's `cites` edge retires and
   the new locator's `cites` edge is live, plus the same move for the
   "relates" edge's `payload.source_memory_id` — ported off raw
   `relations`-row counting, which the same PR's rebind-scope narrowing
   made inert for that table too (`rebind_source_id_inner` stopped
   touching `relations` alongside `page_sources`/`page_evidence`, per its
   own in-code note).
6. `delete_by_source_id_in_transaction`'s remaining-sources recompute
   (db.rs, the per-page loop after cites-edge retirement). Fixed in this
   PR. On any memory delete, every page depending on the deleted locator
   recomputes its `pages.source_memory_ids` mirror by merging the page's
   pre-delete mirror value with a `page_sources` row scan for that page —
   but `page_sources` is never pruned once a citation is legitimately
   dropped (item 1 stopped writing it going forward), so a stale row for a
   citation an earlier edit had already removed got merged straight back
   into the mirror on every LATER, unrelated memory delete — resurrecting
   a dropped citation. Confirmed by a deterministic repro: page cites A and
   B, a content edit legitimately drops B (mirror -> [A, C]), an unrelated
   delete of C recomputes the mirror and B reappears. Fix: merge from live
   `cites` edges instead of `page_sources`
   (`edge_type='cites' AND src_kind='page' AND src_id=?page_id AND
   dst_kind='memory' AND valid_until IS NULL`) — same pattern as instance
   5. The being-deleted locator's edge is already retired earlier in the
   same transaction, so the scan naturally excludes it. Test:
   `delete_by_source_id_recompute_does_not_resurrect_dropped_citation`
   (main_tests.rs) pins the repro above.
7. `delete_page`'s inbound-link orphaning UPDATE (db.rs). Fixed in this
   PR. A distinct class from the frozen-store discovery-scan bugs above —
   a missing INVERSE operation in the item-3 narrowed page_links design,
   not a frozen-table scan. Since item 3, a RESOLVED link never gets a
   `page_links` row (resolution deletes the row and mints a `links` edge
   instead), so `delete_page`'s `UPDATE page_links SET target_page_id =
   NULL WHERE target_page_id = ?1` had nothing to touch for a resolved
   inbound link — the link vanished from both `get_page_outbound_links_scoped`
   read sources (live edges + orphan page_links rows) instead of
   re-orphaning, reproduced as an index-out-of-bounds panic. Fix: in the
   same transaction as the edge retirement, INSERT a fresh orphan
   `page_links` row (label read from the retiring edge's payload, same
   convention as `replace_page_links`) for each inbound live `links` edge
   whose source page survives the delete. Test:
   `page_links_target_delete_becomes_orphan_and_reresolves` (main_tests.rs).
8. `accept_page_merge`'s inbound-link repoint scan (db.rs). Fixed in this
   PR. Same discover-via-frozen-store shape as instances 5/6: the repoint
   read `SELECT pl.source_page_id, pl.label_key, pl.target_page_id,
   p.space, pl.label FROM page_links pl ... WHERE pl.target_page_id = ?1
   OR pl.label_key = ?2` only ever matched a `page_links` row. Since item
   3, neither live writer of that table (`replace_page_links`,
   `resolve_orphan_page_links`) ever leaves a row with a non-NULL
   `target_page_id` — a resolved link's row is deleted the moment it
   resolves, and the minted `links` edge becomes its sole canonical
   representation — so the `pl.target_page_id = ?1` arm of the WHERE
   clause can no longer match anything for a link resolved after
   cutover. A RESOLVED inbound link to the absorbed page was therefore
   invisible to the repoint and stayed pointed at the now-archived page
   instead of following it to the merge survivor. Confirmed live (not
   just theoretical) via `accept_page_merge_reconciles_links_and_evidence_edges`,
   which failed at its "absorbed-target links edge is retired" assertion
   before this fix. Fix: a second scan, `SELECT e.src_id,
   json_extract(e.payload, '$.label'), p.space FROM edges e ... WHERE
   e.edge_type = 'links' AND e.dst_id = ?1 AND e.valid_until IS NULL`,
   folding every live inbound edge into the same `repointed` batch the
   existing mint/retire loop already drains — the page_links-side scan is
   left in place unchanged (harmless backward-compat for any pre-cutover
   row that might still exist, never populated by a live writer going
   forward). Test: `accept_page_merge_reconciles_links_and_evidence_edges`
   (main_tests.rs) — pre-existing test, unmodified, green post-fix.

## PR 2c — vector re-home, then entity-side writers go canonical-only

**Q1 RULED: re-home in Stage 2, option (i), as ordered sub-steps inside 2c
ahead of the writer flip.** The investigation found the re-home target
already exists and is fully populated: `pages` has an `embedding` column
with a live DiskANN index (`idx_pages_embedding`, db.rs:6754, self-heal
recreate at :6771), every entity shadow page already carries its entity's
embedding (912/912, zero NULLs both sides — mirror is atomic in
`refresh_entity_embedding`, db.rs:32906, pinned by existing tests), and
only ONE query depends on `entities_vec_idx` (the Global path,
db.rs:27893; the scoped variant is a brute-force scan, and Global already
has a brute-force fallback). Rejected: deferring to Stage 3 concentrates
three unrelated failure modes into the least-reversible migration; the
`child_vectors` shell is untested-and-inert surface for no gain. The
deciding argument: parity between the two embedding copies is a FREE A/B
oracle that disappears the moment `entities` stops being written — the
re-home is the cheapest it will ever be right now.

Ordered sub-steps:

1. **Query swap.** Re-point the Global ANN query at `idx_pages_embedding`
   with a `kind='entity'` post-filter and 3x over-fetch (the `search_pages`
   precedent, db.rs:48258 — entity pages are 82% of the index, so 3x is
   comfortable at caller limits of 1-5). Switch the scoped brute-force scan
   from `entities` to the shadow pages. `entities_vec_idx` stays as
   fallback during this sub-step. Verification: A/B the two indexes on the
   live store — parity means identical top-k entity sets; any disagreement
   is crowding and gets measured, not assumed.
2. **Mirror inversion.** Rewrite `refresh_entity_embedding` to embed
   directly onto the shadow page (today it writes `entities` then copies —
   post-flip that copies NULL-or-stale). `update_entity_shadow_page`
   retires with the store it mirrors.
3. **Writer flip.** Entity writers stop writing `entities`/`entity_aliases`
   (alias truth already in shadow-page payload since m113/m114);
   `entities_vec_idx` retires with them.

Also in 2c:

- The 3 writer-coupled 1.5a carryovers flip here: the two repair validators
  (`repair.rs:4709`, `:4743`) and `entity_exists`.
- `observations`: writers KEEP writing (reclassified out of the Stage 3
  drop list, 1.5a spec ruling — user veto open). No Stage 2 change.

## Q3 ruling — intentionally-legacy lint audits

**Keep all nine (7 entity-store + 2 provenance cross-store) through
Stage 2; they retire in Stage 3 with the stores they audit.** The tripwire
argument decides it: after the flips a frozen store's audit output should
be CONSTANT — a changing output is precisely the "some writer you believed
flipped is still writing legacy" alarm, and retiring the instrument at the
flip removes it when it earns its keep most. The cost of keeping them is
one stage of a read-only check.

Two Stage 2 obligations fall out:

- **Fix the eight drifted comments.** The 1.5a entity-lint comments say
  "Stays on `entities` until Stage 2 retires the store itself" — factually
  wrong about the program shape (Stage 2 retires writers, Stage 3 drops
  stores): `lint/kg/query.rs:56/:76/:102`, `lint/kg/query/aggregate.rs:30`,
  `lint/deep.rs:96/:220/:273`, `lint/semantic_candidates.rs:846`. Reword to
  the 1.3 form ("retires with the stores at Stage 3",
  `lint/pages/provenance_checks/source.rs:40-42`, which is correct).
- **Program-doc checklist line.** The audits' retirement becomes an explicit
  Stage 3 checklist item in the program doc, not eight scattered code
  comments (they already drifted once).

Same disposition, one relations-side diagnostic found during the discovery
sweep (see "Discovery-scan sweep findings" below): `count_stale_relation_sources`
(db/kg_quality_diagnostics.rs) stays on `relations` and joins the Stage 3
audits-retirement item — pure `log::warn!` visibility via `detect_stale_relations`,
no behavioral effect, so a constant post-cutover count is the tripwire, not a
bug to fix. Classified (a) Tripwire vs. `fold_relation_type`'s discovery
`SELECT`s below, classified (b) Live-bug (`heal_relation_vocabulary` actually
acts on what it discovers) — the two calls read the same table but the
behavioral-effect test puts them on opposite sides of the Stage 2/Stage 3
line.

## Fork register (all ruled 2026-08-06)

| # | Question | Ruling |
|---|---|---|
| Q1 | Entity vector index re-home | Stage 2, inside 2c, ordered ahead of the writer flip (query swap → mirror inversion → flip); parity A/B is the oracle |
| Q2 | cites payload extension + authored NULL-locator rows | Already shipped (m115, 100% live coverage); Stage 2 adds a coverage teeth-test and closes `link_page_evidence` to `#[cfg(test)]`; Stage 3 adds a pre-drop empty-set assertion |
| Q3 | Intentionally-legacy lint audits | Keep through Stage 2 as frozen-store tripwires; retire in Stage 3; fix the eight drifted comments now |

## Exit criteria

- Zero dual-writes: no non-migration code path writes `relations`,
  `page_sources`, `page_evidence` (post-Q2 shape), resolved `page_links`,
  or `entities`/`entity_aliases` (post-Q1 shape) outside the canonical
  stores.
- Oracle machinery gone; closing drift-0 receipts recorded in the PR
  bodies.
- Every dated Stage-1 carryover dispositioned (flipped, retired, or
  explicitly re-dated with a Stage 3 owner).
- Stage 3 preconditions all that remain: backup + operator-verified restore
  drill + declared downgrade barrier + the user's explicit
  point-of-no-return confirmation.
