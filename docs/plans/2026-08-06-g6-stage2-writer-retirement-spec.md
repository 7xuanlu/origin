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
- **Repair-rollback residual** (program doc lines 55-58): the
  `rollback-v1.json` artifact restores legacy-store rows byte-wise. Once
  repair writers are canonical-only, the rollback artifact must
  capture/restore canonical `edges` rows instead; a legacy-row restore
  against a frozen store is dead weight at best and a Stage 3 crash at
  worst. Shape: version the artifact (v2 captures edges rows), refuse to
  apply a v1 artifact whose rows target a retired store.
- Tests per flipped writer: drive the real path, assert the canonical
  mint/retire AND zero rows written to the legacy store (RED control:
  un-flip one writer, its no-legacy-write assertion must fail).

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
