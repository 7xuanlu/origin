# M6 D12 — automatic writer/caller manifest

Date authored: 2026-08-01
Branch: `kg-m6-stage0`, cut from `origin/main` `e39048c7` (release 0.15.2, post-refactor `a028199f`).
Stage: M6 **Stage 0**, artifact 9 of 12 ("Automatic writer/caller manifest with exact fence adapter per symbol").
Status: **contract artifact, docs only.** No fence-adapter code exists yet and none should be written before daemon PR-D. Zero production code changed by this document.

The frozen M6 goal prompt (2026-07-27, Sol closure APPROVE) is not amended here. D12 named five fence targets plus "any production caller discovered by LSP/ast-grep manifests". This document regenerates those anchors against the post-refactor tree and enumerates the callers the frozen contract could not name.

## 1. What "automatic" means here

A caller is **automatic** when the page mutation is not the direct result of an explicit human or API authoring action — it is reached from a scheduler tick, an ambient job, a refinery phase, a post-ingest hook, or a background queue worker.

D12's carve-out is explicit: *existing explicit authored/source creation remains legal through the M0 gate, but does not count as automatic genesis without D1/D2 evidence*. Section 5 lists those explicit paths with the reason each is excluded, so the exclusion is auditable rather than implied by absence.

Two classes of fence policy appear in the manifest:

- **`route`** — the adapter consults the per-space M6 maintenance/genesis generation and, after cutover, hands the write to the M6 writer (D9 attachment finalizer, D10 refresh finalizer, or the D3 genesis finalizer). Before cutover it passes through to the current writer.
- **`pass_through`** — the write is automatic in trigger but its content is either human-authored or a deterministic 1:1 mirror of a non-page row. The adapter exists so the fence and G9 can *see* the write; it must never route the write into M6 genesis, relevance, or coverage. Listing these inside the manifest rather than in the excluded section is the fail-closed choice: a future refactor that turns one of them into a synthesis path then trips the gate instead of slipping through.

## 2. Method and tool provenance

The sweep ran **seam-inward-out**: locate every SQL statement that mutates the `pages` table, fold those statements into the Rust functions that own them, then walk callers outward.

| Step | Tool | What it produced |
|---|---|---|
| Find every `pages`-row mutation | `grep -rnoiE 'INSERT (OR …)? INTO pages\|UPDATE pages\|DELETE FROM pages'` over `crates/*/src` | 105 statements across 15 files; 71 of them in `crates/wenlan-core/src/db.rs` |
| Fold statements into owning functions | script over the file's `fn` definition lines | 43 distinct functions in `db.rs` + 9 outside it |
| Enumerate call sites of the seam API | **`ast-grep` 0.44.0**, Rust grammar, patterns `SYM($$$A)`, `$P::SYM($$$A)`, `$X.SYM($$$A)` | the caller lists in sections 4–6 |
| Classify prod / test / eval | per-file `#[cfg(test)]` boundary walk, then reading each site | reclassified 4 sites the pattern match alone got wrong |
| Confirm every cited line | `Read` on each `file:line` before it entered a table | — |

**LSP was attempted and is not available for this workspace.** A dedicated read-only agent spent roughly twelve minutes trying `findReferences`, `goToDefinition`, `prepareCallHierarchy`, and `workspaceSymbol` against this worktree. `documentSymbol` (syntactic, single-file) worked; every semantic method returned empty. Diagnosis: no `rust-analyzer` process was running for this worktree, no `target/` directory was ever created in it, and `workspaceSymbol` returned hits only from a *different*, already-indexed repository. So the caller graph below rests on `ast-grep`'s Rust AST matching plus per-site reading, not on semantic references.

The honest limit that follows: an AST sweep catches syntactic call expressions and struct-literal constructions under the literal names used. It cannot see a call reached through a trait object or a function pointer. Every symbol in this manifest is a plain free function, an inherent `MemoryDB` method, or an enum variant — no trait indirection is visible in the code read — so the residual risk is low but not zero. Section 8 states the closure argument that does not depend on the reference sweep being exhaustive.

## 3. The write seam

Every page mutation in production reaches SQLite through one of three doors.

```
                    ┌─────────────────────────────────────────┐
  automatic         │  post_write::page_write(db, PageWrite)   │   ← canonical seam
  callers  ───────► │  page_dispatch.rs:77                     │
  (§4)              │    Attach │ Create │ Update │            │
                    │    UpdatePreservingSources │             │
                    │    ReplaceSource │ DocumentSource        │
                    └───────────────┬─────────────────────────┘
                                    │  page_create.rs / page_update.rs
                                    ▼
                    ┌─────────────────────────────────────────┐
  automatic         │  MemoryDB page mutators in db.rs         │
  callers  ───────► │  set_page_stale, clear_page_staleness,   │   ← side door: row
  (§4)              │  replace_page_sources, archive_page,     │     fields, not prose
                    │  set_page_citations_*, update_page_…     │
                    └───────────────┬─────────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────────┐
  entity KG   ─────►│  insert/update_entity_shadow_page        │   ← mirror door:
  mutators          │  db.rs:9709 / db.rs:9752                 │     kind='entity' rows
                    └─────────────────────────────────────────┘
```

The canonical seam is `page_write` at `crates/wenlan-core/src/post_write/page_dispatch.rs:77`, dispatching the `PageWrite` enum (`page_dispatch.rs:13`) into `page_create.rs` / `page_update.rs`. Seven public wrappers sit on top of it: `create_page` (`:202`), `create_page_with_floor` (`:220`), `create_page_with_tuning` (`:238`), `update_page` (`:285`), `update_page_preserving_sources` (`:313`), `update_page_at_source_revision` (`:333`), and `update_page_growth_at_versions` (`:363`).

Every write through the seam records a writer identity, classified by `Writer::classify` at `crates/wenlan-core/src/post_write/page_update.rs:48`. The thirteen writer literals that appear in production code are an independent enumeration of the same population, and section 8 uses them as a cross-check.

## 4. Manifest — fenced automatic writers (machine-readable)

This block is the artifact a G9 structural CI test consumes. Fields are pipe-delimited, one row per fenced symbol, no padding inside fields. `defined_at` is a `file:line` on this branch; `write_seam` is the exact mutation the symbol performs or delegates.

```manifest
# m6-writer-manifest-v1
# branch: kg-m6-stage0 @ e39048c7
# fields: symbol|defined_at|write_seam|writer_identity|trigger|policy|fence_adapter
detect_page_candidates|crates/wenlan-core/src/synthesis/detect.rs:18|page_write PageWrite::Attach @ detect.rs:73|detect|refinery Phase::Detect @ refinery/mod.rs:982|route|m6_fence::detect_attach
distill_one_cluster_with_tuning|crates/wenlan-core/src/synthesis/distill.rs:464|page_write PageWrite::Attach @ distill.rs:501,540; PageWrite::Create @ distill.rs:735|distill,system|refinery Phase::Emergence via distill_pages_scoped_gated|route|m6_fence::emergence_cluster_write
distill_pages_scoped_gated|crates/wenlan-core/src/synthesis/distill.rs:993|delegates to distill_one_cluster_with_tuning @ distill.rs:1110|distill|refinery Phase::Emergence @ refinery/mod.rs:1014|route|m6_fence::emergence_distill_gate
run_page_growth_slice|crates/wenlan-core/src/post_ingest.rs:128|update_page_growth_at_versions @ post_ingest.rs:324|page_growth|scheduler AmbientJob::PageGrowth @ wenlan-server/src/scheduler/ambient.rs:449|route|m6_fence::growth_slice_update
grow_page|crates/wenlan-core/src/post_ingest.rs:822|update_page @ post_ingest.rs:953|page_growth|post-ingest enrichment @ post_ingest.rs:685|route|m6_fence::growth_page_update
enqueue_changed_pages|crates/wenlan-core/src/refinery/mod.rs:1393|set_page_stale @ refinery/mod.rs:1402|source_updated|refinery Phase::ReDistill @ refinery/mod.rs:1087|route|m6_fence::redistill_enqueue_stale
re_distill_stale_pages|crates/wenlan-core/src/refinery/mod.rs:1422|refresh_page @ refinery/mod.rs:1451|re_distill|refinery Phase::ReDistill @ refinery/mod.rs:1092|route|m6_fence::redistill_batch_refresh
run_redistill_page_slice|crates/wenlan-core/src/refinery/mod.rs:1570|refresh_page @ refinery/mod.rs:1609,1681; set_page_stale @ :1680; clear_page_staleness @ :1620,1693|re_distill,source_updated|refinery phase slice @ refinery/mod.rs:663 from scheduler fire_steep_phase @ wenlan-server/src/scheduler.rs:1879|route|m6_fence::redistill_slice_refresh
refresh_page|crates/wenlan-core/src/synthesis/distill.rs:1208|delegates to refresh_page_with_prompt @ distill.rs:1216|re_distill|shared refresh writer for redistill, maintenance, overview|route|m6_fence::page_refresh_write
refresh_page_with_prompt|crates/wenlan-core/src/synthesis/distill.rs:1236|update_page_at_source_revision @ distill.rs:1360|re_distill|called by refresh_page and refresh_overview_page|route|m6_fence::page_refresh_prompt_write
maybe_refresh_overview_page|crates/wenlan-core/src/refinery/mod.rs:1727|refresh_overview_page @ refinery/mod.rs:1737|refinery|refinery Phase::Overview @ refinery/mod.rs:1120|route|m6_fence::refinery_overview_refresh
refresh_overview_page|crates/wenlan-core/src/synthesis/overview.rs:104|replace_page_sources @ overview.rs:115; set_page_stale @ :118; refresh_page_with_prompt @ :124|overview_sync|maintenance tick/slice and refinery Phase::Overview|route|m6_fence::overview_refresh_write
ensure_overview_page|crates/wenlan-core/src/synthesis/overview.rs:70|create_page @ overview.rs:92|maintenance,refinery|called by refresh_overview_page @ overview.rs:111|route|m6_fence::overview_ensure_create
run_maintenance_stage_slice|crates/wenlan-core/src/maintenance.rs:240|refresh_page @ maintenance.rs:479; clear_page_staleness @ :490; refresh_overview_page @ :526|maintenance|scheduler fire_maintenance_stage_safe @ wenlan-server/src/scheduler.rs:1730|route|m6_fence::maintenance_slice_refresh
run_maintenance_tick|crates/wenlan-core/src/maintenance.rs:128|refresh_page @ maintenance.rs:187; refresh_overview_page @ :203|maintenance|no production caller on this branch; dormant public entry|route|m6_fence::maintenance_tick_refresh
run_citation_backfill_with_page_limit|crates/wenlan-core/src/citations.rs:419|set_page_citations_with_changelog_at_version @ citations.rs:454|citation_backfill|scheduler ambient via run_citation_backfill_slice @ wenlan-server/src/scheduler/ambient.rs:507|route|m6_fence::citation_backfill_update
record_annotate_failure|crates/wenlan-core/src/citations.rs:358|set_page_citations_with_changelog_at_version @ citations.rs:374|citation_backfill|same lane as run_citation_backfill_with_page_limit|route|m6_fence::citation_giveup_update
write_document_source_page|crates/wenlan-core/src/document_enrichment.rs:669|page_write PageWrite::DocumentSource @ document_enrichment.rs:679|doc-enrich|doc-enrichment queue worker @ document_enrichment.rs:397,466 from scheduler ambient @ wenlan-server/src/scheduler/ambient.rs:594|pass_through|m6_fence::document_source_write
sync_one_file|crates/wenlan-core/src/sources/page_watcher.rs:126|update_page @ page_watcher.rs:222|fs_edit|vault watcher sync_filesystem_edits @ page_watcher.rs:56 from wenlan-server/src/scheduler.rs:1239|pass_through|m6_fence::vault_edit_update
insert_entity_shadow_page|crates/wenlan-core/src/db.rs:9709|INSERT INTO pages @ db.rs:9716|entity|store_entity @ db.rs:29287 and migration 92 @ db.rs:10020|pass_through|m6_fence::entity_shadow_create
update_entity_shadow_page|crates/wenlan-core/src/db.rs:9752|UPDATE pages @ db.rs:9758|entity|store_entity @ db.rs:29293, add_entity_alias @ :29613, refresh_entity_embedding @ :29849, merge_entities @ :30358, confirm_entity @ :32645|pass_through|m6_fence::entity_shadow_sync
```

Twenty-one rows: **seventeen `route`** (`detect_page_candidates` through `record_annotate_failure`) and **four `pass_through`** (`write_document_source_page`, `sync_one_file`, and the two entity-shadow mutators).

### Notes on individual rows

**`run_maintenance_tick` has no production caller on this branch.** Its own doc comment (`maintenance.rs:216`) calls it "the explicit full-tick API … available to foreground callers", but every `ast-grep` hit outside `#[cfg(test)]` is gone — the scheduler uses `run_maintenance_stage_slice` instead (`wenlan-server/src/scheduler.rs:1730`). It stays in the manifest because it is a `pub` entry that writes pages; leaving it out would let a future consumer bypass the fence without failing G9.

**`refresh_page` and `refresh_page_with_prompt` are the chokepoint** for four of the fenced lanes (redistill batch, redistill slice, maintenance slice/tick, overview). They earn their own rows so a per-caller mutation test can prove that fencing the lane entry is not sufficient on its own.

**`insert_entity_shadow_page` / `update_entity_shadow_page` write `pages` rows but not pages.** They mirror an `entities` row into a `kind='entity'`, empty-`content` shadow. The function's own doc comment states the contract: these rows "stay excluded from retrieval/context, export, and every page mutation" (`db.rs:9707-9708`). They are `pass_through` for exactly that reason — the fence must observe them so G9 cannot be blindsided by the KG lane, and must never let them contribute a genesis root, a relevance candidate, or a coverage claim.

**`sync_one_file` carries human prose.** Its writer identity is `fs_edit`, which `Writer::classify` (`page_update.rs:50`) puts in `Writer::Human`. The scheduler poll is automatic; the content is a person editing markdown in the vault. `pass_through`, and D10's "human-owned page prose stays byte-identical" applies to whatever it writes.

**`write_document_source_page` is the source-page mirror.** It writes the single `creation_kind='source'` page for a document, keyed by a deterministic hash of source id + file path (`document_enrichment.rs:760`). This is the "existing explicit … source creation" D12 leaves legal; the human action is adding the folder, and the per-document write is a 1:1 projection with no synthesis. `pass_through`.

## 5. Excluded — production, explicit human or API authoring

These reach the same write seam but are the direct result of an explicit action, so D12 leaves them unfenced. Each is listed so the exclusion is a decision on the record rather than an omission.

| Symbol | Location | Surface | Why excluded |
|---|---|---|---|
| `handle_create_page` | `crates/wenlan-server/src/page_routes.rs:262` | `POST /api/pages` | Caller supplies title and prose; `create_page_with_tuning` at `:291`. |
| `handle_update_page` | `crates/wenlan-server/src/page_routes.rs:620` | `POST /api/memory/{id}/update-page` | The manual editor; `update_page_preserving_sources` at `:643`, writer `manual_edit`. |
| `handle_refresh_page` | `crates/wenlan-server/src/page_routes.rs:695` | `PUT /api/pages/{id}` | Agent-requested refresh of one named page; `update_page` at `:822`, `update_page_summary` at `:847`, `clear_page_staleness` at `:849`. Writer `agent_refresh`. Distinct from the *automatic* refresh callers D12 fences. |
| `handle_archive_page` | `crates/wenlan-server/src/page_routes.rs:178` | `POST /api/pages/{id}/archive` | `archive_page` at `:186`. |
| `handle_delete_page` | `crates/wenlan-server/src/page_routes.rs:198` | `DELETE /api/pages/{id}` | `delete_page` at `:214`. |
| `handle_distill` | `crates/wenlan-server/src/routes.rs:519` | `POST /api/distill` | On-demand distillation; `clear_user_edited` at `:609`, `distill_pages_scoped` at `:648`, `resolve_orphan_page_links` at `:796`. The *scheduled* entry into the same synthesis code is fenced as `distill_pages_scoped_gated`. |
| `handle_redistill` | `crates/wenlan-server/src/routes.rs:853` | `POST /api/distill/{page_id}` | Human names the page; `clear_user_edited` at `:881`. |
| `handle_accept_revision` | `crates/wenlan-server/src/memory_routes.rs:1220` | revision-card accept | `accept_pending_revision_with_knowledge_path` at `:1231`, writer `revision_accept`. The card is staged automatically, but publication requires the accept. |
| `handle_accept_refinement` | `crates/wenlan-server/src/refinery_routes.rs:124` | refinement-card accept | `apply_refinement_with_decision` at `:138`, which reaches `accept_page_merge` (`refinement_queue.rs:201`), `archive_page` (`:219`), and `apply_cross_space_discovery` → `PageWrite::Create` (`:304`). The only production entry into `apply_refinement_with_decision`; `apply_refinement` (`refinement_queue.rs:117`) has no production caller. Card *emission* is automatic and writes no page rows. |
| `apply_repair_with_pages` | `crates/wenlan-core/src/repair.rs:2060` | lint-repair apply | Behind an approved-manifest digest and a repair fence (`crates/wenlan-server/src/repair_routes.rs:230`). Reaches `apply_deterministic_repair_cas` (`repair.rs:2221`), `regenerate_page_projection_cas` (`:2233`), `apply_quarantine_stale_page_projection` (`:2273`), `apply_rename_page_title` (`:2372`). |
| `cmd_backfill::run` | `crates/wenlan-server/src/cmd_backfill.rs:19` | hidden CLI subcommand | Operator-invoked; `delete_page` at `:95`. |
| Page-draft mutators | `crates/wenlan-core/src/db/page_drafts.rs:204,355,476` | direct draft editor | `create_page_draft_with_id_in_registered_space`, `update_page_draft_in_registered_space`, `delete_page_draft`. Human-authored drafts. Note: **no production caller resolves to any of them on this branch** — the surface is dormant. |

**Adjacent but not a page writer:** `resolve_orphan_page_links` (`crates/wenlan-core/src/db.rs:44108`) runs automatically in the refinery emergence phase (`refinery/mod.rs:1032`) and in `handle_distill` (`routes.rs:796`). It mutates `page_links`, never `pages`. It matters to M6 because D2 signal 2 (orphan wikilink) reads the same table, but it is not a fence target.

## 6. Excluded — test and eval only

Verified by `#[cfg(test)]` boundary walk plus reading each site. Counts, not line lists.

| Location | Sites | Note |
|---|---|---|
| `crates/wenlan-core/src/post_write/post_write_tests.rs` | 19 `page_write`, 14 `create_page`, 42 `update_page`, plus wrapper calls | the seam's own suite |
| `crates/wenlan-core/src/db/main_tests.rs` | ~40 across `archive_page`, `delete_page`, `link_page_source`, `set_page_stale`, `clear_page_staleness` | |
| `crates/wenlan-core/tests/` (`provenance_p2`, `provenance_p3`, `page_citations_e2e`, `distill_redesign_e2e`) | ~20 | hermetic e2e |
| `crates/wenlan-server/tests/` and the `#[cfg(test)]` modules inside `wenlan-server/src` | ~12 | route and scheduler suites |
| `crates/wenlan-core/src/eval/shared.rs:2517` (`store_batch_distilled_page`) | 1 | **eval only**; `create_page_with_tuning`. Also `eval/lifecycle.rs:790`, `eval/runner.rs:190`. |
| `crates/wenlan-core/src/document_enrichment.rs:702` (`write_source_page`) | 2 `page_write` | **`#[cfg(test)]` on the function itself** (`:701`). Sole constructor of `PageWrite::ReplaceSource`. |
| `crates/wenlan-core/src/synthesis/overview.rs:230` (`create_research_page`) | 1 `create_page` | inside `mod tests` (opens `:136`) |
| `crates/wenlan-core/src/page_map_improve.rs:469`, `crates/wenlan-server/src/page_map_routes.rs:597,634`, `crates/wenlan-server/src/routes.rs:1806,2031`, `crates/wenlan-server/src/memory_routes.rs:2612`, `crates/wenlan-server/src/page_routes.rs:1097`, `crates/wenlan-core/src/post_ingest.rs:2553,2631,2704` | 10 | test seed helpers past each file's `#[cfg(test)]` boundary |

Two variants have **no production constructor at all**: `PageWrite::ReplaceSource` (only the `#[cfg(test)]` `write_source_page`) and `PageWrite::UpdatePreservingSources` (only its own wrapper, reached from the explicit `handle_update_page`). `create_page_with_floor` (`page_dispatch.rs:220`) has zero production callers.

## 7. New versus the frozen contract's five

D12 named five targets. Against this tree:

| Frozen target | Status |
|---|---|
| `detect_page_candidates` | confirmed, re-anchored `synthesis/detect.rs:18` |
| automatic `distill_pages_scoped_gated` | confirmed, re-anchored `synthesis/distill.rs:993` |
| `run_page_growth_slice` | confirmed, re-anchored `post_ingest.rs:128` |
| current refresh callers | resolved into six named symbols: `refresh_page`, `refresh_page_with_prompt`, `re_distill_stale_pages`, `run_redistill_page_slice`, `run_maintenance_stage_slice`, `run_maintenance_tick` |
| global `refresh_overview_page` | confirmed, re-anchored `synthesis/overview.rs:104`; plus `ensure_overview_page` and `maybe_refresh_overview_page` |

**Twelve rows are new relative to the frozen five** — the population D12 delegated to "any production caller discovered by LSP/ast-grep manifests":

1. `distill_one_cluster_with_tuning` — the function that actually holds the `PageWrite::Attach` and `PageWrite::Create` calls under the gated distiller.
2. `grow_page` — a second, separate page-growth writer reached from post-ingest enrichment rather than from the ambient slice.
3. `enqueue_changed_pages` — marks pages stale automatically; a refresh *producer* rather than a refresh caller, and invisible to a search for refresh symbols.
4. `re_distill_stale_pages` and 5. `run_redistill_page_slice` — the two distinct refinery redistill entries.
6. `run_maintenance_stage_slice` and 7. `run_maintenance_tick` — the maintenance lane, which the frozen list did not name at all.
8. `refresh_page` and 9. `refresh_page_with_prompt` — the shared refresh chokepoint.
10. `ensure_overview_page` — creates the reserved overview row; the only automatic `create_page` caller in the tree.
11. `run_citation_backfill_with_page_limit` and 12. `record_annotate_failure` — **the citation-backfill lane, entirely absent from the frozen contract.** It runs as a scheduler ambient job (`wenlan-server/src/scheduler/ambient.rs:507`) and writes the `citations` column and a changelog entry through `set_page_citations_with_changelog_at_version`. It does not touch prose, which is presumably why it was overlooked, but it is an automatic page-row update and D10's dependency-invalidation reasoning has to account for it.

Plus the four `pass_through` rows (`write_document_source_page`, `sync_one_file`, and the two entity-shadow mutators), which the frozen contract's exclusion sentence covers in principle but names nowhere. Twelve new `route` rows plus four `pass_through` rows against five frozen targets is how the twenty-one-row total is reached.

### Corrections to the post-M5 revalidation hints

The 2026-07-31 revalidation note gave a starting remap for four symbols. Three verify clean; **one caller attribution in it is wrong and should not be carried forward.**

> **Correction of record.** The note lists `reconcile.rs:866` as a production caller of `run_page_growth_slice`. It is **test code.** `crates/wenlan-core/src/reconcile.rs` opens `#[cfg(test)]` at line 590, and the call at `:866` sits inside that module. `run_page_growth_slice` has exactly one production caller on this branch: `crates/wenlan-server/src/scheduler/ambient.rs:449`, the `AmbientJob::PageGrowth` arm.

Symbol by symbol:

- `detect_page_candidates → synthesis/detect.rs:18` — **correct**; production caller `refinery/mod.rs:982` correct.
- `distill_pages_scoped_gated → synthesis/distill.rs:993` — **correct**; caller `refinery/mod.rs:1014` correct. The second listed caller `distill.rs:983` is inside `distill_pages_scoped` (`:975`), whose only production entry is the explicit `handle_distill` route.
- `run_page_growth_slice → post_ingest.rs:128` — **correct**; caller `wenlan-server/src/scheduler/ambient.rs:449` correct. Its second listed caller is the wrong attribution called out above.
- `refresh_overview_page → synthesis/overview.rs:104` — **correct**; wrapper `maybe_refresh_overview_page` at `refinery/mod.rs:1727` (the note's `:1737` is the call inside it, not the definition); `maintenance.rs:203,526` correct.

## 8. Completeness argument

The sweep does not rest on having found every reference. It rests on the page table having a small, enumerable set of doors.

**Step 1 — every `pages`-row mutation is inside a known function.** A case-insensitive scan for `INSERT INTO pages`, `INSERT OR … INTO pages`, `UPDATE pages`, and `DELETE FROM pages` across `crates/wenlan-core/src` and `crates/wenlan-server/src` returns 105 statements in 15 files. Folding each into its enclosing `fn` yields 43 functions in `db.rs` and 9 outside it. Of the 52:

- 6 are migration bodies (`run_migrations`, `migrate_80_page_scope_fold`, `migrate_89_page_kind_fold`, `migrate_93_page_aliases`, `fold_entity_type`, and migration 92's backfill loop) — schema evolution, not a writer lane;
- 7 are inside `#[cfg(test)]` (`synthesis/distill.rs:2356`, `synthesis/detect.rs:204`, `maintenance.rs:1542`, `eval/seed_contract.rs:498`, and three in test modules of `db/page_drafts.rs`);
- 3 are the page-draft mutators, dormant (section 5);
- 4 are repair CAS bodies behind the approved-manifest fence (section 5);
- 2 are the entity-shadow mirror pair (manifested `pass_through`);
- 8 are cascade/lifecycle bodies invoked only from memory, space, or entity deletion and rebinding (`delete_space`, `reassign_memories_space`, `delete_by_source_id_in_transaction`, `rebind_source_id_inner`, `rebind_source_page_in_transaction`, `replace_memory_page_dependency_in_transaction`, `merge_entities`, `delete_entity`) — they clear or repoint page references when their owning row goes away; they never create page content and cannot mint a page;
- the remaining 22 are the `MemoryDB` page mutators reached either through `page_write`'s impl functions or directly by the callers enumerated in sections 4 and 5.

**Step 2 — every production caller of those 22 is classified.** Each was queried with three `ast-grep` patterns (bare call, path-qualified call, method call), and every non-test hit was read and placed in section 4 or section 5.

**Step 3 — an independent cross-check on writer identity.** Every seam write persists an `edited_by` literal. Scanning production sources for the thirteen literals `Writer::classify` and `PipelineStage` know about yields exactly thirteen in use: `page_growth` (21 sites), `distill` (13), `fs_edit` (9), `re_distill` (7), `manual_edit` (6), `revision_accept` (5), `refinery` (4), `refinery_merge` (3), `doc-enrich` (3), `agent_refresh` (3), `overview_sync` (2), `maintenance` (2), `citation_backfill` (2). Each maps onto a row already in section 4 or section 5 — `page_growth` to the two growth rows, `distill` to the two emergence rows, `re_distill` to the four refresh rows, `overview_sync`/`refinery`/`maintenance` to the overview and maintenance rows, `citation_backfill` to the two citation rows, `doc-enrich` and `fs_edit` to the `pass_through` rows, and `manual_edit`/`agent_refresh`/`revision_accept`/`refinery_merge` to section 5. **No literal is unaccounted for**, which is a second, structurally independent way of reaching the same population: the identity scan does not use the caller graph, and the caller graph does not use the identities.

**What could still be missed, and why it is unlikely.** Trait-object or function-pointer dispatch would be invisible to both the AST sweep and the identity scan; nothing in the code read routes page writes that way. A caller that mutates `pages` through SQL text assembled at runtime — splitting the keyword from the table name across a `format!` — would evade the statement scan; every statement found is a literal string. A new writer added after this commit is exactly what G9 is for.

## 9. Relationship to G9

This document is the **seed**, not the gate.

G9 (`m6_old_writers_are_fenced`) requires a tracked machine-readable writer manifest naming every allowed production `PageWrite`, create, attach, update, refresh, and overview caller plus its fence adapter, and **a structural CI test that enumerates those production symbols and call sites and rejects any unlisted wrapper or consumer**. The frozen contract is explicit that LSP and ast-grep generate review evidence but are not the gate by themselves. The block in section 4 is that review evidence in consumable form; the executable test is owed by the M6 daemon program and is the thing that actually fails a build.

The gate that eventually consumes this block must, per G9: fail when a manifest row is deleted, fail when an unlisted wrapper is added, and fail when one fence is bypassed. Per-caller mutation tests must additionally prove that no listed path bypasses M6/M5 finalization after cutover.

The natural home is a `#[cfg(test)]` lib test alongside the existing drift-defense teeth in `crates/wenlan-core/src/drift_guard.rs`, which are already picked up by the `cargo test --workspace --lib` that CI and pre-push run. That placement is a suggestion for PR-D, not a decision this artifact may make.

Until then, STOP condition 2 of the frozen contract stands: a new automatic page mutation existing outside the writer manifest is a stop-and-surface, not a merge-and-note.
