# M3 PR-1 caller inventory (stage: caller inventory, first deliverable)

Status: committed **before** the `kind`-column / mapping-table / dual-write code,
per the M3 goal prompt ("A committed inventory of every entity consumer... its
counts go in the PR description. A wire surface the inventory can't classify is
a stop condition"). Mirrors the M2 edge-assignment-matrix precedent
(`docs/plans/2026-07-21-m2-edge-assignment-matrix.md`).

Verified against `~/Repos/wenlan/.claude/worktrees/kg-m3-entity-pages-dualwrite`,
branched from main `3246e180`, schema 82.

## The one governing rule (D4)

> Internal core readers flip per-consumer; the wire stays legacy via adapters.
> The ~200 internal `wenlan-core` entity readers flip to page-backed reads
> behind a per-consumer reversible cutover (same pattern as M2's
> `set_reader_cutover` / `reader_uses_edges`, gated on a clean current parity
> watermark). The wire/MCP `entity_id` shape is frozen and preserved by
> adapters, never changed.

Every consumer below gets exactly one of three labels:

- **FLIP** — a `wenlan-core`-internal reader; gets its own reversible
  per-consumer cutover flag in a later PR-1 stage (mapping table + shadow pages
  land first; the actual flips are gated on those).
- **FREEZE** — a wire/MCP-facing surface (HTTP route, MCP tool, `wenlan-types`
  struct, CLI display). Shape never changes; an adapter on the flipped internal
  read translates page-backed data back into this frozen contract.
- **WRITE** — a write path. Not a flip target. PR-1 converts these to
  dual-write (`entities` row + `kind=entity` shadow page in one transaction),
  handled by the separate "collapse the two cascades" and "dual-write shadow
  pages" stages below, not by this inventory's flip/freeze axis.

## Write paths (WRITE — handled by later PR-1 stages, not flip/freeze)

| Cascade | file:line | Terminal sink |
|---|---|---|
| `post_write::create_entity` | `crates/wenlan-core/src/post_write.rs` (agent/MCP/HTTP path) | `MemoryDB::store_entity` (`db.rs:18972`) |
| `importer::resolve_entity_bulk` | `crates/wenlan-core/src/importer.rs` (bulk-import path) | `MemoryDB::store_entity` (`db.rs:18972`) |

`MemoryDB::create_entity` (`db.rs:18944`) — a bare low-level insert (no
resolution cascade, no embedding) — has **zero production callers**. Every
call site found (`db.rs` test module, `post_ingest.rs:1494,1567` inside
`mod tests` at `post_ingest.rs:780`, `lint/snapshot_tests.rs`) is
`#[cfg(test)]`. It is a test-fixture-only primitive, not a third production
entity-creation path, and is **out of scope** for the canonical entity-upsert
collapse.

## FLIP set — `wenlan-core` internal readers/writers touching `entities`

Grouped by subsystem. Function name is the enclosing `fn`; line is the `fn`
signature (confirmed by read, not grep-guessed). Migration-only code (one-shot
backfills: migrations 24/41/50/81, `backfill_edges_from_relations`,
`audit_legacy_cross_space_links`, `run_migration_55_pass_b`) is excluded — it
runs once and never gets an ongoing reader-cutover flag.

### Core CRUD / graph (`db.rs`)

| Function | file:line | What it does |
|---|---|---|
| `entity_canonicals` | `db.rs:431` | distinct canonical names, feeds dedup |
| `fold_entity_type` | `db.rs:455` | UPDATE entity_type fold; called live from `kg_quality.rs:366` (not just migration) |
| `expand_anchor_entities_khop` | `db.rs:14864` | k-hop graph read, JOIN entities |
| `entity_degrees` | `db.rs:15271` | degree-count read (no external caller found — possibly dead, included defensively) |
| `get_memories_for_entities` | `db.rs:15327` | resolves linked memories for entity-id list |
| `expand_entities_khop` / `expand_entities_khop_scoped` | `db.rs:15410` / `15420` | k-hop expansion, JOIN entities |
| `search_entities_by_vector` | `db.rs:15909` | vector-similarity read |
| `get_observations_for_entities` | `db.rs:16000` | batch observations, JOIN entities |
| `store_entity_minhash_bands` / `query_entities_by_band` / `delete_entity_minhash_bands` | `db.rs:19026` / `19073` / `19111` | minhash dedup index write/read/delete |
| `get_entity_name_type` | `db.rs:19126` | SELECT name,type by id; used by `page_map_routes.rs:71` for ref liveness |
| `index_entity_minhash_if_eligible` | `db.rs:19191` | orchestrates minhash indexing |
| `resolve_entity_by_alias` | `db.rs:19205` | alias-table read |
| `add_entity_alias` | `db.rs:19226` | alias write |
| `resolve_entity_type` | `db.rs:19337` | canonical-type read |
| `increment_entity_type_count` | `db.rs:19386` | vocab counter write |
| `search_entities_by_name` | `db.rs:19398` | SELECT by name |
| `refresh_entity_embedding` | `db.rs:19432` | UPDATE embedding |
| `merge_entities` | `db.rs:19496` | dedup merge (repoints relations/observations/memories/page owner); called from `synthesis/refinement_queue.rs:156` |
| `entity_exists` | `db.rs:20047` | existence check |
| `list_entities` | `db.rs:20063` | unscoped list; only caller is `scoped_entities.rs:18` |
| `get_entity_detail` | `db.rs:20117` | full detail incl. relations JOIN; only caller `scoped_entities.rs:58` + `post_ingest.rs:414` |
| `delete_entity` | `db.rs:20248` | DELETE |
| `resolve_entity_by_name` | `db.rs:20321` | exact/substring name lookup |
| `update_memory_entity_id` | `db.rs:20361` | writes `memories.entity_id` FK (not the `entities` table itself) |
| `link_memory_entities` | `db.rs:20380` | writes `memory_entities` junction |
| `memory_entities_degree_stats` / `top_memory_entity_hubs` | `db.rs:20407` / `20491` | hub stats; no external callers found outside db.rs |
| `get_memory_entity_id` | `db.rs:20614` | reads a memory's entity_id |
| `find_memories_without_entities` | `db.rs:20701` | no external callers found |
| `confirm_entity` | `db.rs:20987` | UPDATE confirmed flag |
| `count_memory_entity_links` | `db.rs:22810` | called from `eval/shared.rs:2064` |
| `count_entities` | `db.rs:22900` | COUNT; called from `onboarding.rs:102` |
| `list_recent_relations` | `db.rs:22935` | unscoped, superseded at runtime by the scoped variant |
| `get_page_by_entity` | `db.rs:28931` | queries `pages.entity_id` (bridges entities→pages already); internal to `find_matching_page` |
| `find_matching_page_scoped` | `db.rs:29890` | live entity→page dedup/attach path; called from `post_write.rs:2548`, `post_ingest.rs:618`, `synthesis/distill.rs:543,2636`, `synthesis/detect.rs:61` |
| `search_memory_with_cue` | `db.rs:13431` | **hot path**: inline raw `SELECT id,name FROM entities WHERE id IN (...)` bypassing every named method AND `scoped_entities.rs`'s scope filter — see "Noted, not blocking" below |
| `load_summary_buckets` | `db.rs:16178` | JOIN feeding `synthesis/summary.rs:228` recap bucketing |
| `create_relation` | `db.rs:19949` | reads both entities' `space` to validate same-space before writing a relation |
| `query_distillation_staging_pool` | `db.rs:25639` | `LEFT JOIN entities` for distillation clustering |
| `detect_communities` | `db.rs:21019` | reads all entity ids, writes `community_id`; called from `refinery/mod.rs:794` — see "Noted, not blocking" |
| `list_spaces` / `get_space` | `db.rs:10483` / `10521` | `entity_count` subquery, feeds `wenlan-cli` space display |

### Scope-filter layer (`db/scoped_entities.rs`) — the canonical read boundary between routes and `db.rs`

`list_entities_scoped` (`:18`), `get_entity_detail_scoped` (`:58`),
`list_recent_relations_scoped` (`:230`), `list_entity_suggestions_scoped`
(`:317`), `search_entities_by_vector_scoped` (`:464`),
`get_memories_for_entities_scoped` (`:526`).

### KG-quality / dedup (`kg_quality.rs`)

`find_merge_candidates` (`:150`), `surface_minhash_merge_candidates` (`:195`),
`heal_entity_vocabulary` (`:361`), `refresh_stale_entity_embeddings` (`:429`),
`scan_contradictions` (`:536`).

### Lint / KG integrity checks (read-only)

`lint/deep.rs`: `alias_integrity` (`:89`), `observation_duplicates` (`:198`),
`relation_vocabulary` (`:242`). `lint/kg/query.rs`: `entity_integrity` (`:48`),
`observation_integrity` (`:66`), `relation_integrity` (`:83`),
`link_integrity` (`:100`). `lint/kg/query/aggregate.rs`: `entity_partitions`
(`:30`), `aggregate_counts` (`:54`), `advisory_metrics` (`:71`).

### Semantic dedup candidates (`lint/semantic_candidates.rs`)

`load_entities` (`:738`), `entity_scope_clause` (`:757`, SQL-fragment
builder), `load_relations` (`:819`).

### Derived-artifact eligibility (`derived_artifact_state.rs`)

`summary_eligible_predicate` (`:7`) — builds a predicate joining
`entities`/`memories` to gate summary-node eligibility by community size.

### Repair/lint-repair internal implementation

`post_write.rs`: `complete_entity_extraction_cas_inner` (`:326`) — raw
existence guard inside a CAS transaction, `INSERT INTO memory_entities ...
WHERE EXISTS (SELECT 1 FROM entities WHERE id=?2 ...)`.
`repair.rs`: `validate_selected_entities_on_snapshot` (`:5124`),
`validate_selected_entities_on_connection` (`:5153`),
`capture_memory_entity_link_on_connection` (`:5538`),
`capture_complete_entity_extraction_on_snapshot` (`:4630`) /
`_on_connection` (`:4705`). `repair_plan/deterministic.rs`:
`resolve_memory_entity_links` (`:977`). `repair_plan/semantic.rs`:
`load_record_inventory` (`:164`).

**FLIP count: ~58 distinct reader/writer functions** (excluding migration-only
and test-only code). Call-site fan-out (each is invoked from
`post_write.rs`/`importer.rs`/`memory_routes.rs`/`kg/entity_extraction.rs`/
`kg/reweave.rs`/`synthesis/*.rs`/route handlers) is consistent with the goal
prompt's "~200 internal readers" estimate when counted per call site rather
than per function definition.

## FREEZE set — wire/MCP surfaces carrying `entity_id` (shape never changes)

### HTTP routes (`crates/wenlan-server/src/`)

| Route | Handler | file:line |
|---|---|---|
| `POST /api/memory/entities` | `handle_create_entity` | `memory_routes.rs:1152` |
| `POST /api/memory/relations` | `handle_create_relation` | router `router.rs:134-136` |
| `POST /api/memory/link-entity` | `handle_link_entity` | `memory_routes.rs:1215` |
| `POST /api/memory/entities/list` | `handle_list_entities` | `memory_routes.rs:1391` |
| `POST /api/memory/entities/search` | `handle_search_entities` | `memory_routes.rs:1442` |
| `GET /api/memory/entities/{entity_id}` | `handle_get_entity_detail` | `memory_routes.rs:1410` |
| `GET /api/memory/entity-suggestions` | `handle_get_entity_suggestions` | `memory_routes.rs:1638` |
| `PUT /api/memory/entities/{id}/confirm` | `handle_confirm_entity` | `memory_routes.rs:2290` |
| `DELETE /api/memory/entities/{id}/delete` | `handle_delete_entity` | `memory_routes.rs:2306` |
| `POST /api/memory/entities/{entity_id}/observations` | `handle_add_entity_observation` | `memory_routes.rs:2321` |
| `GET /api/knowledge/recent-relations` | `handle_list_recent_relations` | `knowledge_routes.rs:54` |
| `GET /api/pages/{id}/map` | `handle_get_page_map` (via `compute_ref_state`) | `page_map_routes.rs:64-71,160` |
| `POST /api/memory/store` (entity pre-resolve branch) | `handle_store_memory` | `memory_routes.rs:356` |

### MCP tools (`crates/wenlan-mcp/src/tools.rs`)

`create_entity` (`:2862/2872`, impl `:1892`), `create_relation`
(`:2879/2889`), `create_observation` (`:2896/2906`), `confirm_entity`
(`:2913/2923`, impl `:1962`), `update_observation`/`confirm_observation`/
`delete_observation` (`:2930-2985`), `list_entity_suggestions` (`:3249`, impl
`:2380`), `PrepareLintRepairChoiceParam::CompleteEntityExtraction`
(`:552-585`). No MCP tool exposes entity list/search/detail read (see "Noted,
not blocking").

### Wire types (`crates/wenlan-types/src/`)

`Entity`, `EntitySearchResult`, `EntityDetail` (`entities.rs:8-34`),
`Observation.entity_id`, `Relation.from_entity/to_entity`,
`RelationWithEntity.entity_id`, `RecentRelation.from_entity_id/to_entity_id`,
`EntitySuggestion` (`entities.rs:38-93`), `Page.entity_id`
(`pages.rs:15` — the FK this migration repoints), `StoreMemoryRequest.entity_id`
(`requests.rs:29`), `CreateEntityRequest` (`requests.rs:136`),
`LinkEntityRequest.entity_id` (`requests.rs:175-177`), `ConfirmEntityRequest`
(`requests.rs:564`), `AddEntityObservationRequest` (`requests.rs:570`),
`CreateEntityResponse` (`responses.rs:284`), `ListEntitiesResponse`
(`responses.rs:317`), `SearchEntitiesResponse` (`responses.rs:322`),
`ProposalAction::EntityMerge/SuggestEntity` +
`RefinementPayload::EntityMerge/SuggestEntity` (`responses.rs:697-745`),
`RawDocument.entity_id` (`sources.rs:172`), `MemorySearchResult.entity_id`/
`.entity_name` (`memory.rs:41-43,102`), `SpaceInfo.entity_count`
(`memory.rs:356`).

**Repair/lint-repair wire types** (`repair.rs` in `wenlan-types`, confirmed by
direct read — carries `entity_id`/`entity_ids: Vec<String>` as literal wire
fields): `RepairTargetWire::MemoryEntityLink{entity_id}` /
`::MemoryEntityExtraction{entity_ids}` (`:1684-1731`),
`RepairMutationWire::CompleteEntityExtraction{entity_ids}` (`:2338`) /
`::DeleteMemoryEntityLink{entity_id}` (`:2353`),
`RepairChoiceWire::CompleteEntityExtraction{entity_ids}` (`:3871-3894`),
`RepairRollbackPayloadV2Wire::CompleteEntityExtraction{before_entity_ids}`
(`:294-327`). These fall under D4's blanket rule same as every other
`entity_id`-carrying wire type — not an exception, and not a stop condition
(the corresponding internal implementation, `repair.rs`/`repair_plan/*.rs` in
`wenlan-core`, is already listed under FLIP above).

### CLI (`crates/wenlan-cli/src/`)

No dedicated entity subcommand. `client.rs:139-140` and
`commands/search.rs:103-104` leave `entity`/`entity_id` fields `None` in
request builders (pass-through). `commands/space.rs:130,235` displays
`entity_count` (read-only, sourced from `SpaceInfo`).

### Desktop-app / `EventEmitter`

Checked every `.emit(` call in `wenlan-core`
(`db.rs:4374,4675,4782`, `onboarding.rs:41`,
`synthesis/refinement_queue.rs:801`, `chat_import/bulk_ingest.rs:154,175`).
**None carry entity data.** The Tauri app gets entity data exclusively by
polling the HTTP routes above — no event/push channel to freeze or migrate.

**FREEZE count: ~47 distinct wire-facing items** (13 routes + 7 MCP tools/params
+ ~27 wenlan-types structs/fields).

## Noted during inventory, not blocking (no fork, no stop condition)

1. **`search_memory_with_cue` (`db.rs:13431`) bypasses `scoped_entities.rs`'s
   scope filter.** It does its own inline, ungated `SELECT id, name FROM
   entities WHERE id IN (...)` for display-name resolution on quick-search
   result rows, rather than going through any named scoped method. This is a
   pre-existing quirk, not introduced by this migration. Classified FLIP like
   every other reader; when it flips to reading `pages(kind=entity)` its
   current (ungated) behavior should be preserved as-is — fixing the gate is
   out of scope for a migration PR (surgical changes only).
2. **`find_matching_page` (unscoped, `db.rs:29833`) is production-dead** — only
   test callers remain (`tests/provenance_p2.rs:446`, `db.rs:47829`);
   production exclusively uses `find_matching_page_scoped`. Not a flip target
   since nothing in production calls it; left alone (removal is a separate,
   unrequested cleanup).
3. **`detect_communities` (`db.rs:21019`) writes `entities.community_id`; no
   consumer reads it back** in any SELECT list found. This is FLIP-classified
   like everything else in `db.rs`, but PR-1's shadow pages don't need a
   `community_id` column — `entities` stays a write-compatible shadow per D5,
   so `community_id` simply stays on `entities` untouched by PR-1. Whether it
   ever needs a page-side equivalent is a later-PR question, not a PR-1 blocker.
4. **`memory_entities_degree_stats`/`top_memory_entity_hubs`/
   `find_memories_without_entities`** — defined, touch `entities`, but no
   external callers found anywhere in the crate. Possibly dead. Included in the
   FLIP set defensively (they do read the table); not a blocker either way.
5. **MCP read-gap**: HTTP exposes entity list/search/detail; MCP exposes none
   of them. Once entities are `pages(kind=entity)`, `search_pages`/
   `list_pages_recent` (already MCP-exposed) naturally cover this gap — not a
   migration blocker, just a note that the asymmetry may resolve itself as a
   side effect of D4's per-consumer flip.

## Summary counts (for PR body)

- **2** production write cascades (both converge on one terminal sink) — WRITE, handled by later stages, not flip/freeze.
- **~58** distinct `wenlan-core` internal reader/writer functions — FLIP, each gets its own reversible per-consumer cutover flag in a later stage.
- **~47** distinct wire/MCP-facing surfaces (13 HTTP routes, 7 MCP tools/params, ~27 `wenlan-types` structs/fields) — FREEZE, shape never changes, adapters bridge to flipped internal reads.
- **0** wire surfaces left unclassified (the repair/lint-repair wire types were the one candidate requiring investigation; resolved under D4's blanket rule, not an exception).
- **1** confirmed dead low-level primitive (`MemoryDB::create_entity`, test-only) excluded from the canonical entity-upsert collapse.
- **5** items noted for the record, none blocking, none requiring escalation.
