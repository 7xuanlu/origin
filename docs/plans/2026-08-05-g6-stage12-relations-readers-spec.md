# G6 Stage 1.2 — migrate `relations` readers onto `edges`

Parent program: `docs/plans/2026-08-04-g6-retirement-program.md` (Stage 1, store 2).
Scout report: `docs/superpowers/g6-stage1-relations-scout.md` (gitignored working doc;
its findings are restated here so this spec is self-contained).
Prerequisite: the semantic-payload PR (#486) + review fixes (#487) — `relates` edges
now carry `semantic_type = relation_type` and payload keys `confidence`, `explanation`,
`source_agent` (semantic, refresh-on-reassert) and `span`, `model_version`,
`prompt_version`, `source_memory_id` (provenance, write-once). Migration 115 backfilled
both groups from live legacy rows. Verified on the live DB: 0 of 304 active relates
edges missing `semantic_type`.

## Contract (from the program doc)

- Readers hard-cut to `edges` — no `reader_uses_edges` gate for these consumers; the
  cutover-lever tables retire wholesale in Stage 2.
- Writers keep dual-writing (Stage 2 flips them). The parity sweep derivation for
  `relations` **stays** in `reconcile_edges_parity` — this PR does NOT remove the
  store's last legacy reader (two deliberate carryovers below).
- One PR: reader redirects + migration 116 + behavior-equivalence tests.

## The two traps this spec exists to encode

**Trap 1 — `created_at` recency scramble.** `list_recent_relations(_scoped)` orders and
filters on `relations.created_at`; `get_entity_detail(_scoped)` orders its relations
list the same way. Edges backfilled by migration 81 carry migration-day
`edges.created_at`, so ordering by the edge column would scramble the recency feed and
misdate every pre-m81 relation. Additionally the relations upsert
(`ON CONFLICT(from_entity,to_entity,relation_type)`) keeps the ORIGINAL `created_at`
on re-assert, while a re-asserted edge may be a fresh row. `edges.created_at` is
therefore NOT a substitute for `relations.created_at`.

Fix: a new **semantic payload key `asserted_at`** (integer, epoch seconds, mirrors the
stored `relations.created_at`). Migrated readers order/filter on
`COALESCE(json_extract(e.payload,'$.asserted_at'), e.created_at)`.

**Trap 2 — wire-visible `id` swap.** `RelationWithEntity.id` and `RecentRelation.id`
currently carry the `relations` row UUID. After migration they carry the
content-addressed `edge_id`. Verified safe: no route dereferences a relation id back
into a lookup or delete (`/api/memory/relations` is POST-create only; relation ids are
display-only on the wire). The change is shape-compatible (still a TEXT id). Flag it
in the PR description as a wire-visible value change; do not add a compatibility shim.

## Migration 116

Idempotent, follows the m115 pattern. Two backfills over ACTIVE relates edges
(`edge_type='relates' AND valid_until IS NULL`), joining structurally to the live
legacy row — the join key `(src_id, dst_id, semantic_type)` = `(from_entity,
to_entity, relation_type)` is UNIQUE in `relations` (the upsert conflict target), so
no re-derivation of edge ids is needed:

1. **`asserted_at`** — always set:
   `payload = json_set(COALESCE(payload,'{}'), '$.asserted_at', r.created_at)`.
2. **`source_memory_id`** — fill-if-absent only: set from `r.source_memory_id` when
   the payload key is currently missing/null AND the legacy row has a value. This
   converges edges born before #486 whose legacy row gained `source_memory_id` later
   (the legacy upsert COALESCEs it in). Never overwrite an existing payload value.

Edges with no matching live relations row (retired legacy row, cross-store residue)
are left untouched — the COALESCE fallback in readers covers them.

## Writer-side changes (dual-write stays; patches gain one key)

- `relates_semantic_patch(...)` gains an `asserted_at: i64` parameter, emitted into
  the semantic patch. Every caller threads the STORED relations row's `created_at`
  (the same SELECT-after-write mirror that already sources `confidence`/
  `explanation`/`source_agent` — extend that SELECT with `created_at`). Refresh-on-
  reassert is harmless: the stored `created_at` is immutable under the upsert.
- `source_memory_id` moves from strict write-once-at-birth to **fill-if-absent**,
  mirroring the legacy COALESCE semantics: on re-assert, the patch may supply it and
  the merge must set it ONLY when the stored payload lacks it. (It is still written
  at most once.) Update the payload-contract doc comment on
  `dual_write_edge_with_payload` accordingly, and list `asserted_at` among the
  semantic keys.

## Reader migration table

All migrated readers filter `edge_type='relates' AND valid_until IS NULL`, use
`src_id`/`dst_id` (kind `entity`) as endpoints, `semantic_type` as `relation_type`,
and `COALESCE(json_extract(payload,'$.asserted_at'), created_at)` wherever legacy
read `relations.created_at`.

| # | Reader | Location | Notes |
|---|---|---|---|
| 1 | `expand_anchor_entities_khop` | db.rs ~25500 | topology-only; clean swap |
| 2 | `expand_entities_khop_scoped` | db.rs ~26056 | topology-only; clean swap |
| 3 | `aggregate_counts` | lint/kg/query/aggregate.rs:54 | count active relates edges |
| 4 | `entity_scope_clause` subquery | lint/semantic_candidates.rs:767 | swap subquery source |
| 5 | `get_entity_detail` | db.rs ~33020 | UNION ALL both directions over edges; `id`=edge_id, `relation_type`=semantic_type, `source_agent`=payload `$.source_agent`, `created_at`=COALESCE(asserted_at, created_at); ORDER BY that value DESC |
| 6 | `get_entity_detail_scoped` | db/scoped_entities.rs:84 | same as #5 + scope filter; keep the entity-page name-hydration overlay logic byte-equivalent (it overlays names on the selected rows; only the selection source changes) |
| 7 | `list_recent_relations` | db.rs ~36859 | recency feed over edges; keep the non-empty-name JOIN filters against `entities`; `since_ms` filter and ORDER BY on the COALESCE expression |
| 8 | `list_recent_relations_scoped` | db/scoped_entities.rs:291 | Global still delegates to #7; scope filter on the joined `entities` rows as today; hydration overlay preserved unchanged |
| 9 | `relation_vocabulary` | lint/deep.rs:242 | distinct `semantic_type` over active relates |
| 10 | `distinct_relation_types_for_vocabulary_heal` | db/kg_quality_vocabulary.rs:7 | same; heal writers still write `relations` (dual-write keeps types equal under parity) |
| 11 | `relation_integrity` | lint/kg/query.rs:83 | dangling-endpoint check: src_id/dst_id existence in `entities` |

**Conditional (investigate before migrating — bounded discretion):**

| # | Reader | Location | Condition |
|---|---|---|---|
| 12 | `load_record_inventory` | repair_plan/semantic.rs:164 | Migrate ONLY if the captured record identity is never dereferenced back into `relations` by the repair apply/CAS path (`apply_deterministic_repair_cas` and friends). If the apply path verifies or mutates relations rows BY THE CAPTURED ID, keep this reader on `relations` and say so in the PR description. |
| 13 | `load_relations` | lint/semantic_candidates.rs:829 | Same condition — trace where its ids flow. |

**Deliberate carryovers (stay on `relations`, dated notes in code):**

- `count_stale_relation_sources` — payload `source_memory_id` converges via m116 +
  fill-if-absent, but historical completeness isn't proven yet; migrates with the
  final relations-exit PR.
- `promote_edges_grounded` legacy-existence validation (db.rs ~14614) — the M5
  grounding validator deliberately checks the legacy row as an independent witness;
  redesign belongs to Stage 2, not a reader swap.

These two (plus the conditional outcomes) mean the Stage 1 exit criterion for
`relations` ("zero non-test readers outside migrations and the parity sweep") is NOT
met by this PR; the program doc's per-store status line should record what remains.

## Tests (RED-control discipline: prove each new test fails on the unfixed code)

1. **m116 backfill** — seed relations via `create_relation` pre-m116 (or simulate a
   pre-asserted_at edge by clearing the key), run migrations, assert `asserted_at`
   equals the relations row's `created_at` and fill-if-absent `source_memory_id`
   converged. Include an idempotency rerun (m55/m81-rerun test idiom in
   db/main_tests.rs).
2. **Recency-order equivalence** — seed ≥3 relations with distinct forced
   `created_at` values (UPDATE relations + re-run m116 or patch payloads), plus one
   edge simulating m81 backfill (edges.created_at newer than asserted_at); assert
   `list_recent_relations` order matches `relations.created_at` DESC, not
   edges.created_at; assert `since_ms` filters on the same value.
3. **`get_entity_detail` equivalence** — field-by-field against a seeded fixture:
   directions, names, relation_type, source_agent, created_at values, order. Assert
   `id` is now the active edge's `edge_id`.
4. **Scoped variants** — same assertions under a Space scope; hydration overlay
   still applies when the entity-pages gate is on (existing scoped tests are the
   template).
5. **Writer mirror** — after `create_relation` then a weaker re-assert, the edge
   payload's `asserted_at` still equals the ORIGINAL row `created_at` and
   `confidence` kept the higher value (extends the existing stored-row-mirror test).
6. **Parity stays 0** — existing parity fixture tests keep passing (the relations
   derivation is untouched).

## Gates

`cargo fmt --check`; `cargo clippy -p wenlan-core -- -D warnings` (and `--tests`);
focused test modules first, then the full `-p wenlan-core` suite (background it —
~22 min, exceeds the 10-min foreground cap); `-p wenlan-server` suite (route contract
tests cover `/api/knowledge/recent-relations` and entity detail).

## Out of scope

Writers stay dual-write; parity derivation stays; `relation_id` uses inside grounding
promotion (db.rs ~14966) untouched; no schema DDL beyond payload backfill; entities
readers (Stage 1.5) untouched. Coordination: Stage 1.1 (page_links readers) is in
flight on branch `g6/stage1-readers` touching db/scoped_pages.rs + db.rs ~46850 —
disjoint from this PR's regions; rebase whichever lands second.
