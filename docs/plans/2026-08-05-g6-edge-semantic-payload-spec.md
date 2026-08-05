# G6 edge semantic payload — one source of truth (spec)

Status: authored 2026-08-05 after the user's decision ("one source of truth",
2026-08-05) on the fork surfaced by the Stage 1 scouts. This PR lands FIRST in
Stage 1 — before any reader migration — because every blocked reader in the
three scout reports migrates only once the semantic fields exist on the edge.

Scout evidence base (gitignored working docs, saved verbatim):
`docs/superpowers/g6-stage1-page-links-scout.md` (Stage 1.1),
`docs/superpowers/g6-stage1-relations-scout.md` (Stage 1.2),
`docs/superpowers/g6-stage1-page-sources-evidence-scout.md` (Stage 1.3).

## Problem

The `edges` schema as wired cannot answer the questions its readers ask:

| Store | Missing on the edge | Blocked readers |
|---|---|---|
| `relations` | `relation_type` (hash-input-only discriminator, never stored), `confidence`, `explanation`, `source_agent`; `source_memory_id` conditional + stale after rebind | 9 of 13, incl. product routes `get_entity_detail`, `list_recent_relations`, most lint/repair tooling |
| `page_links` | `label` display text (only `label_key` hashes into the id); orphan rows (`target_page_id IS NULL`) derive no edge at all | both product-route readers behind `GET /api/pages/{id}/links`, all orphan-feed readers |
| `page_sources` + `page_evidence` | `link_reason`, `linked_at`, `title`; 4-way `source_kind` collapses to 2-way `dst_kind` | product route `GET /api/pages/{id}/sources`; provenance lints need the 4-way kind |

## Design

### 1. Columns vs payload JSON

Semantic fields go into the existing write-once `payload` JSON **except**
`relation_type`, which becomes a real nullable TEXT column (`semantic_type`)
on `edges`:

- `relation_type` is filtered/grouped on (vocabulary heal, lint checks,
  entity-detail grouping) — a JSON extract in every hot reader is the wrong
  trade, and it is already a structural component of the edge id.
- The rest (`confidence`, `explanation`, `source_agent`, `label`,
  `link_reason`, `linked_at`, 4-way `source_kind`) are display/audit fields
  read per-row after selection — payload JSON is fine and avoids a wide
  migration.

The column is `semantic_type` (not `relation_type`) because `links` edges can
reuse it later if link kinds ever grow vocabulary; for `cites` edges it stays
NULL.

### 2. Payload becomes update-in-place for semantic keys

Payload is write-once-at-birth today, which is why `source_memory_id` goes
stale after a rebind. The dual-write helpers gain a merge mode: semantic keys
(`label`, `link_reason`, `linked_at`, `source_kind`, `confidence`,
`explanation`, `source_agent`) may be refreshed on re-assertion of the same
edge id; provenance keys (`span`, `model_version`, `prompt_version`,
`source_memory_id`) keep write-once semantics, and `rebind_edges_identity`
already re-stamps `source_memory_id` across renames.

### 3. Orphan `page_links` rows

Orphans (`target_page_id IS NULL`) stay in `page_links` until resolved — they
are by definition not edges (no dst). Stage 3 therefore drops `page_links`
only after the orphan feed moves to a dedicated small table (or the drop list
keeps a narrowed `page_orphan_links`). Decision deferred to the Stage 1.1
migration PR; the payload PR does not need it.

### 4. Cross-table consistency lints

`lint/pages/provenance_checks/source.rs` validates the legacy stores against
each other — that check dissolves when the stores retire (nothing left to
cross-check). Its replacement is edge-internal integrity (payload
`source_kind` present and in-vocabulary, `semantic_type` non-empty on
relates) — added in the Stage 1 reader-migration PRs, not here.

## The PR's contents

1. **Migration N** (next free slot): `ALTER TABLE edges ADD COLUMN
   semantic_type TEXT` + backfill:
   - `relates` edges: `semantic_type` + payload `confidence`/`explanation`/
     `source_agent` joined from the live `relations` row that derives the same
     edge id (re-derive `compute_edge_id` per row over active relations;
     retired edges are left as-is — history keeps its birth payload).
   - `links` edges: payload `label` from `page_links.label` via the same
     re-derivation (label_key = lower(label)).
   - `cites` edges: payload `link_reason`/`linked_at`/`source_kind` from
     `page_sources` (memory kind) and `page_evidence` (all kinds; evidence
     `title` too where present).
   - Backfill is idempotent and skips edges whose legacy row no longer exists.
2. **Writer updates**: every `dual_write_edge(_with_payload)` call site
   supplies the semantic fields it knows (create_relation_with_span,
   commit_entity_enrichment_at_version, fold_relation_type's re-mint,
   replace_page_links, resolve_orphan_page_links + BindPageLink arm,
   insert_resolved_page_evidence, dual_write_page_citations,
   link_page_evidence, accept_page_merge).
3. **Parity extension**: `reconcile_edges_parity` gains a semantic-drift
   check (legacy `relation_type`/`label` vs edge `semantic_type`/payload) so
   the new fields cannot silently rot before Stage 2 — same watermark, drift
   classes extended.
4. **Teeth**: regression tests per writer (semantic fields present after each
   real path), backfill test on a seeded legacy fixture, parity-oracle drift 0
   before/after.

## Non-goals

- No reader migrates in this PR (Stage 1 PRs do that, per store).
- No writer stops writing legacy rows (Stage 2).
- No store drops (Stage 3).
- M5 `supports` edges: unchanged, still fenced out of parity.

## Risks

- **Backfill joins on re-derived ids** are O(store size) with hashing per
  row — acceptable one-time cost; run inside the migration transaction like
  migration 111.
- **Payload merge mode** touches the shared dual-write helper; every call
  site is covered by the Stage 0 parity-oracle tests, which all assert drift
  0 end-state and would catch a regression in edge identity.
- **Schema widening is downgrade-safe**: older daemons ignore the new column
  (SQLite), so this PR does NOT create the Stage 3 downgrade barrier.
