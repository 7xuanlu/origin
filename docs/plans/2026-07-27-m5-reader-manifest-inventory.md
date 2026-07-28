# M5 Stage 0 — reader manifest, generated inventory

Date: 2026-07-27. Binding for M5 PR-B. This is the enumeration half of
`2026-07-27-m5-reader-manifest.md`; read that file for the rules, this one for
the set.

**This file was generated from the merge base (`5ba8a3b4`), not hand-written.**
That is the point: two earlier hand-enumerations in the companion artifact were
wrong in both directions within a day. Every row below traces to a `.route()`
call, a `#[tool(` declaration, or a `Commands` variant in this tree.

## How `page_bearing` was determined

For each HTTP route, the handler's return type was resolved and its fields
scanned transitively (depth 6) for prose-carrying names: `title`, `summary`,
`content`, `excerpt`, `markdown`, `prose`, `body`, `text`, `snippet`, `label`,
`description` — as **substrings**, not whole words.

Two false-negative classes were found and fixed while building this, both worth
recording because they will recur:

- **`label` is a page title by another name.** `PageLinkOutbound.label`,
  `PageLinkInbound.label`, `OrphanLink.label`, and `PageMapNode.label` all carry
  human-readable page names. A scan keyed on `title` misses every one.
- **Word-boundary matching misses compound fields.** `\bsummary\b` does not
  match `delta_summary` (`PageChangelogEntry`), because `_` is a word character.

The scan is deliberately **over-inclusive**. A row may be demoted from
`page_bearing = yes` only by a written reason recorded in this file, never by
tightening the pattern. Over-inclusion costs one redundant adapter; a false
negative ships an unguarded reader.

## The opaque routes — and why they matter most

16 routes return `serde_json::Value`, a bare `Response`, or `&'static str`.
Their payload **cannot be determined from the type**, so static analysis is
blind to them. They include the three primary page readers:

- `GET /api/pages` — `handle_list_pages`
- `POST /api/pages/search` — `handle_search_pages`
- `GET /api/pages/{id}` — `handle_get_page`

Every opaque route is treated as **`page_bearing = yes`**, fail-closed: an
untyped response is exactly the case where a reviewer's guess is least reliable,
so the guess is not permitted to be the permissive one.

Opacity does **not** force the class to `automatic`. The two axes are
independent — an opaque return type answers "is prose in the payload," not "did
a human name the page." `GET /api/pages/{id}` is explicit whatever its return
type says. Since no provisional content reaches any caller that has not declared
the M5 contract, an opaque explicit route is not a hole.

This is also the strongest available argument for the MCP-wrapper rule already in
`AGENTS.md` ("always typed-deserialize, never `serde_json::Value`"): untyped
responses are invisible to every contract check, including this one.

## HTTP — all 151 registered (method, path, handler) triples

52 page-bearing, 16 opaque (⇒ automatic), 83 not page-bearing.

| Method | Path | Page-bearing | Class | Adapter | Evidence |
|---|---|---|---|---|---|
| `GET` | `/api/activities` | yes | automatic | `handle_list_activities` | AgentActivityRow.memory_titles |
| `GET` | `/api/agents` | yes | automatic | `handle_list_agents` | AgentResponse.description |
| `GET` | `/api/agents/{name}` | yes | automatic | `handle_get_agent` | AgentResponse.description |
| `GET` | `/api/briefing` | yes | automatic | `handle_get_briefing` | BriefingResponse.content |
| `GET` | `/api/capture-stats` | unknown | automatic | `handle_capture_stats` | opaque |
| `POST` | `/api/chunks/delete-bulk` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/chunks/time-range` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/chunks/{id}/update` | no | not_applicable | — | no prose fields |
| `GET` | `/api/chunks/{source_id}` | yes | explicit | `handle_get_chunks` | MemoryDetail.content, MemoryDetail.summary, MemoryDetail.title |
| `GET` | `/api/communities` | no | not_applicable | — | no prose fields |
| `GET` | `/api/communities/members` | no | not_applicable | — | no prose fields |
| `GET` | `/api/communities/page-assignments` | no | not_applicable | — | no prose fields |
| `GET` | `/api/communities/proposals` | no | not_applicable | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/accept` | no | not_applicable | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/reject` | no | not_applicable | — | no prose fields |
| `GET` | `/api/config` | yes | automatic | `handle_get_config` | ConfigResponse.skip_title_patterns |
| `PUT` | `/api/config` | yes | automatic | `handle_update_config` | ConfigResponse.skip_title_patterns |
| `GET` | `/api/config/routing` | no | not_applicable | — | no prose fields |
| `GET` | `/api/config/skip-apps` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/config/skip-apps` | no | not_applicable | — | no prose fields |
| `POST` | `/api/context` | yes | automatic | `handle_context` | ChatContextResponse.context, KnowledgeContext.graph_context, SearchRes |
| `GET` | `/api/debug/pipeline` | unknown | automatic | `handle_pipeline_status` | opaque |
| `GET` | `/api/decisions` | yes | automatic | `handle_list_decisions` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `GET` | `/api/decisions/domains` | no | not_applicable | — | no prose fields |
| `POST` | `/api/distill` | unknown | automatic | `handle_distill` | opaque |
| `POST` | `/api/distill/{page_id}` | unknown | automatic | `handle_redistill` | opaque |
| `POST` | `/api/documents/{source_id}/space` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/documents/{source_id}/tags` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/documents/{source}/{source_id}` | no | not_applicable | — | no prose fields |
| `GET` | `/api/health` | no | not_applicable | — | no prose fields |
| `GET` | `/api/health` | no | not_applicable | — | no prose fields |
| `GET` | `/api/home-stats` | yes | automatic | `handle_get_home_stats` | TopMemory.content |
| `POST` | `/api/import/chat-export` | no | not_applicable | — | no prose fields |
| `POST` | `/api/import/memories` | no | not_applicable | — | no prose fields |
| `GET` | `/api/import/state` | no | not_applicable | — | no prose fields |
| `GET` | `/api/indexed-files` | yes | automatic | `handle_list_indexed_files` | IndexedFileInfo.content, IndexedFileInfo.summary, IndexedFileInfo.titl |
| `POST` | `/api/ingest/memory` | no | not_applicable | — | no prose fields |
| `POST` | `/api/ingest/text` | no | not_applicable | — | no prose fields |
| `POST` | `/api/ingest/webpage` | no | not_applicable | — | no prose fields |
| `GET` | `/api/knowledge/count` | no | not_applicable | — | no prose fields |
| `GET` | `/api/knowledge/path` | no | not_applicable | — | no prose fields |
| `GET` | `/api/knowledge/recent-relations` | no | not_applicable | — | no prose fields |
| `POST` | `/api/llm/test` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/by-ids` | yes | explicit | `handle_get_memories_by_ids` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `POST` | `/api/memory/confirm/{source_id}` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/contradiction/{source_id}/dismiss` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/memory/delete/{source_id}` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/entities` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/entities/list` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/entities/search` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/entities/{entity_id}` | yes | automatic | `handle_get_entity_detail` | Observation.content |
| `POST` | `/api/memory/entities/{entity_id}/observations` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/entities/{id}/confirm` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/memory/entities/{id}/delete` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/entity-suggestions` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/link-entity` | unknown | automatic | `handle_link_entity` | opaque |
| `POST` | `/api/memory/list` | yes | automatic | `handle_list_memories` | IndexedFileInfo.content, IndexedFileInfo.summary, IndexedFileInfo.titl |
| `GET` | `/api/memory/nurture` | yes | automatic | `handle_get_nurture_cards` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `POST` | `/api/memory/observations` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}/confirm` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/pending-revision/{source_id}` | yes | explicit | `handle_get_pending_revision` | PendingRevision.content |
| `GET` | `/api/memory/pending-revisions` | yes | automatic | `handle_list_pending_revisions` | PendingRevisionItem.revision_content |
| `GET` | `/api/memory/pinned` | yes | automatic | `handle_list_pinned_memories` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `GET` | `/api/memory/recent` | yes | automatic | `handle_recent_memories` | RecentActivityItem.snippet, RecentActivityItem.title |
| `POST` | `/api/memory/reclassify/{source_id}` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/rejections` | yes | automatic | `handle_get_rejections` | RejectionRecord.content |
| `POST` | `/api/memory/relations` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/accept` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/dismiss` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/search` | yes | automatic | `handle_search_memory` | SearchResult.content, SearchResult.content_hash, SearchResult.last_del |
| `GET` | `/api/memory/stats` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/store` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/unconfirmed` | yes | automatic | `handle_list_unconfirmed_memories` | RecentActivityItem.snippet, RecentActivityItem.title |
| `POST` | `/api/memory/{id}/correct` | unknown | automatic | `handle_correct_memory` | opaque |
| `GET` | `/api/memory/{id}/detail` | yes | explicit | `handle_get_memory_detail` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `POST` | `/api/memory/{id}/pin` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/{id}/revisions` | yes | explicit | `handle_get_memory_revisions` | MemoryRevisionEntry.content_preview, MemoryRevisionEntry.delta_summary |
| `PUT` | `/api/memory/{id}/stability` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/{id}/unpin` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/{id}/update` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/{id}/update-page` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/{id}/versions` | yes | explicit | `handle_get_version_chain` | MemoryVersionItem.content, MemoryVersionItem.title |
| `GET` | `/api/memory/{source_id}/enrichment-status` | yes | explicit | `handle_get_enrichment_status` | EnrichmentStatusResponse.summary |
| `GET` | `/api/on-device-model` | no | not_applicable | — | no prose fields |
| `POST` | `/api/on-device-model/download` | no | not_applicable | — | no prose fields |
| `GET` | `/api/onboarding/milestones` | no | not_applicable | — | no prose fields |
| `POST` | `/api/onboarding/milestones/{id}/acknowledge` | no | not_applicable | — | no prose fields |
| `POST` | `/api/onboarding/reset` | no | not_applicable | — | no prose fields |
| `GET` | `/api/pages` | unknown | explicit | `handle_list_pages` | opaque |
| `POST` | `/api/pages` | no | not_applicable | — | no prose fields |
| `POST` | `/api/pages/export` | no | not_applicable | — | no prose fields |
| `GET` | `/api/pages/orphan-links` | yes | explicit | `handle_list_orphan_links` | OrphanLink.label, OrphanLinksResponse.orphan_labels |
| `GET` | `/api/pages/recent` | yes | explicit | `handle_recent_pages` | RecentActivityItem.snippet, RecentActivityItem.title |
| `GET` | `/api/pages/recent-changes` | yes | explicit | `handle_recent_page_changes` | PageChange.title |
| `POST` | `/api/pages/search` | unknown | automatic | `handle_search_pages` | opaque |
| `GET` | `/api/pages/{id}` | unknown | explicit | `handle_get_page` | opaque |
| `POST` | `/api/pages/{id}/archive` | unknown | automatic | `handle_archive_page` | opaque |
| `POST` | `/api/pages/{id}/export` | no | not_applicable | — | no prose fields |
| `GET` | `/api/pages/{id}/links` | yes | explicit | `handle_get_page_links` | PageLinkInbound.label, PageLinkOutbound.label |
| `GET` | `/api/pages/{id}/map` | yes | explicit | `handle_get_page_map` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/edges` | yes | explicit | `handle_create_map_edge` | PageMapEdge.label |
| `PATCH` | `/api/pages/{id}/map/edges/{edge_id}` | yes | explicit | `handle_patch_map_edge` | PageMapEdge.label |
| `POST` | `/api/pages/{id}/map/improve` | yes | explicit | `handle_improve_page_map` | PageMapEdge.label, PageMapNode.label |
| `PUT` | `/api/pages/{id}/map/layout` | yes | explicit | `handle_put_page_map_layout` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/nodes` | yes | explicit | `handle_create_map_node` | PageMapNode.label |
| `PATCH` | `/api/pages/{id}/map/nodes/{node_id}` | yes | explicit | `handle_patch_map_node` | PageMapNode.label |
| `GET` | `/api/pages/{id}/revisions` | yes | explicit | `handle_get_page_revisions` | PageChangelogEntry.citations_summary, PageChangelogEntry.delta_summary |
| `GET` | `/api/pages/{id}/sources` | yes | explicit | `handle_get_page_sources` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `GET` | `/api/ping` | unknown | automatic | `handle_ping` | opaque |
| `GET` | `/api/profile` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/profile` | no | not_applicable | — | no prose fields |
| `GET` | `/api/profile/narrative` | yes | automatic | `handle_get_profile_narrative` | NarrativeResponse.content |
| `POST` | `/api/profile/narrative/regenerate` | yes | automatic | `handle_regenerate_narrative` | NarrativeResponse.content |
| `GET` | `/api/refinery/queue` | no | not_applicable | — | no prose fields |
| `POST` | `/api/refinery/queue/{id}/accept` | no | not_applicable | — | no prose fields |
| `POST` | `/api/refinery/queue/{id}/reject` | no | not_applicable | — | no prose fields |
| `POST` | `/api/repairs/apply` | yes | automatic | `handle_apply` | RepairTarget.label_key |
| `POST` | `/api/repairs/plan` | no | not_applicable | — | no prose fields |
| `POST` | `/api/repairs/plan/entries` | yes | automatic | `handle_plan_entries` | RepairMutation.after_title, RepairMutation.before_title, RepairSystemA |
| `POST` | `/api/repairs/prepare` | yes | automatic | `handle_prepare` | RepairMutation.after_title, RepairMutation.before_title, RepairTarget. |
| `POST` | `/api/repairs/verify` | no | not_applicable | — | no prose fields |
| `GET` | `/api/retrievals/recent` | yes | automatic | `handle_recent_retrievals` | RetrievalEvent.memory_snippets, RetrievalEvent.page_titles |
| `POST` | `/api/search` | yes | automatic | `handle_search` | SearchResult.content, SearchResult.content_hash, SearchResult.last_del |
| `PUT` | `/api/setup/anthropic-key` | no | not_applicable | — | no prose fields |
| `GET` | `/api/setup/status` | no | not_applicable | — | no prose fields |
| `POST` | `/api/shutdown` | unknown | automatic | `handle_shutdown` | opaque |
| `GET` | `/api/snapshots` | yes | automatic | `handle_list_snapshots` | SessionSnapshot.summary |
| `GET` | `/api/snapshots/{id}/captures` | yes | automatic | `handle_get_snapshot_captures` | SnapshotCapture.window_title |
| `GET` | `/api/snapshots/{id}/captures-with-content` | yes | automatic | `handle_get_snapshot_captures_with_content` | SnapshotCaptureWithContent.content, SnapshotCaptureWithContent.summary |
| `POST` | `/api/snapshots/{id}/delete` | no | not_applicable | — | no prose fields |
| `GET` | `/api/sources` | no | not_applicable | — | no prose fields |
| `POST` | `/api/sources` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/sources/{id}` | no | not_applicable | — | no prose fields |
| `POST` | `/api/sources/{id}/sync` | no | not_applicable | — | no prose fields |
| `GET` | `/api/spaces` | yes | automatic | `handle_list_spaces` | Space.description |
| `POST` | `/api/spaces` | yes | automatic | `handle_create_space` | Space.description |
| `POST` | `/api/spaces/reorder` | no | not_applicable | — | no prose fields |
| `POST` | `/api/spaces/{from}/move-to/{to}` | unknown | automatic | `handle_move_space` | opaque |
| `DELETE` | `/api/spaces/{name}` | unknown | automatic | `handle_delete_space` | opaque |
| `PUT` | `/api/spaces/{name}` | yes | automatic | `handle_update_space` | Space.description |
| `POST` | `/api/spaces/{name}/confirm` | no | not_applicable | — | no prose fields |
| `POST` | `/api/spaces/{name}/pin` | no | not_applicable | — | no prose fields |
| `POST` | `/api/spaces/{name}/star` | unknown | automatic | `handle_toggle_space_starred` | opaque |
| `GET` | `/api/status` | no | not_applicable | — | no prose fields |
| `GET` | `/api/status` | no | not_applicable | — | no prose fields |
| `POST` | `/api/steep` | no | not_applicable | — | no prose fields |
| `GET` | `/api/suggest-tags` | no | not_applicable | — | no prose fields |
| `GET` | `/api/tags` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/tags/{name}` | no | not_applicable | — | no prose fields |
| `GET` | `/ws/updates` | unknown | automatic | `handle_ws_upgrade` | opaque |

## MCP — all 29 `#[tool(` declarations

Taken from `crates/wenlan-mcp/src/tools.rs` **in this tree**. An earlier draft
listed `search_pages`, `get_page`, `get_page_links`, `list_pages_recent`, and
`list_nurture`; none exist here. Those came from a *running* MCP server's
advertised list, which is a different build. A manifest is a contract about
source, and an installed binary is not evidence about the code being changed.

| Tool | Page-bearing | Class |
|---|---|---|
| `accept_refinement` | no | not_applicable |
| `accept_revision` | no | not_applicable |
| `apply_lint_repair` | no | not_applicable |
| `capture` | no | not_applicable |
| `confirm_memory` | no | not_applicable |
| `context` | yes | automatic |
| `create_entity` | no | not_applicable |
| `create_relation` | no | not_applicable |
| `delete_page` | yes | explicit |
| `dismiss_revision` | no | not_applicable |
| `distill` | yes | automatic |
| `forget` | no | not_applicable |
| `get_lint_agent_work_page` | yes | automatic |
| `get_lint_repair_plan_entries` | yes | automatic |
| `get_memory_revisions` | yes | explicit |
| `get_page_revisions` | yes | explicit |
| `get_page_sources` | yes | explicit |
| `lint` | yes | automatic |
| `list_pending` | yes | automatic |
| `list_pending_imports` | yes | automatic |
| `list_pending_revisions` | yes | automatic |
| `list_refinements` | yes | automatic |
| `list_rejections` | yes | automatic |
| `prepare_lint_repair` | yes | automatic |
| `prepare_lint_repair_plan` | yes | automatic |
| `recall` | yes | automatic |
| `reject_refinement` | no | not_applicable |
| `verify_lint_repair` | no | not_applicable |
| `write_page` | yes | explicit |

## CLI — all 19 `Commands` variants

From `crates/wenlan-cli/src/main.rs:29`. The count is 19, not 18: `Connect` is a
tuple variant (`Connect(commands::mcp::ConnectArgs)`) and is easy to miss when
scanning for brace-shaped variants.

| Subcommand | Page-bearing | Class |
|---|---|---|
| `wenlan status` | no | not_applicable |
| `wenlan setup` | no | not_applicable |
| `wenlan background` | no | not_applicable |
| `wenlan restart` | no | not_applicable |
| `wenlan doctor` | no | not_applicable |
| `wenlan lint` | yes | automatic |
| `wenlan models` | no | not_applicable |
| `wenlan keys` | no | not_applicable |
| `wenlan enrichment` | yes | automatic |
| `wenlan connect` | no | not_applicable |
| `wenlan search` | yes | automatic |
| `wenlan recall` | yes | automatic |
| `wenlan pages` | yes | explicit |
| `wenlan sources` | yes | automatic |
| `wenlan capture` | yes | automatic |
| `wenlan memories` | yes | explicit |
| `wenlan curate` | yes | explicit |
| `wenlan agents` | yes | automatic |
| `wenlan spaces` | yes | automatic |

## Projection, export, internal

| Surface | Class | Adapter |
|---|---|---|
| legacy Markdown projection directory | n/a | `projection_directory_invariant` (companion §5) |
| `POST /api/pages/export`, `POST /api/pages/{id}/export` | explicit | listed in the HTTP table |
| RRF page channel | automatic | internal reader table |
| graph voting / community routing | automatic | internal reader table |
| refinery, summary, briefing builders | automatic | internal reader table |

Internal readers have no registry to enumerate against, so "reads a page" is not
a syntactic property here. The containment rule from the companion artifact §2
applies: every page read routes through a small set of named helpers, and a test
asserts no other module calls the underlying query directly.

## The teeth: this file is data, not documentation

A checked-in inventory is the same hand-copied-canonical-thing that produced the
misses above — unless something proves it still matches the tree. PR-B adds a
test that:

1. re-derives the three enumerations (routes via `TrackedRouter`, MCP via the
   tool table, CLI via `Commands`) from the built binary or source;
2. asserts the derived set **exactly equals** this file's rows — extra and
   missing both fail. Key on `(builder, method, path)`: `GET /api/health` and
   `GET /api/status` are each registered by both `build_router` (`router.rs:47`)
   and `build_repair_router` (`router.rs:592`), on separate `Router` instances.
   Keying on `(method, path)` alone reports a phantom drift on both;
3. asserts every row carries a non-empty `class`;
4. asserts no row carries `page_bearing = unknown` **and** `class != automatic`.

Check 2 is the positive control. Without it this file is a snapshot that rots;
with it, adding a route without classifying it fails the build.
