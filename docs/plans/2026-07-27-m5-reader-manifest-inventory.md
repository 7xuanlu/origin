# M5 Stage 0 — reader manifest, generated inventory

Date: 2026-07-27. Binding for M5 PR-B. This is the enumeration half of
`2026-07-27-m5-reader-manifest.md`; read that file for the rules, this one for
the set.

**Generated from the merge base (`5ba8a3b4`), not hand-written.** Every row
traces to a `.route()` call, a `#[tool(` declaration, or a `Commands` variant.

## What the first generated version got wrong

Recorded because the corrections are the reason to trust this version:

- **Wrong file set.** It scanned `router.rs` and `repair_routes.rs` only, and
  reported 151 rows. Routes also register in `lint_routes.rs` and `routes.rs`.
- **Single-line regex.** Chained method routers spanning lines
  (`.route(p, get(a)\n.delete(b))`) lost every method after the first.
- Together those two hid **12 registrations**. The count is 163:
  155 in `router.rs`, 5 in `repair_routes.rs`,
  2 in `lint_routes.rs`, 1 in `routes.rs`.

The rule that follows: enumerate by pattern over the whole crate, never over a
hand-picked file list. This is the third incomplete enumeration in this program
from exactly that mistake.

## How `page_bearing` is determined

Three independent tests; **any** one of them yields `yes`.

1. **Response fields.** The handler's return type is resolved and scanned
   transitively (depth 6) for prose-carrying names — `title`, `summary`,
   `content`, `excerpt`, `markdown`, `prose`, `body`, `text`, `snippet`,
   `label`, `description` — as **substrings**, not whole words.
2. **Opacity.** A return type of `serde_json::Value`, a bare `Response`, or
   `&'static str` says nothing about the payload, so it counts as page-bearing.
3. **Effects.** A route that writes page prose to a destination is page-bearing
   even when its response carries none. `POST /api/pages/export` and
   `POST /api/pages/{id}/export` return only `ExportStats`, and write full page
   prose into the user's Obsidian vault.

Test 3 exists because tests 1 and 2 are both about the *response*, and the most
consequential page reader in the product does its exposure through a side
effect. A response-only method is structurally blind to it.

Two false-negative classes found in test 1 and fixed, both worth recording:

- **`label` is a page title by another name.** `PageLinkOutbound.label`,
  `PageLinkInbound.label`, `OrphanLink.label`, `PageMapNode.label` all carry
  human-readable page names. A scan keyed on `title` misses every one.
- **Word-boundary matching misses compound fields.** `\bsummary\b` does not
  match `delta_summary` (`PageChangelogEntry`), because `_` is a word character.

The scan is deliberately **over-inclusive**. A row may be demoted to
`page_bearing = no` only by a written reason recorded here, never by tightening
the pattern.

### Recorded demotions

Exactly one, and it is the only row in this file where a `no` overrides an
opaque return type:

- **`GET /ws/updates`.** The handler returns a bare `Response` (the upgrade), so
  test 2 flags it. But the messages it can carry are a closed enum:
  `WsServerMessage` (`websocket.rs:34`) has exactly three variants — index
  progress (`files_indexed`/`files_total`), ingest completion (`document_id`,
  `chunks`), and an error string. No page field exists on any of them. The
  evidence here is stronger than the return type, which describes only the
  protocol upgrade.

An earlier draft made this route the headline example of a page-bearing reader
nobody had noticed. That was wrong, and the correction is why demotions must
cite a type rather than an intuition.

## Why every row is `automatic`

An earlier version classified routes `explicit` when the path named a page —
`GET /api/pages/{id}`, `GET /api/pages`, the browse feeds. That is unsound, and
live app code disproves it:

- `SpaceList.tsx:76` calls `listPages(...)` on a `refetchInterval` of 10 s to
  compute sidebar counts. `GET /api/pages` was classified "human browse"; in
  production it is a 10-second poll.
- `HomePage.tsx:75` loads `listRecentChanges(3)` automatically every 30 s.
- `/api/pages/orphan-links` feeds candidate generation
  (`memory_routes.rs:3464`).

**Reader intent is a property of the call, not of the route.** A client that is
globally M5-aware does not thereby prove that a given request was a human
naming a page. So no route earns `explicit` from its path, and the class column
here is `automatic` everywhere.

`explicit` survives only as a **per-call** signal: a request carries an explicit
human-intent marker, the server records it, and only then may provisional
content appear — with both truth axes rendered. Absent the marker, automatic.
That collapses the mixed-caller problem instead of enumerating around it, and it
is strictly simpler than per-route intent guessing.

It also settles the embedded-content question. An explicit grant covers the
**named page only**; other pages' titles inside the payload (`PageLinkOutbound`,
`PageMapNode`, `OrphanLink`) follow the automatic rule unless they carry their
own axes.

## HTTP — all 163 registered `(method, path, handler)` triples

79 page-bearing, 84 not.

| Method | Path | Page-bearing | Class | Adapter | Evidence |
|---|---|---|---|---|---|
| `GET` | `/api/activities` | yes | automatic | `handle_list_activities` | AgentActivityRow.memory_titles |
| `GET` | `/api/agents` | yes | automatic | `handle_list_agents` | AgentResponse.description |
| `DELETE` | `/api/agents/{name}` | yes | automatic | `handle_delete_agent` | opaque response type — fail-closed |
| `GET` | `/api/agents/{name}` | yes | automatic | `handle_get_agent` | AgentResponse.description |
| `PUT` | `/api/agents/{name}` | yes | automatic | `handle_update_agent` | AgentResponse.description |
| `GET` | `/api/briefing` | yes | automatic | `handle_get_briefing` | BriefingResponse.content |
| `GET` | `/api/capture-stats` | yes | automatic | `handle_capture_stats` | opaque response type — fail-closed |
| `POST` | `/api/chunks/delete-bulk` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/chunks/time-range` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/chunks/{id}/update` | no | not_applicable | — | no prose fields |
| `GET` | `/api/chunks/{source_id}` | yes | automatic | `handle_get_chunks` | MemoryDetail.content, MemoryDetail.summary, MemoryDetail.title |
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
| `GET` | `/api/debug/pipeline` | yes | automatic | `handle_pipeline_status` | opaque response type — fail-closed |
| `GET` | `/api/decisions` | yes | automatic | `handle_list_decisions` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `GET` | `/api/decisions/domains` | no | not_applicable | — | no prose fields |
| `POST` | `/api/distill` | yes | automatic | `handle_distill` | opaque response type — fail-closed |
| `POST` | `/api/distill/{page_id}` | yes | automatic | `handle_redistill` | opaque response type — fail-closed |
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
| `GET` | `/api/lint` | yes | automatic | `handle_lint` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheckResu |
| `POST` | `/api/lint` | yes | automatic | `handle_lint_submission` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheckResu |
| `POST` | `/api/llm/test` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/by-ids` | yes | automatic | `handle_get_memories_by_ids` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
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
| `POST` | `/api/memory/link-entity` | yes | automatic | `handle_link_entity` | opaque response type — fail-closed |
| `POST` | `/api/memory/list` | yes | automatic | `handle_list_memories` | IndexedFileInfo.content, IndexedFileInfo.summary, IndexedFileInfo.titl |
| `GET` | `/api/memory/nurture` | yes | automatic | `handle_get_nurture_cards` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `POST` | `/api/memory/observations` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/memory/observations/{id}` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}/confirm` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/pending-revision/{source_id}` | yes | automatic | `handle_get_pending_revision` | PendingRevision.content |
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
| `POST` | `/api/memory/{id}/correct` | yes | automatic | `handle_correct_memory` | opaque response type — fail-closed |
| `GET` | `/api/memory/{id}/detail` | yes | automatic | `handle_get_memory_detail` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `POST` | `/api/memory/{id}/pin` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/{id}/revisions` | yes | automatic | `handle_get_memory_revisions` | MemoryRevisionEntry.content_preview, MemoryRevisionEntry.delta_summary |
| `PUT` | `/api/memory/{id}/stability` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/{id}/unpin` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/memory/{id}/update` | no | not_applicable | — | no prose fields |
| `POST` | `/api/memory/{id}/update-page` | no | not_applicable | — | no prose fields |
| `GET` | `/api/memory/{id}/versions` | yes | automatic | `handle_get_version_chain` | MemoryVersionItem.content, MemoryVersionItem.title |
| `GET` | `/api/memory/{source_id}/enrichment-status` | yes | automatic | `handle_get_enrichment_status` | EnrichmentStatusResponse.summary |
| `GET` | `/api/on-device-model` | no | not_applicable | — | no prose fields |
| `POST` | `/api/on-device-model/download` | no | not_applicable | — | no prose fields |
| `GET` | `/api/onboarding/milestones` | no | not_applicable | — | no prose fields |
| `POST` | `/api/onboarding/milestones/{id}/acknowledge` | no | not_applicable | — | no prose fields |
| `POST` | `/api/onboarding/reset` | no | not_applicable | — | no prose fields |
| `GET` | `/api/pages` | yes | automatic | `handle_list_pages` | opaque response type — fail-closed |
| `POST` | `/api/pages` | no | not_applicable | — | no prose fields |
| `POST` | `/api/pages/export` | yes | automatic | `handle_export_pages` | EFFECT: writes page prose to the requested vault |
| `GET` | `/api/pages/orphan-links` | yes | automatic | `handle_list_orphan_links` | OrphanLink.label, OrphanLinksResponse.orphan_labels |
| `GET` | `/api/pages/recent` | yes | automatic | `handle_recent_pages` | RecentActivityItem.snippet, RecentActivityItem.title |
| `GET` | `/api/pages/recent-changes` | yes | automatic | `handle_recent_page_changes` | PageChange.title |
| `POST` | `/api/pages/search` | yes | automatic | `handle_search_pages` | opaque response type — fail-closed |
| `DELETE` | `/api/pages/{id}` | yes | automatic | `handle_delete_page` | opaque response type — fail-closed |
| `GET` | `/api/pages/{id}` | yes | automatic | `handle_get_page` | opaque response type — fail-closed |
| `PUT` | `/api/pages/{id}` | no | not_applicable | — | no prose fields |
| `POST` | `/api/pages/{id}/archive` | yes | automatic | `handle_archive_page` | opaque response type — fail-closed |
| `POST` | `/api/pages/{id}/export` | yes | automatic | `handle_export_page` | EFFECT: writes page prose to the requested vault |
| `GET` | `/api/pages/{id}/links` | yes | automatic | `handle_get_page_links` | PageLinkInbound.label, PageLinkOutbound.label |
| `DELETE` | `/api/pages/{id}/map` | yes | automatic | `handle_reset_page_map` | opaque response type — fail-closed |
| `GET` | `/api/pages/{id}/map` | yes | automatic | `handle_get_page_map` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/edges` | yes | automatic | `handle_create_map_edge` | PageMapEdge.label |
| `DELETE` | `/api/pages/{id}/map/edges/{edge_id}` | yes | automatic | `handle_delete_map_edge` | PageMapEdge.label |
| `PATCH` | `/api/pages/{id}/map/edges/{edge_id}` | yes | automatic | `handle_patch_map_edge` | PageMapEdge.label |
| `POST` | `/api/pages/{id}/map/improve` | yes | automatic | `handle_improve_page_map` | PageMapEdge.label, PageMapNode.label |
| `PUT` | `/api/pages/{id}/map/layout` | yes | automatic | `handle_put_page_map_layout` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/nodes` | yes | automatic | `handle_create_map_node` | PageMapNode.label |
| `DELETE` | `/api/pages/{id}/map/nodes/{node_id}` | yes | automatic | `handle_delete_map_node` | PageMapNode.label |
| `PATCH` | `/api/pages/{id}/map/nodes/{node_id}` | yes | automatic | `handle_patch_map_node` | PageMapNode.label |
| `GET` | `/api/pages/{id}/revisions` | yes | automatic | `handle_get_page_revisions` | PageChangelogEntry.citations_summary, PageChangelogEntry.delta_summary |
| `GET` | `/api/pages/{id}/sources` | yes | automatic | `handle_get_page_sources` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Memory |
| `GET` | `/api/ping` | yes | automatic | `handle_ping` | opaque response type — fail-closed |
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
| `DELETE` | `/api/setup/anthropic-key` | no | not_applicable | — | no prose fields |
| `PUT` | `/api/setup/anthropic-key` | no | not_applicable | — | no prose fields |
| `GET` | `/api/setup/status` | no | not_applicable | — | no prose fields |
| `POST` | `/api/shutdown` | yes | automatic | `handle_shutdown` | opaque response type — fail-closed |
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
| `POST` | `/api/spaces/{from}/move-to/{to}` | yes | automatic | `handle_move_space` | opaque response type — fail-closed |
| `DELETE` | `/api/spaces/{name}` | yes | automatic | `handle_delete_space` | opaque response type — fail-closed |
| `PUT` | `/api/spaces/{name}` | yes | automatic | `handle_update_space` | Space.description |
| `POST` | `/api/spaces/{name}/confirm` | no | not_applicable | — | no prose fields |
| `POST` | `/api/spaces/{name}/pin` | no | not_applicable | — | no prose fields |
| `POST` | `/api/spaces/{name}/star` | yes | automatic | `handle_toggle_space_starred` | opaque response type — fail-closed |
| `GET` | `/api/status` | no | not_applicable | — | no prose fields |
| `GET` | `/api/status` | no | not_applicable | — | no prose fields |
| `POST` | `/api/steep` | no | not_applicable | — | no prose fields |
| `GET` | `/api/suggest-tags` | no | not_applicable | — | no prose fields |
| `GET` | `/api/tags` | no | not_applicable | — | no prose fields |
| `DELETE` | `/api/tags/{name}` | no | not_applicable | — | no prose fields |
| `POST` | `/chat/completions` | yes | automatic | `move` | opaque response type — fail-closed |
| `GET` | `/ws/updates` | no | not_applicable | — | DEMOTED, see below |

## MCP — all 29 `#[tool(` declarations

From `crates/wenlan-mcp/src/tools.rs` **in this tree**. An earlier draft listed
`search_pages`, `get_page`, `get_page_links`, `list_pages_recent`, and
`list_nurture`; none exist here. Those came from a *running* MCP server's
advertised list, which is a different build. A manifest is a contract about
source, and an installed binary is not evidence about the code being changed.

MCP responses are assembled per tool rather than returned as one typed struct,
so the response scan does not apply; every tool is page-bearing by default and
must be demoted individually with a recorded reason.

| Tool | Page-bearing | Class | Adapter |
|---|---|---|---|
| `accept_refinement` | yes | automatic | tool handler |
| `accept_revision` | yes | automatic | tool handler |
| `apply_lint_repair` | yes | automatic | tool handler |
| `capture` | yes | automatic | tool handler |
| `confirm_memory` | yes | automatic | tool handler |
| `context` | yes | automatic | tool handler |
| `create_entity` | yes | automatic | tool handler |
| `create_relation` | yes | automatic | tool handler |
| `delete_page` | yes | automatic | tool handler |
| `dismiss_revision` | yes | automatic | tool handler |
| `distill` | yes | automatic | tool handler |
| `forget` | yes | automatic | tool handler |
| `get_lint_agent_work_page` | yes | automatic | tool handler |
| `get_lint_repair_plan_entries` | yes | automatic | tool handler |
| `get_memory_revisions` | yes | automatic | tool handler |
| `get_page_revisions` | yes | automatic | tool handler |
| `get_page_sources` | yes | automatic | tool handler |
| `lint` | yes | automatic | tool handler |
| `list_pending` | yes | automatic | tool handler |
| `list_pending_imports` | yes | automatic | tool handler |
| `list_pending_revisions` | yes | automatic | tool handler |
| `list_refinements` | yes | automatic | tool handler |
| `list_rejections` | yes | automatic | tool handler |
| `prepare_lint_repair` | yes | automatic | tool handler |
| `prepare_lint_repair_plan` | yes | automatic | tool handler |
| `recall` | yes | automatic | tool handler |
| `reject_refinement` | yes | automatic | tool handler |
| `verify_lint_repair` | yes | automatic | tool handler |
| `write_page` | yes | automatic | tool handler |

## CLI — all 19 `Commands` variants

From `crates/wenlan-cli/src/main.rs:29`. The count is 19, not 18: `Connect` is a
tuple variant (`Connect(commands::mcp::ConnectArgs)`), easy to miss when
scanning for brace-shaped variants. The CLI renders daemon responses, so it
inherits the daemon's classification and adds the projection-directory reader.

| Subcommand | Page-bearing | Class | Adapter |
|---|---|---|---|
| `wenlan status` | yes | automatic | subcommand renderer |
| `wenlan setup` | yes | automatic | subcommand renderer |
| `wenlan background` | yes | automatic | subcommand renderer |
| `wenlan restart` | yes | automatic | subcommand renderer |
| `wenlan doctor` | yes | automatic | subcommand renderer |
| `wenlan lint` | yes | automatic | subcommand renderer |
| `wenlan models` | yes | automatic | subcommand renderer |
| `wenlan keys` | yes | automatic | subcommand renderer |
| `wenlan enrichment` | yes | automatic | subcommand renderer |
| `wenlan connect` | yes | automatic | subcommand renderer |
| `wenlan search` | yes | automatic | subcommand renderer |
| `wenlan recall` | yes | automatic | subcommand renderer |
| `wenlan pages` | yes | automatic | subcommand renderer |
| `wenlan sources` | yes | automatic | subcommand renderer |
| `wenlan capture` | yes | automatic | subcommand renderer |
| `wenlan memories` | yes | automatic | subcommand renderer |
| `wenlan curate` | yes | automatic | subcommand renderer |
| `wenlan agents` | yes | automatic | subcommand renderer |
| `wenlan spaces` | yes | automatic | subcommand renderer |

## Projection, export, internal

| Surface | Page-bearing | Class | Adapter |
|---|---|---|---|
| legacy Markdown projection directory | yes | n/a | `projection_directory_invariant` (companion §5) |
| `POST /api/pages/export`, `POST /api/pages/{id}/export` | yes | automatic | in the HTTP table; effect-based |
| RRF page channel | yes | automatic | internal reader table |
| graph voting / community routing | yes | automatic | internal reader table |
| refinery, summary, briefing builders | yes | automatic | internal reader table |

Internal readers have no registry to enumerate against, so "reads a page" is not
a syntactic property. The containment rule from the companion §2 applies: every
page read routes through a small set of named helpers, and a test asserts no
other module calls the underlying query directly. **These four rows are
categories, not call sites** — PR-B must replace them with the enumerated helper
call sites, and that replacement is part of PR-B's definition of done.

## The teeth: this file is data, not documentation

A checked-in inventory is the same hand-copied-canonical-thing that produced the
misses above — unless something proves it still matches the tree. PR-B adds a
test that:

1. re-derives the enumerations from source: routes by scanning **every** file
   under `crates/wenlan-server/src` for `.route(` with paren-balanced argument
   parsing, MCP by `#[tool(`, CLI by `Commands`;
2. asserts the derived set **exactly equals** this file's rows — extra and
   missing both fail. Key on `(builder, method, path)`: `GET /api/health` and
   `GET /api/status` are each registered by both `build_router` and
   `build_repair_router`, on separate `Router` instances, so keying on
   `(method, path)` alone reports a phantom drift on both;
3. re-runs the **prose-field scan** and asserts no row marked
   `page_bearing = no` resolves to a type matching the pattern. Without this the
   evidence column rots: a typed response that later gains a `title` field flips
   no→yes with nothing going RED — the exact drift this file exists to catch;
4. asserts every row carries a non-empty `class`;
5. asserts every row in the effect-writer set is `page_bearing = yes`
   regardless of its response type;
6. asserts no row carries `class = explicit`, since `explicit` is a per-call
   signal and never a route property.

Checks 2 and 3 are the positive controls: 2 keeps the row set live, 3 keeps the
evidence live. Without both, this file is a snapshot that rots.
