# M5 Stage 0 — reader manifest, generated inventory

Date: 2026-07-27. Binding for M5 PR-B. This is the enumeration half of
`2026-07-27-m5-reader-manifest.md`; read that file for the rules, this one for
the set.

**Generated from the merge base (`5ba8a3b4`), not hand-written.** Every row
traces to a `.route()` call, a `#[tool(` declaration, or a `Commands` variant.

## Three wrong counts before this one

Recorded because the corrections are the only reason to trust the fourth:

| Claim | Wrong because |
|---|---|
| 151 | scanned a hand-picked file list (`router.rs`, `repair_routes.rs`), missing `lint_routes.rs`; single-line regex truncated chained method routers spanning lines |
| 163 | fixed both of those, then counted a route registered **inside a `#[cfg(test)]` block** in `routes.rs` — its handler was an inline closure, which is why the adapter cell read `move` |
| **162** | current: paren-balanced parse, every file, test modules stripped |

155 in `router.rs`, 5 in `repair_routes.rs`,
2 in `lint_routes.rs`. `routes.rs` contributes none.

Two rules follow, and both are now part of the contract:

- **enumerate by pattern over the whole crate**, never over a hand-picked file
  list;
- **strip `#[cfg(test)]` modules**, or the manifest classifies fixtures as
  production surface.

A garbage adapter cell (`move`) was the visible symptom of the third error. The
adapter column is an enforcement address, so PR-B's test rejects any value that
does not resolve to a function.

## How `page_bearing` is determined

Four independent tests; **any** one yields `yes`.

1. **Response fields.** The handler's return type is resolved and scanned
   transitively (depth 6) for prose-carrying names — `title`, `summary`,
   `content`, `excerpt`, `markdown`, `prose`, `body`, `text`, `snippet`,
   `label`, `description` — as **substrings**, not whole words.
2. **Opacity.** `serde_json::Value`, a bare `Response`, or `&'static str` says
   nothing about the payload, so it counts as page-bearing.
3. **Effects.** A route that writes page prose to a destination is page-bearing
   even when its response carries none. `POST /api/pages/export` and
   `POST /api/pages/{id}/export` return only `ExportStats` and write full page
   prose into the user's Obsidian vault.
4. ~~**The error arm.**~~ **Withdrawn as a `page_bearing` test** — see below.
   159 of 162 handlers return
   `Result<_, ServerError>`, and **every** `ServerError` variant carries a
   free-form `String` (`error.rs:11`). D4's stale-base save conflict is exactly
   where an implementer helpfully writes `current version: <title>`. That risk
   is real; making it a per-route classification is what was wrong.

Tests 1 and 2 inspect the Ok path only. Test 3 was added because the most
consequential page reader in the product exposes nothing through its response.

**Test 4 was wrong as stated, and the table did not obey it.** If a free-form
`ServerError` arm made a route page-bearing, 159 of 162 routes would be
page-bearing; the table says 77. `POST /api/import/memories` and the bulk-delete
route return `Result<_, ServerError>` and are marked `no`. A rule the data
contradicts is not a rule.

The error arm is a **real leak and the wrong axis**. It is not a property of any
individual route — it is one cross-cutting invariant with a single enforcement
point at the error-serialization seam: no `ServerError` body may contain a
provisional page's title or prose, enforced once, for every route including
routes added later. That is both stronger than classifying 159 rows and far less
to maintain. `page_bearing` returns to meaning what it says: does the **Ok**
path, or a write effect, carry page prose.

Two false-negative classes found in test 1 and fixed:

- **`label` is a page title by another name.** `PageLinkOutbound.label`,
  `PageLinkInbound.label`, `OrphanLink.label`, `PageMapNode.label` all carry
  human-readable page names. A scan keyed on `title` misses every one.
- **Word-boundary matching misses compound fields.** `\bsummary\b` does not
  match `delta_summary` (`PageChangelogEntry`), because `_` is a word character.

The scan is deliberately **over-inclusive**. A row may be demoted to
`page_bearing = no` only by a written reason recorded here, never by tightening
the pattern.

### Recorded demotions

Exactly one, and it is the only row where `no` overrides an opaque return type:

- **`GET /ws/updates`.** The handler returns a bare `Response` (the upgrade), so
  test 2 flags it. But the messages it can carry are a closed enum:
  `WsServerMessage` (`websocket.rs:34`) has exactly three variants — index
  progress, ingest completion, and an error string — and no page field exists on
  any of them. The evidence is stronger than the return type, which describes
  only the protocol upgrade.

An earlier draft made this route the headline example of a page-bearing reader
nobody had noticed. That was wrong, and the correction is why demotions cite a
type rather than an intuition.

## Class is always `automatic`; `explicit` is per-call

An earlier version classified routes `explicit` when the path named a page.
Live app code disproves it: `SpaceList.tsx:76` polls `listPages(...)` every 10 s
for sidebar counts, and `HomePage.tsx:75` polls `listRecentChanges(3)` every
30 s. **Reader intent is a property of the call, not the route.**

So no route earns `explicit` from its path. `explicit` exists only as a
**per-call human-intent marker** that the server records, and which binds to the
page IDs the call names (companion §4).

## `marker_shape` — what a marker may do, per route

A boolean `marker_eligible` was the wrong column. It was set `yes` for every
page-bearing HTTP route, which made `POST /api/context`, `POST /api/search`,
and both export routes marker-eligible — the three shapes D3 excludes
unconditionally. Universal eligibility moves the mixed-caller hole instead of
closing it: an agent that transmits the marker on `/api/context` contaminates
exactly the automatic context D3's first sentence protects.

The column is now three-valued and **fail-closed by construction**: `none`
unless a route is on the allowlist below. A route added tomorrow is `none` until
someone deliberately gives it a shape.

| Shape | What the marker grants | Routes |
|---|---|---|
| `none` | nothing — **the request is refused**, not silently downgraded | everything not listed below (153 of 162) |
| `collection` | provisional **entries**: page ID + title + both axes per item, never prose | `GET /api/pages`, `GET /api/pages/recent`, `GET /api/pages/recent-changes`, `POST /api/pages/search` |
| `named_page` | full prose for the page named in the path, both axes | `GET /api/pages/{id}`, `.../links`, `.../map`, `.../revisions`, `.../sources` |

`/api/pages/orphan-links` was on the `collection` list for one draft and does
not belong there. Its items are `OrphanLink { label, count }`
(`wenlan-types/src/responses.rs:1128`) — no page ID and no truth axes. The
carve-out is **conditional** on rendering both axes per item; a shape that
cannot carry them cannot receive the grant. An entry surfacing without its state
is the unearned trust this rung exists to prevent, so the route is `none` and
its labels stay excluded like any other embedded other-page title. The general
rule: **a route qualifies for `collection` only if its item type can carry a
page identity and both axes.**

Refusing on `none` rather than ignoring is deliberate. An ignored marker is a
wiring mistake that behaves correctly today and silently wrong after a refactor;
a refused one fails loudly at the first integration test.

### Two gates, and only one of them is cooperative

The previous draft said eligibility is "enforced client-side by a test on each
surface," which is another way of saying the server does not enforce it. Split
the concern and one half becomes hard:

| Gate | Question | Enforcement |
|---|---|---|
| **route shape** | may a marker do anything here, and what? | **server-side and hard.** No client cooperation. `none` refuses. |
| **marker authenticity** | did a human actually gesture? | **cooperative-tier.** The daemon is loopback and unauthenticated (artifact 5 §1); it cannot tell a forged marker from a real one. |

The first gate closes the *bulk* path: a cooperative agent that forges the
marker and claims to be the app still cannot pull provisional prose out of
`/api/context`, `/api/search`, or an export, because those routes refuse the
marker regardless of caller.

**It does not bound total exposure, and an earlier draft claimed it did.** The
two shapes compose: a forging agent calls a `collection` route to enumerate
provisional page IDs, then calls `named_page` once per ID, and reconstructs the
corpus a page at a time into its own prompt. "Bounded to a page the caller named
by ID" is true per call and worthless in aggregate, because the caller can name
every ID it just discovered.

That composition is not a new hole, and this is the honest place to say so: it
requires **forging the marker**, which artifact 5 §1 already concedes as out of
scope (T11, hostile same-user). Nothing at cooperative tier can prevent it —
the daemon has no signal that separates a forged marker from a real one, so any
"prevention" claim here would be theatre.

What the gates actually buy, stated without inflation:

| Against | Effect |
|---|---|
| a careless integration (an MCP tool wired to send the marker) | **prevented** — per-surface rule, plus every automatic-context and export route refuses regardless |
| an accidental bulk leak (one `/api/context` call returning provisional prose) | **prevented** — shape gate, no cooperation needed |
| a deliberately forging agent enumerating then fetching | **not prevented.** Conceded at T11. Made *visible*: every marked call is recorded with caller identity, the page IDs it named, and a timestamp, so page-at-a-time extraction is an auditable pattern rather than an invisible one |

Under the boolean column, the third row was invisible *and* the first two were
unprotected. The gain is real; the bound is not.

### Per-surface transmission is the cooperative half

| Surface | May transmit | Why |
|---|---|---|
| HTTP, from an interactive client | on shaped routes only | a human gesture can exist |
| MCP tools | **never** | no human gesture; the agent is the caller |
| internal readers | **never** | no caller to gesture |
| non-interactive CLI subcommands | **never** | scripted, not browsed |

A surface that transmits anyway is a failing test on that surface. That gate is
soft, and saying so plainly is what cooperative-tier means. It is also no longer
load-bearing: the shape gate holds even when this one is bypassed.

## HTTP — all 162 registered `(method, path, handler)` triples

77 page-bearing, 85 not.

| Method | Path | Builder | Page-bearing | Class | Marker-shape | Adapter | Evidence |
|---|---|---|---|---|---|---|---|
| `GET` | `/api/activities` | main | yes | automatic | `none` | `handle_list_activities` | AgentActivityRow.memory_titles |
| `GET` | `/api/agents` | main | yes | automatic | `none` | `handle_list_agents` | AgentResponse.description |
| `DELETE` | `/api/agents/{name}` | main | yes | automatic | `none` | `handle_delete_agent` | opaque response type — fail-closed |
| `GET` | `/api/agents/{name}` | main | yes | automatic | `none` | `handle_get_agent` | AgentResponse.description |
| `PUT` | `/api/agents/{name}` | main | yes | automatic | `none` | `handle_update_agent` | AgentResponse.description |
| `GET` | `/api/briefing` | main | yes | automatic | `none` | `handle_get_briefing` | BriefingResponse.content |
| `GET` | `/api/capture-stats` | main | yes | automatic | `none` | `handle_capture_stats` | opaque response type — fail-closed |
| `POST` | `/api/chunks/delete-bulk` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/chunks/time-range` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/chunks/{id}/update` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/chunks/{source_id}` | main | yes | automatic | `none` | `handle_get_chunks` | MemoryDetail.content, MemoryDetail.summary, MemoryDetail.title |
| `GET` | `/api/communities` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/members` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/page-assignments` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/proposals` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/accept` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/reject` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/config` | main | yes | automatic | `none` | `handle_get_config` | ConfigResponse.skip_title_patterns |
| `PUT` | `/api/config` | main | yes | automatic | `none` | `handle_update_config` | ConfigResponse.skip_title_patterns |
| `GET` | `/api/config/routing` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/config/skip-apps` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/config/skip-apps` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/context` | main | yes | automatic | `none` | `handle_context` | ChatContextResponse.context, KnowledgeContext.graph_context, Searc |
| `GET` | `/api/debug/pipeline` | main | yes | automatic | `none` | `handle_pipeline_status` | opaque response type — fail-closed |
| `GET` | `/api/decisions` | main | yes | automatic | `none` | `handle_list_decisions` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `GET` | `/api/decisions/domains` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/distill` | main | yes | automatic | `none` | `handle_distill` | opaque response type — fail-closed |
| `POST` | `/api/distill/{page_id}` | main | yes | automatic | `none` | `handle_redistill` | opaque response type — fail-closed |
| `POST` | `/api/documents/{source_id}/space` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/documents/{source_id}/tags` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/documents/{source}/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/health` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/health` | repair | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/home-stats` | main | yes | automatic | `none` | `handle_get_home_stats` | TopMemory.content |
| `POST` | `/api/import/chat-export` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/import/memories` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/import/state` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/indexed-files` | main | yes | automatic | `none` | `handle_list_indexed_files` | IndexedFileInfo.content, IndexedFileInfo.summary, IndexedFileInfo. |
| `POST` | `/api/ingest/memory` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/ingest/text` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/ingest/webpage` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/count` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/path` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/recent-relations` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/lint` | main + repair | yes | automatic | `none` | `handle_lint` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck |
| `POST` | `/api/lint` | main + repair | yes | automatic | `none` | `handle_lint_submission` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck |
| `POST` | `/api/llm/test` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/by-ids` | main | yes | automatic | `none` | `handle_get_memories_by_ids` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `POST` | `/api/memory/confirm/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/contradiction/{source_id}/dismiss` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/delete/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities/list` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities/search` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/entities/{entity_id}` | main | yes | automatic | `none` | `handle_get_entity_detail` | Observation.content |
| `POST` | `/api/memory/entities/{entity_id}/observations` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/entities/{id}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/entities/{id}/delete` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/entity-suggestions` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/link-entity` | main | yes | automatic | `none` | `handle_link_entity` | opaque response type — fail-closed |
| `POST` | `/api/memory/list` | main | yes | automatic | `none` | `handle_list_memories` | IndexedFileInfo.content, IndexedFileInfo.summary, IndexedFileInfo. |
| `GET` | `/api/memory/nurture` | main | yes | automatic | `none` | `handle_get_nurture_cards` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `POST` | `/api/memory/observations` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/observations/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/pending-revision/{source_id}` | main | yes | automatic | `none` | `handle_get_pending_revision` | PendingRevision.content |
| `GET` | `/api/memory/pending-revisions` | main | yes | automatic | `none` | `handle_list_pending_revisions` | PendingRevisionItem.revision_content |
| `GET` | `/api/memory/pinned` | main | yes | automatic | `none` | `handle_list_pinned_memories` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `GET` | `/api/memory/recent` | main | yes | automatic | `none` | `handle_recent_memories` | RecentActivityItem.snippet, RecentActivityItem.title |
| `POST` | `/api/memory/reclassify/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/rejections` | main | yes | automatic | `none` | `handle_get_rejections` | RejectionRecord.content |
| `POST` | `/api/memory/relations` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/accept` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/dismiss` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/search` | main | yes | automatic | `none` | `handle_search_memory` | SearchResult.content, SearchResult.content_hash, SearchResult.last |
| `GET` | `/api/memory/stats` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/store` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/unconfirmed` | main | yes | automatic | `none` | `handle_list_unconfirmed_memories` | RecentActivityItem.snippet, RecentActivityItem.title |
| `POST` | `/api/memory/{id}/correct` | main | yes | automatic | `none` | `handle_correct_memory` | opaque response type — fail-closed |
| `GET` | `/api/memory/{id}/detail` | main | yes | automatic | `none` | `handle_get_memory_detail` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `POST` | `/api/memory/{id}/pin` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/{id}/revisions` | main | yes | automatic | `none` | `handle_get_memory_revisions` | MemoryRevisionEntry.content_preview, MemoryRevisionEntry.delta_sum |
| `PUT` | `/api/memory/{id}/stability` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/{id}/unpin` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/{id}/update` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/{id}/update-page` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/{id}/versions` | main | yes | automatic | `none` | `handle_get_version_chain` | MemoryVersionItem.content, MemoryVersionItem.title |
| `GET` | `/api/memory/{source_id}/enrichment-status` | main | yes | automatic | `none` | `handle_get_enrichment_status` | EnrichmentStatusResponse.summary |
| `GET` | `/api/on-device-model` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/on-device-model/download` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/onboarding/milestones` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/onboarding/milestones/{id}/acknowledge` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/onboarding/reset` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/pages` | main | yes | automatic | **`collection`** | `handle_list_pages` | opaque response type — fail-closed |
| `POST` | `/api/pages` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/pages/export` | main | yes | automatic | `none` | `handle_export_pages` | EFFECT: writes page prose to the requested vault |
| `GET` | `/api/pages/orphan-links` | main | yes | automatic | `none` | `handle_list_orphan_links` | OrphanLink.label, OrphanLinksResponse.orphan_labels |
| `GET` | `/api/pages/recent` | main | yes | automatic | **`collection`** | `handle_recent_pages` | RecentActivityItem.snippet, RecentActivityItem.title |
| `GET` | `/api/pages/recent-changes` | main | yes | automatic | **`collection`** | `handle_recent_page_changes` | PageChange.title |
| `POST` | `/api/pages/search` | main | yes | automatic | **`collection`** | `handle_search_pages` | opaque response type — fail-closed |
| `DELETE` | `/api/pages/{id}` | main | yes | automatic | `none` | `handle_delete_page` | opaque response type — fail-closed |
| `GET` | `/api/pages/{id}` | main | yes | automatic | **`named_page`** | `handle_get_page` | opaque response type — fail-closed |
| `PUT` | `/api/pages/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/pages/{id}/archive` | main | yes | automatic | `none` | `handle_archive_page` | opaque response type — fail-closed |
| `POST` | `/api/pages/{id}/export` | main | yes | automatic | `none` | `handle_export_page` | EFFECT: writes page prose to the requested vault |
| `GET` | `/api/pages/{id}/links` | main | yes | automatic | **`named_page`** | `handle_get_page_links` | PageLinkInbound.label, PageLinkOutbound.label |
| `DELETE` | `/api/pages/{id}/map` | main | yes | automatic | `none` | `handle_reset_page_map` | opaque response type — fail-closed |
| `GET` | `/api/pages/{id}/map` | main | yes | automatic | **`named_page`** | `handle_get_page_map` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/edges` | main | yes | automatic | `none` | `handle_create_map_edge` | PageMapEdge.label |
| `DELETE` | `/api/pages/{id}/map/edges/{edge_id}` | main | yes | automatic | `none` | `handle_delete_map_edge` | PageMapEdge.label |
| `PATCH` | `/api/pages/{id}/map/edges/{edge_id}` | main | yes | automatic | `none` | `handle_patch_map_edge` | PageMapEdge.label |
| `POST` | `/api/pages/{id}/map/improve` | main | yes | automatic | `none` | `handle_improve_page_map` | PageMapEdge.label, PageMapNode.label |
| `PUT` | `/api/pages/{id}/map/layout` | main | yes | automatic | `none` | `handle_put_page_map_layout` | PageMapEdge.label, PageMapNode.label |
| `POST` | `/api/pages/{id}/map/nodes` | main | yes | automatic | `none` | `handle_create_map_node` | PageMapNode.label |
| `DELETE` | `/api/pages/{id}/map/nodes/{node_id}` | main | yes | automatic | `none` | `handle_delete_map_node` | PageMapNode.label |
| `PATCH` | `/api/pages/{id}/map/nodes/{node_id}` | main | yes | automatic | `none` | `handle_patch_map_node` | PageMapNode.label |
| `GET` | `/api/pages/{id}/revisions` | main | yes | automatic | **`named_page`** | `handle_get_page_revisions` | PageChangelogEntry.citations_summary, PageChangelogEntry.delta_sum |
| `GET` | `/api/pages/{id}/sources` | main | yes | automatic | **`named_page`** | `handle_get_page_sources` | MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me |
| `GET` | `/api/ping` | main | yes | automatic | `none` | `handle_ping` | opaque response type — fail-closed |
| `GET` | `/api/profile` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/profile` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/profile/narrative` | main | yes | automatic | `none` | `handle_get_profile_narrative` | NarrativeResponse.content |
| `POST` | `/api/profile/narrative/regenerate` | main | yes | automatic | `none` | `handle_regenerate_narrative` | NarrativeResponse.content |
| `GET` | `/api/refinery/queue` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/refinery/queue/{id}/accept` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/refinery/queue/{id}/reject` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/repairs/apply` | main + repair | yes | automatic | `none` | `handle_apply` | RepairTarget.label_key |
| `POST` | `/api/repairs/plan` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/repairs/plan/entries` | main | yes | automatic | `none` | `handle_plan_entries` | RepairMutation.after_title, RepairMutation.before_title, RepairSys |
| `POST` | `/api/repairs/prepare` | main | yes | automatic | `none` | `handle_prepare` | RepairMutation.after_title, RepairMutation.before_title, RepairTar |
| `POST` | `/api/repairs/verify` | main + repair | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/retrievals/recent` | main | yes | automatic | `none` | `handle_recent_retrievals` | RetrievalEvent.memory_snippets, RetrievalEvent.page_titles |
| `POST` | `/api/search` | main | yes | automatic | `none` | `handle_search` | SearchResult.content, SearchResult.content_hash, SearchResult.last |
| `DELETE` | `/api/setup/anthropic-key` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/setup/anthropic-key` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/setup/status` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/shutdown` | main | yes | automatic | `none` | `handle_shutdown` | opaque response type — fail-closed |
| `GET` | `/api/snapshots` | main | yes | automatic | `none` | `handle_list_snapshots` | SessionSnapshot.summary |
| `GET` | `/api/snapshots/{id}/captures` | main | yes | automatic | `none` | `handle_get_snapshot_captures` | SnapshotCapture.window_title |
| `GET` | `/api/snapshots/{id}/captures-with-content` | main | yes | automatic | `none` | `handle_get_snapshot_captures_with_content` | SnapshotCaptureWithContent.content, SnapshotCaptureWithContent.sum |
| `POST` | `/api/snapshots/{id}/delete` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/sources` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/sources` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/sources/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/sources/{id}/sync` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/spaces` | main | yes | automatic | `none` | `handle_list_spaces` | Space.description |
| `POST` | `/api/spaces` | main | yes | automatic | `none` | `handle_create_space` | Space.description |
| `POST` | `/api/spaces/reorder` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{from}/move-to/{to}` | main | yes | automatic | `none` | `handle_move_space` | opaque response type — fail-closed |
| `DELETE` | `/api/spaces/{name}` | main | yes | automatic | `none` | `handle_delete_space` | opaque response type — fail-closed |
| `PUT` | `/api/spaces/{name}` | main | yes | automatic | `none` | `handle_update_space` | Space.description |
| `POST` | `/api/spaces/{name}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{name}/pin` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{name}/star` | main | yes | automatic | `none` | `handle_toggle_space_starred` | opaque response type — fail-closed |
| `GET` | `/api/status` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/status` | repair | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/steep` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/suggest-tags` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/tags` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/tags/{name}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/ws/updates` | main | no | not_applicable | `none` | — | DEMOTED, see below |

## MCP — all 29 `#[tool(` declarations

From `crates/wenlan-mcp/src/tools.rs` **in this tree**. An earlier draft listed
`search_pages`, `get_page`, `get_page_links`, `list_pages_recent`, and
`list_nurture`; none exist here. Those came from a *running* MCP server's
advertised list — a different build. A manifest is a contract about source, and
an installed binary is not evidence about the code being changed.

MCP responses are assembled per tool rather than returned as one typed struct,
so the response scan does not apply: every tool is page-bearing by default and
must be demoted individually with a recorded reason. **Every MCP tool is
`marker_shape = none`.**

| Tool | Page-bearing | Class | Marker-shape | Adapter |
|---|---|---|---|---|
| `accept_refinement` | yes | automatic | `none` | tool handler |
| `accept_revision` | yes | automatic | `none` | tool handler |
| `apply_lint_repair` | yes | automatic | `none` | tool handler |
| `capture` | yes | automatic | `none` | tool handler |
| `confirm_memory` | yes | automatic | `none` | tool handler |
| `context` | yes | automatic | `none` | tool handler |
| `create_entity` | yes | automatic | `none` | tool handler |
| `create_relation` | yes | automatic | `none` | tool handler |
| `delete_page` | yes | automatic | `none` | tool handler |
| `dismiss_revision` | yes | automatic | `none` | tool handler |
| `distill` | yes | automatic | `none` | tool handler |
| `forget` | yes | automatic | `none` | tool handler |
| `get_lint_agent_work_page` | yes | automatic | `none` | tool handler |
| `get_lint_repair_plan_entries` | yes | automatic | `none` | tool handler |
| `get_memory_revisions` | yes | automatic | `none` | tool handler |
| `get_page_revisions` | yes | automatic | `none` | tool handler |
| `get_page_sources` | yes | automatic | `none` | tool handler |
| `lint` | yes | automatic | `none` | tool handler |
| `list_pending` | yes | automatic | `none` | tool handler |
| `list_pending_imports` | yes | automatic | `none` | tool handler |
| `list_pending_revisions` | yes | automatic | `none` | tool handler |
| `list_refinements` | yes | automatic | `none` | tool handler |
| `list_rejections` | yes | automatic | `none` | tool handler |
| `prepare_lint_repair` | yes | automatic | `none` | tool handler |
| `prepare_lint_repair_plan` | yes | automatic | `none` | tool handler |
| `recall` | yes | automatic | `none` | tool handler |
| `reject_refinement` | yes | automatic | `none` | tool handler |
| `verify_lint_repair` | yes | automatic | `none` | tool handler |
| `write_page` | yes | automatic | `none` | tool handler |

## CLI — all 19 `Commands` variants

From `crates/wenlan-cli/src/main.rs:29`. The count is 19, not 18: `Connect` is a
tuple variant, easy to miss when scanning for brace-shaped variants. The CLI
renders daemon responses, so it inherits the daemon's classification and adds
the projection-directory reader.

| Subcommand | Page-bearing | Class | Marker-shape | Adapter |
|---|---|---|---|---|
| `wenlan status` | yes | automatic | `none` | subcommand renderer |
| `wenlan setup` | yes | automatic | `none` | subcommand renderer |
| `wenlan background` | yes | automatic | `none` | subcommand renderer |
| `wenlan restart` | yes | automatic | `none` | subcommand renderer |
| `wenlan doctor` | yes | automatic | `none` | subcommand renderer |
| `wenlan lint` | yes | automatic | `none` | subcommand renderer |
| `wenlan models` | yes | automatic | `none` | subcommand renderer |
| `wenlan keys` | yes | automatic | `none` | subcommand renderer |
| `wenlan enrichment` | yes | automatic | `none` | subcommand renderer |
| `wenlan connect` | yes | automatic | `none` | subcommand renderer |
| `wenlan search` | yes | automatic | `none` | subcommand renderer |
| `wenlan recall` | yes | automatic | `none` | subcommand renderer |
| `wenlan pages` | yes | automatic | **`collection` + `named_page`** | subcommand renderer |
| `wenlan sources` | yes | automatic | `none` | subcommand renderer |
| `wenlan capture` | yes | automatic | `none` | subcommand renderer |
| `wenlan memories` | yes | automatic | `none` | subcommand renderer |
| `wenlan curate` | yes | automatic | `none` | subcommand renderer |
| `wenlan agents` | yes | automatic | `none` | subcommand renderer |
| `wenlan spaces` | yes | automatic | `none` | subcommand renderer |

## Projection, export, internal

| Surface | Page-bearing | Class | Marker-shape | Adapter |
|---|---|---|---|---|
| legacy Markdown projection directory | yes | n/a | `none` | `projection_directory_invariant` (companion §5) |
| `POST /api/pages/export`, `/api/pages/{id}/export` | yes | automatic | `none` | in the HTTP table; effect-based |

Both rows are effect-based: neither returns page prose, and both write it. The
internal helper readers are enumerated in full below rather than described as
categories.

## Internal readers — enumerated by a committed generator

An earlier draft listed four categories and deferred enumeration to PR-B. A
later one enumerated 76 call sites. Both were wrong, in different ways, and the
second failure is the instructive one.

**The predicate lived in prose, so every implementation of it disagreed.** Three
rounds produced 76, then a reviewer's 65 and 71, then 54 — not because the tree
changed but because "a function that reads page prose" admits at least four
readings. A count derived from an unpinned predicate is not a fact, and this
file had been presenting one as a fact.

The predicate is therefore **executable and checked in**:
`scripts/m5-reader-sweep.py`. Its docstring is the specification; the number is
whatever it prints. Two implementation defects it now documents in place:

- **body delimitation.** Bodies were delimited by the next `fn` match, which
  merged adjacent functions and let one function's SQL vote for its neighbour.
  Bodies are now brace-matched.
- **predicate scope.** The `pages` match and the prose-column match were tested
  anywhere in the body, so `list_tags_scoped` (selects tags, merely checks page
  existence, `db.rs:19900`) and `tally` (no page SQL at all, `db.rs:18199`)
  counted. Both halves must now hold **inside the same string literal**.

The earlier "differential cross-check" — two test-strippers agreeing on 76 — was
worthless, and worth recording as a trap. Both implementations varied only the
`#[cfg(test)]` stripping and shared the same broken body delimitation. **A
differential oracle that varies the wrong dimension confirms the bug in
stereo.** It is not evidence unless the varied dimension is the one that can be
wrong.

### Current output

| | Count |
|---|---|
| internal page-prose readers | **54** |
| exposure paths — `pub` **and** called from outside `wenlan-core` | **10** |
| internal-only | **44** |
| name-ambiguous — caller edges unresolvable by a name scan | **3** |

`db.rs` holds 28 of them. The four categories the first draft named — "RRF page
channel", "graph voting / community routing", "refinery / briefing builders" —
match none of the modules that actually appear.

Every row is `marker_shape = none`. There is no caller to gesture.

### Exposure is a starting classification, not a proof

Caller edges are resolved by **name**, which over-matches on generic names and
under-matches through trait dispatch and re-exports. Rows whose function name is
declared more than once in the tree are marked name-ambiguous and excluded from
the exposure set rather than given a caller list that would read as evidence —
the two `load` functions are `pub(super)` with core-parent callers only, and an
earlier draft credited them with server callers that do not exist.

PR-B re-resolves every edge with the language server and re-runs this generator,
asserting set equality on the rows and the partition. Until then the split is a
classification to be confirmed, and internal-only is **not** a pass: it means no
*current* caller crosses the boundary, which one `pub` re-export changes.

### Exposure paths (adapter required)

| Address | Function | Visibility | Caller outside core |
|---|---|---|---|
| `core/db.rs:16980` | `reconcile_entity_page_parity` | `pub` | `server/scheduler.rs:2237` |
| `core/db.rs:39619` | `list_recent_retrievals_scoped` | `pub` | `server/routes.rs:1121` |
| `core/db.rs:43455` | `load_page_source_index` | `pub` | `server/routes.rs:928` |
| `core/db.rs:46287` | `list_stale_pages_scoped` | `pub` | `server/routes.rs:970` |
| `core/db.rs:46327` | `find_stale_archived_pages` | `pub` | `server/cmd_backfill.rs:49` |
| `core/db/scoped_entities.rs:12` | `list_entities_scoped` | `pub` | `server/memory_routes.rs:1429` |
| `core/db/scoped_entities.rs:84` | `get_entity_detail_scoped` | `pub` | `server/memory_routes.rs:1445` |
| `core/db/scoped_entities.rs:291` | `list_recent_relations_scoped` | `pub` | `server/knowledge_routes.rs:67` |
| `core/db/scoped_entities.rs:615` | `search_entities_by_vector_scoped` | `pub` | `server/memory_routes.rs:1480` |
| `core/db/scoped_pages.rs:376` | `list_recent_changes_scoped` | `pub` | `server/routes.rs:1141` |

### Internal-only

| Address | Function | Visibility | Why internal-only |
|---|---|---|---|
| `core/db.rs:9240` | `migrate_89_page_kind_fold` | `private` | not `pub` |
| `core/db.rs:26449` | `rebind_source_page_in_transaction` | `private` | not `pub` |
| `core/db.rs:34283` | `oldest_active_page` | `pub` | no caller outside core |
| `core/db.rs:39479` | `list_recent_retrievals` | `pub` | no caller outside core |
| `core/db.rs:39825` | `list_recent_changes` | `pub` | no caller outside core |
| `core/db.rs:40128` | `list_recent_pages_with_badges` | `pub` | no caller outside core |
| `core/db.rs:41041` | `append_page_history` | `private` | not `pub` |
| `core/db.rs:41202` | `insert_page_with_kind_inner` | `private` | not `pub` |
| `core/db.rs:41739` | `get_page_inner` | `private` | not `pub` |
| `core/db.rs:41768` | `get_page_by_entity` | `pub` | no caller outside core |
| `core/db.rs:41843` | `list_pages_inner` | `private` | not `pub` |
| `core/db.rs:41876` | `list_pages_stale` | `pub` | no caller outside core |
| `core/db.rs:41921` | `list_pages_by_space` | `pub` | no caller outside core |
| `core/db.rs:42982` | `find_matching_page` | `pub` | no caller outside core |
| `core/db.rs:43039` | `find_matching_page_scoped` | `pub` | no caller outside core |
| `core/db.rs:43406` | `page_merge_row` | `private` | not `pub` |
| `core/db.rs:43582` | `list_active_page_titles_scoped` | `pub` | no caller outside core |
| `core/db.rs:43624` | `list_relevant_active_page_titles` | `pub` | no caller outside core |
| `core/db.rs:43688` | `find_active_page_id_by_title` | `pub` | no caller outside core |
| `core/db.rs:43723` | `find_unique_active_page_id_by_title_scoped` | `pub` | no caller outside core |
| `core/db.rs:44274` | `backfill_page_embeddings` | `pub` | no caller outside core |
| `core/db.rs:45351` | `get_pages_for_memory` | `pub` | no caller outside core |
| `core/db.rs:46250` | `get_stale_page_after` | `pub` | no caller outside core |
| `core/lint/deep.rs:221` | `page_duplicates` | `private` | not `pub` |
| `core/lint/deep.rs:299` | `page_body_result` | `private` | not `pub` |
| `core/lint/pages/db_checks.rs:233` | `load_rows` | `private` | not `pub` |
| `core/lint/pages/link_checks/orphans.rs:8` | `load` | `pub(super)` | name-ambiguous — caller edges unresolvable by name |
| `core/lint/semantic_candidates.rs:846` | `load_pages` | `private` | name-ambiguous — caller edges unresolvable by name |
| `core/lint/serving/query.rs:16` | `load` | `pub(super)` | name-ambiguous — caller edges unresolvable by name |
| `core/maintenance.rs:557` | `scan_automatic_retro_stub_slice` | `private` | not `pub` |
| `core/maintenance/duplicates.rs:63` | `scan_near_duplicate_slice` | `pub(super)` | not `pub` |
| `core/maintenance/duplicates.rs:281` | `embedding_near_duplicate_pairs` | `private` | not `pub` |
| `core/post_write.rs:583` | `rename_page_title_cas_inner` | `private` | not `pub` |
| `core/post_write.rs:906` | `page_on_connection` | `pub(crate)` | not `pub` |
| `core/post_write.rs:939` | `apply_deterministic_repair_cas` | `pub` | no caller outside core |
| `core/repair.rs:1387` | `prepare_rename_page_title` | `private` | not `pub` |
| `core/repair.rs:3930` | `capture_rename_page_row_on_snapshot` | `private` | not `pub` |
| `core/repair.rs:4159` | `validate_rename_page_title_collision_on_snapshot` | `private` | not `pub` |
| `core/repair.rs:4203` | `validate_rename_page_title_collision_on_connection` | `pub(crate)` | not `pub` |
| `core/repair.rs:6145` | `projection_page_receipt_sql` | `private` | not `pub` |
| `core/repair_plan/deterministic.rs:96` | `resolve_duplicate_page_titles` | `private` | not `pub` |
| `core/repair_plan/deterministic.rs:270` | `renamed_page_title_still_actionable` | `private` | not `pub` |
| `core/repair_plan/deterministic.rs:420` | `resolve_source_pages` | `private` | not `pub` |
| `core/repair_plan/deterministic.rs:1147` | `resolve_orphan_links` | `private` | not `pub` |

## The teeth: this file is data, not documentation

A checked-in inventory is the same hand-copied-canonical-thing that produced
three wrong counts — unless something proves it still matches the tree. PR-B
adds a test that:

1. re-derives the enumerations from source: routes by scanning **every** file
   under `crates/wenlan-server/src` for `.route(` with paren-balanced argument
   parsing and `#[cfg(test)]` modules stripped; MCP by `#[tool(`; CLI by
   `Commands`;
2. asserts the derived set **exactly equals** this file's rows — extra and
   missing both fail. Key on `(builder, method, path)`, which is why the table
   carries a `Builder` column; keying on `(method, path)` alone reports a
   phantom drift. The two counts differ and both are correct:

   | Count | Value | What it counts |
   |---|---|---|
   | call-site triples | **162** | rows in this table: 155 `router.rs` + 5 `repair_routes.rs` + 2 `lint_routes.rs` |
   | runtime builder triples | **166** | `(builder, method, path)` pairs actually installed: 160 `main` + 6 `repair` |

   The +4 is two call sites that each land in **both** builders:
   `lint_routes::register` (2 triples) and `repair_routes::register_execution`
   (2 triples). The second is easy to miss and was missed once:
   `repair_routes::register` **wraps** `register_execution`
   (`repair_routes.rs:25`), so `main` gets all five repair routes, not three,
   while `build_repair_router` installs `apply`/`verify` again
   (`router.rs:591`). An arithmetic that reads only the two `register*` call
   sites in `router.rs` lands on 164 and is wrong.

   `/api/health` and `/api/status` do *not* inflate: each is two separate
   `.route()` call sites, one per builder region, so they are already two rows
   here. `build_router` delegates to `build_router_with_shutdown`
   (`router.rs:24`), so those are one builder, not two;
3. re-runs the **prose-field scan** and asserts no row marked
   `page_bearing = no` resolves to a type matching the pattern. Without this the
   evidence column rots: a typed response that later gains a `title` field flips
   no→yes with nothing going RED;
4. asserts every `adapter` cell resolves to a real function — the `move` cell
   that a closure produced must fail, not sit there looking like an address;
5. asserts every row in the effect-writer set is `page_bearing = yes` regardless
   of its response type;
6. asserts **no row carries `class = explicit`**, since `explicit` is a per-call
   signal and never a route property;
7. asserts the `Marker-shape` column is **fail-closed**: re-derives the
   allowlist and asserts every route not on it is `none`, so a route added
   without a deliberate shape cannot default to eligible;
8. asserts a marker sent to a `none` route is **refused** — not ignored, not
   silently downgraded to automatic — and that `POST /api/context`,
   `POST /api/search`, and both export routes are among the refusals;
9. asserts no surface marked never-transmit (MCP, internal, non-interactive
   CLI) sends the marker;
10. re-runs `scripts/m5-reader-sweep.py` and asserts its output equals the
    internal-reader tables — the generator is the predicate, so this is a real
    positive control rather than a second reading of prose;
11. asserts the exposure partition with the **language server**, not a name
    scan: every internal-only row is either not `pub` or has no caller outside
    `wenlan-core`. A row that gains an outside caller fails until it is moved to
    the exposure table and given an adapter;
12. asserts every `collection`-shaped route's item type can carry a page
    identity **and** both truth axes — the check that keeps
    `OrphanLink { label, count }`-shaped payloads off the allowlist;
13. sentinel test: seed a provisional page, drive every error path that names
    it, and assert its title and prose appear in **no** error body. This is the
    single error-seam invariant, not a per-route classification.

Checks 2 and 3 are the positive controls: 2 keeps the row set live, 3 keeps the
evidence live. Without both, this file is a snapshot that rots.
