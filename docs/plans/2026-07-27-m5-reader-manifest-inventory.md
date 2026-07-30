# M5 Stage 0 — reader manifest, generated inventory

Date: 2026-07-27. Binding for M5 PR-B. This is the enumeration half of
`2026-07-27-m5-reader-manifest.md`; read that file for the rules, this one for
the set.

**Generated from the merge base (`5ba8a3b4`), not hand-written.** Every row
traces to a `.route()` call, a `#[tool(` declaration, or a `Commands` variant.

**Amended for PR #399.** The three `/api/spaces/default` routes landed on `main`
after this file was generated, so the original 162 was correct at `5ba8a3b4` and
stale by the time PR-B cut its branch. They were not found by re-reading this
file -- `TrackedRouter`'s coverage assert failed the first time it ran, which is
the whole point of keeping the enforcement in the router rather than in a
second scan. Current count: 167.

## Three wrong counts before this one

Recorded because the corrections are the only reason to trust the fourth:

| Claim | Wrong because |
|---|---|
| 151 | scanned a hand-picked file list (`router.rs`, `repair_routes.rs`), missing `lint_routes.rs`; single-line regex truncated chained method routers spanning lines |
| 163 | fixed both of those, then counted a route registered **inside a `#[cfg(test)]` block** in `routes.rs` — its handler was an inline closure, which is why the adapter cell read `move` |
| **162** | current: paren-balanced parse, every file, test modules stripped |

160 in `router.rs`, 5 in `repair_routes.rs`,
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
   free-form `String` (`server/error.rs:11`). D4's stale-base save conflict is exactly
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

Thirty-five were recorded; **twenty-one still stand.** Two were reversed at
pre-merge review and twelve more by the provenance re-audit, both below. One of
the survivors — `GET /ws/updates` — is the only row where `no` overrides an
opaque return type; the rest are transparent handlers traced to their data
source.

- **`GET /ws/updates`.** The handler returns a bare `Response` (the upgrade), so
  test 2 flags it. But the messages it can carry are a closed enum:
  `WsServerMessage` (`websocket.rs:34`) has exactly three variants — index
  progress, ingest completion, and an error string — and no page field exists on
  any of them. The evidence is stronger than the return type, which describes
  only the protocol upgrade.

An earlier draft made this route the headline example of a page-bearing reader
nobody had noticed. That was wrong, and the correction is why demotions cite a
type rather than an intuition.

#### Source-traced demotions (2026-07-28)

The scan of section "How `page_bearing` was determined" matched on response
*field names*, so a route returning any `title`/`content`/`summary` field was
marked `yes` regardless of where those bytes came from. Thirty-four rows were
demoted here on 2026-07-28; pre-merge review then returned two of them to
page-bearing and the provenance re-audit returned twelve more (both reversal
sections below), leaving **20**. Each remaining bullet was traced from handler
to the SQL or core function it actually reads, and cites the data-source line
rather than the response type.

**RETRACTED 2026-07-28 — the original cross-cutting claim was false.** An
earlier revision of this section asserted that "no production code path writes
page prose into `memories`," backed by a name-based SQL scan. Pre-merge review
disproved it by reading the code: `stage_page_revision_card`
(`post_write.rs:3090`) builds `RawDocument { title: "Revision: {page.title}",
content: <full new page body>, pending_revision: true }` and upserts it into
`memories` (`post_write.rs:3123`, `:3141`). The scan missed it because the
function names no page table in a SQL literal — it copies prose through a Rust
struct. **A name-based scan cannot establish provenance, and no future revision
of this document should reinstate that claim on scan evidence.**

What actually holds up the memories-backed demotions is narrower and has to be
checked per row: the revision card is written with `pending_revision = 1`, and
the ordinary memory readers filter it out. `list_memories_scoped`
(`db.rs:28545`) and `get_memories_by_source_ids_scoped` (`db.rs:28834`) both
carry `AND pending_revision = 0`; the two pending-revision readers require
`EXISTS (SELECT 1 FROM memories target WHERE target.source_id =
revision.supersedes …)` (`db.rs:35595`, `db.rs:35454`), which a `page_…` id
does not satisfy. So a demotion is sound only where the reader carries that
filter — **the question is the provenance of the bytes, not the name of the
table.**

`GET /api/chunks/{source_id}` is where that distinction bit: `get_chunks_scoped`
(`db.rs:27287`) conditions on `source_id` and `source IN ('memory','file')` with
no `pending_revision` filter, so given a revision-card id it returns the page
title and full body. It is now page-bearing.

Audit status, stated plainly: this section listed 32 source-traced demotions and
`GET /ws/updates` carries a 33rd, demoted higher up for a different reason. All
**33** were re-checked under the provenance question on 2026-07-28. **Twelve
were wrong and are reversed below; 21 stand**, 20 of them still listed here. The
twelve failed the same way: asking which table the handler reads returns
`memories`, which is true and decides nothing, because `stage_page_revision_card`
(`post_write.rs:3060`) writes a full copy of a human-owned page *into* `memories`,
and `dismiss_pending_revision` (`db.rs:35386`) used to leave that copy behind
with `pending_revision = 0` — the one column the surviving demotions lean on.
The "26 under-audited" follow-up this paragraph used to carry is closed.

- **`GET /api/agents`.** `list_agents` (`db.rs:34274`) selects `FROM
  agent_connections`. No join, no page read.
- **`GET /api/agents/{name}`.** `get_agent` (`db.rs:34252`) selects `FROM
  agent_connections WHERE name = ?1`.
- **`PUT /api/agents/{name}`.** `update_agent` (`db.rs:34295`) writes
  `agent_connections`, then the handler re-reads through `get_agent`
  (`db.rs:34252`).
- **`GET /api/config`.** `load_config` (`config.rs:180`) is a
  `std::fs::read_to_string` of `config.json` — no DB at all. The
  `skip_title_patterns` field that the field-name scan matched is a list of
  capture-skip window-title globs (`config.rs:30`, `config.rs:49`), not page
  titles.
- **`PUT /api/config`.** Same file-backed source; the handler returns
  `config_to_response(&cfg)` (`config_routes.rs:15`, `config_routes.rs:116`).
- **`GET /api/decisions`.** `list_memories_scoped` (`db.rs:28498`) reads
  `memories` joined to `enrichment_steps`.
- **`GET /api/memory/pending-revision/{source_id}`.**
  `get_pending_revision_for_scoped` (`db.rs:35448`) selects `FROM memories`.
- **`GET /api/memory/pending-revisions`.** `list_pending_revisions_scoped`
  (`db.rs:35589`) selects `FROM memories`.
- **`GET /api/memory/rejections`.** `handle_get_rejections`
  (`memory_routes.rs:1934`) calls `get_rejections` (`db.rs:41112`), which
  selects `FROM rejected_memories`.
- **`GET /api/memory/{source_id}/enrichment-status`.**
  `get_enrichment_status_scoped` (`db.rs:32439`) composes `get_enrichment_steps`
  (`db.rs:32611`) and `get_enrichment_summary` (`db.rs:32640`), both over
  `enrichment_steps`.
- **`GET /api/profile/narrative`.** Served from `get_cached_narrative`
  (`db.rs:36575`, `FROM narrative_cache`) or generated by `generate_narrative`
  (`narrative.rs:122`) over `get_memories_for_narrative` (`db.rs:36514`).
- **`POST /api/profile/narrative/regenerate`.** Same path, forced regeneration
  (`memory_routes.rs:3273`).
- **`GET /api/snapshots`.** `get_recent_snapshots` (`db.rs:36328`) selects `FROM
  session_snapshots`.
- **`GET /api/snapshots/{id}/captures`.** `get_captures_for_snapshot_scoped`
  (`db.rs:36138`) joins `capture_refs` to `memories`.
- **`GET /api/snapshots/{id}/captures-with-content`.**
  `get_snapshot_captures_with_content_scoped` (`db.rs:36277`) composes
  `db.rs:36138`, `db.rs:27061`, and `get_snapshot_capture_content`
  (`db.rs:36236`, `FROM memories`).
- **`GET /api/spaces`.** `list_spaces` (`db.rs:19226`) selects `FROM spaces`
  with COUNT subqueries over `memories` and `entities`.
- **`POST /api/spaces`.** `create_space` (`db.rs:19317`) reads only `SELECT
  MAX(sort_order) FROM spaces` before inserting.
- **`GET /api/spaces/default`.** `get_default_space` (`space_context.rs:73`)
  resolves through `get_space_by_id` (`space_context.rs:35`), a `spaces` read.
- **`PUT /api/spaces/default`.** `set_default_space` (`space_context.rs:106`)
  returns `get_space_by_id` (`space_context.rs:35`).
- **`PUT /api/spaces/{name}`.** `update_space` (`db.rs:19374`) does touch
  `pages`, but its only page statement (`db.rs:19428`-`db.rs:19446`) is a
  scope-rename cascade that writes space/workspace/version/last_modified and
  selects nothing back; the response is `get_space(new_name)` (`db.rs:19265`).

#### Demotions reversed (2026-07-28, pre-merge review)

Two rows demoted above were wrong and are now page-bearing. Both were missed
the same way: the demotion asked which table the handler reads, when the
question that decides disclosure is where the bytes in that table came from.

- **`GET /api/activities` → `yes`.** `agent_activity.detail` carries page
  titles verbatim, written at three sites — `post_write.rs:2078`
  (`write_document_source_page_impl`), `:2163` (`replace_source_page_impl`),
  and `:2918` (`create_page_impl`) — each as
  `format!("title={title}, sources={}", …)` into `log_agent_activity`. The
  column is selected straight back out (`db.rs:40866`, `:40967`), lands in
  `AgentActivityRow.detail` (`db.rs:40905`, `:41016`), and is returned by
  `handle_list_activities` (`server/memory_routes.rs:2687`). The original
  demotion cited "the only title source is `SELECT DISTINCT title FROM
  memories`" — true of the `memory_titles` field, but it overlooked the
  `detail` column selected by that same statement.
- **`GET /api/chunks/{source_id}` → `yes`.** See the retraction above:
  `get_chunks_scoped` (`db.rs:27287`) has no `pending_revision` filter, so a
  revision-card id returns the staged page title and full body.

Both were found by reading, not by driving a live daemon. The cheapest
confirmation at the wire, if wanted: create a page with a sentinel title, then
`GET /api/activities` and grep the body for it; and capture a
`revision_card_id` from a gated page write, then `GET /api/chunks/{that_id}`.

#### Demotions reversed (2026-07-28, provenance re-audit)

Twelve more. The pair above were found one at a time; this pass put the
provenance question to all thirty-three demoted rows at once and got the same
answer twelve times. One row is the whole mechanism:
`stage_page_revision_card` (`post_write.rs:3060`) copies a human-owned page into
`memories` as `title = "Revision: {page.title}"` (`post_write.rs:3088`),
`content` **and** `source_text` set to the full new page body
(`post_write.rs:3096`, `:3107`), `memory_type = "fact"`,
`source_agent = "page_write"`, `confirmed = false`, `stability = "new"`,
`supersedes = page.id`, `pending_revision = true` (`post_write.rs:3098`-`:3105`).
Every reader below reads `memories`, and none of them knows that row is a page.

Two things end the exposure, and both are narrower than the demotions assumed.
Accept consumes the card: `accept_pending_revision_with_knowledge_path`
(`post_write.rs:4174`) resolves it, and `try_update_page_content` deletes it in
the same transaction as the page write (`db.rs:43284`, `db.rs:43320`). Dismiss
did not — it fell straight through to `db.dismiss_pending_revision`
(`db.rs:35386`), whose statement is `UPDATE memories SET pending_revision = 0,
supersedes = NULL` under the comment "Unstage, not delete" (`db.rs:35405`). A
dismissed page card therefore became a permanent, ordinary, retrievable memory
carrying a full copy of the page's prose — with `pending_revision = 0`, the exact
column the Tier B readers below trust. **This PR closes that asymmetry:**
`dismiss_pending_revision` (`post_write.rs:4211`) now resolves the page card
first and deletes it (`post_write.rs:4216`-`:4217`), mirroring accept, while a
memory card keeps unstage semantics — `resolve_page_revision_card` returns
`None` unless `structured_fields` says `revision_kind = "page_write"` and
`target_kind = "page"` (`post_write.rs:4027`-`:4031`). The fix is forward-only.
Cards dismissed by an older binary are already ordinary rows in a live database,
and nothing in this PR goes back for them.

All twelve are `marker_shape: none` and stay that way. They are page-bearing on
the automatic path; no marker can widen them.

**Tier A — the staged card is enough, with no user action at all.** These
readers carry no `pending_revision` filter, so a card is exposed from the moment
it is staged.

- **`GET /api/memory/nurture` → `yes`.** `handle_get_nurture_cards`
  (`server/memory_routes.rs:1932`) → `get_nurture_cards_scoped`
  (`db.rs:35061`). The `WHERE` is `source='memory' AND stability='new' AND
  COALESCE(is_recap,0)=0` plus the superseder clause
  (`db.rs:35090`-`db.rs:35093`) — no `pending_revision` filter — and
  `ORDER BY c.pending_revision DESC` (`db.rs:35096`) sorts staged cards
  **first**. It selects `c.title`, `c.content` and `c.source_text`, so the queue
  meant for reviewing captures returns the whole page body, top of the list.
- **`GET /api/memory/recent` → `yes`.** `handle_recent_memories`
  (`server/memory_routes.rs:4010`) → `list_recent_memories_scoped`
  (`db.rs:40319`). Its `WHERE` is `source='memory' AND chunk_index=0` plus a
  `supersede_mode` exclusion, no `pending_revision`. `RecentActivityItem.title`
  is the card title verbatim and `snippet` falls back to the first 100 chars of
  content (`db.rs:40432`).
- **`GET /api/memory/unconfirmed` → `yes`.**
  `handle_list_unconfirmed_memories` (`server/memory_routes.rs:4046`) →
  `list_unconfirmed_memories_scoped` (`db.rs:40479`). It filters
  `(confirmed = 0 OR confirmed IS NULL)` (`db.rs:40501`), which a staged card
  satisfies by construction (`post_write.rs:3102`), and carries no
  `pending_revision` filter. Same title-plus-100-chars payload.
- **`GET /api/indexed-files` → `yes`.** `handle_list_indexed_files`
  (`server/memory_routes.rs:2399`) → `list_indexed_files_scoped`
  (`db.rs:27061`), whose only condition is `source != 'episode'`
  (`db.rs:27065`). Narrowest of the twelve: `content` is hard-coded empty
  (`db.rs:27109`), so what leaks is `MAX(title)` — the page's title, which this
  manifest already treats as page prose ("`label` is a page title by another
  name").
- **`GET /api/briefing` → `yes`.** `handle_get_briefing`
  (`server/memory_routes.rs:3292`) → `generate_briefing_scoped`
  (`briefing.rs:181`) → `get_recent_memories_for_briefing_scoped`
  (`db.rs:36873`) / `get_recent_memories_for_briefing` (`db.rs:36834`). Neither
  filters `pending_revision`; both select `title, content` whole. The title
  reaches the wire through `titles_fallback` (`briefing.rs:64`), and on the LLM
  path the full body becomes the prompt — the same off-machine hop the
  re-distillation section records.
- **`GET /api/memory/{id}/revisions` → `yes`.** `handle_get_memory_revisions`
  (`server/memory_routes.rs:5639`) → `walk_supersede_chain_scoped`
  (`db.rs:48696`). The recursive CTE anchors on `source_id = ?1` with no
  `pending_revision` filter and returns `memory.title` plus
  `substr(memory.content, 1, 200)` (`db.rs:48758`-`db.rs:48759`) as
  `MemoryRevisionEntry.title` / `.content_preview`
  (`wenlan-types/src/responses.rs:704`, `:705`). Hand it the card id and it
  answers.
- **`GET /api/memory/{id}/versions` → `yes`.** `handle_get_version_chain`
  (`server/memory_routes.rs:2931`) → `get_version_chain_scoped`
  (`db.rs:33928`). No `pending_revision` filter, and the per-item statement
  selects `MAX(title), MAX(content)` into `MemoryVersionItem.title` / `.content`
  (`wenlan-types/src/memory.rs:154`, `:155`) — the full body, not a preview.

**Tier B — one dismiss.** These do carry `AND pending_revision = 0`, which
stops a *staged* card and not a dismissed one, because dismiss set that column
to 0.

- **`POST /api/memory/list` → `yes`.** `handle_list_memories`
  (`server/memory_routes.rs:1107`) → `list_filtered_confirmed_scoped`
  (`db.rs:29220`), which pushes `pending_revision = 0` at `db.rs:29274`. The
  handler forwards `req.confirmed` unchanged (`server/memory_routes.rs:1124`),
  so a request that omits the field applies no confirmed filter and the row
  comes back with `MAX(title)` and `MAX(content)`.
- **`GET /api/memory/by-ids` → `yes`.** `handle_get_memories_by_ids`
  (`server/memory_routes.rs:2902`) → `get_memories_by_source_ids_scoped`
  (`db.rs:28822`), first condition `pending_revision = 0` (`db.rs:28834`). It
  returns `title`, the concatenated `content` and `source_text` — two full
  copies of the page body per row.
- **`GET /api/memory/{id}/detail` → `yes`.** `handle_get_memory_detail`
  (`server/memory_routes.rs:2877`) → `get_memory_detail_scoped`
  (`db.rs:28811`), a one-id call into that same reader. Same filter, same
  payload.

**Tier B+ — a dismiss and one further ordinary action.** The dismiss clears
`pending_revision`; a second, unremarkable step clears what is left.

- **`GET /api/memory/pinned` → `yes`.** `handle_list_pinned_memories`
  (`server/memory_routes.rs:3378`) → `list_pinned_memories_scoped`
  (`db.rs:33092`), which is `list_memories_scoped(scope, None, None, Some(true),
  100)` (`db.rs:33096`). `pending_revision = 0` (`db.rs:28545`) stops the staged
  card; after a dismiss the surviving gate is `pinned = true`. Pinned, the row
  carries `title`, `content` and `source_text`.
- **`GET /api/home-stats` → `yes`, and its gate is not the one the tier name
  suggests.** `handle_get_home_stats` (`server/memory_routes.rs:1537`) →
  `get_home_stats_scoped` (`db.rs:33631`) / `get_home_stats` (`db.rs:33276`).
  Neither carries a `pending_revision` filter anywhere. `TopMemory.content` is
  `SUBSTR(c.content, 1, 200)` joined through `access_log` (`db.rs:33546`,
  `db.rs:33731`) with an all-time `access_count > 0` fallback (`db.rs:33582`,
  `db.rs:33775`), so the card must have been *retrieved* to appear at all.
  Search is what holds it back — the hybrid path excludes
  `pending_revision != 0` (`db.rs:21724`) — so the access count stays at zero
  until a dismiss makes the row searchable. The reachable sequence is still
  dismiss-then-retrieve; the column doing the work is `access_log`, not
  `pending_revision`.

Found by reading, like the pair above. The cheapest confirmation at the wire:
stage a card against a page with a sentinel title (a gated page write returns
its `revision_card_id`), then `GET /api/memory/nurture` and grep the body for
the sentinel; dismiss that card and repeat against
`GET /api/memory/{card_id}/detail`.

#### Corrected evidence on kept rows (2026-07-28)

Two rows kept `page_bearing = yes` — the right verdict — while citing evidence
that does not carry it. A row whose evidence names the wrong field looks
demotable the moment that field is refactored away, so the cell is load-bearing
even when the verdict does not move.

- **`GET /api/memory/entities/{entity_id}`**, was `Observation.content`, now
  `Entity.name = pages.title (M3)`. An observation is a knowledge-graph string,
  not page prose, so it never justified the verdict. What does: under the M3
  entity reader cutover, `get_entity_detail_scoped`
  (`db/scoped_entities.rs:116`) selects `p.title` from `pages` through
  `entity_page_map` and returns it as the entity's `name`. The page class it
  reads is `kind = 'entity'` — precisely the class `select_visible_pages`
  excludes, so the shared visibility helper never sees this path. The cutover is
  per-consumer and default-OFF (`set_entity_reader_cutover`), so the disclosure
  is conditional; the manifest records the strongest reason, not the current
  default.
- **`POST /api/repairs/apply`**, now `RepairTarget.label_key`. The receipt
  flattens to `RepairTarget::PageLink { source_page_id, label_key, scope }`
  (`wenlan-types/src/repair.rs:1701`), and a wikilink label is a page title
  under another name. A scan over response *types* cannot see this; only
  following the flatten does.

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
| `none` | nothing — **the request is refused**, not silently downgraded | everything not listed below (160 of 167) |
| `collection` | provisional **entries**: page ID + title + both axes per item, never prose | `GET /api/pages`, `POST /api/pages/search` |
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

**PR-C applied that rule to the two rows it had not been applied to.**
`GET /api/pages/recent` and `GET /api/pages/recent-changes` carried
`collection` while their item types fail the same test `orphan-links` failed:
`RecentActivityItem` (`wenlan-types/src/memory.rs:444`) has a prose `snippet`
and neither axis, and `PageChange` (`:415`) is a bare page ID, title, kind and
timestamp. Nothing over-exposed — both adapters are Full-only, so a provisional
page was already dropped rather than reduced — which is exactly what made it
worth fixing now: the shape promised a carve-out the wire types cannot honour,
and the trap was the next person to "fix" the adapters to keep that promise,
reducing a page into a struct with nowhere to put the axes. Both are `none`
until those types grow them. Tooth:
`truth_guard_test.rs::the_recent_feeds_refuse_a_marker_they_cannot_honour`.

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

## HTTP — all 167 registered `(method, path, handler)` triples

60 page-bearing, 107 not.

| Method | Path | Builder | Page-bearing | Class | Marker-shape | Adapter | Evidence |
|---|---|---|---|---|---|---|---|
| `GET` | `/api/activities` | main | yes | automatic | `none` | `handle_list_activities` | AgentActivityRow.detail = title={page.title} |
| `GET` | `/api/agents` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `DELETE` | `/api/agents/{name}` | main | yes | automatic | `none` | `handle_delete_agent` | opaque response type — fail-closed |
| `GET` | `/api/agents/{name}` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `PUT` | `/api/agents/{name}` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `PATCH` | `/api/brief` | main | no | not_applicable | `none` | — | BriefUpdateReceipt carries no page prose |
| `POST` | `/api/brief` | main | yes | automatic | `none` | `handle_read_brief` | Brief.last_session_summary, BriefItem.text, SearchResult.content |
| `GET` | `/api/briefing` | main | yes | automatic | `none` | `handle_get_briefing` | BriefingResponse.content = revision-card titles |
| `GET` | `/api/capture-stats` | main | yes | automatic | `none` | `handle_capture_stats` | opaque response type — fail-closed |
| `POST` | `/api/chunks/delete-bulk` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/chunks/time-range` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/chunks/{id}/update` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/chunks/{source_id}` | main | yes | automatic | `none` | `handle_get_chunks` | MemoryDetail.title/content via revision card |
| `GET` | `/api/communities` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/members` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/page-assignments` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/communities/proposals` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/accept` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/communities/proposals/{id}/reject` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/config` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `PUT` | `/api/config` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/config/routing` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/config/skip-apps` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/config/skip-apps` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/context` | main | yes | automatic | `none` | `handle_context` | ChatContextResponse.context, KnowledgeContext.graph_context, Searc |
| `GET` | `/api/debug/pipeline` | main | yes | automatic | `none` | `handle_pipeline_status` | opaque response type — fail-closed |
| `GET` | `/api/decisions` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/decisions/domains` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/distill` | main | yes | automatic | `none` | `handle_distill` | opaque response type — fail-closed |
| `POST` | `/api/distill/{page_id}` | main | yes | automatic | `none` | `handle_redistill` | opaque response type — fail-closed |
| `POST` | `/api/documents/{source_id}/space` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/documents/{source_id}/tags` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/documents/{source}/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/health` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/health` | repair | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/home-stats` | main | yes | automatic | `none` | `handle_get_home_stats` | TopMemory.content via dismissed revision card |
| `POST` | `/api/import/chat-export` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/import/memories` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/import/state` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/indexed-files` | main | yes | automatic | `none` | `handle_list_indexed_files` | IndexedFileInfo.title via revision card |
| `POST` | `/api/ingest/memory` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/ingest/text` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/ingest/webpage` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/count` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/path` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/knowledge/recent-relations` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/lint` | main + repair | yes | automatic | `none` | `handle_lint` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck |
| `POST` | `/api/lint` | main + repair | yes | automatic | `none` | `handle_lint_submission` | LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck |
| `POST` | `/api/llm/test` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/by-ids` | main | yes | automatic | `none` | `handle_get_memories_by_ids` | MemoryItem.title/content via dismissed card |
| `POST` | `/api/memory/confirm/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/contradiction/{source_id}/dismiss` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/delete/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities/list` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/entities/search` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/entities/{entity_id}` | main | yes | automatic | `none` | `handle_get_entity_detail` | Entity.name = pages.title (M3) |
| `POST` | `/api/memory/entities/{entity_id}/observations` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/entities/{id}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/entities/{id}/delete` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/entity-suggestions` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/link-entity` | main | yes | automatic | `none` | `handle_link_entity` | opaque response type — fail-closed |
| `POST` | `/api/memory/list` | main | yes | automatic | `none` | `handle_list_memories` | IndexedFileInfo.title/content via dismissed card |
| `GET` | `/api/memory/nurture` | main | yes | automatic | `none` | `handle_get_nurture_cards` | MemoryItem.title/content via staged revision card |
| `POST` | `/api/memory/observations` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/memory/observations/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/observations/{id}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/pending-revision/{source_id}` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/memory/pending-revisions` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/memory/pinned` | main | yes | automatic | `none` | `handle_list_pinned_memories` | MemoryItem.title/content via dismissed pinned card |
| `GET` | `/api/memory/recent` | main | yes | automatic | `none` | `handle_recent_memories` | RecentActivityItem.title/snippet via revision card |
| `POST` | `/api/memory/reclassify/{source_id}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/rejections` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/memory/relations` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/accept` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/revision/{id}/dismiss` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/search` | main | yes | automatic | `none` | `handle_search_memory` | SearchResult.content, SearchResult.content_hash, SearchResult.last |
| `GET` | `/api/memory/stats` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/store` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/unconfirmed` | main | yes | automatic | `none` | `handle_list_unconfirmed_memories` | RecentActivityItem.title/snippet via revision card |
| `POST` | `/api/memory/{id}/correct` | main | yes | automatic | `none` | `handle_correct_memory` | opaque response type — fail-closed |
| `GET` | `/api/memory/{id}/detail` | main | yes | automatic | `none` | `handle_get_memory_detail` | MemoryItem.title/content via dismissed card |
| `POST` | `/api/memory/{id}/pin` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/{id}/revisions` | main | yes | automatic | `none` | `handle_get_memory_revisions` | MemoryRevisionEntry.title/content_preview via card |
| `PUT` | `/api/memory/{id}/stability` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/{id}/unpin` | main | no | not_applicable | `none` | — | no prose fields |
| `PUT` | `/api/memory/{id}/update` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/memory/{id}/update-page` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/memory/{id}/versions` | main | yes | automatic | `none` | `handle_get_version_chain` | MemoryVersionItem.title/content via revision card |
| `GET` | `/api/memory/{source_id}/enrichment-status` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/on-device-model` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/on-device-model/download` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/onboarding/milestones` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/onboarding/milestones/{id}/acknowledge` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/onboarding/reset` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/pages` | main | yes | automatic | **`collection`** | `handle_list_pages` | opaque response type — fail-closed |
| `POST` | `/api/pages` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/pages/export` | main | yes | automatic | `none` | `handle_export_pages` | EFFECT: writes page prose to the requested vault |
| `GET` | `/api/pages/orphan-links` | main | yes | automatic | `none` | `handle_list_orphan_links` | OrphanLink.label, OrphanLinksResponse.orphan_labels |
| `GET` | `/api/pages/recent` | main | yes | automatic | `none` | `handle_recent_pages` | RecentActivityItem.snippet, RecentActivityItem.title; NOT Collection — carries prose and no axes |
| `GET` | `/api/pages/recent-changes` | main | yes | automatic | `none` | `handle_recent_page_changes` | PageChange.title; NOT Collection — carries no axes |
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
| `GET` | `/api/profile/narrative` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/profile/narrative/regenerate` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
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
| `GET` | `/api/snapshots` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/snapshots/{id}/captures` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `GET` | `/api/snapshots/{id}/captures-with-content` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/snapshots/{id}/delete` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/sources` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/sources` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/sources/{id}` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/sources/{id}/sync` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/spaces` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/spaces` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `DELETE` | `/api/spaces/default` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/spaces/default` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `PUT` | `/api/spaces/default` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/spaces/reorder` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{from}/move-to/{to}` | main | yes | automatic | `none` | `handle_move_space` | opaque response type — fail-closed |
| `DELETE` | `/api/spaces/{name}` | main | yes | automatic | `none` | `handle_delete_space` | opaque response type — fail-closed |
| `PUT` | `/api/spaces/{name}` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |
| `POST` | `/api/spaces/{name}/confirm` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{name}/pin` | main | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/spaces/{name}/star` | main | yes | automatic | `none` | `handle_toggle_space_starred` | opaque response type — fail-closed |
| `GET` | `/api/status` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/status` | repair | no | not_applicable | `none` | — | no prose fields |
| `POST` | `/api/steep` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/suggest-tags` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/api/tags` | main | no | not_applicable | `none` | — | no prose fields |
| `DELETE` | `/api/tags/{name}` | main | no | not_applicable | `none` | — | no prose fields |
| `GET` | `/ws/updates` | main | no | not_applicable | `none` | — | DEMOTED — proof in the inventory doc |

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
| `wenlan pages` | yes | automatic | **`collection` + `named_page`** | enforce_projection_directory_invariant |
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

Four drafts of this section were wrong in four different ways, and the sequence
is worth more than any of the numbers. The fourth is the one to read: it was
produced by a committed generator, which is the fix for the first three, and it
was still wrong — because nothing checked the generator.

| Draft | Claim | Why it was wrong |
|---|---|---|
| 1 | four categories, enumeration deferred to PR-B | the categories named modules that hold no page reader at all |
| 2 | 76 call sites | bodies delimited by the next `fn` match, so neighbours merged; `pages` and prose matched anywhere in the body, so `list_tags_scoped` and `tally` counted |
| 3 | 54 SQL-bearing definitions | correct as far as it went, and **the wrong set** — see below |
| 4 | 186 readers / 28 exposure, from `scripts/m5-reader-sweep.py` | the generator's `#[cfg(test)]` stripper armed its brace scan on `#[cfg(test)] mod foo;` — an attribute over an item that never opens a brace — and blanked forward to the next unrelated `{`, hiding real code from the sweep |
| 5 | this one, 190 / 22, from the same script with a self-check | the number is whatever the script prints, and the script now proves it can see |

**Draft 4's error is the one that matters most**, because it was invisible.
Drafts 1-3 were wrong in ways a careful reader could argue with; draft 4 was
wrong in a way no reader could see, because the generator silently deleted the
code before counting it. What it hid was not incidental: five HTTP route
handlers, `handle_update_page` and `handle_refresh_page` among them — the page
prose write path, exactly what the manifest governs. A generator is only as
trustworthy as its input filter, and that filter had no check. It has one now
(`python3 scripts/m5-reader-sweep.py --selftest`), asserting both that a
`#[cfg(test)] mod foo;` does not swallow the code after it and that a real
`#[cfg(test)]` block still gets stripped. The exposure count also fell 28 → 22
because a name-keyed caller scan was reporting each function's own definition
line as an external caller of itself.

**Draft 3's error is the instructive one.** SQL-bearing definitions are where
page prose is *materialized*, not where it is *read*. `get_page_inner` is
private (`db.rs:41739`); the reader the governing spec asks for is the citation
backfill that calls `get_page` and hands `page.content` to the LLM
(`citations.rs:423`). A manifest of sources answers a question nobody asked.
"An executable manifest from actual callers" (`kg-m5-goal-prompt.md:118`) means
the callers.

The generator therefore expands the SQL layer through a **transitive caller
closure**, and reports the depth so the layers stay distinguishable rather than
being flattened into one number:

| Depth | What it is | Count |
|---|---|---|
| 0 | SQL-bearing definitions — where prose is materialized | 54 |
| 1 | wrapper layer — `get_page` over `get_page_inner` | 50 |
| 2 | consumers — what the spec asks for | 86 |
| | **total** | **190** |

Of these, **22** are exposure paths (`pub` and called from outside
`wenlan-core`) and **9** are name-ambiguous, excluded from the exposure set
rather than given caller edges a name scan cannot attribute.

### What this enumeration is not

- **Not LSP-resolved.** Edges come from name matching, which over-matches on
  generic names and under-matches through trait dispatch and re-exports. PR-B
  re-resolves with the language server and re-runs this generator, asserting set
  equality on the rows and the partition.
- **Not depth-unbounded.** The closure stops at depth 2. A consumer three hops
  from the SQL is not listed. Depth is a stated parameter of the generator, not
  a claim that nothing lies past it.
- **Not a safety property.** Internal-only means no *current* caller crosses the
  crate boundary, which one `pub` re-export changes.

Every row is `marker_shape = none`. There is no caller to gesture.

### The differential check that proved nothing

An earlier commit added two independently written `#[cfg(test)]` strippers and
recorded that both returned 76. They shared the same broken body delimitation,
so they confirmed the bug in stereo. **A differential oracle that varies the
wrong dimension is not evidence** — it is a second reading of the same mistake,
carrying the authority of agreement.

<!-- m5-reader-sweep:begin -->

### Depth 0 — SQL-bearing definitions

| Address | Function | Visibility | Exposure |
|---|---|---|---|
| `core/db.rs:3625` | `run_migrations` | `pub` | internal-only |
| `core/db.rs:9117` | `migrate_89_page_kind_fold` | `private` | internal-only |
| `core/db.rs:17057` | `reconcile_entity_page_parity` | `pub` | **exposure** — `server/scheduler.rs:2237` |
| `core/db.rs:26686` | `rebind_source_page_in_transaction` | `private` | internal-only |
| `core/db.rs:34325` | `oldest_active_page` | `pub` | internal-only |
| `core/db.rs:39521` | `list_recent_retrievals` | `pub` | internal-only |
| `core/db.rs:39661` | `list_recent_retrievals_scoped` | `pub` | **exposure** — `server/routes.rs:910` |
| `core/db.rs:39867` | `list_recent_changes` | `pub` | internal-only |
| `core/db.rs:40170` | `list_recent_pages_with_badges` | `pub` | internal-only |
| `core/db.rs:41083` | `append_page_history` | `private` | internal-only |
| `core/db.rs:41244` | `insert_page_with_kind_inner` | `private` | internal-only |
| `core/db.rs:41781` | `get_page_inner` | `private` | internal-only |
| `core/db.rs:41810` | `get_page_by_entity` | `pub` | internal-only |
| `core/db.rs:41885` | `list_pages_inner` | `private` | internal-only |
| `core/db.rs:41918` | `list_pages_stale` | `pub` | internal-only |
| `core/db.rs:41963` | `list_pages_by_space` | `pub` | internal-only |
| `core/db.rs:43024` | `find_matching_page` | `pub` | internal-only |
| `core/db.rs:43081` | `find_matching_page_scoped` | `pub` | internal-only |
| `core/db.rs:43448` | `page_merge_row` | `private` | internal-only |
| `core/db.rs:43497` | `load_page_source_index` | `pub` | **exposure** — `server/routes.rs:655` |
| `core/db.rs:43631` | `list_active_page_titles_scoped` | `pub` | internal-only |
| `core/db.rs:43678` | `list_relevant_active_page_titles` | `pub` | internal-only |
| `core/db.rs:43745` | `find_active_page_id_by_title` | `pub` | internal-only |
| `core/db.rs:43780` | `find_unique_active_page_id_by_title_scoped` | `pub` | internal-only |
| `core/db.rs:44331` | `backfill_page_embeddings` | `pub` | internal-only |
| `core/db.rs:45409` | `get_pages_for_memory` | `pub` | internal-only |
| `core/db.rs:46308` | `get_stale_page_after` | `pub` | internal-only |
| `core/db.rs:46345` | `list_stale_pages_scoped` | `pub` | **exposure** — `server/routes.rs:734` |
| `core/db.rs:46385` | `find_stale_archived_pages` | `pub` | **exposure** — `server/cmd_backfill.rs:49` |
| `core/db/maintenance_duplicate_reads.rs:35` | `embedding_near_duplicate_pairs` | `pub(crate)` | internal-only |
| `core/db/maintenance_duplicate_reads.rs:108` | `scan_near_duplicate_slice` | `pub(crate)` | internal-only |
| `core/db/maintenance_retro_scan.rs:17` | `scan_automatic_retro_stub_slice` | `pub(crate)` | internal-only |
| `core/db/repair_deterministic.rs:7` | `apply_deterministic_repair_cas` | `pub` | internal-only |
| `core/db/repair_page_rename.rs:117` | `rename_page_title_cas_inner` | `pub(crate)` | internal-only |
| `core/db/repair_page_rename.rs:717` | `page_on_connection` | `private` | internal-only |
| `core/db/scoped_entities.rs:12` | `list_entities_scoped` | `pub` | **exposure** — `server/memory_routes.rs:1456` |
| `core/db/scoped_entities.rs:84` | `get_entity_detail_scoped` | `pub` | **exposure** — `server/memory_routes.rs:1472` |
| `core/db/scoped_entities.rs:291` | `list_recent_relations_scoped` | `pub` | **exposure** — `server/knowledge_routes.rs:67` |
| `core/db/scoped_entities.rs:615` | `search_entities_by_vector_scoped` | `pub` | **exposure** — `server/memory_routes.rs:1507` |
| `core/db/scoped_pages.rs:391` | `list_recent_changes_scoped` | `pub` | **exposure** — `server/routes.rs:974` |
| `core/lint/deep.rs:221` | `page_duplicates` | `private` | internal-only |
| `core/lint/deep.rs:299` | `page_body_result` | `private` | internal-only |
| `core/lint/pages/db_checks.rs:233` | `load_rows` | `private` | internal-only |
| `core/lint/pages/link_checks/orphans.rs:8` | `load` | `pub(super)` | name-ambiguous |
| `core/lint/semantic_candidates.rs:846` | `load_pages` | `private` | name-ambiguous |
| `core/lint/serving/query.rs:16` | `load` | `pub(super)` | name-ambiguous |
| `core/repair.rs:1500` | `prepare_rename_page_title` | `private` | internal-only |
| `core/repair.rs:3510` | `capture_rename_page_row_on_snapshot` | `private` | internal-only |
| `core/repair.rs:3739` | `validate_rename_page_title_collision_on_snapshot` | `private` | internal-only |
| `core/repair.rs:3783` | `validate_rename_page_title_collision_on_connection` | `pub(crate)` | internal-only |
| `core/repair.rs:5725` | `projection_page_receipt_sql` | `private` | internal-only |
| `core/repair_plan/deterministic.rs:96` | `resolve_duplicate_page_titles` | `private` | internal-only |
| `core/repair_plan/deterministic.rs:270` | `renamed_page_title_still_actionable` | `private` | internal-only |
| `core/repair_plan/deterministic.rs:420` | `resolve_source_pages` | `private` | internal-only |
| `core/repair_plan/deterministic.rs:1147` | `resolve_orphan_links` | `private` | internal-only |

### Depth 1 — wrapper layer

| Address | Function | Visibility | Reaches prose via |
|---|---|---|---|
| `core/db.rs:3286` | `new` | `pub` | `run_migrations` |
| `core/db.rs:3498` | `new_with_shared_embedder` | `pub` | `run_migrations` |
| `core/db.rs:24148` | `augment_with_graph_gated` | `private` | `search_entities_by_vector_scoped` |
| `core/db.rs:26226` | `rebind_source_id_inner` | `private` | `rebind_source_page_in_transaction` |
| `core/db.rs:41174` | `insert_page_with_kind` | `pub(crate)` | `insert_page_with_kind_inner` |
| `core/db.rs:41213` | `insert_document_source_page_at_hash` | `pub(crate)` | `insert_page_with_kind_inner` |
| `core/db.rs:41487` | `replace_source_page_inner` | `private` | `append_page_history` |
| `core/db.rs:41768` | `get_page` | `pub` | `get_page_inner` |
| `core/db.rs:41777` | `get_page_browse` | `pub` | `get_page_inner` |
| `core/db.rs:41863` | `list_pages` | `pub` | `list_pages_inner` |
| `core/db.rs:41876` | `list_pages_browse` | `pub` | `list_pages_inner` |
| `core/db.rs:42369` | `try_update_page_content` | `private` | `append_page_history` |
| `core/db.rs:43253` | `accept_page_merge` | `pub` | `page_merge_row` |
| `core/db.rs:43485` | `find_best_overlapping_page` | `pub` | `load_page_source_index` |
| `core/db.rs:43618` | `list_active_page_titles` | `pub` | `list_active_page_titles_scoped` |
| `core/db.rs:44059` | `resolve_orphan_page_links` | `pub` | `find_unique_active_page_id_by_title_scoped` |
| `core/db.rs:46298` | `list_stale_pages` | `pub` | `list_stale_pages_scoped` |
| `core/db/repair_page_rename.rs:610` | `rename_page_projection_matches_post` | `private` | `page_on_connection` |
| `core/db/scoped_pages.rs:280` | `list_recent_pages_with_badges_scoped` | `pub` | `list_recent_pages_with_badges` |
| `core/lint/deep.rs:30` | `run` | `pub(super)` | `page_body_result` |
| `core/lint/pages/db_checks.rs:54` | `run` | `pub(crate)` | `load_rows` |
| `core/maintenance.rs:240` | `run_maintenance_stage_slice` | `pub` | `get_stale_page_after` |
| `core/maintenance/duplicates.rs:163` | `detect_near_duplicate_pages_inner` | `private` | `embedding_near_duplicate_pairs` |
| `core/onboarding.rs:90` | `check_after_refinery_pass` | `pub` | `oldest_active_page` |
| `core/page_map_improve.rs:253` | `source_suggestions` | `private` | `find_active_page_id_by_title` |
| `core/post_ingest.rs:128` | `run_page_growth_slice` | `pub` | `find_matching_page_scoped` |
| `core/post_ingest.rs:822` | `grow_page` | `pub(crate)` | `find_matching_page_scoped` |
| `core/post_write.rs:207` | `rename_page_title_cas` | `pub(crate)` | `rename_page_title_cas_inner` |
| `core/post_write.rs:1165` | `create_page_impl` | `private` | `find_matching_page_scoped` |
| `core/refinery/mod.rs:1570` | `run_redistill_page_slice` | `pub` | `get_stale_page_after` |
| `core/repair.rs:1370` | `prepare_memory_reclassification_with_pages` | `pub` | `prepare_rename_page_title` |
| `core/repair.rs:2097` | `apply_repair_with_pages_inner` | `private` | `apply_deterministic_repair_cas` |
| `core/repair.rs:5690` | `projection_page_row_from_snapshot` | `private` | `projection_page_receipt_sql` |
| `core/repair.rs:5709` | `projection_page_row_from_connection` | `private` | `projection_page_receipt_sql` |
| `core/repair_plan/deterministic.rs:76` | `resolve_current` | `pub(crate)` | `resolve_duplicate_page_titles` |
| `core/repair_plan/deterministic.rs:214` | `target_still_actionable` | `pub(super)` | `renamed_page_title_still_actionable` |
| `core/synthesis/detect.rs:18` | `detect_page_candidates` | `pub` | `find_matching_page_scoped` |
| `core/synthesis/distill.rs:81` | `build_existing_titles_hint` | `pub(crate)` | `list_active_page_titles_scoped` |
| `core/synthesis/distill.rs:464` | `distill_one_cluster_with_tuning` | `private` | `find_matching_page_scoped` |
| `core/synthesis/overview.rs:70` | `ensure_overview_page` | `private` | `find_active_page_id_by_title` |
| `core/synthesis/wikilinks.rs:71` | `resolve_against_pages` | `pub` | `find_unique_active_page_id_by_title_scoped` |
| `server/cmd_backfill.rs:19` | `run` | `pub` | `find_stale_archived_pages` |
| `server/knowledge_routes.rs:54` | `handle_list_recent_relations` | `pub` | `list_recent_relations_scoped` |
| `server/memory_routes.rs:1443` | `handle_list_entities` | `pub` | `list_entities_scoped` |
| `server/memory_routes.rs:1462` | `handle_get_entity_detail` | `pub` | `get_entity_detail_scoped` |
| `server/memory_routes.rs:1494` | `handle_search_entities` | `pub` | `search_entities_by_vector_scoped` |
| `server/routes.rs:501` | `handle_distill` | `pub` | `list_stale_pages_scoped` |
| `server/routes.rs:896` | `handle_recent_retrievals` | `pub` | `list_recent_retrievals_scoped` |
| `server/routes.rs:960` | `handle_recent_page_changes` | `pub` | `list_recent_changes_scoped` |
| `server/scheduler.rs:2025` | `run_ambient_job` | `private` | `reconcile_entity_page_parity` |

### Depth 2 — consumers

| Address | Function | Visibility | Reaches prose via |
|---|---|---|---|
| `core/citations.rs:419` | `run_citation_backfill_with_page_limit` | `private` | `get_page` |
| `core/db.rs:21793` | `search_memory_with_cue` | `private` | `augment_with_graph_gated` |
| `core/db.rs:24131` | `augment_with_graph` | `pub` | `augment_with_graph_gated` |
| `core/db.rs:24681` | `augment_with_graph_seeded_scoped` | `private` | `augment_with_graph_gated` |
| `core/db.rs:26196` | `rebind_source_id` | `pub` | `rebind_source_id_inner` |
| `core/db.rs:26209` | `rebind_source_id_with_source_page` | `pub` | `rebind_source_id_inner` |
| `core/db.rs:34316` | `first_active_page` | `pub` | `list_pages` |
| `core/db.rs:41136` | `insert_page` | `pub(crate)` | `insert_page_with_kind` |
| `core/db.rs:41431` | `replace_source_page` | `pub(crate)` | `replace_source_page_inner` |
| `core/db.rs:41456` | `replace_source_page_at_document_hash` | `pub(crate)` | `replace_source_page_inner` |
| `core/db.rs:41830` | `find_page_by_source_memory` | `pub` | `get_page` |
| `core/db.rs:42119` | `update_page_content` | `pub` | `try_update_page_content` |
| `core/db.rs:42152` | `try_update_page_content_if_stale` | `pub` | `try_update_page_content` |
| `core/db.rs:42178` | `try_update_page_content_with_changelog_at_source_revision` | `pub` | `try_update_page_content` |
| `core/db.rs:42226` | `try_update_page_content_with_changelog` | `pub` | `try_update_page_content` |
| `core/db.rs:42260` | `try_update_page_content_with_changelog_at_version` | `pub` | `try_update_page_content` |
| `core/db.rs:42295` | `try_update_page_growth_at_versions` | `pub` | `try_update_page_content` |
| `core/db.rs:42333` | `try_accept_page_revision` | `pub(crate)` | `try_update_page_content` |
| `core/db.rs:43007` | `refresh_page_wikilinks` | `pub` | `get_page` |
| `core/db.rs:43537` | `max_page_overlap` | `pub` | `find_best_overlapping_page` |
| `core/db/repair_page_regenerate.rs:123` | `regenerate_page_projection_cas` | `pub(crate)` | `get_page` |
| `core/db/repair_page_rename.rs:400` | `recover_rename_page_title_apply_receipt` | `pub(crate)` | `rename_page_projection_matches_post` |
| `core/db/scoped_pages.rs:472` | `list_pages_scoped_inner` | `private` | `list_pages` |
| `core/db/scoped_pages.rs:693` | `get_page_scoped_inner` | `private` | `get_page` |
| `core/document_enrichment.rs:669` | `write_document_source_page` | `private` | `get_page` |
| `core/eval/answer_quality.rs:138` | `run_e2e_answer_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/answer_quality.rs:407` | `run_e2e_locomo_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/answer_quality.rs:879` | `run_e2e_context_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/answer_quality.rs:1015` | `run_e2e_context_eval_longmemeval` | `pub` | `new_with_shared_embedder` |
| `core/eval/answer_quality.rs:3006` | `run_fullpipeline_lme` | `pub` | `new_with_shared_embedder` |
| `core/eval/context_path.rs:134` | `run_context_path_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/context_path.rs:376` | `run_context_path_eval_longmemeval` | `pub` | `new_with_shared_embedder` |
| `core/eval/longmemeval.rs:1767` | `run_longmemeval_rerank_pool_probe` | `pub` | `new_with_shared_embedder` |
| `core/eval/pipeline.rs:336` | `run_locomo_pipeline_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/pipeline.rs:654` | `run_longmemeval_pipeline_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:353` | `run_quality_cost_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:659` | `run_multi_turn_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:788` | `run_native_memory_augmentation` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:1039` | `run_pipeline_token_eval_simulated` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:1504` | `run_quality_at_scale_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:1665` | `run_scaling_eval` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval.rs:1879` | `run_memory_layer_comparison` | `pub` | `new_with_shared_embedder` |
| `core/eval/retrieval_drift.rs:86` | `capture_rankings` | `pub` | `new_with_shared_embedder` |
| `core/eval/shared.rs:655` | `open_or_seed_scenario_db` | `pub` | `new_with_shared_embedder` |
| `core/maintenance.rs:128` | `run_maintenance_tick` | `pub` | `list_stale_pages` |
| `core/maintenance.rs:656` | `list_distilled_stub_pages` | `private` | `list_pages` |
| `core/maintenance/duplicates.rs:148` | `detect_near_duplicate_pages` | `pub(super)` | `detect_near_duplicate_pages_inner` |
| `core/maintenance/duplicates.rs:156` | `detect_all_near_duplicate_pages` | `pub(super)` | `detect_near_duplicate_pages_inner` |
| `core/maintenance/duplicates.rs:238` | `list_page_source_sets` | `private` | `list_pages` |
| `core/maintenance/page_merge_order.rs:44` | `load_candidate` | `private` | `get_page` |
| `core/page_map_improve.rs:210` | `improve_page_map` | `pub` | `get_page` |
| `core/page_map_improve.rs:360` | `run_proactive_page_maps` | `pub` | `list_pages` |
| `core/post_ingest.rs:391` | `run_post_ingest_enrichment` | `pub` | `grow_page` |
| `core/post_write.rs:377` | `page_write` | `pub` | `create_page_impl` |
| `core/post_write.rs:501` | `write_document_source_page_impl` | `private` | `insert_document_source_page_at_hash` |
| `core/post_write.rs:583` | `replace_source_page_impl` | `private` | `get_page` |
| `core/post_write.rs:1937` | `update_page_impl` | `private` | `get_page` |
| `core/post_write.rs:2508` | `accept_page_revision_card` | `private` | `get_page` |
| `core/refinery/mod.rs:639` | `run_periodic_steep_phase_with_api` | `pub` | `run_redistill_page_slice` |
| `core/refinery/mod.rs:741` | `run_periodic_steep_with_api_scope` | `private` | `check_after_refinery_pass` |
| `core/refinery/mod.rs:1393` | `enqueue_changed_pages` | `pub(crate)` | `list_pages` |
| `core/refinery/mod.rs:1422` | `re_distill_stale_pages` | `pub(crate)` | `list_stale_pages` |
| `core/repair.rs:1361` | `prepare_memory_reclassification` | `pub` | `prepare_memory_reclassification_with_pages` |
| `core/repair.rs:2060` | `apply_repair_with_pages` | `pub` | `apply_repair_with_pages_inner` |
| `core/repair.rs:2372` | `apply_rename_page_title` | `private` | `rename_page_title_cas` |
| `core/repair.rs:5411` | `capture_page_projection_rollback` | `pub(crate)` | `projection_page_row_from_snapshot` |
| `core/repair.rs:5551` | `projection_page_row_on_connection` | `pub(crate)` | `projection_page_row_from_connection` |
| `core/repair_plan.rs:47` | `deterministic_target_still_actionable` | `pub(crate)` | `target_still_actionable` |
| `core/sources/page_watcher.rs:126` | `sync_one_file` | `private` | `get_page` |
| `core/synthesis/distill.rs:129` | `build_page_compile_user_prompt` | `pub(crate)` | `build_existing_titles_hint` |
| `core/synthesis/distill.rs:453` | `distill_one_cluster` | `pub` | `distill_one_cluster_with_tuning` |
| `core/synthesis/distill.rs:993` | `distill_pages_scoped_gated` | `pub(crate)` | `distill_one_cluster_with_tuning` |
| `core/synthesis/distill.rs:1236` | `refresh_page_with_prompt` | `pub(crate)` | `get_page` |
| `core/synthesis/overview.rs:40` | `top_page_source_ids` | `private` | `list_pages` |
| `core/synthesis/overview.rs:104` | `refresh_overview_page` | `pub` | `ensure_overview_page` |
| `core/synthesis/refinement_queue.rs:125` | `apply_refinement_with_decision` | `pub` | `accept_page_merge` |
| `server/main.rs:1059` | `run_daemon` | `private` | `list_pages` |
| `server/memory_routes.rs:2184` | `handle_create_page` | `pub` | `get_page` |
| `server/memory_routes.rs:3727` | `handle_update_page` | `pub` | `get_page` |
| `server/memory_routes.rs:3802` | `handle_refresh_page` | `pub` | `get_page` |
| `server/page_map_routes.rs:49` | `visible_page` | `private` | `get_page` |
| `server/page_map_routes.rs:69` | `ensure_page_is_active` | `private` | `get_page` |
| `server/repair_routes.rs:183` | `handle_prepare` | `private` | `prepare_memory_reclassification_with_pages` |
| `server/routes.rs:991` | `handle_recent_pages` | `pub` | `list_recent_pages_with_badges_scoped` |
| `server/scheduler.rs:1974` | `run_ambient_job_safe` | `private` | `run_ambient_job` |
| `server/scheduler.rs:2326` | `fire_maintenance_stage_safe` | `private` | `run_maintenance_stage_slice` |

<!-- m5-reader-sweep:end -->

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
   | call-site triples | **167** | rows in this table: 160 `router.rs` + 5 `repair_routes.rs` + 2 `lint_routes.rs` |
   | runtime builder triples | **171** | `(builder, method, path)` pairs actually installed: 165 `main` + 6 `repair` |

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
    internal-reader tables at every depth — the generator is the predicate, so
    this is a real positive control rather than a second reading of prose. It
    also asserts the brace scan never hits `MAX_FN_LINES`, so a future
    unbalanced brace in a string cannot silently truncate a body;
11. asserts the exposure partition with the **language server**, not a name
    scan: every internal-only row is either not `pub` or has no caller outside
    `wenlan-core`. A row that gains an outside caller fails until it is moved to
    the exposure table and given an adapter;
12. asserts every `collection`-shaped route's item type can carry a page
    identity **and** both truth axes — the check that keeps
    `OrphanLink { label, count }`-shaped payloads off the allowlist;
13. asserts every marked call writes a durable audit row carrying caller
    identity, the page IDs named, and a timestamp — and that removing the write
    goes RED. The audit record is the sole compensating control for the
    conceded `collection` + `named_page` composition attack, so an untested one
    is a sentence, not a control;
14. sentinel test: seed a provisional page, drive every error path that names
    it, and assert its title and prose appear in **no** error body. This is the
    single error-seam invariant, not a per-route classification.

Checks 2 and 3 are the positive controls: 2 keeps the row set live, 3 keeps the
evidence live. Without both, this file is a snapshot that rots.

#### Known non-HTTP disclosure surfaces (out of scope for check 14)

Check 14's invariant is scoped to the HTTP **error response body**. Three
success-path sites put a page title somewhere durable that is not an error body,
found while auditing the seam and deliberately left alone:

| site | what it writes | why out of scope |
|---|---|---|
| `core/post_write.rs:2918` (`create_page_impl`) | `title={req.title}` into `log_agent_activity`'s `detail` | activity-log row, `Ok` path only |
| `core/post_write.rs:3088` (`stage_page_revision_card`) | `format!("Revision: {}", page.title)` into a staged `RawDocument` | staged row, `Gated` success path only |
| `core/export/obsidian.rs:44` | `log::warn!` naming a page that failed to export | process log, never reaches the wire |

None is reachable through an error response, so none can be driven by a caller
holding no truth grant. If the truth contract later extends to durable
side-channels, these are the known starting set.

Read "out of scope" narrowly: it means *not an error body*, not *not a
disclosure*. The first two rows were later found to be HTTP-reachable by other
routes — that is what promoted `/api/activities` and `/api/chunks/{source_id}`
to page-bearing, and then twelve more readers at the provenance re-audit, every
one of them reaching row two. Being absent from the error seam says nothing
about the read side.

##### Background re-distillation sends page titles to the configured LLM
##### provider (found by cross-model review, 2026-07-28)

A fourth surface, and the only one that leaves the machine. It is not an HTTP
route, so the manifest does not cover it and neither gate sits in its path:

```
scheduler.rs:2494  run_periodic_steep_phase_with_api   (timer, no request)
  -> refinery/mod.rs:1583  get_stale_page_after("source_updated", ...)
       selects stale/active pages directly, no truth predicate
  -> synthesis/distill.rs:1309  build_page_compile_user_prompt(db, &page.title, ..)
       -> :85  list_relevant_active_page_titles(...)   other titles, as a hint
  -> llm.generate(..)   provider from refinery/mod.rs:326 resolve_synthesis,
                        which may be the Anthropic slot or an external endpoint
```

So post-cutover a background timer can hand an unsupported page's title — plus
a hint list of other active titles — to a remote provider, with no HTTP
middleware and no `page_visibility` anywhere in the path. Failure logging in
`synthesis/distill.rs` names the titles too.

Inert today only because nothing is hidden at generation 0. PR-C must decide
whether re-distillation of an unsupported page is permitted at all; if it is,
the permit has to be consumed inside the refinery, not at the HTTP edge, since
there is no request to attach a grant to.

## What PR-C must own before the ceremony (cross-model review, 2026-07-28)

A read-only Codex (gpt-5.6-sol, xhigh) review of the PR-B branch returned
**BLOCK** — not on PR-B's safety, which is inert at generation 0, but on the
claim that advancing the generation activates the contract. It does not. Every
item below was verified at source before being recorded here.

### The asymmetry, stated plainly

The destructive half of this contract is wired and the protective half is not.
`page_visibility` has exactly one production caller — the projection invariant,
which deletes `.md` files. No HTTP adapter reads the grant the guard resolves;
`select_visible_pages` filters scope, trust tier and `kind`, never truth state.
Advancing the generation today would evict pages from the user's vault while
`/api/pages`, `/api/pages/{id}`, `/api/pages/search` and both export routes kept
serving them. Adapters land BEFORE the ceremony, not with it. The warning now
also sits on `set_truth_cutover_generation` itself, where the trigger is.

### Prerequisites

**All eight closed by PR-C**, whose design and per-item reasoning live in
`docs/plans/2026-07-28-m5-prc-adapters.md`. Kept here struck through rather than
deleted, so the review that produced them stays legible next to the table it was
about.

| # | what | closed by |
|---|---|---|
| 1 | ~~Adapters that consume the resolved grant, so hiding actually hides~~ | four shared operations in `wenlan-core/src/truth_adapter.rs`, consumed at every HTTP page reader; `TruthView` is now a `FromRequestParts` extractor |
| 2 | ~~Stop trusting `state.json` as the page enumeration~~ | the pass enumerates the directory by frontmatter `origin_id` and evicts by scanned filename, never through `remove_page` |
| 3 | ~~Both a removal pass and a write-time skip~~ | `write_page_gated` / `write_page_permitted` on both projection-write types |
| 4 | ~~Decide whether a failed invariant may serve traffic~~ | at generation ≥ 1 (or an unreadable generation) a failed pass aborts startup; at 0 it stays a logged error |
| 5 | ~~Flip the audit write to fail-closed for grants~~ | a grant that could not be recorded is refused; automatic and refused outcomes stay best-effort |
| 6 | ~~Rule on re-distillation of unsupported pages~~ | `page_write_permit` on the ambient re-distill op, `filter_page_refs` on the title hint; the explicit `POST /api/distill/{id}` keeps its own caller grant |
| 7 | ~~A tooth on the projection pass's wiring~~ | a source scan asserts the `main.rs` call site exists |
| 8 | ~~Re-audit the demotions never checked for provenance~~ | all 33 re-checked, 12 reversed — see "Demotions reversed (2026-07-28, provenance re-audit)" |

Closing them did **not** produce a clean bill. The re-audit for #8 found a live
disclosure bug rather than a bookkeeping error, and PR-C leaves three named
items to the ceremony: `stage_page_revision_card` is still ungated, the page
write fence is unbuilt, and page-map hiding is unresolved as a graph
transformation. Those are listed under "What PR-C does not do" in its own
document, and the ceremony is not safe to run until they are answered.

On #2: `load_state` returns `KnowledgeState::default()` on any read error
(`knowledge.rs:573`) and `parse_state` falls back to `unwrap_or_default()` on
malformed JSON (`:630`). Either yields an empty page map, so the invariant
returns `Ok(0)` and reports success while every `.md` file stays on disk — and
`wenlan pages` enumerates that directory independently of `state.json`. Since
state saves truncate before writing, a crash mid-save is a realistic way to
reach it.

### Generation 0 is inert in effect, not in every observable

Worth stating because the branch was described as changing nothing. The guard
does not consult the generation, so from this branch onward a request carrying
`x-wenlan-reader-intent` gets an audit row, and can get a 403 on a
`MarkerShape::None` route — headers that were ignored before. Migration 101 also
creates its table and advances `user_version` on first start. No caller that
does not send the new headers sees any change, which is the property that
matters, but "nothing changes" was too strong.
