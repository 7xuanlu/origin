# M5 PR-C — the protecting half

PR-B landed the M5 truth-exposure substrate inert at `truth_cutover_generation = 0`
(squash `f29c2c54`, PR #408). A cross-model review returned BLOCK on the cutover,
correctly, and the finding is the reason this document exists:

> The destructive half of the contract is wired and the protective half is not.
> `page_visibility` has exactly one production caller — the projection invariant,
> which **deletes** the `.md` file of every page the verdict hides. No HTTP
> adapter reads the grant the guard resolves. Advancing the generation today
> would evict pages from the user's vault while every page route kept serving
> them.

PR-C is the protecting half. It ships **still inert** — the generation stays 0 —
because the ordering is the whole point: adapters land before the ceremony, never
alongside it.

The eight prerequisites recorded in
`docs/plans/2026-07-27-m5-reader-manifest-inventory.md` are this document's scope.
Each section below closes one.

## 1. Adapters that consume the resolved grant

### The seam, and why it is the DB and not the handler

A response-layer redactor cannot work: `/api/context` flattens `Page` into a
string and drops the ID before the response exists
(`crates/wenlan-server/src/routes.rs:416`), and both export handlers write prose
to disk before returning a JSON receipt. The filter has to run while the pages
are still typed.

So the adapters are **four shared operations** in
`crates/wenlan-core/src/truth_adapter.rs`, each taking the resolved grant:

| operation | input → output | used by |
|---|---|---|
| `filter_pages` | `Vec<Page>` → `Vec<Page>` | every list/search reader |
| `filter_page` | `Option<Page>` → `Option<Page>` | every by-id reader |
| `filter_page_refs` | `Vec<T>` keyed by page id → `Vec<T>` | revisions, links, recent-changes, retrievals, map |
| `page_write_permit` | one page id → `PagePermit` | projection writes, export writes, re-distillation |

`Hidden` drops the item. `EntryOnly` keeps id + title + both truth axes and
strips every prose-bearing field. `Full` is the identity. At generation 0
`page_visibility` answers `Full` for everything, so all four are pass-through and
this PR changes no behavior — the same property PR-B had, now held by more code.

### Both truth axes have to reach the reader

`TruthGrant::CollectionEntries` exists so a human can still find a page after
migration, when every page is provisional. Its precondition is that the entry
carries its state: an entry that appears without both axes is exactly the
unearned trust this rung exists to prevent.

`Page` carried neither axis. `pages.review_status` is **not** one of them — it is
the distillation-faithfulness gate, machine despite the name
(`crates/wenlan-core/src/pages.rs:85`). So `wenlan-types::pages::Page` gains one
field:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub truth: Option<PageTruth>,   // { supported: bool, human_reviewed: bool }
```

`None` on every read path that did not ask for it, which keeps the wire
byte-identical for existing clients. `filter_pages` populates it on the entries
it reduces, and only there.

### Precedence between the two machine axes

`review_status` (`unconfirmed` | `confirmed`, the faithfulness gate) and
`support_status` (`provisional` | `supported`, the claim-entailment gate) are two
machine judgments over "is this content trustworthy" and nothing said which wins.
Settled here: **they are independent and both are floors.** `review_status` keeps
its existing job of ranking pages into context (`select_pages_for_context`);
`support_status` decides visibility. A page that is `confirmed` + `provisional`
is hidden, because passing the faithfulness gate is not evidence that its claims
are entailed. Neither gate may promote past the other.

## 2. `state.json` is not the page enumeration

`load_state` returns `KnowledgeState::default()` on any read error
(`export/knowledge.rs:573`) and `parse_state` falls back to `unwrap_or_default()`
on malformed JSON (`:630`). Either yields an empty map, so the invariant returns
`Ok(0)` — reporting success — while every `.md` file stays on disk and
`wenlan pages` enumerates that directory independently. State saves truncate
before writing, so a crash mid-save reaches it.

The fix is to enumerate what the reader actually reads: **the directory**. The
invariant now takes the union of `state.json`'s keys and the page IDs recovered
from the frontmatter of every `.md` file in the projection directory.

A file that cannot be attributed to a page ID is **never deleted, and never
counted as a failure either** — the projection directory can be the user's own
vault and may hold files Wenlan did not write. The second half matters as much
as the first: section 4 turns a failed pass into a refusal to start at
generation ≥ 1, so treating one unrecognized note as a stuck eviction would let
a stranger's file take the daemon down. It is skipped. Fail closed on the
decision, never on the user's data. Tooth:
`a_file_with_no_origin_id_is_left_alone`.

**The scan reads bytes, not a `String`.** The frontmatter probe reads a bounded
8 KiB prefix, and that bound lands wherever it lands — `read_to_string` fails
the *entire* read when the cap splits a multi-byte character, which any page
carrying CJK prose past 8 KiB will eventually do. The failure was swallowed by
an `.ok()?`, so the file dropped out of the scan: fail **open**, on a disclosure
boundary, and precisely on the path that exists to catch what `state.json`
forgot. It now reads to a `Vec<u8>` and decodes with `from_utf8_lossy` — the key
is ASCII and sits at the head, so a replacement character in the truncated tail
costs nothing. Tooth:
`a_page_with_multibyte_prose_past_the_scan_cap_is_still_evicted`, which drives
three paddings because only one alignment in three splits a 3-byte character at
the cap.

## 3. A write-time skip as well as the removal pass

The removal pass alone holds the invariant only across uptime: pages are
projected at runtime (`post_write.rs:2851`), so after the cutover every newly
written provisional page is readable until the next restart.

`KnowledgeProjectionWrite::write_page` gains an async twin that consults
`page_write_permit` first and declines to project a page the automatic reader may
not see. Removal stays the load-bearing half — migration 99 backfilled every
pre-existing page to `provisional`, so a writer that merely declined new writes
would leave the entire existing corpus readable — but the two together are what
makes the invariant hold continuously rather than at boot.

## 4. A failed invariant may not serve page traffic

Today a failed pass logs `error!` and the daemon serves anyway
(`main.rs:1534`). At generation 0 that is right: the pass removes nothing, so a
failure records the absence of a restriction that is not in force.

At generation ≥ 1 it is wrong. A failure means a file the reader may not see is
still on disk, and `wenlan pages` reads that directory with no wire gate in
front of it — there is no HTTP response to filter. **So at generation ≥ 1 a
failed invariant aborts startup.** The daemon does not open the door it cannot
hold. The cutover is a deliberate ceremony with an operator present, which is
what makes refusing to start an acceptable answer rather than a brick.

An unreadable generation aborts too, same as the guard: `unwrap_or(1)`. The
pass itself is now tested for the first time — the eviction of a file
`state.json` has forgotten, the inertness of the whole pass at generation 0, and
the rule that an unattributable `.md` is skipped rather than deleted or counted
as a failure that would take the daemon down.

## 5. Fail-closed audit for grants

The audit row is the only compensating control for the one attack D3 concedes:
`Collection` and `NamedPage` compose, so a caller willing to forge the marker can
enumerate provisional IDs and fetch them one at a time. Best-effort was a
generation-0 choice — while nothing is hidden, a lost row records the absence of
a restriction that is not in force.

At generation ≥ 1 a **granted** call whose audit row could not be written is
refused. Automatic and refused outcomes stay best-effort: neither issues
exposure, so failing them closed would trade a real availability loss for no
control. `caller` stays unverified — it is the same cooperative tier as the
marker itself, and the row proves a marked call happened, not who made it.

A daemon with no database takes the same branch: it cannot write the row, and it
cannot be asked what generation it is on either, so unknown resolves to refuse.
That is the reachable form of the failure in a test — a live database whose
audit INSERT fails takes the identical branch by construction, but cannot be
provoked without a fault-injection seam the guard does not have, and is not
worth building one for.

## 6. Unsupported pages are not re-distilled

Background re-distillation sends page titles and prose to the configured LLM
provider, which may be external, with neither gate in the path.

It is not an HTTP route, so there is no request to attach a grant to. The permit
is therefore consumed **inside the refinery**: at generation ≥ 1 a page the
automatic reader may not see is not re-distilled, and is not offered as a title
hint for someone else's distillation either. An unsupported page is not a page we
are willing to spend an external round-trip on.

Two seams, both in `synthesis/distill.rs`:

| seam | what it is | gate |
|---|---|---|
| `refresh_page_with_prompt` | the shared re-distill op | `page_write_permit`, scoped to `RefreshReason::SourceChanged` |
| `build_existing_titles_hint` | other active titles fed to the prompt | `filter_page_refs`, one batched query |

The scoping is the judgment call. `refresh_page_with_prompt` serves both the
ambient path and `POST /api/distill/{id}` (`RefreshReason::Explicit`), and only
the ambient one is grantless — the explicit route has a caller and goes through
the wire guard, so gating it here would apply the automatic verdict to a request
that legitimately carries its own. Every ambient caller reaches the gate:
`run_redistill_page_slice`, `re_distill_stale_pages`, both `maintenance.rs`
callers, and the Overview refresh.

The hint seam needed `list_active_page_titles_scoped` and
`list_relevant_active_page_titles` widened to return `(id, title)` — the queries
never selected the page ID, so there was nothing to gate on. The title-only
public wrapper keeps its signature.

## 7. A tooth on the projection pass's wiring

Deleting the `enforce_projection_directory_invariant` call from `main.rs` leaves
every test green today — the pass is tested, its wiring is not. A source-scan
test asserts the call site exists, the same shape as the F5 scan that proves
`set_truth_cutover_generation` has no production caller.

## 8. The under-audited demotions

The Opus review retracted the premise behind 34 manifest demotions ("no
production code path writes page prose into `memories`" is false —
`post_write.rs:3090`). Eight were re-checked; the rest were recorded as
under-audited rather than known correct. They are re-audited under the
provenance question in this PR, and the manifest is corrected where the
re-audit disagrees.

The re-audit did not return a clean bill. It found a live asymmetry, verified at
source: `accept_pending_revision_with_knowledge_path` consumes a page revision
card (`try_update_page_content(consume_revision_id: …)` →
`delete_by_source_id_in_transaction`, `db.rs:43320`, in the same transaction as
the page write), while `dismiss_pending_revision` had no page branch at all and
fell through to the memory path's `UPDATE memories SET pending_revision = 0,
supersedes = NULL` — "unstage, not delete". So **dismissing a page revision card
turned it into a permanent, ordinary, retrievable memory holding a full copy of
the page's prose**, and cleared the very flag most of the demotion evidence
relied on to exclude it.

That is a generation-0 bug, not a cutover one — it corrupts data today, with the
contract entirely inert — so this PR fixes it: page cards delete on dismiss,
mirroring accept. Memory cards keep their unstage semantics, which are correct
for a genuine independent capture that merely topic-matched; a page card is
prose the daemon manufactured from the page itself, so there is no capture to
preserve. Cards already dismissed in a live database are not retroactively
cleaned by this PR.

## 9. Two marker shapes the wire types cannot honour

The inventory's stated rule is that a route qualifies for `collection` only if
its item type can carry a page identity **and both axes**, and it applies that
rule by name to demote `/api/pages/orphan-links`. Two rows were never held to
it: `GET /api/pages/recent` and `GET /api/pages/recent-changes`.
`RecentActivityItem` carries a prose `snippet` and neither axis; `PageChange` is
a page ID, a title, a kind and a timestamp.

Nothing over-exposed. Both adapters are Full-only, so a provisional page was
already dropped rather than reduced — which is exactly what made this worth
fixing while it was still cheap. The defect was not a leak but a **trap**: the
manifest advertised a carve-out the response types cannot represent, and the
obvious way to "fix" the adapters to honour it is to reduce a page into a struct
with nowhere to put the axes it must be reduced *with*. An entry surfacing
without its state is the unearned trust the whole rung exists to prevent.

Both are `none` until those types grow the axes, and `none` refuses rather than
downgrades, so a client that sends a marker there finds out. Tooth:
`truth_guard_test.rs::the_recent_feeds_refuse_a_marker_they_cannot_honour`.

## 10. Five readers the first pass did not gate

Three were missed, one was excused on reasoning that does not hold, and one
landed on `main` while this branch was open:

- **`POST /api/distill`** was excused as "a write path, already gated
  downstream". True of the write, irrelevant to the **response**, which returns
  stale-page `title`, `summary` and `source_memory_ids`, an
  `existing_page_title` from the overlap probe, and the orphan-label feed. The
  suppression stays ungated deliberately — skipping a cluster that overlaps an
  existing page *uses* the knowledge without disclosing it, and removing that
  would make the agent mint duplicates — but the disclosure is gated, and where
  the overlapping page is invisible the reported `new_memory_count` falls back
  to the full cluster size rather than a figure derived from a hidden page's
  sources.
- **`GET /api/pages/orphan-links`** exposes wikilink labels lifted out of page
  **bodies**. The leak is the *source* page, not the target — an orphan label by
  definition names no target. The existing query aggregates `source_page_id`
  away in SQL and applies `HAVING`/`LIMIT` before anything can filter, so
  post-filtering its output is wrong twice over. It gains an unaggregated twin
  and folds counts in Rust, after the truth filter.
- **`GET /api/retrievals/recent`** carries `page_titles` with a `page_ids` list
  documented 1:1 with it. Where they line up the pairs are filtered; where they
  do not — legacy rows recorded before `page_ids` existed — nothing attributes a
  title to a page, and unknown is not permission, so both lists clear.
- **`POST /api/brief`** arrived on `main` mid-branch with the Space Brief
  feature, already carrying `PageBearing::Yes` and an adapter cell naming
  `handle_read_brief`, which did not gate. Its `related_context` comes from
  `search_memory`, which merges the page channel **inline** — a page's prose
  arrives as a `SearchResult` with `source == "page"` and `source_id` the page
  id. The channel is default-OFF behind `WENLAN_ENABLE_PAGE_CHANNEL`, but an
  exposure contract that holds only while a flag is off is not a contract, so
  the gate is unconditional. Only the `source == "page"` rows go to the
  adapter: keying a memory row by its memory id would drop a legitimate result,
  since no page grant covers it. Surviving pages are filtered in place rather
  than partitioned and re-appended, because `search_memory` has already merged
  them into RRF order and a page that survives belongs where that merge put it.

  The same change re-gates `/api/context`, which `main` rewrote in the same
  feature to fetch its pages *through* this handler. The grant travels as a
  parameter rather than being re-derived, so the two routes cannot disagree.

## What PR-C does not do

- **It does not advance the generation.** That is the ceremony, and it comes
  after this. Ordering is the finding this whole PR answers.
- It does not build the fence that holds page writes during the ceremony.
- It does not resolve page-map hiding as a graph transformation; incident edges
  are keyed by node IDs, not page IDs, so a hidden page's edges need their own
  pass.
- **It does not gate `stage_page_revision_card` itself.** While a card is
  staged, the nurture family reads it with no `pending_revision` filter at all
  (`db.rs` ~35086 — and `ORDER BY c.pending_revision DESC` sorts staged cards
  *first*, so this is deliberate product behavior, not an oversight). At
  generation ≥ 1 that surfaces the title and prose of a page the automatic
  reader may not see. The fix is one gate at the single producer — refuse to
  stage a card for a page with no `page_write_permit` — not adapters on the
  twelve memory readers that can carry one. It is a pure generation-≥ 1
  concern, so it belongs to the ceremony PR alongside the other three.
- It does not reconcile `handle_refresh_page`'s md-first rollback with a gated
  write. That handler restores `existing_md_content` on a later DB failure,
  which assumes the md write happened; post-cutover `write_page_gated` may
  return `Ok(None)`. Idempotent and harmless, but the ceremony needs it in view.
- **It does not gate two page-bearing readers that need a design call first.**
  Both carry `PageBearing::Yes` and a named adapter that does not filter, and
  both have carried that since PR-B — unchanged by this PR, and passing
  `page_bearing_rows_carry_an_adapter_address` because that tooth checks the
  cell is an address, not that the function behind it enforces. They are the
  two remaining holes before the generation may advance:
  - `GET /api/activities` (`handle_list_activities`) — a page title really is
    copied into `AgentActivityRow.detail` at write time
    (`post_write.rs` `title={req.title}` into `log_agent_activity`), so the
    bytes' provenance is the page. But the identity sits inside a formatted
    free-text `detail` rather than a column, so gating means either parsing
    that string or giving the activity row a real page reference. A schema
    question, not a wiring one.
  - `GET /api/memory/entities/{entity_id}` (`handle_get_entity_detail`) —
    evidence reads `Entity.name = pages.title (M3)`, and that may have the
    provenance backwards: the M3 dual-write derives a `kind='entity'` shadow
    page **from** the entity, and every real page reader excludes
    `kind='entity'` explicitly. If the bytes originate in `entities`, the row
    is not page-bearing at all and the correct fix is a demotion rather than an
    adapter. Settling that is a provenance re-audit of its own.

Those are the ceremony PR's scope, and none of them is a prerequisite for the
adapters.
