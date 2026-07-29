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

A file that cannot be attributed to a page ID is **never deleted** — the
projection directory is inside the user's vault and may hold files Wenlan did not
write. It is reported, and an unattributed file makes the pass fail, which
section 4 then acts on. Fail-closed on the decision, never on the user's data.

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

## 6. Unsupported pages are not re-distilled

Background re-distillation sends page titles to the configured LLM provider,
which may be external, with neither gate in the path: `scheduler.rs:2494` →
`refinery/mod.rs:1583` → `synthesis/distill.rs:1309`, plus other active titles
pulled as a prompt hint (`:85`).

It is not an HTTP route, so there is no request to attach a grant to. The permit
is therefore consumed **inside the refinery**: at generation ≥ 1 a page the
automatic reader may not see is not re-distilled, and is not offered as a title
hint for someone else's distillation either. An unsupported page is not a page we
are willing to spend an external round-trip on.

## 7. A tooth on the projection pass's wiring

Deleting the `enforce_projection_directory_invariant` call from `main.rs` leaves
every test green today — the pass is tested, its wiring is not. A source-scan
test asserts the call site exists, the same shape as the F5 scan that proves
`set_truth_cutover_generation` has no production caller.

## 8. The 26 under-audited demotions

The Opus review retracted the premise behind 34 manifest demotions ("no
production code path writes page prose into `memories`" is false —
`post_write.rs:3090`). Eight were re-checked; 26 were recorded as under-audited
rather than known correct. They are re-audited under the provenance question in
this PR, and the manifest is corrected where the re-audit disagrees.

## What PR-C does not do

- **It does not advance the generation.** That is the ceremony, and it comes
  after this. Ordering is the finding this whole PR answers.
- It does not build the fence that holds page writes during the ceremony.
- It does not resolve page-map hiding as a graph transformation; incident edges
  are keyed by node IDs, not page IDs, so a hidden page's edges need their own
  pass.

Those three are the ceremony PR's scope, and none of them is a prerequisite for
the adapters.
