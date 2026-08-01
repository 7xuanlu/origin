# M6 Stage-0 artifact 8 — overview matrix: split, merge, subscription, proposal

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: frozen contract D11, and gate G8
(`m6_overview_identity_survives_rebinding`).
Continues the decision numbering from artifact 7 (`S0-70`).

**Grounding (rev 2, findings 2 and 15).** In-repo `file:line` citations were read
on branch `kg-m6-stage0`, based on **`origin/main` `1c903bec`** — PR #418, *"close
the M5 daemon gaps"*. Rev 1 was written against `e39048c7` (release 0.15.2), which
`#418` has since superseded; every citation in this artifact was mechanically
re-pinned to `1c903bec` and re-verified to resolve to byte-identical source text,
so no claim moved, only the numbers. App-repo citations are read from
**`wenlan-app` `origin/main` `1d71aa4`** — resolved from that ref rather than from
a working tree, because the local app checkout sits behind `origin/main`. That
checkout is the user's; nothing in this work modifies it. Verify a citation with
`git show origin/main:<path>` inside the app repo.

---

## 0. The carried question, answered

Stage-0 artifact 3 left one question for this artifact: *does M6 need to
subscribe overview slots to M4's community-ID rebinding events, or is an explicit
orphan rule enough?* The dispatch asked me to read M4's rebinding path far enough
to propose an answer. I did, and the answer is that **subscribing would be
subscribing to an event that, by construction, cannot move a survivor's ID.**

M4's rebinding runs in three steps:

1. **`rebind_durable_ids_weighted`** (`crates/wenlan-core/src/community_partition.rs:772`,
   called at `crates/wenlan-core/src/community_grouping.rs:496`) computes, for
   every (new group, old community) pair, a weighted-fractional overlap. Its doc
   comment (`crates/wenlan-core/src/community_partition.rs:768`-`:771`) is
   exact: *"Each node contributes one unit split
   across the prior communities reached by its incident grounded weight. A node
   with no grounded adjacency falls back to its own prior identity."* The
   accumulation is `weight / total` per neighbour bucket
   (`crates/wenlan-core/src/community_partition.rs:803`-`:806`), with the
   no-adjacency fallback at `:807`-`:810`.

2. **`assign_rebound_ids`** (`crates/wenlan-core/src/community_partition.rs:820`)
   sorts candidates by descending overlap, then old ID, then group
   (`:830`-`:837` — fully deterministic, no float-tie ambiguity because the two
   tiebreakers are total orders), then claims greedily:

   ```rust
   // crates/wenlan-core/src/community_partition.rs:841-845
   for (_, old_id, group) in candidates {
       if !rebound.contains_key(&group) && claimed_old.insert(old_id.clone()) {
           rebound.insert(group, old_id);
       }
   }
   ```

   The two halves of that condition are the whole property: `!rebound.contains_key`
   means **a new group takes at most one old ID**, and `claimed_old.insert` means
   **an old ID is claimed by at most one new group**. Unclaimed groups get a fresh
   `community-m4-new-{n}` marker (`:847`-`:860`).

3. **`community_grouping`** mints a real UUID for any marker ID
   (`crates/wenlan-core/src/community_grouping.rs:503`-`:512`) and carries the
   prior ID through unchanged otherwise.

So M4 *already implements D11's max-overlap-survivor rule at the ID level*:

| Structural event | What happens to the IDs | D11's requirement | Met? |
|---|---|---|---|
| **split** | the child with maximum weighted overlap keeps the old ID; every sibling gets a fresh UUID | survivor keeps subscription/page, children inherit none | **yes, by construction** |
| **merge** | the surviving new group claims exactly one old ID; every other old ID is claimed by nobody | survivor keeps its overview, losers detach | **partly — the survivor half is automatic, the loser half has no representation** |

> **Decision S0-71 — M6 does not subscribe overview slots to rebind events. It
> adds an explicit detach rule to machine F, keyed on the merge loser.**
>
> Rationale: for splits and for the surviving side of merges, the subscription's
> `scope_id` is an M4 community ID that M4 guarantees not to change, so a
> subscription mechanism would fire zero times and still cost a table and a
> reconciliation pass. The only case with no automatic representation is the
> merge *loser* — an old community ID that no new group claimed — and that is one
> rule, not a subscription system.
>
> The loser is detectable without new machinery. The finalize retires every
> community in the space and then un-retires the survivors:
>
> ```sql
> -- crates/wenlan-core/src/db.rs:13979-13980
> UPDATE communities SET retired_at = ?2, updated_at = ?2
>  WHERE space = ?1 AND retired_at IS NULL
> -- then, per surviving community, crates/wenlan-core/src/db.rs:13997-14001
> INSERT INTO communities (...) VALUES (...)
>  ON CONFLICT(community_id) DO UPDATE SET ..., retired_at = NULL
> ```
>
> **A merge loser is exactly a row left with `retired_at IS NOT NULL` after the
> finalize commits.** M6's detach rule reads that, and cross-checks it against the
> `community_merge` identity event that `detect_community_identity_events`
> already emits (`crates/wenlan-core/src/community_grouping.rs:785`-`:793`,
> called at `crates/wenlan-core/src/db.rs:13894`).

One nuance the contract does not state and an implementer must not guess:
**D11 says "maximum-overlap survivor" without naming the metric, and M4's metric
is weighted-fractional incident-grounded-edge overlap, not raw member count.**
M4 has both — the unweighted member-count variant `rebind_durable_ids` exists at
`crates/wenlan-core/src/community_partition.rs:753` — and production calls the
weighted one. M6 inherits M4's choice rather than re-deriving one.

> **Decision S0-72 — "maximum overlap" in D11 means whatever
> `rebind_durable_ids_weighted` computes; M6 never recomputes it.** Two
> independent implementations of "maximum overlap" that disagree on a boundary
> case would put the subscription on one community and the page on another, which
> is precisely the identity split G8 exists to catch.

---

## 1. The subscription

### 1.1 There is no substrate today

Searching `crates/wenlan-core/src/` for a subscription concept returns nothing
related — the only hits are in `llm_provider.rs` and eval modules, on an
unrelated sense of the word. The entire subscription table is **PR-A-new**.

More consequentially, the overview that exists today is **one global page**, not
a family:

```rust
// crates/wenlan-core/src/synthesis/overview.rs:25
pub const OVERVIEW_PAGE_TITLE: &str = "Overview";
```

`ensure_overview_page` (`crates/wenlan-core/src/synthesis/overview.rs:70`) finds
it by title (`:75`) and creates it with `space: None.into()` (`:83`) and
`creation_kind: Some("research")` (`:89`). So there is exactly one overview per
install, it belongs to no space and no community, and the mechanism that keeps
it single is a title lookup.

> **Decision S0-73 — the existing global "Overview" page is a distinct scope
> kind (`install`), not the space overview and not a community overview.** The
> alternative — retrofitting it as the space overview of some default space —
> would silently change what an existing user's Overview page contains at
> upgrade. It keeps its title, its page ID, and its behavior; M6's scopes are
> additive.

### 1.2 The durable representation

D11: *"At most one active subscription per `(scope_kind, scope_id)`."*

> **Decision S0-74 — the constraint is a primary key on the active-subscription
> table, not a uniqueness check in code.**
>
> ```sql
> CREATE TABLE m6_overview_subscriptions (
>     scope_kind   TEXT NOT NULL CHECK(scope_kind IN ('install','space','community')),
>     scope_id     TEXT NOT NULL,
>     page_id      TEXT NOT NULL,
>     space        TEXT NOT NULL,
>     state        TEXT NOT NULL CHECK(state IN ('active','detached')),
>     created_at   INTEGER NOT NULL,
>     detached_at  INTEGER,
>     PRIMARY KEY (scope_kind, scope_id)
> );
> CREATE UNIQUE INDEX idx_m6_overview_sub_page
>     ON m6_overview_subscriptions(page_id) WHERE state = 'active';
> ```
>
> The primary key gives D11's at-most-one directly. The second index gives the
> converse — one active subscription per page — which D11 does not state but G8's
> "duplicate subscription" case needs, because without it two scopes could both
> point at one page and a detach of either would half-orphan it.
>
> `detached` rows stay in the table rather than being deleted: D11 requires
> detached overviews to *"remain readable"*, and a deleted subscription row makes
> "which scope did this page once belong to" unanswerable.

For `scope_kind = 'community'`, `scope_id` is the M4 `community_id`
(`crates/wenlan-core/src/db.rs:10447`). Per S0-71 that ID is stable across
splits and across the winning side of merges, so the subscription needs no
rebinding pass.

---

## 2. Split

| # | Situation | Subscription | Page | Genesis |
|---|---|---|---|---|
| SP1 | old community C splits into C (max overlap) + D + E | C's row is untouched — `scope_id` still `C` | C's overview page is untouched | D and E have no subscription, so they enter genesis as ordinary community-overview candidates |
| SP2 | C splits and the max-overlap child is *not* the one a human would call "the same topic" | still untouched | still untouched | the mismatch is a rename/transfer question for a human, never an automatic move — see S0-76 |
| SP3 | C splits into children that all fall below the D2 overview threshold | untouched | untouched | no child forms an overview; C's overview persists over a now-smaller community. Not a defect: D11 governs identity, D2 governs formation |
| SP4 | C splits and C's overview is human-edited | untouched | untouched, byte-identical | unchanged from SP1 — split never touches the survivor, so the human-edited case needs no special handling here |

> **Decision S0-75 — split stages no proposal.** D11 requires a transfer/retire
> proposal on *merge* (for the losing overview) and says nothing about split,
> because on split nothing is orphaned: the survivor keeps everything and the
> children never had anything. Staging a "your community split" card for every
> split would be a notification, not a decision, and artifact 5's suppression
> rules exist precisely to keep non-decisions off the review surface.
>
> M4 already emits a `community_split` proposal into `refinement_queue`
> (`crates/wenlan-core/src/db.rs:14089`-`:14092`) for its own naming purposes.
> M6 does not add a second one.

---

## 3. Merge

| # | Situation | Survivor | Loser |
|---|---|---|---|
| MG1 | C and D merge; the new group claims C | C's subscription and page untouched | D's subscription moves to `detached`; D's page stops refreshing, stays readable; one transfer/retire proposal stages |
| MG2 | C and D merge, both have overviews, D's is human-edited | as MG1 | D detaches but is **never** archived, renamed, or overwritten (§4). The proposal offers transfer or retire; neither is applied without a human |
| MG3 | C and D merge, only D has an overview | C gains nothing automatically — no subscription is created for C by the merge | D detaches and proposes transfer to the survivor. **Accepting the transfer is what gives C an overview**, and it is a human's call |
| MG4 | three-way merge C+D+E → C | C untouched | D and E each detach and each stage a proposal — but see S0-77 on coalescing |
| MG5 | a merge whose survivor ID is fresh (no old ID claimed) | there is no survivor subscription | every participant detaches; the new community enters genesis normally |

> **Decision S0-76 — detach is automatic; transfer and retire are not.** D11
> lists three things that must never happen automatically to a human-edited
> overview (transfer, archive, rename) and one thing that must (stop refreshing).
> Stage 0 applies the stricter reading to *all* detached overviews, human-edited
> or not: detaching is a statement about the world (the community it described no
> longer exists), while transferring is a judgment about meaning. The former is
> observation, the latter is editorial.

> **Decision S0-77 — one proposal per `(merge event, losing scope)`, and the
> proposal ID is derived so a re-run coalesces.** MG4 stages two proposals, not
> one batched card, because each names a different page a human must decide
> about. But a re-published generation must not stage them again: the ID is
> `m6_digest("m6-overview-proposal-v1", [action, space, source_generation,
> losing_scope_id])`, inserted with `INSERT OR IGNORE`, mirroring what M4 already
> does for its own proposals (`crates/wenlan-core/src/db.rs:14075`-`:14078` for
> the ID shape, `:14090` for the insert mode).

### 3.1 What "stops refreshing" means concretely

A detached overview must not be picked up by the maintenance refresh. Today the
refresh path is unconditional on the reserved page
(`refresh_overview_page`, `crates/wenlan-core/src/synthesis/overview.rs:104`),
which marks it stale every pass (`:118`) and re-distills.

> **Decision S0-78 — "stops refreshing" is enforced by the sweep's join to the
> subscription table, not by a flag on the page.** A detached overview is a page
> with no `state='active'` subscription row, so a sweep that selects
> `JOIN m6_overview_subscriptions … WHERE state = 'active'` cannot see it. A
> `stopped` boolean on `pages` would be a second, drift-prone truth about the
> same fact, and artifact 5's S0-42 rejected the same shape for the same reason.
>
> Corollary the implementation must respect: a detached overview must also stop
> being *marked* stale, not merely stop being refreshed. A page marked stale
> forever with nothing willing to refresh it is exactly the silent parking D7's
> closing rule forbids.

---

## 4. Human-edited overviews

### 4.1 The predicate

The same one the rest of the page system uses — there is no overview-specific
notion of human editing and M6 must not invent one:

```rust
// crates/wenlan-core/src/post_write/page_update.rs:111-113
pub fn page_is_human_owned(page: &crate::pages::Page) -> bool {
    page.user_edited || page.creation_kind == "authored"
}
```

An overview created by `ensure_overview_page` has `creation_kind = "research"`
(`crates/wenlan-core/src/synthesis/overview.rs:89`), so it starts machine-owned
and becomes human-owned the moment a manual or filesystem edit sets `user_edited`
(`crates/wenlan-core/src/db.rs:42540`).

### 4.2 The four prohibitions

| Prohibition | Enforcement |
|---|---|
| never automatically **transferred** | S0-76 — transfer is only ever a human accepting a proposal |
| never automatically **archived** | detach sets `state='detached'` on the subscription; it never touches `pages.status` |
| never automatically **renamed** | §5, and S0-80 |
| never automatically **overwritten** | the in-statement `AND COALESCE(user_edited, 0) = 0` guard on every content update (`crates/wenlan-core/src/db.rs:42550`, `:42571`) — the same guarantee artifact 7's S0-66 rests on |

> **Decision S0-79 — the four prohibitions are asserted against a human-edited
> overview that has been detached, not merely one that exists.** A test that
> edits an overview and asserts nothing happened proves very little. The G8 case
> is: edit the overview, merge its community away, then assert all four — the
> page's content, title, `status`, and `user_edited` are unchanged, and the only
> new row anywhere is one proposal.

---

## 5. Titles

D11: *"Titles initialize from an accepted community display name or neutral
stable fallback; structural rebinding never changes a title."*

`communities.display_name` is inserted `NULL`
(`crates/wenlan-core/src/db.rs:14000`) and set only when a human accepts a rename
proposal (`crates/wenlan-core/src/db.rs:16371`-`:16373`). Two properties of that
statement matter here, and both are already true in the tree:

- **An accepted name survives regrouping.** The finalize upsert's
  `ON CONFLICT DO UPDATE` list is `space, algo_version, projection_version,
  updated_at, retired_at` (`crates/wenlan-core/src/db.rs:14002`-`:14005`) — it
  does **not** include `display_name`. So republishing a generation cannot erase
  a name a human chose.
- **A retired community cannot be renamed.** The update requires
  `retired_at IS NULL` (`crates/wenlan-core/src/db.rs:16373`) and errors with
  *"community … is missing, retired, or outside proposal space"* when it matches
  nothing (`:16388`-`:16391`). A merge loser is therefore already un-renameable,
  which is consistent with it being detached.

> **Decision S0-80 — the overview title is set once at page creation and is
> never rewritten by any automatic path, including a later display-name
> acceptance.** The tempting behavior is to propagate an accepted community name
> onto the existing overview page. It is refused: a page title is a user-visible,
> link-target-bearing string (wikilinks resolve on it — `label_key` in artifact
> 4 §6), so an automatic retitle would silently repoint every inbound link. A
> rename after the fact is a proposal, not a side effect.

> **Decision S0-81 — the neutral stable fallback is `Overview: <community_id>`,
> and the fallback is chosen by whether `display_name` is `NULL` at page-creation
> time, evaluated once.** "Stable" is the operative word: a fallback derived from
> the community's *members* would change every time the membership changes, which
> is a silent title change by another route.

### 5.1 The title-collision hazard

The lookup that keeps the global overview single is not scoped:

```sql
-- crates/wenlan-core/src/db.rs:43809-43812
SELECT id FROM pages
 WHERE LOWER(title) = LOWER(?1) AND status = 'active'
   AND COALESCE(kind, 'concept') != 'entity'
 LIMIT 1
```

No `space` predicate, case-insensitive, `LIMIT 1`. So a community whose accepted
display name is `Overview` — or `overview`, or a space overview titled the same —
would collide with the reserved install-level page, and `ensure_overview_page`
would return the *wrong page ID* and refresh the community's overview content
into the install overview.

> **Decision S0-82 — M6 overview pages are located by their subscription row,
> never by title lookup.** `find_active_page_id_by_title` stays exactly as it is
> for the existing install-level page (S0-73); M6's community and space overviews
> resolve `scope → page_id` through `m6_overview_subscriptions`, which makes the
> title a display string with no lookup duty and the collision harmless.
>
> This is the G8 adversarial case: accept the display name `Overview` for a
> community, then assert the install overview is untouched.

---

## 6. Space overviews

D11: *"Space overviews are unaffected by community split/merge."*

This falls out of S0-74 rather than needing a rule: a space overview's
subscription is `('space', <space name>)`, and nothing in §2 or §3 reads or
writes a row whose `scope_kind` is `'space'`. Both structural events are scoped to
`scope_kind = 'community'` by their own keys.

> **Decision S0-83 — the space-overview invariant is asserted as "no row with
> `scope_kind='space'` was written during the event", not as "the space overview
> page's content is unchanged".** Content equality would also pass if the space
> overview happened to be refreshed for an unrelated reason during the same
> cycle, and would fail spuriously if it were legitimately refreshed. The write
> assertion tests the actual claim.

---

## 7. The M6 proposal wire

D11: *"Reassignment uses a new versioned M6 proposal wire. Do not mutate the
`deny_unknown_fields` community-v1 payload."*

The frozen surface is `CommunityProposalPayload`
(`crates/wenlan-types/src/communities.rs:98`), with `deny_unknown_fields` on its
serde attribute (`:97`) and three variants — `CommunitySplit` (`:99`),
`CommunityMerge` (`:107`), `CommunityRename` (`:115`). Its own test asserts the
closure (`crates/wenlan-types/src/communities.rs:232`). Adding an M6 variant to
that enum would change the accepted wire for every existing client, so:

> **Decision S0-84 — M6 adds a separate type,
> `OverviewProposalPayload`, gated by a new constant
> `OVERVIEW_PROPOSAL_SCHEMA_VERSION = "m6-overview-proposal-v1"`, alongside the
> untouched `community-read-v1` constant
> (`crates/wenlan-types/src/communities.rs:6`).**
>
> ```rust
> #[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
> pub enum OverviewProposalPayload {
>     OverviewTransfer {
>         space: String,
>         source_generation: i64,
>         losing_scope_kind: String,
>         losing_scope_id: String,
>         losing_page_id: String,
>         surviving_scope_id: String,
>     },
>     OverviewRetire {
>         space: String,
>         source_generation: i64,
>         losing_scope_kind: String,
>         losing_scope_id: String,
>         losing_page_id: String,
>     },
> }
> ```
>
> It carries `deny_unknown_fields` for the same reason the community wire does.
> It lives in a new module rather than in `communities.rs`, so that a future
> change to M6's proposals cannot accidentally edit the frozen enum in a
> neighbouring hunk.

One shared mechanism, deliberately: both proposal families land in
`refinement_queue` with `status = 'awaiting_review'`
(`crates/wenlan-core/src/db.rs:14090`-`:14092`). A second queue would need a
second review surface.

---

## 8. Findings against the tree

**F1 — there is no per-scope overview substrate at all; the current overview is a
single global title-keyed page.** `OVERVIEW_PAGE_TITLE`
(`crates/wenlan-core/src/synthesis/overview.rs:25`) plus a space-less create
(`:83`) plus an unscoped title lookup (`crates/wenlan-core/src/db.rs:43809`-`:43812`)
means D11's entire `(scope_kind, scope_id)` model is PR-A-new. This is larger
than "add a table": the existing overview's *singleton-ness is implemented by its
title*, which is why S0-82 has to move M6 off title lookup entirely rather than
adding a second title constant.

**F2 — M4's merge-loser signal is a second-order effect of the retire/un-retire
sequence, not an explicit output.** Losers are identifiable only as rows still
carrying `retired_at IS NOT NULL` after the finalize
(`crates/wenlan-core/src/db.rs:13979`-`:13980` then `:13997`-`:14005`). That is
sound but implicit — nothing names it, and a future refactor that changed the
retire strategy to a targeted `UPDATE … WHERE community_id IN (…)` would preserve
correctness for M4 and silently break M6's detach rule. **Recommendation for
PR-A: make the detach rule read the `community_merge` identity event
(`crates/wenlan-core/src/community_grouping.rs:785`-`:793`) as its primary
signal and use `retired_at` only as a cross-check**, so M6 depends on a named
output rather than on a side effect of a bulk update.

**F3 — `assign_rebound_ids` can leave a merge with no ID continuity at all**
(case MG5). If every old ID in a merge is claimed by some *other* group first —
possible because claiming is greedy over a global sort
(`crates/wenlan-core/src/community_partition.rs:841`-`:845`), not per-event — the
merged group falls through to a fresh `community-m4-new-{n}`
(`:847`-`:860`) and then a fresh UUID
(`crates/wenlan-core/src/community_grouping.rs:503`-`:512`). D11's merge rule
("the survivor keeps its overview") has no survivor in that case. MG5 covers it
by treating every participant as a loser, which is the only reading that loses no
data, but it is worth flagging that **D11's merge language presumes a survivor
that M4 does not always produce.** Reported, not resolved — this is a contract
question, not an implementation one.

**F4 — the existing global overview is refreshed unconditionally.**
`refresh_overview_page` (`crates/wenlan-core/src/synthesis/overview.rs:104`)
ensures the row exists, replaces its sources (`:115`), and marks it stale
(`:118`) on every maintenance pass, with no subscription concept to consult.
S0-78's subscription join is therefore a genuine behavior change to this
function, not just a new query — PR-A must keep the install-scope page refreshing
exactly as it does today (S0-73) while the new scopes go through the join.

---

## 9. Gate mapping — G8

`m6_overview_identity_survives_rebinding` must *"cover split, merge, partitioner
swap, label proposal, stale-generation acceptance, duplicate subscription,
human-edited overview, detached loser, and space overview,"* asserting stable
IDs, no silent title change, and no data loss. Each named case maps to a section
here:

| G8 case | Source | Assertion |
|---|---|---|
| split | SP1–SP4 | survivor's `scope_id`, `page_id`, and title all unchanged; children hold zero subscription rows |
| merge | MG1–MG5 | survivor unchanged; each loser has `state='detached'` and exactly one proposal |
| partitioner swap | S0-72 | changing the partitioner changes membership but the subscription is keyed on the community ID, so no subscription row is written |
| label proposal | §5, S0-80 | accepting a display name updates `communities.display_name` and **not** `pages.title` |
| stale-generation acceptance | §7 | a proposal carrying a `source_generation` older than the current one is rejected, and the loser stays detached rather than being retro-transferred |
| duplicate subscription | S0-74 | the primary key and the partial unique index each reject their direction; assert both, since one index catches only one shape |
| human-edited overview | S0-79 | the four prohibitions, asserted on a detached human-edited page |
| detached loser | S0-78 | readable, not refreshed, **and not marked stale** |
| space overview | S0-83 | no `scope_kind='space'` row written during a community event |
| **title collision (added)** | S0-82, F1 | accept the display name `Overview` for a community; the install overview page is untouched |

The last row is not in G8's list and should be added to it: F1 makes it reachable
with no adversarial effort beyond choosing an ordinary word as a community name.

---

## 10. Decisions introduced here

`S0-71` no rebind subscription; an explicit merge-loser detach rule instead ·
`S0-72` "maximum overlap" means M4's weighted metric, never recomputed ·
`S0-73` the existing global Overview is its own `install` scope kind ·
`S0-74` at-most-one is a primary key, plus a partial unique index for the converse ·
`S0-75` split stages no M6 proposal ·
`S0-76` detach is automatic; transfer and retire are not ·
`S0-77` one proposal per (merge event, losing scope), with a derived coalescing ID ·
`S0-78` "stops refreshing" is a subscription join, not a page flag — and it must also stop marking stale ·
`S0-79` the four prohibitions are asserted on a *detached* human-edited overview ·
`S0-80` overview titles are set once and never automatically rewritten ·
`S0-81` the neutral fallback is `Overview: <community_id>`, evaluated once ·
`S0-82` M6 overviews resolve by subscription row, never by title lookup ·
`S0-83` the space-overview invariant is asserted as "no row written" ·
`S0-84` a separate `OverviewProposalPayload` type; `community-v1` stays frozen.
