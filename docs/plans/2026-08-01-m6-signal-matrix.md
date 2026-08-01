# M6 Stage-0 artifact 1 — four-signal input, eligibility, and threshold matrix

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

**Sources.** Frozen goal prompt D1 (relaxed independence-group floor) and D2 (four signals, admission thresholds), plus the scope fence. G2 (`m6_genesis_counts_groups_not_rows`) is the executable consumer.

**How to read this.** The per-signal eligibility tables are normative. Each predicate row carries a **grounding verdict**:

| Verdict | Meaning |
|---|---|
| `EXISTS` | a table and column on this branch answers the predicate today |
| `DERIVED` | answerable from existing tables, but no query does it today — PR-A writes the query, adds no schema |
| `PR-A-new` | no substrate; PR-A must add schema. Named, never invented behavior |
| `lane 1` | substrate exists, its writer is in flight on `m5-truth-derivation` — see §7.1. Same meaning as artifact 12's `lane 1` |
| ~~`BLOCKED`~~ | *retired rev 8.* It meant blocked on an open contract ruling (§7.2). R-2 was ruled on 2026-08-01, so no row in this artifact carries it. The legend row survives only to record that this artifact's `BLOCKED` was never artifact 12's — a collision §7.1 called out, and one worth keeping legible now that both are discharged |

**Status.** Contract only. No schema, no code.

---

## 1. What the floor counts

D1 fixes the counting unit before any signal is defined:

> Every automatic genesis signal counts distinct `independence_group_id` values represented by active, grounded external roots. A pre-page provenance root has no M5 `support_status`; do not invent one.

Three separate qualifiers sit in that sentence, and each maps to a column that exists today:

| Qualifier | Concrete predicate | Location | Verdict |
|---|---|---|---|
| distinct `independence_group_id` | `COUNT(DISTINCT provenance_roots.independence_group_id)` | `crates/wenlan-core/src/db.rs:8792` | `EXISTS` |
| active root | `provenance_roots.status = 'active'` — the CHECK enumerates `'ingesting' \| 'active' \| 'failed'`, default `'active'` | `crates/wenlan-core/src/db.rs:8793` | `EXISTS` |
| grounded, still-asserted edge | `edges.grounded = 1 AND edges.valid_until IS NULL` — exactly the predicate the partial index `idx_edges_active_grounded_space_type` is built on | `crates/wenlan-core/src/db.rs:8827`–`:8828` | `EXISTS` |
| external (not agent-authored) | `root_kind = 'document_ingest'`; the M3g promoter that mints these is gated to `source_agent = 'folder'` | `crates/wenlan-core/src/db.rs:8791`; `crates/wenlan-core/src/edge_grounding.rs:2114` | `EXISTS` |
| a pre-page root has no support status | `provenance_roots` has **no** support column, and none should be added | DDL, `crates/wenlan-core/src/db.rs:8787`–`:8796` | `EXISTS` (as an absence — confirmed by reading the full DDL) |

The last row is a positive confirmation rather than a gap: D1 says *do not invent one*, and the table indeed has nothing to invent from. Any M6 code that reaches for a root's "support" is wrong by construction.

**The canonical count expression.** Every floor in D2 reduces to one shape, which PR-A implements once and the four signals parameterise:

```
COUNT(DISTINCT r.independence_group_id)
  FROM edges e
  JOIN provenance_roots r ON r.root_id = e.root_id
 WHERE e.grounded = 1
   AND e.valid_until IS NULL
   AND r.status = 'active'
   AND r.root_kind <> 'generated'
   AND <per-signal scope>
```

`DERIVED` — no query of this shape exists on the branch today. The join path (`edges.root_id → provenance_roots`) is real and indexed (`idx_edges_root`, `crates/wenlan-core/src/db.rs:8824`); nothing has yet had a reason to aggregate over it.

---

## 2. Signal 1 — evidence cluster

> **D2.1**: current durable M4 community; at least `3` groups.

**Input set.** The entity nodes of one M4 community, and the active grounded external roots reachable from them.

| # | Eligibility predicate | Concrete grounding | Verdict |
|---|---|---|---|
| 1.1 | the community exists and is not retired | `communities.retired_at IS NULL`; DDL at `crates/wenlan-core/src/db.rs:10446` (`community_id TEXT PRIMARY KEY` at `:10447`) | `EXISTS` |
| 1.2 | **durable** — the community's membership is the published one, not an in-flight recompute | three conditions together: `community_members.published_generation = space_graph_state.published_generation`, `space_graph_state.dirty = 0`, and `grouping_generation = published_generation`. The join recurs at `crates/wenlan-core/src/db.rs:15970` and `:15497` | `EXISTS` |
| 1.3 | the publication was actually finalized, not merely stamped | a matching `community_publication_receipts` row (membership digest + algo/projection version), enforced by the fail-closed reader gate `community_reader_durable_gate_sql` at `crates/wenlan-core/src/db.rs:2642` | `EXISTS` |
| 1.4 | ≥ 3 distinct independence groups over the community's members | the §1 count expression, scoped to edges whose endpoint is a `community_members.node_id` of this community | `DERIVED` |
| 1.5 | overview evidence excluded | see §5, rule R5; title-rule interim per S0-164 | PR-A |

**Scope clause for §1's count.** `e.src_kind = 'entity' AND e.src_id IN (SELECT node_id FROM community_members WHERE community_id = ?)`, unioned with the same over `dst_kind`/`dst_id`. `community_members` is keyed `(space, node_id)` with `node_kind` defaulting to `'entity'` (`crates/wenlan-core/src/db.rs:10459`–`:10460`), so an entity endpoint is the only member kind today.

**Decision S0-13 — "durable" means all three of 1.2 and 1.3, not `retired_at IS NULL` alone.** D2 says "current durable M4 community" without decomposing it. The tree has four separate signals that could each be read as "durable", and only the conjunction is safe: a community row can be un-retired while its membership belongs to a superseded generation, and `space_graph_state.published_generation` can advance while no receipt confirms the membership digest. M6 reuses the existing fail-closed gate rather than assembling its own conjunction, so a future change to what "published" means cannot silently widen M6's admission.

---

## 3. Signal 2 — orphan wikilink

> **D2.2**: at least `2` distinct active, supported, non-overview referring pages and at least `3` underlying groups.

**Input set.** One normalized wikilink label with no resolved target, and the pages that reference it.

| # | Eligibility predicate | Concrete grounding | Verdict |
|---|---|---|---|
| 2.1 | the link is orphaned | `page_links.target_page_id IS NULL` — the partial index `idx_page_links_orphan` is built exactly on that predicate (`crates/wenlan-core/src/db.rs:6680`–`:6681`). There is no separate resolved/unresolved flag; NULL target **is** the state | `EXISTS` |
| 2.2 | links group by normalized label, not raw text | `page_links.label_key`, lowercased at write time (`crates/wenlan-core/src/db.rs:43916`); `label` keeps first-seen casing for display. Primary key is `(source_page_id, label_key)`, DDL `crates/wenlan-core/src/db.rs:6670` | `EXISTS`, but see decision S0-14 |
| 2.3 | ≥ 2 **distinct** referring pages | `COUNT(DISTINCT page_links.source_page_id)`; the PK already prevents one page from contributing twice for one label | `EXISTS` |
| 2.4 | referring pages are **active** | `pages.status = 'active'`; `resolve_orphan_page_links` already joins on exactly this (`crates/wenlan-core/src/db.rs:44117`) | `EXISTS` |
| 2.5 | referring pages are **supported** | `page_truth_state.support_status = 'supported'`; DDL and CHECK at `crates/wenlan-core/src/db/claim_identity.rs:279`–`:300` | **`lane 1`** — §7.1 |
| 2.6 | referring pages are **non-overview** | the title rule now — literally `lower(title) <> 'overview'`; `pages.kind <> 'overview'` (CHECK at `crates/wenlan-core/src/db.rs:9177`) only after the follow-up PR makes `kind` truthful on every path — S0-164 | PR-A |
| 2.7 | ≥ 3 underlying groups | D1: the union of active grounded external roots supporting **the exact current claim revisions that contain the link** — not the referring page as a whole | `DERIVED` |
| 2.8 | a stale or provisional referring page contributes nothing | D1, explicit. Depends on 2.5 | **`lane 1`** — §7.1 |

**Decision S0-14 — the D8 wikilink key normalization is stricter than `label_key` and PR-A must not conflate them.** `label_key` today is `to_lowercase()` and nothing else (`crates/wenlan-core/src/db.rs:43916`). D8 requires NFKC, lowercase, whitespace collapse, alias/fragment stripping, control and bidi rejection, and a `1..=128` scalar bound. Those are different functions: `to_lowercase()` accepts a 4000-character label containing a bidi override, and NFKC folds characters that `to_lowercase` leaves distinct. PR-A adds the D8 normalizer as a **separate** function used for the M6 slot ID and admission, and leaves `label_key` untouched — rewriting `label_key` would rewrite the primary key of every existing `page_links` row. Where the two disagree, admission uses the D8 form and the existing link rows are read through it. Two labels that collide under D8 but not under `label_key` are one candidate; the reverse cannot happen, since D8's normalization is a refinement of lowercasing.

**Decision S0-15 — "underlying groups" is scoped to the claim revisions containing the link, and PR-A must not widen it to the whole page.** D1 says "the exact current claim revisions that contain the link". The cheap implementation — count every root supporting the referring page — would let an unrelated well-sourced paragraph on the same page supply the floor for a link it never mentions. That is the row-counting failure G2 exists to catch, one level up. The narrow reading is the contract; PR-A implements it, and if the claim-revision-to-link binding turns out not to exist, that is a `PR-A-new` item to name, not a reason to widen.

---

## 4. Signals 3 and 4 — community and space overview

> **D2.3**: current published community; at least `5` grounded nodes and `3` groups; no active overview subscription.
> **D2.4**: at least `5` grounded nodes and `3` groups; no active space-overview subscription.

| # | Eligibility predicate | Concrete grounding | Verdict |
|---|---|---|---|
| 3.1 | current published community (signal 3 only) | same conjunction as 1.2 + 1.3 | `EXISTS` |
| 3.2 | ≥ 5 **grounded** nodes | `community_members.attachment = 'core'` only — see decision S0-16 | `EXISTS` |
| 3.3 | ≥ 3 groups | the §1 count expression, scoped as in signal 1 (signal 3) or to the space (signal 4) | `DERIVED` |
| 3.4 | no active overview subscription | — | **`PR-A-new`** |
| 4.1 | ≥ 5 grounded nodes in the space | as 3.2, scoped by `community_members.space` | `EXISTS` |
| 4.2 | no active space-overview subscription | — | **`PR-A-new`** |
| 3.5 / 4.3 | overview candidates record witness roots, never concept coverage | artifact 2 machine B, `claim_role = 'witness'` | `PR-A-new` (artifact 2's table) |

**Decision S0-16 — a "grounded node" is `attachment = 'core'`, excluding both isolated attachment classes.** M4 membership is not uniform. `project_grounded_relates` produces `attachment = "core"` (`crates/wenlan-core/src/community_grouping.rs:516`); nodes with no grounded edge are then attached either by strongest ungrounded neighbour, `attachment = "isolated_ungrounded"` (`:590`), or by nearest embedding centroid, `attachment = "isolated_embedding"` (`:598`).

Counting all `community_members` rows toward the ≥ 5 floor would let embedding proximity supply a floor — which the scope fence forbids in as many words ("turn embedding similarity into a genesis floor") and D2's tie-break rule forbids again. So the floor counts `'core'` only. This is the single most consequential grounding decision in this artifact: the naive `COUNT(*) FROM community_members` is both the obvious implementation and a direct scope-fence violation.

**On overview subscriptions.** A repo-wide scan for `subscription` / `subscribe` over `wenlan-core`, `wenlan-server`, and `wenlan-types` finds only the WebSocket client message `Subscribe { channels }` (`crates/wenlan-server/src/websocket.rs:25`), a transient per-connection UI channel with no durable state. There is no table, column, or wire type expressing a page's subscription to a community or space. Both 3.4 and 4.2 are `PR-A-new` in full — schema, writer, and reader.

---

## 5. D1's counting rules

Each rule, its enforcing predicate, and where the data lives.

| # | D1 rule | Enforcing predicate | Data location | Verdict |
|---|---|---|---|---|
| R1 | Independent documents and UI-authorized human capture/correction groups count | `root_kind IN ('document_ingest','human_capture','human_edit_delta')` | `crates/wenlan-core/src/db.rs:8791` | `EXISTS` |
| R2 | **Generated roots count zero** | `root_kind <> 'generated'` | same CHECK, `crates/wenlan-core/src/db.rs:8791` | `EXISTS` |
| R3 | Chunks, mirrors, and same-session captures **collapse through the independence group** | not a filter — an assignment property. Two chunks of one file receive the same `independence_group_id` because they share `source_identity`, and near-dups are unioned by the LSH overlay. Asserted today by `distinct chunks of one file share one independence_group_id` (`crates/wenlan-core/src/edge_grounding.rs:2251`) | artifact 3 §2 | `EXISTS` |
| R4 | **Unknown independence routes to human review and cannot auto-publish** | today `acquire_provenance_root` returns `Err` and mints nothing (`crates/wenlan-core/src/db.rs:18525`–`:18532`) | artifact 3 §5 | **partial** — the refusal exists, the durable review artifact does not |
| R5 | **Overview pages and generated overview evidence never contribute to genesis** | the title rule now — literally `lower(title) <> 'overview'`; `pages.kind <> 'overview'` only after the follow-up PR (S0-164) | `crates/wenlan-core/src/db.rs:9177` | PR-A |
| R6 | M5 support applies to page-mediated inputs; a stale/provisional referring page contributes nothing | `page_truth_state.support_status = 'supported'` | `crates/wenlan-core/src/db/claim_identity.rs:282` | **`lane 1`** — §7.1 |
| R7 | New M6 prose stays invisible unless its M5 claim/support publication succeeds | artifact 2 machine E — the page and its truth state commit in one transaction, so there is no window where prose exists without published support | artifact 2 §8.2 | `PR-A-new` (by construction) |

R4 deserves its own line because the contract and the tree differ in kind, not degree. D1 wants unknown independence to *route to human review*. The tree **refuses the mint** — `acquire_provenance_root` returns an error, no root row is created, and no durable artifact records that a review is owed. The refusal is the safe half (nothing inflates the count); the missing half is that a human is never told. PR-A must add the review artifact. Detail in artifact 3 §5.

---

## 6. Boundary cases — the G2 seed

G2 requires that "each D2 threshold has boundary tests". These are the cases, stated so a test can be written from the row alone. Every "groups" count is the §1 count expression.

| Case | Signal | Setup | Expected |
|---|---|---|---|
| B1 | evidence cluster | exactly 2 independence groups | **reject** |
| B2 | evidence cluster | exactly 3 independence groups | **admit** |
| B3 | evidence cluster | 4 groups, community retired (`retired_at` set) | reject (1.1) |
| B4 | evidence cluster | 4 groups, membership at a superseded `published_generation` | reject (1.2) |
| B5 | evidence cluster | 4 groups, `space_graph_state.dirty = 1` | reject (1.2) |
| B6 | evidence cluster | 4 groups, no `community_publication_receipts` row | reject (1.3) |
| B7 | orphan wikilink | 1 referring page, 5 groups | **reject** (2.3) |
| B8 | orphan wikilink | 2 referring pages, 5 groups | **admit** |
| B9 | orphan wikilink | 3 referring pages, 2 groups | **reject** (2.7) |
| B10 | orphan wikilink | 3 referring pages, 3 groups | **admit** |
| B11 | orphan wikilink | 2 referring pages, one `provisional` | reject — only 1 supported page (2.5) |
| B12 | orphan wikilink | 2 referring pages, one an overview page | reject — only 1 non-overview page (2.6) |
| B13 | orphan wikilink | 2 referring pages, one `status <> 'active'` | reject (2.4) |
| B14 | orphan wikilink | link already resolved (`target_page_id` non-NULL) | reject (2.1) |
| B15 | community overview | 4 core nodes, 3 groups | **reject** (3.2) |
| B16 | community overview | 5 core nodes, 3 groups | **admit** |
| B17 | community overview | 5 core nodes, 2 groups | **reject** (3.3) |
| B18 | community overview | 4 core + 3 `isolated_embedding` nodes, 3 groups | **reject** — embedding attachment does not count (S0-16) |
| B19 | community overview | 4 core + 2 `isolated_ungrounded` nodes, 3 groups | **reject** (S0-16) |
| B20 | community overview | 5 core nodes, 3 groups, an active overview subscription | reject (3.4) |
| B21 | space overview | 5 core nodes in the space, 3 groups | **admit** |
| B22 | space overview | 5 core nodes, 3 groups, active space-overview subscription | reject (4.2) |
| B23 | any | 3 groups, one contributed only by a `root_kind = 'generated'` root | **reject** — 2 effective groups (R2) |
| B24 | any | 3 chunks of one document, no other evidence | **reject** — 1 group (R3) |
| B25 | any | 3 groups, one root `status = 'ingesting'` | reject (§1, active) |
| B26 | any | 3 groups, one root's only edge has `valid_until` set | reject (§1, still-asserted) |
| B27 | any | 3 groups, one root's edge has `grounded = 0` | reject (§1, grounded) |
| B28 | any | 3 groups, all from `human_edit_delta` roots authored by one person | **reject** — all human authorship shares one group (artifact 3 §2), so this is 1 group, not 3 |
| B29 | positive control | 3 genuinely independent documents | **admit** (R1) |
| B30 | positive control | 3 UI-authorized human capture groups | **admit** (R1) — but see B28: these must be genuinely distinct capture groups, not three deltas from one author |
| B31 | any | exactly 64 eligible links on a page | admit at the cap (D8) |
| B32 | any | 65 eligible links on a page | the 65th is excess and remains frontier-visible (D8) |

B28 and B30 are in tension by design and both are correct: R1 says human capture groups count, and the human-root rule collapses all human authorship into one group. The resolution is that "UI-authorized human capture/correction groups" must be distinct *groups*, and one author produces one. A test that seeds three human deltas and expects admission is asserting the bug.

---

## 7. Findings that strain the frozen contract

Both are cases where a D1/D2 predicate names a column that exists but is not maintained, so the predicate would silently evaluate the same way for every row. Raised under the STOP-13 instruction rather than resolved unilaterally — and both have since been answered, so neither is open *(rev 9, round-7 finding 2: rev 8 answered them below but left this line reading "reported, not resolved")*. §7.1 resolved into `lane 1`: the promoter is in flight on `m5-truth-derivation`. §7.2 was **ruled on 2026-08-01** — R-2 = option 1, with the title rule as the named interim until `kind` is truthful (S0-164).

### 7.1 No production writer promotes `support_status` to `supported`; every page is `provisional`

> **Status (rev 2): resolved into a lane, no longer pending a ruling.** The
> claim-derivation promoter is in flight on `m5-truth-derivation`, so artifact 12
> catalogues the dependent G2 clauses as **`lane 1` — prerequisite in flight**,
> not blocked-pending-ruling. Rev 1's `BLOCKED` marks on the support-dependent
> rows were both stale and, worse, used a different meaning of the word than
> artifact 12 does (there, `BLOCKED` means the merge-no-survivor ruling alone).
> Corrected below; §7.2's escalation is a genuinely different one and stays open.

**Confirmed, with the scope `#418` forces (rev 2, finding 15).** Rev 1 said
`page_truth_state` "has no live writer", which is no longer true and was always
broader than the claim M6 rests on. At `1c903bec` there are **two** production
writers: the migration-99 backfill (`crates/wenlan-core/src/db/claim_identity.rs:502`,
called once from `crates/wenlan-core/src/db.rs:11756`), and the presence-review
upsert `#418` added, which sets `human_reviewed = 1` and deliberately leaves the
machine axis at `'provisional'`
(`crates/wenlan-core/src/db/presence_review.rs:326`-`:341`, reasoning at
`:321`-`:325`). The claim that survives, and the only one D1 and D2 depend on, is
narrower: **no production writer promotes `support_status` to `'supported'`** — a
repository-wide search for that literal at `1c903bec` returns no production hit,
and the derivation queue is still unserved, `claim_derivation_jobs` having no
production reader or writer outside its own DDL. The migration's doc comment
states the resulting floor:

> Migration 99 (M5 PR-A): fail-closed page truth-state backfill. **Every page becomes `provisional` and unreviewed.** Nothing is read from a legacy field and turned into a truth claim — see `backfill_page_truth_state` for why that is the whole point rather than a conservative default.
> — `crates/wenlan-core/src/db.rs:11742`–`:11747`

There is no derivation worker **on this branch**. `claim_derivation_jobs` — the table whose lease columns exist precisely so "a crashed worker [is] reclaimable instead of parking a page forever" (`crates/wenlan-core/src/db/claim_identity.rs:304`–`:305`) — has no production reader or writer at `1c903bec`: the only references outside its own DDL at `:306`–`:322` are in test files. One is being built on `m5-truth-derivation`, which is what makes these rows `lane 1` rather than open-ended.

**Consequence for M6.** On any real database on this branch, `support_status = 'supported'` matches zero rows. Predicate 2.5 is therefore not merely unverified — it is false for every page, so **the orphan-wikilink signal cannot admit a single candidate**, and D1's rule R6 ("a stale/provisional referring page contributes nothing") excludes everything. A PR-B genesis shadow for orphan wikilinks would measure exactly zero on every install and read as "the signal is rare" rather than "the substrate is dead" — the same failure mode the eval seed contract exists to prevent elsewhere in this repo.

**Why this is a report and not a blocker I should resolve.** G1 already requires "M5 readiness/cutover at 100%" before M6 code begins, and the sequencing is now known rather than guessed: the derivation worker is being built on `m5-truth-derivation`. The consequence for M6 is unchanged — **PR-B's orphan-wikilink shadow is unmeasurable until that writer lands** — and so is the recommendation that G1's prerequisite check assert a live writer rather than the presence of the table. Artifact 11's S0-153 census is that assertion made concrete.

### 7.2 `pages.kind` is never set on insert, so `kind = 'overview'` cannot identify overview pages

> **Status: RULED 2026-08-01 — option 1 (make `kind` truthful), delivered in two
> steps.** The write-path half is in flight on the kind-fix lane, which makes
> `pages.kind` truthful at insert. The three stale paths it deliberately deferred
> — rename, SOURCE-replace, archive, none of which re-derive `kind` — close in a
> follow-up PR after that lane merges. Until both land, **R5's exclusion predicate
> is the title rule, written literally as `lower(title) <> 'overview'`** — option
> 2 as a named interim rather than as a silent fallback. See S0-164 below.

> **Decision S0-164 *(new rev 8 — applies R-2, ruled 2026-08-01)* — M6 reads
> overview-ness through the title rule until the `kind` column is truthful on
> every path, and does not route on `kind` before then.** The ordering is not
> caution for its own sake; two facts force it.
>
> First, **the title rule is currently the more accurate predicate.** `kind` is
> derived at insert and never re-derived, so a page that is renamed, has its
> source replaced, or is archived carries whatever `kind` it was born with. A
> read-time title predicate cannot go stale that way. Adopting `kind` before the
> stale paths close would trade a correct predicate for an authoritative-looking
> one.
>
> Second, **a drift-guard tooth enforces the ordering mechanically.** The
> kind-fix lane's tooth fails the build if production read routes on any `kind`
> other than `'entity'`. M6 routing on `kind = 'overview'` before that fence
> lifts does not produce a subtle bug; it produces a red build. Whoever lifts the
> fence does it in the follow-up PR that closes the stale paths, in that order,
> and this decision names that as the sequence rather than leaving it to be
> discovered.

**Confirmed.** The `kind` column was added by migration 89 with `DEFAULT 'concept'` and a CHECK over `('entity','concept','source','overview','authored')` (`crates/wenlan-core/src/db.rs:9177`). The only writer of `'overview'` is that migration's own backfill, `CASE WHEN LOWER(title) = 'overview' AND status = 'active' THEN 'overview'` (`crates/wenlan-core/src/db.rs:9254`).

The generic page-creation path does not set `kind` at all. Both arms of the insert in `insert_page` list their columns explicitly and `kind` is absent from both (`crates/wenlan-core/src/db.rs:41400` and `:41407`), so every page created after migration 89 takes the `'concept'` default — **including overview pages**, which `ensure_overview_page` creates through that same path. Only the entity-shadow writer sets `kind` explicitly (`crates/wenlan-core/src/db.rs:9720`).

Production code already works around this. Overview pages are identified by **title**: `OVERVIEW_PAGE_TITLE = "Overview"` (`crates/wenlan-core/src/synthesis/overview.rs:25`), looked up via `find_active_page_id_by_title` at `:75`, `:269`, `:343`, and `:451`, and compared with `eq_ignore_ascii_case` at `:52` and `:304`. The maintenance sweeps filter overviews with `lower(title) != 'overview'` (`crates/wenlan-core/src/db/maintenance_retro_scan.rs:28`; `crates/wenlan-core/src/db/maintenance_duplicate_reads.rs:53`–`:54`, `:70`–`:71`, `:121`–`:122`). No production read site queries `WHERE kind = 'overview'`.

**Consequence for M6.** D1's rule R5 ("overview pages and generated overview evidence never contribute to genesis") and D2.2's "non-overview referring pages" are exclusion predicates protecting the floor from self-reference — an overview page citing its own community must not help mint a page about that community. Implemented as `kind <> 'overview'`, the predicate would admit every overview created since migration 89, which on a young install is all of them. Implemented as `lower(title) <> 'overview'`, it inherits a title convention that any user can defeat by renaming a page.

This one is squarely inside M6's scope to fix, so it went to the reviewer as a decision rather than to another milestone as a dependency. **Ruled 2026-08-01: option 1**, delivered in the two steps §7.2's status block names. The three options are kept below rather than collapsed to the answer, because the sequencing S0-164 fixes is only legible against the option that was chosen and the two that were not:

1. **Make `kind` truthful** — set it at every insert site, backfill by the same title rule, and add a `drift_guard`-style structural test that fails when a new `INSERT INTO pages` omits `kind`. Costs a migration and touches the write path; gives every later milestone a reliable page-kind axis.
2. **Keep the title convention** and define R5 as `lower(title) <> 'overview'`, documenting that the exclusion is defeatable by rename.
3. **Derive overview-ness from M6's own tables** — an overview page is one an M6 overview candidate published — and treat pre-M6 overviews via the title rule.

**Option 1 was recommended and option 1 was ruled.** It is the only option under which R5 means what D1 says it means, the write-path change is small and mechanical, and options 2 and 3 both leave M6 with a self-reference hole that G2 would have to encode as a known exception. It is a schema change on a shared table, which is why it was the reviewer's call rather than mine; the reviewer made it on 2026-08-01. Option 2 survives as the *interim* and only as the interim — that is what S0-164 fixes: the title rule carries R5's exclusion until `kind` is truthful on every path, and not one step past it.

---

## 8. The embedding tie-break rule

> **D2**: Embedding similarity may break an exact tie only. It can never supply a missing floor, group, grounded root, or supported page.

Restated as what an implementation may and may not do:

| May | May not |
|---|---|
| order two candidates that are otherwise **exactly** equal on every admission input — same group count, same root set, same thresholds cleared | contribute to any count in §1 |
| choose between two equally-ranked slot titles once admission has already passed | attach a node that then counts toward the ≥ 5 grounded-node floor (this is S0-16) |
| — | substitute for a missing grounded root, a missing group, or a `provisional` page |
| — | be consulted before the thresholds are evaluated |

**Decision S0-17 — the tie-break is evaluated strictly after admission, on an equal-input set, and its input is recorded in the candidate fingerprint.** Two consequences. First, admission is a pure function of the D2 predicates; embedding state cannot change whether a candidate exists, only which of two identical candidates is ordered first. Second, because a tie-break that changed with the embedding model would silently change candidate identity across a model upgrade, the model version already in D5's fingerprint covers it — no separate mechanism.

**Decision S0-18 — a tie that the embedding does not break is resolved by the deterministic slot ID, ascending.** D2 does not say what happens when the embedding also ties. Leaving it unspecified would make candidate ordering non-deterministic across runs, which G3's replay tests cannot tolerate. The slot ID is already deterministic (D5) and already computed at this point.

---

## 9. Relationship to G2

G2 (`m6_genesis_counts_groups_not_rows`) asserts that "generated roots, chunks, mirrors, inactive/ungrounded external roots, M5-provisional page-mediated inputs, same-session captures, and overview evidence cannot inflate any signal", with positive controls for independent documents and UI-authorized human groups.

Every clause maps onto a row above: generated roots → R2 / B23; chunks and mirrors → R3 / B24; inactive and ungrounded roots → §1's active and grounded qualifiers / B25–B27; M5-provisional inputs → R6 / B11 (`lane 1`, §7.1); same-session captures → R3 and artifact 3 §4; overview evidence → R5 / B12 (ruled, §7.2); positive controls → B29 and B30, read alongside B28.

Two of G2's clauses had no way to fail on the branch as found, because their predicates are constant across all rows (§7.1, §7.2). A gate that cannot fail is not a gate, so G2's RED phase must include the positive control that proves the predicate discriminates — a supported page next to a provisional one, an overview page next to a concept page. The §7.2 half is buildable now: R-2's ruling makes the interim predicate `lower(title) <> 'overview'`, which discriminates on data that exists today, so the overview-versus-concept fixture needs nothing further. The §7.1 half still waits on the promoter, which is exactly what `lane 1` means. That dependency is recorded in artifact 12's mutation catalog.
