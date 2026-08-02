# M6 Stage-0 artifact 5 — frontier policy: differential query, cursor, surfacing, suppression, quarantine, compaction

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: frozen contract D7, plus the D8 caps' interaction with the frontier.
Builds directly on machine F of artifact 2 (`2026-08-01-m6-state-machines.md` §9),
which fixed the six states and twelve transitions; this artifact fixes the
queries, the clocks, the scopes, and the exhaustion behaviour that machine F left
open.

**Approved amendment (2026-08-02, D2=A and D3=A).** Suppression history is
append-only and repeatable, with stored liveness (`lapsed_at IS NULL`), and
card bindings are retained per group with stored liveness
(`closed_at IS NULL`). Every reference below to "exactly one reason" means
exactly one **live** reason; closed/lapsed history remains queryable and does
not count.

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

## 0. The one sentence this artifact has to make checkable

D7's closing rule:

> Cursor wrap, restart, quota exhaustion, and a permanently small space may delay
> work but may never lose or silently park evidence.

Everything below is either a mechanism that makes that true or a test that
detects it being false. §8 is the enumeration the rule demands: every path that
could park evidence, with the guard that stops it.

---

## 1. The differential query

### 1.1 Six states, six durable homes

Machine F's six states are not a `state` column. Each is the presence of a
**live** row in a different table, and "exactly one durable reason" is literally
"exactly one of these six has a live row for this group." Retained closed card
bindings and lapsed suppressions are history, not current reasons.

| State | Durable home | New? |
|---|---|---|
| `covered` | `genesis_group_coverage` | PR-A-new |
| `exclusively_claimed` | `genesis_candidate_roots` with a live reservation | PR-A-new (D4) |
| `waiting_frontier` | `genesis_frontier` | PR-A-new |
| `surfaced_card` | `genesis_card_binding WHERE closed_at IS NULL` → `refinement_queue` (`crates/wenlan-core/src/db/migrations_v004_v009.rs:49`) | binding new; queue **exists** |
| `suppressed` | `genesis_suppression WHERE lapsed_at IS NULL` | PR-A-new |
| `quarantined` | `genesis_quarantine` | PR-A-new |

> **Decision S0-42 — the six states are six tables, not one enum column.** A
> single `state` column would make the invariant true by construction and
> therefore untestable: the row would always have exactly one value, including
> when the *reason* behind it had been rolled back. Six tables make "exactly one"
> a real count over live rows, which is what G5 has to assert. Historical rows
> remain in those same tables but are excluded by their stable liveness marker.
> It also matches what the transitions already are — F4 writes a coverage row, F7 writes a
> suppression row — so no transition gains a second write.

### 1.2 The eligible set

Reusing artifact 1's canonical count expression, restricted to one space:

```sql
WITH eligible AS (
  SELECT DISTINCT r.independence_group_id AS gid
    FROM edges e
    JOIN provenance_roots r ON r.root_id = e.root_id
   WHERE e.space      = :space
     AND e.grounded   = 1
     AND e.valid_until IS NULL
     AND r.status     = 'active'
     AND r.root_kind <> 'generated'
)
```

Grounded in the tree: `edges.space` (`crates/wenlan-core/src/db.rs:8813`),
`edges.grounded` (`:8811`), `edges.valid_until` (`:8820`), `edges.root_id`
(`:8812`), `provenance_roots.status` (`:8793`), `provenance_roots.root_kind`
(`:8791`), `provenance_roots.independence_group_id` (`:8792`).

The partial index `idx_edges_active_grounded_space_type ON edges(space, edge_type)
WHERE valid_until IS NULL AND grounded = 1` (`db.rs:8827`-`:8828`) covers the
`edges` side of the predicate. The join to `provenance_roots` goes through
`idx_edges_root ON edges(root_id) WHERE root_id IS NOT NULL` (`db.rs:8824`) and
then the `provenance_roots` primary key (`db.rs:8788`). **PR-A should measure
this before enabling the scan**: the space filter is index-supported, the
root-kind and status filters are not, so the cost scales with grounded edges in
the space, not with groups.

### 1.3 The exclusivity check — G5's body

```sql
SELECT e.gid,
       (cov.independence_group_id IS NOT NULL)
     + (clm.independence_group_id IS NOT NULL)
     + (fro.independence_group_id IS NOT NULL)
     + (crd.independence_group_id IS NOT NULL)
     + (sup.independence_group_id IS NOT NULL)
     + (qua.independence_group_id IS NOT NULL) AS reason_count
  FROM eligible e
  LEFT JOIN genesis_group_coverage  cov ON cov.independence_group_id = e.gid AND cov.space = :space AND cov.coverage_epoch = :epoch
  LEFT JOIN live_reservations      clm ON clm.independence_group_id = e.gid AND clm.space = :space AND clm.coverage_epoch = :epoch
  LEFT JOIN genesis_frontier       fro ON fro.independence_group_id = e.gid AND fro.space = :space AND fro.coverage_epoch = :epoch
  LEFT JOIN genesis_card_binding   crd ON crd.independence_group_id = e.gid AND crd.space = :space AND crd.coverage_epoch = :epoch
                                      AND crd.closed_at IS NULL
  LEFT JOIN genesis_suppression    sup ON sup.independence_group_id = e.gid AND sup.space = :space AND sup.coverage_epoch = :epoch
                                      AND sup.lapsed_at IS NULL
  LEFT JOIN genesis_quarantine     qua ON qua.independence_group_id = e.gid AND qua.space = :space AND qua.coverage_epoch = :epoch
                                      AND qua.lifted_at IS NULL
 WHERE reason_count <> 1;
```

`live_reservations` is `genesis_candidate_roots` joined to `genesis_candidates`
and filtered to candidates in a non-terminal state — the reservation is only a
*reason* while the candidate is alive, which is exactly what makes F5 (reservation
released → back to frontier) mandatory rather than housekeeping.

The card and suppression predicates are stored markers on purpose. A wall-clock
predicate such as `expires_at > unixepoch()` cannot be a safe partial-index
predicate, and it would let the same committed database change from one reason
to zero without a transaction. The reconciler first stamps `lapsed_at`, then
restores `genesis_frontier`, in one transaction. The partial unique indexes on
each table permit at most one open/unlapsed row per `(space,
independence_group_id, coverage_epoch)`.

**Two failure shapes, two different meanings.**

| `reason_count` | Meaning | Reconciler action | Gate |
|---|---|---|---|
| `0` | The group is eligible and nothing accounts for it — **this is silent parking** | insert a `genesis_frontier` row (transition F1) | G5 fails if a scan ever *finds* a 0 without an explanation; the reconciler's insert is the repair, and the count of repairs is the signal |
| `> 1` | Double-booked — e.g. covered *and* suppressed | **no automatic repair**; refuse and surface | G5 fails hard. Two reasons means two transitions committed that should have been mutually exclusive, which is a transaction-boundary bug and must not be papered over |

> **Decision S0-43 — the reconciler repairs `0` and refuses `> 1`.** A `0` is
> recoverable and its recovery is the frontier's whole job (F1 exists for exactly
> this). A `> 1` is evidence that an invariant already broke, and auto-picking a
> winner would destroy the only trace of which transition was wrong.

### 1.4 Why a query and not a queue

The differential query is total: it derives the frontier from the evidence rather
than maintaining it incrementally. That single choice is what makes most of §8's
guards unnecessary — a lost cursor, a dropped event, a crashed worker, a
never-delivered notification all cost a rescan and none of them can lose a group,
because nothing about the group's presence in the eligible set depends on
anything M6 wrote.

---

## 2. Cursor semantics

### 2.1 What the cursor is

D7's ordering key is `next_scan_at, first_seen_at, group_id`. The supporting
index on `genesis_frontier` is exactly that tuple, so the scan is an index walk.

The cursor is a **scan position within one pass**, and nothing else.

> **Decision S0-44 — the cursor is a pure optimization; correctness never depends
> on it.** Losing it, corrupting it, or resetting it costs one extra pass over
> already-accounted-for groups. This follows from §1.4 and is the reason the
> cursor gets a single `app_metadata` key rather than a table with its own
> integrity story.

### 2.2 Storage and wrap

Storage follows the tree's existing cursor idiom exactly: one `app_metadata`
row (`crates/wenlan-core/src/db.rs:5672`-`:5675`, `key TEXT PRIMARY KEY, value
TEXT NOT NULL`), written after each slice, cleared to the empty string on wrap.
The precedent is the automatic-maintenance cross-space cursor —
`AUTOMATIC_CROSS_SPACE_CURSOR_KEY` (`crates/wenlan-core/src/maintenance.rs:31`),
advanced at `:421` and cleared at `:425` when the slice reports no more work.

> **Decision S0-45 — one key, `m6_frontier_cursor_v1`, value =
> `"<next_scan_at>|<first_seen_at>|<space>|<gid>"`; empty string means "start from
> the beginning".** One key rather than one per space because the ordering key is
> global and a per-space cursor would need its own fairness story between spaces.
> The space is inside the value so the resume point is exact.

**Wrap behaviour.** On reaching the end of the ordering, the cursor is cleared
and the next pass starts over. A wrap is **not** a checkpoint and asserts
nothing: a group inserted mid-pass at an ordering position the scan already
passed is simply seen on the following pass. The frontier row was created by F1
in the meantime, so the group is accounted for the entire time it is waiting —
being *unscanned* and being *unaccounted for* are different conditions, and only
the second one is parking.

### 2.3 `next_scan_at` is bounded

`next_scan_at` delays when a frontier row is *examined*. Unbounded, it is a
parking mechanism: push it forward on every pass and the 7-day timer is never
evaluated.

> **Decision S0-46 — `next_scan_at` may never be set more than 24 hours ahead of
> `unixepoch()`.** The 7-day below-floor timer is therefore evaluated at least
> seven times before it can fire, and no code path can defer a row past its own
> surfacing deadline. Enforced as a CHECK-style guard at every write site plus one
> G5 assertion (`MAX(next_scan_at) - unixepoch() <= 86400`).

---

## 3. Surfacing — the coalesced unformed-topic card

### 3.1 Scope

D7 says below-floor evidence older than 7 days creates **one coalesced**
unformed-topic card. The dispatch asks: one per what?

> **Decision S0-47 — one card per `(space, coverage_epoch)`.**
>
> Per-group is what D7 explicitly rules out ("coalesced"). Per-install would mean
> a user with five spaces gets one card that mixes unrelated work and cannot be
> dismissed for one space without dismissing it for all. Per-space is the unit the
> user actually reasons about, and the epoch is in the key because a contract-version
> epoch is a fresh accounting era (machine D) — a card dismissed under the old
> epoch should not silence the new one.

The card lists the below-floor groups by count and by their nearest-miss floor,
never by content. See §3.4.

The durable binding is one retained row per group and card:
`genesis_card_binding(space, independence_group_id, coverage_epoch, card_id,
page_id, first_seen_at, created_at, closed_at)`. One shared card therefore has
many rows. Only `closed_at IS NULL` rows count in §1.3, and a partial unique
index permits at most one open binding per group/epoch.

### 3.2 Card identity and the write that must not be used

Card ID: `m6_unformed_topic_<space-digest>_<epoch>`, where the space digest is an
`m6_digest` (artifact 4 §2) so a space name never appears in an ID.

The tree has two ways to write into `refinement_queue` and **they behave
differently on a dismissed card**:

| Writer | Statement | Effect on an existing dismissed row |
|---|---|---|
| `insert_refinement_proposal` (`crates/wenlan-core/src/db.rs:36845`, statement at `:36862`) | `INSERT OR REPLACE INTO refinement_queue (id, action, source_ids, payload, confidence)` | **Resurrects it.** `status` is not in the column list, so REPLACE deletes the row and the new one takes the `DEFAULT 'pending'` (`migrations_v004_v009.rs:55`) |
| `insert_lint_review_if_absent` (`crates/wenlan-core/src/db.rs:36871`, statement at `:36882`-`:36884`) | `INSERT OR IGNORE … VALUES (…, 'awaiting_review')` | **Leaves it alone.** Its doc comment (`:36868`-`:36870`) says so: *"`INSERT OR IGNORE` deliberately never resurrects a dismissed item."* |

> **Decision S0-48 — the unformed-topic card is written with `INSERT OR IGNORE`
> and an explicit `status`, never through `insert_refinement_proposal`.** A card
> the user dismissed must stay dismissed; re-emitting it every scan is the exact
> nagging failure D7's suppression window exists to prevent. See finding F1 —
> the existing cross-space discovery card guards this at the *caller*
> (`maintenance.rs:844`-`:846`, a check-then-insert) rather than in the writer,
> which is a race M6 should not copy.

### 3.3 The clock

`refinement_queue.created_at` is `TEXT DEFAULT (datetime('now'))`
(`migrations_v004_v009.rs:56`) — a text timestamp. Every M6 timer is an integer
`unixepoch()` per S0-11 ("one clock, and a test can move it only by moving stored
values").

> **Decision S0-49 — every M6 clock lives on an M6 table as an INTEGER
> `unixepoch()` column; `refinement_queue.created_at` is never read as a clock
> input.** The frontier row owns `first_seen_at`; the suppression row owns
> `expires_at`; the candidate row owns `payload_compacted_at`. The card is a
> *projection* of a decision M6 already made, not the record of when M6 made it.
> Mixing the two clock formats is precisely what S0-11 forbids, and it would make
> the 7-day timer untestable by the value-moving technique every other M6 timer
> uses.

### 3.4 What the card may say

Permitted: the space (as displayed to its owner), the count of below-floor
groups, how many groups short of the floor the nearest one is, the epoch, and the
allowed actions.

Forbidden, per D13: any root content, any wikilink label, any capability
material, any user identifier, any prompt text. The card says *"3 topics in this
space have evidence from only 2 independent sources"* — never what the topics
are. This is not a UX preference; the card is durable, exportable content and D13
governs it.

### 3.5 The below-floor timer does not reset on re-entry

The 7-day timer starts at `genesis_frontier.first_seen_at`. Transitions F5
(reservation released), F8 (card expired), F11 (suppression lapsed), and F12
(quarantine lifted) all re-insert a frontier row.

> **Decision S0-50 — `first_seen_at` is preserved across every re-entry within a
> coverage epoch; only F1's genuinely-first insertion sets it.** If re-entry reset
> the clock, a group that is repeatedly claimed and released — which is the normal
> shape of retry-then-fail — would never reach 7 days and would never surface.
> That is silent parking dressed as activity, and it is the most likely way to
> violate D7 while every individual transition looks correct. A new coverage epoch
> does reset it, because the epoch is a new accounting era by construction.

### 3.6 Shared-card closure

Dismissal reads every live binding for the shared `card_id`, closes **all** of
them, and only then writes each group's post-card transition. Closure and all
replacement reasons share one transaction. A crash after all bindings close or
after any individual suppression insert rolls the whole transaction back, so
every group still has its open card reason; a commit leaves every binding
closed and every group with its suppression reason. Binding rows are retained
after closure.

---

## 4. Suppression (180 days)

Written by F7 when a candidate is suppressed or a human dismisses a card.

| Property | Value |
|---|---|
| Row | `genesis_suppression(space, independence_group_id, coverage_epoch, page_id, reason, first_seen_at, suppressed_at, expires_at, lapsed_at)` |
| Clock | `expires_at = unixepoch() + 180*86400`, set in-statement (S0-11) |
| Lapse | F11: the reconciler observes `expires_at <= unixepoch()`, stamps `lapsed_at`, then re-inserts the frontier row in the same transaction |
| Identity | `page_id`, durable **forever**, past both lapse and compaction (D7, D14) |

> **Decision S0-51 — lapse is a stored marker, not deletion.** The join in §1.3
> carries `AND sup.lapsed_at IS NULL`; `expires_at <= unixepoch()` makes a row
> eligible for reconciliation but does not itself change liveness. The
> reconciler stamps `lapsed_at` before restoring frontier, in the same
> transaction. Deleting the suppression would erase the record that a human
> already said no once, which D7 ("suppression identities remain durable") and D14
> (forward-safe rollback) both require to survive.

A consequence worth stating: there is no zero-reason wall-clock edge at the
180-day mark. Until the next differential reconciliation, the unlapsed row is
still the one live reason. That reconciliation commits the marker and frontier
replacement together. A later suppression in the same epoch appends a new row
with a new `suppressed_at`; the lapsed identity remains queryable, and the
partial unique index still allows at most one unlapsed row.

---

## 5. Quarantine

The only one of the six states that is never entered automatically.

> **Decision S0-52 — `genesis_quarantine.reason` is `TEXT NOT NULL` with a
> `CHECK(length(trim(reason)) > 0)`, and there is no code path that inserts a
> default reason.** D7 says the quarantine reason must be explicit. A nullable or
> defaultable reason column is how "explicit" decays into "whatever the last
> handler happened to pass", and a group can then arrive in quarantine by
> omission — which is parking with extra steps.

The retained row is
`genesis_quarantine(space, independence_group_id, coverage_epoch, reason,
first_seen_at, quarantined_at, lifted_at)`, keyed by `(space,
independence_group_id, coverage_epoch, quarantined_at)`. Only
`lifted_at IS NULL` counts in §1.3, and a partial unique index permits at most
one live quarantine per group/epoch. Lifting (F12) stamps `lifted_at` before
restoring frontier in the same transaction; the row is retained and a later
quarantine appends a new history row, mirroring §4.

---

## 6. Compaction (90 days)

D7: terminal candidate payloads compact after 90 days; receipts, page genesis,
human decisions, and suppression identities remain durable.

| Compacted | Retained forever |
|---|---|
| `genesis_candidates.payload` (the inference input/output blob) | the candidate row itself: ID, slot ID, page ID, state, reason, timestamps |
| — | `operation_receipts` rows (`db.rs:8217`-`:8224`) |
| — | the published page and its provenance |
| — | human decisions (card dismissals, reviews) |
| — | suppression identities (§4) |

> **Decision S0-53 — compaction sets `payload = NULL` and stamps
> `payload_compacted_at = unixepoch()`; it never deletes a candidate row.** The
> row is what carries the candidate's *reason*, and §1.3 counts reasons. Deleting
> a terminal candidate row would drop a group's `exclusively_claimed` or coverage
> accounting to `reason_count = 0`, and the reconciler would faithfully re-insert
> a frontier row for a group that was already handled — an infinite rediscovery
> loop caused by a retention policy. The stamp is what makes "compacted" and
> "never had a payload" distinguishable.

Eligibility: `state IN (terminal set) AND state_entered_at <= unixepoch() -
90*86400 AND payload IS NOT NULL`. The terminal set is machine A's, from artifact 2.

---

## 7. Quota exhaustion

D8's caps, and what each does to the excess. **In every row, the excess stays in
`waiting_frontier`** — that is the whole content of D8's "excess work remains
frontier-visible."

| Cap | Value | On hit | Where the excess goes |
|---|---|---|---|
| eligible links per page | 64 | links past the 64th are not counted toward the orphan-wikilink signal | the page's other links are unaffected; nothing is written, so nothing is parked |
| roots per candidate | 64 | the candidate carries 64 roots; further roots do not join this candidate | their groups are **not** reserved by it, so they stay eligible and stay in the frontier |
| pending candidates per space | 128 | no new candidate is prepared for this space this cycle | every unprepared group keeps its frontier row; `next_scan_at` advances by the S0-2 backoff, bounded by S0-46 |
| candidate prepares per space per cycle | 16 | the 17th group is not prepared | same as above |
| automatic LLM finalizations per refinery turn | 1 | the second finalization gets `LlmError::NotAvailable` from `AmbientBudgetProvider` (`crates/wenlan-server/src/scheduler.rs:455`, `:495`-`:503`) | the candidate stays in its pre-finalization state; its reservation still holds, so the group reads `exclusively_claimed`, not `0` |

> **Decision S0-54 — no cap may write a terminal candidate state or retire a
> frontier row.** A cap is a *rate* limit, and converting a rate limit into a
> terminal outcome is the single most natural way to violate D7 while looking
> like good hygiene ("we were over quota, so we dropped it"). Every cap's only
> permitted effects are: don't start new work, and advance `next_scan_at`.

The roots-per-candidate row deserves the extra sentence: a candidate capped at 64
roots reserves 64 groups and leaves the rest eligible. Those leftover groups can
be picked up by a *different* candidate for a different slot, which is correct —
they are independent evidence that this candidate did not consume.

---

## 8. Every path that could park evidence

D7's closing rule, enumerated. "Parked" means the group is eligible and
`reason_count = 0` with nothing scheduled to change that.

| # | Path | How it would park | Guard | Guard lives in |
|---|---|---|---|---|
| P1 | Cursor lost or corrupted | scan resumes past the group | the differential query is total (§1.4); the cursor only picks a starting point | S0-44 |
| P2 | Cursor wrap mid-insert | group inserted at an already-passed position | the frontier row exists from F1 onward; unscanned ≠ unaccounted | §2.2 |
| P3 | `next_scan_at` pushed forward repeatedly | 7-day timer never evaluated | 24h ceiling on `next_scan_at` | S0-46 |
| P4 | Crash between claim and publish | reservation row exists, candidate is dead | S0-5's recovery scan fires F5 for every orphaned reservation | artifact 2 |
| P5 | Lease held by a dead process | no worker can claim the group | `grouping_leases.expires_at` + the reap arm (`crates/wenlan-core/src/db.rs:13432`-`:13434`) | existing M4 substrate |
| P6 | Retry exhaustion | candidate stops retrying | S0-12: `attempt > 5` → `stale` with `reason='retry_exhausted'`, which releases the reservation (F5) and returns the group to the frontier | artifact 2 |
| P7 | Quota exhaustion (any of 5 caps) | work not started | no cap may terminalize or retire | S0-54 |
| P8 | Suppression never lapsing | group suppressed forever | `expires_at` is set in-statement; reconciliation stamps `lapsed_at` before restoring frontier, and the live join uses that stable marker | §4, S0-51 |
| P9 | Quarantine never lifted | group quarantined forever | quarantine requires an explicit human/policy reason and is visible as a distinct state in the differential query; lift stamps the stable marker before restoring frontier, and reactivation appends history | S0-52 |
| P10 | Below-floor forever in a small space | group never reaches the admission floor | the 7-day card fires regardless of floor (F3); this is G5's positive control | §3 |
| P11 | Card dismissed, never re-emitted | user dismissed once, silence forever | dismissal atomically closes every live binding for the shared card and writes one suppression per group; §4's marked lapse returns each group | §3.6 + §4 |
| P12 | Compaction deletes the reason | terminal candidate row removed → `reason_count = 0` → rediscovery loop | compaction nulls the payload only | S0-53 |
| P13 | Timer reset by re-entry churn | claim/release cycling keeps resetting the 7-day clock | `first_seen_at` preserved across re-entry | S0-50 |
| P14 | Epoch change disables genesis | new-epoch genesis blocked pending migration (D5) | the block is on *new genesis*, not on the frontier; groups keep their frontier rows and the differential query keeps running under the old epoch until the migration completes | machine D, artifact 2 |
| P15 | Group's roots all go `status='failed'` | group drops out of the eligible set | **not parking** — the group is no longer eligible, so there is no evidence to lose. Recorded here so a reader does not mistake the drop for a leak | §1.2 |
| P16 | Mirror collapse (F10) | group marked covered without a candidate | the coverage row is a real durable reason, and D4 explicitly wants this: *"a durable group-coverage row makes future mirrors of an already-covered group covered immediately"* | artifact 2 F10 |

P9 and P15 are the two rows where the honest answer is "this is not a violation."
Both are included because a reviewer checking the enumeration for completeness
will look for them, and an enumeration that silently omits its non-violations
reads as an enumeration that missed them.

---

## 9. Findings against the tree

**F1 — the existing discovery-card guard is a check-then-insert, not an atomic
write.** `emit_cross_space_discovery_card`
(`crates/wenlan-core/src/maintenance.rs:834`-`:865`) reads
`get_refinement_proposal(&id)` at `:844` and returns early if a row exists, then
calls `insert_refinement_proposal` at `:854`, whose statement is
`INSERT OR REPLACE` (`db.rs:36862`). Two concurrent emitters between the check and
the insert would replace a dismissed card back to `pending`. The daemon is the
single writer, so this is currently unreachable rather than broken — but M6 must
not copy the shape, and the `INSERT OR IGNORE` writer at `db.rs:36882` is the one
to copy. S0-48.

**F2 — `refinement_queue` timestamps are TEXT, not `unixepoch()` integers**
(`crates/wenlan-core/src/db/migrations_v004_v009.rs:56`, `created_at TEXT DEFAULT
(datetime('now'))`). Not a bug in its own context; it is a hazard the moment an
M6 timer is tempted to read it. S0-49 closes it by putting every M6 clock on an
M6 table.

**F3 — `refinement_queue.status` has no CHECK constraint**
(`migrations_v004_v009.rs:55`, `status TEXT DEFAULT 'pending'`), while the code
uses at least `pending`, `awaiting_review`, `resolved`, and `dismissed`
(`db.rs:16201`, `:16341`, `:16483`). M6's card binding should not depend on the
status string being well-formed; the binding row carries M6's own view of whether
the card is open. Reported, not resolved — adding a CHECK to a live table is a
migration with its own risk and is outside M6's scope.

**F4 — the eligible-set query has no covering index for the root-side
predicates.** `provenance_roots.status` and `root_kind` are unindexed
(`db.rs:8791`, `:8793`; the only indexes on that table are the primary key at
`:8788` and the `UNIQUE(identity_version, identity_digest)` at `:8795`). At
current corpus sizes this is almost certainly fine; PR-A should measure before
enabling the scan, because the M4 parity sweep's 18.88s single-connection hold at
10k entities (documented in `crates/wenlan-core/AGENTS.md`) is the precedent for
how a full-pass scan degrades foreground latency on this daemon's one connection.

---

## 10. Gate mapping

| Gate | What this artifact hands it |
|---|---|
| G5 (`m6_frontier_has_no_missing_root`) | §1.3's live-row query verbatim as the assertion body; the `0` vs `> 1` split (S0-43) as two distinct failure modes; suppress → lapse → frontier → suppress again while both histories remain queryable; shared-card failure injection after closure and after every replacement reason; S0-46's `next_scan_at` ceiling as a second assertion |
| G6 (abuse bounds) | §7's table — one case per cap, each asserting the excess is still in `waiting_frontier` |
| G-catalog | P1–P16 (§8) as one crash/adversarial case each; P10 as the positive control for a permanently small space; P12 as the retention-policy regression test |
| G10 (epoch) | P14: the frontier keeps running under the old epoch while new-epoch genesis is blocked |

---

## 11. Decisions introduced here

`S0-42` six states are six tables, and G5 counts exactly one live row rather than all retained history ·
`S0-43` reconciler repairs `reason_count = 0`, refuses `> 1` ·
`S0-44` the cursor is a pure optimization ·
`S0-45` one `app_metadata` cursor key, cleared to `""` on wrap ·
`S0-46` `next_scan_at` capped at 24h ahead ·
`S0-47` one unformed-topic card per `(space, coverage_epoch)` ·
`S0-48` card written with `INSERT OR IGNORE`, never `INSERT OR REPLACE` ·
`S0-49` every M6 clock is an INTEGER `unixepoch()` on an M6 table ·
`S0-50` `first_seen_at` survives re-entry within an epoch ·
`S0-51` suppression is repeatable append-only history; the reconciler marks expiry with `lapsed_at` before restoring frontier ·
`S0-52` quarantine reason is non-empty and never defaulted; retained rows use `lifted_at` as stable liveness and may repeat ·
`S0-53` compaction nulls the payload, never deletes the row ·
`S0-54` no cap may terminalize a candidate or retire a frontier row.
