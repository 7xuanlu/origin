# M6 PR-B — genesis shadow implementation spec

Status: draft spec, pre-implementation. Written against worktree HEAD `c620d7e2`
(main tip). Nothing in this document has been built or run; every code reference
was read on that tree.

Normative sources, in precedence order:

1. `2026-07-27-kg-m6-goal-prompt.md` (the frozen M6 goal prompt, in the
   `wenlan-app` repo under `docs/superpowers/plans/`) — "## PR-B — genesis
   shadow", "## Stage 0", the RED-first gate rules, and the rollback rules.
2. The twelve Stage-0 contract artifacts, `docs/plans/2026-08-01-m6-*.md`.
3. `docs/plans/2026-08-02-m6-pr-a-followup-schema.md` (PR-A receipt).
4. Shipped code: `crates/wenlan-core/src/m6/`, migrations 108/109, the M5
   promoter loop in `crates/wenlan-server/src/main/runtime.rs`.

Where an artifact and the shipped code disagree, §11 records it rather than
resolving it.

---

## 1. Scope and non-goals

### 1.1 In scope

PR-B builds the **genesis shadow**: the complete machinery that reconciles the
four D2 signals into deterministic candidates, reserves independence-group
claims, drives the durable frontier, and runs finalization all the way to the
last CAS gate — and then **stops**, one step before the model call and one step
before any page write. It produces evidence, not pages.

Concretely, PR-B ships:

- The four signal readers and the one canonical independence-count expression.
- The candidate state machine (A), the claim/reservation machine (B), the
  coverage-epoch machine (D), and the frontier half of machine F, wired to the
  already-staged writers in `crates/wenlan-core/src/m6/frontier_policy.rs`.
- One M6 phase-lease facade over the **existing** `grouping_leases` registry
  (D6: no parallel lease system).
- A dry-run finalization that verifies all eight CAS inputs, records the
  verdict, and performs **no LLM call and no page write**.
- An incremental-vs-full recomputation oracle plus mutation oracles.
- Per-space readiness and benchmark evidence recording.
- One shadow loop in the daemon, beside (never inside) the M5 truth loop.

### 1.2 Non-goals — explicit, and each one a review tripwire

| Not in PR-B | Why | Lands in |
|---|---|---|
| Any write to `pages`, `page_sources`, `page_links`, `entities`, `relations`, `communities`, `community_members` | Shadow means advisory; user-visible behavior must be byte-identical | PR-E1…E4 |
| Any LLM call on the genesis path | D6's "≤1 LLM finalization per refinery turn" is a cap on a thing that does not yet exist here | PR-E1…E4 |
| `page_projection_outbox` rows in any state other than absent | Machine E's publish half | PR-E1 |
| Enabling genesis for any space (`genesis_coverage_state.genesis_enabled = 1`) | PR-B ships the machinery, not the enablement | PR-E1…E4, gated by readiness |
| Relevance scoring (`m6_pair_stats`, `m6_adjacency`) | Bounded relevance is PR-C's contract | PR-C |
| Refresh jobs (`genesis_refresh_jobs`, `m6_refresh_dependencies`) | Guarded refresh is PR-C's contract | PR-C |
| Overview subscription lifecycle writes | Overview identity is D5/PR-C | PR-C |
| Any binding to strict whole-page visibility suppression | **That product decision is PAUSED product-wide.** Genesis activation must not depend on it, must not read its state, and must not be blocked by its absence | — |
| Any new production read route on `kind='overview'` (or any non-`entity` kind) | `drift_guard` teeth #16 still fences it; see §3.2 R5 | after M6 re-derives `kind` on every mutation path |
| New migrations, unless proven necessary | 108 + 109 are believed complete; see §2.7 | — |

### 1.3 The one-sentence contract

> After PR-B, on every real install, the genesis substrate is fully populated
> with candidates, claims, coverage and frontier rows that a later cutover can
> publish from — and a byte-for-byte diff of every user-visible surface against
> the pre-PR-B build is empty.

---

## 2. The six contract items, mapped to code

Legend: **[wire]** = a staged PR-A API exists, PR-B calls it; **[new]** = new
code; **[read]** = existing production code PR-B reads and must not change.

### 2.1 Item 1 — reconcile all four genesis signals into deterministic dry-run candidates

| Piece | File | Status |
|---|---|---|
| Canonical distinct-group count expression | `crates/wenlan-core/src/m6/independence.rs` | **[new]** |
| Four signal readers → `Vec<CandidateProposal>` | `crates/wenlan-core/src/m6/signals.rs` | **[new]** |
| Durable community gate for signals 1 and 3 | `community_reader_durable_gate_sql`, `crates/wenlan-core/src/db.rs:2663` | **[read]** |
| `slot_id` / `page_id` / `candidate_id` / `active_root_digest` / `CandidateFingerprint` | `crates/wenlan-core/src/m6/identity.rs` | **[wire]** |
| Length-prefixed digest primitives | `crates/wenlan-core/src/m6/digest.rs` | **[wire]** |
| Domain tags, signal tags, label scalar caps | `crates/wenlan-core/src/m6/constants.rs` | **[wire]** |
| D8 wikilink normalization | `normalize_label_key`, `crates/wenlan-core/src/m6/label_key.rs` | **[wire]** |

`signals.rs` is a **pure reader**: it takes a `&libsql::Transaction` and a space,
and returns proposals. It opens no transaction and writes nothing. Determinism
is structural — every proposal's identity is a digest of its inputs, and the
admission order is `(admitted, tie-break, slot_id ASC)` per S0-17/S0-18, so two
runs over identical state produce an identical proposal list in an identical
order. That property is what §6's oracle compares.

### 2.2 Item 2 — populate frontier and coverage/witness claims through durable leases

| Piece | File | Status |
|---|---|---|
| M6 phase-lease facade (acquire / verify / release, phases `genesis`, `frontier`) | `crates/wenlan-core/src/m6/leases.rs` | **[new]**, thin |
| Underlying lease registry: DDL `db.rs:10554`, reap `13880`, acquire `13893`, release `14203`, ownership CAS `14373`, consume `14584` | `crates/wenlan-core/src/db.rs` | **[read]** — extended only by new `phase` **values**, never a new table |
| Candidate machine A + reservation machine B + coverage machine D | `crates/wenlan-core/src/m6/candidates.rs` | **[new]** |
| Frontier differential query, cursor, six reconciliation states | `crates/wenlan-core/src/m6/frontier.rs` | **[new]** |
| `suppress_frontier_group` (F7), `bind_frontier_groups_to_card` (F3), `reconcile_expired_suppressions` (F11), `quarantine_frontier_group` (F9), `lift_quarantine_to_frontier` (F12), `dismiss_card_to_suppression` | `crates/wenlan-core/src/m6/frontier_policy.rs` | **[wire]** — these are exactly the writers PR-A's module doc says "PR-B/PR-C wire" |
| `genesis_candidates`, `genesis_candidate_roots` + the two partial unique indexes, `genesis_coverage_state`, `genesis_group_coverage`, `genesis_frontier` | `crates/wenlan-core/src/db/genesis_schema.rs` (migration 108) | **[read]** — DDL already correct |

`frontier_policy.rs` owns no transaction; every entry point takes a
caller-owned `&libsql::Transaction`. PR-B's `frontier.rs` and `candidates.rs`
must preserve that discipline: the **turn driver** owns the transaction, the
writers are transaction-scoped, and I-5 (no guard spans a model call) is
trivially satisfied because PR-B makes no model call at all.

### 2.3 Item 3 — enforce every D1/D2/D4/D7/D8 predicate

See §3. One enforcement point per predicate; a predicate with two enforcement
points is a review finding, not a belt-and-braces bonus (I-2's lesson: the two
partial unique indexes are the *entire* exclusion mechanism, and a duplicate
code-level check would be a second, weaker one).

### 2.4 Item 4 — dry-run finalization verifying all CAS inputs, no LLM, no page

| Piece | File | Status |
|---|---|---|
| The eight CAS gates E-1…E-8, in order, plus the E3 rollback | `crates/wenlan-core/src/m6/finalize.rs` | **[new]** |
| M5 preconditions consulted by E-8 (`evaluate_page_support` `crates/wenlan-core/src/db/claim_derivation.rs:3590`, `page_truth_state.support_status`) | `crates/wenlan-core/src/db/claim_derivation.rs` | **[read]** |
| M4 community generation consulted by E-7 | `crates/wenlan-core/src/db.rs:2663` | **[read]** |

See §5 for the gate list and exactly what "dry run" means at each one.

### 2.5 Item 5 — compare incremental state to full recomputation and mutation oracles

| Piece | File | Status |
|---|---|---|
| Full recomputation from scratch + structural diff against incremental state | `crates/wenlan-core/src/m6/oracle.rs` | **[new]** |
| Mutation oracle harness (the RED half of every gate in §8) | `crates/wenlan-core/tests/m6_genesis_shadow.rs` | **[new]** |

See §6.

### 2.6 Item 6 — record readiness and benchmark evidence per space

| Piece | File | Status |
|---|---|---|
| `initialize_readiness`, `readiness_fence`, `transition_is_legal`, `transition_readiness`, `ReadinessPhase`, `SoakEvidence`, `record_soak_receipt` | `crates/wenlan-core/src/m6/refresh_readiness.rs` | **[wire]** |
| Per-space shadow statistics + benchmark receipt recording | `crates/wenlan-core/src/m6/evidence.rs` | **[new]** |
| Monotone `m6_mutation_count` (the zero-mutation proof) and `m6_counters` | `crates/wenlan-core/src/m6/remaining_substrate.rs` | **[wire]** — read for the proof, and the *point* is that PR-B never increments it |

See §7.

### 2.7 Migration posture

**Expectation: no new migration.** Migration 108
(`crates/wenlan-core/src/db/genesis_schema.rs`) creates the five genesis
objects; migration 109 (`crates/wenlan-core/src/m6/frontier_policy.rs`,
`refresh_readiness.rs`, `overview_subscriptions.rs`, `remaining_substrate.rs`)
creates fourteen tables plus two coverage columns, per the PR-A receipt
inventory.

PR-B must **verify this before writing a line of DDL** by enumerating every
column its writers touch against the shipped DDL. If a gap is found:

- The migration must be **additive and inert** — new table or new nullable
  column only; no backfill that changes an existing read; no new trigger on an
  existing table; no index that changes an existing query plan on a production
  read path.
- The PR body must carry a one-paragraph justification naming the Stage-0
  decision the gap violates, so the gap is recorded as a PR-A miss rather than a
  PR-B design choice.
- A migration that is not additive-and-inert is out of scope: it belongs in a
  separate PR-A follow-up with its own review.

---

## 3. The D-predicates, concretely

### 3.1 D1 — the relaxed independence floor

**Predicate.** A candidate is admissible only if the count of **distinct
independence groups** over its supporting evidence is **≥ 3**.

**The count expression** (canonical, artifact 1; one home in
`crates/wenlan-core/src/m6/independence.rs`, every signal calls it):

```sql
COUNT(DISTINCT r.independence_group_id)
  FROM edges e JOIN provenance_roots r ON r.root_id = e.root_id
 WHERE e.grounded = 1 AND e.valid_until IS NULL
   AND r.status = 'active' AND r.root_kind <> 'generated'
   AND <per-signal scope>
```

Each conjunct is load-bearing and each is a separate mutation row in §8:

- **R1 — groups, not rows.** `COUNT(DISTINCT independence_group_id)`. Chunks,
  mirrors, and same-session captures collapse through the group, so ten chunks
  of one document is one group. Counting `root_id` or `edge_id` is the classic
  inflation bug (G2's whole reason for existing).
- **R2 — active roots only.** `r.status = 'active'`. A retracted root's group
  does not count.
- **R3 — grounded, live edges only.** `e.grounded = 1 AND e.valid_until IS NULL`.
- **R4 — generated roots count zero.** `r.root_kind <> 'generated'`. This is the
  no-self-bootstrapping rule: M6's own output can never justify more M6 output.
  Overview pages and generated overview evidence never contribute, under any
  signal.
- **R5 — overview pages never contribute.** Enforced as
  `lower(p.title) <> 'overview'` per **S0-164**, *not* as `p.kind <> 'overview'`.
  `drift_guard` teeth #16 (`crates/wenlan-core/src/drift_guard.rs:10657`) forbids
  routing a production read on a non-`entity` page kind; `kind` is stamped at
  insert (`page_kind_for`, `crates/wenlan-core/src/db.rs:41904`) and repaired by
  migration 107, but rename/archive/replace still never re-derive it. Routing on
  `kind` is a red build, not a subtle bug.
- **R6 — unknown independence cannot auto-publish.** A root whose group cannot
  be assigned routes to human review; in shadow that means the candidate reaches
  `review_required` (machine A) and is recorded, never admitted.
- **R7 — human capture counts, but as one group.** UI-authorized human capture
  and correction groups count toward the floor. Per artifact 3 §7.2, *all* human
  authorship collapses into the single group `human:local`, so three human
  deltas can never clear a 3-group floor on their own (boundary case B28). That
  is the contract behaving correctly, not a bug to work around.
- **R8 — page-mediated inputs inherit M5 support.** For any input reached
  through a page (orphan-wikilink), the referring page must be the **current
  `supported` version**, and the underlying groups are the union of active
  grounded external roots supporting the **exact current claim revisions that
  contain the link** (S0-15) — not the whole page's evidence.

**Enforcement point.** `independence::distinct_group_count(tx, scope)` —
a single function returning `i64`, with the scope supplied per signal. Every
admission decision in `signals.rs` reads that one function. There is no second
count anywhere in the M6 tree.

### 3.2 D2 — the four signals and their thresholds

All four evaluate against the same D1 count function; they differ only in scope
and in their extra structural conditions.

| # | Signal | Structural precondition | Group floor | Extra |
|---|---|---|---|---|
| 1 | `evidence-cluster` | membership in a **current durable** M4 community | ≥ 3 | grounded nodes are `community_members.attachment = 'core'` **only** (S0-16) |
| 2 | `orphan-wikilink` | ≥ 2 **distinct** referring pages, each active, `supported`, and non-overview | ≥ 3 underlying | link target normalized by D8; underlying groups scoped per R8 |
| 3 | `community-overview` | a **current published** community, and **no active overview subscription** for it | ≥ 3 | ≥ 5 grounded nodes |
| 4 | `space-overview` | **no active space-overview subscription** for the space | ≥ 3 | ≥ 5 grounded nodes |

- "Current durable M4 community" is the fail-closed
  `community_reader_durable_gate_sql` (`crates/wenlan-core/src/db.rs:2663`):
  `published_generation` match **and** `dirty = 0` **and**
  `grouping_generation = published_generation` **and** a
  `community_publication_receipts` row (S0-13). PR-B calls the existing gate SQL;
  it does not re-express the four conjuncts.
- **Embeddings break exact ties only.** They never admit and never reject. If an
  embedding value is unavailable, admission is unchanged and the fallback
  tie-break is `slot_id` ascending (S0-18).
- Signals 3 and 4 use **non-exclusive witness roots** (D4), so their claim rows
  carry `claim_role = 'witness'`.

**Enforcement point.** One function per signal in `signals.rs`, each returning
`Option<CandidateProposal>` per candidate slot. The threshold constants live in
`crates/wenlan-core/src/m6/constants.rs` (PR-B adds them; PR-A deliberately left
them out because it had no consumer).

### 3.3 D4 — exclusive concept claims vs non-exclusive witness roots

**Predicate.** Within one `coverage_epoch`:

- At most one **active** `claim_role = 'concept'` row per `root_id`.
- At most one **active** `claim_role = 'concept'` row per
  `independence_group_id`.
- `claim_role = 'witness'` rows are unconstrained by both, and a witness row
  must **never** be readable as coverage (I-3).
- Every group **outside the waiting frontier** must have **exactly one** durable
  reason. Zero reasons ⇒ repair by inserting a frontier row. Two or more ⇒
  refuse and surface (S0-43).

**Enforcement point.** The two partial unique indexes in migration 108,
verbatim:

```sql
CREATE UNIQUE INDEX idx_genesis_root_claim
    ON genesis_candidate_roots(root_id, coverage_epoch)
    WHERE claim_role = 'concept' AND released_at IS NULL;
CREATE UNIQUE INDEX idx_genesis_group_claim
    ON genesis_candidate_roots(independence_group_id, coverage_epoch)
    WHERE claim_role = 'concept' AND released_at IS NULL;
```

PR-B's reservation code inserts and lets SQLite refuse. It must **not**
pre-check with a `SELECT` and skip the insert — that converts a database
guarantee into a TOCTOU race, and it is the exact shape the PR-A review round-2
"two-strike" finding rejected. The `released_at IS NULL` liveness marker is a
**stored** bit (S0-6), reconciled only by the recovery scan, because a SQLite
partial index may only reference columns of its own table.

Reservation terminal semantics per exit are machine B's table (B1–B8); the exit
matrix in §8 asserts them as a total function.

`genesis_group_coverage` makes durable group coverage permanent within an epoch,
so a future mirror of an already-covered group is covered **immediately** rather
than re-entering the frontier. No transition deletes a coverage row and the
epoch never decreases (I-7).

### 3.4 D6 — one durable lease registry

**Predicate.**

- There is exactly **one** durable phase-lease table: the physical M4
  `grouping_leases` (`crates/wenlan-core/src/db.rs:10554`), extended with the new
  `phase` **values** `genesis` and `frontier`. Creating a parallel lease system
  is a contract violation.
- A manual call and a scheduler call for the same
  `(phase, space, input_generation)` **join the same operation** — they do not
  race, and the loser observes the winner rather than duplicating work.
- **≤ 1 LLM finalization per refinery turn.** In PR-B this cap is satisfied
  vacuously (zero calls), and the gate that asserts it is written now so it
  cannot regress silently when PR-E makes it non-vacuous.

**Enforcement point.** `crates/wenlan-core/src/m6/leases.rs` — a facade with no
storage of its own, delegating to the existing acquire/release/CAS/reap
statements. TTLs per S0-3: frontier 120s, relevance 300s, genesis 900s, refresh
900s. **No renewal** (S0-4): a lease either completes inside its TTL or is
reaped. All clocks are SQLite `unixepoch()` evaluated in-statement (S0-11) — no
Rust-side `chrono::Utc::now()` on any M6 lease or frontier path.

### 3.5 D7 — the durable frontier

**Predicate.**

- The frontier is keyed `(space, independence_group_id, coverage_epoch)` and
  ordered `next_scan_at, first_seen_at, group_id` — exactly
  `idx_genesis_frontier_scan`.
- Six reconciliation states, per machine F.
- Below-floor evidence **older than 7 days** produces **one coalesced
  unformed-topic card**, one per `(space, coverage_epoch)` (S0-47).
- Suppression lasts **180 days** (`SUPPRESSION_SECONDS` in
  `crates/wenlan-core/src/m6/frontier_policy.rs`).
- Terminal payloads **compact after 90 days** — `payload` is nulled and
  `payload_compacted_at` stamped (S0-10, S0-53). The row is never deleted.
- **No cap may terminalize** (S0-54). Cursor wrap, restart, quota exhaustion,
  and a small space may all *delay* evidence; none may lose it or silently park
  it. Artifact 5's P1–P16 is the complete enumeration of paths that could park
  evidence, and every one of them is a mutation row in §8.

**Enforcement point.** The differential query in
`crates/wenlan-core/src/m6/frontier.rs` is a **total** query over eligible
groups, LEFT JOINing the six durable homes and computing `reason_count`. It is
not a delta feed — totality is what makes "exactly one durable reason" checkable
rather than assumed. Cursor state is the single `app_metadata` key
`m6_frontier_cursor_v1` with value `"<next_scan_at>|<first_seen_at>|<space>|<gid>"`
(S0-44/S0-45). `next_scan_at` is never set more than 24h ahead (S0-46). All
frontier inserts are `INSERT OR IGNORE`, never `INSERT OR REPLACE` (S0-48), so
`first_seen_at` survives re-entry (S0-50).

### 3.6 D8 — normalization and hard caps

**Normalization pipeline** (`normalize_label_key`,
`crates/wenlan-core/src/m6/label_key.rs`, already shipped — PR-B wires it):

pre-cap 1024 scalars → NFKC → lowercase → NFKC again → structural reject
(`#`, `|`, `[`, `]`) → whitespace collapse → reject any `Cc`/`Cf` scalar →
length must be `1..=128`.

Rejections `R0` (too long raw), `R1` (structural), `R2` (control/format),
`R3` (length) **drop the link and never fail ingest**. A rejected link is
invisible to the orphan-wikilink signal; it is not an error.

**Hard caps** (all in `constants.rs`, all enforced at the read that produces the
bounded set, never by truncating after the fact):

| Cap | Value |
|---|---|
| eligible links per page | 64 |
| roots per candidate | 64 |
| pending candidates per space | 128 |
| candidate prepares per space per cycle | 16 |
| automatic LLM finalizations per refinery turn | 1 (vacuous in PR-B) |

**Excess work remains frontier-visible.** This is the predicate that makes the
caps safe: hitting a cap must leave the un-processed remainder on the frontier
with a future `next_scan_at`, never mark it done, suppressed, or terminal.

---

## 4. Shadow-loop design

### 4.1 Where it hooks in

`crates/wenlan-server/src/main/runtime.rs`, immediately after the M5
truth-maintenance loop (currently lines 183–314). A **second, sibling**
`tokio::spawn`, not a new stage inside the M5 loop — the two must be able to
fail, back off, and be disabled independently.

Shape copied deliberately from the M5 loop, because that shape is proven:

1. Guarded by `optional_runtime_workers_allowed(repair_recovery_pending)` — the
   same predicate the M5 loop uses, so repair/recovery still suppresses both.
2. `tokio::spawn` after a `lifecycle::sleep_or_shutdown` startup delay. PR-B
   uses **3s** rather than M5's 1s so the two loops do not contend for the
   single DB connection mutex at boot.
3. Per turn: re-snapshot state, **drop the read guard before any await**.
   (Root `AGENTS.md`: never hold a `RwLock` guard across `.await`.)
4. One `run_genesis_shadow_turn(...)` per tick, inside a **shutdown-biased**
   `tokio::select! { biased; _ = wait_for_shutdown(..) => return; result = .. }`.
5. Adaptive delay: 100ms when the turn did work, 1s when idle, 5s after an
   error, with de-duplicated error logging (log the first occurrence and each
   change, not every tick).
6. `lifecycle::sleep_or_shutdown` at the loop bottom.

The turn function returns a `GenesisTurn` enum mirroring M5's `PromotionTurn`:

```rust
pub enum GenesisTurn {
    Idle,
    Frontier { space: String, admitted: usize, deferred: usize },
    Prepared { candidate_id: String },
    DryRunPassed { candidate_id: String },
    DryRunRefused { candidate_id: String, gate: CasGate },
    Requeued { candidate_id: String, reason: String },
    Parked { candidate_id: String, reason: String },
    RefusedBudget,
}
```

`RefusedBudget` is distinct from `Requeued` for a contract reason: **S0-9 — a
budget refusal does not increment `attempt`.** Folding it into `Requeued` would
burn retries on backpressure and eventually park healthy work.

### 4.2 Cadence and bounds per turn

One turn does **at most one** of the following, in this priority order, and
returns:

| Priority | Step | Bound |
|---|---|---|
| 1 | Startup recovery scan (once per process, eagerly, before the first normal turn — S0-5) | whole scan, one transaction per candidate |
| 2 | Frontier reconciliation for one space | 1 space, cursor-resumed, ≤ 512 rows |
| 3 | Candidate prepare for one space | ≤ 16 prepares (D8 cap), ≤ 64 roots each |
| 4 | Dry-run finalization for one candidate | exactly 1 |

"One turn, one unit of work" is the M5 loop's discipline and it is what makes
shutdown responsive and the adaptive delay meaningful.

### 4.3 Leases held per step

| Step | Phase | TTL | Scope |
|---|---|---|---|
| frontier reconciliation | `frontier` | 120s | `(phase, space, input_generation)` |
| candidate prepare + dry-run finalize | `genesis` | 900s | `(phase, space, input_generation)` |

No renewal (S0-4). Backoff on retry is `60s * 4^(attempt-1)` capped at 4h, with
**no jitter** (S0-2); `attempt > 5` is exhaustion, which lands the candidate in
`stale` carrying a reason — **not** an eleventh state (S0-12).

### 4.4 The startup recovery scan

Artifact 2 §11, three steps, run eagerly at process start:

1. Delete expired leases across **all** phases.
2. Release orphaned reservations, return their groups to the frontier, and move
   the candidate to `stale` with `reason = 'lease_lost'` — **one transaction per
   candidate**, so a crash mid-scan leaves a consistent prefix rather than a
   half-released candidate.
3. Requeue `handed_off` rows in `page_projection_outbox`. In PR-B this must find
   **zero rows** (§1.2), and the gate in §8 asserts that the zero is real rather
   than a query that cannot see the table.

### 4.5 Restart-resumability

Every step's progress is durable before the turn returns: the frontier cursor is
an `app_metadata` key, candidate state is a row, reservations are rows with a
stored liveness marker, and leases expire on their own. There is no in-memory
work queue. A `kill -9` between any two turns costs at most one turn's work and
one lease TTL of latency.

---

## 5. Dry-run finalization semantics

### 5.1 What "dry run" means, exactly

Finalization opens **one outer IMMEDIATE transaction** (machine E), evaluates
the eight CAS gates **in order**, and then — where the real path would call the
model and write a page — records a verdict row and **rolls back or commits only
shadow state**. Specifically:

- **No LLM provider is consulted.** The provider argument is not threaded into
  `finalize.rs` at all. This is stronger than a runtime `if` and is checkable by
  a signature: `finalize.rs` must not import or name `LlmProvider`.
- **No `pages` row is created, updated, or touched.** No `page_projection_outbox`
  row is written in any state.
- **All eight CAS inputs are still verified**, in order, against live state — a
  dry run that skipped gates would prove nothing about the real path.
- The verdict (pass, or the first gate that failed) is recorded as shadow
  evidence per §7.

### 5.2 The eight CAS gates, in order

| Gate | Verifies | On miss |
|---|---|---|
| E-1 | the lease token still belongs to this worker | E3 rollback |
| E-2 | `input_generation` unchanged | E3 rollback |
| E-3 | active-root-set digest unchanged (`active_root_digest`, `identity.rs`) | E3 rollback |
| E-4 | `coverage_epoch` **and** `epoch_state` unchanged | E3 rollback |
| E-5 | claims still unreleased; witness rows still present | E3 rollback |
| E-6 | evidence liveness — every root still active, every edge still grounded and live | E3 rollback |
| E-7 | M4 community generation still current and durable | E3 rollback |
| E-8 | M5 page/dependency truth preconditions hold | E3 rollback |

Any miss ⇒ **E3 rollback: publish nothing, requeue via A11.** Not park, not
terminal — a CAS miss means the world moved, which is normal.

Two properties the gates must have, and which §8 mutates:

- **Order is load-bearing.** E-1 before E-2 before E-3 …; a reordering that
  checks evidence liveness before the lease can act on state it does not own.
- **Recomputation is not a substitute for a stored CAS input.** The M5 code
  already learned this: `finalize_page_support`
  (`crates/wenlan-core/src/db/claim_derivation.rs:4545`) checks
  `current_eligibility_generation(&tx) != job.eligibility_generation` *and*
  separately recomputes the verdict, with the comment that recomputing is not a
  substitute because `active → draining` can leave the verdict unchanged while
  invalidating the regime the worker leased under. M6's E-2/E-4 have the same
  shape and the same reason.

### 5.3 Invariants asserted at finalization

- **I-1** exactly one live durable reason per group.
- **I-4** publish is one atomic fact (vacuous in PR-B; asserted so it cannot
  regress).
- **I-5** no guard spans a model call (structurally true — no model call).
- **I-8** a retry reuses the same candidate, slot, page id, lease, and receipt
  rather than minting new identity. This is why `candidate_id` is the derived
  digest and the natural primary key.

---

## 6. Incremental-vs-full recomputation oracle

### 6.1 The oracle

`crates/wenlan-core/src/m6/oracle.rs` exposes:

```rust
pub struct GenesisSnapshot { /* candidates, claims, coverage, frontier — normalized */ }

pub async fn snapshot_incremental(tx: &libsql::Transaction, space: &str) -> Result<GenesisSnapshot, WenlanError>;
pub async fn recompute_full(tx: &libsql::Transaction, space: &str) -> Result<GenesisSnapshot, WenlanError>;
pub fn diff(a: &GenesisSnapshot, b: &GenesisSnapshot) -> Vec<Divergence>;
```

`recompute_full` derives the entire genesis state for a space **from primary
evidence only** — edges, roots, pages, communities — ignoring every
`genesis_*` row. `snapshot_incremental` reads what the shadow loop actually
built. `diff` returns an empty vector iff they agree.

Normalization matters: the snapshot must exclude fields that legitimately differ
between an incremental build and a from-scratch build — `created_at`,
`updated_at`, `first_seen_at`, `attempt`, `next_attempt_at`, `lease_token`. It
must **include** everything identity-bearing: `candidate_id`, `slot_id`,
`page_id`, `signal_kind`, `coverage_epoch`, `active_root_digest`, `state`, the
full claim set with roles, the coverage set, and the frontier key set. Excluding
a field that should be compared is the way this oracle silently stops working,
so the field list is itself asserted by a test (a struct-field census, in the
spirit of the R4 test-support census that the PR-A receipt records at
`docs/plans/2026-08-02-m6-pr-a-followup-schema.md:114`).

### 6.2 Where the oracle runs

1. **In hermetic tests** — after every scripted mutation sequence (§9), assert
   `diff().is_empty()`.
2. **In the shadow loop, sampled** — one space per N turns (N configurable,
   default large enough to be negligible), read-only, recording a divergence
   count as evidence per §7. A divergence is logged and recorded; it never
   auto-repairs, because auto-repair would hide the bug the oracle exists to
   find.

### 6.3 Mutation oracles

Beyond full-vs-incremental, the harness scripts each mutation class and asserts
the resulting delta:

| Mutation | Expected genesis delta |
|---|---|
| new grounded edge on a new root in a new group | group leaves frontier or the count rises by exactly 1 |
| new grounded edge on a root in an **existing** group | count unchanged (R1) |
| root retracted (`status <> 'active'`) | count drops by 1 iff it was that group's last active root |
| edge retracted (`valid_until` set) | same, via R3 |
| generated root added | count unchanged (R4) |
| page renamed to/from `Overview` | signal eligibility flips via `lower(title)`, per R5 |
| community becomes dirty | signal 1 and signal 3 stop admitting (durable gate) |
| M5 support drops from `supported` | orphan-wikilink underlying groups shrink (R8) |
| space renamed | every `space`-keyed M6 row follows; no proof value changes |

---

## 7. Readiness and benchmark evidence

### 7.1 Readiness

PR-B's readiness target is stage **`B_genesis_shadow`**, whose entry condition
in artifact 11 §2 is literally "PR-B deployed, jobs dry-run".

Wiring (all **[wire]**, `crates/wenlan-core/src/m6/refresh_readiness.rs`):

- `initialize_readiness` for each `(stage, signal)` key on first observation.
- `readiness_fence` read before any state transition.
- `transition_is_legal` — only `off → preparing`, `preparing → off`,
  `preparing → committed`. PR-B drives at most `off → preparing`; it never
  reaches `committed`, because committed is a cutover fact.
- `transition_readiness` — epoch always increments.

`record_soak_receipt` requires ≥ 259200s (72h) of window, ≥ 20 turns, ≥ 1 daemon
start, and zero mutations / violations / regressions, enforced by the
`m6_soak_receipt_fence_guard` trigger. PR-B **records the evidence that feeds
it** and may write a receipt once a real 72h shadow soak has run; the receipt is
not a merge gate for PR-B itself (see §11 Q9).

### 7.2 Per-space shadow statistics

`crates/wenlan-core/src/m6/evidence.rs` records, per space per turn:

- frontier size, admitted / deferred / suppressed / quarantined counts
- candidates by state
- dry-run outcomes by first-failing CAS gate
- oracle divergence count (should be 0)
- the observed value of `genesis_coverage_state.m6_mutation_count` (should be
  unchanged, forever — this is the zero-mutation proof)

### 7.3 Benchmarks

Artifact 8 §9 defines the Stage-0 representative corpus (S0-97: 100k memories,
5k pages, 8 spaces at 40/20/12/10/8/5/3/2%, 12k independence groups, Zipf 1.1
truncated at 5000, hubs at 5000/1024/65, mean fanout 8, ages 0–720d, 15%
generated roots, 5% retracted edges, seed `0x6D36_0000`, digest-identified
manifest). PR-B builds against **that** corpus generator; it does not invent a
second one.

Artifact 8 §10 gives pass/fail limits for **relevance only** (`R-BENCH-MAX` 50ms
hard, `R-BENCH-Q` ≤ 4 queries, `R-BENCH-ROWS` ≤ 512 rows, `R-BENCH-HUB`). No
equivalent limits exist for the frontier or genesis paths anywhere in the twelve
artifacts, while the goal prompt's verification floor demands
"representative-scale relevance/frontier/genesis/lock benchmarks". PR-B
therefore **measures and records** frontier and genesis timings, query counts,
and row counts against the corpus, and states them as observed values — it does
not invent thresholds and call them contract. See §11 Q8.

---

## 8. RED-first executable gates

Every gate below ships with **(a)** a RED mutation that makes it fail and **(b)**
a discriminating positive control that stays green under that same mutation
(S0-135/S0-155). One condition per row (S0-154). A clause with no mapped row is
by definition ungated (S0-134). Empty-inventory vacuous truth is mandatory
(S0-137) — every gate must also state what it asserts when its input set is
empty, because a gate that passes vacuously on an empty install is not a gate.

Row IDs below are the artifact-12 catalog IDs, so the catalog stays the index.

### G2 — the independence floor counts groups, not rows

| Row | RED mutation | Positive control |
|---|---|---|
| G2.1 | `COUNT(DISTINCT independence_group_id)` → `COUNT(*)` | ten chunks of one document admit exactly zero candidates |
| G2.2 | drop `r.status = 'active'` | a retracted root's group stops counting |
| G2.3 | drop `e.grounded = 1` | an ungrounded edge contributes nothing |
| G2.4 | drop `e.valid_until IS NULL` | a retracted edge contributes nothing |
| G2.5 | drop `r.root_kind <> 'generated'` | a generated root contributes nothing (no self-bootstrapping) |
| G2.6 | floor 3 → 2 | a 2-group cluster is refused |
| G2.7 | `lower(title) <> 'overview'` → always true | an Overview page contributes nothing |
| G2.8 | collapse human groups → per-capture groups | three human deltas do **not** clear the floor (B28) |

Vacuous case: an install with zero grounded edges admits zero candidates and
writes zero rows — asserted, not assumed.

### G3 — overlapping candidates publish once

| Row | RED mutation | Positive control |
|---|---|---|
| G3.1 | drop `idx_genesis_root_claim` | two candidates cannot both claim one root |
| G3.2 | drop `idx_genesis_group_claim` | two candidates cannot both claim one group |
| G3.3 | `claim_role` CHECK removed, third role inserted | a witness row is never readable as coverage (I-3) |
| G3.4 | pre-check-then-insert instead of insert-and-refuse | concurrent reservation still yields exactly one winner |
| G3.5 | `released_at` written by a non-recovery path | reservation liveness stays reconciler-owned (S0-6) |
| G3.6 | `candidate_id` made a surrogate key | a retry reuses the same candidate row (I-8) |
| G3.7 | coverage row deleted on epoch change | coverage is permanent within an epoch (I-7) |

### G4 — finalization is all-or-nothing, and the dry run really is dry

| Row | RED mutation | Positive control |
|---|---|---|
| G4.1a–k | disable each CAS gate E-1…E-8 in turn (plus the three ordering permutations) | each disabled gate makes exactly one scripted concurrent-mutation scenario publish-eligible that must not be |
| G4.2a–g | replace a stored CAS input with a recomputation | the `active → draining` analogue still refuses |
| G4.3 | remove the E3 rollback | a failed gate leaves no partial write |
| G4.4 | let `finalize.rs` reach a page write | zero `pages` mutations across a full shadow run |
| G4.5a–d | let `finalize.rs` reach an LLM call | `finalize.rs` names no provider type; zero inference calls recorded |

Vacuous case: a candidate set of size zero produces zero verdicts and zero
`page_projection_outbox` rows.

### G5 — the frontier loses nothing

| Row | RED mutation | Positive control |
|---|---|---|
| G5.1 | make the differential query a delta feed | a group with zero durable reasons is repaired by a frontier insert |
| G5.2 | allow `reason_count > 1` to pass | two reasons refuse and surface (S0-43) |
| G5.3 | `INSERT OR IGNORE` → `INSERT OR REPLACE` | `first_seen_at` survives re-entry (S0-50) |
| G5.4 | let a cap mark work terminal | hitting the 16-prepare cap leaves the remainder frontier-visible (S0-54) |
| G5.5 | drop the cursor / reset it on restart | restart resumes at the same position, loses nothing (P1–P16) |
| G5.6 | `next_scan_at` allowed > 24h ahead | S0-46 holds |
| G5.7 | compaction deletes the row instead of nulling `payload` | terminal rows survive compaction (S0-53) |

### Exit-matrix gate (S0-139)

The 8 finalization exits × 6 resting states are asserted as a **total function**:
every exit maps to exactly one resting state, and the test enumerates all 48
cells. A missing cell fails; an extra cell fails.

### Catalog-completeness gate (S0-144)

A test asserts that every artifact-12 row whose lane is PR-B has a matching
executable gate in this PR, by ID. The catalog's completeness is itself a test —
so a row added to the catalog later without a gate fails the build rather than
sitting silently unimplemented.

---

## 9. Test plan — hermetic, no GPU

Everything below runs on a hosted runner with no Metal, no model files, and no
API key. There is no L7 leg in PR-B, because there is nothing to run a model
against.

### 9.1 Unit tests (`#[cfg(test)]`, in-module)

| Module | Coverage |
|---|---|
| `independence.rs` | the count expression against a fixture graph; every R1–R8 conjunct |
| `signals.rs` | each of the four signals at threshold − 1, threshold, threshold + 1; tie-break determinism; embedding unavailability changes nothing |
| `candidates.rs` | machine A's 19 transitions, machine B's 8, machine D's 6, each as a legal/illegal pair |
| `frontier.rs` | the differential query's `reason_count` arms (0 / 1 / >1); cursor encode/decode round-trip; 24h clamp |
| `leases.rs` | acquire/CAS/release/reap against the real `grouping_leases`; the manual-joins-scheduler case |
| `finalize.rs` | each CAS gate in isolation; gate ordering; E3 rollback leaves no rows |
| `oracle.rs` | the field census; a deliberately divergent snapshot is detected |

### 9.2 Integration test — `crates/wenlan-core/tests/m6_genesis_shadow.rs`

One in-memory libSQL database, migrations applied, a scripted world:

1. Seed a small graph (documents, roots, groups, edges, pages, one community).
2. Run N shadow turns to quiescence.
3. Assert the oracle diff is empty.
4. Apply each mutation from §6.3, re-run to quiescence, assert the expected
   delta and an empty oracle diff.
5. Kill and restart mid-flight (simulate by abandoning a lease), assert the
   recovery scan restores consistency and loses nothing.
6. Assert the zero-mutation proof: `m6_mutation_count` unchanged, zero rows in
   `pages` / `entities` / `relations` / `communities` changed, zero
   `page_projection_outbox` rows.

### 9.3 Benchmark test (`#[ignore]`, manual)

Generate the S0-97 corpus from its seed, run frontier reconciliation and
candidate prepare, record wall time, query count, and rows scanned. Marked
`#[ignore]` because 100k memories is not a 20-minute CI budget; run locally and
paste the receipt into the PR.

### 9.4 Test-support census

The PR-A receipt (`docs/plans/2026-08-02-m6-pr-a-followup-schema.md:114`)
records that a completed workspace run failed on "2 R4 test-support guards …
because the six new test-only libSQL helpers were absent from the frozen
manifest/census". PR-B will add test-only helpers and **must** register them in
that same census in the same commit, or the full workspace suite fails after the
focused suite passes — an expensive way to learn it.

---

## 10. Behavior-unchanged proof

Reuse the M5 promoter's proof method, which is the reason PR-B is allowed to
ship at all.

### 10.1 The hash proof

For a fixed corpus and a fixed sequence of API calls, capture a stable digest of
every user-visible surface before and after the PR-B build, and assert equality:

- `/api/search`, `/api/context`, `/api/memory/search` result sets and order
- `/api/memory/list`, `/api/pages` (list, get, sources, links, revisions)
- `/api/memory/entities`, `/api/memory/relations`, `/api/memory/observations`
- `/api/status` (minus fields that legitimately move: uptime, counters)
- the projected page vault on disk — file list plus per-file content hash

The vault leg needs the three-axis isolation the root `AGENTS.md` calls out:
`WENLAN_DATA_DIR` does **not** cover the page vault, so the isolated daemon's
data dir must be seeded with a `config.json` carrying
`{"knowledge_path": "<scratch>/pages"}` and the path confirmed via
`GET /api/knowledge/path` before the run. Otherwise a "behavior-unchanged" run
writes into the real vault while proving it did not.

### 10.2 The structural proofs

Three claims that a hash diff alone cannot make:

1. **`finalize.rs` names no LLM provider type** — a source-level assertion, not
   a runtime one.
2. **`m6_mutation_count` is unchanged** on every space after a full shadow soak.
   The counter is monotone and trigger-guarded (`remaining_substrate.rs`), so an
   unchanged value is a durable proof rather than an observation.
3. **Zero rows in `page_projection_outbox`** in any state, asserted against the
   real table so the assertion cannot pass because the table is invisible.

### 10.3 Rollback

Per the goal prompt's rollback rule for the PR-B/PR-C shadows: **stop jobs,
invalidate leases, retain frontier / coverage / stats for diagnosis, and leave
readers and writers unchanged.** Rollback is therefore a flag flip plus a lease
sweep — no data deletion, no migration reversal, no reader change. The retained
state is the diagnostic value; deleting it on rollback would throw away the
evidence that explains why the rollback happened.

---

## 11. Open questions and contract ambiguities

Flagged, not resolved. Each names the artifacts that disagree.

**Q1 — The `supported` writer is now live, invalidating a stated lane
assumption.** Artifact 1 §7.1 and artifact 12's F2 both assert that no
production writer promotes `support_status` to `'supported'`, place 16 catalog
rows in "lane 1", and state that "PR-B's orphan-wikilink shadow is unmeasurable
until that writer lands". HEAD `c620d7e2` ("feat: add shadow claim support
promoter") **is** that writer: `finalize_page_support`
(`crates/wenlan-core/src/db/claim_derivation.rs:4482`) writes
`support_status = 'supported'` at line 4595 via the `?3` parameter bound at
4559. Consequence: G1.2, G2.5, G6.9, G7.1–G7.3 and P1–P5/P6b are executable
**now**, and the orphan-wikilink signal is measurable in PR-B. The catalog's
lane counts (LIVE 32 / PR-A 156 / lane 1 16) are stale by 16 rows.

**Q2 — `pages.kind` is still fenced, so D1 R5 and D2.2 must key on the title.**
The write-path half of the kind fix shipped (`page_kind_for`,
`crates/wenlan-core/src/db.rs:41904`; migration 107 repair at
`crates/wenlan-core/src/db.rs:12131`), but `drift_guard` teeth #16
(`crates/wenlan-core/src/drift_guard.rs:10657`) still forbids production read
routing on any non-`entity` kind, and its own doc comment names the surviving
gaps: rename, archive, and replace never re-derive `kind`, and migration 89
folded only `creation_kind = 'imported'`. S0-164 already says to use
`lower(title) <> 'overview'`. This spec follows S0-164; the question is when the
fence comes down, and whether that is inside M6 at all.

**Q3 — `label_key` re-keying is undecided.** Artifact 4's finding F5 gives PR-A
three options — (a) migrate and merge colliding rows, (b) compute the M6 key at
read time and forfeit `idx_page_links_orphan`
(`crates/wenlan-core/src/db.rs:6701`), (c) add a second column — and PR-A chose
none. Production still stores `let label_key = link.label.to_lowercase();`
(`crates/wenlan-core/src/db.rs:44425`), which is not the D8 pipeline. PR-B's
orphan-wikilink signal needs a decision: option (b) is the only one that is
inert, and it costs the index on a query PR-B runs per space per cycle.

**Q4 — D1 R4's refused-mint review artifact has no owner in PR-B.** S0-23
specifies it; the PR-A receipt explicitly says "The refused-mint review artifact
remains owned by the later edge-grounding writer lane"
(`docs/plans/2026-08-02-m6-pr-a-followup-schema.md:86`). So R4's "unknown
independence routes to human review" has a state (`review_required`) but no
review surface. PR-B can record the state; something else must surface it.

**Q5 — `grounded = 1 AND root_id IS NULL` has no named substrate.** S0-20
requires this case be surfaced. No table in migration 108 or 109 holds it, and
no artifact names one. PR-B's independence code will encounter it (the count
expression's `JOIN` silently drops such edges) and currently has nowhere to
record it.

**Q6 — Space identity is asymmetric across the M6 tables.** `genesis_frontier`,
`genesis_suppression`, `genesis_card_binding`, and `genesis_quarantine` key on
the **renameable** `space` name. Migration 109 bound `genesis_coverage_state`
and `m6_counters` to the immutable `spaces.id`
(`docs/plans/2026-08-02-m6-pr-a-followup-schema.md:35`, rounds 3–5). The gap is
closed today only by the rename cascade (`SPACE_RENAME_TABLES` in
`crates/wenlan-core/src/m6/frontier_policy.rs`). A rename that fails partway
leaves the two halves disagreeing, and nothing detects it.

**Q7 — The frontier cursor's ordering does not match the shipped index.** S0-45
specifies a global cursor ordered `next_scan_at, first_seen_at, space, gid`, but
`idx_genesis_frontier_scan`
(`crates/wenlan-core/src/db/genesis_schema.rs:170`) is
`(space, next_scan_at, first_seen_at, independence_group_id)` — space **first**.
A global cursor scan against a space-first index either sorts or scans every
space. Either the cursor is per-space (contradicting S0-45's single
`app_metadata` key holding a space component) or the index is wrong.

**Q8 — "Active reservation" has two definitions.** Artifact 5 speaks of
`live_reservations` in terms of the candidate's non-terminal state; S0-6 and
migration 108's partial indexes define it as the stored `released_at IS NULL`
bit on the claim row. These diverge exactly when a candidate goes terminal
without the recovery scan having released its claims — which is the normal
window, not an edge case. PR-B will implement the stored-bit definition
(because it is the one the indexes enforce) and the divergence window is
observable.

**Q9 — No frontier or genesis benchmark thresholds exist.** Artifact 8 §10
defines `R-BENCH-MAX` / `R-BENCH-Q` / `R-BENCH-ROWS` / `R-BENCH-HUB` for
relevance only. The goal prompt's verification floor demands
"representative-scale relevance/frontier/genesis/lock benchmarks". PR-B will
report observed numbers with no pass/fail line; someone has to set one.

**Q10 — S0-3's lease TTLs were never re-derived.** S0-3 states the TTLs
(frontier 120s, relevance 300s, genesis 900s, refresh 900s) "are to be
re-derived from the configured LLM timeout at PR-A". PR-A did not do it. The
900s genesis TTL is currently a number with no stated relationship to anything
the daemon is configured to wait for.

**Q11 — Does opening a coverage epoch enable genesis?** Migration 108's
`genesis_coverage_state` has `genesis_enabled INTEGER NOT NULL DEFAULT 0`, and
its comment says "a space with NO ROW is genesis-disabled". Machine D's D1
transition opens an epoch. PR-B must open epochs to have a `coverage_epoch` to
key candidates on — so it will create rows with `genesis_enabled = 0`. The
artifacts do not state whether epoch-open is legal while genesis is disabled, or
whether `genesis_enabled = 1` is a precondition of D1. This spec assumes the
former (row exists, flag stays 0); if the latter is intended, PR-B cannot
populate candidates at all and the shadow is empty.

**Q12 — Is a soak receipt a PR-B merge gate?** `record_soak_receipt`
(`crates/wenlan-core/src/m6/refresh_readiness.rs`) enforces a 72h / 20-turn /
1-daemon-start / zero-mutation contract via the `m6_soak_receipt_fence_guard`
trigger. Artifact 11's stage machine gives `B_genesis_shadow` the entry
condition "PR-B deployed, jobs dry-run" — which is satisfied at merge, before
any soak. Whether the receipt gates the *merge* or the *stage advance* is
unstated. This spec assumes stage advance.

### 11.1 Adjudication (2026-08-03)

Resolved in this draft so implementation is not blocked. Each is a spec-level
call, not a merge: the independent review at the integrated boundary still owns
the final word, and any of these can be reversed there.

**Q7 → per-space cursor; leave the shipped index alone.** S0-45's global
ordering and `idx_genesis_frontier_scan`'s space-first ordering cannot both
stand. The index is shipped and the corpus has 25 spaces, so a per-space cursor
is a 25-key walk that the index serves directly, with round-robin across spaces
giving the same starvation-freedom the global order was reaching for. Amend
S0-45 rather than the index.

**Q3 → option (b), read-time key computation.** Production stores
`label.to_lowercase()` (`crates/wenlan-core/src/db.rs:44425`); the D8 pipeline
wants more. Options (a) and (c) both write to production rows, which the
shadow-mode floor forbids outright, so (b) is the only inert choice and the
decision reduces to whether forfeiting `idx_page_links_orphan` is affordable.
Measured on the production corpus: `page_links` holds 209 rows, 88 of them
orphans. A full scan at that scale is free. Revisit only if `page_links` grows
past ~100k rows.

**Q11 → opening a coverage epoch is legal while `genesis_enabled = 0`.** Both
the flag default (`genesis_schema.rs:142`, `DEFAULT 0`) and the absent-row rule
independently mean "disabled", so a row with the flag at 0 is disabled by the
same contract that an absent row is. PR-B therefore creates rows and never
writes the flag. Gate G4 gains a positive control: with epochs opened for every
space and `genesis_enabled = 0` throughout, no page, support, or card row is
written.

**Slicing → three stacked PRs** (§12.2). ~2,900 production lines does not fit
one honest review, and the split keeps the zero-mutation proof in the same PR as
the loop that could break it.

---

## 12. Size estimate and slicing

### 12.1 Estimate

| Area | New | Wired | Test |
|---|---|---|---|
| `independence.rs`, `signals.rs` | ~450 | — | ~500 |
| `candidates.rs` (machines A/B/D) | ~600 | — | ~700 |
| `frontier.rs` | ~400 | `frontier_policy.rs` | ~450 |
| `leases.rs` | ~150 | `db.rs` registry | ~250 |
| `finalize.rs` (8 CAS gates) | ~450 | — | ~800 |
| `oracle.rs` | ~350 | — | ~200 |
| `evidence.rs` | ~200 | `refresh_readiness.rs` | ~150 |
| `recovery.rs` | ~200 | — | ~250 |
| `runtime.rs` shadow loop | ~120 | — | — |
| integration test | — | — | ~900 |

Roughly **2900 lines of production code and 4200 of test**. That is well past a
reviewable single PR, and past the point where a review can hold the whole thing
in mind at once.

### 12.2 Recommended slicing — three PRs

**PR-B1: read side (no writes at all).**
`independence.rs`, `signals.rs`, `oracle.rs::recompute_full`, plus the G2 gate
family and the D8 wiring. Ships a pure function from a database to a candidate
proposal list, with the full independence-floor gate suite. Reviewable in
isolation because it cannot break anything — it writes nothing. ~950 production
lines.

**PR-B2: durable state and the frontier.**
`leases.rs`, `candidates.rs`, `frontier.rs`, `recovery.rs`,
`oracle.rs::snapshot_incremental` + `diff`, the G3 and G5 gate families, the
exit matrix, and the mutation oracles. This is where the state machines and I-1
through I-9 live. ~1350 production lines.

**PR-B3: dry-run finalization, the loop, and evidence.**
`finalize.rs`, `evidence.rs`, the `runtime.rs` shadow loop, the G4 gate family,
the behavior-unchanged proof, and the benchmark receipt. ~800 production lines.

The split point is chosen so the **zero-mutation proof lands in B3 with the loop
that could violate it** — B1 and B2 have no daemon-side driver, so nothing runs
in production until B3 merges. That makes B1 and B2 genuinely inert, and gives
B3 a small enough diff for the review that actually matters: the one that checks
the dry run is dry.

### 12.3 If it must be one PR

Ship it as one PR but review it in the three slices above, in order, with the
gate suite for each slice green before the next is read. The catalog-completeness
gate (S0-144) is what makes that reviewable — it proves no PR-B-lane row was
skipped, regardless of how the diff was chunked.
