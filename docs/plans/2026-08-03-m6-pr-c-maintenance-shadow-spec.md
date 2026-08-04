# M6 PR-C — maintenance shadow: implementation spec

Status: implementation spec. Successor to
`docs/plans/2026-08-03-m6-pr-b-genesis-shadow-spec.md`, which this file assumes
has been read — its §1 legend, §11 open-question ledger, and §12.2 slicing
rationale are reused rather than restated.

**Grounding.** Every `file:line` below was read on this branch
(`m6-truth-followup`, worktree `.claude/worktrees/m5-truth-promoter`) while
writing. Claims that were *not* verified against source carry `[unverified]`
inline. Stage-0 `S0-NN` decisions are normative; where shipped code contradicts
an artifact, **the code wins and §13 records the amendment**.

**One-sentence contract.** PR-C runs a second bounded background lane that
maintains M6-owned relevance statistics, evaluates refresh jobs as dry runs,
reconciles overview-subscription lifecycle, and records the stage-C soak
evidence PR-D's cutover consumes — mutating no user-visible row, consulting no
language model, and publishing nothing.

---

## 1. Scope

### 1.1 What PR-B landed that PR-C builds on

PR-B3 is an **interface**, not internals. Four seams are stable and PR-C binds
to them; nothing else about PR-B3's modules is assumed.

| Seam | Where | What PR-C uses it for |
|---|---|---|
| The sibling-lane shape | `crates/wenlan-server/src/main/runtime.rs:338`–`:417` | The literal template for PR-C's lane: 3s startup delay, shutdown-biased `tokio::select!` (`:355`), de-duplicated error log (`:396`–`:408`), 5s error backoff (`:409`), `lifecycle::sleep_or_shutdown` (`:413`) |
| The maintenance guard | `runtime.rs:359` — `maintenance_for_genesis.begin_background()` | PR-C's lane takes the same guard so an approved repair suspends it |
| Monotone counters | `crates/wenlan-core/src/m6/evidence.rs:83` (`bump`), `:194` (`ShadowStats`), `:219` (`observe_space`) | PR-C's turn counters, soak components, and the runtime zero-mutation assertion |
| Readiness transitions | `crates/wenlan-core/src/m6/refresh_readiness.rs:370` (`transition_is_legal`, "Public since PR-B3"), `:381` (`transition_readiness`), `:329` (`initialize_readiness`), `:347` (`readiness_fence`) | Driving stage `C` |

PR-B3 also settled facts PR-C must not re-derive: the frontier reconciler
already reads `genesis_suppression` and `genesis_quarantine` as live reasons
(`crates/wenlan-core/src/m6/frontier.rs:115`, `:119`), F11 suppression lapse is
wired (`frontier.rs:176`–`:177`), and D7 payload compaction is shipped and
tested (`crates/wenlan-core/src/m6/candidates.rs:935`, catalog row `G5.6` bound
to `compaction_nulls_the_payload_and_keeps_the_row` at
`crates/wenlan-core/src/m6/catalog_test.rs:91`).

### 1.2 The four contract items, and what is actually left

The task names four items. Three of the four have **zero production callers
today**; the fourth is two-thirds already wired. Determined by caller grep, not
by the `#[allow(dead_code)]` attributes at `crates/wenlan-core/src/m6/mod.rs:33`,
`:36`, and `:38` — those are a signal, and the grep is the proof.

| # | Contract item | Substrate | Wired by PR-B? | PR-C's remainder |
|---|---|---|---|---|
| 1 | Bounded relevance scoring | `m6_pair_stats` (`relevance.rs:31`), `m6_adjacency` (`:47`) | **No** — only the migration-109 DDL call at `db.rs:12275` | Everything: pair-stat maintenance, adjacency materialization, the oracle, the budget instruments |
| 2 | Guarded refresh jobs | `genesis_refresh_jobs` (`refresh_readiness.rs:15`), `m6_refresh_dependencies` (`:47`) | **No** for the job/dependency half; **yes** for the readiness half (PR-B3's `evidence.rs:150` wires `initialize_readiness`/`readiness_fence`/`transition_is_legal`/`transition_readiness`) | `acquire_refresh_lease` (`:149`), `replace_refresh_dependencies` (`:229`), `space_refresh_dependencies_ready` (`:261`), `record_soak_receipt` (`:438`) |
| 3 | Overview subscription lifecycle (D5 overview identity) | `m6_overview_subscriptions` | **DDL only.** `crates/wenlan-core/src/m6/overview_subscriptions.rs` is 42 lines and contains exactly one function, `ensure_overview_subscription_tables` (`:8`). The table is *read* at `crates/wenlan-core/src/m6/signals.rs:632` and written by nothing | The detach half and the `install`-scope registration — see §5.3 for why the create half cannot land here |
| 4 | `frontier_policy.rs` writers | `crates/wenlan-core/src/m6/frontier_policy.rs` | **Three of six wired** | Three unwired, of which **PR-C wires zero** — see §5.4 |

#### 1.2.1 The frontier_policy inventory, resolved by caller grep

| Writer | Machine-F transition | Definition | Caller today |
|---|---|---|---|
| `bind_frontier_groups_to_card` | F3 / F6 | `frontier_policy.rs:189` | `frontier.rs:712`, `candidates.rs:873` — **wired** |
| `reconcile_expired_suppressions` | F11 | `frontier_policy.rs:235` | `frontier.rs:177` — **wired** |
| `dismiss_card_to_suppression` | F7 (from card) | `frontier_policy.rs:408` | `candidates.rs:911` — **wired** |
| `suppress_frontier_group` | F7 (from `exclusively_claimed`) | `frontier_policy.rs:151` | none |
| `quarantine_frontier_group` | F9 | `frontier_policy.rs:302` | none |
| `lift_quarantine_to_frontier` | F12 | `frontier_policy.rs:355` | none |

The three unwired writers have **no automatic driver in any Stage-0 artifact**
(§5.4 and Q3). PR-C declares them deferred with an owner rather than inventing
a policy for them; that is a scope decision, not an omission, and §8.7 makes it
countable.

### 1.3 Non-goals

Each row names the rung that owns it and why PR-C cannot.

| Deferred | To | Why not PR-C |
|---|---|---|
| Writing a relevance **attachment** to a page | PR-E | An attachment is a `page_links`/`pages` write. Zero-mutation forbids it |
| Staging the merge-loser **proposal** into `refinement_queue` with `status='awaiting_review'` (S0-77, `docs/plans/2026-08-01-m6-overview-matrix.md:230`) | PR-E | The row surfaces in `/curate`. `refinement_queue` is not on the task's enumerated table list, but a row a human is asked to review is user-visible state by any honest reading — see Q6 |
| Extending `try_update_page_content` for refresh finalization (S0-69, `docs/plans/2026-08-01-m6-refresh-matrix.md:399`) | PR-E | Publishes prose |
| Staging a refresh **revision card** into `memories` | PR-E | Enumerated table |
| Creating an overview subscription for a *new* community/space overview page | PR-E | Requires a `page_id` only PR-E mints (§5.3) |
| Flipping `genesis_coverage_state.genesis_enabled` | PR-E1–E4 | Publication flag; PR-C never writes it (§7.3) |
| Advancing readiness past stage `C` | PR-D | S0-125 (`docs/plans/2026-08-01-m6-readiness-cutover.md:297`) puts every PR-D precondition inside the epoch-advancing transaction; PR-C supplies the *evidence*, PR-D spends it |
| The D12 old-writer fence | PR-D (G9) | PR-C's stage-C soak component 1 is therefore honest only for stage C — see Q8 |
| `suppress_frontier_group` / `quarantine_frontier_group` / `lift_quarantine_to_frontier` | PR-D | §5.4 |

---

## 2. Contract items mapped to code

Legend, carried from PR-B's spec §2: **[wire]** = an existing function gains a
production caller; **[new]** = new code; **[read]** = read-only consumption.

### 2.1 Relevance (D1, D9)

| Piece | Disposition | Where |
|---|---|---|
| `ensure_relevance_tables` | already called | `crates/wenlan-core/src/db.rs:12275` (migration 109) |
| `qualified_co_citation` | **[wire]** | `relevance.rs:101` — derived at read time, never stored (S0-91, `docs/plans/2026-08-01-m6-relevance-contract.md:363`) |
| `apply_group_eligibility_change` | **[wire]** | `relevance.rs:111` |
| `normalized_pair_stats_digest` | **[wire]** | `relevance.rs:165` — the oracle's comparison |
| `common_neighbor_counts` | **[wire]** | `relevance.rs:233` — query (c). Note it takes `&libsql::Connection`, not a transaction |
| Pair-stat incremental maintenance | **[new]** | `m6/relevance_sweep.rs` |
| Adjacency top-64 materialization | **[new]** | `m6/relevance_sweep.rs` |
| The four budget instruments (S0-95, `…relevance-contract.md:446`) | **[new]** | `m6/relevance_sweep.rs` + a bench-only NVISIT harness (S0-157, `:583`) |

### 2.2 Refresh (D10)

| Piece | Disposition | Where |
|---|---|---|
| `ensure_refresh_readiness_tables` | already called | `refresh_readiness.rs:11` |
| `acquire_refresh_lease` | **[wire]** | `refresh_readiness.rs:149` |
| `replace_refresh_dependencies` | **[wire]** | `refresh_readiness.rs:229` |
| `space_refresh_dependencies_ready` | **[wire]** | `refresh_readiness.rs:261` — PR-D precondition 5 |
| `record_soak_receipt` | **[wire]** | `refresh_readiness.rs:438` |
| Stale-page selection sweep (S0-55: no durable `queued` row, `…refresh-matrix.md:65`) | **[new]** | `m6/refresh_sweep.rs` |
| Anchoring bijection (S0-58, `:146`) evaluated as a dry run | **[new]** | `m6/refresh_sweep.rs` |
| `reconcile_supported_pages` / `evaluate_page_support` | **[read]** | `crates/wenlan-core/src/db/claim_derivation.rs:2996`, `:3590` |

### 2.3 Overview subscriptions (D11)

| Piece | Disposition | Where |
|---|---|---|
| `ensure_overview_subscription_tables` | already called | `crates/wenlan-core/src/db.rs:12271` |
| `has_active_overview_subscription` | **[read]** | `signals.rs:632` |
| `resolve_space_name` | **[read]** | `signals.rs:606`; its doc at `:601`–`:605` records that every evidence table keys on the **name**, not the id |
| Merge-loser detach (S0-76 `…overview-matrix.md:222`, S0-163 `:477`) | **[new]** | `m6/overview_lifecycle.rs` |
| `install`-scope registration of the existing global Overview page (S0-73, `:137`) | **[new]** | `m6/overview_lifecycle.rs` |

### 2.4 The lane

| Piece | Disposition | Where |
|---|---|---|
| `run_maintenance_shadow_turn` | **[new]** | `m6/maintenance.rs` |
| `MaintenanceState`, `MaintenanceTurn` | **[new]** | `m6/maintenance.rs`, mirroring `ShadowState`/`GenesisTurn` (`crates/wenlan-core/src/m6/shadow.rs:105`, `:60`) |
| `LeasePhase::{Relevance, Refresh}` | **[new]**, in an existing enum | `crates/wenlan-core/src/m6/leases.rs:37` |
| `m6_maintenance_shadow_enabled()` | **[new]** | `crates/wenlan-core/src/db.rs`, beside `entity_page_reconcile_enabled` (`:1707`) |
| The sibling spawn | **[new]** | `crates/wenlan-server/src/main/runtime.rs`, after the genesis block ending at `:417` |

---

## 3. The D-predicates in force, and where each is enforced

Only the predicates PR-C can break. Each names the single place a reviewer
checks, so "is D9 enforced?" is a lookup and not a reading exercise.

| # | Predicate | Enforcement point | Fails how, if wrong |
|---|---|---|---|
| **D1** | Independence floor: co-citation requires ≥ 3 *distinct independence groups*, undecayed and unsmoothed (S0-88, `…relevance-contract.md:288`) | `qualified_co_citation` (`relevance.rs:101`) called at read time on `distinct_group_count`; the bit is never stored | A stored `floor_cleared` column would let a retraction leave a stale `true`. The schema has no such column — `relevance.rs:31`–`:43` — so this is structural |
| **D6** | One durable lease registry; `grouping_leases` is the only table granting exclusive rights to an automatic **phase** (I-6, `docs/plans/2026-08-01-m6-state-machines.md:478`) | `LeasePhase::ALL` (`leases.rs:47`) widens from 2 to 4; `reap_expired_m6_phases` (`:182`) is written over `ALL` and widens for free | A new per-phase lease table. §7.4 gates it |
| **D8** | Work bounds per turn; a cap refuses to start work and never terminalizes (S0-54, via `constants.rs:84`, `:88`) | §4.3's per-turn table, each cap a named constant in `m6/constants.rs` | A cap that marks a row done instead of leaving it selectable |
| **D9** | Bounded relevance: ≤ 32 candidate endpoints, ≤ 64 neighbors/endpoint, ≤ 2016 pairs/group, hub weight `min(1, 64/d)` (`…relevance-contract.md:315`–`:321`, `:344`) | `MAX_CANDIDATE_ENDPOINTS` / `MAX_NEIGHBORS_PER_ENDPOINT` (`relevance.rs:20`–`:21`) bound the **query**, not a post-hoc `Vec` truncation (S0-90, `:349`) | A `LIMIT` on an unindexed predicate visits the whole table and still reports 512 rows — which is why G6.11 is instrumented, not textual |
| **D10** | Guarded refresh: an ambiguous anchor or a dropped support rejects the **entire** result, never a per-claim skip (S0-57 `…refresh-matrix.md:112`, S0-142 `docs/plans/2026-08-01-m6-mutation-catalog.md:480`) | One `RefreshVerdict::Rejected { reason }` returned from the bijection check; there is no per-claim skip path to take | Skipping the ambiguous claim and "publishing the rest" is the natural implementation and the exact failure G7.2 exists to catch |
| **D11** | Overview pages are located by their **subscription row**, never by title lookup (S0-82, `…overview-matrix.md:350`) | `has_active_overview_subscription` (`signals.rs:632`) is the only locator PR-C calls | A `SELECT … FROM pages WHERE title = 'Overview'` anywhere in the maintenance modules |
| **D14** | Rollback: nothing PR-C writes may make resuming the old writer unsafe | `genesis_coverage_state.m6_mutation_count` stays 0 (§7.1); every PR-C write lands in a class D14 keeps for diagnosis (`…readiness-cutover.md:357`, the **B / C** row) | Bumping the counter for an M6-table write would falsely retire the reverse-ledger escape hatch (S0-128, `:365`) |
| **S0-11** | One clock: `unixepoch()` evaluated in-statement, never a Rust-side `now` parameter (`…state-machines.md:133`) | Every new statement in `relevance_sweep.rs` / `refresh_sweep.rs` / `overview_lifecycle.rs` | **Already contradicted by shipped code** — `acquire_refresh_lease` takes `request.now: i64` and `lease_expires_at: i64` (`refresh_readiness.rs:143`, `:139`). §13.2 is the amendment |

---

## 4. The maintenance lane

### 4.1 Second sibling lane, not extra work inside the genesis turn

**Decision.** A second `tokio::spawn` beside the genesis lane.

The DB-mutex argument is the one most likely to be made backwards, so state it
first. `MemoryDB` holds a single `tokio::sync::Mutex<libsql::Connection>` and
both lanes reach it through the same `Arc<MemoryDB>`. **A second lane therefore
buys no database parallelism at all** — it adds a second contender for one
mutex. `WENLAN_ENABLE_ENTITY_PAGE_RECONCILE`'s note in
`crates/wenlan-core/AGENTS.md` records the failure mode concretely: one sweep
held that mutex 18.88s at 10k entities and every foreground request queued
behind it. That hazard is identical under both options, and the thing that
actually fixes it is identical too — a per-turn work bound plus never holding a
guard across an `await`. **The lane choice is not a mutex-contention decision.**

Two arguments do discriminate, and both favor a sibling:

1. **Independent failure.** The genesis loop backs off 5s on any error
   (`runtime.rs:409`). Folding maintenance into `run_genesis_shadow_turn` makes
   a relevance-sweep error throttle candidate preparation, and vice versa. That
   is exactly the reasoning PR-B3 wrote down for splitting genesis off M5's
   truth loop (`runtime.rs:316`–`:319`: *"the two must be able to fail, back
   off, and be disabled independently"*). The same sentence applies verbatim
   one rung out.
2. **Cadence.** The genesis loop ticks at 100ms while working and 1s idle
   (`runtime.rs:391`, `:393`). Maintenance work is naturally slower: every
   comparable ambient lane in this repo (`WENLAN_ENABLE_EDGES_RECONCILE`,
   `WENLAN_ENABLE_ENTITY_PAGE_RECONCILE`, `WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE`)
   runs on a 30-minute interval. Running a relevance slice on a 100ms tick is
   waste; running it on its own interval *inside* the genesis turn is a second
   lane wearing the first lane's clothes, with a shared error state it does not
   want.

**Counter-argument, stated and rejected.** A third background task is a third
thing to reason about at shutdown and a third place to get the guard wrong. It
is answered by making the new lane a near-literal copy of the block at
`runtime.rs:338`–`:417` — same guard (`:359`), same biased select (`:355`),
same `sleep_or_shutdown` exit (`:413`) — so the review question is "does it
differ from its sibling?" rather than "is it correct from first principles".

### 4.2 Cadence, and why the idle interval is 300s

| Condition | Next delay | Reason |
|---|---|---|
| Turn did work | 1s | Slices drain a backlog one at a time; 1s keeps a 10k-pair backlog draining without pinning the connection mutex |
| Turn idle | 300s | The soak floor is the constraint, not throughput — see below |
| Turn errored | 5s | Matches `runtime.rs:409` |

300s is derived, not picked. `m6_readiness_soak_receipts` requires
`observed_turns >= 20` and a window `>= 259200` seconds
(`refresh_readiness.rs:104`–`:105`), ratified as S0-126
(`…readiness-cutover.md:316`, and the ratification note at `:621`). At a 300s
idle poll a wholly idle daemon accumulates 20 turns in 100 minutes, so the
turn-count component can never be the thing that stalls a 72-hour soak — which
is the correct division of labour, because S0-126's whole point is that
*elapsed time is never sufficient*. A 30-minute interval matching the other
ambient lanes would take 10 hours to reach 20 turns and would make the turn
count a second, accidental time gate.

### 4.3 Per-turn bounds, and the ordering rule

One turn does **at most one** of the numbered items, in priority order, and
returns.

> **Ordering rule — drain before intake. "I cannot take on more work" must
> never preempt "finish the work I have."** Every step that *finishes* an
> existing unit is ordered ahead of every step that *starts* a new one, and a
> refusal to start is never an early return past a drain step. §4.4 is the
> shipped counterexample this rule is written from.

This is a **program invariant, not a local convention of this driver.** It was
reached twice, independently and from opposite directions: from the genesis
lane's livelock, where a pending-cap refusal returns before the only step that
drains the cap (§4.4), and from the refresh table's stranded lease, where a
takeover placed inside acquire would let intake preempt the reclaim that makes
intake possible (§5.2.1). Two rungs, two different resources — a per-space
candidate cap and a partial unique index — one rule. PR-D and PR-E inherit it:
any new ambient step is placed by asking whether it finishes an existing unit
or starts a new one, and the answer fixes its priority.

| Priority | Kind | Work | Bound | Constant |
|---|---|---|---|---|
| 1 | recovery | Startup scan, once per process | reap expired `relevance`/`refresh` leases; count one daemon start | `LeasePhase::ALL` (`leases.rs:47`) |
| 2 | **drain** | Reclaim one expired `'leased'` refresh job — the Q10 statement, §5.2.1 | one job | `REFRESH_LEASE_TTL_SECONDS` |
| 3 | **drain** | One page's refresh dry run, from an already-leased job | exactly one page; ≤ 64 roots | `constants::ROOTS_PER_CANDIDATE_CAP` (`constants.rs:79`) |
| 4 | **intake** | Lease one newly-selected stale page | one page | `idx_genesis_refresh_jobs_active` (`refresh_readiness.rs:37`) |
| 5 | maintenance | One space's relevance slice | ≤ 32 endpoints, ≤ 64 neighbors/endpoint, ≤ 4 queries, ≤ 512 materialized rows | `relevance.rs:20`, `:21`; S0-95's instruments |
| 6 | maintenance | One space's overview-lifecycle reconcile | one space | new `OVERVIEW_RECONCILE_SPACES_PER_TURN = 1` |
| — | | otherwise | `MaintenanceTurn::Idle` | — |

Priorities 2 and 3 sit ahead of 4 for exactly the §4.4 reason: leasing a new
page is intake, and evaluating an already-leased one is the only thing that
retires a row from the active partial index. Inverting them reproduces the
livelock with different nouns.

**Two properties the enum must carry**, both of which the shipped genesis
driver gets wrong (§4.4):

1. `MaintenanceTurn::did_work()` is **false** for every refusal variant
   (`LeaseHeld`, `RefusedActiveCap`), not just for `Idle`. A refusal wrote
   nothing, so it must take the 300s idle delay, not the 1s working delay.
2. `state.rotation` advances on **every** turn that did not do durable work —
   refusals included — not only on the fall-through path. A refused space must
   not be re-selected on the next turn.

Rotation itself reuses `ShadowState`'s pattern (`shadow.rs:105` carries a
`rotation: usize`, chosen as an index rather than a name so a deleted space
cannot wedge it).

### 4.4 The livelock PR-C must not reproduce

Verified in shipped `shadow.rs` on this branch, link by link:

| # | Link | Where |
|---|---|---|
| 1 | A pending-cap refusal returns a `Some` turn: `ObserveOutcome::RefusedPendingCap => return Ok(Some(GenesisTurn::RefusedBudget))` | `shadow.rs:350` |
| 2 | `Some` from `prepare_one` makes `prepare` commit and return `Some` | `shadow.rs:303`–`:306` |
| 3 | `Some` from priority 3 returns before priority 4 (`finalize_one`) is ever reached | `shadow.rs:176`–`:181` |
| 4 | `did_work()` is `!matches!(self, GenesisTurn::Idle)`, so `RefusedBudget` reads as work → the 100ms delay | `shadow.rs:91`–`:93`, `runtime.rs:390`–`:393` |
| 5 | `state.rotation` advances only after all three priorities returned `None` | `shadow.rs:184` |

Composed: once a space reaches `PENDING_CANDIDATES_PER_SPACE_CAP` (128,
`constants.rs:84`), the driver spins on that space at 100ms forever. It never
reaches `finalize_one` — **the only step that moves a candidate to a terminal
state and therefore the only thing that drains the pending count** — and it
never rotates, so every other space starves too.

The instructive part is not the bug, it is why 12 passing driver tests did not
see it: **a wedged lane writes nothing forbidden.** Gates shaped "the shadow
does not write X" are all satisfied by a shadow that does nothing at all. A
suite that is exhaustive on safety and silent on liveness cannot distinguish
"correctly inert" from "wedged", and PR-C's gate families inherit that blind
spot unless §8.5 is written. That is why §8.5 exists as a family rather than a
single test.

### 4.5 The flag

```
WENLAN_ENABLE_M6_MAINTENANCE_SHADOW      # default OFF; 1|true|yes enables
```

- **Parsed** by a new `m6_maintenance_shadow_enabled()` in
  `crates/wenlan-core/src/db.rs`, copying `entity_page_reconcile_enabled`
  (`:1707`) and its pure inner `entity_page_reconcile_enabled_value` (`:1712`,
  matching exactly `"1" | "true" | "yes"` after trim + lowercase). The pure
  half is what the unit test drives.
- **Gated at the spawn**, not inside the turn: when the flag is off, no task
  exists, so an idle poll costs nothing and the lane cannot be re-enabled by a
  mid-life environment change.
- **Default OFF** for the same reason every comparable lane is: the RSS and
  foreground-latency ceilings of a full-store relevance pass have not been
  measured. Say that in the doc entry, in those words, so the burn-down
  condition is explicit.
- **Documented in `crates/wenlan-core/AGENTS.md`**, in the "Retrieval env
  flags" list, **in the same PR that adds the `std::env::var` read**.
  `drift_guard` teeth #2 is fail-closed: a new undocumented `WENLAN_*` flag
  fails the build. This is a hard ordering constraint on the slice that lands
  the flag (§12), not a follow-up.

### 4.6 Leases

`LeasePhase` gains two variants. S0-3 (`…state-machines.md:125`) fixes the
TTLs, and `leases.rs:33`–`:35` already names the plan in a doc comment:
*"`relevance` and `refresh` are reserved by S0-3 and land with PR-C; they are
absent here rather than declared unused, because a phase value with no acquirer
is a value nothing checks."*

| Variant | Stored `phase` | TTL | Constant to add |
|---|---|---|---|
| `Relevance` | `relevance` | 300s | `RELEVANCE_LEASE_TTL_SECONDS` |
| `Refresh` | `refresh` | 900s | `REFRESH_LEASE_TTL_SECONDS` |

`ALL` becomes `[LeasePhase; 4]`. Two things widen for free and one must be
checked: `reap_expired_m6_phases` (`leases.rs:182`) builds its `IN (…)` from
`ALL.len()` with generated digit placeholders, so it widens correctly and keeps
phase values bound; the acquire (`:73`) and `owns` (`:119`) already parameterise
`phase`. The thing to check is that widening `ALL` does **not** start reaping
M4's `community` rows — it does not, because `community` is not a `LeasePhase`
variant, and `leases.rs:173`–`:181` records that scoping as deliberate.

**Both TTLs are un-re-derived**, exactly as `constants.rs:96`–`:97` says of the
genesis TTL. S0-3's binding rule is `TTL > (model call timeout + finalize
budget)`; PR-C makes no model call, so both are vacuously satisfied *in shadow*
and must be re-derived before PR-E. Carried as Q9.

---

## 5. Per-item design

### 5.1 Relevance

#### 5.1.1 The decay clock, and the gap the artifacts leave

The estimator cells are **decayed sums**, not counts
(`…relevance-contract.md:238`):

```
contribution(g) = hub_weight(g) * 0.5 ^ (age_days(g) / 180)
```

with `age_days` measured from the group's most recent contributing root's
`provenance_roots.created_at` (`:244`–`:246`). `provenance_roots` has that
column (`crates/wenlan-core/src/db.rs:8894`) and a
`status IN ('ingesting','active','failed')` CHECK at `:8893` that the
eligibility spine reads.

Decay is therefore a function of *(stored root timestamp, evaluation time)*.
That is the gap: **nothing pins the evaluation time.** S0-91 states the oracle
as byte equality of a normalized snapshot "at a fixed relevance generation"
(`:363`–`:377`), and S0-92 fixes accumulation order and 9-dp rounding to make
the equality exact (`:388`). Neither pins the *instant*. An incremental update
that decays a touched pair to `unixepoch()` at T₁ and a full recompute at
T₂ > T₁ produce different cells for every untouched pair, so the oracle fails
continuously for a reason that is not a bug — and worse, it fails *later*, in a
way that looks like an incremental defect.

`space_graph_state` cannot supply the pin: its columns are `space`,
`graph_generation`, `grouping_generation`, `published_generation`, `dirty`
(`crates/wenlan-core/src/db.rs:10575`–`:10581`). No timestamp.

**Adjudication (Q1): the decay reference is a per-space monotone counter in
`m6_counters`, and no migration is needed.**

`m6_counters` is `(space_id, space, name, value INTEGER >= 0)` with
`PRIMARY KEY(space_id, name)` (`remaining_substrate.rs:147`–`:154`) and five
triggers: monotone on update (`:155`), monotone on insert-replace (`:161`),
two identity guards (`:172`, `:186`), and no-delete (`:202`). A unixepoch
stored there is a monotone integer that may never decrease, which is exactly
the safety property a decay reference needs — a reference that moved backwards
would make a pair's decayed weight *increase*.

```
name  = "relevance_decay_reference"
value = unixepoch() at the moment this space's pair table was last re-referenced
```

Every incremental update decays to the **stored** reference, never to
`unixepoch()`. The reference advances only in a full re-reference pass, which
rewrites every pair row for the space in one transaction. Consequences:

- The oracle becomes exact by construction: every row at a given reference
  decays to the same instant, so incremental and full agree without a
  tolerance, and S0-92's fixed accumulation order plus 9-dp rounding do the
  remaining work.
- S0-11 is honored: `unixepoch()` is still evaluated in-statement; it is stored
  once rather than read per row.
- Staleness is bounded and visible: the reference's age is a query, so "this
  space's relevance is decayed as of N days ago" is answerable, and the
  re-reference pass is a normal bounded slice.

The negative control S0-91 demands (`:383`–`:386`) survives: the test asserts
the incremental path's row-visit counter stayed within the incremental bound,
so an implementation that re-references on every mutation fails.

#### 5.1.2 The slice

One turn, one space, one transaction per commit point:

1. Acquire `LeasePhase::Relevance` for `(space, grouping_generation)` via
   `leases::acquire` (`leases.rs:73`). `None` → another holder; return `Idle`.
   `input_generation` is `space_graph_state.grouping_generation` per S0-1
   (`…state-machines.md:123`).
2. Select ≤ 32 candidate endpoints — S0-90's cap **at the query**
   (`…relevance-contract.md:349`), with the truncation recorded.
3. For each endpoint, `common_neighbor_counts` (`relevance.rs:233`, query (c)),
   bounded to 64 neighbors.
4. Recompute the affected pairs' four cells against the **stored decay
   reference**, accumulating in `independence_group_id ASC` and rounding to
   9 dp (S0-92).
5. `UPSERT` into `m6_pair_stats`; rewrite the endpoint's `m6_adjacency` rows,
   ordered `(support_recency DESC, page_id ASC)` per S0-89 (`:325`), ranks
   1..=64 (`relevance.rs:52` CHECK).
6. Release the lease in the same transaction as the last write.

`m6_pair_stats.updated_generation` records the `grouping_generation` the row
was computed under — the column exists (`relevance.rs:40`) and is otherwise
unused. It is **not** overloaded to carry the decay reference; that would be a
second meaning on one column and is the kind of thing §13 exists to prevent.

#### 5.1.3 The budget

S0-95's four instruments (`…relevance-contract.md:446`) plus S0-157's
aggregation rule (`:583`). Three run in ordinary tests; the fourth is
bench-only:

| Instrument | Bound | Where asserted |
|---|---|---|
| Queries per evaluation | ≤ 4 | Counting wrapper around the connection, unit test — **CI-gated** |
| Materialized rows | ≤ 512 | Decoded-row counter, unit test — **CI-gated** |
| Wall clock | ≤ 50ms on a 5k-degree hub | `R-BENCH-MAX`, bench — **measured, not gated** (§8.6's ruling) |
| `SQLITE_SCANSTAT_NVISIT`, **summed over every scan loop of every statement** | ≤ 2,176 | Bench-only build with `LIBSQLITE3_FLAGS=SQLITE_ENABLE_STMT_SCANSTATUS` — R-1 option (b′), S0-157 |

The fourth is the one that fails quietly if approximated: the catalog is
explicit (`…mutation-catalog.md:443`–`:448`) that a textual `LIMIT 512` on an
unindexed predicate visits the whole table and still reports 512 rows. It is a
**bench gate, not a CI gate**, because the scanstatus build is not the CI
build; §8.7 records that as a declared deferral rather than a silent one.

### 5.2 Refresh

#### 5.2.1 Selection

S0-55 (`…refresh-matrix.md:65`) forbids a durable `queued` row — it is the
sweep's selection, re-derived each turn. The DDL agrees: `genesis_refresh_jobs`
restricts `state` to `'leased' | 'retry' | 'finalized' | 'revision_card'`
(`refresh_readiness.rs:25`–`:26`) with a comment saying `queued` and
`generated` may never become durable states. So the sweep re-derives its
selection from live page/claim state every turn and holds no cursor.

The active-job exclusion is the partial unique index
`idx_genesis_refresh_jobs_active ON (page_id, base_page_version) WHERE state IN
('leased','retry')` (`:37`–`:39`). This is insert-and-refuse, matching the
project's exclusion discipline: `acquire_refresh_lease` (`:149`) does
`INSERT OR IGNORE`, and on zero rows falls through to a due-retry `UPDATE`
guarded by `state = 'retry' AND COALESCE(next_attempt_at, 0) <= ?12`
(`:197`–`:200`). No check-then-insert anywhere.

**Expiry takeover is missing and C2 adds it (Q10).** `lease_expires_at` is a
write-only column. It occurs exactly six times in `refresh_readiness.rs` — the
DDL at `:28`, the struct field at `:139`, and four writes: the INSERT column
list `:158` with its bind `:170`, and the UPDATE `SET` `:191` with its bind
`:209`. It appears in no `WHERE` clause. Symmetrically, `state = 'retry'` is
never written: `'retry'` occurs three times — the CHECK at `:26`, the partial
index predicate at `:39`, and the retry arm's own guard at `:199`. **The state
the retry arm waits for has no producer, and the column that would detect
expiry has no reader**, so a `'leased'` job whose worker dies is stranded
permanently: the `INSERT OR IGNORE` is refused by the partial unique index and
the retry `UPDATE` matches nothing.

C2 adds the reclaimer as **priority 2 of the turn** (§4.3) — its own drain step
ahead of intake, not a clause bolted onto acquire. One job per turn, per space:

```sql
UPDATE genesis_refresh_jobs
   SET state = 'retry', lease_token = NULL, lease_expires_at = NULL,
       next_attempt_at = NULL, updated_at = unixepoch()
 WHERE job_id = (
     SELECT job_id FROM genesis_refresh_jobs
      WHERE space = ?1
        AND state = 'leased'
        AND lease_expires_at IS NOT NULL
        AND lease_expires_at <= unixepoch()
      ORDER BY lease_expires_at, job_id
      LIMIT 1
 )
```

`job_id` is `INTEGER PRIMARY KEY` (`:16`), so the subquery is a rowid lookup,
and `idx_genesis_refresh_jobs_space_compatibility` (`:40`–`:41`) is
space-leading. The expiry predicate is M5's, verbatim in shape:
`lease_next_derivation_job` carries `status = 'leased' AND lease_expires_at IS
NOT NULL AND lease_expires_at <= ?1` in both of its statements — the reap at
`crates/wenlan-core/src/db/claim_derivation.rs:3392`–`:3395` and the acquire's
own selection subquery at `:3418`–`:3421`.

Three design points, each settled against shipped code rather than by analogy:

- **Every turn, not only at startup.** M6's only startup reap is
  `recovery::scan` → `leases::reap_expired_m6_phases` (`recovery.rs:60`,
  `leases.rs:182`), and `shadow.rs:149`–`:152` runs recovery **once per
  process**. A turn that dies mid-dry-run does not restart the daemon, so a
  startup-only sweep would strand the job until an operator restarts it. M5
  agrees by construction: both of its expiry predicates live in the acquire
  path and neither is a startup step.
- **Not keyed to the selection either.** A takeover keyed on the
  `(page_id, base_page_version)` the sweep just re-derived would only reclaim
  pages the selection picks again — and selection is re-derived from live
  page/claim state every turn (above), so a page that stopped being stale would
  never be revisited and its `'leased'` row would hold
  `idx_genesis_refresh_jobs_active` against that key permanently. The reclaimer
  therefore scans by expiry, not by selection.
- **A separate statement, not a widened retry guard.** Folding the expiry
  predicate into `:199` as `state = 'retry' OR (state = 'leased' AND
  lease_expires_at <= ?)` would make takeover implicit at every acquire and
  unnameable in a gate. As its own step it returns its own row count, so the
  **expiry takeover** row of G7 (§8.2) asserts takeover happened rather than
  inferring it from a successful acquire — and priority 2 sitting ahead of
  priority 4 is the §4.3 ordering rule applied to this table: reclaiming a dead
  worker's job finishes work, leasing a new page starts it.

`genesis_refresh_jobs` is a **separate table** from `m6_leases`, so widening
`LeasePhase::ALL` does not reach it; the startup reap in §8.8 covers it as a
distinct statement, and that gap is amendment 13.1.

#### 5.2.2 The dry run

Per page: compute the anchoring, evaluate it as a **bijection** over the claim
inventory (S0-58, `…refresh-matrix.md:146`), and return a verdict. Nothing is
published; `try_update_page_content` is never called (S0-69 is PR-E's, `:399`).

```
enum RefreshVerdict {
    WouldPublish { dependencies: usize },
    Rejected { reason: RejectReason },   // wrong anchor | ambiguous | support dropped
    Requeued { gate: &'static str },
}
```

`Rejected` is whole-result by construction (D10, §3) — there is no per-claim
skip to take, which is what makes G7.2 falsifiable rather than aspirational.

#### 5.2.3 The dependency snapshot, and why the shadow writes it

`m6_refresh_dependencies` is documented as "the CURRENT exact dependency
snapshot for a page", replaced wholesale by the finalizer
(`refresh_readiness.rs:43`–`:45`), and `replace_refresh_dependencies` (`:229`)
deletes then re-inserts inside the caller's transaction. In shadow there is no
finalizer, so a naive reading says PR-C writes nothing here.

**Adjudication (Q2): PR-C writes the snapshot from the dry run.** PR-D
precondition 5 (`…readiness-cutover.md:243`) is the anti-join

```sql
m6_refresh_dependencies d LEFT JOIN provenance_roots r ON r.root_id = d.root_id
WHERE d.space = ?1 AND (r.root_id IS NULL OR r.status <> 'active')
```

shipped as `space_refresh_dependencies_ready` (`refresh_readiness.rs:261`–`:282`).
Over an empty table it returns `true` — a green precondition that has checked
nothing. That is precisely the vacuous-truth failure the catalog makes
mandatory to test against (`…mutation-catalog.md:219`, *"the vacuous-truth
mutation is mandatory and must be written first"*). Leaving the table empty
through the whole of PR-C would hand PR-D a precondition that passes on an
empty implementation and a broken one alike.

The write is legal and semantically correct: `m6_refresh_dependencies` is
M6-owned, and the snapshot is "what a refresh of this page at this version
depends on", which the dry run computes from live state — the same value the
finalizer would write. The table has no FK on `root_id` **by design**, so that
deleting a live root leaves the missing identity observable to the anti-join
(`refresh_readiness.rs:45`–`:46`); PR-C must not add one.

#### 5.2.4 Readiness stage C and the soak receipt

PR-B3's `evidence.rs` writes stage `"B"`. PR-C drives stage `C`:

| Step | Call | Guard |
|---|---|---|
| Create the row | `initialize_readiness(space, 'C', '-')` (`refresh_readiness.rs:329`) | The DDL CHECK at `:73`–`:78` enforces `signal = '-'` for stages A–D — S0-152's non-null sentinel (`…readiness-cutover.md:102`) |
| `off → preparing` | `transition_readiness` (`:381`), legality via `transition_is_legal` (`:370`) | Every transition bumps the epoch (S0-120, `:135`) |
| `preparing → committed` | same | Once the space has produced at least one verdict of each of the three kinds |
| Record the soak | `record_soak_receipt` (`:438`) with `SoakEvidence` (`:425`) | The `m6_soak_receipt_fence_guard` trigger (`:111`–`:122`) aborts a receipt that does not name the current readiness fence |

The receipt's six numeric components are carried as `m6_counters` rows so they
survive restart and cannot decrease:

| Counter name | Feeds | Must be |
|---|---|---|
| `maintenance_soak_window_started_at` | `window_started_at` | set once |
| `maintenance_turns` | `observed_turns` | ≥ 20 (`:105`) |
| `maintenance_daemon_starts` | `daemon_starts` | ≥ 1 (`:106`) |
| `maintenance_old_writer_mutations` | `old_writer_mutations` | 0 (`:107`) |
| `maintenance_work_bound_violations` | `work_bound_violations` | 0 (`:108`) |
| `maintenance_support_regressions` | `support_regressions` | 0 (`:109`) |

The three zero-CHECKs are the teeth: a counter that ever bumps makes the
receipt un-insertable, so a soak cannot be talked into passing. **The 72-hour
window CHECK (`:104`) means no test can wait for a real soak** — tests inject
stored values, which is exactly the testability property S0-11 was chosen for
(`…state-machines.md:133`: *"a test can move it only by moving stored values,
which is what makes the crash matrix reproducible"*).

**Honesty limit, carried as Q8.** S0-126 component 1 is "zero mutations to
pages in this space from any caller **outside the D12 manifest**"
(`…readiness-cutover.md:319`). The manifest fence is G9's, and G9 is PR-D's.
PR-C's `maintenance_old_writer_mutations` can only assert the M6 half — that
M6 itself wrote nothing. A stage-C receipt is therefore honest *for stage C*
and is **not** sufficient evidence for stage D. The spec says so rather than
letting a stage-D reviewer infer a stronger claim from a green row.

### 5.3 Overview subscriptions

#### 5.3.1 The chicken-and-egg, stated plainly

`m6_overview_subscriptions` gates signals 3 and 4: `space_overview`
(`signals.rs:65`) refuses when `has_active_overview_subscription(tx, "space",
&space)` is true (`:71`), and the evidence-cluster signal at `:110` sits beside
it. The locator is `SELECT EXISTS (… WHERE scope_kind = ?1 AND scope_id = ?2
AND state = 'active')` (`signals.rs:632`).

So a subscription row **suppresses** genesis for its scope. Creating one
requires a `page_id` — the DDL keys the row to a page — and the only page a
community- or space-overview subscription could name is one PR-E mints. If
PR-C created subscriptions for scopes whose pages do not exist yet, it would
permanently starve the two signals it is meant to unblock: the shadow would
refuse to propose overviews forever, and the refusal would look like a correct
gate.

**Adjudication (Q4): PR-C wires only the halves that need no minted page.**

| Half | PR-C? | Reason |
|---|---|---|
| Merge-loser **detach** (`active → detached`) | **Yes** | A state flip on an M6-owned row. S0-76 (`…overview-matrix.md:222`) makes detach automatic while transfer and retire are not; S0-163 (`:477`) extends it to the no-survivor case (all participants detach) |
| `install`-scope registration of the **existing global** Overview page | **Yes** | S0-73 (`:137`) makes the existing global Overview a distinct scope. `scope_kind='install'` is invisible to signals 3/4, which query `'space'` and `'community'`, so registering it starves nothing |
| Split → **no** proposal | **Yes**, as an asserted no-op | S0-75 (`:199`): split stages no proposal. The gate is that nothing is written |
| Creating a `space`/`community` subscription | **No** | Needs a minted page — PR-E |
| Staging the merge **proposal** into `refinement_queue` | **No** | §1.3 and Q6 |

#### 5.3.2 The vacuity problem, and the mandatory empty-inventory case

Detach is **vacuous today**: with zero subscription rows, a detach sweep runs
over an empty set and passes. That is exactly the shape the catalog forbids
shipping unmarked. Per S0-137's mandatory vacuous-inventory rule
(`…mutation-catalog.md:227`), the detach gate must ship with an explicit case
that **constructs** subscription rows in the fixture — a synthetic merge with a
determinate survivor (G8.7a) and one with none (G8.7b) — so the assertion has
something to be false about. A detach test with an empty fixture is not a
weaker test; it is not a test.

### 5.4 The frontier-policy remainder: PR-C wires none of the three

The three unwired writers are `suppress_frontier_group` (F7 from
`exclusively_claimed`, `frontier_policy.rs:151`), `quarantine_frontier_group`
(F9, `:302`), and `lift_quarantine_to_frontier` (F12, `:355`).

**Adjudication (Q3): all three are deferred to PR-D, declared, not omitted.**

Three findings carry it:

1. **The read side is already complete.** Machine F's six states are all
   reconciled today: `frontier.rs:115` and `:119` LEFT JOIN
   `genesis_suppression` and `genesis_quarantine` into the differential query,
   so a quarantined group already reads as quarantined and I-1 already holds
   over it. Nothing about the frontier is broken by leaving the writers
   unwired.
2. **F9 has no automatic trigger anywhere in Stage-0.** Artifact 5's parking
   enumeration is explicit — P9 (`docs/plans/2026-08-01-m6-frontier-policy.md:431`)
   says quarantine *"requires an explicit human/policy reason"*, and S0-52
   (`:344`) makes `reason` `TEXT NOT NULL` with no default precisely so a group
   cannot arrive in quarantine by accident. There is no operator surface in a
   shadow lane and no policy rule named in any artifact. Wiring F9 would mean
   inventing the policy, which is speculative surface.
3. **F7-from-`exclusively_claimed` has no machine-A trigger.** Machine F's F7
   row (`…state-machines.md:456`) admits `exclusively_claimed | surfaced_card →
   suppressed`, triggered by "candidate suppressed, or card dismissed by a human
   (A17)". But machine A has exactly one edge into `suppressed` — A17, from
   `review_required` (`:195`), which is the `surfaced_card` leg and is already
   wired through `dismiss_card_to_suppression` at `candidates.rs:911`. **No
   machine-A transition reaches `suppressed` from `prepared`, `inferencing`, or
   `validating`.** So `suppress_frontier_group`'s remaining leg is either dead
   surface or a machine-A gap. §13.3 records it as an artifact contradiction
   rather than specifying against it.

F12 follows F9: a lift is only meaningful once something can quarantine.

Each of the three ships as a `Coverage::Deferred` row in the catalog registry
(`catalog_test.rs:41`) naming PR-D as owner, so an unexecuted writer is a
visible attributed entry rather than a silent absence — the property
`deferrals_are_declared_and_bounded` (`catalog_test.rs:220`) enforces, and
whose pinned count that test's assertion at `:228` must be updated to match.

### 5.5 The frozen benchmark corpus — a genuine PR-C dependency

#### 5.5.1 Why this is blocking rather than nice-to-have

**PR-B's benchmark is measure-only; PR-C's relevance benchmark is a gate.**
Artifact 6 §10 (`…relevance-contract.md:854`–`:861`) sets hard pass/fail
limits — `R-BENCH-MAX` 50ms, `R-BENCH-Q` ≤ 4, `R-BENCH-ROWS` ≤ 512,
`R-BENCH-HUB` — and S0-98 (`:863`) refuses to soften the 50ms into a p99,
answering flakiness by *specifying the measurement* instead: the maximum over
"the Stage-0 representative 100k-memory/5k-page corpus, warm cache, single
evaluation at a time, no concurrent refinery turn" (`:876`–`:879`). S0-99
(`:883`) then forbids fixing a benchmark failure by tuning a constant.

§8.6 rules that only the two **count** limits become required CI checks and the
wall-clock ones are measured locally. That is a ruling about *where a limit is
enforced*, not a softening of the limit — S0-98's specified measurement is
exactly what the local receipt records, and S0-99 binds the recorded numbers
just as hard. The corpus is blocking either way: an ungated measurement over a
drifting corpus is worth even less than a gated one.

A threshold with no frozen corpus is not a gate: the number drifts with the
corpus and nobody can tell drift from regression. Artifact 6 says it in one
line at `:813` — *"A benchmark whose corpus is not reproducible is not a
gate."*

**The generator does not exist.** Verified: there is no
`crates/wenlan-core/benches/` directory, and a repository search for the seed
`0x6D36_0000` and for `R-BENCH` finds hits only in
`docs/plans/2026-08-01-m6-relevance-contract.md` (`:832`, `:1024`, and the §10
table) and in PR-B's spec. S0-97 (`:815`) freezes eleven `R-CORPUS-*`
constants and nothing implements them.

#### 5.5.2 The precedent, read rather than recalled

M5 froze its corpus properly and the parts are on disk:

| Part | Where |
|---|---|
| Checked-in generator, **405 lines** | `crates/wenlan-core/src/eval/m5_bench_corpus.rs` |
| Seed as a `const`, not a flag: `M5_BENCH_SEED: u64 = 0x4d35_0001` | `:21` |
| Encoding tags that make the digest domain explicit | `M5_CORPUS_ENCODING` `:24`, `M5_MANIFEST_ENCODING` `:25` |
| Deterministic RNG | `SplitMix64` `:376`, seeded at `:273` |
| Manifest digest over corpus + inputs | `canonical_manifest_digest` `:302`–`:313` |
| Refuse-on-mismatch | `verify_manifest_digest` `:316` |
| Checked-in digest fixture (one 64-hex line) | `crates/wenlan-core/tests/fixtures/m5_bench_corpus.sha256` |
| The bench that consumes it, and refuses | `crates/wenlan-core/tests/m5_bench.rs:28` (`include_bytes!`), `:209`, `:218`, `:228`, `:238`, `:255` |

The reasoning is stated at
`docs/plans/2026-07-27-m5-entailment-budget-spec.md:290`–`:312`: a corpus is
frozen only if a reviewer can regenerate it byte-for-byte and check that they
did; naming a size and a property is not freezing.

Two corrections to the brief that reached this spec, both checked: the module
is **405** lines, not 644 (644 is its file mode); and the bench lives in
`crates/wenlan-core/tests/`, not in a `benches/` directory — which matters,
because CI's differential planner routes integration targets by path.

#### 5.5.3 Adjudication (Q13): a sibling module, reusing the freeze machinery

**Sibling `crates/wenlan-core/src/eval/m6_bench_corpus.rs`, not an extension of
the M5 module.**

M5's corpus is the right *pattern* and the wrong *shape*. Its public surface is
page-shaped end to end — `PageSizeDistribution` (`:34`), `PageSizeBucket`
(`:44`), `FIXED_BUCKET_BOUNDS` (`:30`), `draw_page_size` (`:334`),
`AccuracyCategory` (`:52`), `write_corpus_stream` (`:268`) emitting a record
stream against a page-size distribution. None of S0-97's graph layer is
expressible in it: no provenance roots, no independence groups, no Zipf degree
distribution, no hub topology, no generated/retracted fractions. Extending it
would put two incompatible corpus shapes behind one manifest, and
`canonical_manifest_digest`'s signature (`corpus_sha256`, `distribution_bytes`,
`accuracy_bytes`, `:302`–`:306`) is already specific to M5's three inputs — a
fourth and fifth argument for M6's graph inputs would change M5's digest domain
and invalidate its checked-in fixture. That is a real cost paid for no reuse.

What *is* reused is the freeze machinery, which is generic and currently
private: `SplitMix64` (`:376`), `sha256_hex` (`:360`), `validate_sha256_hex`
(`:364`), and `DigestWriter` (`:394`) are all module-private and must be lifted
to `pub(crate)` — a mechanical change with no behavioral effect, and the only
edit PR-C makes to M5's module.

Frozen identity, on M5's pattern exactly:

| Property | M6 value |
|---|---|
| Seed | `M6_BENCH_SEED: u64 = 0x6D36_0000` — a `const`, never a flag or env var (`R-CORPUS-SEED`, `:832`) |
| Composition | The eleven `R-CORPUS-*` constants of S0-97 (`:818`–`:832`), each a named `const` so an amendment names what it changes (S0-99) |
| Encoding tags | `M6_CORPUS_ENCODING` / `M6_MANIFEST_ENCODING`, distinct strings from M5's |
| Digest fixture | `crates/wenlan-core/tests/fixtures/m6_bench_corpus.sha256`, one 64-hex line |
| Refusal | `verify_manifest_digest`'s shape: the bench **refuses to run** on mismatch rather than reporting an incomparable number |
| Bench target | `crates/wenlan-core/tests/m6_relevance_bench.rs`, matching `m5_bench.rs`'s placement |

The three hub degrees are the part most likely to be built wrong and must be
generated exactly: **5,000 / 1,024 / 65** (`R-CORPUS-HUB`, `:827`). S0-97's own
note at `:834`–`:837` says why 65 is there — it is the smallest degree at which
top-64 selection actually truncates, so it is the case an implementation with
`>` instead of `>=` gets wrong, and a corpus that omits it lets that bug pass.

#### 5.5.4 CI routing — ruled: a separate `m6-platform` filter

This is the part that makes C0 a slice rather than a chore, and it was found by
reading rather than reasoning. `drift_guard.rs:2694`–`:2724` asserts four
things about M5's platform routing:

| Assertion | Lines |
|---|---|
| the `m5-platform` change filter's path set is **exactly** seven paths — including `m5_bench_corpus.rs`, its `.sha256` fixture, and `m5_bench.rs` — compared with `!=` | `:2694`–`:2706` (paths at `:2696`–`:2702`) |
| each of the seven schedules **both** the macOS and Windows platform owners | `:2707`–`:2713` |
| the step named `"M5 bench platform controls"` carries an exact `if` condition | `:2714`–`:2716` |
| **that step's `run` is pinned verbatim** — `cargo nextest run -p wenlan-core --features eval-harness --test m5_bench` | `:2717`–`:2718` |

**The pinned `run` is what decides it.** Folding M6's paths into `m5-platform`
would route an M6 corpus edit to a step whose command names `--test m5_bench`.
There is no way to make one filter run two different bench targets without
either unpinning the `run` — which is the tooth's entire content — or running
the wrong bench on every M6 change. The fold is not untidy, it is
**structurally unavailable**. Separate `m6-platform` filter, its own exact-set
tooth, its own CI step running `--test m6_relevance_bench`.

Two constraints on how C0 lands it:

- **The fixture and the tooth update ship in the same commit.** Same
  fail-closed reasoning as the flag/`AGENTS.md` pairing (§12): `drift_guard` is
  a `#[cfg(test)]` lib test selected whenever the planner includes
  `wenlan-core`, so a new routed path landing one commit ahead of its expected
  set breaks the build for whoever pulls in between.
- **The M5 tooth is not weakened to make room.** The new tooth is purely
  additive; `m5_platform_inputs` stays exactly seven paths and its `!=`
  comparison is untouched. The gate row in §8.4 enforces this, with the fold
  itself as the RED mutation.

---

## 6. Migration posture: no migration 111

**Claim: PR-C needs no migration.** `PRAGMA user_version = 110` at
`crates/wenlan-core/src/db.rs:12332` is the highest (108 at `:12245`, 109 at
`:12281`), so 111 is unclaimed and available — the question is whether it is
*necessary*. Applying the standard from
`docs/plans/2026-08-03-m6-pr-a-followup-2-scope.md` §7.2 — prove necessity at
the call sites, or specify a no-migration design — the proof is per call site:

| PR-C needs | Existing home | Verified at |
|---|---|---|
| Pair statistics | `m6_pair_stats` | `relevance.rs:31` |
| Bounded adjacency | `m6_adjacency` | `relevance.rs:47` |
| Refresh work items with an exclusion | `genesis_refresh_jobs` + `idx_genesis_refresh_jobs_active` | `refresh_readiness.rs:15`, `:37` |
| Dependency snapshot | `m6_refresh_dependencies` | `refresh_readiness.rs:47` |
| Readiness stage C | `m6_readiness`, whose CHECK already admits `'C'` | `refresh_readiness.rs:63` |
| Soak receipt with teeth | `m6_readiness_soak_receipts` + fence trigger | `refresh_readiness.rs:84`, `:111` |
| Subscription lifecycle state | `m6_overview_subscriptions.state` | `overview_subscriptions.rs:8` |
| Turn/soak counters, monotone, per space | `m6_counters` | `remaining_substrate.rs:147` |
| **The decay reference** | `m6_counters` — the one non-obvious case, §5.1.1 | `remaining_substrate.rs:147`–`:171` |
| Two new lease phases | `grouping_leases.phase`, a `TEXT` column with no CHECK constraining its values | `…state-machines.md:63`–`:73` |

The last row is worth a sentence: adding `relevance` and `refresh` to the
`phase` column is a Rust-side enum change, not a schema change, because the
column is unconstrained text. That is the whole reason D6 says *extend the
registry, do not build a second one* (I-6, `…state-machines.md:478`).

The decay reference is the only place where a migration was a live option — a
dedicated `m6_relevance_generation(space, generation, decayed_as_of)` table is
the obvious design. It is rejected because `m6_counters` already provides
exactly the semantics needed (per-space, monotone, undeletable) and a second
table would be a second truth about the same fact.

**Consequence for §7.4's structural gate:** since PR-C adds no DDL, a gate that
asserts `PRAGMA user_version` is unchanged by the whole of PR-C is cheap and
exact. It ships.

---

## 7. The proof obligations, as checkable things

### 7.1 Zero mutation of user-visible state

**The obligation.** No `INSERT`, `UPDATE`, or `DELETE` against `pages`,
`page_*`, `entities`, `relations`, `observations`, `edges`, `memories`, or
`chunks` may execute on any PR-C code path. Three independent proofs, because
each catches what the others miss.

| # | Proof | Mechanism | RED mutation |
|---|---|---|---|
| 1 | **Structural (source text)** | A test reads `m6/{relevance_sweep,refresh_sweep,overview_lifecycle,maintenance}.rs` and asserts every statement whose first keyword is `INSERT`/`UPDATE`/`DELETE` names a table on a fixed M6 allowlist. Modeled on `catalog_test.rs:138`'s `m6_test_function_names`, which already reads M6 source text rather than trusting a string | Add `UPDATE pages SET title = ?1 …` to `relevance_sweep.rs` → the allowlist check fails |
| 2 | **Runtime counter** | After N maintenance turns over a seeded corpus, `ShadowStats.m6_mutation_count` (`evidence.rs:194`) reads 0 for every space. The counter is trigger-guarded monotone (`remaining_substrate.rs:50`), so it cannot be un-bumped | Make any PR-C write bump `m6_mutation_count` → assertion fails on the first turn |
| 3 | **Behavioral hash** | Fingerprint every enumerated table before and after the N turns; assert byte equality. This is the one that catches a mutation arriving through a *called* function rather than a literal statement — proof 1 reads only PR-C's own files | Have `refresh_sweep` call `try_update_page_content` → the `pages` fingerprint moves |

Proof 3 is not redundant with proof 1: the whole hazard of a refresh lane is
that it is one call away from the M5 finalizer, and a repo-wide grep over PR-C's
own modules is blind to a helper that mutates on PR-C's behalf.

**PR-C must not bump `m6_mutation_count`.** S0-128 (`…readiness-cutover.md:365`)
introduces the counter to answer one question: *may the old writer be resumed?*
A write to `m6_pair_stats` does not make resuming the old writer unsafe — the
old writer neither reads nor owns that table. Bumping the counter for M6-table
writes would falsely retire D14's reverse-ledger escape hatch on the first
maintenance turn, and would do it silently. The counter tracks user-visible M6
writes only; PR-C makes none, so it stays 0. Carried as Q5 because the artifact
sentence says "any M6 write" and this reading narrows it.

### 7.2 No LLM, checkable by signature

**The obligation.** The maintenance lane consults no provider — and is
checkable without reading any function body.

| Check | Assertion | RED mutation |
|---|---|---|
| Source text | None of `m6/{relevance_sweep,refresh_sweep,overview_lifecycle,maintenance}.rs` contains `LlmProvider`, `LlmEngine`, `llm_provider`, or `engine::` | `use crate::llm_provider::LlmProvider;` in `maintenance.rs` → fails |
| Signature | `run_maintenance_shadow_turn(conn: &libsql::Connection, state: &mut MaintenanceState)` takes no provider and returns no provider-bearing type | Add a provider parameter → fails |
| Lane | The `runtime.rs` maintenance block reads no `ServerState` snapshot | The genesis block already records this argument at `runtime.rs:324`–`:329`: the M5 loop re-reads `state.llm` every turn *because it needs a provider*; a lane that never touches `ServerState` has no guard to hold across an await and no seam a provider could later be threaded through unnoticed. PR-C's block copies that property; adding a `shared.read().await` fails |

The signature check is the strong one. A body check answers "does it call the
model today"; a signature check answers "could it", which is the property that
survives a later edit.

### 7.3 `genesis_enabled` is never written

The publication flag on `genesis_coverage_state` belongs to PR-E1–E4. PR-C's
proof-1 allowlist covers the table but not the column, so the gate is separate:
assert `genesis_coverage_state.genesis_enabled` is byte-identical before and
after N turns, **including for a space where it is already 1** — the
interesting case is not "the shadow leaves 0 alone" but "the shadow does not
flip a value that is already set, in either direction". RED mutation: have the
lane write `genesis_enabled = 1` for a space that completed stage C.

### 7.4 No parallel lease registry, no new migration

Two cheap structural gates, both in the spirit of G9:

- **I-6**: the set of tables with a `lease_token`-like exclusive-rights column
  is unchanged by PR-C, except that `genesis_refresh_jobs` is declared as a
  **work-item** lease, not a phase lease (§13.1). RED: add a
  `m6_relevance_leases` table.
- **Migration**: `PRAGMA user_version` is 110 before and after PR-C. RED: add a
  migration 111.

---

## 8. Gates, mutation controls, and the exit matrix

PR-C's gate families are **G6** (relevance), **G7** (refresh), **G8**
(overview), plus the four structural proof gates of §7. G6/G7/G8 rows already
exist in the catalog; PR-C's job is to bind them, and the binding mechanism is
the shipped registry.

### 8.1 G6 — `m6_relevance_is_bounded_and_safe`

| Row | Weaken | Test asserts | Deliberate break that must fail it |
|---|---|---|---|
| G6.4 | incremental pair state diverges from full recomputation | `normalized_pair_stats_digest` (`relevance.rs:165`) byte-equal after each of the eight mutations in S0-91 (`…relevance-contract.md:378`–`:381`), then a randomized interleaving | Accumulate cell sums in arrival order instead of `independence_group_id ASC` → the 9-dp digest diverges (S0-92) |
| G6.4 negative control | a full recompute masquerading as incremental | Row-visit counter stayed within the incremental bound | Re-reference the whole space on every mutation → the counter blows the bound while the digest still matches |
| G6.7 | > 2016 pairs from one group | pair count ≤ `C(64,2)` | Cap the hub *weight* but not the *selection* |
| G6.8 | > 32 candidate endpoints | endpoint count ≤ `MAX_CANDIDATE_ENDPOINTS` (`relevance.rs:20`) | Move the cap from the query to a post-hoc `Vec::truncate` — the count still passes, so the test must assert the **query plan**, not the result length |
| G6.10 | drop `64/d` hub weighting | a `d=5000` hub contributes `0.0128`, not `1.0` | Return `1.0` from `hub_weight` |
| G6.11b | a fifth query | query counter ≤ 4 | Split the adjacency read into two statements |
| G6.11c | > 512 materialized rows | decoded-row counter ≤ 512 | Materialize the candidate set before filtering |
| G6.11d / G6.14 | exceed 50ms / 2,176 NVISIT | bench-only (§8.6) | Drop `idx_m6_pair_stats_space_page_a_generation` (`relevance.rs:44`) |
| D1 floor | store `floor_cleared` | `qualified_co_citation` derived at read time; a retraction below 3 immediately reads `false` | Cache the bit in a column → a retraction leaves a stale `true` |

### 8.2 G7 — `m6_refresh_preserves_truth`

| Row | Weaken | Test asserts | Deliberate break |
|---|---|---|---|
| G7.2 | resolve an ambiguous anchor by picking one | the **entire** verdict is `Rejected` (S0-142, `…mutation-catalog.md:480`) | Skip the ambiguous claim and return `WouldPublish` for the rest — the single most natural implementation, which is why the row exists |
| G7.3 | publish with support dropped for any claim | verdict is `Rejected` | Same shape |
| G7.6 | two cards for one refresh | one coalesced verdict per `(page_id, base_page_version)` — S0-62 (`…refresh-matrix.md:207`) | Key the coalescing on `page_id` alone |
| lease exclusion | two sweeps lease one page | second `acquire_refresh_lease` returns `false` | Drop `WHERE state IN ('leased','retry')` from the partial index → both insert |
| retry claim | a not-yet-due retry is claimed | `COALESCE(next_attempt_at, 0) <= now` holds | Remove the `next_attempt_at` guard at `refresh_readiness.rs:200` |
| expiry takeover | a dead worker's lease is permanent | priority 2 flips a `'leased'` row whose `lease_expires_at` is in the past to `'retry'` (**one** row reported), and the next `acquire_refresh_lease` for that key then returns `true` (§5.2.1, Q10) | Delete the priority-2 statement — the `INSERT OR IGNORE` is refused by the partial index and the retry `UPDATE` matches nothing, so the page is stranded forever |
| expiry takeover — positive control | takeover fires on a live lease | a `'leased'` row with `lease_expires_at` in the **future** is left alone and `acquire_refresh_lease` returns `false` | Drop `lease_expires_at <= unixepoch()` from the predicate → the row above still passes, which is why the control is a separate row (S0-155) |
| expiry takeover — unselected page | takeover only reaches pages the sweep re-derives | a `'leased'` expired row whose page is **no longer stale** is still reclaimed | Key the statement on the turn's selected `(page_id, base_page_version)` → both rows above still pass, and the row stays wedged forever |
| PR-D precondition 5 | vacuous green | `space_refresh_dependencies_ready` returns `false` when a dependency's root is deleted or non-`active` **and** the table is non-empty | Run the assertion against an empty table — it passes, which is the failure S0-137 makes mandatory to guard |

### 8.3 G8 — `m6_overview_identity_survives_rebinding`

| Row | Weaken | Test asserts | Deliberate break |
|---|---|---|---|
| G8.5 | duplicate subscription row | `idx_m6_overview_sub_scope_active` / `idx_m6_overview_sub_page_active` (both `WHERE state='active'`, `overview_subscriptions.rs:8`) reject the second | Drop the partial predicate |
| G8.7a | merge loser stays attached when a survivor is determinate | loser's `state` is `detached` | Detach nobody |
| G8.7b | any participant stays attached when there is **no** survivor | every participant is `detached` (S0-163, `…overview-matrix.md:477`) | Treat no-survivor as "no merge" and detach nobody — G8.7a still passes, which is why the row is split |
| G8.8 | resolve a space overview by title lookup | the locator is `has_active_overview_subscription` (`signals.rs:632`) and no `WHERE title =` appears in `overview_lifecycle.rs` | Add a title lookup fallback |
| S0-75 | a split stages a proposal | zero rows written on a split | Stage one |
| vacuity | an empty-fixture detach passes | the fixture **constructs** subscription rows first (§5.3.2) | Run the detach gate on an empty table |

### 8.4 Structural gates (§7)

| Gate | RED mutation |
|---|---|
| `maintenance_writes_only_m6_owned_tables` | `UPDATE pages …` in `relevance_sweep.rs` |
| `maintenance_lane_names_no_llm_type` | `use crate::llm_provider::LlmProvider;` in `maintenance.rs` |
| `maintenance_never_writes_genesis_enabled` | write `genesis_enabled = 1` after stage C commits |
| `maintenance_adds_no_migration` | add migration 111 |
| `user_visible_tables_are_byte_identical_after_n_turns` | call `try_update_page_content` from `refresh_sweep.rs` |
| `m6_platform_routing_is_its_own_filter` (extends `drift_guard`) | Fold M6's corpus/fixture/bench paths into `m5-platform` instead of the new filter. The tooth asserts `m5_platform_inputs` is still exactly the seven paths of `drift_guard.rs:2696`–`:2702` **and** that the M6 set routes to a step whose `run` names `--test m6_relevance_bench`, so the fold fails on both halves (§5.5.4) |

### 8.5 Liveness — the family §4.4 says the gates would otherwise miss

Every gate above is shaped *"the shadow does not write X"*, and a lane that
does nothing at all satisfies all of them. These four assert the complement:
progress is made, and no space starves. They are the only gates in this spec
whose RED mutation is a **stall** rather than a write.

| Gate | Asserts | Deliberate break that must fail it |
|---|---|---|
| `a_full_active_queue_still_drains` | Fill the refresh job table to its active cap, then run N turns: the leased count strictly decreases and reaches zero | Move the intake step (priority 4) ahead of the drain steps (2–3) — the exact §4.4 inversion. Every safety gate still passes |
| `a_refused_turn_does_not_read_as_work` | `MaintenanceTurn::did_work()` is `false` for `LeaseHeld` and `RefusedActiveCap` | `did_work()` as `!matches!(self, Idle)` — the shipped genesis definition at `shadow.rs:91`–`:93` |
| `a_refused_space_does_not_starve_its_siblings` | With space A wedged at its cap and space B holding ready work, B's work completes within `spaces.len()` turns | Advance `rotation` only on the fall-through path — the shipped genesis behavior at `shadow.rs:184` |
| `every_priority_is_reachable_from_a_saturated_state` | From a state where each of priorities 2–6 has work waiting, N turns execute at least one of each | Any early return that skips a lower priority unconditionally |

The first gate is the load-bearing one. Write it as a **property over a
sequence of turns**, not a single-turn assertion — a single turn cannot
distinguish "did one unit of work" from "will do exactly one unit of work
forever", and that distinction is the whole bug.

### 8.6 The relevance benchmark gate

Distinct from §8.1's unit gates: §8.1 asserts the caps against a fixture,
§8.6 asserts the budgets against the frozen corpus of §5.5. Both are needed —
a cap that holds on eight rows and breaks on a 5,000-degree hub passes §8.1.

**Ruling: the limits split by determinism, and only the deterministic half is
a required check.** A count is a property of the implementation; a millisecond
is a property of the machine. Hosted GitHub runners are shared, noisy hardware,
and this repo already carries `docs/ci-flake-policy.md` because an intermittent
required check costs real time. A wall-clock gate that gets rerun until it
passes is not a gate — it is a delay with a green tick. The split below is
therefore deliberate and load-bearing, not an omission.

The split costs nothing in coverage, because the counts are the limits that
actually encode the algorithmic contract. The regression this benchmark exists
to catch is a per-pair query loop replacing the set-based read; that breaks
`R-BENCH-Q` on the first extra statement, long before it costs 50ms on any
machine. The wall clock is the weaker instrument of the two.

**CI-gated — deterministic, hardware-independent:**

| Gate | Bound | Break |
|---|---|---|
| `R-BENCH-Q` / `R-BENCH-ROWS` | ≤ 4 queries / ≤ 512 materialized rows, **always** | Split the adjacency read in two |
| corpus identity | the bench **refuses to run** when the manifest digest does not match the checked-in fixture | Report the number anyway — an incomparable result cited as a gate result |

**Measured and recorded, never a pass condition — L7 local:**

| Instrument | Recorded | Why not gated |
|---|---|---|
| `R-BENCH-MAX` | max evaluation ms, warm cache, one at a time, no concurrent refinery turn (S0-98, `…relevance-contract.md:876`) | Wall clock on shared runners. Recorded with the corpus digest and the machine, as PR-B3 does for its §7.3 numbers |
| `R-BENCH-HUB` | the 5,000-degree hub's figures under all of the above | Same, plus it needs the frozen corpus at full size |
| `R-BENCH-P50` / `R-BENCH-P99` | reported percentiles | S0-98 already forbids promoting these to a pass condition; the ruling above extends the same treatment to the max |

The NVISIT instrument additionally needs the
`LIBSQLITE3_FLAGS=SQLITE_ENABLE_STMT_SCANSTATUS` build. Every ungated
instrument is a **declared** deferral in the registry per §8.7, not a silent
one, and the local receipt is the artifact that discharges it.

Two teeth keep the split honest rather than convenient:

- The recorded numbers ship with the corpus manifest digest and the machine
  identity. A timing without its corpus identity is not evidence, and §5.5's
  refuse-on-mismatch already makes an unidentified run impossible.
- **A benchmark failure stops for a contract amendment and may not be answered
  by tuning a constant** (S0-99, `:883`) — restate this in the bench's own
  module doc. It binds the ungated half too: "we measured 80ms" is an amendment
  question, not a footnote. A limit nobody gates is the single most tempting
  constant to quietly relax.

### 8.7 Catalog registration

`catalog_test.rs` parses only `G3.*`/`G5.*` today (`:126`) and its `REGISTRY`
is `[(&str, Coverage); 14]` (`:41`) with a deferral budget pinned at exactly 1
(`:228`). PR-C must:

1. Widen the parser at `:126` to `G6.`/`G7.`/`G8.` and widen `REGISTRY`
   accordingly.
2. Bind the rows §8.1–§8.3 cover to `Coverage::Gate(<test name>)`.
3. Declare the rest — G6.11d and G6.14 (bench-only), G6.1/G6.2/G6.9 (attachment,
   PR-E), G7.1/G7.4/G7.5 (publication, PR-E), G8.1–G8.4/G8.6/G8.9–G8.11
   (rebinding and rename, PR-D), and the three frontier-policy writers of §5.4 —
   as `Coverage::Deferred` with an owner.
   **Including §8.6's ungated instruments**: `R-BENCH-MAX`, `R-BENCH-HUB`, the
   two percentiles, and NVISIT are `Coverage::Deferred("L7 local receipt")`,
   not absent rows. That is what makes the determinism split auditable — a
   reader can see which limits are enforced by CI and which by a receipt,
   without reading the workflow file.
4. Update the pinned deferral count at `:228` in the same commit.

Step 4 is the one that gets forgotten; `deferrals_are_declared_and_bounded`
(`:220`) is what catches it, which is the mechanism working as designed.

**Catalog row G3.4** (`catalog_test.rs:53`) is currently deferred with the
reason *"coverage is written by publication, and PR-B's dry run publishes
nothing — PR-C"*. **That attribution is wrong and PR-C must re-point it to
PR-E.** PR-C publishes nothing either; group coverage is written by machine E's
`F4` inside the genesis finalize transaction, which lands with publication. A
deferral naming the wrong owner is worse than one naming none, because it reads
as covered by the next rung. Fixing the string is a one-line edit in the same
commit that widens the registry.

### 8.8 Exit matrix

What a crash at each point leaves, and what the next turn does. The lane holds
no state a restart needs: `MaintenanceState` is in-memory rotation and turn
counts, and every durable fact is a row.

| Crash point | Durable residue | Next turn |
|---|---|---|
| Before acquiring a lease | none | re-derives selection, acquires |
| Holding a `relevance` lease, before any write | lease row, live until TTL (300s) | Blocked for ≤ 300s, then C7/C8 reap. Nothing lost |
| Mid-slice, between two pair upserts | Partially-updated `m6_pair_stats` at the **stored decay reference** | The next slice recomputes the same endpoints; the reference did not move, so no drift accumulates. **This is why §5.1.1's pinned reference is a correctness property, not an optimization** |
| After the pair upsert, before the adjacency rewrite | pair rows current, adjacency stale | Adjacency is derived; the next slice rewrites it |
| Holding a `refresh` lease, mid-dry-run | `genesis_refresh_jobs` row in `'leased'` with `lease_expires_at` in the past | **On shipped code, stranded forever**: the retry arm at `refresh_readiness.rs:182`–`:216` requires `state = 'retry'`, which nothing writes. C2's acquire-time takeover statement (§5.2.1) flips it to `'retry'` on the next selection of that page |
| After `replace_refresh_dependencies`, before the verdict | snapshot written | Correct: the snapshot is a statement about live state, not about the verdict |
| Mid-detach | some subscriptions `detached` | Detach is idempotent on `state='active'`; the next pass finishes |
| After counters bump, before `record_soak_receipt` | counters ahead of the receipt | Monotone counters over-count turns at worst, which is safe: the CHECKs are floors (`>= 20`, `>= 1`) and ceilings of zero |
| Between `transition_readiness` and the next turn | epoch bumped (S0-120) | A pre-crash capture cannot win a later CAS — the ABA property the epoch exists for |

Startup, once per process, mirroring S0-5 (`…state-machines.md:127`) scoped to
PR-C's phases: reap expired `relevance`/`refresh` leases via
`reap_expired_m6_phases` (`leases.rs:182`, now covering four phases), sweep
expired `genesis_refresh_jobs` rows to `'retry'` — **a separate table the
`m6_leases` reap does not reach**, so widening `LeasePhase::ALL` does not cover
it (§13.1) — and bump `maintenance_daemon_starts`.

The startup sweep is defence in depth, not the reclaimer: `recovery::scan` runs
once per process (`shadow.rs:149`–`:152`, `recovery.rs:60`), and a turn that
dies mid-dry-run does not restart the daemon. The reclaimer that matters is the
acquire-time takeover in §5.2.1. That bump is what makes the soak's
`daemon_starts >= 1` CHECK (`refresh_readiness.rs:106`) satisfiable — S0-126's
rationale (`…readiness-cutover.md:337`) is that every crash-recovery edge runs
only on startup, so a soak with no restart has never observed the recovery half.

---

## 9. Hermetic test plan

All tests are L4 (in-process, no GPU, no network), reusing
`m6/genesis_test_support.rs` as the fixture seam.

| Module | What it drives |
|---|---|
| `m6/relevance_sweep_test.rs` | The eight-mutation oracle sequence + interleaving; the four caps; the hub weighting; the decay-reference pin |
| `m6/refresh_sweep_test.rs` | Lease exclusion and retry claim; the three expiry-takeover rows (§8.2) including the unselected-page case; the three rejection reasons; the dependency snapshot and its non-vacuous anti-join; the stage-C transitions and the soak receipt with injected window values |
| `m6/overview_lifecycle_test.rs` | Detach with a determinate survivor and with none; the install-scope registration; split-writes-nothing; the constructed-fixture vacuity guard |
| `m6/maintenance_test.rs` | Priority ordering and per-turn bounds; the exit matrix rows of §8.8; the four structural gates of §8.4; **the four liveness gates of §8.5** |
| `crates/wenlan-core/tests/m6_relevance_bench.rs` | The frozen-corpus round trip and refuse-on-mismatch; §8.6's five budget rows |
| `crates/wenlan-core/src/db.rs` unit test | `m6_maintenance_shadow_enabled_value` over `1|true|yes|0|no|""|None` |
| bench (not CI) | G6.11d wall clock, G6.14 NVISIT, under `LIBSQLITE3_FLAGS=SQLITE_ENABLE_STMT_SCANSTATUS` |

The soak-receipt tests inject `SoakEvidence` (`refresh_readiness.rs:425`)
directly. There is no way to wait 72 hours and no reason to try; the CHECKs at
`:104`–`:109` are the contract and injection is how they get exercised in both
directions.

---

## 10. Behavior-unchanged proof

With the flag off — the default — PR-C is inert by construction: no task is
spawned, so no code runs. The gate is a test asserting that with
`WENLAN_ENABLE_M6_MAINTENANCE_SHADOW` unset, no `relevance` or `refresh` lease
row is ever created and every counter stays at its prior value. RED mutation:
spawn the lane unconditionally, as the genesis lane does today.

With the flag on, §7's three zero-mutation proofs are the behavior-unchanged
argument for user-visible state, and §7.3 covers the publication flag.

---

## 11. Open-question ledger

Thirteen adjudicated, three flagged.

### 11.1 Adjudicated

**Q1 — What pins the decay clock?** *Adjudicated: a per-space monotone
`m6_counters` row.* Evidence: the decay formula is a function of
`provenance_roots.created_at` and an evaluation time
(`…relevance-contract.md:241`–`:246`); `space_graph_state` carries no timestamp
(`db.rs:10575`–`:10581`); `m6_counters` is per-space, monotone, and undeletable
(`remaining_substrate.rs:147`–`:206`), and monotone is the *right* constraint
because a backwards reference would increase a decayed weight. Full argument
§5.1.1.

**Q2 — Does the shadow write `m6_refresh_dependencies`?** *Adjudicated: yes.*
Otherwise `space_refresh_dependencies_ready` (`refresh_readiness.rs:261`)
returns `true` over an empty table and PR-D precondition 5 is vacuous — the
failure the catalog makes mandatory to guard (S0-137, `…mutation-catalog.md:227`). The
write is to an M6-owned table and is semantically the same value a finalizer
would write. §5.2.3.

**Q3 — Which frontier-policy writers does PR-C wire?** *Adjudicated: none of
the three remaining.* The read side is already complete (`frontier.rs:115`,
`:119`); F9 has no automatic trigger by construction (S0-52 forbids a default
reason, P9 requires an explicit human/policy one); F7's remaining leg has no
machine-A transition into it. All three ship as declared deferrals. §5.4.

**Q4 — How does the overview lifecycle avoid starving its own signals?**
*Adjudicated: wire detach and install-scope registration only.* A
`space`/`community` subscription suppresses signals 3/4 (`signals.rs:71`) and
requires a page only PR-E mints, so creating one in shadow would starve genesis
permanently while looking like a correct gate. §5.3.

**Q5 — Does PR-C bump `m6_mutation_count`?** *Adjudicated: no.* S0-128's
counter answers "may the old writer be resumed"
(`…readiness-cutover.md:365`); an `m6_pair_stats` write does not make that
unsafe, and bumping would silently retire D14's reverse-ledger hatch on turn
one. The artifact says "any M6 write" and this narrows it to user-visible
writes — flagged as a wording amendment in §13.4 rather than a silent
reinterpretation.

**Q6 — Is `refinement_queue` user-visible?** *Adjudicated: yes for PR-C's
purposes.* It is not on the task's enumerated table list, so the mechanical
gate would permit it, but a row staged `awaiting_review` is what `/curate`
shows a human. PR-C does not write it; the merge proposal (S0-77,
`…overview-matrix.md:230`) goes to PR-E. Recorded here because the enumerated
list and the intent diverge at exactly this table, and the next reader deserves
to know the divergence was seen.

**Q7 — Sibling lane or same turn?** *Adjudicated: sibling.* On independent
failure (`runtime.rs:316`–`:319`) and cadence (§4.2), not on mutex contention —
the single connection mutex is contended identically either way. §4.1.

**Q13 — Does the M6 benchmark corpus extend `m5_bench_corpus.rs` or ship as a
sibling?** *Adjudicated: sibling `m6_bench_corpus.rs`, reusing the freeze
machinery.* M5's public surface is page-shaped end to end (`:34`, `:44`,
`:268`, `:334`) with no graph layer, and `canonical_manifest_digest`'s
signature is specific to M5's three inputs (`:302`–`:306`), so extending it
would change M5's digest domain and invalidate its checked-in fixture. The
generic half — `SplitMix64`, `sha256_hex`, `validate_sha256_hex`,
`DigestWriter` — is reused by lifting it to `pub(crate)`. §5.5.3.

**Q14 — How are backpressure and drain ordered?** *Adjudicated: drain before
intake, and a refusal is never work.* Every finishing step precedes every
starting step; `did_work()` is false for refusals; rotation advances on
refusal. Derived from the shipped livelock verified link-by-link at §4.4, and
gated by §8.5 rather than left as a convention.

**Q8 — Is a stage-C soak receipt sufficient for stage D?** *Adjudicated: no,
and the spec says so.* S0-126 component 1 requires zero mutations from callers
outside the D12 manifest (`…readiness-cutover.md:319`); the manifest fence is
G9's and lands in PR-D. PR-C's counter proves only the M6 half. §5.2.4.

**Q10 — What reclaims an expired `'leased'` refresh job?** *Adjudicated:
nothing on shipped code, and C2 adds an explicit acquire-time takeover
statement.* Stated without hedging because it was verified twice, once by the
author and once independently: `lease_expires_at` is a **write-only column** —
six occurrences in `refresh_readiness.rs` (DDL `:28`, struct field `:139`,
INSERT `:158`/`:170`, UPDATE `:191`/`:209`), none in a `WHERE` clause — and
`state = 'retry'` is **never written** — three occurrences (`:26` CHECK, `:39`
index predicate, `:199` the retry arm's own guard). The state the retry arm
waits for has no producer; the column that would detect expiry has no reader; a
dead worker's job is stranded permanently. The fix and its three design points
are §5.2.1; the gate rows are §8.2. This entry moved from flagged to
adjudicated once the shipped M5 precedent settled the mechanism.

**Q15 — Does the M6 corpus extend `m5-platform`'s CI filter or get its own?**
*Adjudicated: its own `m6-platform` filter, with its own exact-set tooth and CI
step.* The deciding fact is not tidiness but that `drift_guard.rs:2717`–`:2718`
pins the M5 step's `run` **verbatim**, `--test m5_bench` included. One filter
cannot run two bench targets without unpinning the command the tooth exists to
pin, so the fold is structurally unavailable rather than merely ugly. Two
constraints follow: the fixture and the tooth update ship in one commit, and
the M5 tooth stays exactly seven paths — additive only. §5.5.4, gated by the
new §8.4 row whose RED mutation is the fold itself.

**Q16 — Are the relevance benchmark's limits all CI gates?** *Adjudicated: no
— the counts gate, the wall clock is measured.* `R-BENCH-Q` ≤ 4 and
`R-BENCH-ROWS` ≤ 512 are properties of the implementation and become required
checks; `R-BENCH-MAX` 50ms and `R-BENCH-HUB` are properties of the machine and
are recorded locally with the corpus digest, per `docs/ci-flake-policy.md`'s
existing cost. Coverage is not reduced: the regression this bench exists to
catch is a per-pair query loop, which breaks the query count first. S0-98 and
S0-99 bind the recorded numbers exactly as hard. §8.6.

### 11.2 Flagged

| # | Question | Blocks | Cheapest resolution |
|---|---|---|---|
| **Q9** | S0-3's lease TTLs are un-re-derived. The binding rule is `TTL > (model call timeout + finalize budget)`; `constants.rs:96`–`:97` already carries this as PR-B's spec §11 Q10 for the genesis TTL, and PR-C adds two more vacuous ones | **PR-E**, not PR-C — the shadow makes no model call, so 300s/900s are vacuously satisfied | Read the configured LLM timeout once, before PR-E1, and re-derive all four |
| **Q11** | The M6 genesis lane spawns **unconditionally** (`runtime.rs:338`) — no flag. Three independent arguments that this is a defect, not a style choice: (a) PR-C's lane is flag-gated per the task's constraint 7, so the two M6 lanes differ in operability; (b) every comparable ambient lane has one (`WENLAN_ENABLE_EDGES_RECONCILE`, `WENLAN_ENABLE_ENTITY_PAGE_RECONCILE`, `WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE`), all default OFF pending measured RSS and foreground-latency ceilings; (c) **strongest** — PR-B's own spec §10.3 defines PR-B rollback as "stop jobs, invalidate leases, retain frontier / coverage / stats … Rollback is therefore a flag flip plus a lease sweep" (`…m6-pr-b-genesis-shadow-spec.md:810`–`:813`). With no flag there is no flip, so **the spec's own rollback procedure is unexecutable** | Nothing in PR-C | `WENLAN_ENABLE_GENESIS_SHADOW`, default OFF, gated at the `tokio::spawn`, documented in `crates/wenlan-core/AGENTS.md` in the same commit (teeth #2 is fail-closed). **Not PR-C's to change**; PR-B3 is a live edit in that file and has been asked for it directly |
| **Q12** | The whole relevance lane rests on a `supported` predicate the relevance contract declares BLOCKED — and that block has been false since the M5 promoter landed. §13.5 | Nothing, once §13.5's amendment is accepted. Flagged because five sections of an artifact still read as blocked | Amend `…relevance-contract.md` §0/F2/S0-85 with the citation in §13.5 |

---

## 12. Slicing and line budget

PR-C as one PR is roughly **2,780 production / 3,220 test lines** — well past
the ~1,400-line ceiling. Reading that ceiling as **production lines per shipped
PR** (PR-B's own three slices each landed ~900–1,000 production against
~1,400 test, per its spec §12), PR-C slices into four.

The split point follows PR-B's stated principle verbatim — *the proof ships
with the code that could break it* — and each slice's proof is named, not
implied.

| Slice | Contents | Prod | Test | The proof that ships with it |
|---|---|---|---|---|
| **C0** | `eval/m6_bench_corpus.rs` (S0-97's eleven constants, seed `0x6D36_0000`, manifest digest, refuse-on-mismatch); `tests/fixtures/m6_bench_corpus.sha256`; `tests/m6_relevance_bench.rs` harness; `pub(crate)` lift of M5's freeze primitives; the `m6-platform` CI filter + its `drift_guard` tooth | ~650 | ~500 | **Corpus reproducibility**: a reviewer regenerates it byte-for-byte and the bench refuses on mismatch. This is the *only* proof C0 can carry, and it is exactly the proof a threshold gate needs before the threshold exists |
| **C1** | `m6/relevance_sweep.rs`; `LeasePhase::Relevance`; the decay-reference counter; the four budget instruments; §8.6's count gates and timing receipt wired to C0's corpus | ~700 | ~900 | The incremental-equals-full oracle **and its negative control**. Both are properties of the incremental maintenance code, and neither is checkable before it exists |
| **C2** | `m6/refresh_sweep.rs`; `LeasePhase::Refresh`; the dependency snapshot; stage-C readiness; the soak receipt; Q10's expiry reclaimer (priority 2) | ~580 | ~820 | The whole-result-rejection gate (G7.2/G7.3) and the **non-vacuous** anti-join. The anti-join gate is meaningless until something populates the table, which is C2's own write |
| **C3** | `m6/overview_lifecycle.rs`; `m6/maintenance.rs` turn driver; the flag + its `AGENTS.md` entry; the `runtime.rs` lane; catalog registry widening | ~850 | ~1,000 | **The zero-mutation, no-LLM, and §8.5 liveness proofs**, plus the behavior-unchanged (flag-off) gate. C3 introduces the loop that could violate all four; before C3 nothing runs on its own, so they would be asserting about code with no driver |

**C0 is a slice, not a chore, on three counts** — and the recommendation is to
land it first:

1. **A gate needs its corpus before its threshold.** §8.6's count limits are
   hard pass/fail (S0-98) and S0-99 forbids answering a failure by tuning a
   constant. Landing C1's thresholds against an ad-hoc fixture and
   retrofitting the corpus afterwards means the first frozen run is also the
   first run anyone could have argued with. The ungated timing half needs the
   corpus just as much — an unreproducible measurement is worth less than an
   unreproducible gate, not more.
2. **Its cost is real and mostly not in the generator.** The S0-97 recipe is
   ~350 lines; the CI-routing work ruled in §5.5.4 is the rest, and it is the
   item most likely to surprise whoever picks this up.
3. **It is independently reviewable.** C0 touches no M6 runtime code at all,
   so its review is "is this corpus reproducible", which is a different and
   much cheaper question than "is this sweep correct".

**Ordering:**

- **C0 first.** Recommended, not forced: C1 could land with a `#[ignore]`d
  bench and bind it later, but §8.6's count limits are *gates*, and a gate that
  arrives after the code it gates has never once failed.
- C1 before C2 only by convenience (they are independent).
- **C3 must be last.** Its `runtime.rs` spawn is what makes C1's and C2's
  writers reachable from a running daemon; landing the lane first would run
  half-built sweeps.
- **The flag and its `AGENTS.md` entry must be in the same commit** —
  `drift_guard` teeth #2 is fail-closed, so a flag read that lands one commit
  ahead of its documentation fails the build.
- **C0's fixture and its `drift_guard` tooth update must be in the same
  commit**, for the same fail-closed reason at `drift_guard.rs:2704`.

Until C3, C1's and C2's modules have no production caller and their tests are
the only drivers — the same shape PR-B's B1/B2 had, and the reason
`#[allow(dead_code)]` at `mod.rs:33`–`:39` exists. C3 removes the attribute
from `frontier_policy`? **No** — §5.4 leaves those three writers unwired, so
`frontier_policy` keeps its attribute. `relevance` and `refresh_readiness` lose
theirs in C1 and C2 respectively.

---

## 13. Where Stage-0 is now stale against shipped code

Five amendments. Each names the artifact, the false sentence, and the code that
falsifies it.

### 13.1 `genesis_refresh_jobs` is a fourth non-registry lease surface

**Artifact 2 §2.2** (`…state-machines.md:99`–`:103`) enumerates exactly three
surfaces that look like the lease registry and are not:
`claim_derivation_jobs`, `CutoverLease`, and `MaintenanceCoordinator`
reservations. Invariant I-6 (`:478`) then says the set of durable rows granting
exclusive execution rights to an automatic phase is exactly `grouping_leases`.

PR-A shipped a fourth: `genesis_refresh_jobs` carries `lease_token` and
`lease_expires_at` (`refresh_readiness.rs:27`–`:28`) under a partial unique
index (`:37`–`:39`).

**Amendment.** It is a **work-item** lease of the `claim_derivation_jobs`
class, keyed `(page_id, base_page_version)` rather than
`(phase, space, input_generation)`, so I-6 is intact — I-6 is about *phase*
leases. Artifact 2 §2.2's table should gain a fourth row saying so. Until it
does, a reviewer reading I-6 literally will read PR-C's refresh lane as a
violation.

The omission is not only editorial. Because the table is outside the phase
registry, it is also outside the phase registry's **reaper**:
`reap_expired_m6_phases` iterates `LeasePhase::ALL` over `m6_leases`
(`leases.rs:182`, `:47`) and never sees `genesis_refresh_jobs`. That is the
structural reason `lease_expires_at` shipped as a write-only column and
`'retry'` shipped with no producer (Q10, §5.2.1): the surface got a lease
lifecycle but not the sweep that every registered phase inherits for free.

**What this predicts.** The defect is not specific to refresh. Any future M6
table that grows its own `lease_token` / `lease_expires_at` columns instead of
taking a row in the phase registry will ship the same way — a lease it can
acquire and no sweep that can reclaim it — and it will pass review, because the
columns look complete in the DDL and the missing half is in a different file.
The check is mechanical and worth running at every subsequent rung: for each
table carrying lease columns, name the statement that reads `lease_expires_at`
in a `WHERE`. If there is none, the lease is one-way. `LeasePhase::ALL`
(`leases.rs:47`) is the cheap answer for anything that can be a phase; a table
that cannot be a phase needs its reclaimer written by hand and gated, as
§5.2.1 does.

### 13.2 S0-11's single-clock rule is already broken by shipped code

**S0-11** (`…state-machines.md:133`): *"Every durable timestamp and every timer
comparison is `unixepoch()` evaluated by SQLite inside the statement itself,
never a Rust-side `now` passed as a parameter."*

`acquire_refresh_lease` takes `now: i64` (`refresh_readiness.rs:143`) and
`lease_expires_at: i64` (`:139`) as caller-supplied parameters, and binds `now`
as `?12` into both `created_at`/`updated_at` and the retry guard
(`:174`, `:200`, `:213`). `m6_refresh_dependencies.created_at` does use
`DEFAULT (unixepoch())` (`:53`), so the divergence is inside one function.

**Amendment.** Shipped code wins. PR-C's new statements use in-statement
`unixepoch()`; PR-C does **not** rewrite `acquire_refresh_lease`'s signature,
because that is PR-A surface with tests against it and the change is not needed
to make PR-C correct. The consequence to record: the refresh lane has two
clocks, and a test that moves stored values (S0-11's testability argument) can
be defeated for the refresh job table by a caller passing a different `now`.
Callers in PR-C pass a value read from `unixepoch()` in the same transaction,
which restores the property behaviorally without a signature change.

### 13.3 Machine F's F7 admits a transition machine A cannot produce

**Machine F, transition F7** (`…state-machines.md:456`): `exclusively_claimed |
surfaced_card → suppressed`, triggered by *"candidate suppressed, or card
dismissed by a human (A17)"*.

**Machine A** has exactly one edge into `suppressed` — A17, `review_required →
suppressed` (`:195`), which is the `surfaced_card` leg. No transition reaches
`suppressed` from `prepared`, `inferencing`, or `validating`, i.e. from
`exclusively_claimed`.

**Amendment.** Either F7's `exclusively_claimed` leg is dead and should be
struck, or machine A is missing an edge. This spec does not decide it —
deciding would mean specifying a suppression trigger no artifact names, which
is the speculative surface §5.4 declines. It is recorded so that
`suppress_frontier_group`'s permanently-uncalled state is a known contradiction
rather than an implementation oversight.

### 13.4 S0-128's "any M6 write" is broader than the predicate it serves

**S0-128** (`…readiness-cutover.md:365`) introduces the per-space monotone
counter *"incremented in the same transaction as any M6 write"*, to answer
"before the first M6 mutation for that space" — the gate on resuming the old
writer.

Taken literally, a write to `m6_pair_stats` would bump it and retire D14's
reverse-ledger escape hatch on the first maintenance turn, even though the old
writer neither reads nor owns that table.

**Amendment.** The counter tracks **user-visible** M6 writes. The sentence
should read "in the same transaction as any M6 write to user-visible state".
PR-C proceeds on the narrower reading (Q5); if the caller prefers the literal
one, PR-C's design changes only in that the counter bumps and every
zero-assertion in §7.1 proof 2 inverts — which would also mean PR-C's first
turn permanently forecloses a D14 rollback path, and that is the reason for the
narrow reading.

### 13.5 The `supported` block is false — the promoter has landed

This is the largest one, and it invalidates rows in two artifacts.

**`docs/plans/2026-08-01-m6-relevance-contract.md`** §0, finding F2, and
decision S0-85 (`:84`) assert that no production writer promotes `supported`,
mark every co-citation input row **BLOCKED**, and call a claim-derivation
promoter *"a hard prerequisite of PR-B's genesis shadow"*.
**`docs/plans/2026-08-01-m6-mutation-catalog.md`** F2 (`:754`–`:758`) repeats
it: *"no production code writes that value today"*, citing
`crates/wenlan-core/src/db/claim_identity.rs:511`–`:512`.

Both are now false. `finalize_page_support`
(`crates/wenlan-core/src/db/claim_derivation.rs:4482`) maps
`SupportOutcome::Supported => ("supported", None, Some(now))` at `:4559` and
binds it as `?3` into
`INSERT INTO page_truth_state … ON CONFLICT(page_id) DO UPDATE SET …
support_status = ?3` at `:4582`–`:4588`, under the eligibility-generation CAS
at `:4545`.

**Rows that go stale with it:**

| Where | Stale claim |
|---|---|
| relevance contract §1.2 | "provisional contributes zero" / "pairs come from supported claim revisions" / "the candidate is the current supported version" — all read as unreachable |
| relevance contract §4 (`:346`) | the candidate predicate is marked **BLOCKED** |
| relevance contract §6 field 2 | same |
| relevance contract §13 | G6's "vacuously true" note |
| mutation catalog §14 F2 (`:754`) | "sixteen of the 204 cases cannot go red until lane 1 lands" — the count is now wrong, and `lane 1` is 16 in the status table at `:669` |
| mutation catalog §1.2 (`:237`, `:240`) | "no production writer promotes `supported`" and "a repository-wide search for the literal `'supported'` at `1c903bec` finds no production hit" — both false at `claim_derivation.rs:4559` |

**Effect on PR-C:** none blocking. The relevance lane's inputs are live, so
G6's co-citation rows are executable rather than vacuous — which is *better*
than the artifacts predict, and is the reason §8.1's oracle can ship with C1
instead of waiting. **This extends PR-B's spec §11 Q1**, which found the
mutation catalog's lane counts stale by 16 rows; the finding here is the cause
of that staleness, not a second instance of it.
