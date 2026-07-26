# M4 Stage 0 — Gate PASS/FAIL criteria (authored before code)

**Rung:** M4 (persisted communities). Companion to
`2026-07-25-m4-communities-mechanics.md`. Gate 1.1–1.3, Gate 1.5, and Gate 2.1–2.3 run
and PASS in the gate stage **before any persistence/implementation code** (D1). The
whole-job portion of Gate 1.4 and Gate 2.4 are unmeasurable until the real publication path
exists, so their unchanged acceptance test is committed RED before all PR-1 persistence
implementation and must turn GREEN on that real path before PR-1 may merge or either gate
may be declared fully PASS (sequencing rulings below). This doc also fixes
the numeric bars the PR-1/PR-2 acceptance tests are judged against (§4), so no measurement
grades itself after the fact (§3: *"the projection must be written down before any
benchmark"*; the M4 §7 row: the two executable gates "written before code"). A gate may be
**MET** or **re-argued to the user with evidence**; it may never be silently loosened to get
green (floor: *"never weaken an existing check to get green"*).

### Sequencing rulings (2026-07-25 and 2026-07-26; Fable sign-off; no bar changes)

1. Gate 1.1–1.3, Gate 1.5, and Gate 2.1–2.3 run and must PASS in the gate stage before any
   persistence/implementation code. At gate stage, 2.1–2.3 are measured on the
   grouping assignment output — the exact artifact PR-1 will persist as
   `community_members`. References to `community_members` / durable ids in 2.1–2.3 read
   accordingly. This measures the algorithm under test; it is not a fake publication path.
2. The pre-persistence part of Gate 1.4 proves the indexed bounded SELECT and that projection
   and partition begin after its connection guard is released. Lease acquisition, finalize
   generation-CAS, cumulative mutex time, and transaction structure can only be measured on
   the real PR-1 job, so 1.4 remains PROVISIONAL until that path exists.
3. Gate 2.4 is likewise unmeasurable before the publication path exists. Its measurement
   moves to PR-1: the **first commit after the gate-stage commits on the PR-1 branch** is a
   RED integration acceptance test asserting both the whole-job 1.4 bar and the 2.4 bar
   verbatim against the real lease, generation-CAS grouping job, and real published
   snapshot. No in-memory or disposable SQLite substitute counts as evidence.
4. That RED test commit precedes all persistence-implementation commits. Persistence
   commits may then land while the test is RED. PR-1 may not merge, Gate 2 may not be
   declared PASS, Gate 1.4 may not be declared MET, and no completion claim may be made
   until the real-path 1.4 and 2.4 assertions are GREEN at the unchanged bars in a
   full-suite run.
5. Gate 2.5 is recorded at gate stage as
   `PROVISIONAL PASS — 2.1–2.3 MET, 2.4 pending PR-1`; it becomes final only when 2.4 turns
   GREEN.
6. Gate 1 is recorded at gate stage as
   `PROVISIONAL PASS — 1.1–1.3 and 1.5 MET; 1.4 SELECT preflight MET, real job pending PR-1`.
7. Nothing in these rulings changes Gate 1.4 or Gate 2.4's bar or FAIL semantics.

**Gates + implementation WAIT on M3g** (Q-A ruling): Gate 1 and Gate 2 measure over a
**real grounded subgraph** produced by M3g. Stage-0 authoring is data-independent and runs
in parallel, but no gate number is collected until M3g has drained a grounded substrate.

Scale reference for all corpora: §6.5 illustrative target = **100k memories / 5k pages on a
laptop**; the M2 acceptance gate already runs at this size, and M4's gates run against it.
Cautionary datum (carried from M3g / entity-page reconcile): a sweep that holds the single
DB connection's mutex for a full pass measured **18.88s at 10k entities on an M2 Pro**
(`crates/wenlan-core/AGENTS.md`), queueing every foreground request behind it. M4's grouping
job MUST NOT repeat that — its mutex hold is bounded (Gate 1.4), partition compute runs
outside the mutex (mechanics §5.3, §10.1).

---

## 0. Provisional constants (frozen here; tuned against the gates, never re-chosen after measurement)

These are the mechanics-doc knobs, fixed NOW so the projection the benchmark runs on is
written down. The PR tunes each to MEET a gate bar without weakening any gate; a change to
any value re-runs the affected gate.

| Constant | Provisional value | Governs |
|---|---|---|
| `W_relates` | `1.0` | base per-edge-type weight (§3.2) |
| `PARALLEL_EDGE_CAP` | `3.0 × W_relates` | folded-pair aggregate weight cap (§3.3) |
| `HUB_DEGREE_CAP` | `50` | grounded degree above which incident weights soft-down-weight (§3.4) |
| `M4_MIN_PARTICIPANTS` | `10` | viability floor — a space below this publishes nothing (§2.3) |
| `FULL_REPARTITION_FRACTION` | `0.25` | dirty-node fraction above which a full re-partition replaces warm-start (§4.3) |
| `MAX_INCREMENTAL_FRONTIER_FRACTION` | `0.25` | expanded dirty + old/new one-hop frontier above which a full re-partition replaces warm-start (§4.3) |
| `LEIDEN_RESOLUTION` (γ) | `1.0` | standard modularity resolution |
| `LEIDEN_SEED` | fixed constant (e.g. `0x4D34`) | RNG determinism (§4.2) |
| `T_hi` | `0.50` | page→community assign threshold, normalized score (§8.2) |
| `T_lo` | `0.30` | page→community drop threshold (`T_hi > T_lo`) (§8.2) |
| `MIN_PAGE_ENTITIES` | `2` | below this a page routes by embedding fallback (§8.3) |

---

## Gate 1 — Leiden spike + on-device benchmark on the exact §3 projection

Runs the partitioner on the **§3-authored projection** (mechanics §3) over a **real M3g
grounded subgraph** at §6.5 scale. Four measured sub-gates; all four must PASS. **On any
FAIL → STOP-fork Q-B** (adopt the label-propagation fallback under the same D6/D7 contract —
a smaller rung — vs re-scope; the USER rules, the agent never silently downgrades).

**Precondition — partitioner seedability (spike gate, blocks 2.1).** Before any timing
number, the spike verifies the chosen crate exposes a **fixed RNG seed AND a deterministic
node-iteration order** — the two properties Gate 2.1's HARD-ZERO determinism depends on. If
the crate is fundamentally unseedable (internal parallelism / unseeded RNG that no public knob
fixes), Gate 2.1 is unmeetable by construction; that is **not an in-house fix** — it routes to
**Q-B** (crate swap / label-prop fallback; §5). Confirm this first, so a determinism failure
observed later (2.1) is correctly read as a crate-capability limit routing to Q-B, not a
wiring bug to patch.

**Bar / hard-fail band (applies to every sub-gate).** Each sub-gate states a **bar** and a
**hard-fail**. At or better than the bar = MET. At or past the hard-fail = STOP (Q-B). The
band **between** bar and hard-fail is **re-argue-to-the-user-with-evidence** territory (per
the header floor) — explicitly **never** a silent auto-pass.

### 1.0 Benchmark corpus (stated, reproducible)

The 100k-memory / 5k-page M2 scale corpus, with a **grounded `relates` subgraph of stated
minimum size** seeded so the benchmark is reproducible regardless of how far M3g's live
drain has progressed: **≥ 5 000 active grounded `relates` edges over ≥ 2 000 participating
entities, spread across ≥ 3 spaces** (seeded directly via bulk insert, mirroring the M2
`m81_bulk_insert_trigger_benchmark` precedent, `db.rs` scale benches — grounded edges
minted with `grounded=1` + a real `root_id`). The projection (§3) is applied to this exact
subgraph. If M3g's live drain has produced MORE than this, the bench also reports on the
live subgraph as a secondary datum, but the fixed seeded subgraph is the graded one.

### 1.1 Partition runtime (CEILING)

- **Full-partition ceiling: a full per-space Leiden partition of the graded subgraph
  completes in ≤ 10 s per space (p95 over ≥ 5 runs), OUTSIDE any DB transaction.** A full
  partition happens on first publication / version change / dirty-fraction breach (§4.3).
- **Hard fail: any full partition > 30 s** — the projection or the crate is too slow at
  scale; escalate to Q-B rather than accept.

### 1.2 Incremental warm-start (FLOOR — a full re-partition per cycle is a FAIL)

Invariant §3: *"incrementality is required, not aspirational."*

- **Bar: an incremental warm-start around a dirty frontier of ≤ 1% of a space's nodes costs
  ≤ 10% of that space's full-partition time (p95 over ≥ 20 single-edge-addition cycles),**
  and scales with the frontier size, not the graph size.
- **Observable scaling oracle:** at a fixed one-node frontier, growing a sparse graph from
  2,048 to 32,768 nodes increases incremental p95 by no more than `3×`; on the 32,768-node
  graph, growing the dirty set `8×` (8→64 nodes) increases p95 by no more than `12×`.
  Graph-wide state initialization is allowed only on first publication, process restart,
  algorithm/projection-version change, or an explicit full-repartition threshold.
- **Expanded-frontier cap:** if dirty nodes plus their old/new one-hop neighbors exceed
  `MAX_INCREMENTAL_FRONTIER_FRACTION`, the call routes to full repartition. The cap does not
  silently truncate optimization and the resulting full/incremental output remains subject
  to Gate 1.5.
- **FAIL: incremental cost ≥ 50% of full-partition cost** for a ≤ 1% frontier (warm-start is
  not actually local — effectively a full re-partition per cycle). This is the §3
  "full re-partition per cycle is a fail" made numeric → Q-B.

This gate bounds the **partition optimizer** reached by the production composer. It does
not claim that PR-1's exact snapshot I/O or post-hoc attachment recomputation is independent
of space size: both deliberately read/write the frozen whole-space snapshot, and centroid
changes can affect every isolated attachment. Those costs remain governed by Gate 1.3
(partition RSS only), Gate 1.4 (mutex/foreground), and Gate 2.4 (correction-cycle
latency). The composer reports elapsed time and loaded/written row counts, but its
whole-snapshot peak RSS is **not measured by PR-1**; Gate 1.3 must not be cited as
composer-I/O memory evidence. The composer must nevertheless report its executed branch
(`CoreReused`, `Incremental`, or a reasoned `Full`) so a whole-space partition cannot hide
inside those permitted snapshot costs.

### 1.3 Peak memory (CEILING)

- **Bar: peak additional RSS for the first full partition in a fresh child process at the
  graded scale ≤ 256 MB.** The no-partition child baseline constructs the identical graph;
  the partition child runs the first Leiden operation, and OS peak RSS is differenced.
  (A few
  thousand nodes / edges is a small graph; this leaves generous headroom and catches an
  accidental all-pairs / dense-matrix representation.)
- **Scope:** partition implementation only. This child-process oracle excludes database
  snapshot loading, attachment recomputation, and publication writes in the production
  composer.
- **Hard fail: > 1 GB** — a representation blow-up; fix or Q-B.

### 1.4 Mutex-hold / foreground-latency (CEILING — the 18.88s guard)

The job holds the single DB connection mutex only for the bounded subgraph SELECT, the lease
acquire, and the finalize CAS (mechanics §5.3, §10.1) — never during projection/partition.

- **Bar: per-firing cumulative DB-mutex hold ≤ 500 ms (p95 over ≥ 20 firings)** at graded
  scale. ~38× under the 18.88s datum.
- **Hard fail: any single firing > 2 s** of cumulative mutex hold (an unbounded scan or a
  compute call leaked under the mutex).
- **Structural assertion (primary guard, timing-independent):** a test/inspection confirms
  **no partition, projection, rebinding, or embedding call occurs inside any `BEGIN…COMMIT`**
  in the job path (§6.3) — the same worst-case guard M3g's Gate 3 uses. The ms ceiling is the
  quantitative backup.
- **Gate-stage status:** the real-path lease/finalize pieces do not exist before persistence.
  The indexed SELECT timing and outside-lock compute are a PROVISIONAL preflight only; §1.4
  becomes MET only when the first PR-1 RED integration test turns GREEN on the actual job.

### 1.5 Grouping quality vs label-prop (FLOOR)

- **Bar: Leiden modularity ≥ the label-prop baseline's modularity** on the SAME projection
  (Leiden must be no worse), AND **Leiden produces zero badly-disconnected communities**
  (a community whose induced subgraph is disconnected — the failure mode Leiden fixes over
  Louvain/label-prop). Measured on the graded subgraph.
- **FAIL: Leiden modularity < label-prop modularity**, or any disconnected community. A
  modularity loss means the projection is mis-weighted (§3) — question the projection (the
  three-fix / systematic-debugging rule), do not accept.

### 1.6 PASS / FAIL (Gate 1)

- **Gate-stage provisional PASS:** 1.1–1.3 and 1.5 MET, plus the 1.4 indexed-SELECT
  preflight MET; record exactly
  `PROVISIONAL PASS — 1.1–1.3 and 1.5 MET; 1.4 SELECT preflight MET, real job pending PR-1`.
- **Final PASS:** 1.1 ≤ 10 s full / 1.2 warm-start ≤ 10% / 1.3 ≤ 256 MB / 1.4 ≤ 500 ms p95 mutex +
  no-compute-in-txn / 1.5 modularity ≥ label-prop with zero disconnected communities — all
  MET, receipts attached.
- **FAIL:** any sub-gate hard-fail, or a floor missed → **surface as Q-B** (label-prop
  fallback vs re-scope; user rules). Never worked around.

---

## Gate 2 — Community churn rate + correction latency (measured independently)

The two dials of invariant #12 (*"stability and truthfulness get separate, measured dials"*).
These gate the **routing/rebinding design** (mechanics §4.4, §6, §8), not just the partitioner.

### 2.1 Determinism churn (HARD-ZERO)

- **Bar: two runs of the grouping job over an IDENTICAL frozen grounded subgraph produce
  byte-identical `community_members` — exactly 0 nodes change community.** With the fixed
  seed + deterministic node order (§4.2), any nonzero churn on a frozen subgraph is a bug,
  not RNG.
- **FAIL: any node changes community across two frozen-subgraph runs.** Fix (find the
  nondeterminism), never accept.
- **"Stable corpus" definition (authored, mechanics §0.2 call 1):** the frozen subgraph is
  taken with **M3g promotion OFF** (`WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE=0`) so the
  grounded set does not move under the measurement. M3g's designed gradual drain is a
  substrate-growth process; isolating it is what makes silent-rebinding churn measurable.

### 2.2 Incremental churn (CEILING — invariant #16 silent-rebinding stability)

Add a single grounded edge (bump one space's generation), re-run the job, measure the
fraction of that space's nodes whose **durable `community_id`** changed.

- **Bar: mean incremental churn ≤ 2% of the space's nodes, p95 ≤ 10%, over ≥ 50 single-edge
  additions** on the graded subgraph. Because durable ids come from max-overlap rebinding
  (§6), a local re-optimization that barely changes member sets keeps ids stable → low churn.
- **FAIL: mean > 2% or p95 > 10%** — the rebinding is thrashing ids (either warm-start is not
  local, §4.3, or the overlap accumulator is mis-assigning, §6). This is what distinguishes
  a stable persisted community from today's overwrite-every-run label-prop.
- **Why 2% / 10% (basis, provisional-pending-baseline).** The natural baseline is today's
  label-prop, which **overwrites `entities.community_id` every run**: a single-edge change can
  reshuffle an unbounded fraction of a space's assignments because it carries **no** id
  persistence at all. A durable persisted community that keeps **≥ 98% of a space's ids stable
  (mean) across a single grounded-edge addition, ≥ 90% at p95** is the qualitative line
  separating "durable" from "overwrite-transient." 2% is a first-pass ceiling, not a measured
  optimum — it is **re-argued against the measured label-prop churn baseline** once M3g's
  substrate exists (same discipline as M3g's "why 80%" tunable: a value fixed now, revisited
  when the measurement that calibrates it lands).

### 2.3 Poison robustness (CEILING — the M3g-imperfect-substrate design input)

M3g's entailment judge is beatable (declaratively-framed injections); the grounded substrate
may carry a bounded false-grounding fraction. **The fraction below is a conservative
stand-in, not a calibrated rate.** M3g's Gate-1 measures **zero** false-grounding on its own
fixture and the **live** false-grounding rate is currently **unmeasured** — there is no true
rate to calibrate against yet. `f_poison = 5%` is therefore a deliberately-pessimistic stress
fraction chosen to prove the partition degrades gracefully, **not** a claim about how dirty
the substrate actually is. Seed **5% of the grounded edges as adversarial cross-community
bridges** (edges between entities in genuinely different clusters, simulating a false
grounding), then re-measure §2.2.

**Recalibration (tied to M3g Gate-1 hardening output).** When M3g's in-flight Gate-1
hardening bounds the real false-grounding rate — and/or a live measurement lands — reset
`f_poison` to that bound (or a small multiple of it) and re-run 2.3. Until then the 5%
stand-in stands. This mirrors M3g's own tunable discipline (a value fixed now, revisited when
the measurement that would calibrate it exists). See mechanics §0.3.

- **Bar: with 5% poison edges, (a) mean incremental churn ≤ 4% (≤ 2× the clean ceiling), and
  (b) the published community COUNT does not collapse below 70% of the clean-substrate count**
  (the partition does not smear into one giant community). Proves M4 degrades gracefully, not
  catastrophically, under an imperfect grounded substrate.
- **FAIL: churn > 4% mean, or community count < 70% of clean.** A collapse means the hub
  normalization (§3.4) / parallel-edge cap (§3.3) are too weak to absorb a few bad bridges —
  fix the projection bounds, don't accept.

### 2.4 Correction latency (CEILING)

A correction — a new grounded edge, an entity merge, or a retraction of a grounded edge
(each bumps `graph_generation`, mechanics §5.2) — must reach the **published** assignment
quickly.

- **Bar: a correction is reflected in `community_members` within ≤ 1 grouping cycle** (the
  next `Phase::CommunityDetection` firing for that space after the bump) in the common case,
  and **≤ 2 cycles worst case** (a mid-run edge arrival fails the CAS and re-queues, costing
  one extra cycle — mechanics §5.3). Measured as: after the bump, assert the published
  snapshot reflects the corrected edge within the cycle bound.
- **FAIL: > 2 cycles**, or a correction that never propagates (the dirty bit was cleared on a
  stale input — a generation-guard bug).

### 2.5 PASS / FAIL (Gate 2)

- **Gate-stage provisional PASS:** 2.1 determinism-zero / 2.2 churn ≤ 2% mean, ≤ 10% p95 /
  2.3 poison churn ≤ 4% and count ≥ 70% are MET, with receipts attached; record exactly
  `PROVISIONAL PASS — 2.1–2.3 MET, 2.4 pending PR-1`.
- **Final PASS:** the gate-stage provisional PASS plus 2.4 latency ≤ 1 cycle (≤ 2 worst)
  proven GREEN against the real PR-1 publication path.
- **FAIL:** any bar missed. Determinism (2.1) and correction-latency (2.4) failures are bugs
  to fix; churn (2.2/2.3) failures point at the projection/rebinding design (three-fix rule
  → question the design).

---

## 2.6 Reported datum (NON-GATING) — participation fraction

**Not a gate.** A **reported** number the gate stage records alongside Gate 1/2, so the
"agent-only corpus ⇒ zero communities" consequence (report spec-level risk #1) is visible
rather than silent. Per space:

> **participation fraction = (# entities with grounded degree ≥ 1) / (# total entities in the
> space)** — the share of a space that actually enters the grounded partition.

A space below the `M4_MIN_PARTICIPANTS` viability floor (§0 constant; mechanics §2.3) publishes
**no** communities; this datum makes that hold legible — a low fraction *explains* an empty
community set instead of it looking like a bug. It **never** PASS/FAILs a gate; it only
annotates the result.

**Adopted default (controller, user veto open).** The controller adopted the recommendation:
the **viability-floor HOLD** (mechanics §0.2 call 5) **plus** reporting this fraction as a
**non-gating** datum. This is an **adopted default pending the user's nod** — if the user
instead wants participation *gated* (a minimum-coverage bar that can FAIL), that converts this
datum into a gate and is a scope change to re-author here.

---

## 3. Mutation-proof requirements (goal-prompt floor — red-proof-only logs)

Each load-bearing gate assertion is mutation-proven: break the product code, watch the test
fail, restore it. Green in a mutation log is NOT evidence; green = a full-suite run at the
gate.

- **Determinism (2.1):** remove the fixed seed / deterministic node order → frozen-run churn
  goes nonzero → 2.1 fails. Restore.
- **Generation-guard (§4.1):** replace the CAS `WHERE graph_generation = input_generation`
  with an unconditional clear → the mid-run-arrival test publishes on a stale input and the
  space is wrongly un-dirtied → §4.1 test fails. Restore.
- **Phase lease (§4.2):** replace the durable lease with a process mutex → the concurrent-job
  test double-publishes → §4.2 fails. Restore.
- **Rebinding (§6 / 4.3):** replace max-overlap with fresh-id-every-run → §4.3 durable-id
  stability test fails and 2.2 churn explodes. Restore.
- **Hub normalization / parallel cap (2.3):** disable the soft down-weight → poison
  robustness collapses the community count → 2.3 fails. Restore.
- **Frontier locality (1.2):** replace the reusable incremental statistics with per-call
  all-node/all-edge rebuilds → the fixed-frontier multi-size oracle fails. Remove the
  expanded-frontier check → the high-degree-hub cap test fails. Restore.

---

## 4. PR acceptance invariant bars (exact/structural — fixed now)

Numbers the PR-1/PR-2 acceptance tests are judged against, authored here so they can't be
set after the fact. Each is a boolean/exact-count check (pass/fail by construction).

### 4.1 Generation-guard (PR-1, HARD)

An edge arriving mid-run leaves the space **queued, dirty NOT cleared**: after the CAS with a
stale `input_generation`, `space_graph_state.dirty = 1`, `published_generation` unchanged,
and the stale snapshot is **not** written. Exact: the CAS `UPDATE … WHERE graph_generation =
input_generation` matches **0 rows**. Any published-on-stale-input = FAIL.

### 4.2 Phase-lease exclusion (PR-1, HARD)

Two concurrent jobs for the same `(phase='community', space, input_generation)`: exactly one
acquires the lease and publishes; the second declines/supersedes and does **not** double-write
(`published_generation` advances **exactly once**). A double-publish = FAIL.

### 4.3 Rebinding stability (PR-1, HARD)

When membership barely changes (a single-edge addition that does not move a community's
member set materially), the durable `community_id` of the affected community is **unchanged**
(max-overlap keeps the id). A stable-membership run that mints a fresh id = FAIL.

### 4.4 Shadow isolation (PR-1, HARD)

A test proves a durable `community_members` row is minted with `algo_version` +
`projection_version`, AND the label-prop `entities.community_id` (migration 27) is
**byte-unchanged** by the Leiden job (write-only shadow, D3). Any mutation of
`entities.community_id` by the new job = FAIL.

### 4.5 Migration replay-safety + backup/restore (PR-1, HARD)

Migration killed mid-run, rerun, **converges** (fresh-DB and upgraded-DB schemas agree; the
daemon refuses to open a newer-schema DB, §6.9). Pre-migration online backup + integrity
receipt + a restore drill that **actually restores** (close/replace/reopen + a post-restore
read). Any non-convergent replay or a drill that doesn't restore = FAIL.

### 4.6 Per-consumer differential cutover (PR-2, HARD)

For each of the **two daemon-side** consumers of `entities.community_id` — (1) T18 summary
rollups (`load_summary_buckets`, `db.rs:19392`, + `refinery/summary.rs`) and (2) peer-entity
eligibility (`summary_eligible_predicate`, `derived_artifact_state.rs`) — a differential test
shows **durable-backed output EQUALS label-prop-backed output on a stable corpus**, then
diverges **only** where the grounded/rebound structure genuinely differs. (The spec's third
consumer, map-region/overview rollups, is **client-side** — no daemon reader exists in this
tree, so it is deferred to the app / M6 and is **not** part of PR-2's differential surface.)
Cutover is **per-consumer reversible** (flip back to `entities.community_id`) until soak
clears. A consumer whose durable-backed output differs on a case where the structure is
identical = FAIL.

### 4.7 Routing hysteresis (PR-2, HARD)

A page scoring `≥ T_hi` for a community **assigns**; a page scoring in `[T_lo, T_hi)` **holds
its prior community**; a page scoring `< T_lo` **drops**. Exact three-way behavior; any band
mis-transition = FAIL.

### 4.8 Split/merge as proposal, not silent rename (PR-2, HARD — D8, invariant #16)

A split/merge updates membership (silent, structural) but a `display_name`/label change
surfaces as a **review proposal** — a test proves no `display_name` changes without an
accepted proposal. A silent rename = FAIL.

### 4.9 Soak internal consistency at 100k/5k (PR-2, HARD-ZERO)

A reconciliation sweep asserts, over the graded corpus: **every entity with a community
assignment (core participant OR isolated-attached) is in exactly one community per space per
published generation; fully-isolated entities (zero incident edges of any kind) are
legitimately NULL** (not a violation — §3.5, mechanics §0.2 call 4). The
`community_members` PK `(space, node_id)` enforces single membership structurally either way.
Also: the **published snapshot's `published_generation` equals
its `input_generation`**; no orphan `community_members` (community_id with no `communities`
row) and no member of a `retired_at` community. Exact-zero violations. Any = FAIL.

### 4.10 Index acceptance at scale (PR-2, HARD)

`EXPLAIN QUERY PLAN` confirms the grounded-subgraph scan uses
`idx_edges_active_grounded_space_type` (`db.rs:8578-8579`) — **no full-table scan** — and the
`community_members` reads use `idx_community_members_community`; bulk benchmark at graded scale
attached. A full-table scan on the hot path = FAIL.

---

## 5. STOP conditions (surface, do not work around)

- **Gate 1 FAIL** (any sub-gate) → **Q-B**: adopt the label-propagation fallback under the
  same D6/D7 contract (a smaller rung) vs re-scope. The USER rules; the agent never silently
  downgrades the algorithm.
- **A gate's authored PASS/FAIL turns out unmeasurable at §6.5 scale** (e.g. the graded
  grounded subgraph cannot be constructed because M3g has grounded too few edges and seeding
  grounded rows directly would violate an invariant) → STOP, surface to the user; do not lower
  the bar.
- **Determinism churn (2.1) nonzero** → first distinguish the cause. A **wiring bug** (seed
  not actually threaded through, node order not sorted, tie-break not smallest-id) → fix it
  (three-fix rule), never accept. A **crate-capability limit** — the chosen partitioner has no
  public fixed-seed knob or has unseeded internal parallelism (the Gate-1 seedability
  precondition above should have caught this) → **NOT an in-house fix**; route to **Q-B**
  (crate swap / label-prop fallback; the user rules).
- **Modularity < label-prop (1.5)** → a mis-weighted projection; question the projection
  (three-fix rule → question the design), never accept.
- **`community_id` wire exposure would touch a frozen M3 stage-F surface** → **Q-C** (PR-2),
  verified against the merged #390 `deny_unknown_fields` teeth.
- **Three fix attempts on one root cause failed** → question the architecture
  (systematic-debugging).

---

## Summary table

| Gate | Bar | Kind | Corpus | Fail |
|---|---|---|---|---|
| **1.1 full-partition runtime** | ≤ 10 s/space p95 | ceiling | graded grounded subgraph @ 100k/5k | > 30 s → Q-B |
| **1.2 warm-start incremental** | ≤ 10% of full for ≤1% frontier; multi-size locality oracle | floor | single-edge cycles + 2k→32k fixed-frontier | ≥ 50% → Q-B |
| **1.3 peak memory** | ≤ 256 MB | ceiling | first full partition in fresh child @ scale | > 1 GB → Q-B |
| **1.4 mutex hold** | ≤ 500 ms p95; no compute in txn | ceiling | ≥ 20 real job firings @ scale; SELECT-only preflight is provisional | > 2 s single |
| **1.5 quality vs label-prop** | modularity ≥ label-prop, 0 disconnected | floor | graded subgraph | < baseline → question projection |
| **2.1 determinism churn** | exactly 0 on frozen subgraph | hard | frozen (M3g OFF) | any change |
| **2.2 incremental churn** | ≤ 2% mean, ≤ 10% p95 | ceiling | ≥ 50 single-edge additions | > bar |
| **2.3 poison robustness** | churn ≤ 4%, count ≥ 70% clean | ceiling | 5% poison edges (stand-in, §2.3) | collapse |
| **2.4 correction latency** | ≤ 1 cycle (≤ 2 worst) | ceiling | bump→publish | > 2 cycles |
| **4.1–4.10 PR acceptance** | exact/boolean per row | hard | PR-1 units / PR-2 @ scale | any violation |

Gate 1 + Gate 2 attach to the gate-stage evidence; §4 bars attach to the PR-1/PR-2 bodies as
receipts (D1 per-stage acceptance).
