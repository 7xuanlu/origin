# M4 Stage 0 — Persisted-communities mechanics (projection + routing + publication)

**Rung:** M4 (persisted communities), spec `2026-07-18-kg-unified-model-spec.md` §3.
**Status:** binding Stage-0 spec, committed BEFORE implementation (D1). Authored by
`investigator`. Covers the goal prompt's Stage-0 deliverables (a) projection spec,
(b) routing spec, and (c) generation-guarded publication design in one contract; the
gate PASS/FAIL numbers (d) live in `2026-07-25-m4-gate-criteria.md`. Where this doc
fixes a number, an algorithm, or a routing, that choice is frozen so the benchmark in
the gate doc cannot grade itself (§3: *"the projection … must be written down before
any benchmark. The benchmark runs on this exact projection."*).

All code citations verified against worktree `kg-m3g-edge-grounding`
(`SCHEMA_VERSION = 94`, `crates/wenlan-core/src/db.rs:579`) — the M3g substrate M4
reads. Line numbers are as of that HEAD; **re-verify against `origin/main` at branch
time** (the tree moves; M3g and #390 may have merged, renumbering migrations and
shifting `SCHEMA_VERSION`).

---

## 0. Forks — inherited status + authored-with-veto calls (read first)

Per the goal prompt, forks the spec leaves open are boxed here, not silently decided.
None of the below **blocks** Stage-0 authoring; Q-A (the one PRIMARY blocker) was
resolved by the user before this doc was written.

### 0.1 Inherited forks (from the M4 goal prompt)

| Fork | Status entering Stage 0 |
|---|---|
| **Q-A** — the grounded subgraph is empty; what does M4 group over? | **RESOLVED (user, 2026-07-25):** a prerequisite grounding rung **M3g** was inserted; §3's grounded-only participation rule stands **unamended**. M4 groups over the grounded subgraph M3g fills. Stage-0 authoring proceeds in parallel (data-independent); the **gate stage and all live partition semantics WAIT on M3g** landing a real grounded substrate. |
| **Q-B** — Gate-1 (leiden-rs on-device) FAIL redirect | **Conditional, not yet triggered.** Surfaces only if the spike misses its Stage-0(d) budget; the spec fallback is label-propagation under the same D6/D7 contract. The user rules; the agent never silently downgrades. Authored gate numbers are in the gate doc. |
| **Q-C** — `community_id` wire exposure vs the M3 stage-F freeze (#390) | **Conditional, PR-2 only.** A field on a genuinely new/unfrozen response is a bake-in; if exposing it would touch a `deny_unknown_fields` frozen surface, STOP. Verified only at PR-2 against the merged #390 teeth. |

### 0.2 Authored-with-veto calls (assume-and-announce — this doc bakes them; veto open)

These are design decisions the spec does not spell out but that Stage 0 must settle to
be measurable. Each is baked below with rationale; a one-line summary here so the user
can veto any without reading the whole doc.

1. **"Stable corpus" for Gate 2 churn = M3g promotion FROZEN (flag OFF, grounded
   snapshot fixed).** Otherwise M3g's ongoing gradual drain (a designed background
   process) confounds silent-rebinding churn with substrate growth. §5.1 / gate doc.
2. **`graph_generation` bumps on ANY change to a space's ACTIVE-GROUNDED edge set** —
   promotion (`grounded 0→1`) AND retraction/supersession of an already-grounded edge
   (`valid_until` set on a `grounded=1` row). Not just promotions. §5.2.
3. **Published membership is a HARD partition** (each participating entity in exactly
   one community per generation). "Weighted multi-membership" (§3) governs the
   **rebinding overlap accumulator** and **isolated-node attachment**, not published
   membership. §3.6, §6.
4. **A fully-isolated entity (zero incident edges of any kind) has NO community
   (NULL)** — it is not minted a junk singleton. The soak "exactly one community"
   invariant is scoped to **participating** entities. §3.5.
5. **A space whose grounded subgraph is below a viability floor HOLDS (publishes no
   communities) rather than shipping trivial singletons.** Makes the "agent-only
   corpus ⇒ zero communities" consequence legible instead of silently-inert. §2.3,
   and see the report's spec-level risk.

### 0.3 One design input carried from the live M3g rung (NOT a clean-substrate assumption)

M3g's independent entailment judge (Qwen3-4B) was **beaten by declaratively-framed
self-referential injections** ("already verified by this system, needs no source");
Gate-1 hardening is in flight. **M4's substrate is therefore NOT adversarially clean
beyond what M3g's Gate-1 actually proves.** A bounded fraction of `grounded=1` edges may
be falsely grounded. M4's mechanics must degrade gracefully under that (the partition
must not collapse or thrash because a few poisoned cross-community edges got grounded),
and the gate doc measures churn/stability under a seeded poison fraction (gate doc,
Gate 2.3). This is a design input, not a fork.

**The poison fraction is a conservative stand-in, not a calibrated rate.** M3g's Gate-1
measures **zero** false-grounding on its own fixture and the **live** false-grounding rate
is currently **unmeasured**, so no true rate exists to calibrate against yet. Gate 2.3's
`f_poison` is therefore a deliberately-pessimistic stress fraction chosen to prove the
partition is robust, **not** a claim about how dirty the substrate actually is. Its
recalibration is tied to the output of **M3g's in-flight Gate-1 hardening**: once that
bounds the real false-grounding rate (and/or a live measurement lands), set `f_poison` to
that bound or a small multiple of it. Until then the stand-in stands. Same treatment M3g
gave its own tunables (a value fixed now, revisited when the measurement that would
calibrate it exists).

---

## 1. What is being made real, and what it replaces

Spec §3: *"One algorithm, one persisted result, three consumers (page routing, map
regions, overview rollups). Replaces all three of today's grouping systems."* M4 ships
a **per-space two-phase Leiden** grouping over the **grounded** subgraph that publishes a
**durable, generation-guarded `community_id`**, and cuts the **daemon-side** consumers over
to it. Only two of the spec's three consumers are daemon-side (below); the third
(map-region/overview) is client-side and lands with the app / M6, not in M4's cutover.

**What exists today (verified):**

- **The system M4 replaces — `detect_communities` (`db.rs:26287`).** It is **GLOBAL,
  not per-space** (`SELECT id FROM entities` with no space scope, `db.rs:26291-26292`),
  **label propagation** (max 100 iterations, `db.rs:26389`), over active `relates`
  adjacency weighted by parallel-edge count (`db.rs:26371-26376`). It reads `edges` when
  `reader_uses_edges("communities")` is flipped-and-clean (`db.rs:26332`), else legacy
  `relations` (`db.rs:26348`), and its `edges` path filters `edge_type='relates' AND
  valid_until IS NULL` with a cross-space-legacy carve-out but **NO `grounded` filter**
  (`db.rs:26340-26346`) — it reads ALL active `relates`, grounded or not. It writes a
  transient `entities.community_id INTEGER` (migration 27, `db.rs:5143-5144`; index
  `idx_entities_community`, `db.rs:5144`), overwritten every run — **not durable, no
  overlap-rebinding, no generation guard, no display name, no space scope**. Its write
  loop uses a **bare unprotected `BEGIN`/`COMMIT`** (`db.rs:26434`/`:26446`) — the
  issue-#389 class D9 fixes. It runs as `Phase::CommunityDetection` (refinery,
  `refinery/mod.rs:928`/`:942-943`).
- **Consumers of the transient `entities.community_id` (M4's PR-2 cutover surface):**
  (1) **T18 summary rollups** — `load_summary_buckets` (`db.rs:19392`; doc comment
  `db.rs:19387`, `SELECT m.source_id, m.title, m.content, e.community_id … WHERE …
  AND e.community_id IS NOT NULL ORDER BY e.community_id, m.last_modified DESC`,
  `db.rs:19399-19416`; grouped into per-community buckets in a Rust `BTreeMap`, NOT SQL
  `GROUP BY`, `db.rs:19422-19423`) feeding `refinery/summary.rs` (`:214`, `:243`, `:258`
  — `sum_b_{community_id}` nodes); (2) **peer-entity community eligibility** for summary
  refresh — `summary_eligible_predicate` (`derived_artifact_state.rs:7-32`, groups by
  `peer_entity.community_id`, `community_id IS NOT NULL`, `HAVING COUNT(*) >= minimum`).
  These are the **two daemon-side** consumers M4's PR-2 cuts over. Map-region/overview
  rollups are the spec's third §3 consumer but are **client-side** today (the app's degree
  heuristic) — no daemon reader exists, so they are not in this tree's cutover surface
  (deferred to the app / M6).
- **No durable community substrate.** Grep for `graph_generation` / `community_generation`
  over `crates/*/src` is **empty** (verified). M4 builds the `communities` table and the
  per-space `graph_generation` counter net-new.
- **No partitioner crate.** `leiden` / `louvain` / `petgraph` are absent from the
  workspace `Cargo.toml`s (verified). Gate-1's spike adds one.
- **The grounded substrate M4 reads (from M3g):** `edges.grounded INTEGER NOT NULL
  CHECK(grounded IN (0,1))` (`db.rs:8562`), `edges.root_id TEXT REFERENCES
  provenance_roots(root_id)` (`db.rs:8563`), `edges.space TEXT NOT NULL` (`db.rs:8564`),
  `edges.weight REAL` nullable (`db.rs:8565`), `edges.valid_until` (`db.rs:8571`). The
  partial index `idx_edges_active_grounded_space_type ON edges(space, edge_type) WHERE
  valid_until IS NULL AND grounded = 1` (`db.rs:8578-8579`) is exactly the access path
  M4's grounded-subgraph scan rides. M3g's `promote_edges_grounded` (`db.rs:10705`,
  `UPDATE edges SET grounded=1, root_id=?, payload=? WHERE edge_id=? AND grounded=0`,
  `db.rs:10721-10722`) is the monotone `grounded 0→1` write M4's `graph_generation`
  hooks onto (§5.2, D-G).

---

## 2. The subgraph and participation (§3, honoring Q-A grounded-only)

### 2.1 The projection subgraph

M4's Leiden runs **per space**, over the **active grounded `relates` subgraph** of that
space:

```
edges WHERE edge_type = 'relates'
        AND valid_until IS NULL          -- active
        AND grounded = 1                 -- Q-A: grounded-only, unamended
        AND space = ?<space>             -- per-space
```

This is exactly the `idx_edges_active_grounded_space_type` index shape (`db.rs:8578-8579`)
plus `edge_type='relates'`. `mentions` is not a live producer (schema-CHECK only, per M3g
D-B); `cites`/`supports`/`links` are structural, not entity↔entity, and are excluded.

Endpoints are entity ids (`src_kind='entity'`, `dst_kind='entity'`). The nodes of the
graph are the **entities** appearing as an endpoint of ≥1 such edge.

### 2.2 Participation rule (§3, two-phase)

- **A node PARTICIPATES in partitioning iff it has ≥1 active grounded incident `relates`
  edge in its space** (grounded degree ≥ 1). These nodes and the grounded edges among
  them are the graph Leiden optimizes over.
- **Grounded-degree-0 nodes are assigned AFTER partitioning** to their strongest
  attachment and **cannot perturb the objective** (§3, isolated-node handling — §3.5).

### 2.3 Viability floor (authored, §0.2 call 5)

A space publishes communities only when its participating-node count clears a floor
`M4_MIN_PARTICIPANTS` (provisional value in the gate doc, tuned against the gate). Below
the floor the space **HOLDS** — it publishes nothing and stays queued — rather than
minting a degenerate all-singletons partition. Rationale: M3g fills the grounded subgraph
**gradually** (entailment-bounded, one promotion per ambient turn — `edge_grounding.rs`
`run_edge_grounding_slice`), so early in the drain a space has very few grounded edges;
publishing near-empty communities would (a) churn violently as the drain proceeds and (b)
present a misleading map. Holding is the legible state. **Consequence to surface:** a
corpus whose knowledge is entirely agent-captured (`source_agent != 'folder'`) has an
empty grounded subgraph forever under M3g's Q-G1 folder-only ruling, so it stays below the
floor and publishes **zero communities** — see the report's spec-level risk.

---

## 3. Projection spec (Stage 0a — §3, written before the benchmark)

The projection turns the §2.1 subgraph into the weighted undirected graph Leiden
optimizes. **The benchmark runs on THIS projection.** Every knob below is fixed here;
the provisional numeric constants are collected in the gate doc and tuned against the
gate, never re-chosen after measurement.

### 3.1 Direction folding

`relates` is undirected in meaning but each edge has a stored `src_id`/`dst_id`. Fold to
an undirected pair by **canonical endpoint ordering**: the unordered pair
`{min(src_id,dst_id), max(src_id,dst_id)}` (lexicographic on the entity id string) is the
projection node-pair key. Both directions of the same logical relation collapse to one
undirected weighted edge. (M3g's content-addressed `edge_id` already makes a logical
relation a single row, but a corpus may legitimately hold both `A relates B` and
`B relates A` as distinct relation types; folding is on the endpoint pair, aggregating
across §3.3.)

### 3.2 Per-edge-type weight scaling

Only `relates` participates, so there is one edge type and a single base weight
`W_relates` (provisional in the gate doc). The per-edge-type-weight machinery is authored
now (a `HashMap<edge_type, f64>` with `relates → W_relates`) so a later rung that grounds
a second extracted edge type slots in without reshaping the projection. `edges.weight`
(nullable, `db.rs:8565`) is **not assumed populated** — the current `relates` producer
does not set it (verify in PR); the projection treats `weight IS NULL` as "use the base
type weight," and uses a present `weight` as a multiplier only if the PR confirms it is
meaningfully populated. Default path: base type weight only.

### 3.3 Parallel-edge aggregation

Multiple grounded edges over one folded endpoint pair (distinct relation types, or
re-extractions) aggregate to a single projected weight by **summation of per-edge base
weights, capped** at `PARALLEL_EDGE_CAP × W_relates` (provisional cap in the gate doc).
The cap prevents a single over-extracted pair from dominating the objective. This mirrors
the intent of today's parallel-edge count (`db.rs:26371-26376`) but bounded and grounded.

### 3.4 High-degree source-page / hub normalization

A few entities (generic hubs — "Project", "2024", a frequently-co-mentioned org) accrue
disproportionate grounded degree and would smear communities together. Normalize a hub's
incident weights so it cannot act as a bridge that collapses distinct communities:
down-weight each incident edge of a node with grounded degree > `HUB_DEGREE_CAP` by
`HUB_DEGREE_CAP / degree` (a soft cap, not a hard drop), so a hub still participates but
its per-edge pull is bounded. Constants provisional (gate doc). This is the §3
"high-degree source-page normalization" made concrete for an entity graph.

### 3.5 Isolated-node handling (grounded-degree-0)

After Leiden partitions the participating nodes, each grounded-degree-0 node is attached
**post-hoc** (it never entered the objective):

1. If it has ≥1 incident **ungrounded** active `relates` edge, attach to the community of
   its strongest such neighbor (by §3.2-§3.4 projected weight over the ungrounded
   adjacency), i.e. the "strongest attachment" of §3.
2. Else if the routing model (§8) has a page-embedding for it, attach to the nearest
   community centroid (the entity-embedding fallback echoing §3's page-embedding fallback).
3. Else (a fully-isolated entity — zero incident edges of any kind) it gets **NO community
   (NULL)**; it is not minted a singleton (§0.2 call 4). This is a distinct, legible state.

Post-hoc attachment cannot change the partition of participating nodes (it reads the
frozen partition, writes only the isolated node's own assignment), preserving §3's
"cannot perturb the objective."

### 3.6 Weighted multi-membership (rebinding only — NOT published)

Published membership is a **hard partition** (§0.2 call 3): each participating entity is
in exactly one community per generation (the soak invariant, scoped to participants per
§3.5). "Weighted multi-membership" (§3) is used **only** inside the old→new rebinding
accumulator (§6): a boundary node whose grounded edges straddle two old communities
contributes **fractional overlap weight** to each, so max-overlap rebinding is stable
under a node that genuinely sits between two groups. It never surfaces as a node belonging
to two published communities.

### 3.7 Determinism of the projection

The projection is a pure deterministic function of the subgraph: edges scanned in a fixed
order (`ORDER BY edge_id`), folded/aggregated/normalized by the fixed rules above, so the
same subgraph always yields byte-identical projected adjacency. This is the precondition
for the determinism story in §4.2 and the differential oracle in the gate doc.

---

## 4. Algorithm, determinism, and stability

### 4.1 Algorithm — two-phase Leiden (fallback: label propagation)

The intended algorithm is **Leiden** (the spike adds the crate — `leiden-rs` or whatever
Gate-1 proves viable on-device; see the gate doc + Q-B). Leiden's guarantees (well-connected
communities, no badly-disconnected community that Louvain can produce) matter for the map's
legibility. **Label propagation is the contract-bound FALLBACK** under the SAME D6/D7
durable + generation-guarded contract (§3, invariant #5): if Gate-1 fails (Q-B, user-ruled)
or Leiden is unavailable on a host, the fallback produces communities under identical
identity/rebinding rules. Identity survives the partitioner swap — Leiden→label-prop is a
rebinding event under §6's max-overlap rule, **not a reset** (invariant #5, §1 Community).

### 4.2 Determinism (a gate precondition)

Leiden is randomized (random node-visit order / tie-breaking). M4 pins:

- a **fixed RNG seed** (a constant, stored with the algorithm version), and
- a **deterministic node iteration order** (entities sorted by id), and
- deterministic tie-breaking (smallest-id wins, mirroring today's label-prop tie-break at
  `db.rs:26406`).

So on a **frozen** subgraph the job is a pure function → identical assignment every run.
This is what makes Gate 2's churn measurement meaningful: any churn on a frozen subgraph is
a bug, not RNG.

### 4.3 Incrementality — warm-start, not full re-partition (§6.5, required not aspirational)

A full re-partition per cycle is a **Gate-1 fail** (§3). The job is incremental:

- It captures the **dirty node set** = entities incident to any `relates` edge whose
  `(grounded, valid_until)` membership in the active-grounded set changed since the last
  published generation for the space (from the generation bookkeeping, §5.2).
- It **seeds Leiden with the previous published membership** (§6 gives the durable prior
  assignment), and runs **bounded local re-optimization** on the dirty frontier + its
  1-hop neighborhood only — the rest of the partition is carried forward unchanged.
- A **full re-partition** happens only on first publication for a space, on an algorithm/
  projection-version change (§5.4 — versions changed ⇒ derived state recomputed), or when
  the dirty fraction exceeds a `FULL_REPARTITION_FRACTION` threshold (provisional, gate
  doc) where incremental would cost more than a full pass.

Gate-1 measures the incremental warm-start cost around a changed subgraph and fails a
per-cycle full re-partition.

### 4.4 Stability vs truthfulness are separate dials (invariant #12)

Two independent measured dials, per invariant #12 and Gate 2:
- **Churn rate** — how much *membership* reshuffles per cycle on a frozen corpus
  (silent-rebinding stability, invariant #16). Driven down by determinism (§4.2) +
  warm-start (§4.3) + overlap-rebinding (§6).
- **Correction latency** — how fast a correction / merge / new grounded edge propagates
  into the published assignment. Driven by the generation-guard trigger (§5) + incremental
  re-optimization (§4.3).
These are traded against each other by the incrementality thresholds and are measured
independently (gate doc Gate 2). Neither is self-graded.

---

## 5. Generation-guarded publication (Stage 0c — D7)

### 5.1 Data model (durable substrate, additive migration)

New tables (migration number = next free on `origin/main` at branch time; this tree is at
`SCHEMA_VERSION = 94`). All DDL is `CREATE TABLE IF NOT EXISTS`, replay-safe, additive.

```sql
-- Community identity (durable; survives regrouping — invariant #5).
CREATE TABLE IF NOT EXISTS communities (
    community_id   TEXT PRIMARY KEY,      -- stable durable id (ULID/uuid), NOT a partitioner label
    space          TEXT NOT NULL,
    display_name   TEXT,                  -- changes ONLY by proposal (D8); NULL until named
    algo_version   TEXT NOT NULL,         -- §6.6: the algorithm version that produced it
    projection_version TEXT NOT NULL,     -- §6.6: the §3 projection version
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL,
    retired_at     INTEGER                -- set when a community dissolves (no members); never hard-deleted while in rollback window
);
CREATE INDEX IF NOT EXISTS idx_communities_space ON communities(space) WHERE retired_at IS NULL;

-- Membership — the versioned assignment snapshot (recomputable, not source-of-truth §6.6).
-- One row per participating entity per space: its CURRENT durable community + the
-- generation that published it.
CREATE TABLE IF NOT EXISTS community_members (
    space          TEXT NOT NULL,
    node_id        TEXT NOT NULL,         -- entity id
    node_kind      TEXT NOT NULL DEFAULT 'entity',
    community_id   TEXT NOT NULL REFERENCES communities(community_id),
    published_generation INTEGER NOT NULL,
    attachment     TEXT NOT NULL,         -- 'core' | 'isolated_ungrounded' | 'isolated_embedding' (§3.5)
    PRIMARY KEY (space, node_id)
);
CREATE INDEX IF NOT EXISTS idx_community_members_community ON community_members(community_id);

-- Per-space monotonic generation + published bookkeeping (D7).
CREATE TABLE IF NOT EXISTS space_graph_state (
    space          TEXT PRIMARY KEY,
    graph_generation      INTEGER NOT NULL DEFAULT 0,  -- bumped by active-grounded edge writes (§5.2)
    published_generation  INTEGER,                     -- last generation whose assignment is live
    dirty                 INTEGER NOT NULL DEFAULT 0    -- CAS-cleared only via WHERE graph_generation = input_generation
);

-- Durable §6.2 phase lease keyed (phase, space, input_generation) — a process mutex is not a lease.
CREATE TABLE IF NOT EXISTS grouping_leases (
    phase          TEXT NOT NULL,         -- 'community'
    space          TEXT NOT NULL,
    input_generation INTEGER NOT NULL,
    token          TEXT NOT NULL,
    expires_at     INTEGER NOT NULL,
    attempt        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (phase, space, input_generation)
);
```

The durable `community_members.community_id` is the **write-only shadow** in PR-1 — the
label-prop `entities.community_id` (migration 27) is untouched; consumers still read it.
PR-2 flips consumers onto `community_members`.

Rebinding old→new keys on **member-set overlap, not partitioner labels** (§1 Community,
D6) — so the identity in `communities` is defined by which entities were in it, and the
`community_id` string is opaque.

### 5.2 What bumps `graph_generation` (authored, §0.2 call 2)

`graph_generation` for a space bumps by **exactly one on any committed change to that
space's ACTIVE-GROUNDED `relates` edge set**:

- **promotion** `grounded 0→1` — hooked onto M3g's `promote_edges_grounded`
  (`db.rs:10705`) in the SAME transaction as the flip (D-G: M3g reserves nothing; M4 adds
  the bump here);
- **retraction / supersession** of an already-`grounded=1` edge — `valid_until` set on a
  grounded row (deletion-cascade, entity merge, edge retraction, §2 of the spec) bumps too,
  because it changes the subgraph the partition is defined over.

Both mark the space `dirty=1` in `space_graph_state` in the same transaction. A pure
`grounded=0` write (ordinary extraction) does **not** bump — it is not in the grounded
subgraph. Rationale: the spec says "bumped by grounded-edge writes"; a retraction of a
grounded edge is such a write, and omitting it would leave a space silently stale after a
correction. The bump is monotonic and never decreases.

### 5.3 The publication CAS (generation-guard, not check-then-act)

A grouping job for a space:

1. **Acquires the durable phase lease** `(phase='community', space, input_generation)` in
   `grouping_leases` (§6.2) — token + expiry; a second concurrent job for the same
   `(space, input_generation)` is excluded at the DB, not by a process mutex. Expired
   leases recover at startup.
2. **Captures `input_generation`** = the space's current `graph_generation` at job start.
3. Reads the subgraph, projects (§3), partitions (§4), rebinds (§6) — **all outside any
   SQLite transaction** (no txn spans a compute-heavy or embedding call, §6.3).
4. **Finalizes in one CAS-guarded transaction**: writes the new `community_members`
   snapshot + `communities` upserts + sets `published_generation = input_generation`, and
   **clears dirty ONLY via `UPDATE space_graph_state SET dirty=0, published_generation=?
   WHERE space=? AND graph_generation = input_generation`**. If a grounded edge arrived
   mid-run, `graph_generation != input_generation`, the CAS matches zero rows, the snapshot
   is **discarded** (nothing visible published on a stale input), and the space stays
   `dirty` and re-queues. A mid-run edge is never silently lost (§3, runtime finding 19).

The finalize transaction uses the **rollback-protected `BEGIN` idiom** (`fold_relation_type`,
`db.rs:304`; D9) — no bare `BEGIN`/`COMMIT` (unlike today's `detect_communities` at
`db.rs:26434`/`:26446`, which M4 rewrites).

### 5.4 Version stamping (§6.6 — recomputable, not durable)

Every `communities` row carries `algo_version` + `projection_version`. Scores/assignments
from different versions are never compared under one threshold; a version change forces a
**full re-partition** (§4.3) and re-derivation, never an in-place reinterpretation. The
assignment (`community_members`) is defined **recomputable-not-durable** (§6.6): it can be
regenerated from `edges` + the stored versions. The community IDENTITY (`communities`
rows, invariant #5) is the durable part. Preserving the versions is what makes the PR-1
rollback ("disable and recompute") reproducible (§9).

---

## 6. Old→new rebinding (D6) — the inverted accumulator, splits/merges as proposals

Community identity is **member-set-overlap-defined** (§1 Community, D6), never the
partitioner's internal labels. After a (re)partition produces new label groups, rebind them
to durable ids:

1. **Build the inverted node→old-community accumulator** (never all-pairs, §3 / D7): for
   each node in each new group, look up its previous durable `community_id` from
   `community_members` and accumulate an overlap score into `overlap[new_group][old_id]`.
   Boundary nodes contribute **weighted-fractional** overlap (§3.6) split across the old
   communities their grounded edges straddle.
2. **Assign each new group the old `community_id` of maximum overlap**, subject to **each
   old id claimed by at most one new group** (the max-overlap claimant). Tie-break by
   larger absolute overlap, then smallest old `community_id` (deterministic).
3. **Unclaimed new groups mint a fresh `community_id`.** Unclaimed old ids (no new group
   inherited them) are **retired** (`retired_at` set; kept through the rollback window,
   never hard-deleted).
4. **Splits and merges are REVIEW PROPOSALS, not silent identity resets** (D6/D8): when one
   old community's members land mostly in ≥2 new groups (split), or ≥2 old communities
   collapse into one new group (merge), the structural rebinding **still applies** (the
   membership is updated), but the **identity event** (which new group keeps the old
   name/id, whether a second community is proposed) surfaces on the review-proposal path.
   The mental map never moves silently.

**Partitioner-swap invariance:** a Leiden→label-prop swap runs the identical rebinding —
the new algorithm's groups are rebound to the existing durable ids by the same max-overlap
rule, so identity survives the swap (invariant #5, §4.1).

---

## 7. Names change only by proposal (D8, invariant #16)

Two orthogonal kinds of change:

- **Membership rebinding is SILENT and structural** — a node moving communities, a group
  minting/retiring an id, updates `community_members`/`communities` directly (no user
  gesture). This is the churn the Gate-2 stability dial measures.
- **A user-visible name change** — a community/overview `display_name` or a map-region
  label — happens **ONLY through the review-proposal path** (the same path as splits/merges,
  §6). Rebinding may change *who is in* a community without ever changing what it is
  *called*; renaming requires a proposal a human accepts.

`display_name` is nullable and defaults NULL (unnamed); naming/renaming is always a
proposal. This is invariant #16 made concrete: *"rebinding is silent, relabeling never is."*

---

## 8. Routing spec (Stage 0b — §3 page↔community, hysteresis, triggers)

Communities are computed over **entities**; the **page-level** §3 consumers (page routing, and
the client-side map/overview) need a **page→community** mapping. Routing derives it. The `T_hi`/`T_lo` constants and weights are authored provisional
here and **tuned against the gate** — internal, not product-visible (bake, don't STOP).

### 8.1 Page→community score

For a page `P` (any `kind`), compute a score to each candidate community `C` from the page's
entities:

- Each entity the page is built on / mentions (via `edges` `mentions`/`cites` to the page's
  memories, and — for a `kind='entity'` shadow page — the entity itself) casts a vote for
  its own community, weighted by the entity's grounded degree within `C` (a well-connected
  member votes more strongly than a peripheral one).
- `score(P, C) = Σ_entities-in-C  entity_weight` , normalized by the page's total entity
  weight so scores are comparable across pages of different sizes.

### 8.2 Hysteresis (assign above `T_hi`, hold between, drop below `T_lo`)

- **Assign** `P → C` when `score(P, C) ≥ T_hi` and `C` is the arg-max.
- **Hold** its prior community when the arg-max score is in `[T_lo, T_hi)` (hysteresis band
  — avoids flip-flopping a page between two near-tied communities).
- **Drop** the assignment (page unassigned) when the arg-max `score < T_lo`.

`T_hi > T_lo`; both provisional in the gate doc. Hysteresis is what keeps page↔community
assignment from adding churn on top of membership churn.

### 8.3 Page-embedding fallback for entity-poor pages

A page with too few entities to score reliably (below `MIN_PAGE_ENTITIES`) routes by
**page embedding**: nearest community centroid, where a community centroid is the mean
embedding of its member entities' shadow pages (M3 `kind=entity` pages carry embeddings).
The fallback is subject to the same `T_hi`/`T_lo` hysteresis on the cosine score.

### 8.4 Update triggers and invalidation

Page→community routing recomputes on: (1) a **new grounded edge** touching one of the page's
entities; (2) a **community rebinding** that moved one of the page's entities; (3) a **page
refresh** (content/entity-set change). On any community **rebinding** (§6), the affected
pages' assignments are **invalidated** and recomputed under §8.2 — an assignment can never
outlive the community structure it was computed against.

---

## 9. Incremental update + rollback semantics

### 9.1 Incremental update (steady state)

The grouping job lives in the existing **`Phase::CommunityDetection` steep-phase slot**
(`refinery/mod.rs:928`/`:942-943`), replacing the `detect_communities` call — NOT a new
ambient sweep (per the goal prompt; the incrementality design does not argue for a separate
sweep). It fires for a space when `space_graph_state.dirty=1` (its `graph_generation`
advanced past `published_generation`). Each firing does the §5.3 CAS publication with the
§4.3 warm-start. Bounded per firing (§10).

### 9.2 Rollback (§6.9, §7 M4 row)

- **PR-1 rollback (shadow):** derived data — **disable the Leiden job and recompute**. The
  label-prop `entities.community_id` and its consumers were never touched, so the system
  falls back cleanly. The stored `algo_version`/`projection_version` are **preserved** so a
  recompute is reproducible. The migration is additive; the durable `community_id` is a
  shadow → recoverable.
- **PR-2 rollback (cutover):** flip each consumer back to `entities.community_id` (the
  label-prop producer stays live as the D6/D7 fallback); the wire field is additive and
  ignorable. Community assignments are **recomputable, never authoritative source data** —
  a restore recomputes them from `edges` + versions; it never loses ground truth.
- **App-heuristic retirement** waits out the rollback window precisely so this fallback
  stays available on the client too (deferred to the app repo, §Deferred).

---

## 10. Caps, bounds, and flag surface

### 10.1 Per-firing bounds (foreground-safe, §6.3 / §6.5)

Mirroring the reconcile-sweep discipline (`reconcile.rs`) and heeding the 18.88s-mutex
cautionary datum (`crates/wenlan-core/AGENTS.md`), the grouping job:

- holds the single DB connection mutex only for the **bounded subgraph SELECT**, the
  **lease acquire**, and the **finalize CAS transaction** — never during projection,
  partitioning, rebinding, or any embedding call (§5.3, §6.3);
- caps the per-firing work: a space's re-optimization is bounded by the dirty-frontier size
  (§4.3), and a full re-partition is chunked so its mutex hold stays under the Gate-3-style
  ceiling (gate doc); constants (`M4_MIN_PARTICIPANTS`, `HUB_DEGREE_CAP`,
  `PARALLEL_EDGE_CAP`, `FULL_REPARTITION_FRACTION`, `T_hi`, `T_lo`, `W_relates`,
  `MIN_PAGE_ENTITIES`) are provisional in the gate doc, tuned against the gate;
- no SQLite transaction spans an embedding call (the routing centroid embeddings, §8.3, are
  read outside the finalize txn).

### 10.2 Flag surface (default-OFF, drift-teeth #2 documented)

A new default-OFF `WENLAN_*` flag gates whether the Leiden job runs (the "shadow" per D3 is
"flag off"), parsed by a `wenlan_core::db::*_enabled()` helper mirroring
`edge_grounding_promote_enabled` / `edges_reconcile_enabled`, checked in the phase gate.
Proposed name `WENLAN_ENABLE_COMMUNITY_LEIDEN` (final name + its `crates/wenlan-core/AGENTS.md`
flag-doc entry land in the PR, following the existing entry format). The per-consumer PR-2
cutover uses the existing `set_reader_cutover` / `reader_uses_edges` cutover-plane pattern
(as `detect_communities` already does for `"communities"`, `db.rs:26332`) so each **daemon-side**
consumer (the two in §1) flips reversibly behind a clean parity watermark. The client-side
map/overview consumer is not on this plane (app / M6).

---

## 11. Non-interference (preserve explicitly)

- **M2 soak:** M4 reads `edges` natively (D2) and adds no legacy reader-cutover of its own.
  It writes only NEW tables (`communities`, `community_members`, `space_graph_state`,
  `grouping_leases`) and — for `graph_generation` — hooks the bump onto M3g's promotion
  write. It never mutates a structural `edges` column, so the M2 parity oracle
  (`reconcile_edges_parity`, structural-only) sees zero drift. M2's soak (which gates M3's
  reader-cutover flips) stays valid.
- **M3g:** M4 consumes M3g's grounded substrate read-only; it does not touch
  `promote_edges_grounded`'s logic except to add the `graph_generation` bump in the same
  transaction (D-G, coordinated at branch time — M3g must have merged first). M4's gates
  wait on M3g landing a real grounded subgraph.
- **M3 (#390):** unrelated — the `entity_id↔page_id` adapter seam is not a communities
  concern (D4). M4 branches from `origin/main`, not on #390. If exposing `community_id` on
  the wire would touch a frozen surface, that is Q-C (STOP), verified at PR-2.
- **issue #389:** M4 uses the rollback-protected `BEGIN` idiom everywhere (D9) and does not
  depend on #389's separate durability work; it rewrites `detect_communities`' bare
  `BEGIN`/`COMMIT` to the protected idiom as it replaces it.

---

## 12. Left to the implementation stage (not decided here)

- Final flag name + its `crates/wenlan-core/AGENTS.md` flag-doc entry (drift-teeth #2), and
  the per-consumer cutover keys.
- The partitioner crate choice (`leiden-rs` vs alternative) — decided by the Gate-1 spike.
- The exact migration numbers (re-read `SCHEMA_VERSION` on `origin/main` at branch time).
- The concrete `community_id` id scheme (ULID vs uuid) and centroid-embedding storage detail.
- Whether `edges.weight` is meaningfully populated by the `relates` producer (verify; the
  projection defaults to base type weight if not, §3.2).
- Wire-exposure shape of `community_id` for the app (PR-2, gated on Q-C).
- The provisional constants' final tuned values (§10.1) — set to meet the gate bars without
  weakening any gate.
