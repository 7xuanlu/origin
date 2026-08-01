# M6 Stage-0 artifact 6 — relevance contract

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: frozen contract D9 in full, and gate G6
(`m6_relevance_is_bounded_and_safe`).
Continues the decision numbering from artifact 8 (`S0-84`).

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

## 0. Read this first: which rows are blocked

D9 requires that co-citation pair inputs come only from roots supporting *"exact
current M5-supported page claim revisions"*, and that *"every candidate page must
be the current M5 `supported` version."* Artifact 1 flagged a dead-substrate STOP
against that predicate. This artifact confirms it, narrows it, and marks every
affected row rather than writing a contract that reads as live.

**The M5 claim/truth substrate is fully built, and no production writer promotes
`supported`.** Rev 1 said "entirely inert", which `#418` has made false and which
was already broader than the evidence under it. `#418` shipped a live production
writer into `page_truth_state`: `POST /api/pages/{id}/review`
(`crates/wenlan-server/src/page_routes.rs:42`) reaches an upsert that sets
`human_reviewed = 1` (`crates/wenlan-core/src/db/presence_review.rs:326`-`:341`).
The substrate therefore has a writer — on the **human** axis. It deliberately does
not touch the machine axis, and says so at the seam: the inserted row is
`'provisional'` with a NULL `evaluated_at` because *"a human saying 'I read this'
is not evidence about whether the machine found support — inventing `supported`
here would collapse the separation the whole rung exists to make"*
(`:321`-`:325`). The claim that survives, and the only one D9 rests on, is the
narrow one: **nothing outside tests writes `support_status = 'supported'`** — a
repository-wide search for the literal at `1c903bec` finds no production hit.

That narrowing does not weaken S0-153's census: a page marked human-reviewed still
carries `evaluated_at IS NULL`, so clause (b) correctly counts it as un-judged
rather than as decided.

Every table
D9 needs exists — `claims`, `claim_revisions`, `claim_anchors`,
`page_version_claims`, `claim_derivation_markers`, `page_truth_state`,
`claim_derivation_jobs` (`crates/wenlan-core/src/db/claim_identity.rs:148`,
`:162`, `:194`, `:215`, `:237`, `:279`, `:306`) — with immutability triggers,
indexes, and a coverage-checked backfill. What does not exist is anything that
*promotes* a page:

- Migration 99 backfills **every** pre-existing page as `'provisional'` with the
  reason `'never evaluated: predates claim derivation'`
  (`crates/wenlan-core/src/db/claim_identity.rs:511`-`:512`), and hard-errors if
  any page is left without a truth row (`:539`-`:543`).
- No production code path writes `support_status = 'supported'`. The only such
  write in the tree is a test
  (`crates/wenlan-core/src/export/projection_invariant_test.rs:469`).
- `claim_derivation_jobs` is created and indexed
  (`crates/wenlan-core/src/db/claim_identity.rs:306`, `:319`-`:322`) and then
  read and written by nothing — there is no enqueue, no worker, and no scheduler
  lane.

So on any real install today, **every page is `Unevaluated`** — which the read
side already distinguishes from judged-and-failed via the three-state
`Support` enum (`crates/wenlan-core/src/db/truth_exposure.rs:158`-`:162`, whose
comment records exactly why: reading `provisional` as "the evidence does not back
this" *"made every page that predates claim derivation an eviction target"*).

Consequence for D9, stated plainly: **a literal implementation of the relevance
contract retrieves zero candidates and forms zero co-citation pairs on every
install.** It is not wrong, it is unreachable. Rows below carry one of:

| Mark | Meaning |
|---|---|
| **LIVE** | the predicate can be evaluated against real data today |
| **BLOCKED** | schema exists; the predicate is constant-false until a claim-derivation promoter ships |
| **PR-A-new** | no substrate; PR-A must create it |

> **Decision S0-85 — Stage 0 does not weaken the `supported` predicate to make
> the contract reachable.** The tempting move is to accept `Unevaluated` pages as
> candidates so PR-B's shadow numbers are non-zero. It is refused: D9's whole
> safety argument is that relevance rests on judged evidence, and admitting
> unjudged pages converts the contract into "attach things to things", which is
> what M6 exists to replace. The correct sequencing is that **a claim-derivation
> promoter is a hard prerequisite of PR-B's genesis shadow**, not a follow-up.
> This is a scheduling consequence for the program plan, and I am reporting it
> rather than resolving it.

---

## 1. The formula

```
score = (4 * co_citation
         + 3 * direct
         + 1.5 * common_neighbor
         + kind_affinity) / 9.5
```

Each term is in `[0, 1]`, so `score` is in `[0, 1]`. The divisor is the sum of
the weights, which makes the maximum exactly `1.0`.

| Term | Weight | Max contribution | Constant ID |
|---|---|---|---|
| `co_citation` | 4 | 0.42105 | `R-W-COCITE` |
| `direct` | 3 | 0.31579 | `R-W-DIRECT` |
| `common_neighbor` | 1.5 | 0.15789 | `R-W-CNEIGH` |
| `kind_affinity` | 1 | 0.10526 | `R-W-KIND` |
| divisor | 9.5 | — | `R-W-DIVISOR` |

### 1.1 A structural property worth naming

Because `4/9.5 = 0.42105 < 0.50`, **co-citation alone can never reach the assign
threshold**, at any NPMI including a perfect `1.0`. Solving
`4c + 3d + 1.5n + k >= 4.75` for the co-citation needed at each combination of
the other terms:

| direct | common-neighbor | kind | co-citation needed to assign |
|---|---|---|---|
| 0 | 0 | 0 | 1.1875 — **impossible** |
| 0 | 0 | 0.5 | 1.0625 — **impossible** |
| 0 | 0 | 1 | 0.9375 |
| 0 | 1 | 0 | 0.8125 |
| 0 | 1 | 1 | 0.5625 |
| 1 | 0 | 0 | 0.4375 |
| 1 | 1 | 1 | already above threshold |

This is a good property and G6 should assert it directly: it means D9's
*"either a direct signal or qualified co-citation"* rule has teeth even before
the rule is separately enforced, because the two weakest configurations cannot
cross on co-citation at all.

### 1.2 Term predicates, grounded

The shared eligibility spine for every graph term. `edges` is the substrate
(`crates/wenlan-core/src/db.rs:8803`), widened by migration 98 to carry
`claim_revision` and `root` endpoints and the `attests` edge type
(`crates/wenlan-core/src/db.rs:11737`):

```sql
-- the eligible-edge predicate, used by all three graph terms
    valid_until IS NULL          -- active            (db.rs:8820)
AND grounded = 1                 -- validator-grounded (db.rs:8811)
AND root_id IS NOT NULL          -- externally rooted  (db.rs:8812)
AND lineage <> 'legacy'          -- excludes legacy-ungrounded (db.rs:8810)
AND space = ?                    -- no cross-space
```

joined to `provenance_roots` (`crates/wenlan-core/src/db.rs:8787`) for:

```sql
    pr.status = 'active'                 -- (db.rs:8793)
AND pr.root_kind <> 'generated'          -- (db.rs:8791)
```

| D1/D9 rule | Enforcing predicate | Where the data lives | Status |
|---|---|---|---|
| generated roots contribute zero | `pr.root_kind <> 'generated'` | `provenance_roots.root_kind` (`db.rs:8791`) | **LIVE** |
| inactive roots contribute zero | `pr.status = 'active'` | `provenance_roots.status` (`db.rs:8793`) | **LIVE** |
| retracted contributes zero | `e.valid_until IS NULL` | `edges.valid_until` (`db.rs:8820`) | **LIVE** |
| legacy-ungrounded contributes zero | `e.grounded = 1 AND e.lineage <> 'legacy'` | `edges.grounded`, `edges.lineage` (`db.rs:8811`, `:8810`) | **LIVE** |
| provisional contributes zero | `pts.support_status = 'supported'` | `page_truth_state` (`claim_identity.rs:279`) | **BLOCKED** (§0) |
| pairs come from supported claim revisions | join through `page_version_claims` (`claim_identity.rs:215`) at the page's current version | `page_version_claims`, `claims` | **BLOCKED** (§0) |
| collapse through the independence group | `pr.independence_group_id` | `provenance_roots.independence_group_id` (`db.rs:8792`) | **LIVE** |
| candidate is the current supported version | `pts.page_version = p.version AND pts.support_status='supported'` | `page_truth_state` | **BLOCKED** (§0) |
| precomputed bounded adjacency | — | nothing | **PR-A-new** (§8) |
| pair statistics | — | nothing | **PR-A-new** (§8) |

**Co-citation input set.** A pair is formed between two *pages* by the
independence groups that co-support them:

```sql
SELECT DISTINCT pr.independence_group_id, pvc.page_id
  FROM edges e
  JOIN provenance_roots pr ON pr.root_id = e.root_id
  JOIN page_version_claims pvc ON pvc.claim_revision_id = e.src_id
  JOIN page_truth_state pts ON pts.page_id = pvc.page_id
 WHERE e.edge_type = 'supports'
   AND e.src_kind = 'claim_revision'         -- (claim_identity.rs:432)
   AND e.lineage  = 'evidence'               -- (claim_identity.rs:433)
   AND e.valid_until IS NULL
   AND e.grounded = 1
   AND e.root_id IS NOT NULL
   AND pr.status = 'active'
   AND pr.root_kind <> 'generated'
   AND pts.support_status = 'supported'      -- BLOCKED
   AND pts.page_version = pvc.page_version   -- exact current revision
```

**Direct.** A one-hop `relates` edge meeting the eligibility spine, read through
the PR-A precomputed adjacency (§8), capped at 64 rows per endpoint. Binary:
`1` if any eligible edge exists, else `0`.

**Common-neighbor.** Bounded weighted Jaccard over the same adjacency, capped at
64 neighbour rows per endpoint:
`|N(a) ∩ N(b)| / |N(a) ∪ N(b)|`, both sets already truncated to 64.

> **Decision S0-86 — the caps are applied before the Jaccard, not after.** A
> Jaccard computed over full neighbour sets and then truncated is a different
> number from one computed over truncated sets, and only the second is
> computable within the 512-row budget. Stating it prevents an implementation
> from "improving" the estimate into a budget violation.

**Kind affinity** (`R-KIND-*`):

| Target kind | Value | Rule |
|---|---|---|
| entity | `1` — **only** with a grounded entity relation, else `0` | `R-KIND-ENTITY` |
| concept | `1` | `R-KIND-CONCEPT` |
| authored | `0.5`, and **review-only** | `R-KIND-AUTHORED` |
| source / overview | excluded from ordinary attachment entirely | `R-KIND-EXCLUDED` |

> **Decision S0-87 — "review-only" means an authored target never
> auto-attaches regardless of score.** It is a gate applied after scoring, not a
> score cap: with `direct=1, cn=1, kind=0.5` the score is already above `0.50`
> (§1.1), so a cap would not stop it. The routing is to the review queue.

---

## 2. The NPMI estimator

### 2.1 The four cells

For a candidate pair of pages `(A, B)`, over independence groups `g`:

| Cell | Meaning |
|---|---|
| `n11` | groups supporting both A and B |
| `n10` | groups supporting A but not B |
| `n01` | groups supporting B but not A |
| `n00` | groups supporting neither (within the space's eligible group universe) |

Each cell is a **decayed** sum, not a count:

```
contribution(g) = hub_weight(g) * 0.5 ^ (age_days(g) / 180)
```

where `age_days` is measured from the group's most recent contributing root's
`provenance_roots.created_at` (`crates/wenlan-core/src/db.rs:8794`), and
`hub_weight` is §3's `64/d`.

Decay reference values (`R-DECAY-HALFLIFE = 180 days`):

| age | factor |
|---|---|
| 0 d | 1.0000 |
| 90 d | 0.7071 |
| 180 d | 0.5000 |
| 360 d | 0.2500 |
| 540 d | 0.1250 |

### 2.2 The estimator

With `alpha = 0.5` (`R-NPMI-ALPHA`) added to all four cells:

```
ñ11 = n11 + α   ñ10 = n10 + α   ñ01 = n01 + α   ñ00 = n00 + α
Ñ   = ñ11 + ñ10 + ñ01 + ñ00            (= N + 2.0)

p11 = ñ11 / Ñ
p1• = (ñ11 + ñ10) / Ñ
p•1 = (ñ11 + ñ01) / Ñ

PMI  = ln( p11 / (p1• · p•1) )
NPMI = PMI / ( -ln p11 )
co_citation = max(0, NPMI)                    -- positive clip
```

### 2.3 Worked numbers

| case | `n11,n10,n01,n00` | `p11` | `PMI` | `NPMI` | clipped |
|---|---|---|---|---|---|
| strong | 8, 2, 2, 988 | 0.0084830 | +4.2540 | +0.8919 | 0.8919 |
| weak | 1, 10, 10, 979 | 0.0014970 | +2.3454 | +0.3606 | 0.3606 |
| at the floor | 3, 1, 1, 995 | 0.0034930 | +4.9436 | +0.8739 | 0.8739 |
| **never co-occurring** | **0, 12, 12, 976** | **0.0004990** | **+1.0867** | **+0.1429** | **0.1429** |

The last row is a finding, not an illustration — see F1 in §12.

### 2.4 The 3-group raw-support floor

> **Decision S0-88 — the floor (`R-NPMI-FLOOR = 3`) is evaluated on the count of
> **distinct independence groups** co-supporting the pair, **undecayed and
> unsmoothed**; below 3, `co_citation = 0` and no NPMI is computed.**
>
> Three separate choices, each load-bearing:
>
> - **Distinct groups, not pages or roots.** This is what makes D1's collapse
>   rules bite: chunks and mirrors of one document collapse to one group
>   (`provenance_roots.independence_group_id`, `db.rs:8792`), so a single source
>   split into ten chunks cannot manufacture a floor-clearing pair.
> - **Undecayed.** The floor is a statement about how much independent evidence
>   ever existed, not about how fresh it is. Applying decay first would let a
>   pair with ample old evidence silently drop below the floor and re-enter it
>   on any new touch, flapping the score.
> - **Unsmoothed.** Smoothing exists to make the estimator well-defined, not to
>   satisfy the floor. `0 + 0.5` must never read as support.

The floor is also the **only** thing standing between the estimator and F1's
positive-score-for-zero-co-occurrence artifact. That makes it a safety
mechanism rather than a noise filter, which is a materially different thing to
review and to test.

---

## 3. Hub bounds

| Constant | Value | ID |
|---|---|---|
| hub degree threshold | 64 pages | `R-HUB-DEGREE` |
| hub weight | `min(1, 64/d)` where `d` = pages the group touches | `R-HUB-WEIGHT` |
| pages forming pairs | deterministic top 64 | `R-HUB-TOPK` |
| max pairs per group | `C(64,2) = 2016` | `R-HUB-MAXPAIRS` |
| adjacency rows per endpoint | 64 | `R-ADJ-CAP` |

Weight reference: `d=32 → 1.0`, `d=64 → 1.0`, `d=128 → 0.5`, `d=5000 → 0.0128`.

> **Decision S0-89 — the deterministic top-64 is ordered by
> `(support_recency DESC, page_id ASC)`, and the tiebreak is `page_id` because
> it is the only total order available that does not move.** Ordering by score
> would be circular (the selection feeds the score). Ordering by recency alone
> is not total — two pages supported in the same second tie, and a
> nondeterministic tiebreak means the same group produces different pair sets on
> two runs, which breaks §5's incremental-equals-full oracle *silently*, in a way
> that looks like a genuine incremental bug.

`2016 = C(64,2)` is arithmetic, not an independent constant: it follows from
`R-HUB-TOPK`. G6 asserts it as a bound to catch an implementation that caps
weight without capping selection.

---

## 4. Candidate retrieval

| Constant | Value | ID |
|---|---|---|
| candidates returned | ≤ 32 | `R-CAND-CAP` |

Every candidate must be the current M5-`supported` version (**BLOCKED**, §0) and
must not be a source or overview page (`R-KIND-EXCLUDED`).

> **Decision S0-90 — the 32-cap is applied at the query, and a truncated
> candidate set is recorded in the receipt as truncated.** D7's closing rule
> (delay is legal, silent parking is not) applies here too: an attachment
> decision made over a silently truncated candidate set is a decision whose
> inputs cannot be reconstructed. The receipt carries `candidates_considered`
> and `candidates_truncated`.

---

## 5. The incremental-equals-full oracle

D9: *"Pair statistics are maintained incrementally and must equal full
recomputation."*

> **Decision S0-91 — the oracle is stated as a testable equality over a
> normalized snapshot, and it is checked after a mutation sequence, not at a
> single point.**
>
> - **What state.** The full pair table, normalized to
>   `sorted_set((page_a, page_b, n11, n10, n01, n00))` with `page_a < page_b`
>   lexicographically, at a fixed relevance generation.
> - **Compared how.** Byte equality of `m6_digest("m6-pairstats-v1", …)` over
>   that normalized set (artifact 4 §2), not a per-row float tolerance. The
>   cells are sums of `f64` decay factors, so a tolerance-based comparison would
>   hide exactly the accumulation-order bugs the oracle exists to catch —
>   see S0-92.
> - **When.** After each of: root activation, root retraction, root
>   reactivation, page support gain, page support loss, community rebinding,
>   page deletion, and space move. Each one individually, and then a randomized
>   interleaving of all eight.
>
> The negative control matters as much as the positive one: an implementation
> that recomputes fully on every mutation trivially passes. The test must assert
> that the incremental path did **not** perform a full recomputation, by
> asserting the row-visit counter (§7) stayed within the incremental bound.

> **Decision S0-92 — decayed cell sums are accumulated in a fixed order
> (`independence_group_id ASC`) and rounded to 9 decimal places before the
> digest.** Floating-point addition is not associative, so an incremental path
> that adds contributions in arrival order and a full recompute that adds them
> in scan order will differ in the last bits, and the oracle would fail for a
> reason that has nothing to do with correctness. Fixing the order makes the
> equality exact; the rounding absorbs the last-bit difference between "add then
> decay" and "decay then add".

---

## 6. The attachment snapshot and CAS

D9's septuple, captured before committing an automatic attachment:

| # | Field | Source | Status |
|---|---|---|---|
| 1 | `target_page_version` | `pages.version` | LIVE |
| 2 | `target_support_version` | `page_truth_state.page_version` (`claim_identity.rs:281`) | BLOCKED |
| 3 | `relevance_generation` | PR-A counter | PR-A-new |
| 4 | `community_generation` | `space_graph_state.published_generation` (`db.rs:10475`) | LIVE |
| 5 | `dependency_generation` | `space_graph_state.grouping_generation` (`db.rs:10474`) | LIVE |
| 6 | `active_root_set_digest` | `m6_digest` over sorted `(root_id, status)` | LIVE |
| 7 | `candidate_set_digest` | `m6_digest` over the sorted candidate page IDs | PR-A-new |

> **Decision S0-93 — field 1 is the pair `(version, source_revision)`, per
> artifact 7's S0-70.** The same blind spot applies: `link_page_evidence`
> advances `source_revision` while leaving `version` fixed
> (`crates/wenlan-core/src/db.rs:45248`-`:45253`), so a version-only snapshot
> cannot see an evidence change between ranking and commit — which is precisely
> one of the five events D9 requires to write nothing and requeue.

The finalizer CASes all seven, re-verifies the target is the current
M5-supported version, and in one short transaction writes the attachment and
dependency state, page history/changelog where the canonical write seam requires
them, stale/refresh enqueue state, the operation receipt
(`operation_receipts`, `crates/wenlan-core/src/db.rs:8217`, PK
`(caller_id, operation_id)` at `:8223`), and lease completion.

> **Decision S0-94 — "writes nothing and requeues" is implemented as a CAS
> failure that returns the job to `queued`, never as a retry inside the
> transaction.** A retry inside the transaction would re-rank against state read
> under a lock held across the ranking work, which artifact 2's machine E and the
> repo's own async rule both forbid.

---

## 7. The query budget and its instrumentation

| Constant | Value | ID |
|---|---|---|
| indexed queries per route evaluation | ≤ 4 | `R-BUDGET-QUERIES` |
| rows materialized per route evaluation | ≤ 512 | `R-BUDGET-ROWS` |
| **index entries visited per route evaluation** | **≤ 2,176** | **`R-BUDGET-VISITS`** |
| wall time per route evaluation | ≤ 50 ms, **hard** | `R-BUDGET-MS` |

D9 is explicit that *"`LIMIT` in SQL text alone is not proof of visited work."*

> **Decision S0-95 *(amended in rev 2, finding 11 — number kept)* — four
> instruments, and the normative proof of *visited work* is SQLite's own
> statement counters, not decoded rows and not `EXPLAIN QUERY PLAN`.**
>
> 1. **A process-side decoded-row counter**, incremented once per row decoded
>    from a `libsql::Rows` cursor. Normative for `R-BUDGET-ROWS`, which is a
>    *materialization* budget — D9's word is "materializes".
> 2. **A query counter**, one per `query`/`execute` call in the evaluation, for
>    `R-BUDGET-QUERIES`.
> 3. **`SQLITE_STMTSTATUS_VM_STEP`** read per statement, summed across the four,
>    normative for the new `R-BUDGET-VISITS`; and **`SQLITE_STMTSTATUS_FULLSCAN_STEP`
>    asserted equal to zero** on every one of them.
> 4. **`EXPLAIN QUERY PLAN` assertions** that each query uses the named index
>    from §8 and contains no `SCAN` over `edges`, `pages`, or `provenance_roots`.
>
> **Why rev 1 was wrong here.** It made instrument (1) normative for the bound
> and then said, in its own next sentence, that instrument (1) "can be satisfied
> by a query that scans a million rows and returns 512" — naming the defect and
> adopting it in the same decision. It then leaned on `EXPLAIN QUERY PLAN` to
> close the gap, which it cannot: EQP reports the *plan*, not the *work*. A plan
> that correctly uses an index range scan over a 5k-degree hub is a passing EQP
> assertion and an unbounded traversal. D9 anticipated exactly this — *"`LIMIT` in
> SQL text alone is not proof of visited work"* — and G6 asks for *"instrumented
> row visits"* (`gp@wenlan-app:613`-`:614`), which is a runtime counter or it is
> nothing.
>
> **What suffices, concretely, and what it costs.** SQLite exposes real visit
> counters through `sqlite3_stmt_status`: `SQLITE_STMTSTATUS_VM_STEP` (4) scales
> with work actually performed, and `SQLITE_STMTSTATUS_FULLSCAN_STEP` (1) is
> non-zero exactly when a full scan happened. They are reachable from this
> workspace, but not from the API it currently uses: `libsql::Statement`
> (`libsql` 0.9.30, the workspace pin) exposes no status accessor, while
> `libsql_sys::Statement::get_status(status: i32) -> i32` wraps
> `sqlite3_stmt_status` directly. `libsql-sys` is **not** a direct dependency
> today, so PR-A must add it — pinned to the identical `0.9.30`, because this
> crate already carries a warning about exactly this hazard: rusqlite is
> deliberately on `buildtime_bindgen` rather than `bundled` since *"having two
> bundled SQLite builds in the same binary causes libsql's thread-mode assertion
> to fail at runtime"* (`crates/wenlan-core/Cargo.toml:63`-`:67`). A second
> SQLite arriving through a version-skewed `libsql-sys` would reproduce that
> failure. If PR-A finds it cannot take the dependency, the honest consequence is
> that G6's visit clause is not gated and must be raised as such — not that
> decoded rows quietly become the proof again.

The four queries are exactly: (a) candidate retrieval, (b) adjacency for the
source endpoint, (c) adjacency for the candidate endpoints, (d) pair statistics
for the candidate pairs.

### 7.1 The 32 × 64 blowup, and why only the visit counter sees it (finding 11)

Query (c) is where the frozen constants collide, and rev 1 did not do the
arithmetic. Candidate retrieval returns **at most 32** pages
(`gp@wenlan-app:284`) and adjacency is capped at **64 rows per endpoint**
(`gp@wenlan-app:287`). Fetching adjacency for the candidate endpoints is therefore
`32 × 64 = 2,048` rows in the worst case — **four times** `R-BUDGET-ROWS`, which
is 512. Two frozen numbers and one derived number that cannot all hold.

> **Decision S0-156 *(rev 2, finding 11)* — query (c) aggregates in SQLite and
> returns one row per candidate, so the evaluation materializes ~160 rows while
> visiting up to 2,048 index entries. `R-BUDGET-VISITS` is set at 2,176 and is
> the bound that actually constrains the hub.** Common-neighbour needs
> `|N(source) ∩ N(candidate)|` — a *count*, not the rows. Query (c) is
>
> ```sql
> SELECT endpoint_id, count(*) FROM m6_adjacency
>  WHERE space = ?1 AND endpoint_id IN (<the ≤32 candidates>)
>    AND neighbor_id IN (<the ≤64 source neighbours>)
>  GROUP BY endpoint_id
> ```
>
> which materializes at most 32 rows. Whole-evaluation materialization is then
> 32 (a) + 64 (b) + 32 (c) + ≤32 (d) = **≤ 160**, comfortably inside 512, and the
> 512 budget stops being the binding constraint on anything.
>
> **This is the exact case that makes finding 11's two halves one insight.** The
> aggregation moves the 2,048-row traversal *inside* SQLite, where a decoded-row
> counter cannot see it: instrument (1) reports 32 and passes while the engine
> walks 2,048 index entries. `EXPLAIN QUERY PLAN` also passes — it is a correct
> indexed range scan. Only `SQLITE_STMTSTATUS_VM_STEP` moves. A design that
> respects the materialization budget by pushing work into the engine is exactly
> the design that makes visit instrumentation load-bearing rather than belt-and-
> braces, and it is why D9 wrote *"`LIMIT` in SQL text alone is not proof of
> visited work"* rather than trusting a row count.
>
> The 2,176 figure is `2,048` for query (c) plus `32 + 64 + 32` for the other
> three, i.e. the arithmetic worst case with no slack — deliberately, so that a
> regression that adds one unindexed lookup fails rather than fitting in a
> margin. `R-BENCH-HUB` asserts it on the 5,000-degree hub, where every candidate
> genuinely has a full 64-row adjacency and the worst case is real rather than
> theoretical.

---

## 8. Indexes PR-A must create

Against the two PR-A-new tables plus the existing `edges`:

| # | Index | Serves |
|---|---|---|
| I1 | `m6_pair_stats(space, page_a, page_b)` PRIMARY KEY | query (d) |
| I2 | `m6_pair_stats(space, page_a, updated_generation)` | incremental invalidation |
| I3 | `m6_adjacency(space, endpoint_kind, endpoint_id, rank)` PRIMARY KEY | queries (b), (c); `rank` is the deterministic 1..64 slot from S0-89 |
| I4 | `m6_adjacency(space, neighbor_id)` | reverse invalidation on root retraction |
| I5 | `page_truth_state(support_status, page_id)` | candidate retrieval; extends the existing status-only index (`claim_identity.rs:301`-`:302`) |

The existing `idx_edges_active_grounded_space_type`
(`crates/wenlan-core/src/db.rs:8827`-`:8828`, `ON edges(space, edge_type) WHERE
valid_until IS NULL AND grounded = 1`) already matches the eligibility spine and
serves the adjacency *rebuild*, not the route evaluation.

> **Decision S0-96 — the route evaluation never touches `edges` directly; it
> reads only `m6_adjacency` and `m6_pair_stats`.** This is what makes the 4-query
> / 512-row budget achievable against a 5000-degree hub: the bound is enforced by
> the precomputed table's shape (64 rows per endpoint, by construction of I3),
> not by a `LIMIT` on a query over a table where the hub's 5000 rows exist.

---

## 9. The Stage-0 representative corpus

D9's budget is stated *"on the Stage-0 representative 100k-memory/5k-page
corpus."* A benchmark whose corpus is not reproducible is not a gate.

> **Decision S0-97 — the corpus is generated by a seeded deterministic recipe,
> committed as a script, and identified by a digest that the benchmark records.**
>
> **Composition** (`R-CORPUS-*`):
>
> | Element | Value | ID |
> |---|---|---|
> | memories | 100,000 | `R-CORPUS-MEM` |
> | pages | 5,000 | `R-CORPUS-PAGE` |
> | spaces | 8, with sizes following a 40/20/12/10/8/5/3/2 % split | `R-CORPUS-SPACE` |
> | independence groups | 12,000 | `R-CORPUS-GROUP` |
> | group degree distribution | Zipf, exponent 1.1, truncated at 5,000 | `R-CORPUS-ZIPF` |
> | explicit hub groups | 3, at degree exactly 5,000 / 1,024 / 65 | `R-CORPUS-HUB` |
> | pages per group (mean) | 8 | `R-CORPUS-FANOUT` |
> | root age distribution | uniform over 0–720 days | `R-CORPUS-AGE` |
> | generated-root fraction | 15% | `R-CORPUS-GENFRAC` |
> | retracted-edge fraction | 5% | `R-CORPUS-RETFRAC` |
> | RNG seed | `0x6D36_0000` | `R-CORPUS-SEED` |
>
> **Why these three hub degrees.** 5,000 is G6's named worst case; 1,024 exercises
> `64/d` well inside the cap; **65 is the off-by-one boundary** — the smallest
> degree at which top-64 selection actually truncates, and therefore the case an
> implementation with `>` instead of `>=` gets wrong.
>
> **Why 15% generated and 5% retracted.** Both must be non-trivial fractions or
> the "contributes zero" clauses are tested only by unit tests and never at
> scale, where an index that silently ignores the predicate would still look
> fast and would produce wrong numbers.
>
> **Identification.** The recipe writes a manifest containing the constants above
> and its own digest; every benchmark result records that digest. A result whose
> corpus digest does not match the committed manifest is not comparable and must
> not be cited — the same discipline `app/eval/AGENTS.md` already applies to eval
> numbers.

---

## 10. Benchmark pass/fail

| Metric | Pass | Fail | ID |
|---|---|---|---|
| route evaluation **max** | ≤ 50 ms | any evaluation > 50 ms | `R-BENCH-MAX` |
| route evaluation p50 | reported | — (tracking only) | `R-BENCH-P50` |
| route evaluation p99 | reported | — (tracking only) | `R-BENCH-P99` |
| queries per evaluation | ≤ 4, always | any evaluation > 4 | `R-BENCH-Q` |
| rows per evaluation | ≤ 512, always | any evaluation > 512 | `R-BENCH-ROWS` |
| 5,000-degree hub evaluation | within all of the above | any breach | `R-BENCH-HUB` |

> **Decision S0-98 *(amended in rev 2, finding 11 — number kept)* —
> `R-BUDGET-MS` is a **hard** 50 ms limit. The p50 and p99 figures are reporting,
> not the gate.** Rev 1 reinterpreted it as a p99 on the reasoning that a hard
> maximum is flaky on a laptop under load. That reasoning is about measurement
> conditions and the fix belongs there, not in the constant: D9 says *"completes
> within `50 ms`"* (`gp@wenlan-app:293`) and G6 calls it *"the frozen `50 ms`
> route budget"* (`gp@wenlan-app:613`). **Stage 0 does not get to reinterpret a
> constant the contract froze** — a Stage-0 artifact that relaxes a frozen number
> because the relaxed version is easier to measure is doing the thing S0-99
> forbids one paragraph later, with the extra problem that a p99 over a
> thousand-evaluation run tolerates ten breaches by construction, and hub
> pathology is a tail event.
>
> Flakiness is answered by specifying the measurement instead: the gate is the
> **maximum** over the Stage-0 representative 100k-memory/5k-page corpus, warm
> cache, single evaluation at a time, no concurrent refinery turn — the same
> fixture conditions D9 names. `R-BENCH-P50` and `R-BENCH-P99` remain in the
> benchmark table as reported figures for tracking drift; neither is a pass
> condition on its own, and `R-BENCH-MAX` is.

> **Decision S0-99 — a benchmark failure stops for a Sol-reviewed contract
> amendment and may not be fixed by tuning a constant.** This is D9's own rule
> (*"production may not silently tune them"*), restated here because a benchmark
> limit is the single most tempting constant to adjust. Every constant in this
> document has an ID precisely so an amendment names what it changes.

---

## 11. Hysteresis and the tie-break

M6 reuses M4's thresholds, which exist in the tree:

```rust
// crates/wenlan-core/src/community_routing.rs:7-8
pub const COMMUNITY_ROUTE_ASSIGN_THRESHOLD: f64 = 0.50;
pub const COMMUNITY_ROUTE_DROP_THRESHOLD: f64 = 0.30;
```

with the three-way decision at
`crates/wenlan-core/src/community_routing.rs:55`-`:78` and its exact-boundary
test at `:138`.

| Rule | Value | ID |
|---|---|---|
| new auto-attachment | score `>= 0.50` | `R-HYST-ASSIGN` |
| top-two margin | `>= 0.10` | `R-HYST-MARGIN` |
| existing attachment holds | score `>= 0.30` | `R-HYST-HOLD` |
| additional requirement | direct signal **or** qualified co-citation | `R-HYST-SIGNAL` |

M4 has no margin rule — `decide_page_community_route`
(`crates/wenlan-core/src/community_routing.rs:55`) takes a single
`best_candidate` and never sees the runner-up. `R-HYST-MARGIN` is therefore
**PR-A-new** even though `R-HYST-ASSIGN` and `R-HYST-HOLD` are LIVE.

> **Decision S0-100 — "qualified co-citation" means a co-citation term computed
> from a pair that cleared the 3-group floor, not merely a non-zero
> co-citation.** After the positive clip, an unqualified co-citation can be
> non-zero for a pair the floor would have rejected (F1). Defining "qualified" as
> "the floor was cleared" is the only reading under which the rule adds safety
> rather than restating `> 0`.

> **Decision S0-101 — an "exact tie" for the embedding tie-break is bit-equality
> of the two scores' `f64` representations after the §5 rounding to 9 decimal
> places.** Anything looser is a threshold in disguise: a tie defined as
> `|a - b| < ε` lets embeddings decide cases separated by up to `ε`, which is
> exactly the "cannot cross a threshold" prohibition. With rounding already
> applied for the oracle, bit-equality is well-defined and cheap.
>
> Corollary G6 must assert: the tie-break may reorder two candidates, and may
> never change how many candidates are above `0.50`, below `0.30`, or within
> `0.10` of the leader.

---

## 12. Findings

**F1 — `alpha=0.5` four-cell smoothing assigns a *positive* co-citation score to
pairs that have never co-occurred, whenever the pair is rare.** With
`n11=0, n10=n01=12, n00=976`, the estimator returns `NPMI = +0.1429` (§2.3). The
mechanism: the smoothed co-occurrence floor `α = 0.5` exceeds the independence
expectation `Ñ · p1• · p•1 ≈ 0.169` for rare marginals, so the pair reads as
positively associated on the strength of the smoothing alone. The positive clip
does not help — the value is already positive. Solving
`α·Ñ > (α+m)²` for the symmetric case shows the artifact persists until each
group appears in roughly `√(α·Ñ)` pages: about 22 pages at `Ñ ≈ 1002`.

The 3-group floor (S0-88) closes this completely, because `n11 = 0` never clears
a floor of 3. **The finding is therefore not "the estimator is broken" but "the
floor is the only thing that makes the estimator safe, and nothing in the
contract says so."** A future amendment that lowered the floor to 1, or applied
it to decayed counts, or read it as "3 pages" rather than "3 groups", would
reintroduce the artifact silently. **G6 must include the never-co-occurring rare
pair as a named negative control** — it is a known-negative in D9's own sense,
and today's gate list does not name it.

**F2 — the M5 truth substrate is complete, and no `supported` promoter exists.**
§0 in full. The schema, triggers, indexes, and backfill all exist and are well
built; a human-axis writer now exists too (`#418`), but no machine-axis promoter
does. Every D9 clause resting on `supported` is constant-false today.
This is a program-sequencing consequence (S0-85), not a defect in either M5 or
D9, but it means PR-B's genesis shadow would measure exactly zero without a
claim-derivation promoter shipping first.

**F3 — `R-HYST-MARGIN` has no M4 antecedent to reuse.** D9 says "reuse M4
hysteresis", and two of the three thresholds do exist
(`crates/wenlan-core/src/community_routing.rs:7`-`:8`). The top-two margin does
not: `decide_page_community_route` (`:55`) receives one candidate and cannot
compute a margin. An implementer reading "reuse M4 hysteresis" as "call the
existing function" would ship without the margin rule and the tests would pass,
because the existing function's own boundary test (`:138`) does not know about
margins.

**F4 — `edges.edge_type` and the endpoint-kind CHECKs in the base DDL do not
list the M5 values that migration 98 adds.** The base table
(`crates/wenlan-core/src/db.rs:8806`-`:8810`) constrains `src_kind` to
`page|memory|entity|external` and `edge_type` to
`mentions|relates|cites|supports|links`, while the live post-migration schema
accepts `claim_revision` endpoints and the `attests` type
(`crates/wenlan-core/src/db.rs:11737`, used at
`crates/wenlan-core/src/db/claim_identity.rs:432`). Anyone grounding a predicate
by reading the `CREATE TABLE` alone will write a filter against constraints that
no longer hold. Not a bug — a rebuild-migration is the correct SQLite idiom —
but it is a trap for exactly the kind of schema-grounding work this artifact
does, and worth one line in the PR-A notes.

---

## 13. Gate mapping — G6

| G6 clause | Where satisfied |
|---|---|
| no cross-space or known-negative auto-attachment | §1.2 spine (`space = ?`); **plus F1's new negative control** |
| exact score/margin/hysteresis boundary tests | §11, with `R-HYST-MARGIN` marked PR-A-new (F3) |
| incremental pair state equals full recomputation | §5, S0-91 and S0-92 |
| generated/inactive/retracted/legacy-ungrounded contribute zero | §1.2 table, all four **LIVE** |
| `EXPLAIN QUERY PLAN` uses fixed indexes | §7 instrument (4), §8 index list |
| instrumented row visits, not a textual `LIMIT` | §7 instrument (3) and §7.1 — `R-BUDGET-VISITS` via `SQLITE_STMTSTATUS_VM_STEP` |
| no group forms more than 2016 pairs | §3, `R-HUB-MAXPAIRS` |
| candidate retrieval never exceeds 32 | §4, `R-CAND-CAP` |
| a provisional candidate is never retrieved or attached | §1.2 — **BLOCKED**; the clause is currently vacuously true, which is worse than failing, so the test must assert the predicate is *evaluated*, not merely that no provisional page appeared |
| a hub cannot dominate after 64/d weighting | §3, `R-HUB-WEIGHT` |
| 5k-degree hub within all budgets | §10, `R-BENCH-HUB`; achievable only via S0-96 |

---

## 14. Constants index

Every constant has an ID so an amendment has an address.

`R-W-COCITE` 4 · `R-W-DIRECT` 3 · `R-W-CNEIGH` 1.5 · `R-W-KIND` 1 ·
`R-W-DIVISOR` 9.5 · `R-NPMI-ALPHA` 0.5 · `R-NPMI-FLOOR` 3 groups ·
`R-DECAY-HALFLIFE` 180 d · `R-HUB-DEGREE` 64 · `R-HUB-WEIGHT` 64/d ·
`R-HUB-TOPK` 64 · `R-HUB-MAXPAIRS` 2016 · `R-ADJ-CAP` 64 · `R-CAND-CAP` 32 ·
`R-KIND-ENTITY` 1 · `R-KIND-CONCEPT` 1 · `R-KIND-AUTHORED` 0.5 ·
`R-KIND-EXCLUDED` source/overview · `R-HYST-ASSIGN` 0.50 · `R-HYST-MARGIN` 0.10 ·
`R-HYST-HOLD` 0.30 · `R-HYST-SIGNAL` direct-or-qualified ·
`R-BUDGET-QUERIES` 4 · `R-BUDGET-ROWS` 512 · `R-BUDGET-MS` 50 ·
`R-BENCH-MAX` 50 ms hard · `R-BENCH-P50` reported · `R-BENCH-P99` reported · `R-CORPUS-MEM` 100k ·
`R-CORPUS-PAGE` 5k · `R-CORPUS-SPACE` 8 · `R-CORPUS-GROUP` 12k ·
`R-CORPUS-ZIPF` 1.1 · `R-CORPUS-HUB` 5000/1024/65 · `R-CORPUS-FANOUT` 8 ·
`R-CORPUS-AGE` 0–720 d · `R-CORPUS-GENFRAC` 15% · `R-CORPUS-RETFRAC` 5% ·
`R-CORPUS-SEED` `0x6D36_0000`.

---

## 15. Decisions introduced here

`S0-85` Stage 0 does not weaken the `supported` predicate; a promoter is a PR-B prerequisite ·
`S0-86` caps apply before the Jaccard, not after ·
`S0-87` authored targets are review-only as a post-scoring gate ·
`S0-88` the 3-group floor is distinct groups, undecayed, unsmoothed ·
`S0-89` deterministic top-64 ordered by `(support_recency DESC, page_id ASC)` ·
`S0-90` truncated candidate sets are recorded as truncated ·
`S0-91` the oracle is a digest equality over a normalized snapshot, after eight mutation kinds ·
`S0-92` decayed sums accumulate in fixed order and round to 9 dp ·
`S0-93` snapshot field 1 is the `(version, source_revision)` pair ·
`S0-94` CAS failure requeues; never retry inside the transaction ·
`S0-95` three required instruments for the budget proof ·
`S0-96` the route evaluation never touches `edges` directly ·
`S0-97` the corpus is a seeded deterministic recipe identified by digest ·
`S0-98` *(amended rev 2)* 50 ms is a hard maximum; p50/p99 are reporting only ·
`S0-99` a benchmark failure is an amendment, never a tuning ·
`S0-100` "qualified co-citation" means the 3-group floor was cleared ·
`S0-101` an exact tie is bit-equality after 9-dp rounding.

**Added in rev 2:** `S0-156` query (c) aggregates in SQLite; `R-BUDGET-VISITS` at
2,176 is the bound that constrains the hub, and it is the only instrument that
sees the 32 × 64 traversal.
