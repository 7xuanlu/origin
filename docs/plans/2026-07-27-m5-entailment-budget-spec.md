# M5 Stage 0 — entailment, cache, and budget spec

Date: 2026-07-27. Binding for M5 PR-A. Implements D6 of
`2026-07-27-kg-m5-goal-prompt.md`.

**Production constants are NOT set by this document.** It freezes the corpus,
the method, the pass/fail rule, and provisional safety ceilings. PR-A implements
and runs the benchmark; the final constants are locked by independent review
(Sol and Fable) before shadow jobs are enabled or PR-A merges. Missing or failed
benchmark evidence is a **STOP**, not permission to pick convenient numbers.

## 1. The judge is separate

The support judge is a distinct constrained pass, never the synthesizing model
in a second role. Reused M3g safety seams (`edge_grounding.rs`), which are
correct and stay:

- schema-constrained output,
- untrusted-input fencing,
- fail-closed parsing,
- bounded calls per tick,
- no tool access,
- **all model work outside every SQLite transaction and DB mutex**
  (`edge_grounding.rs:100` states this invariant explicitly).

M3g also refuses to judge under a non-pinned backend
(`skipped_non_on_device`, `edge_grounding.rs:171`) because a cloud model would
score a fixed threshold differently. M5 keeps that refusal.

## 2. The M3g cache key is NOT reused

Verified: M3g's version tag is

```rust
// edge_grounding.rs:154
format!("{EDGE_GROUNDING_ENTAILMENT_PROMPT_VERSION}|{model_id}")
```

`prompt_version | model_id`. It carries **no model version** and **no content
digests**, because it tags a *cursor*, not a per-item cache. Adopting it as an
M5 cache key would mean two different weight-sets of the same `model_id` share
cached scores, which is exactly the conflation D6 forbids.

M5's cache identity is the full five-part key:

```
(claim_text_digest, source_span_digest, model_id, model_version, prompt_version)
```

Changing **any** component misses. Consequences, all deliberate:

- scores from different model versions are **never compared under one
  threshold**;
- a prompt edit invalidates every cached score;
- a claim-text or span change misses, so a cached score can never be attributed
  to text the judge never saw.

`claim_text_digest` is the `canonical_text_digest` of artifact 1 §2.
`source_span_digest` is the anchor `span_digest` of artifact 1 §3. Reusing those
exact digests means an invalidated anchor cannot silently hit cache.

### Model metadata recorded per row

`model_id`, `model_version`, `prompt_version`, `backend`, `threshold_at_write`,
`scored_at`. `threshold_at_write` is recorded so a later threshold change is
detectable per row rather than silently reinterpreting stored scores.

## 3. Threshold

One threshold per `(model_id, model_version, prompt_version)`. There is no
global threshold constant. M3g's `EDGE_GROUNDING_ENTAILMENT_THRESHOLD = 0.5`
(`edge_grounding.rs:51`) is scoped to its own pinned model and is **not**
inherited by M5.

A claim revision is supported only when at least one active support edge scores
**at or above** the threshold for *its own* key triple. Below-threshold, absent,
and unparseable all mean `provisional` (artifact 2 §1).

## 4. Retry and lease

- Durable derivation jobs with leases. A lease has an owner, an expiry, and an
  attempt count.
- Expired leases are reclaimable. Reclaim does **not** discard cached scores —
  the cache is keyed by content, not by job.
- Attempts are capped. A job exceeding the cap is parked with a durable reason,
  and its page stays `provisional`. Parked is a visible terminal state, never a
  silent retry loop.
- **Partial results never publish** (artifact 2, row 6). A run that judges some
  claims and fails others requeues whole.
- Retry is idempotent through the D8 finalizer CAS on
  `(page_version, dependency_generation, active_root_set_digest)`. Any CAS miss
  writes zero visible rows and requeues.

## 5. Indexes

| Index | Serves |
|---|---|
| cache PK on the full five-part key | exact hit/miss |
| `(model_id, model_version)` | invalidate a model version wholesale |
| `(prompt_version)` | invalidate a prompt version wholesale |
| `(scored_at)` | retention reaping |
| jobs `(status, lease_expires_at)` | claim the next job, reclaim expired |
| jobs `(page_id, page_version)` | dedupe per page version |

Plus the four `edges` partial indexes from artifact 3 §6, which carry the
support lookups this spec's budgets measure.

## 6. Budget dimensions

Frozen dimensions, with **provisional safety ceilings** in force until the
benchmark replaces them. Provisional ceilings are chosen to be safe, not
optimal; PR-A may lower them on evidence and may raise them only on evidence.

| Dimension | Provisional ceiling | Rationale |
|---|---|---|
| active claims per page | 200 | above this, derivation parks rather than degrades |
| support candidates per claim | 8 | bounded fan-out |
| judge items per batch | 25 | matches M3g's validated per-tick cap (`edge_grounding.rs:44`) |
| model calls per page per cycle | 200 | claims × 1 pass, cache misses only |
| tokens per page per cycle | 250k | throughput guard |
| retry attempts per job | 5 | then park |
| lease duration | 10 min | > p99 page derivation |
| cache retention | 90 days since last hit | bounded growth |

## 7. Benchmark

**Corpus (frozen at Stage 0):** a representative synthetic corpus of
**100k memories / 5k pages**, generated by a seeded, checked-in generator so any
reviewer reproduces the same bytes. Page-size distribution matches the observed
production distribution rather than a uniform average, because tail pages, not
median pages, are what breach a budget.

Plus a hand-built **judge accuracy set**: labelled (claim, span, entails?)
triples including the adversarial cases from artifact 1 §7 — negation (N1) and
quantifier change (N8) — which a lenient judge will wrongly call supported.

**Method:**

- run on the pinned on-device model, single machine, cold and warm cache;
- report p50/p95/p99, never a mean — a mean hides the tail that breaks a budget;
- record query counts and lock-hold times separately from wall time, since the
  binding constraint is DB mutex occupancy, not throughput;
- N≥3 runs with stddev for any number that leaves this repo (repo eval-citation
  discipline);
- environment-stamped.

**Measured:**

1. support-status evaluation latency per page (artifact 2 §1 conditions);
2. per-page derivation wall time, cold and warm cache;
3. cache hit rate across a realistic edit sequence;
4. DB mutex hold per operation;
5. query count per operation at 100k/5k;
6. judge precision/recall on the accuracy set.

**Pass/fail:**

| Check | Rule |
|---|---|
| every budget dimension | measured p99 within its ceiling |
| DB mutex hold | no regression against the pre-M5 baseline on the same corpus |
| judge on N1/N8 | must **not** score above threshold — a judge that entails a negation is disqualified regardless of aggregate accuracy |
| benchmark absent or failed | **STOP** |

The N1/N8 rule is a hard veto rather than a contribution to an aggregate score,
because an aggregate lets a judge that is 95% accurate overall still be
systematically wrong on the one class of error that silently marks false claims
supported.

## 8. Mutation checks

| Weakening | Must fail |
|---|---|
| drop `model_version` from the cache key | cross-version hit test |
| drop `prompt_version` | prompt-change hit test |
| drop either digest | changed-text hit test |
| compare scores across model versions under one threshold | §3 test |
| use one global threshold constant | §3 test |
| run model work inside a transaction or under the DB mutex | §1 invariant test |
| publish partial results | artifact 2 row 6 |
| discard cached scores on lease reclaim | §4 test |
| retry forever instead of parking | §4 cap test |
| let a non-pinned backend judge | §1 refusal test |
| accept the benchmark with a mean instead of p99 | §7 method test |
| set production constants without benchmark evidence | §7 STOP rule — review gate |
