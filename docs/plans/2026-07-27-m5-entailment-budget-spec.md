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
(`skipped_non_on_device`, `edge_grounding.rs:175`) because a cloud model would
score a fixed threshold differently. M5 keeps that refusal.

## 2. The M3g cache key is NOT reused

Verified: M3g's version tag is

```rust
// edge_grounding.rs:155, in grounding_version_tag (:154)
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

## 2a. Support-candidate sourcing — including human-authored prose

`supports` targets memory spans (artifact 3 §2). That raises the question the
rest of this spec depends on: **where does evidence for a human-written
sentence come from?**

If the answer were "nowhere," one human-added sentence would make its page
permanently `provisional` — `supported` requires *every* claim supported
(artifact 2 §1) — and automatic readers exclude at page granularity (artifact
4). Users would learn that editing their own wiki removes it from their agent's
context, which trains people to stop curating. For a curation product that is
fatal, and no amount of `human_reviewed` rescues it, because that axis
deliberately does not feed `support_status`.

### The path, from D4 — with D4's actual preconditions

**Only** a save that supplies a valid exact `base_version + base_content_digest`
mints a grounded `human_edit_delta`, computed against the exact immutable
`page_history` snapshot and carrying `operation_id`. Per D4
(`2026-07-27-kg-m5-goal-prompt.md:197`):

| Save shape | Result |
|---|---|
| valid exact base supplied | grounded `human_edit_delta` minted |
| **stale** base supplied | **conflict, nothing written** — no delta |
| base **omitted** | saves prose, mints **no** grounded delta |

The omitted-base case is the one that matters here and the one an earlier draft
of this section erased by saying "human saves mint grounded deltas" flatly. A
no-base save produces prose with **no** evidence, so its claims are
`provisional` and its page cannot reach `supported` until a based save or a
derivation supplies evidence. That is correct, and it must be stated rather than
papered over.

D4 also constrains **which** assertion kinds may act as evidence
(`2026-07-27-kg-m5-goal-prompt.md:190`):

- **`observation`** — may support another claim, as **one independence group**,
  only after exact-delta entailment. This is the only kind that does.
- **`correction`** — requires the exact current target revision and digest; a
  stale target conflicts. Cross-page consequences stage review; they never
  silently rewrite another page.
- **`preference`** — subject-bound, never generalized into factual structure.
- **`speculation`** — retains modality; cannot support an unqualified factual
  claim.
- legacy/missing kind — NULL, unclassified, **non-voting**.

So "human prose is evidence" is false in general and true only for an eligible
`observation` delta from an exact-base save. Any broader reading would let a
preference or a speculation support a factual claim, which D4 forbids outright.

This is not circular. Entailment against the human's own text checks that the
derived claim does not **overreach** what the human actually wrote. A claim that
faithfully restates the sentence is supported — correctly, since
`support_status` means "every claim traces to evidence," not "every claim is
independently verified." A claim that generalizes, strengthens a quantifier, or
drops a hedge fails, which is exactly the N1/N8 class in artifact 1 §7.

Because a human-authored claim rests on exactly one independence group, it can
never accumulate independent corroboration from its own page. That is intended
and must not be worked around by counting the same delta twice.

### Verified gap — PR-A must build this

`provenance_roots.root_kind` permits `human_capture` and `human_edit_delta`
(`db.rs:8898`), but **nothing mints either**. The only production minter is
`acquire_provenance_root("document_ingest", …)` (`edge_grounding.rs:537`); the
two human kinds appear elsewhere only inside a `provenance.rs` unit test.

Two Stage 0 contracts therefore depend on code that does not exist:

1. this section's human-prose evidence path;
2. artifact 3 §5, which requires an attesting root to be `human_capture` or
   `human_edit_delta`.

PR-A must deliver the human-root minter and the span-addressable delta store.
Neither is optional, and neither can be deferred to PR-B: without them,
attestation has no valid source root and every human-edited page is
permanently unsupported.

### Shadow-phase metric (required)

During the PR-A shadow phase, report supported-fraction **by page class** —
distilled, human-edited, human-authored — never as one aggregate. An aggregate
dominated by distilled pages would hide a human-authored class sitting at zero,
which is precisely the failure this section exists to catch. That number gates
PR-C (artifact 7 §4a).

## 3. Threshold

One threshold per `(model_id, model_version, prompt_version)`. There is no
global threshold constant. M3g's `EDGE_GROUNDING_ENTAILMENT_THRESHOLD = 0.5`
(`edge_grounding.rs:51`) is scoped to its own pinned model and is **not**
inherited by M5.

A claim revision is supported only when at least one active support edge scores
**at or above** the threshold for *its own* key triple. Below-threshold, absent,
and unparseable all mean `provisional` (artifact 2 §1).

### Model-version eligibility is ROLLING, not instant

Artifact 2 row 14 demotes a page when its supporting model version becomes
ineligible. If a judge upgrade made the old version ineligible *immediately*,
the entire corpus would flip `provisional` in one step — the migration cliff
recurring at every model bump, with no readiness fence, a mass quarantine of the
projection directory (artifact 7 §4), and a decode-bound re-derivation storm.

Eligibility is therefore a durable per-version state with a drain:

| State | Meaning |
|---|---|
| `active` | new judgments use this version |
| `draining` | no new judgments; **existing scores remain eligible** |
| `retired` | scores no longer count; affected pages demote |

An upgrade moves the old version `active → draining` and the new one to
`active`. Re-derivation proceeds under the new version at the normal drain rate.
The old version moves to `retired` only when no page still depends on it, or
when it is retired **deliberately** — a judge found to be wrong should demote
its corpus immediately, and that path stays available as an explicit operator
action rather than an accident of upgrading.

Per-version thresholds (above) are what make this coherent: scores from the two
versions are never compared, they are simply each valid under their own
threshold while their version is `active` or `draining`.

### Eligibility generation — how bulk demotion stays atomic

Artifact 2 requires demotion in the trigger's own transaction. A retirement or
threshold change affects an unbounded number of pages, so "synchronous **and**
batched" is a contradiction: either one enormous transaction, or a window in
which eligibility has changed while stored rows still read `supported`.

Neither is acceptable, so bulk eligibility changes do **not** rewrite page rows
to take effect. They bump a single durable **eligibility generation**:

- the eligibility table holds `(model_id, model_version, prompt_version) →
  {state, threshold, generation}`;
- one monotonic generation counter covers all of it;
- **every** support-status read joins through that table, so retiring a version
  or raising a threshold changes what every affected page reports **at the
  instant the generation commits** — one row written, no window;
- stored `support_status` is then reconciled to match, batched and bounded,
  purely as a materialization. Because reads already join eligibility, a page
  whose stored value is momentarily stale still **reads** `provisional`.

This is the M4 control-plane shape again — one global monotonic generation, one
central gate — and it is why bulk changes need no unbounded transaction.
Per-edge events (a retraction) stay synchronous as artifact 2 specifies; only
the unbounded class routes through the generation.

### The finalizer must CAS eligibility too

D8's finalizer CASes `(page_version, dependency_generation,
active_root_set_digest)`. None of those move when a model is retired or a
threshold is raised, so a job that began under the old regime can finalize
afterwards and **republish stale support** — a verdict from a retired judge
written as fresh evidence.

The finalizer therefore also captures and re-checks the **eligibility
generation** before commit. A mismatch is a CAS miss: zero visible rows,
requeue, re-judge under the current regime.

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

A corpus is "frozen" only if a reviewer can regenerate it byte-for-byte and
check that they did. Naming a size and a property is not freezing. The following
identifiers are the freeze, and PR-A creates each at these exact paths:

| Artifact | Path | Frozen by |
|---|---|---|
| generator | `crates/wenlan-core/src/eval/m5_bench_corpus.rs` | checked in |
| seed | `M5_BENCH_SEED: u64 = 0x4d35_0001` | a constant, not a flag |
| corpus digest | `crates/wenlan-core/tests/fixtures/m5_bench_corpus.sha256` | asserted by the bench before it runs |
| size-distribution snapshot | `crates/wenlan-core/tests/fixtures/m5_page_size_dist.json` | derived once from a real corpus, checked in, cited by digest |
| judge accuracy set | `crates/wenlan-core/tests/fixtures/m5_judge_accuracy.jsonl` | checked in, digest-asserted |

**Corpus:** **100k memories / 5k pages**, produced by that generator under that
seed. Page sizes are drawn from the snapshot above rather than a uniform
average, because tail pages, not median pages, are what breach a budget. The
snapshot is a checked-in file precisely so "observed production distribution"
names something a reviewer can open — an undefined phrase would let any later
run silently change the shape of the test.

The bench recomputes the corpus digest at startup and **refuses to run** on a
mismatch. A benchmark whose corpus can drift measures nothing across time.

**Judge accuracy set:** labelled `(claim, span, entails?)` triples, including
every adversarial case from artifact 1 §7 — negation (N1), quantifier change
(N8), terminator change (N11), acronym case change (N12) — which a lenient judge
will wrongly call supported.

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

Rows marked **[gate]** are human review gates, not executable tests. They are
listed because they must happen, but they are not teeth — a table that mixes the
two lets a process promise stand in for a failing build. Every unmarked row is
an executable test that goes RED under its weakening.

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
| set production constants without benchmark evidence | **[gate]** §7 STOP rule |
| leave human-authored prose with no evidence path | §2a — a page edited by an **exact-base save whose claims are eligible `observation`s** must be able to reach `supported` |
| mint a grounded delta for a no-base or stale-base save | §2a save-shape table |
| let `preference` or `speculation` support a factual claim | §2a kind table |
| let an unclassified/legacy-kind claim vote | §2a kind table |
| let a human delta support a claim that overreaches it | §2a exact-delta test, N1/N8 class |
| count one human delta as two independence groups | §2a test |
| retire a model version instantly on upgrade | §3 rolling-eligibility test |
| demote pages while the old version is `draining` | §3 test |
| let a support-status read skip the eligibility join | §3 — retire a version, page must read `provisional` immediately |
| omit the eligibility generation from the finalizer CAS | §3 — retire mid-job, finalize must write zero rows |
| rewrite page rows as the mechanism of bulk demotion | §3 — unbounded-transaction / stale-window test |
| report supported-fraction as one aggregate | §2a metric test |
