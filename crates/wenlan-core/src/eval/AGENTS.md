# crates/wenlan-core/src/eval - Rust runner conventions

Applies to agents working under `crates/wenlan-core/src/eval/`. Read alongside root `AGENTS.md` and `app/eval/AGENTS.md` for fixture + artifact context.

---

## runner shapes

Two runner families exist. Choose based on what question you are answering.

### Ephemeral-per-conversation runners

Create a `tempdir` + `MemoryDB::new` + upsert observations per conversation. Each conversation gets a fresh isolated DB. No page distillation runs because pages are a background task that requires a full enrichment pipeline.

Examples: `run_locomo_eval`, `run_locomo_eval_cross_rerank`, and their LME equivalents.

Use when: measuring the published baseline bar, or any retrieval-only benchmark where pages should not be a factor.

### From-cached-DB runners (PR-B)

Accept a pre-seeded `&MemoryDB` for a consolidated scenario DB that already contains memories, KG entities, and distilled pages. Skips ingest entirely. The DB was seeded once via the fullpipeline harness.

Examples: `run_locomo_eval_cross_rerank_from_db`, `run_longmemeval_eval_cross_rerank_from_db`.

Use when: measuring page-channel impact, or any question that requires pages to already exist in the DB.

Tradeoff: consolidated DB introduces cross-conversation noise because all conversations share one DB. Results are comparable to the OFF variant of the same runner, not to the ephemeral-per-conversation bar.

---

## source_id conventions

Source IDs must match exactly between the seeder and the evaluator. A mismatch produces silent zero NDCG with no error.

### LoCoMo

```rust
source_id: format!("locomo_{}_obs_{}", sample.sample_id, i)
```

Used in: ephemeral runner seed paths and the fullpipeline harness. The `run_locomo_eval_cross_rerank_from_db` runner re-derives this mapping from the fixture file, so the cached DB must have been seeded with the same format.

### LongMemEval

```rust
// Use the helper, not a hand-rolled format!:
memory_source_id(question_id, session_idx, turn_idx)
// defined at longmemeval.rs:165-167
```

The helper exists precisely so the format is defined once. Do not inline an equivalent `format!` call elsewhere.

---

## build_*_env stamping

`build_locomo_env` and `build_lme_env` produce a `ReportEnv` struct that stamps every baseline. Key fields:

- `variant`: the `variant_tag` string, a human-readable differentiator for the layered baseline path. As of Task #30, `comparable_env_hash` also includes `flags`, so the hash itself now distinguishes flag-only differences; `variant` is retained for readability.
- `flags`: human-readable `Vec<String>` with `"key=value"` entries. Wired into `comparable_env_hash` (Task #30, sorted so order is insignificant) and used for audit. Always populate it.
- `is_single_run`: always `true` for single-run tests. Any baseline with `is_single_run = true` must not be cited externally (see root AGENTS.md Eval Citation Discipline).

### When adding a new A/B variant

1. Branch `variant_tag` on the distinguishing env var. Pattern from `locomo.rs`:

```rust
let page_channel_state = if crate::db::page_channel_enabled() {
    "on"
} else {
    "off"
};
let variant_tag = if page_channel_state == "on" {
    "cross_rerank_v2_pages"
} else {
    "cross_rerank_v2_no_pages"
};
```

2. Push the state into `flags`:

```rust
env_stamp.flags.push(format!("page_channel={}", page_channel_state));
env_stamp.flags.push("scenario_db=consolidated".to_string());
```

3. Mirror the filename suffix in `eval_harness.rs` (legacy path). The harness branches `__with_pages` vs `__no_pages` on `WENLAN_ENABLE_PAGE_CHANNEL` (opt-in, default OFF).

---

## filename suffix on legacy path

`eval_harness.rs` branches the `app/eval/baselines/` filename suffix on `WENLAN_ENABLE_PAGE_CHANNEL`:

```rust
let suffix = if wenlan_core::db::page_channel_enabled() {
    "__with_pages"
} else {
    "__no_pages"
};
```

When adding a new A/B variant driven by a different env var, add an analogous suffix branch so page-ON and page-OFF artifacts don't overwrite each other at the legacy path.

---

## smoke vs full run sizing

| `EVAL_*_LIMIT` | Time | Use case |
|---|---|---|
| `2` | ~1 min | Wiring smoke: verify the test reaches the eval loop and saves a baseline |
| `20` | ~30 min | Subset eval: direction check for A/B comparison |
| unset | ~30 min-3h | Full fixture: for results worth citing |

Reranker first-run downloads the model weights (~1.1GB for the default bge-reranker-base; ~600MB-2.27GB for others). Account for that on cold caches.

---

## pages_count sanity gate (PR-B convention)

When using a cached scenario DB, check that pages exist BEFORE the eval loop. The PR-B runners SKIP with a clear message when the table is empty:

```rust
let pages_count = db.count_active_pages().await.expect("count_active_pages failed");
if pages_count == 0 {
    println!(
        "SKIP: cached scenario DB has 0 active pages at {}. Run scripts/seed-scenario-dbs.sh from the repo root then verify with cached_scenario_db_compat_check.",
        db_dir.display()
    );
    return;
}
```

SKIP semantics (rather than panic) match the surrounding fixture-missing branches so a contributor without seeded DBs gets actionable output instead of a thread panic. Without this gate, a corrupt or empty page table silently produces page-OFF metrics stamped with the page-ON variant tag. The mislabeling would contaminate any external citation.

---

## ephemeral vs cached: decision table

| Question | Runner shape | DB |
|---|---|---|
| What is the baseline NDCG for retrieval-only? | ephemeral-per-conv | tempdir |
| Does page-channel improve NDCG? | from-cached-DB, run twice (ON/OFF) | scenario_seeded |
| Does a new retrieval signal help? | ephemeral-per-conv (no pages needed) | tempdir |
| Does distillation quality affect retrieval? | from-cached-DB (pages already distilled) | scenario_seeded |

---

## seed completeness: one route, one contract (no drift)

Seeding a cached scenario DB is ONE orchestrator, not a scatter of STEP tests: run `seed_scenario_dbs_complete` (`tests/eval_harness.rs`). It chains event_date inject → classify → `memory_entities` sweep → episodes → distill, then asserts `SeedExpectations::complete()`. Never hand-run the individual `seed_*` STEP tests; they are its internals.

`seed_contract.rs` is the single liveness contract, gating BOTH ends:
- **Producer:** `seed_scenario_dbs_complete` asserts `complete()` — hard-fails on `memory_entities=0` (graph), `event_date=0` (temporal), or `pages=0` active (page channel; matches `MemoryDB::count_active_pages`). Presence checks, not percentages (percentages rot).
- **Consumer:** every per-query collector calls `assert_feature_substrate_live(conn, feature)` at entry — a graph/temporal/page A/B over an empty substrate **errors ("EVAL REFUSED")** rather than emitting a null that reads as "doesn't help". A starved-substrate lie is structurally impossible.

Adding a channel with an A/B: add its step to the orchestrator, its floor to `SeedExpectations`, its key to `assert_feature_substrate_live`, and a `seed_contract.rs` unit test. See `crates/wenlan-core/AGENTS.md` "Eval seed + eval read: ONE route, ONE contract (no drift)".

---

## Eval architecture + workflow (diagram)

```
SUBSTRATE  (what you retrieve from)                          pool/CE decision valid?
  deep-S faithful   per-Q DB, SeedPerQuestion + _s fixture    mem~285, isolated   ✅ the only valid one
  oracle            per-Q DB, SeedPerQuestion + oracle fixture mem~12, gold-only   ✗ pool-blind
  consolidated      ONE shared DB (all questions merged)       11397 mem, leakage  ⛔ BANNED (code-gated)

PHASES  —  P1+P2 = expensive SEED (cache once per substrate-version);  P3+P4 = cheap READ (per experiment)

  ┌──────────── SEED  (cache once, contract-gated) ─────────────┐   ┌────── READ  (per experiment) ──────┐
  │ P1 store + embed            (FastEmbed BGE-Base-Q)           │   │ P3 answer   (on-device qwen3.5-9b) │
  │ P2 enrich: classify→entity→distill(pages)   (qwen3-4b)       │   │ P4 judge    (claude CLI OAuth)     │
  │ gate: P1+P2 content-hash + SeedExpectations                  │   │ subset N (category-stratified)     │
  │       (mem≥2×pool, entities>0, pages>0)                      │   │                                    │
  └─────────────────────────────────────────────────────────────┘   └────────────────────────────────────┘
       cache → scenario_seeded/lme_s_deep_<codehash>/fullpipeline/lme/<qid>/
       (lazy-load the 9B answer model AFTER seeding — it sits idle in VRAM during P2 and slows the 4B enricher)

  ⚠ TWO "PHASE" NUMBERINGS — DISTINCT SYSTEMS, DO NOT CONFLATE:
    • OUTER pipeline (this diagram): P1 store+embed · P2 enrich · P3 answer · P4 judge.
        P1+P2 = SEED (cache once); P3+P4 = READ (per experiment). The seed runs ONLY P1+P2.
    • INNER enrich stages (enrich_db_for_eval_local, shared.rs): outer P2 expands into THREE
        sub-stages the CODE + SEED LOGS also label "Phase 1/2/3":
          Phase 1 = classify        — within-Q parallel @ EVAL_ENRICHMENT_CONCURRENCY → fills width ~8
          Phase 2 = post_ingest/mem  — per-memory link+extract+title; the DOMINANT cost, width ~1.5
          Phase 3 = distill_pages    — clustering (global barrier) + per-cluster synth (DISTILL_CLUSTER_CONCURRENCY)
    ⇒ A seed log line "[enrich_local] Phase 2" = INNER post_ingest ⊂ OUTER P2 (enrich), NOT outer P3 (answer).
      EVAL_SCENARIO_CONCURRENCY (cross-Q fan-out) is the ONLY parallelism feeding inner Phase 2's batch
      (post_ingest is a serial within-Q loop), so inner-Phase-2 width ≤ conc; classify adds within-Q
      parallelism on top → fills the batch. See "post_ingest serial: what's actually order-dependent" below.

DECISION METRICS
  pool depth + CE on/off + model  →  RETRIEVAL: NDCG@10 + recall@5 + P50 latency   ← READ minus P3/P4
                                      deterministic · no LLM · no judge · minutes · resolvable at N=30-60
  end-task confirmation           →  ANSWER-ACC (judged)   ← full READ · N=150 · pool effect ~2-3pp < MDE ⇒ confirm-only

WORKFLOW
  1. seed deep-S ONCE   (conc=16 ≈860s/Q, contract + shape gated)                  → canonical cache
  2. rerank_pool_retrieval_probe_emit × {CE-off,  CE-on × pool{10,20,50,adaptive} × model{bge-base,turbo}}
       → PerQueryRow jsonl → analyze_paired.py → PICK pool depth + model + CE-on/off  (retrieval metrics)
  3. (optional, citable)  confirm the winner via answer-accuracy @ N=150
```

## CE rerank pool / model eval methodology (boule-verified 2026-06-19)

Cross-model adversarial review (claude/main-loop + gpt-5.5 + gemini-3.1-pro, unanimous approve-with-changes). Binding rules for any CE fetch-pool-depth or reranker-model decision:

1. **Decide pool depth on RETRIEVAL metrics, not answer accuracy.** Pool-depth answer-accuracy effects are ~2-3pp — below even the full N=479 paired-binary MDE (~5pp), so an answer-accuracy A/B can return a NULL that gets misread as "depth doesn't matter" (the dead-substrate trap). Score the CE-reranked top-10 against LME `answer_session_ids`: **NDCG@10 + recall@k + coverage_recall** (continuous, bigger deltas +0.05..0.18, deterministic, no judge → resolvable at N=30-60). Answer accuracy (judged) is CONFIRMATORY; latency P50/P99 is a co-equal decision axis (bge-base ~1.2s vs jina-turbo +20ms).

2. **Substrate shape gate: `mem ≥ 2×max_pool` per question** (≥100, ideally the full ~285-memory deep-S haystack). A pool sweep only fully collapses at `mem ≤ min_pool`, but exercising pool=50 + adaptive headroom against real distractors needs `mem ≫ max_pool`. Shallow gold/oracle DBs (mem<50) are NOT pool-testable. This is a hard `SeedExpectations` assert, not a guideline.

3. **Faithful per-question isolation only.** The consolidated/merged cross-conversation substrate (one shared 11397-memory haystack) is BANNED for CE pool / on-off decisions — a question retrieves another's evidence, inflating deep-pool gains. Code teeth: `run_fullpipeline_lme` refuses `DbSource::Consolidated` for confounder-isolation arms unless `ALLOW_CONSOLIDATED_CE=1`. See root `AGENTS.md` + memory `feedback_no_consolidated_ce_substrate`.
   - **`DbSource::SeedPerQuestion` depth depends on the FIXTURE:** + `longmemeval_oracle.json` (the runner default) = gold turns only (mem~12, pool-BLIND — "rarely changes membership"); + `longmemeval_s.json` (`LME_S_FIXTURE`) = the deep ~285-mem isolated haystack (the correct substrate). Same enum value, opposite validity.
   - **PROVENANCE WARNING:** the historical CE figures (`+0.178 NDCG@10` — still cited under the `WENLAN_RERANK_SKIP_PREFERENCE` flag in `crates/wenlan-core/AGENTS.md`; `+0.233@pool=50`, no longer cited in any tracked AGENTS.md) were measured on **oracle (pool-blind)** or **consolidated (banned)** substrates — NOT faithful deep-S. Do NOT cite them for a deep-S pool/CE decision; regenerate retrieval NDCG on `SeedPerQuestion + _s`.

### emitter correctness (boule adversarial review 2026-06-19 — 2 critical fixes shipped)
The retrieval-NDCG emitter (`run_longmemeval_rerank_pool_probe`, `score_retrieval`) and its runner must obey:
4. **NDCG IDCG over ALL gold, not the retrieved set.** Grading only retrieved ids inflates IDCG as each gold is surfaced → in the multi-gold case a real recall gain registers as an NDCG *drop* (anti-correlated with the objective). `score_retrieval` builds grades over the full `has_answer` set; regression test `score_retrieval_rewards_surfacing_buried_gold`. This DIVERGES from the legacy retrieved-set baselines (which measured a pool-blind quantity) — intentional.
5. **CE-off baseline MUST control the graph stream.** `search_memory_cross_rerank_cued` sets `allow_graph_stream = reranker.is_none()`, so a `reranker=None` CE-off arm runs WITH the +0.0545-NDCG graph stream while CE-on runs without it — a confound bigger than the CE signal. The runner forces `WENLAN_GRAPH_MEMORY_STREAM=0` on every arm. (Production-realistic CE-on/stream-off vs hybrid/stream-on is a SEPARATE experiment.)
6. **Stratify, don't restrict.** Pool depth can only help on BURIED-evidence questions (gold NOT in the base/CE-off top-10); easy questions (gold already @1) are at ceiling and dilute the aggregate. Measure the buried-evidence FRACTION first and pre-declare strata; do NOT filter to buried-only (selecting on the outcome overstates the production gain).
7. **Emit `recall@pool_size` as a diagnostic** (not a decision blocker). Pool depth acts entirely on stage (a) "did gold enter the CE INPUT pool"; final top-10 alone can't attribute a null to recall-saturation vs CE-won't-promote. A null final-top-10 gain is still actionable: pick the cheaper/shallower pool. [TODO — not yet emitted]
8. **Stats:** define the `adaptive` policy explicitly; report P50 + **p95/p99** latency; apply an MDE / multiple-comparison treatment across the pool×model arms. [TODO]
   - Binary `has_answer` relevance = confirmed correct for this selector.

### deep-S N=31 pool/model sweep — RESULTS + substrate caveats (2026-06-19)

First faithful deep-S retrieval-NDCG sweep: 9 arms × N=31 via `rerank_pool_retrieval_probe_emit` (one binary, `WENLAN_GRAPH_MEMORY_STREAM=0` forced on every arm). `ce_off` = 0.614 ndcg@10. Verdicts split by what the substrate can support:

- **CE on/off = FIRM.** Every CE-on arm +0.107..+0.164 ndcg@10 (t=2.1..3.4); bge at all pools + turbo@pool10 survive Bonferroni (α/8). Robust even on the easy-skewed sample below — a harder/complete substrate can only help CE more. **Safe to record + act on now.**
- **pool depth: 10 = peak/floor (DIRECTIONAL).** bge p10/p20/p50 = 0.764/0.778/0.770 (p50−p20 Δ−0.008, deeper *hurts*); turbo monotone DOWN with depth (0.747/0.725/0.721). recall@5 rises with depth, ndcg@10 does not. pool=50 dominated (4.4s P50).
- **adaptive @ rel_gap=0.15 = DOMINATED (DIRECTIONAL).** Expands eff-pool to ~17, gets LOWER ndcg at HIGHER latency than fixed pool=10 — the deeper-helps premise fails on this substrate, and rel_gap→0 just collapses adaptive back to fixed pool=10. No reason to ship the machinery here.
- **model bge vs turbo = NOT RESOLVABLE at N=31.** bge directionally higher at all 4 pool settings (+0.018..+0.053) but t<1.3 and 16-19/31 queries TIE → decide on latency, not quality. turbo p10 = 280ms vs bge p10 = 551ms / p20 = 1253ms.

**Pareto frontier (4 non-dominated arms):** ce_off(200ms, .614) → turbo_p10(280ms, .747, +0.132) → bge_p10(551ms, .764) → bge_p20(1253ms, .778). turbo_p10 is the KNEE (~80% of max CE gain for +80ms). Recommended default = **CE ON · jina-turbo · pool=10**; bge-base = quality-ceiling opt-in; KILL pool=50 + adaptive.

**SUBSTRATE VALIDITY — two defects (why only CE-on is ship-firm):**
1. **Not representative.** The 31 DBs (smoke 24 + batchtest 8, opportunistic — NOT stratified) over-weight single-session-user (36% vs full LME-S 14%) and starve multi-session + temporal-reasoning (13% each vs 27% each) — the two hard categories where deep pools help most. → aggregate ndcg inflated; pool-depth verdict under-powered exactly where it matters.
2. **Enrichment near-complete (CORRECTED 2026-06-19 — earlier "pages=0" was a COUNT bug).** All 31 DBs are LIVE on classify (100%), `memory_entities` (329-431), `event_date` (261-299 ≈ 100%), AND **pages** (239 active total, mean 7.7/container, ALL 31 non-zero, real topical titles e.g. "ANZAC Identity Origins", "gender pay gap"). The prior "pages=0" was a measurement bug: **`COUNT(*) FROM pages` mis-plans against the libsql vector-index shadow and returns 0** even when rows exist — enumerate instead (`SELECT id FROM pages WHERE status='active' | wc -l`). **PREVENTION:** before trusting any "channel starved" finding, run `scripts/probe-scenario-liveness.sh <db-dir>` (enumerate-based, all channels) — never hand-roll `COUNT(*)`. A "starved" verdict must come from row enumeration, and `immutable=1` (the sandbox CANTOPEN workaround) ignores the `-wal`, so checkpoint or drop it when recent writes matter. Only **episodes=0** is a genuine gap (no episode-backfill step). The page channel did NOT contribute to the N=31 sweep ONLY because `WENLAN_ENABLE_PAGE_CHANNEL` was unset — the pages EXIST, so page-channel-ON is testable on the existing pool31 with NO reseed. ⚠ OPEN: `count_active_pages` (db.rs:16572, used by the seed contract's page floor) runs the SAME `SELECT COUNT(*) FROM pages WHERE status='active'` that returned 0 in the sqlite3 CLI. The CLI mis-plans on libsql vector-index shadows; whether **libsql's own** `COUNT(*)` is also affected is UNVERIFIED — if it is, the `complete()` page floor falsely rejects live-page DBs. Needs a Rust-level check (count via libsql vs row enumeration).

**CHANNEL UTILIZATION — the sweep ran CE over a PLAIN hybrid base, NOT the full stack.** Traced `search_memory_cross_rerank_cued`: the probe calls it with `temporal_cue=None` and every enrichment flag at its default-OFF. Only vector + FTS (RRF) + CE fired. Dormant channels split two ways:
- *Legitimately off (no gap):* graph→memory stream is code-skipped under any live CE (`allow_graph_stream = reranker.is_none()`) — its dormancy MATCHES production CE-on, and graph×CE measured non-significant (don't stack). Temporal soft boost = measured CLOSED-NULL on LME (cue fires ~3%, event-anchored TR doesn't trigger the deictic gate). Global prelude = opt-in.
- *Page channel — flag-off, NOT absent (corrected):* `WENLAN_ENABLE_PAGE_CHANNEL` was default-OFF in the sweep, so pages never entered the CE candidate pool — but the pages EXIST (mean 7.7/container). So page-channel-ON is testable on the EXISTING pool31 NOW (no reseed). SCORING CAVEAT: the probe scores top-10 vs gold MEMORY ids, so a page surfaced into the top-10 counts as a MISS unless page→source expansion (HippoRAG-style, PR #203 `coverage_recall`) is applied — wire that before reading a page-on NDCG, else the page arm looks falsely negative. Episodes genuinely absent (minor channel).

**GATE — confirm pool/model/adaptive at N=90 (corrected scope 2026-06-19).** Pages are ALREADY live in pool31 (no reseed needed for the page channel), so:
- **Immediate (no reseed):** re-run the N=31 sweep with `WENLAN_ENABLE_PAGE_CHANNEL=1` on the existing pool31 to get the **page-channel-ON family** — but FIRST wire page→source expansion into the probe's scorer (else pages count as misses). This tests "does pool depth shift when page candidates join the CE pool" without any seeding.
- **The remaining real reason to reseed N=90 = REPRESENTATIVENESS only** (the category skew: SSU 36% vs full 14%, MS/TR starved), NOT pages. Stratify to full LME-S proportions (KU 14 · MS 24 · SSA 10 · SSP 5 · SSU 13 · TR 24 = 90). Episodes (genuinely absent) can be added in the same reseed if the episode channel is ever wired into retrieval.
- The N=90 reseed MUST use the batched enrichment path (n_seq_max=8); the seed-only `enrich_fullpipeline_lme_only` path measured at n_seq_max=1 / 0.4s/call (serial) — see throughput note. CE-on ships now; turbo↔bge flip + pool-depth + adaptive-kill wait for the representative N=90. Per-query latency N-independent (turbo p10 stays 280ms).

## choosing N: statistical power

Paired binary-accuracy MDE (p≈0.6, inter-arm corr≈0.7, α.05, power.80): **N=30 ≈ ±19pp, N=60 ≈ ±14pp, N=150 ≈ ±9pp, N=479 ≈ ±5pp**. (rho/discordance are assumed, not measured — treat as order-of-magnitude.) Therefore: N=30 is blind to few-pp effects; N≈150 is the directional floor for answer-accuracy; N=479 (full LME-S) is the citable gold standard used by prior shipped flags. Quick-tier subsets MUST be **category-stratified** (LME-S categories are uneven: multi-session 133 / temporal 133 / knowledge-update 78 / SSU 70 / SSA 56 / SSP 30) and the ship rule pre-registered before the run. NDCG-based pool decisions (rule 1) are resolvable at far lower N than answer-accuracy.

## on-device seed throughput (cost planning)

Batch is **M=8-capped** (per-slot KV = ctx/n_seq_max; raising M truncates prompts — probe `a4db3a65`). M=8 is a fixed ceiling, but realized fill has headroom: full-pipeline seed runs ~5/8 (classify+extract+distill+answer interleave with CPU gaps; distill is ~3% of wall, not the bottleneck). **Scenario-concurrency >8 raises realized throughput toward M=8** (MEASURED: conc=8 → time-weighted 5.33/8; conc=16 → **7.83/8, 96% of GPU-time at full M=8** over 396 calls = ~1.47×) — it does not raise the ceiling. Config that hit 7.83/8: `EVAL_SCENARIO_CONCURRENCY=16 WENLAN_LLM_PARALLEL_SEQS=8 WENLAN_LLM_CTX_SIZE=16384 WENLAN_LLM_COALESCE_MS=10`. Deep-S seed ≈ 1262s/Q at conc=8 → **≈860s/Q at conc=16** → N=150 ≈ 36h, N=479 ≈ 4.8 days. Surface this cost as a planning input; do NOT let a cost wall rationalize the consolidated-substrate ban. **conc is RAM-BOUND (~1.6GB/Q working set), so the saturation point is per-box: conc=16 was measured on a free-RAM box; on a 16GB box conc=16 SWAPS and runs SLOWER than serial, and the safe cap is conc≈3 (runbook `ram_warn` computes `1.6·conc+4` GB and warns).** Width-8 fill on a RAM-capped box comes from WAY B's within-Q batching (`WENLAN_SEED_BATCHED_POSTINGEST`), NOT from raising conc.

**Reading the seed log — the per-Q number is concurrency-inflated.** The `[lme_enrich_only] … wall=Xs/conc N≈Ys/Q total=Zm` line: `wall` is the IN-FLIGHT wall for that one question, but `.buffer_unordered(N)` runs N questions through ONE GPU worker at once, so `wall` is ≈N× the true throughput cost. Read `≈Ys/Q` (= `wall/conc`, the honest per-Q rate when GPU-bound) or `total ÷ done`, NEVER raw `wall`, for throughput/ETA. (Pre-2026-06-21 the field was the bare `scenario=Xs` wall and was repeatedly misread as per-Q cost — e.g. a `scenario=3192s` line at conc3 is ~1064s/Q ≈ 18min, not 53min. The fix: emit `wall/conc≈Ys/Q` inline so the inflated number can't stand alone. A GPU at 99% util with flat per-call decode ms/tok IS optimal — a rising per-Q `wall` there is the cache thinning to fully-fresh questions, not a regression.)

## on-device perf levers (merged status + default state)

The "is decode / prefill / distill addressed?" question has ONE answer table — check HERE before
claiming a lever is missing or "ruled out" (a 2026-06-20 audit found the throughput notes
self-contradicting and a diagram that wrongly called decode unaddressed):

| lever | flag / knob | merged | engine default | seed runbook | targets |
|---|---|---|---|---|---|
| WAY B extract batching | `WENLAN_SEED_BATCHED_POSTINGEST` | ✅ `b4460d59` | OFF | **ON** | extract slot occupancy — width 1.69→8.00 measured (the realized lever) |
| continuous-batch slot-backfill | `WENLAN_LLM_SLOT_BACKFILL` | ✅ #276 `6724b757` | OFF | **ON** | decode ragged-finish slots — `drain_cap` m→4m keeps M=8 full as seqs finish early |
| prefill prefix-KV cache | `WENLAN_LLM_PREFIX_KV_CACHE` | ✅ #278 `0503f451` | OFF | **ON** | shared-prefix prefill (≥32 tok) — caches the KG-template prefix only, NOT per-memory content |
| distill cluster concurrency | `DISTILL_CLUSTER_CONCURRENCY` | ✅ `distill.rs:717` | 1 (serial), max 4 | **2** | distill across independent clusters — ~3% of wall, small prize |
| flash-attn | — | n/a | — | — | REFUTED (no gain on this path) — the ONLY genuinely ruled-out idea |

**Why default-OFF in the engine but ON in the seed:** slot-backfill rewrites the shared on-device
inference path that CI can't validate on Metal (opt-in until staged-rollout confidence); #278 +
distill-conc were gated behind WAY B Phase-2 measurement. They are SAFE to enable on the seed:
per-sequence semantics are unchanged (the substrate is already non-deterministic under M=8 batch
composition — these flags don't add a new class), and `SeedExpectations` /
`assert_feature_substrate_live` hard-fail if any flag ever zeroed out extraction. The runbook turns
all four ON via overridable `${VAR:-default}` and prints a `[perf]` banner at start, so the active
levers are visible per run — never silent.

## post_ingest serial: what's actually order-dependent

Inner Phase 2 (`run_post_ingest_enrichment` per memory, `post_ingest.rs`) is a serial within-Q
loop. Grounded breakdown of WHY — only PART of it needs ordering:

| step (post_ingest.rs)              | LLM? (decode) | order-dependent? | why |
|---|---|---|---|
| 1. auto_link_entity                | maybe (extract on miss) | YES | links to entities CREATED by prior memories |
| 2. entity extract (`extract_single_memory_entities`) | YES (heavy) | NO | depends only on THIS memory's content |
| 3. check_page_contradiction        | maybe | YES | reads pages written by prior memories/distill |
| 5. enrich_title                    | YES (heavy) | NO | depends only on THIS memory's content |

So the decode-bound LLM work (extract + title) is order-INDEPENDENT; only the link/contradiction
DB commits need insertion order. The serial loop is a fidelity-preserving SIMPLICITY choice
(matches the production single-memory store path, "Rules of ML" #32), NOT a hard requirement on
the LLM work.

PROPOSED LEVER — ENDORSED-WITH-CHANGES by boule `wfapcs1sm` (degraded panel: codex gpt-5.5
approve-with-changes + gemini revised-to-approve; claude dropped by contamination gate; medium
conf). Decouple compute from commit — pre-compute extract+title in a `buffer_unordered` batch
(fills M=8), then apply link/contradiction serially in insertion order, preserving the substrate.
DIRECTION confirmed (batch-fill is the primary lever). CORRECTION 2026-06-20: only flash-attn was
refuted. prefix-KV (#278 `0503f451`) and continuous-batch slot-backfill (#276 `6724b757`) are now
MERGED and are COMPLEMENTARY engine levers — both default-OFF flags (see "on-device perf levers"
table below); distill cluster-concurrency was always BUILT (`distill.rs:717`). Do NOT re-assert
these are "ruled out" — for the SEED's extract workload prefill often DOMINATES the call (live
`[batch_timing]`: prefill 51-60% of total in 2 of 3 sampled M=8 batches), so #278 (prefill-side)
is plausibly the higher-value parked lever, NOT decode.
MANDATORY GATES before any code (council dissent):
  1. The "~3-5×" was OVERSTATED + conceded. Magnitude is Amdahl-bound by the serial commit tail —
     first MEASURE the independent-extract/title vs dependent-commit split of post_ingest wall time.
  2. Width 1.5 proves starvation, NOT scalable headroom — confirm with a tokens/sec-vs-batch curve
     (GPU may be memory-bandwidth-bound before M=8).
  3. PURITY UNVERIFIED: prove `extract_single_memory_entities` + `enrich_title` are pure functions of
     one memory. The duplicate-entity hazard (two same-Q memories about a NEW entity → parallel
     extract+create dupes that the serial `auto_link_entity` path would have merged) is the dominant
     fidelity risk and must be golden-tested against the serial substrate.
  4. Buffered precompute of ~285 memories may worsen the conc=3 RAM cap — bound the buffer.
  5. Deterministic sampling + golden output tests, else seed reproducibility breaks.

distill (inner Phase 3) is NOT serial-by-design: clustering (`find_distillation_clusters`) is a
genuine global barrier, but per-cluster synthesis already honors `DISTILL_CLUSTER_CONCURRENCY`
(`distill.rs:717`, default 1, max 4). It's ~3% of wall — a non-target regardless.

## content-addressed seed (no silent-stale)

A cached seed is valid only for an exact P1+P2 **code-content hash** — the hand-list {fixture-rev, embedder, enricher-model} is insufficient (omits chunker, embedding quantization, distill-code version, session-construction, schema). Hash the seeding+enrichment code + config + fixture bytes; `SeedExpectations` is the backstop. A P2 enrichment change flips the hash → reseed once, contract-gated; a P3 flag/pool/model change leaves it untouched → reuse. Pay the seed once per substrate-version, never per experiment.

---

## scenario DB cache env flags

Three env flags govern how `open_or_seed_scenario_db` handles a stale or mismatched cached scenario DB.

| Flag | Default | Effect |
|---|---|---|
| `EVAL_ALLOW_WIPE` | unset (refuses) | Permits wiping + reseeding from scratch: a partial-state DB via `clear_all_for_eval`, or a cache-env-mismatched DB by removing its `origin_memory.db` file (`std::fs::remove_file`). |
| `EVAL_PARALLEL_OK` | unset (refuses) | Allows concurrent access to a locked scenario DB (results may be corrupted). |
| `EVAL_MIGRATE_STALE` | unset (refuses) | Migrates a schema-stale cached scenario DB IN PLACE, without wipe or re-seed. |

### EVAL_MIGRATE_STALE details

`EVAL_MIGRATE_STALE=1` triggers the migrate path when `open_or_seed_scenario_db` detects that the DB's `cache_env.json` has a schema/migrations stamp mismatch but the enricher provenance (`enricher_provider` + `enricher_model`) matches the current run. It refuses cloud-enriched or unstamped-legacy DBs (anti-laundering: a DB seeded by a different model cannot be silently promoted as if it were seeded by the current one). The migrate guard requires the DB to already be fully enriched (`enriched == mem_count`, no partial state) and substrate-live (temporal, graph, pages). If either check fails, the path returns an error and you must re-seed (or set `EVAL_ALLOW_WIPE=1`).

After the migration guard passes, execution falls through into the shared Phase-1 classification backfill (the same block that runs on a normal cache hit). This ensures that a migrated OLD DB that predates the classification pass gets `importance`/`quality` backfilled before the eval loop sees it, preventing T8/T11/T15 training-serving skew. There is exactly one `write_cache_env_stamp` + `return Ok(db)` for both the migrate and cache-hit paths (at the bottom of the shared block).

## paired A/B reading: the G3 gate (analyze_paired.py)

`scripts/analyze_paired.py` applies the Eval-Trust v3 G3 "A/A-floored attributed liveness" gate on top of Wilcoxon + BH-FDR. Any `<feature>_aa_<bench>.jsonl` in the input dir is an A/A no-op control (flag OFF on both arms) and sets that bench's noise floor (aggregate + per-category |meanΔndcg|). Every other feature gets a `G3` verdict: `SIGNAL` (above floor, right direction, attributed, BH-sig), `WEAK` (same minus BH-sig), `NOISE-FLOOR`, `WRONG-DIR`, `UNATTRIBUTED` (<90% of moved queries carry the channel's touch flag — precedence: `channel_touched` (generic, highest) > `temporal_touched` > `graph_skipped`; features with a wired probe: `graph_stream` arms — probe is whether the stream would contribute ≥1 linked memory (non-person anchors under hub cap with linked memories, mirrors `MemoryDB::graph_stream_touches`); `rerank_skip_pref` — probe is `is_preference_query`, the bypass IS the channel; `rerank`/`rerank_model_*` — probe is CE top-10 differing from base top-10, collectors fetch base ids outside the latency window; `rerank_graph_stack` — stream probe (arm retired on graph-stream-default-on branch); page/episode/fact/global_prelude carry NO probe intentionally — verdicts stay `*` until an internals probe exists), `NO-FLOOR` (no A/A run for the bench — emit one, e.g. the `rerank_window_aa` pattern, before trusting verdicts). Per-category verdicts are conditions 1+2 only (no per-category p-value). Selftest: `python3 scripts/analyze_paired.py --selftest`.

Trust calibration: a `*` suffix (`SIGNAL*`/`WEAK*`) means the rows carried no touch flag, so attribution was vacuous — `graph_stream`, `rerank_skip_pref`, `rerank`/`rerank_model_*`, and `rerank_graph_stack` arms now emit `channel_touched` probes (condition 3 is real for those); page/episode/fact/global_prelude intentionally carry no probe yet — their `*` verdicts are honest, not a bug, and stay until an internals probe is wired. Floors are bench-keyed, not path-keyed: don't co-locate a CE-path A/A with base-path feature files (noise characteristics differ; the CE-path A/A measured NON-deterministic — a full-ndcg flip between identical OFF arms at smoke n — which the analyzer surfaces as a `WARNING` when a floor source has `n_touched > 0`). `paired_ab_emit` itself emits no `_aa` file; co-locate an A/A from the matching path/bench or every verdict reads `NO-FLOOR`.

## retrieval drift oracle (differential / equivalence)

`eval::retrieval_drift` is a deterministic DRIFT detector for retrieval ranking — the first concrete instance of the loop-engineering differential-oracle pattern (`~/.agents/loop-engineering.md`). It catches silent reorderings the labeled NDCG eval (`eval::retrieval`) misses, because that eval is noisy (±~19pp at small N) and post-merge.

- **Honest framing:** it detects DRIFT from a frozen reference, NOT absolute correctness. It is trustworthy only **layered UNDER** `eval::retrieval`: the labeled NDCG eval vouches each baseline and gates every golden refresh. Never sell it as a correctness oracle.
- **Invariant:** top-weighted Rank-Biased Overlap (`eval::rank_overlap::rbo`) — asserts ranking ORDER agreement (a stable property), never byte-exact scores. Catches a within-top-K demotion; tolerates benign cross-arch f32 ULP near-tie swaps. Scores are stored for diagnosis only.
- **Goldens (committed):** `crates/wenlan-core/src/eval/goldens/retrieval_ranking.{current,anchor}.json`. Dual anchor — `current` refreshed on intentional changes, `anchor` drifts only by explicit human decision (delete it to re-anchor) to catch slow cumulative drift.
- **Cadence:** L6 (relevant core/eval-owner push to `main`, plus manual dispatch). `ranking_drift_vs_golden` is `#[ignore]`d and selected by its exact full test name alongside `test_run_quality_cost_eval_basic`; never use the broad `eval::retrieval` substring because it also selects manual/scaling benches. The metric + regression-injection tests are synthetic (no model) and run per-PR. NOT a pre-push gate (wenlan has none by design).
- **Refresh discipline:** `EVAL_GOLDEN_BLESS=1 cargo test -p wenlan-core --lib eval::retrieval_drift::tests::bless_goldens -- --ignored` — ONLY when `eval::retrieval` is green AND the ranking change is intentional. Capture/refresh in the env the gate runs in (Linux CI) to match arch; the RBO tolerance absorbs residual ULP.
- **Verify-the-verifier:** the synthetic injection tests (`injected_reverse_top2_is_red`, `injected_drop_best_is_red`, `injected_full_reverse_is_red_even_at_anchor`, `identical_is_green`, `benign_deep_neartie_swap_is_green`) prove the metric goes red on real regressions and green on benign noise — an oracle never seen red is not an oracle.

## cross-reference

- Fixture population, env vars, seed scripts, baseline layout: see `app/eval/AGENTS.md`.
- Citation rules (single-run, schema-version, receipt-only): see root `AGENTS.md` "Eval Citation Discipline".
