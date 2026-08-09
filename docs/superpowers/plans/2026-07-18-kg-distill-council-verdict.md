# KG communities + distill architecture — council verdict (boule:debate)

**Run:** 2026-07-18, 3-lab adversarial council (Claude main-loop, GPT-5.5 via
Codex, Gemini 3.1 Pro), form → attack → defend → stake-free judge,
position-stable across both counterbalanced orderings. One attack
(gemini→claude) lost to a transient classifier; all other 13 agents completed.

**Judgment: `needs-more-info` (medium confidence).** Initial tally was
unanimous approve-with-changes (3/3); after the adversarial round codex and
gemini both revised to needs-more-info (2–1 operative). The *direction* has
genuine three-way consensus — what blocks approval is a short list of
load-bearing unverified/unspecified items, not the architecture.

Proposal debated: phased Leiden plan (daemon `community_id` → shared
co-citation → community-scoped maintenance → GraphRAG-shaped unified graph),
with the pages-in-or-out boundary, page-evolution stability, and distill
truthfulness posed as open problems.

## Settled by consensus (all three members)

1. **Daemon-side Leiden with persisted `community_id`** beats the frontend
   heuristic — but seeding + hysteresis is NOT enough for stability: an
   explicit old→new **partition rebinding step (max-overlap matching)** and
   **durable community IDs** (region names keyed to the durable ID) are
   required, or labels permute across re-runs and the Atlas visibly thrashes.
2. **Pages enter the graph as edge-bearing NON-VOTING participants**
   (option i — the answer to "should pages be left out of Leiden inputs"):
   wikilink/entity edges present, excluded from the partition objective, with
   **thresholded multi-community assignment** and a **page-embedding fallback**
   for entity-poor pages. Strict pages-out is wrong: it orphans memoryless
   human/agent-created pages and gives edit cascades no path to travel.
3. **Page edits become typed first-class captures** riding the existing
   ingest path (no new cascade mechanism) — BUT derived/agent-edit captures
   must be **lineage-gated out of** the ≥3 seed floor AND out of co-citation,
   relation-confidence, and synthesis-support statistics, or Phase 2's
   structural signal inflates on the system's own outputs (pages transitively
   seeding pages).
4. **Page identity is durable and decoupled from community assignment.**
   Community merges/splits surface as review proposals, never silent identity
   changes. Pinning granularity must be **block/claim level** — page- or
   section-level pinning guarantees staleness.
5. **Truthfulness gates are currently inadequate and mis-placed.**
   Embedding-cosine passes fluent paraphrased hallucinations; memory-ID
   citation proves traceability, not entailment. Required: **per-claim
   source-span support scoring gating PRE-publication**, with a
   machine-readable *provisional* status on the agent read path — because
   agents consume pages before any human clears a review queue; the queue is
   the human-curation channel only.
6. **Community-scoped maintenance (Phase 3) needs three backstops:**
   a bounded **new-topic staging scan** (as written, novel topics never
   accumulate the ≥3 seed floor — new-page creation silently breaks);
   **dependency/staleness invalidation** for page↔source, not only periodic
   sweeps; and **multi-community routing** — one memory can legitimately bear
   on several communities.

## What blocks approval (the needs-more-info list)

- **leiden-rs verification spike** — crate maturity is unverified beyond
  registry metadata; does it expose partition seeding?
- **On-device re-partition benchmark** — the full-Leiden recompute cost on
  consumer hardware is load-bearing for the entire incremental premise.
- **Concrete page↔community routing model** — once 1:1 community=page softens
  to a scope hint: membership weights, update triggers, coverage/invalidation
  criteria. The "community stays the ROUTING key via page subscriptions"
  rebuttal held directionally but is unspecified.
- **Assertion-type taxonomy for human edits** — correction vs speculation vs
  preference vs observation, with distinct propagation rules (user assertions
  display as such; they are never document-backed evidence).
- **Co-citation weighting** — raw PMI over-weights rare accidental
  co-mentions in small local corpora; use smoothed PPMI/NPMI with a
  minimum-support floor plus temporal decay, validated against
  wrong-attachment rates.
- **Hysteresis tuning risk** — churn suppression also suppresses *necessary
  truthful* reassignments; stability and truthfulness are co-equal in the UX
  contract and need separate dials.

## Dissent

One member held approve-with-changes throughout: direction fully converged,
Phase 1 has a shipped rollback (the frontend heuristic remains), and the
unknowns are resolvable as **gated prerequisites inside an approval** rather
than blockers. Practically: run the spike + benchmark first, then the phased
build proceeds as consented.

## Sequencing consequence

Phase 1 (Leiden + `community_id` + rebinding + durable IDs) is still the
right first move — its rollback is shipped and it touches nothing in distill.
The unified-graph endgame must wait for the routing model and benchmark.
Related docs: `2026-07-18-page-map-mind-map.md` (Page Map carve-out).
