# Knowledge-graph revamp — close plan

**Status:** authored 2026-08-04 from the post-review corrected picture (adversarial source review + independent re-verification). Supersedes the three-step close sketch. Companion to the unified-model spec (`2026-07-18-kg-unified-model-spec.md`, v3, in the `wenlan-app` repo under `docs/superpowers/plans/`) and the council verdict.

**Binding scope decision (user, 2026-08-03):** the goal is *a working correct knowledge graph*. The automated page-generation rung (M6) is parked entirely — substrate and overview page both. KEEP: retiring the duplicate legacy link stores, shipping the truth gate to the app, and the measurement work. DEFER: retiring the app's map-guess heuristic. PRs #470/#471 close unmerged.

**~~The one open user decision~~ — RESOLVED 2026-08-04, no decision needed.** The grounding sweep carried a documented known limitation (`HV3_H2`) whose 2026-07-25 ruling blocked any default-ON flip until it was closed or re-adjudicated. It is now **closed**, and not by accepting the flaw: the defect was in the test case, not the model. Its wording was ambiguous about whose records it appealed to, and it contradicted the record-keeping rule that 10 of the 57 coverage cases depend on — so no prompt setting could pass both gates. Rewriting the case to name this system unambiguously closed the leak with no prompt change. Measured on the pinned model: 35 cases, 25 entailment calls, **promoted=0, leak set empty**. See the RESOLVED block in `2026-07-25-m3g-gate-criteria.md`. **Gate 1 no longer blocks the flip; the flag stays default-OFF purely for want of the Gate 2/3 measurements in step G3.**

---

## Part A — what the spec now gets wrong or leaves underspecified

Each item is labeled: **SPEC EDIT** (the spec text must change), **CODE FIX** (the spec is right, the code never honored it), or **DEBT** (spec commitment, absent, consciously deferred with a tripwire).

1. **Origin classification has no mechanism — SPEC EDIT + CODE FIX.** The grounding rule correctly defines external as "a document ingest or a human statement," but the spec never says *how* a connector declares document-ingest. In that vacuum the promotion rung's mechanics doc narrowed external to the literal string `source_agent='folder'` (mechanics doc §5.2), so Obsidian imports (`source_agent='obsidian'`) — genuine document ingests — can never ground, and every future connector re-breaks the gate by design (the connector guide tells each to invent a new string). Spec edit: an explicit, daemon-authoritative origin classification (document_ingest / human_capture / generated) recorded at the ingest boundary, which promotion reads instead of string-matching. Code follows. Rider: vault imports must skip Wenlan's own projected pages (they carry `origin_id` frontmatter) or the system re-ingests and eventually believes its own output.

2. **Origin honesty is not enforced at the API boundary — CODE FIX (the spec already demands it).** §5.6 says no wire request may select anything that decides grounding, but `/api/memory/store` persists the client-supplied `source_agent` raw, with no reserved-value guard (`memory_routes.rs:462`). Any agent can send `source_agent:"folder"` today and make its own extracted relations promotion-eligible. Fix: normalize/reject reserved origin-bearing values at the boundary. One-line spec edit naming `source_agent` as an origin-bearing field.

3. **Q2 (seed-floor strictness) must be closed — SPEC EDIT.** The shipped genesis floor is the *inverse* of the grounding rule: only agent captures can seed pages and ingested documents never can (`db.rs:1966-1969`, `db.rs:2210`). The spec left "captures-only vs documents-count-as-one" open as Q2 — but the captures-only variant is incoherent under the spec's own §5.6, which makes every agent capture `generated`: with no human-capture path built, strict Q2 means nothing could ever seed. Close Q2: a document counts as one independence group; generated captures never count. (Anti-fragmentation survives — one book = one group, still below the ≥3 floor alone.) This is recorded now and *implemented only if genesis is ever unparked*: changing the floor without the surfacing backstop would silently zero page creation on a capture-heavy corpus — the two ship together or not at all.

4. **"One edge store" needs an enforced writer inventory — SPEC EDIT + CODE FIX.** M2 requires every mutation to update all live stores in one transaction, but never enumerated the writers, and the always-on ambient entity-extraction path was missed: it inserts relations bare and even deletes competing relation rows with no edge write (`db.rs:32286`, `db.rs:32324`), so the canonical store is bypassed *and* driven into permanent parity drift by the path that runs most. Spec edit: a required, checked writer inventory (parallel to the M2 assignment matrix) with the parity sweep failing loud on any writer outside it. Code fix: route the ambient path through the dual-writing relation call (`create_relation_with_span`, which already classifies lineage/space correctly).

5. **Promotion must be repair-aware — CODE FIX + one-line rung note.** The grounding sweep permanently advances its cursor past relations that have no edge row yet (`edge_grounding.rs:422-425`) and only ever rescans on a model/prompt version change. Any backfill therefore lands behind the cursor, invisible. The repair in Part B resets the cursor explicitly; the rung note says repairs require it.

6. **Debt register (spec commitments, absent, deferred deliberately):**
   - *Embedding/model versioning (§6.6):* stored embeddings carry no model-version column. Tripwire: must land before any embedding-model change, or similarity comparisons silently mix incompatible vectors.
   - *Export/import round-trip (§6.8):* absent by design — export is one-way, provenance is stripped on ingress. Tripwire: never present export as a backup until this lands; the operative safety net is the §6.9 SQLite online-backup + restore drill (step 6's gate).
   - *Retention caps (§6.5):* no prune for page history or superseded edges. Ops hygiene, not correctness.
   - *Human-edit → memory loop (§5.1):* helper exists, nothing calls it; the page-update route lacks the bound-base digest it needs. The spec's interim rule (human edits win prose, no voting weight) operates and is spec-sanctioned. App-side work when wanted.
   - *Q3 human-assertion taxonomy:* zero presence in the tree; interim conservative rule operates. Was spec-marked "needed before M5 finishes" — the truth-gate step below must state which call the merged M5 code made rather than discover it at cutover. `[unverified: what merged M5 decided]`
   - *Unformed-topic surfacing (council Trap 1):* never shipped (explicitly deferred in code); nothing regresses by parking it. Coupled to item 3 — both-or-neither.

---

## Part B — the close plan, in dependency order

Two tracks. Track T is independent of everything in track G and can start immediately.

### Track T — truth gate to the app (kept: M5 App PR + cutover)

**T1. Ship the "this page's claims aren't verified yet" signal to the app, then run the switch-on ceremony.**
*Plain:* every page an agent reads will carry an honest badge saying whether its statements are backed by sources, because agents read pages the instant they're written — before any human reviews them.
*Why independent:* the truth-gate worker runs on its own; its support links deliberately don't participate in grounding. Nothing in track G blocks it.
*Exit:* agent read paths return the support status; cutover generation > 0; app displays both fields. Needs the app repo. `[needs app repo: current app-side state]`
*Owner:* agent task; the cutover ceremony is operator-run with the user present.

### Track G — make the graph's links real

**G1. Make every link-writer write the one canonical link store.**
*Plain:* one background process that runs all the time has been writing links into the old bookkeeping only, and even deleting entries there without telling the new store — so the new store can never match the old one until this stops.
*Why first:* the mismatch it creates blocks the "old and new stores agree" check that gates every later switch — including the retirement the user kept. Everything downstream reads through this.
*What:* route ambient extraction through the existing dual-writing call; run the existing backfill for historical relations; **reset the grounding sweep's cursor** (it has permanently skipped everything that lacked an edge row); normalize entity spaces so backfilled links classify as first-class rather than legacy.
*Exit:* a new ambient extraction produces a canonical edge row (test); the parity sweep reports zero drift and stays at zero across a week of normal use.
*Owner:* agent task.

**G2. Make "where did this come from" honest and tamper-proof.**
*Plain:* the system decides what counts as outside information by looking at a label any client is allowed to write, and it doesn't recognize Obsidian imports as outside information at all — both halves get fixed at the daemon so no agent can vote its own notes into the graph.
*Why here:* grounding must not be switched on over a spoofable or too-narrow origin test; G1's backfilled edges are judged under this gate.
*What:* Part A items 1 + 2, including the self-recapture exclusion for Wenlan's own projected pages.
*Exit:* a store request claiming a reserved origin is normalized/rejected (test); an Obsidian-imported document's relations become promotable (test); a Wenlan-projected note re-imported from the vault does not (test); spec addendum merged.
*Owner:* agent task.

**G3. Measure before switching anything on.**
*Plain:* run the background jobs on a realistic copy of the data and write down what they cost (memory, how long they block the daemon) and how well grounding performs, so switches are flipped on evidence instead of hope.
*Why here:* every later flip is gated on receipts that do not exist today — the coverage and latency gates have runnable benchmarks but no committed results, and the one scary latency number in the docs (18.88 s) is prose with no receipt behind it. `[unverified until measured]`
*What:* the two parity sweeps (memory + foreground-latency ceilings; add chunked sweeping if the measured hold breaches the gate bars); the grounding sweep's coverage floor (≥ 80 % on the labelled positive set) and latency ceiling (≤ 500 ms p95 mutex-hold per tick), re-running the zero-false-grounding fixture against the G2 origin gate. The genesis-lane flag is *excluded* — it exists only for the parked rung; measuring it is work for a dropped feature.
*Exit:* receipts committed per the gate doc's three gates. Needs the live corpus (or a representative copy). `[needs live DB: corpus document-vs-capture mix — this also determines how much of the graph can ground at all]`
*Owner:* agent task.

**G4. ~~USER DECISION~~ — DONE 2026-08-04, no decision required.**
*Plain:* the one crafted sentence known to fool the verifier turned out to be a badly-written test, not a weakness in the model — it was ambiguous about whose records it appealed to, and it contradicted a rule that a sixth of the coverage cases rely on. Rewriting the sentence to say what it meant closed the leak outright.
*Result:* pinned model, 35 cases, 25 entailment calls, promoted=0, leak set empty. No prompt change; the record-keeping rule the coverage gate needs is untouched. The blocking clause in the 2026-07-25 ruling is satisfied.
*Residual:* nothing. Gate 1 is at unqualified exact-zero again; only G3's unmeasured coverage/latency gates keep the flag off.

**G5. Switch on grounding, then move readers to the new store, then soak.**
*Plain:* turn on the checker that promotes verified links, point the product's reading paths at the new store one consumer at a time, and watch for a quiet period to prove nothing broke.
*Why here:* flips are cheap only after G1-G4; each reader cutover is already gated on a clean parity check by construction.
*Exit:* grounding sweep enabled per G4's decision; each reader cutover flipped with its watermark clean; an agreed soak window passes with drift still zero.
*Owner:* agent task (flip order proposed by agent, confirmed by user).

**G6. Retire the duplicate old stores.**
*Plain:* delete the five old link-bookkeeping tables and the old entity table so there is exactly one source of truth — the bug the whole refactor exists to kill.
*Why last:* irreversible; only safe after G5's soak, and only with a proven restore path.
*Gate:* a pre-migration SQLite online-backup plus a rehearsed restore drill (the spec's own rollback contract for this stage is restore-from-backup only), and a declared downgrade barrier. Export/import is NOT that safety net (debt register).
*Exit:* retirement migration merged; backup + restore-drill receipt attached.
*Owner:* agent task; user confirms the point of no return.

### Ledger — open by choice, recorded so "closed" stays honest

- **Page genesis still believes agent output.** Today only agent captures can seed pages and documents cannot — the inverse of the grounding rule. Fixing it is the coupled pair in Part A item 3 (grounded floor + surfacing backstop, both or neither), parked with M6 by the user's decision. The close delivers a grounded *graph*; page *creation* remains ungrounded, knowingly.
- **Communities:** the Leiden shadow stays off; its two daemon-side summary readers serve parked overview surfaces, and the app-side map keeps its uniform heuristic (no truth-mixing exists — nothing consumes community assignments yet). Retirement of the heuristic: deferred by user decision.
- **Debt register items** (Part A item 6) with their tripwires.
- **PRs #470/#471:** closed unmerged by user decision.

---

*Claims in this document marked `[unverified …]` or `[needs …]` were not provable from the monorepo tree at authoring time. Everything else was verified against source on 2026-08-03/04.*
