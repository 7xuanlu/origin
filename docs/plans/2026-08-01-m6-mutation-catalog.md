# M6 Stage-0 artifact 12 — RED mutation catalog and positive controls

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: Stage-0 item 12 — a RED mutation catalog and positive controls for
**every** gate G1–G11.
Continues the decision numbering from artifact 11 (`S0-133`).

### Citation conventions

In-repo citations are read on branch `kg-m6-stage0`. Two files carry most of
them, and a short-form `basename:NNN` always means one of these:

- `docs/plans/2026-07-27-m5-truth-state-matrix.md`
- `crates/wenlan-core/src/db/claim_identity.rs`

Citations written **`gp@wenlan-app:NNN`** are foreign: line NNN of the frozen M6
contract, `docs/superpowers/plans/2026-07-27-kg-m6-goal-prompt.md` in the
`wenlan-app` repository — an untracked file (artifact 11, F5).

---

## 0. How to read this catalog

A gate is a claim that some weakening of the system is detectable. This catalog
is the enumeration of those weakenings. Each row names **one** thing to break and
**one** test that must go RED when you break it.

**Status marks:**

| Mark | Meaning |
|---|---|
| **LIVE** | the substrate exists on this branch; the mutation is writable today |
| **PR-A** | needs schema or code that PR-A creates; writable the moment it lands |
| **lane 1** | prerequisite in flight — the claim-derivation promoter being built on `m5-truth-derivation`. Not vapor, not a follow-up: the RED test is specified here against the frozen predicate so it becomes executable the day that branch lands |
| **BLOCKED** | blocked pending the merge-no-survivor ruling. Not weakened, not dropped, not narrowed — carried at full strength with its dependency named |

> **Decision S0-134 — a gate clause with no mutation row in this catalog is not
> gated.** The gate definitions in the frozen contract are prose; prose passes by
> reading. A clause that nobody can name a breaking mutation for is a clause no
> test protects, and discovering that at review time is too late. The catalog is
> therefore the gate's index, and completeness against the clause list is itself
> checked (§13).

> **Decision S0-135 — every RED row is paired with a discriminating positive
> control that differs in exactly the mutated condition.** A test that goes red
> when you break something proves nothing unless it goes green when you do not:
> a test that fails for an incidental reason (missing fixture, wrong setup)
> looks identical to a working tooth. The standard is already practiced in this
> repo — commit `37b369b5` on `m5-truth-derivation` records it as *"Mutation-
> proven: with the evidence left in the page's own space the write SUCCEEDS and
> returns an edge id, so the tooth discriminates on the space condition rather
> than on any incidental failure."* Every row below inherits that bar.

> **Decision S0-136 — a `lane 1` row is specified against the frozen four-
> condition predicate, not against whatever the worker turns out to do.** The
> predicate is committed and stable at
> `docs/plans/2026-07-27-m5-truth-state-matrix.md:21`-`:35`; the implementation
> is not yet. Writing the tests against the document means the worker is tested
> against its contract rather than against itself, and it means these rows are
> reviewable now.

---

## 1. The `supported` predicate — the shared dependency

Six gate rows — G1.2, G2.5, G6.9, G7.1, G7.2, G7.3 — plus the six predicate
rows below all depend on a page being `supported`. The predicate has four
conditions, and **every** one must hold; any failure of any
kind yields `provisional`
(`docs/plans/2026-07-27-m5-truth-state-matrix.md:21`-`:35`):

1. an exact-page-version `claim_derivation_complete` marker exists whose recorded
   page-version digest **and** `extractor_version` both equal the current ones;
2. that marker's membership inventory is **nonempty**;
3. every active claim revision in the inventory has at least one `supports` edge
   that is simultaneously active, above threshold, and produced by a
   currently-eligible model version;
4. no claim revision in the inventory is in a deferred, timed-out, or malformed
   support state.

The matrix calls this *"a whitelist, not a blacklist — an unanticipated state is
provisional by construction rather than by enumeration"* (`:37`-`:38`).

### 1.1 The four condition mutations, plus the one that matters most

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| P1 | accept the marker on digest match alone, ignoring `extractor_version` | a page whose text is unchanged but whose extractor version moved reads `supported` | lane 1 |
| P2 | allow an empty inventory to satisfy the predicate | **see §1.2** | lane 1 |
| P3 | accept a `supports` edge that is inactive, below threshold, or from an ineligible model version | each of the three reads `supported` | lane 1 |
| P4 | ignore deferred / timed-out / malformed support states | a page with one deferred revision reads `supported` | lane 1 |
| P5 | publish a partial derivation run | a run completing some claims and failing others flips the page rather than leaving prior state | lane 1 |
| P6 | infer `support_status` from `human_reviewed`, or the reverse | either axis moves the other | lane 1 |

Positive control for P1–P6: one page satisfying all four conditions reads
`supported`, and reads `provisional` again after exactly one condition is
withdrawn.

### 1.2 The vacuous-truth mutation is mandatory and must be written first

The matrix singles this out: condition 2 exists to forbid *"all zero claims are
supported"*, and it is *"the single most dangerous state in the design, because
`∀x ∈ ∅` is trivially true in every natural implementation — a plain `.all()`
over an empty inventory returns `true` and silently marks an underived page as
fully supported"* (`:40`-`:47`).

> **Decision S0-137 — the empty-inventory mutation is a required RED row, and
> the positive control asserts a genuinely empty page stays permanently
> `provisional`.** This is the one mutation whose absence is invisible: the
> natural implementation is already broken, so a missing test does not fail —
> it silently passes on a system that marks every underived page supported. The
> matrix states the intent plainly: a genuinely empty page is *"permanently
> `provisional`. That is intended and is not a bug to be 'fixed' later"*
> (`:49`-`:50`).

**Why this is `lane 1` and not `BLOCKED`.** The M5 substrate is fully built and
fully inert — `backfill_page_truth_state` marks every page provisional with the
reason *"never evaluated: predates claim derivation"*
(`crates/wenlan-core/src/db/claim_identity.rs:511`-`:512`), and no production
code writes `supported`. Artifact 6 recorded that as a STOP. It is now a lane
with a branch, so the twelve dependent rows are specified, not deferred.

---

## 2. G1 — `m6_prerequisites_are_durable`

Two halves that fail differently: a runtime handshake and merge-time evidence
(`gp@wenlan-app:534`-`:550`).

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G1.1 | accept an incomplete or non-current M4 assignment generation | handshake passes on a stale generation | LIVE |
| G1.2 | accept M5 readiness below 100% | handshake passes at 99% | lane 1 |
| G1.3 | accept an incompatible app-contract version | handshake passes against an unsupported contract | PR-A |
| G1.4 | let the M5 automatic-reader manifest fail open | an unlisted reader is tolerated | LIVE |
| G1.5 | replace the old-binary fixture with a current daemon asserting old binaries refuse | **see below** | PR-A |
| G1.6 | drop an M4 or M5 squash commit from the ancestry list | `git merge-base --is-ancestor` is not run for it, or its failure does not block | PR-A |
| G1.7 | accept an unsigned or mismatched app compatibility manifest | a manifest whose artifact digest, daemon contract version, or supported schema range disagrees is accepted | PR-A |

**G1.5 is the row most likely to be discharged wrongly.** The contract is
explicit that *"a current daemon claiming that old binaries refuse is not
evidence"* (`gp@wenlan-app:540`-`:541`) — it requires launching the oldest
supported pre-M6 binary against a **copied** migrated database. Artifact 11
(S0-133) established the refusal mechanism is already live and correct; what is
missing is this evidence form.

**Positive control:** one fully ready fixture — all seven conditions true —
passes both halves.

---

## 3. G2 — `m6_genesis_counts_groups_not_rows`

Seven exclusions, each of which must be independently unable to inflate a signal
(`gp@wenlan-app:552`-`:557`).

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G2.1 | count generated roots as evidence | a signal crosses its threshold on generated roots alone | LIVE |
| G2.2 | count chunks of one document as independent | one document crosses a multi-group threshold | LIVE |
| G2.3 | count mirrors as independent | two mirrors of one source cross a two-group threshold | LIVE |
| G2.4 | count inactive or ungrounded external roots | a retracted root still contributes | LIVE |
| G2.5 | count M5-provisional page-mediated inputs | a provisional page's claims contribute | lane 1 |
| G2.6 | count same-session captures as independent | one capture session crosses a threshold | LIVE |
| G2.7 | count overview evidence toward concept coverage | an overview's own evidence inflates the concept signal | **BLOCKED** |
| G2.8 | move any D2 threshold by one unit without failing | each threshold's boundary test | LIVE |

**Why G2.7 is BLOCKED and stays at full strength.** The exclusion is stated over
*overview evidence*, which presupposes that every overview has a determinate
subject community. The merge-no-survivor case (artifact 8, F3) is precisely the
case where a merge retires every input community and no side survives to own the
resulting overview. An overview whose subject is undefined cannot be classified
as overview evidence or concept evidence, so the exclusion is not evaluable for
it, and the STOP's resolution decides which way it falls.

> **Decision S0-138 — G2.7 is carried at full strength with its dependency
> named, and the interim behavior does not become the contract.** Artifact 8's
> MG5 all-losers treatment stands as the safe interim, which means today the row
> is testable *under that interim*. Writing the test against the interim and
> marking the row green would convert a placeholder into a settled decision by
> the back door. The row stays BLOCKED until the ruling lands, and the interim
> test is labelled as testing the interim.

**Positive controls:** independent documents cross the threshold as expected;
UI-authorized human groups count.

---

## 4. G3 — `m6_overlapping_candidates_publish_once`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G3.1 | let candidates claiming `{1,2,3}` and `{2,3,4}` both publish | two concept pages exist for overlapping groups | PR-A |
| G3.2 | make the page ID non-deterministic | a retry produces a second ID | PR-A |
| G3.3 | allow two receipts for one publication | duplicate retry writes a second receipt | PR-A |
| G3.4 | drop group coverage on publish | a covered group reads uncovered | PR-A |
| G3.5 | leave an orphan claim after any exit | a claim survives its candidate | PR-A |
| G3.6 | let lease expiry hand the same work to two runners without a takeover check | both proceed | PR-A |
| G3.7 | let a manual run and the scheduler execute the same candidate | two publications | PR-A |

**The reservation exit matrix.** Eight terminal transitions — published, bounded
retry, review-required, stale, suppressed, superseded, exhausted retry,
compaction — and each must *atomically* leave every group in one of: permanent
coverage, an active bounded reservation, frontier, surfaced review, suppression,
or quarantine (`gp@wenlan-app:565`-`:568`).

> **Decision S0-139 — the exit matrix is 8 exits × 6 legal resting states
> asserted as a total function, not 8 tests that each assert "something
> reasonable happened."** The invariant is that no group falls out of all six.
> A per-exit test that checks the expected resting state passes while a group
> silently lands in none of them on some *other* exit; only asserting the
> disjunction over every group after every exit catches that. Crash injection is
> applied at each exit, since the contract says atomically.

**Positive control:** non-overlapping candidates both publish.

---

## 5. G4 — `m6_finalize_is_all_or_nothing`

Six concurrent events injected at **every** finalize boundary: root retraction,
dependency change, human edit, community generation change, lease expiry, claim
loss (`gp@wenlan-app:572`-`:576`).

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G4.1 | move any of the eleven components outside the one outer transaction | that component advances while another does not | PR-A |
| G4.2 | drop the CAS on any injected event | the finalize commits against changed state | PR-A |
| G4.3 | expose the filesystem projection before SQLite truth commits | stale or provisional prose is readable | PR-A |
| G4.4 | skip restart reconciliation of the outbox | a projection is left partially advanced | PR-A |
| G4.5 | hold a SQLite transaction, connection mutex, truth-state lock, or DB guard across the model call | the re-entrant provider test deadlocks | PR-A |

The eleven components are named in the contract: page, claim, support,
dependency, history, receipt, lease, candidate, coverage, truth, and durable
projection-outbox state.

> **Decision S0-140 — G4.3's assertion is "never stale or provisional", not
> "always present".** The contract deliberately permits a crash to leave the
> supported projection *temporarily absent* (`gp@wenlan-app:578`-`:580`), so a
> test asserting the file exists after every crash point is testing the wrong
> invariant and will be "fixed" by weakening the real one. The right assertion
> is the disjunction: after any crash, the projection is either absent or
> exactly the supported content — never stale, never provisional.

> **Decision S0-141 — G4.5's instrumentation asserts over held guards, not over
> elapsed time.** A latency-based check ("the model call took longer than the
> lock was held") passes on a fast model and fails on a slow machine. The
> assertion is that no guard is *held* across the call, which is a structural
> fact the instrumentation can observe directly.

**Positive control:** an uninterrupted finalize advances all eleven components
exactly once.

---

## 6. G5 — `m6_frontier_has_no_missing_root`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G5.1 | let an eligible uncovered group be in none of the six states | the differential query finds it | PR-A |
| G5.2 | drop cursor wrap handling | a group past the wrap is never visited | PR-A |
| G5.3 | lose frontier rows across restart | a group present before restart is absent after | PR-A |
| G5.4 | let quota exhaustion silently drop rather than defer | a group vanishes under quota pressure | PR-A |
| G5.5 | ignore a liveness transition | a group that became eligible is never surfaced | PR-A |
| G5.6 | compact away an unsurfaced group | compaction removes a group still owed surfacing | PR-A |
| G5.7 | break the seven-day surfacing clock | a group waits indefinitely | PR-A |

The differential query is the gate: every eligible uncovered root/group is
covered, claimed, waiting, surfaced, suppressed, or quarantined
(`gp@wenlan-app:591`-`:592`). G3's reservation crash matrix is repeated here
against the frontier invariant (`:595`-`:598`).

**Positive control:** a permanently small space surfaces **once** — not zero
times, and not repeatedly.

---

## 7. G6 — `m6_relevance_is_bounded_and_safe`

Catalogued in full in artifact 6; the mutation view:

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G6.1 | allow cross-space auto-attachment | a page attaches across a space boundary | LIVE |
| G6.2 | allow a known-negative to auto-attach | the negative case attaches | PR-A |
| G6.3 | move the 0.50 / 0.10 / 0.30 hysteresis boundaries by one unit | each boundary test | LIVE |
| G6.4 | let incremental pair state diverge from full recomputation | the oracle | PR-A |
| G6.5 | count generated, inactive, retracted, or legacy-ungrounded roots/edges in co-citation or common-neighbour statistics | each of the four contributes non-zero | LIVE |
| G6.6 | let `EXPLAIN QUERY PLAN` drift off the fixed indexes | the plan assertion | PR-A |
| G6.7 | let a group form more than `2016` pairs | the pair cap | PR-A |
| G6.8 | let candidate retrieval exceed `32` | the retrieval cap | PR-A |
| G6.9 | retrieve or attach a current M5-provisional candidate page | a provisional page is attached | lane 1 |
| G6.10 | drop the `64/d` hub weighting | a hub dominates | PR-A |
| G6.11 | exceed `64` adjacency rows / `4` queries / `512` rows / `50 ms` on a 5k-degree hub | the instrumented row-visit counters | PR-A |
| G6.12 | let any of the five CAS-losing events commit | each writes attachment/dependency/history/receipt state instead of requeueing | PR-A |
| G6.13 | let an embedding cross a threshold | the embedding tie-break test | PR-A |

**G6.11's instrumentation is the row that fails quietly.** The contract requires
*"instrumented row visits proving the bound rather than a textual SQL `LIMIT`"*
(`gp@wenlan-app:613`-`:614`) — a `LIMIT 512` on an unindexed predicate visits
the whole table and still reports 512 rows returned. Artifact 6's S0-98 names
the three counters.

**Positive controls:** direct co-citation attachment commits; qualified
co-citation attachment commits; a stable candidate set commits **exactly once**.

---

## 8. G7 — `m6_refresh_preserves_truth`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G7.1 | anchor an old supported claim to a wrong target revision | the anchoring assertion | lane 1 |
| G7.2 | resolve an ambiguous anchor by picking one | the whole result must reject, not choose | lane 1 |
| G7.3 | publish with support dropped for any claim | the result must reject entirely | lane 1 |
| G7.4 | let truth, page, history, dependencies, staleness, or receipt publish non-atomically | any one advances alone | PR-A |
| G7.5 | modify one byte of human prose | the byte-identity assertion | LIVE |
| G7.6 | emit two cards for one refresh | the coalescing assertion | PR-A |

> **Decision S0-142 — G7.2's RED assertion is that the *entire result* is
> rejected, not that the ambiguous claim is skipped.** The contract says wrong
> target, ambiguity, and dropped support *"reject the entire result"*
> (`gp@wenlan-app:626`). Skipping the ambiguous claim and publishing the rest is
> the natural implementation and the exact failure this gate exists to catch —
> it publishes a refresh whose claim inventory silently shrank. Artifact 7's
> bijection check (S0-58) is the mechanism.

**Positive control:** a safe machine refresh publishes.

---

## 9. G8 — `m6_overview_identity_survives_rebinding`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G8.1 | let a split change the surviving community's durable ID | ID stability | LIVE |
| G8.2 | let a merge change the winner's durable ID | ID stability | LIVE |
| G8.3 | let a partitioner swap rebind IDs wholesale | ID stability across algo version | LIVE |
| G8.4 | accept a label proposal against a stale generation | the stale-generation rejection | LIVE |
| G8.5 | allow a duplicate subscription row | uniqueness | PR-A |
| G8.6 | let a machine overwrite a human-edited overview | the ownership guard | LIVE |
| G8.7 | leave a merge loser attached | the detach rule (artifact 8, MG5) | **BLOCKED** |
| G8.8 | let a space overview resolve by title lookup | the collision hazard (artifact 8, S0-82) | LIVE |
| G8.9 | rename an overview silently on any of the above | no-silent-title-change | LIVE |
| G8.10 | lose subscription, proposal, or history rows on any of the above | no-data-loss | PR-A |

**G8.7 is the merge-no-survivor STOP's home clause.** Where G2.7 is blocked
*because* the STOP makes an exclusion unevaluable, G8.7 is blocked because the
detach rule itself is what the ruling decides. Same status, different reason,
and the distinction matters: a ruling that resolves G8.7 automatically resolves
G2.7, but not the reverse.

**Positive control:** a split and a merge that each have a determinate survivor
preserve IDs, titles, and every subscription.

---

## 10. G9 — `m6_old_writers_are_fenced`

The frozen contract names three mutations directly (`gp@wenlan-app:644`-`:645`):

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G9.1 | delete a manifest row | the structural CI test | LIVE (artifact 9) |
| G9.2 | add an unlisted wrapper | the structural CI test | LIVE |
| G9.3 | bypass one fence | the per-caller mutation test | PR-A |
| G9.4 | add an unlisted *consumer* (not just a wrapper) | the structural CI test | LIVE |
| G9.5 | make the enumeration depend on LSP or ast-grep output | **see below** | LIVE |

> **Decision S0-143 — G9.5 is a real mutation row, not a process note.** The
> contract says *"LSP/ast-grep generate review evidence but are not the gate by
> themselves"* (`gp@wenlan-app:642`-`:643`). That is easy to satisfy on day one
> and easy to erode later, when a maintainer replaces a slow structural
> enumeration with a fast tool query. The mutation: make the gate's enumeration
> read from a tool rather than from the compiled symbol set, and assert the gate
> now passes with an unlisted wrapper present — which is the whole point, since a
> tool that fails to parse a file reports zero matches rather than an error.

**Positive control:** the manifest exactly matching the production symbol set
passes, and every listed path routes through M6/M5 finalization after cutover.

---

## 11. G10 — `m6_app_is_version_and_coverage_safe`

Catalogued in artifact 10; the mutation view:

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G10.1 | accept a `schema_version` other than the pinned constant | the equality check | LIVE |
| G10.2 | declare a space ready on incomplete pagination | `interrupted` | LIVE |
| G10.3 | accept mismatched generations across the drained set | `generation-mismatch` | LIVE |
| G10.4 | drop any of the four cross-row coherence checks | each of `member-count-mismatch`, `unknown-community`, `conflicting-assignment`, `foreign-space` | LIVE |
| G10.5 | treat 404/405 as a transport error | `old-daemon` fallback is not taken | LIVE |
| G10.6 | mix durable and fallback assignments within a space | the no-mixing assertion | LIVE |
| G10.7 | drop the per-space segment from a fallback ID | duplicate local numeric IDs collide across spaces | LIVE |
| G10.8 | connect a region or edge across spaces | the cross-space assertion | LIVE |
| G10.9 | silently skip an unknown proposal variant | **the card must still render** (artifact 10, S0-115) | PR-A |
| G10.10 | let a stale action apply instead of conflicting | the typed conflict | PR-A |
| G10.11 | accept an M6 action without presence | each of the five mutating actions | PR-A |
| G10.12 | leave any of the thirteen rendered states missing in any of three locales | the enumeration assertion | PR-A |

**G10.9 is the row with a wrong-looking natural implementation.** Filtering
unknown variants out of the list is the default behavior of every renderer, and
a test that asserts "no crash" passes on it. The assertion must be that the card
is *present in the list*, shows an unsupported state, and offers no actions.

---

## 12. G11 — `m6_signal_cutover_is_independent`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G11.1 | share one readiness generation across signals | enabling one enables another | PR-A |
| G11.2 | share one canary across signals | one canary covers two | PR-A |
| G11.3 | let a failed later signal roll back an earlier healthy one | E2's failure disables E1 | PR-A |
| G11.4 | enable a genesis signal before maintenance is soaked | the soak precondition | PR-A |
| G11.5 | let the readiness row be a bare generation integer | the ABA sequence (artifact 11, S0-120) | PR-A |
| G11.6 | let an unreadable readiness row default to permitted | the fail-closed direction (artifact 11, S0-121) | PR-A |
| G11.7 | let a `committed` space return to `off` | the one-way door (artifact 11, S0-122) | PR-A |

**Positive control:** enabling E1 on one space leaves E2, E3, E4 disabled on that
space and every signal disabled on every other space.

---

## 13. Completeness, and how it is checked

> **Decision S0-144 — the catalog's completeness against the frozen gate clauses
> is itself a test, and it is a text-level test rather than a judgment call.**
> Each gate section in the frozen contract is a list of clauses. PR-A adds a test
> that parses those clause lists and asserts every clause maps to at least one
> catalog row ID. Without it, a clause added to the contract later acquires no
> mutation row and nobody notices — which is exactly the failure mode S0-134
> describes, arriving through drift instead of oversight.

Current coverage: **11 gates, 87 gate mutation rows, 6 predicate rows, 93
total.**

| Status | Rows |
|---|---|
| LIVE | 31 |
| PR-A | 48 |
| lane 1 | 12 |
| BLOCKED | 2 |

(Counts are mechanical — every catalog row is a table line whose ID matches
`G<n>.<n>` or `P<n>`, and the status is its last cell.)

---

## 14. Findings

**F1 — two gates are blocked by one ruling, and resolving them is ordered.**
G8.7 (merge loser detach) and G2.7 (overview evidence exclusion) are both blocked
on the merge-no-survivor STOP, but not symmetrically: G8.7 *is* the rule the
ruling decides, and G2.7 is downstream of it, because an overview with no
determinate subject cannot be classified either way. A ruling on G8.7 therefore
resolves both; a ruling that addressed only the G2 exclusion would leave the
detach rule undecided. Worth knowing when the ruling is drafted, so it is not
scoped to the narrower question.

**F2 — twelve of the ninety-three rows cannot go red until lane 1 lands, and
they are concentrated in the gates that protect truth.** G1.2, G2.5, G6.9,
G7.1–G7.3, and the six predicate rows all rest on a page being `supported`, and no
production code writes that value today
(`crates/wenlan-core/src/db/claim_identity.rs:511`-`:512`). This is not an
argument for weakening them — it is an argument for the ordering the lead has
already set: the promoter is a prerequisite of PR-B, and the catalog is written
so those rows become executable on the day it lands rather than needing design
then.

**F3 — three rows are gates against a *natural* implementation, meaning their
absence is silent.** P2 (vacuous truth over an empty inventory), G7.2 (skipping
an ambiguous claim rather than rejecting the result), and G10.9 (filtering an
unknown variant out of a list) share a shape: the broken behavior is what an
ordinary implementation does by default, and the corresponding "reasonable" test
passes on the broken version. A missing test here does not show up as a gap in
coverage; it shows up as a green build on a system that is wrong. These three
should be written before the code they gate, not after.

**F4 — the catalog cannot express "this gate is green" and should not try.**
Every row here is a mutation and a required RED. None of it says the
corresponding positive path works today. That is deliberate — artifact 6's
experience was that a bounds check reported clean while the content was wrong,
and a catalog that tracked its own pass/fail state would invite exactly that
confusion. Gate status lives in CI; this document is the specification of what CI
must check.

---

## 15. Decisions introduced here

`S0-134` a gate clause with no mutation row is not gated; the catalog is the gate's index ·
`S0-135` every RED row is paired with a discriminating positive control differing in exactly the mutated condition ·
`S0-136` `lane 1` rows are specified against the frozen four-condition predicate, not against the eventual implementation ·
`S0-137` the empty-inventory vacuous-truth mutation is required, and a genuinely empty page stays permanently provisional ·
`S0-138` G2.7 is carried at full strength; the interim treatment is labelled as interim and does not become the contract ·
`S0-139` the reservation exit matrix is asserted as a total function over 8 exits × 6 resting states ·
`S0-140` G4.3 asserts "never stale or provisional", not "always present" ·
`S0-141` G4.5 asserts over held guards, not elapsed time ·
`S0-142` G7.2 rejects the entire result; skipping the ambiguous claim is the failure ·
`S0-143` G9.5 is a mutation row — the enumeration may not come from a tool query ·
`S0-144` catalog completeness against the frozen clause lists is itself a test.
