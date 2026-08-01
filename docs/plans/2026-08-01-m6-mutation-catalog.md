# M6 Stage-0 artifact 12 — RED mutation catalog and positive controls

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: Stage-0 item 12 — a RED mutation catalog and positive controls for
**every** gate G1–G11.
Continues the decision numbering from artifact 11 (`S0-133`).

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
| **RULING** | the clause is frozen but its mapping waits on a ruling; the row is an index entry, **not** coverage (S0-160). *No row carries this as of rev 8 — R-1 ruled and G6.14 became a real case. The status stays defined for the next such clause.* |
| **BLOCKED** | blocked pending the merge-no-survivor ruling. Not weakened, not dropped, not narrowed — carried at full strength with its dependency named. *No row carries this as of rev 8 — R-3 ratified as S0-163 and G8.7/G2.7 unblocked. The status stays defined, and the full-strength discipline is why the rows were testable the moment the ruling landed.* |

> **Decision S0-160 *(rev 4, round-3 N5)* — a frozen clause whose mapping waits
> on a ruling carries a `RULING` row; it is never simply absent.** Withdrawing
> rev 3's visit proof (artifact 6, S0-157) left G6's visit clause with no row at
> all, which under S0-134 means *not gated* — correct, but invisible. An absent
> row and an unnoticed clause look identical in this catalog, and the catalog is
> the gate's index, so invisibility is the one failure it cannot afford.
>
> A `RULING` row names the clause, names the decision it waits on, and counts. It
> is not coverage and must never be totalled as though it were: the status exists
> precisely so the hole is legible in the coverage table instead of hiding in the
> difference between two numbers. When the ruling lands, the row becomes a normal
> mutation row with a real control, or the clause is restated and the row is
> retired with a note. **G6.14 was the only one, and it retired on 2026-08-01
> when R-1 ruled (b′): the row is now a normal mutation case with a real
> control.** The status stays defined rather than being deleted with its last
> row — the next frozen clause that outruns its instrument needs somewhere to
> land, and re-deriving this decision at that moment is exactly the invisibility
> S0-160 exists to prevent.
>
> *(rev 5, round-4 item 4: rev 4 filed this row as `G10.16`, inside the G10 app
> gate. The clause is G6's (`gp@wenlan-app:600`) and extends G6.11's limits, and
> S0-144's completeness test resolves a clause to a row by the row ID's gate
> prefix — so a `G10.` row could never satisfy a G6 clause, and the row would
> have read as covered in one gate while leaving a hole in another. Refiled as
> `G6.14`, which makes the mapping machine-resolvable by construction and needs
> no separate mapping table. G10's numbering is unaffected: 16 was its last
> row.)*

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

> **Decision S0-154 *(rev 2, finding 7)* — a mutation row names exactly one
> condition, and a row that named several is split into lettered sub-rows keeping
> its number.** S0-135 requires a control "differing in exactly the mutated
> condition", and that is unsatisfiable for a row that mutates four things at
> once: whichever way the bundled test goes, it does not say which condition
> carried it. Rev 1 had thirteen such rows. They are now lettered (`G6.5a`…`d`
> and so on), which keeps every original number resolvable while making each line
> one testable claim.

> **Decision S0-155 *(rev 2, finding 7)* — each row's positive control is derived
> from the row by a fixed rule, and the derivation is the contract: for a row
> reading "weaken X", the control is the identical fixture with X intact,
> asserted to reach its success outcome.** Rev 1 gave one control per gate
> section — three controls for thirteen G6 rows — which is not what S0-135 says
> and does not discriminate: a section-level control passing tells you the
> section's happy path works, not that row 7's tooth bites on row 7's condition.
> A stated derivation rule is better than 203 hand-written cells, which would
> drift; it works precisely *because* S0-154 makes every row single-conditioned.
> Where the derivation does not apply, the row carries an explicit control in its
> RED column and says so.

> **Decision S0-136 — a `lane 1` row is specified against committed frozen text,
> not against whatever the worker turns out to do.** For the nine predicate rows
> in §1 that text is the frozen four-condition predicate
> (`docs/plans/2026-07-27-m5-truth-state-matrix.md:21`-`:35`); the implementation
> is not yet committed. Writing the tests against the document means the worker
> is tested against its contract rather than against itself, and it means these
> rows are reviewable now.
>
> *(rev 2, finding 12: rev 1 said every `lane 1` row is specified against the
> four conditions. P6b is `lane 1` and is not — it is the axis-independence
> claim that sits beside the predicate, whose frozen text is
> `:14`-`:17` of the same matrix. The rule is the same, the source text is a
> different paragraph, and rev 1's wording denied that the distinction existed.
> See "P6 is not predicate coverage" below.)*

---

## 1. The `supported` predicate — the shared dependency

Six gate rows — G1.2, G2.5, G6.9, G7.1, G7.2, G7.3 — plus the nine predicate
rows below all depend on a page being `supported`. (The two P6 rows in the
following section do not: P6 is about axis independence, not about the
predicate — see below.) The predicate has four
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
| P3a | accept an **inactive** `supports` edge | a page whose only edge is inactive reads `supported` | lane 1 |
| P3b | accept a **below-threshold** `supports` edge | a page whose only edge is below threshold reads `supported` | lane 1 |
| P3c | accept an edge from an **ineligible model version** | a page whose only edge is from an ineligible model reads `supported` | lane 1 |
| P4a | ignore the **deferred** support state | a page with one deferred revision reads `supported` | lane 1 |
| P4b | ignore the **timed-out** support state | a page with one timed-out revision reads `supported` | lane 1 |
| P4c | ignore the **malformed** support state | a page with one malformed revision reads `supported` | lane 1 |
| P5 | publish a partial derivation run | a run completing some claims and failing others flips the page rather than leaving prior state | lane 1 |

*(rev 4, round-3 group 7: rev 3's G10.13a–c named a locale but not a **cell** —
"an M6 state" in that locale, which is not one condition and cannot have a control
differing in exactly one. S0-117 requires each of thirteen states present and
non-empty in each of three locales, so the contract's own unit is the
state×locale cell and there are exactly **39** of them. The letters a…m index
S0-117's thirteen states in its order, so `G10.14f` is unambiguously
`zh-Hans` × "overview transfer proposal". The three row-families materialize all
39 and are counted as 39 by the mechanical rule below.)*

*(rev 5, round-4 item 2: S0-117's row 1 reads "candidate card, one per variant"
and does not itself order the four variants, so a…d were not yet unique. **The
order is `M6CandidateCard`'s declaration order** (artifact 10 §5,
`docs/plans/2026-08-01-m6-app-wire.md:200`-`:209`): `a` = EvidenceCluster,
`b` = OrphanWikilink, `c` = CommunityOverview, `d` = SpaceOverview. Declaration
order rather than an invented one, because the enum is the thing the renderer
switches on — any other order would make the catalog and the code disagree about
which cell a failing row names. `e`…`m` then follow S0-117's rows 5…13 as
already stated.)*

*(rev 2, finding 12: rev 1's P3 and P4 each bundled three states behind a single
row, and P4's RED column asserted only the deferred one — timed-out and malformed
had no assertion at all, so two of condition 4's three states were uncatalogued
while the row was counted as covering them. Split per S0-154; the assertions are
now one per state.)*

Positive control (per S0-155): one page satisfying all four conditions reads
`supported`, and reads `provisional` again after exactly the one condition of
that row is withdrawn.

**P6 is not predicate coverage, and it is LIVE.** Rev 1 listed it here as a
seventh predicate mutation, which inflated the predicate's coverage count: the
frozen predicate has four conditions
(`docs/plans/2026-07-27-m5-truth-state-matrix.md:21`-`:35`), and "infer
`support_status` from `human_reviewed`, or the reverse" is not one of them — it is
the **axis-independence** claim that sits beside the predicate. It is also no
longer `lane 1`. Since `#418` the mutation is writable today:

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| P6a | make the review endpoint write `support_status = 'supported'` alongside `human_reviewed = 1` | a page reads `supported` on the strength of a human having opened it | **LIVE** |
| P6b | make a machine promotion to `supported` also set `human_reviewed = 1` | the human axis moves without a human | lane 1 |

P6a is live because the exact seam exists and is deliberately written the other
way: the review upsert inserts `'provisional'` with a NULL `evaluated_at`, and
says why — *"a human saying 'I read this' is not evidence about whether the
machine found support — inventing `supported` here would collapse the separation
the whole rung exists to make"*
(`crates/wenlan-core/src/db/presence_review.rs:321`-`:325`, write at `:326`-`:341`).
Changing that literal is a one-word mutation against shipped code, which makes
P6a the only tooth in this section that can be proven today.

### 1.2 The vacuous-truth mutation is mandatory and must be written first

The matrix singles this out: condition 2 exists to forbid *"all zero claims are
supported"*, and it is *"the single most dangerous state in the design, because
`∀x ∈ ∅` is trivially true in every natural implementation — a plain `.all()`
over an empty inventory returns `true` and silently marks an underived page as
fully supported"* (`docs/plans/2026-07-27-m5-truth-state-matrix.md:42`-`:47`).

> **Decision S0-137 — the empty-inventory mutation is a required RED row, and
> the positive control asserts a genuinely empty page stays permanently
> `provisional`.** This is the one mutation whose absence is invisible: the
> natural implementation is already broken, so a missing test does not fail —
> it silently passes on a system that marks every underived page supported. The
> matrix states the intent plainly: a genuinely empty page is *"permanently
> `provisional`. That is intended and is not a bug to be 'fixed' later"*
> (`docs/plans/2026-07-27-m5-truth-state-matrix.md:49`-`:50`).

**Why this is `lane 1` and not `BLOCKED`.** The M5 substrate is fully built, and
no production writer promotes `supported` — `backfill_page_truth_state` marks
every page provisional with the reason *"never evaluated: predates claim
derivation"* (`crates/wenlan-core/src/db/claim_identity.rs:511`-`:512`), and a
repository-wide search for the literal `'supported'` at `1c903bec` finds no
production hit. ("Fully inert" would be the wrong word since `#418`: the human
axis now has a live writer. See artifact 6 §0.) Artifact 6 recorded that as a STOP. It is now a lane
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
| G1.7a | accept a manifest whose **artifact digest** disagrees | that manifest is accepted | PR-A |
| G1.7b | accept a manifest whose **daemon contract version** disagrees | that manifest is accepted | PR-A |
| G1.7c | accept a manifest whose **supported schema range** disagrees | that manifest is accepted | PR-A |
| G1.7d | accept an **unsigned** manifest | an unsigned manifest is accepted | PR-A |

**G1.5 is the row most likely to be discharged wrongly.** The contract is
explicit that *"a current daemon claiming that old binaries refuse is not
evidence"* (`gp@wenlan-app:540`-`:541`) — it requires launching the oldest
supported pre-M6 binary against a **copied** migrated database. Artifact 11
(S0-133) established the refusal mechanism is already live and correct; what is
missing is this evidence form.

**Suite-level control** (row controls are per S0-155): one fully ready fixture — all seven conditions true —
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
| G2.4a | count an **inactive** external root | an inactive root contributes | LIVE |
| G2.4b | count an **ungrounded** external root | an ungrounded root contributes | LIVE |
| G2.5 | count M5-provisional page-mediated inputs | a provisional page's claims contribute | lane 1 |
| G2.6 | count same-session captures as independent | one capture session crosses a threshold | LIVE |
| G2.7 | count overview evidence toward concept coverage | an overview's own evidence inflates the concept signal | PR-A |
| G2.8 | move any D2 threshold by one unit without failing | each threshold's boundary test | LIVE |

**G2.7 unblocked 2026-08-01.** The exclusion is stated over *overview evidence*,
which presupposes that every overview has a determinate subject community — the
one thing the merge-no-survivor case (artifact 8, F3) did not supply, since a
merge that retires every input community leaves no side to own the resulting
overview. S0-163's D11 amendment supplies it: in an all-losers merge every
participant is a loser, so every overview has a determinate disposition and the
exclusion is evaluable again.

*(The dependency direction this catalog recorded held. It said a ruling on G8.7's
clause resolves G2.7 and not the reverse; the ruling went to G8.7's clause and
both unblocked. Worth one line because it is the artifact set predicting its own
resolution correctly rather than being corrected by it.)*

> **Decision S0-138 — G2.7 is carried at full strength with its dependency
> named, and the interim behavior does not become the contract.** Artifact 8's
> MG5 all-losers treatment stands as the safe interim, which means today the row
> is testable *under that interim*. Writing the test against the interim and
> marking the row green would convert a placeholder into a settled decision by
> the back door. The row stays BLOCKED until the ruling lands, and the interim
> test is labelled as testing the interim.
>
> *(amended rev 8: ruled 2026-08-01. The interim did **not** become the contract
> by default — it was ratified deliberately as S0-163, which is the outcome this
> decision was written to force rather than to prevent. The row's status moves to
> PR-A and the test stops being labelled as testing an interim.)*

**Suite-level controls** (row controls are per S0-155): independent documents cross the threshold as expected;
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

**Suite-level control** (row controls are per S0-155): non-overlapping candidates both publish.

---

## 5. G4 — `m6_finalize_is_all_or_nothing`

Six concurrent events injected at **every** finalize boundary: root retraction,
dependency change, human edit, community generation change, lease expiry, claim
loss (`gp@wenlan-app:572`-`:576`).

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G4.1a–k | move **one** of the eleven components outside the one outer transaction — one row per component | that component advances while another does not | PR-A |
| G4.2a–g | drop the CAS on **one** injected event — one row per event: failure, concurrent root retraction, dependency change, human edit, community generation change, lease expiry, claim loss | the finalize commits against the state that event changed | PR-A |
| G4.3 | expose the filesystem projection before SQLite truth commits | stale or provisional prose is readable | PR-A |
| G4.4 | skip restart reconciliation of the outbox | a projection is left partially advanced | PR-A |
| G4.5a–d | hold one of the four across the model call — one row per holder: SQLite transaction, connection mutex, truth-state lock, DB guard | the re-entrant provider test deadlocks on that holder | PR-A |

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

**Suite-level control** (row controls are per S0-155): an uninterrupted finalize advances all eleven components
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

**Suite-level control** (row controls are per S0-155): a permanently small space surfaces **once** — not zero
times, and not repeatedly.

---

## 7. G6 — `m6_relevance_is_bounded_and_safe`

Catalogued in full in artifact 6; the mutation view:

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G6.1 | allow cross-space auto-attachment | a page attaches across a space boundary | PR-A |
| G6.2 | allow a known-negative to auto-attach | the negative case attaches | PR-A |
| G6.3a | move the `0.50` assign boundary by one unit | the assign boundary test | LIVE |
| G6.3b | move the `0.30` drop boundary by one unit | the drop boundary test | LIVE |
| G6.3c | move the `0.10` top-two margin by one unit | the margin boundary test | PR-A |
| G6.4 | let incremental pair state diverge from full recomputation | the oracle | PR-A |
| G6.5a | count **generated** roots/edges in co-citation or common-neighbour statistics | generated input contributes non-zero | PR-A |
| G6.5b | count **inactive** roots/edges | inactive input contributes non-zero | PR-A |
| G6.5c | count **retracted** roots/edges | retracted input contributes non-zero | PR-A |
| G6.5d | count **legacy-ungrounded** roots/edges | legacy-ungrounded input contributes non-zero | PR-A |
| G6.6 | let `EXPLAIN QUERY PLAN` drift off the fixed indexes | the plan assertion | PR-A |
| G6.7 | let a group form more than `2016` pairs | the pair cap | PR-A |
| G6.8 | let candidate retrieval exceed `32` | the retrieval cap | PR-A |
| G6.9 | retrieve or attach a current M5-provisional candidate page | a provisional page is attached | lane 1 |
| G6.10 | drop the `64/d` hub weighting | a hub dominates | PR-A |
| G6.11a | exceed `64` adjacency rows for one endpoint on a 5k-degree hub | the adjacency-row cap assertion | PR-A |
| G6.11b | issue a fifth query in one evaluation | the query counter (S0-95 instrument 2) | PR-A |
| G6.11c | materialize more than `512` rows in one evaluation | the decoded-row counter (S0-95 instrument 1) | PR-A |
| G6.11d | exceed `50 ms` on a 5k-degree hub | `R-BENCH-MAX` (S0-98) | PR-A |
| G6.12a–e | let a CAS-losing event commit — one row per event: provisionalization, root retraction, community rebinding, relevance-stat update, candidate-set change | that event writes attachment/dependency/history/receipt state instead of requeueing | PR-A |
| G6.13 | let an embedding cross a threshold | the embedding tie-break test | PR-A |
| G6.14 | exceed `2,176` visited index entries in one evaluation | `SQLITE_SCANSTAT_NVISIT` asserted on the bench-only instrumented build (S0-157, R-1 option (b′), ruled 2026-08-01) | PR-A |

**G6.11's instrumentation is the row that fails quietly.** The contract requires
*"instrumented row visits proving the bound rather than a textual SQL `LIMIT`"*
(`gp@wenlan-app:613`-`:614`) — a `LIMIT 512` on an unindexed predicate visits
the whole table and still reports 512 rows returned. Artifact 6's **S0-95** names
the four instruments (rev 2, finding 16: rev 1 cited S0-98, which is the 50 ms
budget, and counted three instruments before S0-95 gained the row-visit counter).

> **Rev 2, finding 8 — three G6 rows were marked LIVE against substrate that does
> not exist.** Checked at `1c903bec`: a repository-wide search finds **no**
> co-citation or common-neighbour code and **no** auto-attachment code, so G6.1
> and the whole of G6.5 are PR-A, not LIVE. G6.3 was two-thirds true —
> `COMMUNITY_ROUTE_ASSIGN_THRESHOLD = 0.50` and
> `COMMUNITY_ROUTE_DROP_THRESHOLD = 0.30` are real and mutable today
> (`crates/wenlan-core/src/community_routing.rs:7`-`:8`), while the `0.10` top-two
> margin has no antecedent at all, which is artifact 6's own F3. Splitting the row
> per S0-154 is what let the true and false halves separate; bundled, the row
> could only carry one status and it carried the flattering one.

**Positive controls** are per-row by the S0-155 derivation. The three
section-level controls rev 1 listed here — direct co-citation attachment commits,
qualified co-citation attachment commits, a stable candidate set commits exactly
once — are retained as **G6 suite-level** sanity checks, which is what they always
were; they are not row controls and no longer stand in for any.

---

## 8. G7 — `m6_refresh_preserves_truth`

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G7.1 | anchor an old supported claim to a wrong target revision | the anchoring assertion | lane 1 |
| G7.2 | resolve an ambiguous anchor by picking one | the whole result must reject, not choose | lane 1 |
| G7.3 | publish with support dropped for any claim | the result must reject entirely | lane 1 |
| G7.4a–f | let one of the six publish non-atomically — one row per kind: truth, page, history, dependencies, staleness, receipt | that one advances alone | PR-A |
| G7.5 | modify one byte of human prose | the byte-identity assertion | LIVE |
| G7.6 | emit two cards for one refresh | the coalescing assertion | PR-A |

> **Decision S0-142 — G7.2's RED assertion is that the *entire result* is
> rejected, not that the ambiguous claim is skipped.** The contract says wrong
> target, ambiguity, and dropped support *"reject the entire result"*
> (`gp@wenlan-app:626`). Skipping the ambiguous claim and publishing the rest is
> the natural implementation and the exact failure this gate exists to catch —
> it publishes a refresh whose claim inventory silently shrank. Artifact 7's
> bijection check (S0-58) is the mechanism.

**Suite-level control** (row controls are per S0-155): a safe machine refresh publishes.

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
| G8.7a | leave a merge loser attached when the merge HAS a determinate survivor | the detach rule (artifact 8, MG1–MG4) | PR-A |
| G8.7b | leave any participant attached when the merge has NO survivor | the all-losers detach rule (artifact 8, MG5 + S0-163) | PR-A |
| G8.8 | let a space overview resolve by title lookup | the collision hazard (artifact 8, S0-82) | LIVE |
| G8.9 | rename an overview silently on any of the above | no-silent-title-change | LIVE |
| G8.10a | lose **subscription** rows on any of the above | no-data-loss, subscription | PR-A |
| G8.10b | lose **proposal** rows on any of the above | no-data-loss, proposal | PR-A |
| G8.10c | lose **history** rows on any of the above | no-data-loss, history | PR-A |
| G8.11a | derive M6 identity from the space **name** instead of `spaces.id` | renaming a space re-keys a `slot_id`, a `page_id`, or a card | PR-A |
| G8.11b | let a rename silently drop a card that minted before it | the card stops appearing with no refusal and no receipt | PR-A |
| G8.11c | skip the rename closure over the space-keyed substrate | a row keyed to the old name survives the rename and stays claimable | PR-A |

**G8.7 is the merge-no-survivor STOP's home clause.** Where G2.7 is blocked
*because* the STOP makes an exclusion unevaluable, G8.7 is blocked because the
detach rule itself is what the ruling decides. Same status, different reason,
and the distinction matters: a ruling that resolves G8.7 automatically resolves
G2.7, but not the reverse.

**G8.11a-c is the rename case, and it is one scenario asserted three ways**
*(new rev 6, round-5 blocker)*. Rename a space that already has M6 cards, then
assert all three: identity is unchanged (`slot_id`, `page_id`, and card identity
survive, because the digest input is `spaces.id` and not the name — artifact 10,
S0-161), nothing disappears silently (a card that minted before the rename still
mints after it, and any refusal is an explicit fail-closed refusal rather than an
absence), and no orphaned state is left behind (no row in the eleven-table
community substrate still names the old space — artifact 10, S0-162, whose
enumeration is closed and whose one deliberate exclusion is recorded there).

The three are separated because they fail independently and a single assertion
would hide two of them: stable-ID derivation alone satisfies G8.11a while still
failing G8.11b, since a stale `communities.space` misses the lookup no matter
what the digest is made of. G8.11c is the one that has to be *added* to the
substrate rather than merely observed — `update_space`'s cascade stops at
memories, entities, and pages today (`db.rs:19266`-`:19298`), and the scheduler
claims dirty spaces without joining a live `spaces` row
(`db/community_grouping_state.rs:84`), so an orphaned row is not guaranteed to
retire on its own.

**Suite-level control** (row controls are per S0-155): a split and a merge that each have a determinate survivor
preserve IDs, titles, and every subscription; a rename with no M6 cards in the
space is a no-op on every count.

---

## 10. G9 — `m6_old_writers_are_fenced`

The frozen contract names three mutations directly (`gp@wenlan-app:644`-`:645`):

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G9.1 | delete a row from the **D12 M6 writer** manifest | its structural CI test | PR-A |
| G9.2 | add an unlisted wrapper | the same test | PR-A |
| G9.3 | bypass one fence | the per-caller mutation test | PR-A |
| G9.4 | add an unlisted *consumer* (not just a wrapper) | the same test | PR-A |
| G9.5 | make the enumeration depend on LSP or ast-grep output | **see S0-143** | PR-A |
| G9.6 | delete a row from the **M5 reader** inventory | `drift_guard::m5_reader_inventory_matches_current_tree` | **LIVE** |

> **Decision S0-143 — G9.5 is a real mutation row, not a process note.** The
> contract says *"LSP/ast-grep generate review evidence but are not the gate by
> themselves"* (`gp@wenlan-app:642`-`:643`). That is easy to satisfy on day one
> and easy to erode later, when a maintainer replaces a slow structural
> enumeration with a fast tool query. The mutation: make the gate's enumeration
> read from a tool rather than from the compiled symbol set, and assert the gate
> now passes with an unlisted wrapper present — which is the whole point, since a
> tool that fails to parse a file reports zero matches rather than an error.

> **Rev 2, finding 8 — G9.1, G9.2, G9.4 and G9.5 were marked LIVE against a test
> that does not exist for this manifest.** There is exactly one structural
> manifest tooth in the tree at `1c903bec`, and it guards the **M5 reader**
> inventory: `m5_reader_inventory_matches_current_tree`
> (`crates/wenlan-core/src/drift_guard.rs:5718`-`:5737`) shells out to
> `scripts/m5-reader-sweep.py --check` and fails on drift. The **D12 M6 writer**
> manifest that G9 is actually about, and its structural test, are both PR-A. Rev
> 1's LIVE marks conflated the two manifests — the same words ("the structural CI
> test") named a real tooth and an unbuilt one. The rows are now PR-A, and the
> live tooth is catalogued as its own row G9.6, which is worth having: it is the
> working precedent the D12 test should be modelled on, and mutating it proves the
> pattern bites before PR-A copies it.

**Suite-level control:** the manifest exactly matching the production symbol set
passes, and every listed path routes through M6/M5 finalization after cutover.
Row controls are per S0-155.

---

## 11. G10 — `m6_app_is_version_and_coverage_safe`

Catalogued in artifact 10; the mutation view:

| # | Weaken this | Must go RED | Status |
|---|---|---|---|
| G10.1 | accept a `schema_version` other than the pinned constant | the equality check | LIVE |
| G10.2 | declare a space ready on incomplete pagination | `interrupted` | LIVE |
| G10.3 | accept mismatched generations across the drained set | `generation-mismatch` | LIVE |
| G10.4a–d | drop one cross-row coherence check — one row each for `member-count-mismatch`, `unknown-community`, `conflicting-assignment`, `foreign-space` | that check's assertion | LIVE |
| G10.5 | treat 404/405 as a transport error | `old-daemon` fallback is not taken | LIVE |
| G10.6 | mix durable and fallback assignments within a space | the no-mixing assertion | LIVE |
| G10.7 | drop the per-space segment from a fallback ID | duplicate local numeric IDs collide across spaces | LIVE |
| G10.8 | connect a region or edge across spaces | the cross-space assertion | LIVE |
| G10.9 | silently skip an unknown proposal variant | **the card must still render** (artifact 10, S0-115) | PR-A |
| G10.10 | let a stale action apply instead of conflicting | the typed conflict | PR-A |
| G10.11a–e | accept one M6 action without presence — one row per action: candidate create, merge, dismiss, overview transfer, overview retire | that action commits unauthenticated | PR-A |
| G10.12a–m | drop one rendered state from the M6 enumeration — one row per state, the thirteen artifact 10's S0-117 enumerates | that state's present-and-non-empty assertion | PR-A |
| G10.13a–m | **`en`** has no own string for state a…m — one row per state | that cell renders the raw key | PR-A |
| G10.14a–m | **`zh-Hans`** has no own string for state a…m — one row per state | that cell silently renders the English fallback | PR-A |
| G10.15a–m | **`zh-Hant`** has no own string for state a…m — one row per state | that cell silently renders the English fallback | PR-A |

**G10.9 is the row with a wrong-looking natural implementation.** Filtering
unknown variants out of the list is the default behavior of every renderer, and
a test that asserts "no crash" passes on it. The assertion must be that the card
is *present in the list*, shows an unsupported state, and offers no actions.

**Suite-level control** (rev 2, finding 7 — rev 1 supplied none for G10, the only
gate with no control at all): a fully ready space at a matching contract version
renders every card kind, offers exactly the actions its presence state permits,
and completes one typed daemon/Tauri/TypeScript action round trip end to end
(artifact 10, S0-150). Row controls are per S0-155.

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

**Suite-level control** (row controls are per S0-155): enabling E1 on one space leaves E2, E3, E4 disabled on that
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

Current coverage: **11 gates, 193 gate mutation cases, 11 predicate cases, and
no `RULING` placeholder — 204 rows counted mechanically.**

*(rev 5, round-4 item 5: rev 4 labelled the then-200 as "189 gate mutation
cases", which counted the `RULING` row as a mutation and contradicted S0-160's
own sentence that it must never be totalled as coverage. The row is still a row,
but the semantic split keeps it out of the coverage figure.)*

*(rev 6, round-5 blocker: +3 — `G8.11a`-`c`, the rename case. 188 → 191 gate
mutation cases, 200 → 203 mechanical rows, PR-A 149 → 152. No claim was
reinterpreted and no row was split; this is three genuinely new assertions
arriving with S0-161's reversal and S0-162.)*

| Status | Cases |
|---|---|
| LIVE | 32 |
| PR-A | 156 |
| lane 1 | 16 |
| BLOCKED | 0 |
| RULING | 0 |

(Counts are mechanical — every catalog row is a table line whose ID matches
`G<n>.<n>[a-z]` or `P<n>[a-z]`, and the status is its last cell. A row written
`G4.5a–d` is one line standing for four cases and counts as four.)

**`RULING` is a status, and it is not a synonym for "not yet written."** A row
marked `RULING` names a frozen clause that this catalog deliberately does not
map, because the mapping depends on a decision that has not been made. S0-134
says an unmapped clause is not gated; a clause that is silently absent is
indistinguishable from one nobody noticed, which is the exact failure S0-134
exists to catch. Carrying the row with an explicit `RULING` status keeps the
index complete and makes the hole countable. **No row carries it today** —
G6.14 held it until R-1 ruled on 2026-08-01, and the count is now zero. The
status and its filing rule survive the row: a `RULING` row is filed under the
gate that owns the clause, not the gate that happened to be under revision when
the hole was found, because S0-144 resolves a clause to a row by the row's gate
prefix and filing it elsewhere would leave the clause unmapped while looking
mapped.

*(rev 8: the morning rulings landed and **BLOCKED reached zero**. G6.14 stops
being a `RULING` placeholder and becomes a real case (R-1 → (b′)); G8.7 and G2.7
unblock (R-3 ratified as S0-163). 203 → 204: three rows changed status and
exactly one was added — `G8.7b`, the all-losers merge. That row exists because
the ruling changed what the contract *says*, and a changed contract should leave
a test behind: "no survivor" is not an instance of "a loser stays attached", so
covering it inside G8.7's prose would have left exactly the unmapped clause
S0-134 exists to catch.)*

*(rev 6: rev 5 reported 200 and rev 6 reports 203. `G8.11a`-`c` — the rename
case — arrives with S0-161's reversal to stable-`spaces.id` identity and with
S0-162's rename-closure rule. Unlike every earlier movement in this list, these
three ARE new claims rather than a regrain: nothing in rev 5 asserted anything
about a rename, because rev 5's identity rule made the rename gap a documented
behaviour instead of a tested one.)*

*(rev 5: total unchanged at 200. Round 4's item 5 corrected only the semantic
split — 188 gate mutations + 1 `RULING` + 11 predicate cases — after rev 4
totalled the `RULING` row as coverage.)*

*(rev 4: rev 3 reported 163. Round 3's group 7 held G10.13a–c non-concrete —
it named a locale but never the state×locale **cell** S0-135 requires — so the
locale condition is now one row per locale per state: G10.13a–m (`en`),
G10.14a–m (`zh-Hans`), G10.15a–m (`zh-Hant`), the 39 cells S0-117 demands.
G6.14 adds the `RULING` row for G6's visit clause. Total **200**. Again no new
claim: the same coverage, spelled at the grain the contract asks for.)*

*(rev 3: rev 2 reported 145. Round 2's finding 7 found two rows still bundled —
G6.11 carried four independent limits and G10.12 carried thirteen rendered states
crossed with three locales — so they are now G6.11a–d, G10.12a–m and G10.13a–c,
and the total is **163**. Same rule, two rows rev 2's own split missed.)*

*(rev 2: rev 1 reported 93 rows. The number rose to 145 without a single new
claim being added — S0-154's split turned fifteen bundled rows into their
constituent conditions, which is the point: the bundling was hiding untested
conditions inside rows that already counted as covered. G4.1 alone was eleven
components behind one row, and G4.2 seven injected events behind another —
finding 7's exact objection. Statuses also moved, always downward and never up:
G9.1, G9.2, G9.4 and G9.5 lost their LIVE marks on inspection (finding 8 — they
named a tooth that guards a different manifest), and the G6.3/G6.5 splits left
the PR-A-new components — the `0.10` margin, and the four ineligible-input
classes — correctly separated from the two threshold constants that really are
live. One row was added rather than relabelled: G9.6 catalogues the tooth the
old LIVE marks were borrowing.)*

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

**F2 — sixteen of the 203 cases cannot go red until lane 1 lands, and they are
concentrated in the gates that protect truth.** G1.2, G2.5, G6.9, G7.1–G7.3, and
the ten remaining predicate cases all rest on a page being `supported`, and no
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
`S0-136` `lane 1` rows are specified against committed frozen text — the four-condition predicate for §1's nine rows, the axis-independence rule for P6b — never against the eventual implementation ·
`S0-137` the empty-inventory vacuous-truth mutation is required, and a genuinely empty page stays permanently provisional ·
`S0-138` G2.7 is carried at full strength; the interim treatment is labelled as interim and does not become the contract ·
`S0-139` the reservation exit matrix is asserted as a total function over 8 exits × 6 resting states ·
`S0-140` G4.3 asserts "never stale or provisional", not "always present" ·
`S0-141` G4.5 asserts over held guards, not elapsed time ·
`S0-142` G7.2 rejects the entire result; skipping the ambiguous claim is the failure ·
`S0-143` G9.5 is a mutation row — the enumeration may not come from a tool query ·
`S0-144` catalog completeness against the frozen clause lists is itself a test.

**Added in rev 2:** `S0-154` a mutation row names exactly one condition; bundled
rows are split into lettered sub-rows keeping their number ·
`S0-155` each row's positive control is derived from the row by a fixed rule, and
the derivation is the contract.

**Added in rev 4:** `S0-160` a frozen clause whose mapping is waiting on a ruling
carries a `RULING` catalog row naming the ruling it waits on — an unmapped clause is
visibly ungated per S0-134, never silently absent. *(rev 5: the row is filed under
the gate that owns the clause — G6.14, not G10.16 — so S0-144's prefix-based
completeness test resolves it, and it is never totalled as coverage. rev 8: its
only row retired when R-1 ruled; the status survives the row.)*
