# M5 Stage 0 — truth-state transition matrix

Date: 2026-07-27. Binding for M5 PR-A. Implements D2 of
`2026-07-27-kg-m5-goal-prompt.md`.

Truth has **two independent axes**. They are computed by different processes,
mean different things, and may never be derived from each other:

| Axis | Values | Derived by | Scope |
|---|---|---|---|
| `support_status` | `supported \| provisional` | machine, fail-closed | current page version |
| `human_reviewed` | `true \| false` | explicit human action only | exact page version + digest |

The single most important rule in this document: **neither axis may be inferred
from the other, and neither may be inferred from any legacy field.** Human
attestation never makes a page `supported`. Machine support never makes a page
`human_reviewed`.

## 1. `support_status` — the machine axis

A page version is `supported` only when **every** condition holds. Any failure,
of any kind, yields `provisional`.

1. An exact-page-version `claim_derivation_complete` marker exists, and its
   recorded page-version digest equals the current page-version digest.
2. That marker's membership inventory is **nonempty**.
3. Every active claim revision in the inventory has at least one `supports` edge
   that is simultaneously: active, above threshold, and produced by a
   currently-eligible model version.
4. No claim revision in the inventory is in a deferred, timed-out, or malformed
   support state.

Everything else is `provisional`. This is a whitelist, not a blacklist — an
unanticipated state is provisional by construction rather than by enumeration.

### The empty-page rule

Condition 2 exists to forbid the vacuous truth **"all zero claims are
supported."** A page whose derivation produced zero claims is `provisional`,
never `supported`. This is the single most dangerous state in the design,
because `∀x ∈ ∅` is trivially true in every natural implementation — a plain
`.all()` over an empty inventory returns `true` and silently marks an
underived page as fully supported.

A genuinely empty page (no prose, nothing to claim) is therefore permanently
`provisional`. That is intended and is not a bug to be "fixed" later.

### Failure modes, all provisional

| Failure | Why provisional |
|---|---|
| no derivation marker | never derived |
| marker digest ≠ current page digest | derived against different text |
| marker present, inventory empty | vacuous-truth guard (§1) |
| partial derivation | incomplete evidence |
| support edge missing for any revision | incomplete evidence |
| support below threshold | insufficient evidence |
| support from an ineligible model version | evidence not currently trusted |
| support edge retracted/inactive | evidence withdrawn |
| anchor digest mismatch on the source span | evidence no longer points at that text |
| derivation deferred, timed out, or malformed | unknown, and unknown is not true |

**Partial support results never publish.** A derivation run that completes some
claims and fails others publishes nothing; the page keeps its prior state. This
prevents a half-finished run from flipping a page to `supported` on the strength
of the claims that happened to succeed first.

## 2. `human_reviewed` — the human axis

`human_reviewed` becomes `true` only through an explicit review action carrying
**both** the exact page version and its digest. If either is absent or stale,
the action is rejected rather than applied to the current version.

`human_reviewed` resets to `false` on every new page version. Approval is of a
specific text, not of a page in perpetuity.

### Actions that do NOT set `human_reviewed`

Explicitly enumerated by D2, because each is a plausible-looking near-miss:

| Action | Sets `human_reviewed`? |
|---|---|
| editing prose | **no** |
| accepting a revision card | **no** |
| attesting a single claim | **no** |
| any bulk or implicit action | **no** |
| explicit page review bound to version + digest | **yes** |

Editing is authorship, not review; a human who wrote a sentence has not
independently verified it. Attesting one claim is not attesting the page.

## 3. Transition matrix

`SS` = `support_status`, `HR` = `human_reviewed`. "current" means the action
carries the exact current page version and digest.

| # | Writer / action | Base state | Support outcome | Review outcome | Result |
|---|---|---|---|---|---|
| 1 | page created (nonempty) | — | not yet derived | — | `SS=provisional`, `HR=false` |
| 2 | page created (empty) | — | inventory empty | — | `SS=provisional`, `HR=false` |
| 3 | derivation completes, all supported | `provisional` | all pass | unchanged | `SS=supported` |
| 4 | derivation completes, any unsupported | `provisional` | ≥1 fails | unchanged | `SS=provisional` |
| 5 | derivation completes, inventory empty | any | vacuous | unchanged | `SS=provisional` |
| 6 | derivation partial | any | incomplete | unchanged | **no publish**, state unchanged |
| 7 | derivation digest ≠ page digest | any | stale | unchanged | `SS=provisional` |
| 8 | prose edited (new version) | any | invalidated | reset | `SS=provisional`, `HR=false` |
| 9 | explicit review, current version | any | unchanged | approved | `HR=true`, **`SS` unchanged** |
| 10 | explicit review, stale version | any | unchanged | rejected | unchanged; action refused |
| 11 | claim attested | any | unchanged | not a page review | `HR` unchanged, `SS` unchanged |
| 12 | revision card accepted | any | re-derive required | not a page review | `HR` unchanged |
| 13 | support edge retracted | `supported` | now missing | unchanged | `SS=provisional` |
| 14 | model version becomes ineligible | `supported` | now ineligible | unchanged | `SS=provisional` |
| 15 | threshold raised above existing scores | `supported` | now below | unchanged | `SS=provisional` |
| 16 | migration of a pre-M5 page | — | not derived | not reviewed | `SS=provisional`, `HR=false` |

Rows 9 and 13–15 are the axis-independence teeth. Row 9 shows human approval
leaving `SS` untouched; 13–15 show `SS` falling back to `provisional` with no
human action involved, and without disturbing `HR`.

Rows 13–15 also establish that `supported` is **not monotonic**. It is a
statement about currently-valid evidence, so it can be lost without the page
changing at all.

### Who drives demotion

`support_status` is **stored**, not computed on read — evaluating §1 per page on
every read would not meet the budgets in artifact 6. Stored state with
event-driven invalidation needs a named owner, or rows 13–15 are aspirations
that no component performs.

| Event | Trigger | Finds affected pages via |
|---|---|---|
| support edge retracted / invalidated | the write that retracts it, same transaction | `idx_edges_supports_rev` (artifact 3 §6) |
| model version → `retired` | the eligibility transition (artifact 6 §3) | entailment cache `(model_id, model_version)` index |
| threshold raised | the threshold change | same index |
| anchor invalidated | the document write that invalidates it | anchor → claim revision |

Rules:

- Demotion is **synchronous with its trigger**, in the same transaction. An
  asynchronous sweep would leave a window in which a page reads `supported` on
  evidence that is already gone — the exact false-trust this rung forbids.
- Promotion is asynchronous (it needs model work); demotion never is. The
  asymmetry is deliberate: losing trust must be immediate, gaining it may wait.
- A bulk trigger (version retirement, threshold change) demotes through the same
  path, batched, and is bounded by the same budgets. It never bypasses the
  per-page write.
- A **startup reconciler** re-evaluates §1 for any page whose stored status
  cannot be proven consistent with current evidence — the durable backstop for a
  crash between a retraction and its demotion.

## 4. Migration from legacy fields

Pre-M5 pages migrate to `support_status=provisional`, `human_reviewed=false`,
unconditionally.

**Nothing** is inferred from `review_status`, `confirmed`, `user_edited`,
authored status, or prose provenance. Those fields recorded different intents
under different rules; treating any of them as evidence of machine support or of
version-bound human review would fabricate trust that was never established.

The cost is real and accepted: every existing page becomes `provisional` at
migration and stays there until genuinely derived. That is the correct direction
of error.

## 5. Legacy `review_status` — compatibility only

After M5, `review_status` is a **mirror of the machine axis only**:

| `support_status` | mirrored `review_status` |
|---|---|
| `provisional` | `unconfirmed` |
| `supported` | `confirmed` |

`human_reviewed` does **not** participate in this mapping. A page that is
human-reviewed but unsupported still mirrors to `unconfirmed`, because
`review_status` is consumed by old clients that read it as a trust signal, and
the trust they need is machine support.

`review_status` is never authoritative, never written directly by M5 logic, and
never read back as an input to either axis. It is derived output.

## 6. Mutation checks

Each weakening must turn at least one listed row RED:

| Weakening | Must fail |
|---|---|
| remove the nonempty-inventory condition | row 2, row 5 |
| let `.all()` run over an empty inventory | row 5 |
| let explicit review set `SS=supported` | row 9 |
| let claim attestation set `HR=true` | row 11 |
| let prose edit preserve `HR` | row 8 |
| publish partial derivation results | row 6 |
| skip the digest equality check | row 7 |
| infer `SS` from legacy `review_status` | row 16 |
| infer `HR` from `user_edited` or authored status | row 16 |
| treat `supported` as monotonic (never fall back) | rows 13, 14, 15 |
| accept a review action without version+digest | row 10 |
| demote asynchronously instead of in the trigger's transaction | §3 "who drives demotion" — crash-window test |
| leave a retraction with no demotion path | rows 13–15 owner test |
| skip the startup reconciler | crash-between-retraction-and-demotion test |
