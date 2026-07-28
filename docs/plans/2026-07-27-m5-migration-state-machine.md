# M5 Stage 0 — A/B/C migration, readiness, and rollback state machine

Date: 2026-07-27. Binding for M5 PR-A, PR-B, and PR-C. Implements the D3
cutover ceremony and the PR-A/B/C migration rules of
`2026-07-27-kg-m5-goal-prompt.md`.

Merge base: `5ba8a3b4`, `SCHEMA_VERSION = 96` (`db.rs:582`).

## 1. Durable state variables

Every one is durable, and no two are inferred from each other.

| Variable | Owner | Values |
|---|---|---|
| `user_version` | PR-A | 96 → **97** |
| `edges_migration_state` | PR-A | `stage`, `cursor`, `batch_checksum`, `epoch` (table exists, `db.rs:8936`) |
| `claim_migration_state` | PR-A | same shape, own row |
| truth-contract version | PR-B | integer, per client declaration |
| readiness watermark | PR-B | 0–100%, durable |
| **cutover generation** | PR-C | `off` → `preparing` → `committed` |
| projection manifest digest | PR-C | over (page, version, content digest) |
| projection watermark | PR-C | + pending-outbox count |
| writer fence / cutover lease | PR-C | held / released |

The cutover generation is the single fence every page writer CASes. One
variable, one gate — the M4 control-plane shape (ONE global generation, ONE
watermark, ONE central fail-closed gate), reused deliberately because it worked.

## 2. PR-A — schema, shadow, no behavior change

Adds: logical claims, immutable revisions + predecessor chains, page-version
membership/anchors, exact-page-version derivation-completion markers, the
five-part entailment cache, page truth state with version-bound review evidence,
durable jobs/leases/receipts, the one-shot nonce table, and the widened edge
kinds/types + indexes (artifact 3).

Migration rules, all fail-closed:

- existing pages ⇒ `support_status=provisional`, `human_reviewed=false`,
  unconditionally (artifact 2 §4);
- **nothing** inferred from `review_status`, `confirmed`, `user_edited`,
  authored status, or prose provenance;
- existing edges survive the guarded rebuild byte-for-byte, row-for-row
  (artifact 3 §7);
- resumable, replay-safe, checksummed;
- pre-migration online backup + integrity receipt + **restore drill executed**,
  before the rebuild — a backup never restored is a hope, not a rollback plan.

Shadow derivation computes claims/supports/status and publishes only through the
D8 finalizer. **Reader trust behavior is unchanged.**

### Downgrade barrier

Already present: `db.rs:3547` refuses to open a database whose `user_version`
exceeds `SCHEMA_VERSION`. Stamping 97 therefore locks out every pre-M5 binary
for free. PR-A adds a test asserting this specific refusal at 97, so a later
refactor cannot quietly loosen it.

Ordering, from the M4 lesson: **all DDL, triggers, and guards commit before
`user_version` is stamped.** An interrupted upgrade leaves ≤96 and never a
half-built 97.

## 3. PR-B — compatibility, inventory, readiness

- typed `support_status` + `human_reviewed` on every M5-aware page-bearing wire
  representation;
- explicit truth-contract negotiation/versioning; **no caller-selected content
  filter** — a client declares that it renders both axes, it never requests
  provisional content;
- every reader adapter from artifact 4 installed and mutation-tested with
  `cutover_generation=off`;
- readiness watermark computed and published.

**PR-B does not activate D3.** Behavior is identical to PR-A. This is the phase
where the adapters are proven correct while inert.

### Readiness

Readiness reaches 100% only when all hold:

1. every page has a derivation outcome (supported or an explicit provisional
   reason) — no page in an unknown state;
2. zero pending derivation jobs and zero parked jobs;
3. the projection outbox is empty;
4. the projection manifest digest matches the built supported-only directory;
5. every reader adapter is installed and its mutation test green.

Readiness is durable and monotone **only while its inputs hold**. A new page or
a retracted support edge lowers it. Treating readiness as a latch would let a
cutover proceed on a number that was true an hour ago.

**Readiness proves coverage, not health.** Condition 1 counts "provisional with
an explicit reason" as a derivation outcome, so 100% readiness is fully
compatible with a corpus that is 90% permanently unsupported. Readiness would
read green while cutover emptied the user's automatic context. That is why §4a
exists.

## 4a. PR-C entry gate — a human looks at the numbers

Readiness = 100% is necessary and **not sufficient**. Before the ceremony runs,
a human reviews the shadow-phase supported-fraction **by page class** —
distilled, human-edited, human-authored (artifact 6 §2a) — and explicitly
approves the cutover.

This gate is not automatable, and that is the point. Every other check in this
document asks "is the machinery consistent?" This one asks "does the product
survive the switch?" A corpus where human-authored pages sit near zero supported
is a signal to fix the evidence path first, not to flip the fence and discover
it in production.

Aggregate supported-fraction does **not** satisfy this gate. An aggregate
dominated by distilled pages hides exactly the class most likely to be broken.

## 4. PR-C — the fenced cutover ceremony

Filesystem projection and SQLite truth state are **not** one atomic store. The
ceremony is fail-closed, not falsely atomic.

1. **Acquire** the global page-write/cutover lease. Every page writer CASes the
   same cutover generation, so no writer can cross the fence.
2. **Drain** all claim/support and projection work for generation `G`.
3. **Build** a fresh supported-only generation directory; record its canonical
   (page, version, content-digest) manifest; persist a matching projection
   watermark with **zero** pending outbox rows.
4. **Swap** the legacy directory to the prepared generation **first**, then
   commit the SQLite cutover **only if** page/truth generation, manifest digest,
   watermark, empty outbox, and writer lease all still match.
5. **Release** the writer fence only after commit. If commit fails, atomically
   restore the previous directory *before* releasing.

Directory-first is the deliberate choice: a legacy reader may briefly see a page
missing, but must **never** see stale provisional prose after cutover. Omission
is recoverable; false trust is not.

### Steady-state projection transitions

| Transition | Order |
|---|---|
| supported → provisional | atomically rename/remove the legacy file into private quarantine **first**, then commit truth state |
| provisional → supported | commit DB truth **first**, then publish via temp-file + atomic rename |

Both orders put the failure window on the side of *file absent*. A crash before
the DB commit in the first case is safe because the file is already gone, and
startup reconcile restores it only if the DB still says supported. A projection
failure in the second case leaves the file absent, which is safe by the same
argument.

A durable projection outbox + reconciler makes every crash point fail closed.

## 5. Recovery states

Crash recovery consults the durable cutover state — it never infers from what is
on disk:

| Durable state | Recovery |
|---|---|
| `off` | restore the old directory |
| `committed` | keep the prepared directory |
| `preparing` / indeterminate | **expose no legacy directory** until reconciliation proves one side |

The third row is the whole design in one line. An indeterminate cutover serves
nothing through the legacy path rather than guessing, because both guesses are
wrong in a way the user cannot detect: the old directory may hold prose that is
now provisional, and the new one may be incomplete.

## 6. Rollback

| Phase | Rollback |
|---|---|
| PR-A pre-`user_version` | drop new tables; old schema intact |
| PR-A post-`user_version` | forward-only; restore from the §2 backup (drill-verified) |
| PR-B | flip adapters off; no durable state changed |
| PR-C `preparing` | restore previous directory, release lease, state → `off` |
| PR-C `committed` | reverse cutover: rebuild the full directory, set generation `off`, under the same lease |

`committed` → `off` is a supported operation, not an emergency improvisation. It
runs the same ceremony in reverse under the same fence, and it is exercised by a
test rather than documented and hoped for.

## 7. Ordering invariants

Each is a separate test:

1. DDL/triggers/guards commit **before** `user_version` (§2).
2. Directory swap **before** SQLite cutover commit (§4.4).
3. Quarantine **before** truth commit on supported → provisional (§4).
4. Truth commit **before** publish on provisional → supported (§4).
5. Writer fence released **after** commit, or after restore on failure (§4.5).
6. Nonce consumption **inside** the mutation transaction (artifact 5 §6).
7. Receipt replay lookup **before** presence validation (artifact 5 §4).

## 8. Mutation checks

| Weakening | Must fail |
|---|---|
| stamp `user_version` before guards commit | §7.1 |
| infer any truth field from a legacy field | artifact 2 §4 |
| let readiness latch at 100% | §3 test — retract a support edge, readiness must drop |
| commit SQLite before the directory swap | §7.2 |
| commit truth before quarantining on demotion | §7.3 |
| publish before truth commit on promotion | §7.4 |
| release the fence before commit | §7.5 |
| infer recovery state from disk contents | §5 test |
| serve the legacy directory in an indeterminate state | §5 row 3 |
| let a writer skip the cutover-generation CAS | §4.1 test |
| open a v97 database with a pre-M5 binary | §2 downgrade test |
| accept a backup without a restore drill | §2 — release checklist |
| ship `committed` with no reverse path | §6 reverse-cutover test |
| treat readiness = 100% as sufficient for cutover | §4a gate |
| satisfy §4a with an aggregate supported-fraction | §4a |
| ship PR-A without the human-root minter | artifact 6 §2a / artifact 3 §5 |
