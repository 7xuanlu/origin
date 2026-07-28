# M5 PR-A — review follow-ups carried into PR-B

Two independent reviews ran against PR-A (schema + shadow derivation): an
in-house deep review (APPROVE-WITH-FIXES) and a cross-model Codex review
(BLOCK). Between them they raised ten findings. Eight are fixed in PR-A with a
RED-first test each, mutation-proven by deleting the rule and watching the test
turn red. The two below are **deliberately deferred**, and this file is the
record so they are carried rather than dropped.

Both are architectural rather than local: they describe invariants that PR-A's
writers uphold but that nothing outside those writers is obliged to preserve.
Neither has a production caller today — PR-A's substrate is written only by its
own tests — so neither is a live defect. Both become live the moment PR-B
wires a reader or a UI save path, which is why they are named here and not
left to be re-discovered.

## F5 — Parent-side mutation leaves active, now-invalid edges

The space fence and the human-root rule are enforced by triggers on `edges`
INSERT and UPDATE (`crates/wenlan-core/src/db/edges_rebuild.rs`). They ask
"is this edge valid at the moment it is written". Nothing asks the question
again when the rows the edge *points at* change underneath it.

Four concrete paths, all in `crates/wenlan-core/src/db.rs`:

| Mutation | What the edge then asserts |
|---|---|
| `set_page_workspace` moves a claim's page | an active support edge still stamped with the old space |
| moving the evidence memory's space | an active edge that is now cross-space, fence never fired |
| deleting a page | claims/revisions cascade away; polymorphic edge endpoints carry no FK, so orphan active support/attestation rows survive |
| flipping an attesting root's `root_kind` human → `generated` | an active human-style attestation over a generated root |

Three shapes of fix are open, and choosing between them is a design call, not
a patch: (a) refuse the parent mutation while a dependent active edge exists,
(b) invalidate/retract the dependent edges inside the parent's transaction, or
(c) let a reconcile sweep detect and retract, on the model
`WENLAN_ENABLE_EDGES_RECONCILE` already uses. (c) fits the existing machinery
and keeps parent mutations cheap; (b) is the only one that is correct at the
instant of the write. The decision belongs with PR-B's reader cutover, because
what a stale active edge *costs* depends entirely on who is reading it.

## F6 — Identical human deltas in two spaces alias one space-bound memory

Provenance roots are content-addressed (`crates/wenlan-core/src/provenance.rs`)
and the delta memory's id derives from the root alone, so the same sentence
added to a page in space A and a page in space B resolves to ONE memory —
filed in space A, because that call ran first. `mint_human_edit_delta` stores
with `INSERT OR IGNORE` and does not compare the existing row's space, so the
second call reports success while handing back evidence the space fence will
later refuse to cite.

PR-A closes the *silent* half: the minter now refuses by name when the memory
it resolved to is filed in a different space than the page being edited, so the
failure surfaces at the call that caused it rather than at an unrelated support
write later. What remains deferred is the underlying identity question — whether
one sentence written by one human in two spaces is one piece of evidence or two.
Folding space into the root's independence signals makes them two; keeping the
root global and letting a memory span spaces makes them one. That is a data-model
decision for M5's identity axis, not a bug fix.

## Coverage gap

`scripts/m5-reader-sweep.py` (+209 lines in PR-A) was reviewed by neither
reviewer. It is a read-only inventory script with no production caller; it
should be read before anything depends on its output.
