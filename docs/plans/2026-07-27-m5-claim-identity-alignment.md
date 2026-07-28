# M5 Stage 0 — claim identity and alignment contract

Date: 2026-07-27. Binding for M5 PR-A. Implements D1 of
`2026-07-27-kg-m5-goal-prompt.md`; amends §6.7 of the unified-model spec.

This document is the authority for how a logical claim keeps its identity across
page revisions. Its central rule: **false-new is safer than false-same.** Support
and attestation carry trust, and attaching either to text a human never approved
is the one failure this rung exists to prevent. Every ambiguous case below
therefore resolves toward minting a new logical claim.

## 1. Identifiers

| Identifier | Shape | Rule |
|---|---|---|
| `claim_id` | durable, opaque | Never reused, never reassigned, never content-derived. Survives edits that preserve alignment. |
| `claim_revision_id` | content-addressed | `H(claim_id, predecessor_revision_id, canonical_text_digest, claim_kind)`. |
| `canonical_text_digest` | digest of canonical form | Defined in §2. |

`predecessor_revision_id` is the empty string for a claim's first revision, so
the hash is total and the chain has one root. Because `claim_id` participates in
the hash, two claims with identical text in the same page produce different
revision IDs, and a revision ID can never be silently shared between logical
claims.

Supports and attestations reference `claim_revision_id` only. Neither may
reference a bare `claim_id`, a page span, or a character offset. This is what
makes trust immutable: the object a human or model approved cannot change
underneath the approval.

## 2. Canonicalization

`canonical_text` is derived from claim text by, in order:

1. Unicode NFC normalization.
2. Replacing every Unicode whitespace run with a single space (U+0020).
3. Trimming leading and trailing whitespace.
4. Removing a single trailing sentence terminator (`.`, `!`, `?`) if present.
5. Case-folding with `str::to_lowercase`.

`canonical_text_digest` is the SHA-256 of the UTF-8 bytes of `canonical_text`.

Canonicalization is deliberately conservative — it absorbs only reformatting.
It does **not** normalize punctuation inside the sentence, expand contractions,
strip markdown emphasis, resolve pronouns, or reorder clauses, because each of
those can change meaning. Two texts that differ after canonicalization are
different claims, full stop.

**Non-goal:** canonicalization is not similarity. There is no edit-distance
threshold anywhere in identity. See §5.

## 3. Anchors

An anchor binds a claim to the source text that produced it:

| Field | Meaning |
|---|---|
| `source_doc_id` | the document the span lives in |
| `source_version` | exact document version the span was read from |
| `span_start`, `span_end` | byte offsets into that exact version |
| `span_digest` | SHA-256 of the exact span bytes |

An anchor is **valid** only if `span_digest` still matches the bytes at those
offsets in that document version. Offsets alone are never trusted: a document
edit shifts offsets, and a stale offset pointing at plausible text is precisely
how support gets attached to the wrong sentence. A digest mismatch invalidates
the anchor, and an invalid anchor can never contribute to alignment or support.

Anchors are per-`claim_revision_id`, immutable, and recorded at derivation time.

## 4. Alignment — the one-to-one rule

Alignment runs between the claim set of the previous page version (`P`) and the
newly derived claim set of the current page version (`C`). A claim in `C`
inherits a `claim_id` from `P` **only** through a match that is one-to-one under
both of the following, evaluated independently:

- **Text key** — equal `canonical_text_digest`.
- **Anchor key** — equal `(source_doc_id, span_digest)` for at least one valid
  anchor on each side.

A candidate pair `(p, c)` is an **inheritance match** only when all hold:

1. `p` and `c` agree on the text key, **or** agree on the anchor key.
2. `p` is the *only* member of `P` matching `c` under the key used.
3. `c` is the *only* member of `C` matching `p` under the key used.
4. `p` is not retired.
5. `p.claim_kind == c.claim_kind`.

Condition 5 is deliberate: a claim whose kind changed is not the same assertion
even when its words are identical, so it must not inherit support.

When both keys are available and they disagree — text matches `p1` while the
anchor matches `p2` — the match is **ambiguous** and `c` mints a new logical
claim. Disagreeing evidence is weaker than no evidence, not stronger.

Everything that is not an inheritance match mints a fresh `claim_id`.

### Position is not identity

Claim order within a page carries no identity weight. Moving a claim, alone,
produces neither a new logical claim nor a new revision — its text digest, kind,
and anchors are unchanged, so the revision hash is unchanged. This is the
explicit D1 requirement that repositioning does not mint a revision.

## 5. Case behavior

| Case | Shape | Outcome | Why |
|---|---|---|---|
| **Unchanged** | 1:1 text + anchor match, identical digest | Same `claim_id`, **same** `claim_revision_id`, no new row | Nothing changed; minting a revision would churn support. |
| **Edited, unambiguous** | 1:1 match on exactly one key, other key absent or agreeing | Same `claim_id`, **new** successor revision | Alignment is certain; support must be re-derived against new text. |
| **Duplicate** | 2+ members of `C` share a text key with one `p` | **All** mint new `claim_id`s | Fails one-to-one in the `C→P` direction. No principled way to pick an heir. |
| **Reordered, unambiguous** | 1:1 match, position differs | Same `claim_id`, same revision | Position is not identity (§4). |
| **Reordered, ambiguous** | reorder plus 2+ equally good matches | New `claim_id`s for the ambiguous members | Explicit D1 requirement. |
| **Split** | one `p` → 2+ members of `C` | All new `claim_id`s; `p` retired | One-to-one fails; neither fragment inherits approval of the whole. |
| **Merge** | 2+ members of `P` → one `c` | New `claim_id`; all matched `p` retired | The merged sentence asserts something no predecessor did alone. |
| **Kind change** | text matches, `claim_kind` differs | New `claim_id` | Condition 5. |
| **Deleted** | `p` has no match in `C` | `p` retired | — |
| **Anchor invalidated** | `p`'s anchors all fail digest check | `p` cannot be matched on the anchor key | §3; may still match on text key. |

Split and merge both retire predecessors *and* mint new IDs. Retirement records
the relationship for audit (§6) without transferring any trust.

## 6. Retirement

Retirement marks a logical claim no longer present on the current page version.

- Retirement is durable and auditable: `retired_at`, `retired_reason`
  (`deleted | split | merged | ambiguous | kind_changed`), and
  `successor_claim_ids` (possibly empty).
- A retired `claim_id` is **never** reassigned and never re-inherited, even if
  identical text reappears in a later version. Reappearing text mints a new
  logical claim.
- Revisions and supports of a retired claim are retained, not deleted — the
  audit trail of what was once approved must survive.

## 7. Oracle corpus

Stage 0 ships a fixture corpus exercised by `G1 — m5_claim_identity_red`. Each
case declares the before/after claim sets and the exact expected identity
outcome; the test asserts identity, not similarity.

**Positive cases — identity MUST be preserved:**

| # | Case |
|---|---|
| P1 | byte-identical text and anchors |
| P2 | whitespace reflow (line wrap, double space) |
| P3 | Unicode NFD → NFC normalization difference |
| P4 | trailing period added |
| P5 | case-only change (`Rust` → `rust`) |
| P6 | claim moved from position 1 to position 4, nothing else changed |
| P7 | surrounding claims edited, target claim untouched |
| P8 | one of two anchors invalidated, the other still valid and uniquely matching |

**Negative cases — identity MUST NOT be preserved:**

| # | Case |
|---|---|
| N1 | negation inserted (`is` → `is not`) |
| N2 | duplicated claim — two identical sentences in the new version |
| N3 | one claim split into two |
| N4 | two claims merged into one |
| N5 | reorder with two equally-matching candidates |
| N6 | text matches `p1` while the anchor matches `p2` |
| N7 | `claim_kind` changed, text identical |
| N8 | quantifier changed (`all` → `some`) |
| N9 | claim deleted, then identical text reappears two versions later |
| N10 | anchor digest mismatch with no valid alternate anchor |

N1 and N8 are the corpus's teeth: both survive any similarity threshold loose
enough to be useful, which is exactly why identity here is digest equality and
never similarity.

## 8. Mutation checks

The oracle is only worth what it catches, so each of these deliberate weakenings
must turn at least one listed case RED:

| Weakening | Must fail |
|---|---|
| drop uniqueness condition 3 (`C→P` direction) | N2 |
| drop uniqueness condition 2 (`P→C` direction) | N3 |
| accept a match when text and anchor keys disagree | N6 |
| drop `claim_kind` from condition 5 | N7 |
| drop `claim_kind` from the revision hash | N7 |
| trust anchor offsets without `span_digest` | N10 |
| add any similarity-threshold fallback | N1, N8 |
| make position part of identity | P6 |
| allow retired IDs to be re-inherited | N9 |
