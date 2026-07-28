# M5 — the claim extractor

Date: 2026-07-27. Binding for M5 PR-A, before shadow derivation. Ninth Stage 0
artifact, added after the final design gate observed that artifact 1 fixes claim
*identity* given a claim set, artifact 6 specs the *judge*, and nothing said what
turns page prose into the claim set in the first place.

Merge base: `5ba8a3b4`.

## 1. There is already an extractor, and it is duplicated

The gap is worse than "unspecified". `citations.rs` has segmented page prose
into claims and judged them since the citation-grounding rung shipped:

| Decision | Where | What it does |
|---|---|---|
| marker handling | `citations.rs:167-171` | boundaries computed on a marker-free **bare** copy, marker offsets recorded before removal |
| segmentation | `citations.rs:196-203` | its **own** `[.!?]+\s+` scan producing `(start, end)` spans |
| paragraph spans | `citations.rs:211-218` | blank-line delimited, for the fallback only |
| judgment | `citations.rs:256` | `bidirectional_support(sentence, union) >= 0.5` |
| support metric | `citations.rs:120` | `max(overlap_fraction(a,b), overlap_fraction(b,a))` — token overlap, **no LLM** |
| fallback scope | `citations.rs:257-268` | a span failing at sentence scope retries against its enclosing paragraph |
| recorded scope | `wenlan-types/src/pages.rs:149` | `"sentence"` or `"paragraph"`, keeping the weaker guarantee visible |

**The segmentation rule exists twice.** `faithfulness::split_sentences`
(`faithfulness.rs:13`) and `citations.rs:196` both hard-code the regex
`(?m)[.!?]+\s+`, kept in sync by a comment (`citations.rs:195`) and nothing
else. A third site, `eval/page_faithfulness.rs:62`, calls the function.

The two copies do not even agree today. `split_sentences` drops empty spans
(`faithfulness.rs:15`, `.filter(|s| !s.trim().is_empty())`); `citations.rs`
keeps them, because its span indices must line up with recorded marker offsets.
So "sentence *n*" means different prose in the two paths on any body with a
double delimiter.

M5 must therefore **not** be told to "reuse `split_sentences`" — that is the
naive reading and it silently breaks marker attribution, because the function
returns `&str` with no offsets and drops the empties the citation path counts
on. PR-A extracts **one** offset-returning span function, and both existing
sites call it. Adding a third copy is the failure this section exists to
prevent.

## 2. The decisive constraint: extent must not depend on judgment

The citation path's two-tier scope is a **verification** fallback, not a
segmentation choice, and M5 must not inherit it as a claim boundary.

If scope were part of a claim's extent, the same prose would yield a
sentence-sized claim when it verifies and a paragraph-sized claim when it does
not. Claim extent would then be a function of support strength — and since M5
revisions are content-addressed over claim text (artifact 1), the claim's
*identity* would change when its *support* changed.

That breaks the rung's central invariant. D2 requires `support_status` and
`human_reviewed` to be two independent axes, neither inferred from the other or
from legacy state. An identity that moves when support moves is inference by
another name: a reviewer's `human_reviewed=true` would silently detach from the
claim it was attached to the moment the judge's verdict flipped.

So:

- **claim extent is always the sentence span**, computed before and
  independently of any judgment;
- the paragraph fallback stays where it is — a property of the *verdict*,
  recorded as the scope that decided it, never a property of the claim.

## 3. Extraction is deterministic, and that is a requirement

The extractor is a span scan, not a model call. This is not a simplification to
be revisited when someone wants richer propositions.

Content-addressed immutable revisions mean claim identity is a pure function of
claim text. An LLM extractor makes that function depend on a model version, so
every model upgrade re-cuts every claim in the corpus, mints a new revision for
each, and invalidates every `attests` edge a human ever wrote. The corpus would
lose its human review state on a routine dependency bump.

A deterministic extractor makes claim sets stable under model upgrades. The
model still does the hard part — judging support — and its version already
participates in the entailment cache key, which is the correct place for model
dependence to live.

| Stage | Deterministic? | Versioned by |
|---|---|---|
| extraction (prose → claim spans) | **yes** | `extractor_version` (§4) |
| judgment (claim × source → verdict) | no, model call | `model_id`, `model_version`, `prompt_version` (artifact 6) |

## 4. `extractor_version`

A stamp is still required, because the span scan itself can change: an
abbreviation exception, a change to marker stripping, the code-fence rule in §5.
Any of those re-cuts claims.

- `extractor_version` is a monotone integer, bumped by any change to
  segmentation, marker stripping, or normalization that can alter a span
  boundary.
- It participates in **derivation-marker identity**. Artifact 2 §1 condition 1
  validates a `claim_derivation_complete` marker by page-version digest alone;
  that is necessary and not sufficient, because the same page text under a
  changed extractor yields a different claim set. The marker therefore records
  `extractor_version` too, and condition 1 checks both. A bump invalidates
  markers and requeues derivation.
- It does **not** participate in claim revision identity. A revision is
  addressed by its text; two extractor versions that produce byte-identical
  claim text produce the same revision, which is the correct outcome and the
  reason the stamp sits on the marker rather than the content hash.
- The entailment cache key stays five-part
  (`claim_text_digest`, `source_span_digest`, `model_id`, `model_version`,
  `prompt_version`). Extraction does not enter it: the key is over claim *text*,
  and text that survives an extractor bump unchanged has a still-valid verdict.
  Re-keying on `extractor_version` would throw away a correct cache for no gain.

## 5. Granularity, and what today's scan actually does

Today's behavior is stated per row before the M5 rule. Two rows — headings and
code fences — are inherited quirks that PR-A deliberately keeps, and they are
written down so that "obviously we should fix this" happens behind an
`extractor_version` bump instead of inside PR-A. The only real divergence in
this artifact is §6.

| Input | Today | M5 rule |
|---|---|---|
| a sentence with an inline marker | markers stripped, boundaries on the bare copy, offsets mapped back (`citations.rs:167-171`) | unchanged |
| a heading (no terminal punctuation) | **absorbed into the following span** — the scan cuts only at `[.!?]+\s+`, so `"# Title\n\nA claim."` is one span | unchanged for PR-A. A heading fused to its first sentence is ugly, not wrong: the claim text still contains the claim, and changing it costs an `extractor_version` bump that invalidates markers corpus-wide. Revisit behind that bump, not inside PR-A. |
| a list item | terminated items split; a bare fragment fuses to its neighbour, same rule | unchanged, same reasoning |
| a fenced code block | **no fence handling exists** (zero occurrences in `citations.rs`) — code is cut at every `.`/`!`/`?` in it | unchanged for PR-A. Code fragments become claims that no source entails, so they land `provisional`, which is the fail-closed direction. Fixing it is a boundary change, hence an `extractor_version` bump; scheduled, not smuggled in. |
| an empty page | zero spans | zero claims is a **valid** claim set, not an error |

Zero claims is a derivation **success** with an empty inventory, and the
`claim_derivation_complete` marker **must still be written**. The tempting
implementation — no claims, so skip the marker — lands the page in artifact 2's
"no derivation marker → never derived" row, which is an unknown state, not an
outcome. Readiness condition 1 (artifact 7 §3) counts pages with an explicit
outcome, so every zero-claim page would hold readiness under 100% forever with
no diagnosis pointing at the cause.

With the marker written, artifact 2 §1 condition 2 (nonempty inventory) makes
the page `provisional` — permanently and intentionally, per that artifact's
empty-page rule. Derivation succeeded; the page just supports nothing.

## 6. The inherited fail-open, which M5 must not inherit

`overlap_fraction` returns **1.0** when the input has no content tokens —
`faithfulness.rs:29` calls it "vacuously faithful", `:32-33` implements it.
`bidirectional_support` takes the **max** of both directions
(`citations.rs:120-123`), so the vacuous 1.0 wins whenever *either* side is
token-empty. Content tokens are alphanumeric runs of length ≥ 4 that are not
stopwords (`faithfulness.rs:24`).

Two live consequences:

- a claim of only short or stopword words — `"It is what it is."`,
  `"Not any more."` — scores 1.0 against **any** source and is badged verified
  (every token is under 4 chars or a stopword: `what` and `more` are both in the
  `STOPWORDS` list at `faithfulness.rs:4-9`);
- a **source** with no content tokens verifies **every** claim checked against
  it.

That is defensible for a citation badge, which is advisory. It is not
defensible for `support_status`, which D2 requires to be fail-closed and which
gates whether prose is served at all after D3. A claim that cannot be
distinguished from noise must not be `supported` because the scorer had nothing
to measure.

**M5 rule:** a claim or source with zero content tokens yields
`support_status = provisional` with an explicit `insufficient_signal` reason. It
is never `supported`. The citation path's own badge behavior is left alone —
this is an M5 truth-axis rule, not a change to `overlap_fraction`, whose
vacuous-1.0 contract other callers (`score_sentence_faithful`, the faithfulness
bench) depend on.

## 7. Mutation checks

Every row is an executable test that goes RED under its weakening.

The first row is **RED right now**, before any M5 code: the delimiter literal
occurs twice today (`citations.rs:196`, `faithfulness.rs:14`). It is the
RED-first test for §1's dedup, and the way to make it green is to remove the
duplicate — never to loosen the count.

| Weakening | Must fail |
|---|---|
| add a third copy of the `[.!?]+\s+` rule instead of one shared span function | §1 — assert exactly one occurrence of the delimiter regex literal in `crates/*/src` |
| have M5 call `split_sentences` and index spans by its output | §1 — a body with a double delimiter must yield identical span indices in both paths |
| let claim extent widen to the paragraph on a failed verdict | §2 — same prose, flipped verdict, claim text byte-identical |
| derive claim extent from any support or review state | §2 — the identity/support independence test |
| make extraction a model call | §3 — identical claim sets across two different `model_id`s |
| change a span boundary without bumping `extractor_version` | §4 — boundary-changing edit + unchanged stamp must fail |
| validate a derivation marker by page-version digest alone | §4 — same page text, bumped `extractor_version`, marker must be rejected |
| let a stale `extractor_version` marker satisfy derivation completeness | §4 — bump the stamp, assert markers invalidate and derivation requeues |
| add `extractor_version` to the entailment cache key | §4 — bump the stamp, assert a verdict on unchanged claim text still hits cache |
| skip the derivation marker when a page yields zero claims | §5 — zero-claim page must carry a marker, be `provisional` by artifact 2 §1 condition 2, and readiness must still reach 100% |
| add fence or heading handling inside PR-A without an `extractor_version` bump | §5 — same test as the boundary-change row, stated separately because these two are the tempting ones |
| let a token-empty claim reach `supported` | §6 — `"It is what it is."` against any source must be `provisional`/`insufficient_signal` |
| let a token-empty source support any claim | §6 — same, source side |
| "fix" `overlap_fraction` to return 0.0 on empty input instead | §6 — `score_sentence_faithful` and the faithfulness bench must keep the vacuous-1.0 contract |
