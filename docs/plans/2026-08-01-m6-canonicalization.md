# M6 Stage-0 artifact 4 — deterministic canonicalization: slots, page IDs, fingerprints, wikilink keys

Status: Stage-0 contract artifact. Normative for PR-A onward.
Scope: frozen contract D5 (deterministic identity and retries) + D8 (normalization
and hard caps, normalization half only — the caps themselves are artifact 5's and
artifact 6's business).
Companions: artifact 1 (`2026-08-01-m6-signal-matrix.md`), artifact 2
(`2026-08-01-m6-state-machines.md`), artifact 3
(`2026-08-01-m6-independence-matrix.md`).

Every `file.rs:NNN` citation in this document was read on branch `kg-m6-stage0`
at authoring time. Unicode claims marked **[measured]** were produced by running
the stated operation, not read out of a table.

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

---

## 0. What this artifact binds

Four distinct identities, easy to conflate and consequential to conflate:

| Identity | Answers | Changes when | Lives in |
|---|---|---|---|
| `slot_id` | *which topic-shaped hole is this?* | the topic's defining inputs change | `genesis_candidates` (PR-A-new) |
| `page_id` | *which page does that hole become?* | never, given a slot | `pages.id` |
| `candidate_fingerprint` | *would re-running produce the same output?* | any output-affecting input changes | `genesis_candidates` (PR-A-new) |
| `label_key` | *are these two wikilinks the same label?* | never, given a label | `page_links.label_key` (db.rs:6673) |

The rule that keeps them apart: **`slot_id` and `page_id` are identity; the
fingerprint is freshness.** A candidate whose fingerprint changed is stale
(machine A transition A7, artifact 2) and re-prepares into the *same* slot and
the *same* page ID. A candidate whose slot changed is a different candidate for a
different topic and shares nothing with the old one. Collapsing these is how a
retry silently forks a page.

---

## 1. The tree already has a deterministic-ID convention. It has two.

The dispatch asked for the existing convention. There are two, they disagree, and
the disagreement is not cosmetic.

| Site | Construction | Separator scheme | Output | Collision-safe by construction? |
|---|---|---|---|---|
| `compute_edge_id`, `crates/wenlan-core/src/provenance.rs:192`-`:207` | SHA-256 over 6 parts | **length-prefixed** (`u64` LE length, then bytes) | `format!("{:x}", …)`, 64 hex | **Yes** |
| `community_relevant_spaces_digest`, `crates/wenlan-core/src/db.rs:2491`-`:2501` | SHA-256 over a sorted+deduped space list | length-prefixed | `hex::encode(…)`, 64 hex | Yes |
| `community_membership_digest`, `crates/wenlan-core/src/db.rs:2697`-`:2711` | SHA-256 over sorted `(node, community, attachment)` triples | length-prefixed | `hex::encode(…)`, 64 hex | Yes |
| `page_write_digest`, `crates/wenlan-core/src/post_write/page_update.rs:349`-`:383` | SHA-256 over the request's deciding fields | length-prefixed (via a local `field` closure) | `format!("{:x}", …)`, 64 hex | Yes |
| `identity_digest`, `crates/wenlan-core/src/provenance.rs:104`-`:113` | SHA-256 over version, kind, content digest | `b":"` separators | 64 hex | Only *by input*: a CHECK'd enum and a hex digest, neither of which can contain `:` |
| `source_page_id`, `crates/wenlan-core/src/document_enrichment.rs:760`-`:769` | SHA-256 over `source_page::`, source id, `::`, file path | `b"::"` separators | `src_` + **first 16 hex chars** | **No** — see finding F1 |

`compute_edge_id`'s own doc comment (`provenance.rs:184`-`:191`) states the
argument that decides this for M6:

> Each part is **length-prefixed** (its byte length as a u64 LE, then the bytes)
> so no two distinct tuples can collide by concatenation ambiguity. A bare NUL
> *separator* is not enough: a part may itself contain a NUL byte (Rust `&str`
> is arbitrary UTF-8, and locators/labels are unvalidated), so `("x\0", "y")`
> and `("x", "\0y")` would hash identically under a separator-only scheme.

That argument applies with more force to M6 than it did to edges. Two of M6's
four slot input sets contain **user-controlled free text**: the orphan-wikilink
label, and (indirectly) the space name. A separator-only scheme over
user-controlled parts is a collision the user can construct on purpose, not one
they might hit by accident.

> **Decision S0-26 — M6 uses the length-prefixed framing (`compute_edge_id`),
> not the separator framing (`identity_digest` / `source_page_id`).**
> Rationale: it is the only one of the two that is safe independent of what its
> inputs contain, and M6's inputs include attacker-chosen strings. This is a
> choice to match the *newer and stronger* of two in-tree conventions, and the
> divergence is deliberate rather than an oversight — see findings F1 and F2.

---

## 2. The M6 digest primitive

One function. Every M6 deterministic ID in this document is a call to it.

```
m6_digest(domain: &str, parts: &[&[u8]]) -> String

    hasher = Sha256::new()
    # the domain tag is itself a length-prefixed part, never a raw prefix
    write_part(hasher, domain.as_bytes())
    for p in parts:
        write_part(hasher, p)
    return format!("{:x}", hasher.finalize())        # 64 lowercase hex chars

write_part(hasher, bytes):
    hasher.update((bytes.len() as u64).to_le_bytes())   # exactly 8 bytes, LE
    hasher.update(bytes)
```

> **Decision S0-27 — SHA-256, `u64` little-endian length prefixes, full 64-char
> lowercase hex output, no truncation.** SHA-256 and LE length prefixes because
> that is what `compute_edge_id` (`provenance.rs:200`-`:206`) and the three
> `db.rs` digests already do; full-length output for the reason in S0-32.
> `format!("{:x}", …)` and `hex::encode(…)` produce byte-identical lowercase hex,
> so either spelling satisfies this decision — the tree uses both.

> **Decision S0-28 — the domain tag is passed through `write_part`, not
> concatenated as a raw prefix.** A raw prefix reintroduces exactly the ambiguity
> the length prefixes remove: with a raw prefix, `("m6-page-v1", "Xabc")` and
> `("m6-page-v1X", "abc")` are the same byte stream. `source_page_id`
> (`document_enrichment.rs:763`) uses a raw prefix and is safe only because its
> tag is a compile-time constant; M6 has several tags and a version axis, so the
> ambiguity would be reachable.

**Encoding rules that apply to every part.** These are part of the contract; a
reimplementation that varies any of them produces different IDs.

| Rule | Value |
|---|---|
| Text parts | UTF-8 bytes of the Rust `String`, no BOM, no trailing NUL |
| Integer parts | ASCII decimal, no leading zeros, no `+`, `-` only for negatives |
| Boolean parts | the literal bytes `t` / `f` |
| Absent optional parts | a zero-length part (length prefix `0`, no bytes) — **not** the string `"none"`, which a real value could equal |
| Set parts | see S0-30 |

---

## 3. `slot_id` — the four signals

D5 fixes the inputs; this section fixes the bytes.

### 3.1 Common frame

```
slot_id = m6_digest("m6-slot-v1", [ signal_tag, space, <per-signal parts…> ])
```

`signal_tag` is one of the literals `evidence-cluster`, `orphan-wikilink`,
`community-overview`, `space-overview`. It is a distinct part rather than being
folded into the domain tag so that all four signals share one domain and one
version axis.

`space` is the raw space string as stored in `communities.space` /
`community_members.space` (`db.rs:10448`, `:10460`). It is **not** normalized —
see S0-31.

### 3.2 Per-signal part vectors

| Signal | Parts after `space` | Source of each part |
|---|---|---|
| evidence cluster | `sorted_set(initial independence_group_id set)` | `provenance_roots.independence_group_id` (`db.rs:8792`) for the roots grounding the cluster's edges, snapshotted at machine-A `observed` |
| orphan wikilink | `normalized_label` | §6 of this document, applied to `page_links.label` (`db.rs:6674`) |
| community overview | `community_id` | `communities.community_id` (`db.rs:10447`), the durable M4 ID |
| space overview | *(none)* | the space alone; one overview slot per space |

The space-overview slot has an empty per-signal part vector. That is intended and
is why the signal tag is a separate part: without it, the space-overview slot for
space `S` and a hypothetical zero-part signal for the same space would collide.

### 3.3 Set encoding

> **Decision S0-30 — a set part is encoded as: dedup, sort **byte-lexicographically
> on the UTF-8 bytes**, then emit the element count as one integer part followed
> by each element as its own part.**
>
> ```
> sorted_set(xs) => [ count(dedup(xs)) as int-part,  e_1, e_2, …, e_n ]
> ```
>
> Three things are load-bearing. **Byte-lexicographic, not locale-collated** —
> `independence_group_id` values are opaque tokens (`src:`/`turn:`/`batch:`
> prefixed, `provenance.rs:171`-`:177`, or an LSH-overlay winner's id), so any
> locale-sensitive ordering is a nondeterminism bug with no upside. **Dedup before
> count**, so a duplicated group cannot inflate the count. **The count is emitted**
> so that `{a}` ∪ `{b}` and `{ab}` cannot alias, which length-prefixing alone
> already prevents but which makes the intent readable at the call site.
> `community_relevant_spaces_digest` (`db.rs:2492`-`:2494`) already does
> sort-then-dedup-then-length-prefix; this is that pattern with the count made
> explicit.

### 3.4 What pins the evidence-cluster slot, and when

The evidence-cluster slot is keyed on the **initial** group set. Groups discovered
later do not change the slot.

> **Decision S0-29 — the evidence-cluster slot set is snapshotted at the machine-A
> `observed` transition and never recomputed.** Rationale: D5 says "sorted initial
> `independence_group_id` set" and D5's closing paragraph says "routine restart,
> recomputation, or root/mirror lifecycle does not reset successful group coverage
> or change a slot." A slot that tracked the live set would change every time a
> new root grounded an edge in the cluster — i.e. the page would change identity
> because it gained evidence, which is backwards. The live set is a **fingerprint**
> input (§5), so growth makes the candidate stale and re-prepares it into the same
> slot. This is exactly the identity-vs-freshness split from §0.

Consequence to accept, not fix: two clusters that start from different group sets
and later converge on identical evidence get two slots and two pages. Machine F's
`covered` state and D4's exclusive-claim rule are what stop both from publishing;
slot identity is not the deduplication mechanism and should not be asked to be.

### 3.5 The community-overview slot inherits M4's ID churn

`community_id` is a **UUIDv4 minted at first appearance and rebound across
recomputes** — `crates/wenlan-core/src/community_grouping.rs:503`-`:511`: a
rebound ID starting with `__m4-new-node-` or `community-m4-new-` gets a fresh
`uuid::Uuid::new_v4()`, otherwise the rebinding carries the previous ID forward.

So the community-overview slot is stable exactly as long as M4's rebinding keeps
the ID stable, and no longer. When a community splits and the split half is
treated as new, it gets a new UUID, hence a new slot, hence a new page. The old
page is not automatically retired — machine A's `superseded` exit and machine F's
coverage bookkeeping are the only things that can retire it, and neither fires on
a bare UUID change.

**This was written as a gap; artifact 8 has since closed it (rev 2, finding 16).**
The two options were (a) subscribe overview slots to M4's rebinding events, or
(b) accept orphaned overview pages and give machine F an explicit rule. This
artifact said Stage 0 could not pick without knowing M4's rebind semantics under
split/merge — but artifact 8 reached that far and picked **(b)**: *"M6 does not
subscribe overview slots to rebind events. It adds an explicit detach rule to
machine F, keyed on the merge loser"* (`2026-08-01-m6-overview-matrix.md`, S0-71).
Its reasoning is that for splits and for the surviving side of a merge the
subscription would fire zero times, because M4 guarantees those community IDs do
not change; the merge loser is the only uncovered case and is one rule rather than
a subscription system. **The decision is S0-71, not an open choice here.** It is
still PR-A-new work, and G-catalog case `C-slot-community-rebind` still exists —
now as the test of S0-71's detach rule rather than as a placeholder for an
undecided fork. The merge-loser half remains dependent on the open
merge-no-survivor ruling (STOP-3), which is a different question from whether M6
subscribes.

---

## 4. `page_id`

```
page_id = "m6p_" + m6_digest("m6-page-v1", [ slot_id ])
```

> **Decision S0-32 — `m6p_` prefix, full 64-hex digest, no truncation.**
>
> The prefix matches the tree's habit of making a page's origin legible from its
> ID (`src_…` for source pages, `document_enrichment.rs:768`), and `pages.id` is
> an unconstrained `TEXT` column, so length costs nothing.
>
> The refusal to truncate is the substantive half. `source_page_id` truncates to
> 16 hex chars = **64 bits**, which is fine for its inputs — a source ID and a
> file path, neither of which an attacker enumerates cheaply. M6's orphan-wikilink
> slot is derived from a **label the attacker writes**, so a 64-bit page ID is
> grindable at ~2³² work: write two labels that collide, and two unrelated topics
> land on one page. Sixty-four hex chars closes it for free. Do not "match the
> tree" here.

Why hash `slot_id` at all rather than use it directly: D5 specifies
`page_id = H("m6-page-v1", slot_id)`, and the indirection means a future
`m6-page-v2` can re-derive page IDs for the same slots without disturbing slot
identity, and vice versa. The two version tags are independent axes on purpose.

---

## 5. Candidate fingerprint

```
candidate_fingerprint = m6_digest("m6-fingerprint-v1", [ …fields below… ])
```

D5's field list, each grounded:

| # | Field | Encoding | Source | Why it is in |
|---|---|---|---|---|
| 1 | `slot_id` | text | §3 | binds the fingerprint to its slot; a fingerprint alone can never be mistaken for another slot's |
| 2 | signal version | int | Stage-0 constant per signal | a signal-logic change must invalidate every candidate it produced |
| 3 | M4 community ID | text, empty part when N/A | `communities.community_id` (`db.rs:10447`) | evidence-cluster and community-overview only |
| 4 | M4 published generation | int, empty part when N/A | `community_members.published_generation` (`db.rs:10464`) / `space_graph_state.published_generation` (`db.rs:10475`) | a re-grouping that keeps the community ID still changes what the community *is* |
| 5 | coverage epoch | int | machine D (artifact 2) | a contract-version epoch change invalidates everything, per D5's closing paragraph |
| 6 | `sorted_set(root ids)` | set (S0-30) | `provenance_roots.root_id` (`db.rs:8788`) | the actual evidence, not just its group summary |
| 7 | input generation | int | `space_graph_state.grouping_generation` (`db.rs:10474`) | the M4 lease/CAS generation the prepare read under (`db.rs:13953`-`:13955`) |
| 8 | active-root digest | text (a `m6_digest` over `sorted_set` of `(root_id, status)` pairs) | `provenance_roots.status` (`db.rs:8793`) | a root going `active → failed` changes the answer without changing field 6 |
| 9 | model version | text | the pinned LLM identifier | a model swap must re-derive before it re-judges |
| 10 | projection version | text | `communities.projection_version` (`db.rs:10451`) | already M4's own re-derivation axis; reused rather than reinvented |
| 11 | prompt version | text | Stage-0 constant | a prompt edit changes the output; D13 forbids the *prompt text* from entering any receipt, so the version tag is the only legal carrier |

> **Decision S0-33 — field 11 (prompt version) is added to D5's list.** D5
> enumerates model version but not prompt version. A prompt edit changes the
> output exactly as a model swap does, and D13 ("capability material, prompts,
> and user identifiers never enter receipts, logs, any export, or committed
> evidence") means the prompt cannot be hashed by content into anything durable.
> An opaque monotone version tag is the only construction that satisfies both.

> **Decision S0-34 — field 3 and field 4 use a zero-length part when
> inapplicable, per §2's absent-optional rule, and are never omitted.** Omitting
> a part shortens the part vector, so an orphan-wikilink fingerprint and a
> community-overview fingerprint could otherwise align their remaining fields.
> Length-prefixing does not save you from a *different number of parts*; a fixed
> arity does.

---

## 6. D8 wikilink key normalization

The input is the raw `target` capture from `WIKILINK_RE`
(`crates/wenlan-core/src/sources/obsidian.rs:16`-`:17`):

```
(!?)\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]
```

The regex already excludes `]`, `|`, and `#` from group 2, and
`synthesis::wikilinks::extract_wikilinks` additionally rejects targets containing
`[` or `]` (`crates/wenlan-core/src/synthesis/wikilinks.rs:52`-`:58`). **Both of
those guards run on pre-normalization bytes**, which is where the first real
problem lives (§6.3).

### 6.1 The ordered algorithm

D8's order with two additions, both justified below. Every step names its
rejection.

```
 0. PRE-CAP        reject if raw target > 1024 Unicode scalars     -> R0
 1. NFKC           normalize
 2. LOWERCASE      str::to_lowercase (Unicode Default Case Conversion)
 3. NFKC AGAIN     re-normalize                                    [addition A]
 4. STRUCTURAL     reject if result contains any of  # | [ ]       [addition B] -> R1
                   (equivalently: strip alias/fragment — see 6.3)
 5. WS COLLAPSE    every maximal run of White_Space -> one U+0020; trim both ends
 6. CONTROL/BIDI   reject if any scalar is category Cc or Cf       -> R2
 7. LENGTH         reject unless 1..=128 Unicode scalars           -> R3
```

The result is `label_key`. The raw target is preserved separately as
`page_links.label` (`db.rs:6674`), unchanged, so a rejected or folded link is
still legible to a human.

### 6.2 Addition A — the second NFKC pass

**Lowercasing can break NFKC form.** [measured] `"\u{0130}".to_lowercase()` in
Rust yields the two scalars `U+0069 U+0307` (`i` + COMBINING DOT ABOVE). Without
a second pass the pipeline emits a string that is not in NFKC, so two inputs that
*should* fold can end up as different keys depending on which case they arrived
in. Normalize → case → normalize is the standard shape (UTS #46 does exactly this
for domain labels).

> **Decision S0-35 — a second NFKC pass runs after lowercasing.** Cheap, and
> without it the pipeline's own output is not a fixed point of the pipeline.

### 6.3 Addition B — the structural rejection must run *after* NFKC

This is the finding that changes D8's order rather than just extending it.
[measured] NFKC folds all four structural characters out of their fullwidth forms:

| Input | Category | NFKC output |
|---|---|---|
| `U+FF03` FULLWIDTH NUMBER SIGN | `Po` | `U+0023` `#` |
| `U+FF5C` FULLWIDTH VERTICAL LINE | `Sm` | `U+007C` `|` |
| `U+FF3B` FULLWIDTH LEFT SQUARE BRACKET | `Ps` | `U+005B` `[` |
| `U+FF3D` FULLWIDTH RIGHT SQUARE BRACKET | `Pe` | `U+005D` `]` |

So `[[Rust＃Ownership]]` — with the fullwidth `＃` — passes `WIKILINK_RE` (the
regex only excludes ASCII `#`), passes the bracket guard at `wikilinks.rs:56`,
and *then* NFKC turns it into `rust#ownership`: a `label_key` containing a
fragment separator that the parser is structurally incapable of producing. Same
route for `｜` producing an alias separator and `［` producing a bracket.

D8's stated order puts "strip aliases/fragments" after NFKC, which handles the
`＃`/`｜` cases correctly. It says nothing about brackets, and the tree's bracket
guard is upstream of normalization, so the `［` case is unguarded today.

> **Decision S0-36 — step 4 is a post-normalization *rejection* of `#`, `|`, `[`,
> `]`, not a strip.** Rejection rather than stripping because by step 4 the parser
> has already split off any legitimate fragment or alias (regex groups 3 and 4,
> `obsidian.rs:166`-`:167`); a separator surviving into the target means the input
> was constructed to smuggle one, and folding it away silently would make
> `[[A＃B]]` and `[[A]]` the same key. The G-catalog gets one case per character.

### 6.4 Addition C — the pre-cap, and why the length check must be last

[measured] the worst single-scalar NFKC expansion across the whole code space is
**1 → 18**: `U+FDFA` ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM expands to 18
scalars. `U+FB03` (ﬃ) is 1 → 3; `U+3392` (㎒) is 1 → 3.

Two consequences. The `1..=128` cap **must** be measured after normalization
(D8's order already puts it last — correct), otherwise 128 accepted input scalars
become up to 2304 stored ones. And the *input* needs its own bound, or a
megabyte-long target burns normalization work before being rejected on length.

> **Decision S0-37 — a 1024-scalar pre-cap (R0) rejects the raw target before
> NFKC.** 1024 is 8× the post-normalization cap, comfortably above the worst
> legitimate expansion ratio for real prose while bounding the work. The number is
> a Stage-0 choice, not a contract value; veto freely.

### 6.5 The control/bidi rejection, stated honestly

> **Decision S0-38 — step 6 rejects every scalar in general category `Cc`
> (control) or `Cf` (format), with no exceptions.**

[measured] categories and NFKC behaviour of the relevant scalars:

| Scalar | Category | NFKC | Verdict |
|---|---|---|---|
| `U+202E` RIGHT-TO-LEFT OVERRIDE | `Cf` | invariant | reject (R2) |
| `U+2066` LEFT-TO-RIGHT ISOLATE | `Cf` | invariant | reject (R2) |
| `U+200B` ZERO WIDTH SPACE | `Cf` | invariant | reject (R2) |
| `U+FEFF` ZERO WIDTH NO-BREAK SPACE / BOM | `Cf` | invariant | reject (R2) |
| `U+200C` ZERO WIDTH NON-JOINER | `Cf` | invariant | reject (R2) — see cost below |
| `U+00A0` NO-BREAK SPACE | `Zs` | → `U+0020` | folded at step 5, accepted |
| `U+3000` IDEOGRAPHIC SPACE | `Zs` | → `U+0020` | folded at step 5, accepted |
| `U+2028` LINE SEPARATOR | `Zl` | invariant | collapsed at step 5 (it is `White_Space`), accepted |

Note the ordering dependency this creates: NBSP and IDEOGRAPHIC SPACE only reach
the whitespace collapse *as spaces* because NFKC ran first. Reversing steps 1 and
5 would leave them intact and then… they would still be caught, because both are
`White_Space=yes`. So this particular pair is order-insensitive; it is called out
because it looks order-sensitive and a future reader will otherwise re-derive it.

**The known cost of the no-exceptions rule.** `U+200C` (ZWNJ) and `U+200D` (ZWJ)
are `Cf` but are *orthographically required* in Persian, and in some Indic and
Malayalam spellings, where they distinguish real words. Rejecting them means
those labels cannot be wikilink keys at all — a genuine harm to a non-Latin-script
user, not a theoretical one.

The alternative — permitting ZWNJ/ZWJ — means two labels that render identically
become different keys, which in M6 means two pages for one topic, and the whole
point of the orphan-wikilink signal is that repeated labels converge. Neither
option is free.

Stage 0 takes rejection because it is D8's literal text and because over-merging
distinct concepts is the failure M6's independence floors exist to prevent, so
the pipeline should not introduce a new merge route. **Upgrade path if a user
reports it:** permit ZWNJ/ZWJ through step 6 and add a confusable-skeleton check
(UTS #39) at step 7 so the rendering-identical pair is caught as a *conflict* to
surface rather than a *merge* to perform. That is strictly more machinery than
Stage 0 should specify on speculation.

### 6.6 Two things `to_lowercase` does not do

Named because a reader will assume otherwise.

**It is not case folding.** [measured] `"\u{00DF}".to_lowercase()` is `ß`, and
`"\u{1E9E}".to_lowercase()` (LATIN CAPITAL LETTER SHARP S) is also `ß` — neither
becomes `ss`. So `[[Straße]]` → `straße` and `[[STRASSE]]` → `strasse` are
**different keys**. Rust's standard library has no `to_casefold`, the tree uses
`to_lowercase` in both existing places (`synthesis/wikilinks.rs:59`,
`db.rs:43916`), and matching it keeps M6's key compatible with the rows already
in `page_links`. Accepted ceiling; the upgrade is a `caseless` dependency, not a
rewrite.

**It is not locale-aware, and that is the desirable property here.** Rust applies
Unicode Default Case Conversion unconditionally, so a Turkish user and a German
user compute the same key for the same label. [measured] `U+FF21` FULLWIDTH LATIN
CAPITAL A lowercases to `U+FF41` fullwidth `a` — which is why NFKC must precede
lowercasing (step 1 folds `Ａ` → `A`, then step 2 gives `a`); the reverse order
yields fullwidth `ａ` and a different key. [measured] `U+212A` KELVIN SIGN
lowercases directly to `k`.

### 6.7 Worked examples

| # | Raw wikilink | Result | Path |
|---|---|---|---|
| W1 | `[[Rust Ownership]]` | `rust ownership` | plain |
| W2 | `[[Rust  Ownership]]` (two spaces) | `rust ownership` | step 5 |
| W3 | `[[ Rust Ownership ]]` | `rust ownership` | step 5 trim |
| W4 | `[[Rust Ownership#borrowing]]` | `rust ownership` | parser splits the fragment (regex group 3) |
| W5 | `[[Rust Ownership\|borrowing rules]]` | `rust ownership` | parser splits the alias (regex group 4) |
| W6 | `[[Ａ Ｂ]]` (fullwidth letters) | `a b` | step 1 then step 2 — **not** step 2 then step 1 |
| W7 | `[[MHz]]` written with `U+3392` | `mhz` | step 1, 1→3 expansion |
| W8 | `[[STRASSE]]` vs `[[Straße]]` | `strasse` vs `straße` | §6.6, distinct on purpose |
| W9 | `[[Café]]` NFC vs `[[Café]]` NFD | `café` (identical) | step 1 |
| A1 | `[[Rust‮Ownership]]` (`U+202E` RLO) | **rejected R2** | bidi override; would render as `rustpihsrenwO` |
| A2 | `[[Rust​Ownership]]` (`U+200B` ZWSP) | **rejected R2** | invisible split of one visible label into two keys |
| A3 | `[[Rust＃Ownership]]` (fullwidth `#`) | **rejected R1** | §6.3 — fragment smuggling past the parser |
| A4 | `[[Rust｜Ownership]]` (fullwidth `\|`) | **rejected R1** | alias smuggling |
| A5 | `[[Rust［Ownership]]` (fullwidth `[`) | **rejected R1** | bracket smuggling past `wikilinks.rs:56` |
| A6 | `[[ﷺﷺﷺ…]]` (44× `U+FDFA`) | **rejected R3** | 44 raw scalars → 792 normalized, over the 128 cap |
| A7 | 2000-scalar target | **rejected R0** | pre-cap, before any normalization work |
| A8 | `[[﻿Rust]]` (leading `U+FEFF`) | **rejected R2** | BOM is `Cf`, not `White_Space` |
| A9 | `[[]]` / `[[   ]]` | **rejected R3** | 0 scalars after trim; the regex's `+` already rejects the empty form, the whitespace-only form reaches R3 |

W1–W9 and A1–A9 are the G-catalog seed for the normalization gate.

### 6.8 What a rejection does

Every rejection path behaves identically, and none of them is a user-facing
error.

| Code | Trigger | Action |
|---|---|---|
| R0 | raw target > 1024 scalars | drop the link, do not write `page_links` |
| R1 | `#`/`\|`/`[`/`]` after normalization | drop the link, do not write `page_links` |
| R2 | any `Cc`/`Cf` scalar | drop the link, do not write `page_links` |
| R3 | normalized length outside `1..=128` | drop the link, do not write `page_links` |

In all four cases: the containing page is written normally, the raw link text
survives in the page body, a counter is incremented per `(space, code)`, and one
`log::debug!` line records the code and the page ID. **The rejected label itself
is never logged** — D13 forbids user content in logs, and a wikilink label is
user content by definition.

> **Decision S0-39 — a rejected link is dropped, never an ingest failure.** A
> page with one adversarial link is still a page the user wants. Failing the whole
> write would turn a normalization edge case into data loss, and would give an
> attacker who can get one link into a document a way to block ingest of the whole
> document.

Rejected links are invisible to the orphan-wikilink signal by construction: the
signal counts `page_links` rows (D2.2, artifact 1 §3), and a rejected link never
becomes one. This is the intended containment — normalization is the *gate*, not
a filter applied later.

---

## 7. Retry and replay identity

### 7.1 The convention M6 inherits

`operation_receipts` (`crates/wenlan-core/src/db.rs:8217`-`:8224`):

```sql
CREATE TABLE IF NOT EXISTS operation_receipts (
    caller_id      TEXT NOT NULL,
    operation_id   TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    response       TEXT NOT NULL,
    created_at     INTEGER NOT NULL,
    PRIMARY KEY (caller_id, operation_id)
);
```

Its semantics are set by `post_write::page_update`: the retry identity is the
`(caller_id, operation_id)` pair (`page_update.rs:502`), the request digest is
`page_write_digest` (`page_update.rs:349`), and a receipt whose digest matches
replays the stored response while a mismatch is a conflict
(`page_update.rs:388`-`:391` doc comment). That is the shape D5's "same
operation/digest replays; collision conflicts" describes.

### 7.2 M6's mapping

| Receipt column | M6 value |
|---|---|
| `caller_id` | the literal `m6-genesis` (the daemon is the only caller; there is no user identity here, per D13) |
| `operation_id` | `candidate_id` — see S0-40 |
| `request_digest` | `candidate_fingerprint` (§5) |
| `response` | the finalization outcome payload (page ID, claim disposition, terminal reason) |

> **Decision S0-40 — `candidate_id = m6_digest("m6-candidate-v1", [slot_id,
> coverage_epoch])`, not a UUID.** D5 requires that a retry *reuse* the candidate;
> a minted UUID would have to be looked up to be reused, which means the lookup
> key is really `(slot_id, coverage_epoch)` and the UUID is an indirection with a
> crash window in it. Deriving the ID removes the window: a process that crashes
> between observing a slot and writing its row recomputes the identical
> `candidate_id` on restart. The coverage epoch is in the derivation because D5
> says a new epoch does not reuse prior coverage, so a new epoch must be a new
> candidate for the same slot.

### 7.3 Row-level outcomes

| Situation | `(caller, op)` | digest | Outcome |
|---|---|---|---|
| Honest retry — same inputs | hit | equal | **Replay.** Return the stored `response`. No new lease, no LLM call, no page write. |
| Re-prepare after staleness | hit | **differs** | **Not a conflict.** The candidate legitimately re-fingerprints (machine A, transition **A19**). The receipt row is *replaced* under the same key with the new digest. See S0-41. |
| Distinct concurrent finalize | hit | differs | **Conflict** if the existing receipt is terminal. Refuse; the first finalization won. |
| First attempt | miss | — | Proceed. |

> **Decision S0-41 — a non-terminal receipt row is replaceable under a new
> fingerprint; a terminal one is not.** This is where M6 diverges from
> `page_update`'s flat rule (any digest mismatch is a conflict), and it has to:
> `page_update`'s operation ID comes from an external caller who is expected to
> mint a new one per logical write, whereas M6's is *derived* from the slot, so
> the same operation ID legitimately recurs across re-preparations of the same
> topic. Without the terminal/non-terminal split, the second preparation of any
> slot would conflict with the first forever.
>
> "Terminal" is exactly machine A's terminal set from artifact 2 (`published`,
> `suppressed`, `superseded`, and `review_required` while the review is open).
> `stale` is deliberately not in that set, and artifact 2 rev 2 now agrees — see
> its S0-151.

### 7.4 What "retry reuses candidate, slot, page ID, lease operation, receipt" is, row by row

| Thing reused | The row | The mechanism |
|---|---|---|
| candidate | `genesis_candidates` (PR-A-new), PK `candidate_id` | derived, S0-40 |
| slot | the `slot_id` column of that same row | derived, §3; never recomputed after `observed` (S0-29) |
| page ID | `pages.id` | derived from `slot_id`, §4; `INSERT … ON CONFLICT(id) DO NOTHING` converges, the same pattern `compute_edge_id` uses for edges (`provenance.rs:180`-`:183`) |
| lease operation | `grouping_leases`, PK `(phase, space, input_generation)` (`db.rs:10479`, PK at `:10486`) | the existing M4 registry with M6's phase values (artifact 2, machine C); a retry re-acquires the same key, and `ON CONFLICT … DO NOTHING` (`db.rs:13447`) makes a live holder win |
| receipt | `operation_receipts`, PK `(caller_id, operation_id)` (`db.rs:8223`) | §7.2 |

All five are keyed on derived values. There is no minted identifier anywhere in
the retry path, which is the property that makes crash-restart identity hold
without a recovery scan having to reconstruct anything.

---

## 8. Findings against the tree

Reported, not resolved. F1 and F2 are pre-existing and out of M6's scope to fix;
F3 and F5 are things PR-A must decide.

**F1 — `source_page_id` is separator-framed over an unvalidated string.**
`crates/wenlan-core/src/document_enrichment.rs:760`-`:769` hashes
`b"source_page::" ‖ source_id ‖ b"::" ‖ file_path`. A `source_id` containing `::`
aliases with a shorter `source_id` and a longer `file_path`. Not currently
exploitable as far as this artifact checked — source IDs appear to be
daemon-minted — but it is the exact construction `compute_edge_id`'s doc comment
argues against, in a function written later. Worth a one-line follow-up, not an
M6 blocker.

**F2 — `source_page_id` truncates to 64 bits** (`document_enrichment.rs:768`,
`&hex[..16]`). Same file, same function. Fine for its inputs; called out because
"match the existing page-ID convention" would silently import it into M6 where the
inputs are attacker-chosen. S0-32 declines it deliberately.

**F3 — the bracket guard runs before normalization.**
`crates/wenlan-core/src/synthesis/wikilinks.rs:52`-`:58` rejects targets
containing `[` or `]`, with the stated rationale that such labels "would poison
page_links and the orphan-by-count feed." [measured] `U+FF3B` NFKC-folds to `[`,
so a fullwidth bracket passes this guard and becomes an ASCII bracket the moment
M6 normalizes. The guard is correct for today's pipeline (which never normalizes)
and insufficient for M6's. S0-36 covers it; the existing guard should stay where
it is rather than be moved, since it also protects the raw `label` column.

**F4 — the community-overview slot inherits M4's UUID rebinding.** §3.5. PR-A-new.

**F5 — `page_links.label_key` today is `to_lowercase()` and nothing else**
(`crates/wenlan-core/src/db.rs:43916`, `let label_key = link.label.to_lowercase();`).
M6's key is strictly stronger, so every existing row's `label_key` is either
identical to the M6 key (the overwhelmingly common ASCII case) or weaker. Since
`label_key` is half the `page_links` primary key (`db.rs:6675`), re-keying is a
migration with a collision case: two existing rows on one page whose old keys
differ but whose M6 keys agree. **PR-A must choose** between (a) migrate and
merge colliding rows, (b) leave `label_key` alone and have M6 compute its key at
read time, or (c) add a second column. Option (b) is the smallest and forfeits the
orphan index (`idx_page_links_orphan ON page_links(label_key)`, `db.rs:6681`),
which the orphan-wikilink signal wants; that tradeoff is PR-A's to make, not
Stage 0's.

---

## 9. Gate mapping

| Gate | What this artifact hands it |
|---|---|
| G2 (signal boundaries) | W1–W9, A1–A9 (§6.7) as normalization boundary cases; the R0–R3 drop semantics (§6.8) as the assertion that a rejected link never reaches a floor count |
| G4 (identity/retry) | §7.3's four outcomes as the receipt table's contract; §7.4's five reused rows as the crash-restart assertion; S0-40's derivation as the property that makes the assertion checkable without a recovery scan |
| G6 (abuse bounds) | the pre-cap (S0-37), the post-normalization length cap, and A6/A7 |
| G-catalog | `C-slot-community-rebind` (§3.5), one case per structural character (§6.3), the ZWNJ decision's upgrade path (§6.5) |

---

## 10. Decisions introduced here

`S0-26` length-prefixed framing, not separator framing ·
`S0-27` SHA-256 / u64 LE / full 64-hex ·
`S0-28` domain tag is a length-prefixed part ·
`S0-29` evidence-cluster slot set pinned at `observed` ·
`S0-30` set encoding: dedup, byte-lex sort, count-prefixed ·
`S0-31` space string is not normalized ·
`S0-32` `m6p_` + full digest, no truncation ·
`S0-33` prompt version added to the fingerprint ·
`S0-34` inapplicable fingerprint fields are zero-length, never omitted ·
`S0-35` second NFKC pass after lowercasing ·
`S0-36` post-normalization structural rejection ·
`S0-37` 1024-scalar pre-cap ·
`S0-38` reject all `Cc`/`Cf`, no ZWNJ exception ·
`S0-39` a rejected link is dropped, not an ingest failure ·
`S0-40` `candidate_id` is derived, not minted ·
`S0-41` non-terminal receipts are replaceable, terminal ones are not.

S0-31 is stated inline in §3.1 and repeated here for completeness: the `space`
part is the raw stored string. Spaces are daemon-managed identifiers with their
own uniqueness discipline, and normalizing them inside M6 would create a second
opinion about space identity that the rest of the daemon does not share. If space
names ever need normalization, that belongs to the spaces layer and M6 inherits
it for free.
