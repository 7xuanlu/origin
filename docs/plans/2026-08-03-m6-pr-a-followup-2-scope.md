# M6 PR-A follow-up 2 — scope for the two substrate gaps blocking PR-B contract item 1

**Status.** Scope and design only. No schema, no code, no migration written here.

**Grounding.** Every `file:line` below was read on branch `m6-pr-b1` in
`/Users/lucian/Repos/wenlan/.claude/worktrees/m5-truth-promoter` during this
investigation. Nothing was compiled or executed — the build lane belongs to
another agent — so every claim here is a claim about **source text I read**, and
the few claims that would need a run to settle are marked `[unverified]`
explicitly.

**Blocking relationship.** PR-B's contract item 1 ("reconcile all four genesis
signals") needs both gaps closed. PR-B2 needs Gap B alone. The two gaps are
independent of each other in every dimension — files, tables, migration, tests.

---

## 1. The two gaps, restated against code I read

### Gap A — no claim-revision-to-link binding

Decision S0-15 (`docs/plans/2026-08-01-m6-signal-matrix.md:104`) requires the
orphan-wikilink signal's "underlying groups" to be scoped to *"the exact current
claim revisions that contain the link"*, and forbids widening to the whole
referring page. It also says that if the binding does not exist, that is a
`PR-A-new` item to name.

**It does not exist as a stored column, and the report is accurate.**

| Fact | Anchor |
|---|---|
| `page_links` is `(source_page_id, target_page_id, label_key, label)`, PK `(source_page_id, label_key)` — no claim, revision, or offset column | `crates/wenlan-core/src/db.rs:6691`–`:6702` |
| Its only production writer is `replace_page_links`, which DELETEs the page's whole row set and re-INSERTs | `crates/wenlan-core/src/db.rs:44393`–`:44510`; insert at `:44426`–`:44436` |
| `label_key` is `to_lowercase()` and nothing more, confirming S0-14 | `crates/wenlan-core/src/db.rs:44425` |
| Orphan rows deliberately produce **no** `edges` row | `crates/wenlan-core/src/db.rs:44438`–`:44440` |
| The extractor returns labels only — `obsidian::Wikilink` carries `target`/`heading`/`display`/`is_embed`, no byte offsets | `crates/wenlan-core/src/sources/obsidian.rs:144`–`:151`, `:154`–`:170` |

**But the binding is recomputable, and PR-A already built the half that makes it
so.** `claim_anchors` binds each claim revision to a byte span **in the page
body**:

- The derivation worker writes `claim_anchors` with
  `source_doc_id = job.page_id`, `source_version = job.page_version`,
  `span_start/span_end = cut.raw_start/raw_end`, `span_digest = revision_content_digest(raw)`
  — `crates/wenlan-core/src/db/claim_derivation.rs:1416`–`:1434`.
- `raw_start`/`raw_end` are offsets into the **raw page body**, mapped back from
  the marker-free body by `marker_free_body_with_offset_map` —
  `crates/wenlan-core/src/db/claim_derivation.rs:1154`–`:1176`.
- That mapper strips **only** `[N]` numeric citation markers. `[[Label]]` is not
  stripped: at the first `[` the scan requires at least one ASCII digit before
  the closing `]`, and the next byte is `[` — `crates/wenlan-core/src/db/claim_derivation.rs:193`–`:207`.
  So wikilink syntax survives verbatim inside every anchored span **and** inside
  `claim_revisions.canonical_text`.
- `claim_anchors` DDL and its immutability trigger:
  `crates/wenlan-core/src/db/claim_identity.rs:205`–`:223`.

So the answer to "does the binding exist" is: **not as a column, but as a
derivable join over data PR-A already stores.** That changes the shape of the
fix materially (§2).

**A second finding, not in the original gap statement, that would have silently
produced a dead signal.** Every `supports` edge is written with `grounded = 0`
— literal `0` in the VALUES list at
`crates/wenlan-core/src/db/claim_identity.rs:1391`, with the reasoning at
`:954`–`:961` ("grounding is inherited and never asserted"). `root_id` **is**
written (`:1396`). Meanwhile the canonical count requires `e.grounded = 1`
(`crates/wenlan-core/src/m6/independence.rs:70`). The obvious scope predicate —
`e.edge_type = 'supports' AND e.src_id IN (<revisions>)` — therefore returns
**zero groups on every install, forever**, and would read as "orphan wikilinks
are rare" rather than "the scope is structurally empty". §2.3 gives the scope
shape that avoids this.

### Gap B — no genesis consumer for the community durable-reader gate

| Fact | Anchor |
|---|---|
| Two consumer literals exist | `crates/wenlan-core/src/db.rs:2509`–`:2510` |
| `community_reader_durable_gate_sql` fail-closes to `"0=1"` for anything else | `crates/wenlan-core/src/db.rs:2663`–`:2670` |
| A second literal list gates `is_known_community_reader` | `crates/wenlan-core/src/db.rs:15019`–`:15024` |
| A third literal list drives `reconcile_pending_community_readers` | `crates/wenlan-core/src/db.rs:15084`–`:15101` |
| A fourth appears as a SQL literal inside the reconciler's candidate query | `crates/wenlan-core/src/db.rs:15197` |
| A fifth is the per-consumer output-delta branch | `crates/wenlan-core/src/db.rs:15397`–`:15420` |
| Migration seeds cutover **intent** for both consumers at `enabled = 1` | `crates/wenlan-core/src/db.rs:11391`–`:11408` |
| `set_community_reader_cutover` is the manual lever, validated by `is_known_community_reader` | `crates/wenlan-core/src/db.rs:14991`–`:15017` |

**One correction to the gap statement as handed over.** The task says the new
consumer needs "migration-seeded rows across `community_reader_cutover` /
`community_reader_current_input` / `community_reader_watermark` /
`community_reader_space_proof`". That is not how the existing consumers work.
The migration seeds **only** `community_reader_cutover`
(`crates/wenlan-core/src/db.rs:11396`–`:11407`). The other three are written
**exclusively by the reconciler**, in one transaction at the end of
`reconcile_community_reader_parity` —
`crates/wenlan-core/src/db.rs:15578`–`:15647`. There is no other writer; I
searched every `INSERT`/`UPDATE`/`DELETE` naming those three tables.

That correction is load-bearing for §3: seeding proof/watermark rows by
migration would be **manufacturing a proof nobody took**, which is exactly the
failure the whole M4 gate exists to prevent.

---

## 2. Gap A design

### 2.1 Recommendation: derive at read time. No schema, no migration, no backfill.

The binding is `containment of a wikilink occurrence's byte range inside an
anchored claim-revision span, in the page's current body`. Every input is
already stored. The read side computes it in Rust, which is where it has to be
anyway: S0-14 requires the D8 normalizer
(`crates/wenlan-core/src/m6/label_key.rs:57`–`:107`) to decide which raw targets
collapse onto one candidate label, and that normalizer cannot be expressed in
SQLite (NFKC, double-lowercase fixed point, Unicode `Cf` rejection). The read
side is already Rust-filtering labels; adding containment costs it one more
loop, not a new architecture.

**The algorithm, per candidate label:**

1. Candidate referring pages come from the existing orphan predicate:
   `page_links.target_page_id IS NULL`, D8-normalize `page_links.label` in Rust,
   keep the pages whose normalized key equals the candidate's.
2. For each referring page, read `pages.content` and `pages.version`.
3. Load the current revision set with their spans:
   ```sql
   SELECT pvc.claim_revision_id, ca.span_start, ca.span_end, ca.span_digest
     FROM page_version_claims pvc
     JOIN claim_anchors ca
       ON ca.claim_revision_id = pvc.claim_revision_id
      AND ca.source_doc_id     = pvc.page_id
      AND ca.source_version    = pvc.page_version
    WHERE pvc.page_id = ?1 AND pvc.page_version = ?2
   ```
   `page_version_claims` is DELETEd and rewritten per version
   (`crates/wenlan-core/src/db/claim_derivation.rs:1306`–`:1313`,
   `:1435`–`:1449`), so "the current version's rows" **is** the current revision
   set — there is nothing to compute.
4. **Verify each span before trusting it**: `revision_content_digest(content[span_start..span_end]) == span_digest`.
   A mismatch drops that revision, silently and fail-closed. This is the exact
   check the derivation aligner already performs against a prior version
   (`crates/wenlan-core/src/db/claim_derivation.rs:1269`–`:1271`), so it is a
   reuse of an established idiom, not a new invariant.
5. Scan the page body for `[[...]]` occurrences **with byte offsets**, D8-normalize
   each raw target, and keep the occurrences whose key matches the candidate.
6. A revision *contains* the link iff some kept occurrence's range lies inside
   that revision's verified span. Collect those `claim_revision_id`s.

**The one new piece of code** is an offset-carrying wikilink extractor. Today
`obsidian::extract_wikilinks` (`crates/wenlan-core/src/sources/obsidian.rs:154`–`:170`)
throws the offsets away, but it already computes them — `cap.get(0)` is taken at
`:160` for the code-block filter, and `cap.get(3)`/`cap.get(4)` are read at
`:166`–`:167`, so `cap.get(2)` (the target group) yields a `Match` with
`.start()`/`.end()` for free. Add a sibling function that returns the range
alongside the existing fields, and express the current function in terms of it
so **one regex and one code-block filter** serve both the write path and M6.
That is the property that makes read-time derivation safe: M6 is not re-deriving
what the writer derived, it is reusing the same extractor over the same bytes.

### 2.2 Why not a stored table

A `claim_revision_links(claim_revision_id, label_key, …)` table written inside
the derivation transaction (the insert loop at
`crates/wenlan-core/src/db/claim_derivation.rs:1383`–`:1449` already holds
`cut.raw_start..cut.raw_end` and the body) is a coherent design and would take
about the same number of lines. It buys SQL-joinability and avoids a per-read
body scan. It costs: a migration, a backfill over already-derived pages,
supersession semantics to define and test, and a layering inversion — the M5
derivation writer would have to call M6's D8 normalizer to compute the key, or
store raw targets and force the read side to normalize anyway, which is where we
started.

Recommend read-time derivation. Volume is small by construction: the signal
needs ≥ 2 referring pages, D8 caps links per page at 64, and the frontier policy
bounds candidates per pass.

**Explicitly rejected: a `canonical_text LIKE '%[[' || ? || ']]%'` binding.**
`claim_revisions.canonical_text` does contain the wikilink syntax verbatim, so
this looks like a one-line answer. It is wrong: `%`/`_` inside a user label are
LIKE metacharacters needing an `ESCAPE` clause, aliases (`[[X|display]]`) and
fragments (`[[X#h]]`) do not match the literal, and the D8 normalization cannot
run in SQLite — so the predicate would silently disagree with the label grouping
that produced the candidate.

### 2.3 The count scope, written so it is not structurally empty

`GroupCountScope.predicate` may reference only `e` and `r`, reaching other
tables through a correlated `EXISTS`
(`crates/wenlan-core/src/m6/independence.rs:34`–`:44`). Given the `grounded = 1`
finding in §1, the scope must make `e` a **grounded** edge and bridge to the
claim revisions through the shared `root_id` that the supports edge records:

```
EXISTS (
  SELECT 1 FROM edges s
   WHERE s.edge_type   = 'supports'
     AND s.src_kind    = 'claim_revision'
     AND s.src_id      IN (<the revision ids from §2.1>)
     AND s.valid_until IS NULL
     AND s.root_id     = e.root_id
)
```

`e` remains a grounded, active, non-generated, non-overview edge (R1–R5 stay
owned by `independence.rs`), and the narrowing says "…whose provenance root is
one that a live support edge on a containing revision names". The revision ids
bind as positional params. Note that `scoped_grounded_node_count`
(`crates/wenlan-core/src/m6/signals.rs:187`–`:216`) duplicates `scope.params`
for its two UNION arms, so a multi-param scope must number consistently across
both — a real constraint on how the `IN` list is emitted.

### 2.4 Write path, backfill, supersession

- **Who writes it:** nobody. There is no new stored state.
- **When:** at read time, inside the caller's transaction, per the PR-B reader
  contract (`crates/wenlan-core/src/m6/signals.rs:1`–`:5`: pure readers, open no
  transaction, write nothing).
- **Backfill:** none. Existing installs and fresh installs are identical the
  moment the code ships.
- **Supersession:** structurally handled. A superseded revision is one that is
  no longer in `page_version_claims` at the page's **current** version, so step 3
  never returns it. A revision that survives a page edit keeps its
  `claim_revision_id` only when the aligner matched it
  (`crates/wenlan-core/src/db/claim_derivation.rs:1368`–`:1381`); a changed
  sentence mints a new revision id, and the new one is what step 3 returns.
- **A page whose derivation is stale** (body at version N, anchors at N−1)
  yields an empty revision set at step 3 and therefore contributes zero groups.
  That is fail-closed and it costs nothing in practice, because predicate 2.5
  already requires the referring page to be `supported`, and support is
  **version-bound**: `finalize_page_support` stamps
  `page_truth_state.page_version` to the finalized version
  (`crates/wenlan-core/src/db/claim_derivation.rs:4582`–`:4595`). A page that is
  `supported` at its current version has current-version anchors by
  construction. PR-B's 2.5 check must compare
  `page_truth_state.page_version = pages.version`, not merely
  `support_status = 'supported'` — that comparison is what carries this
  guarantee, and it is worth a test of its own.

---

## 3. Gap B design

### 3.1 The wiring points, exhaustively

| # | Site | Change |
|---|---|---|
| 1 | `crates/wenlan-core/src/db.rs:2509`–`:2510` | add `COMMUNITY_GENESIS_CONSUMER` (proposed literal: `"m6_genesis"`) |
| 2 | `crates/wenlan-core/src/db.rs:2664`–`:2667` | add the constant to the gate's `matches!`; the `"0=1"` default at `:2669` is untouched |
| 3 | `crates/wenlan-core/src/db.rs:15019`–`:15024` | add the constant to `is_known_community_reader` |
| 4 | `crates/wenlan-core/src/db.rs:15086`–`:15089` | add the constant to the reconcile loop |
| 5 | `crates/wenlan-core/src/db.rs:15196`–`:15207` | **no SQL change needed** — the branch is `?1 <> 'summary_buckets'`, so genesis inherits the no-pending-revision-filter arm, which is the correct universe for it. Add a comment naming genesis so the inheritance is deliberate rather than accidental |
| 6 | `crates/wenlan-core/src/db.rs:15397`–`:15420` | add a genesis arm to the output-delta branch (see §3.2) |
| 7 | migration 111 | seed one `community_reader_cutover` row at `enabled = 0` (see §3.3) |
| 8 | new, M6-owned | one named seam so `m6/signals.rs` does not import from `db.rs` (see §3.5) |

**Nothing about the gate body changes.** Its five blocking terms — `enabled = 1`
(`:2684`), the contract-version match (`:2685`), `unexplained_drift_count = 0`
(`:2686`), the space-proof count equality (`:2687`–`:2690`), and the per-space
`NOT EXISTS` over state and receipts (`:2691`–`:2710`) — apply to any consumer
by construction, because the SQL is parameterised only by the consumer literal.
Adding a third literal weakens nothing.

### 3.2 The reconciliation output, and why the stakes are lower than they look

**First, the fact that decides the shape of the argument.** `output_delta` does
**not** gate. The blocking quantity is
`unexplained_total = source_coverage_delta + invalid_publication_proofs`
(`crates/wenlan-core/src/db.rs:15442`), and `output_delta` only ever becomes
`explained_structural_delta_count`, and only when coverage is already clean
(`:15443`–`:15447`). The gate reads `unexplained_drift_count = 0` (`:2686`) and
never reads `output_delta_count`. So the choice below changes what the receipt
**records**, not whether the reader opens.

**Recommendation: genesis's output is the partition set — the same computation
as `summary_buckets`.** M6's community-consuming signals read exactly one thing
about M4: which nodes are in which community. Signal 1 scopes edges to a
community's `community_members.node_id` set; signal 3 counts
`attachment = 'core'` members. The partition symmetric-difference is the
strictest legacy-comparable statement about that.

**Why not "the admitted-candidate set".** It is the tempting answer — make the
delta about what M6 would actually emit. It cannot work. The legacy side is
`entities.community_id` written by label propagation, which has no `attachment`
column and no published generation, so the legacy arm of an
attachment-and-floor-sensitive computation is **empty by construction**. The
symmetric difference would then equal the entire durable set, `output_delta`
would be permanently nonzero, and — while that would not block the gate — the
receipt would report constant drift that no operator could ever clear. A number
that can only ever be wrong is worse than no number.

**Why not zero.** Defensible (it drops a field that does not gate), but it
throws away the one legacy-vs-durable signal genesis actually consumes, and it
makes genesis the only consumer whose receipt says nothing about output.

Concretely, wiring point 6 becomes: the partition branch fires for
`summary_buckets` **or** genesis; the eligibility branch stays exclusive to
`summary_eligibility`.

### 3.3 Fresh install versus existing install

Both reach the identical state, and the reason is that **absence and
`enabled = 0` are the same thing**: `community_reader_cutover.enabled` is
`NOT NULL DEFAULT 0 CHECK(enabled IN (0,1))`
(`crates/wenlan-core/src/db.rs:10974`–`:10978`), and the gate's `EXISTS` finds
no row at all when none was seeded, which evaluates false exactly as
`enabled = 0` does.

| | Fresh install | Existing install |
|---|---|---|
| after migration 111 | one genesis cutover row, `enabled = 0` | same row, inserted `OR IGNORE` |
| `community_reader_uses_durable("m6_genesis")` | `false` | `false` |
| `reconcile_pending_community_readers` | skips genesis — `community_reader_parity_needs_reconcile` requires `enabled = 1` (`crates/wenlan-core/src/db.rs:15056`–`:15059`) | same |
| watermark / space-proof / current-input rows | none | none |
| older daemon reading a 111-stamped DB | one unknown row in a control-plane table it never queries by that literal; its own two consumers are untouched | same |

**Enabling genesis is a deliberate two-step, and that is the rollout control.**
`set_community_reader_cutover("m6_genesis", true)` records intent; the gate stays
closed until the next `CommunityDetection` phase runs a reconcile pass that
writes a clean, current watermark and a full set of space proofs. There is no
path from "flip the flag" to "reader is live" inside one call.

**Why seed the row at all, given absence already means disabled.** One reason,
and it is a fence rather than a feature: the existing migration seeds cutover
intent by **looping over a consumer array at `enabled = 1`**
(`crates/wenlan-core/src/db.rs:11392`–`:11408`). The next person who adds a
consumer will pattern-match that loop, and genesis would inherit `enabled = 1`
— which destroys the rollout-control property the ruling protects. An explicit,
separately-written seed at `0`, carrying a comment saying genesis is deliberately
not in that loop, is a cheap standing objection. If the reviewer prefers zero
migrations, dropping the seed is behaviourally identical and the fence becomes a
test-only assertion (§5, G-B1).

### 3.4 The five literal lists

Adding a third consumer means editing five places (§3.1 rows 2–6) that each hold
their own copy of the consumer set, with no mechanism forcing agreement. A
divergence is silent and asymmetric: a consumer in `is_known_community_reader`
but missing from the gate's `matches!` is permanently `"0=1"` — dark, not
broken, which is the hardest failure to notice.

This repo already refuses that pattern one table over:
`PARITY_GUARD_TRIGGERS` is *"written once so the installer and every validator
read the same text and cannot drift apart"* (`crates/wenlan-core/src/db.rs:2525`–`:2546`).
Recommend the same here: one `COMMUNITY_READER_CONSUMERS: [&str; 3]`, with
rows 2–4 reading from it, plus a test asserting that every element yields
non-`"0=1"` SQL and that a non-element yields `"0=1"`. Rows 5–7 stay explicit —
the candidate-query branch and the `enabled = 1` seed loop are per-consumer
*policy*, not membership, and folding them in is how genesis would end up
enabled by default. This touches M4 gate code, so it is a ruling (§7 Q4), not
something to do quietly.

### 3.5 The seam for PR-B

The constraint is that PR-B's read side must not reach into `db.rs`. The gate
SQL builder is `pub(crate)` and `m6` is in the same crate, so `signals.rs`
*could* call it directly; the constraint says it should not.

Recommend **one M6-owned wrapper** — a small module exposing, say,
`m6::community_gate::durable_gate_sql()` and
`m6::community_gate::reader_is_live(tx)` — that makes the single call into
`db.rs` on M6's behalf. Signals then depend on an M6 name.

Do **not** relocate `community_reader_durable_gate_sql` out of `db.rs`. It is
M4's central fail-closed gate, it survived four independent review rounds, and
moving it is a large blast radius for zero behaviour change.

---

## 4. Migration plan

**One migration, number 111, and it carries Gap B only.** Gap A ships as pure
code with no schema and no migration (§2.4). The latest applied version is 110
(`crates/wenlan-core/src/db.rs:12299`), and `migrate_110_judge_eligibility`
(`:12257`–`:12304`) is the shape to mirror.

- **Content:** one `INSERT OR IGNORE INTO community_reader_cutover` for the
  genesis consumer at `enabled = 0`, inside the migration's own immediate
  transaction, then the `PRAGMA user_version = 111` bump.
- **Posture:** additive and inert. It creates no table, alters no column,
  changes no reader or writer fence, and its one row is semantically identical to
  the row's absence (§3.3). The established idiom for M6 migrations is
  `crate::m6::<module>::ensure_*(&tx)` (`:12235`–`:12243`); this one has no
  table to ensure, so it is a bare seed.
- **Ordering:** Gap A can merge before, after, or without 111. Gap B's code
  half (constants and match arms) is likewise independent of the migration —
  with no seeded row the gate still fail-closes correctly. If the reviewer takes
  the zero-migration option (§7 Q3), the follow-up ships with **no** migration at
  all.
- **Older daemons reading a 111-stamped database:** unaffected. Migrations in
  this tree are forward-only version-gated blocks; a daemon at 110 sees
  `user_version = 111`, runs none of the `version < N` arms, and queries
  `community_reader_cutover` only by its own two consumer literals
  (`:2683`, `:15058`). The extra row is invisible to it.

---

## 5. RED-first gate list

Each gate names the mutation that must turn it red, and the positive control
that proves it discriminates rather than merely passing.

### Gap A

| Gate | Assertion | Mutation (must fail) | Positive control |
|---|---|---|---|
| `m6_orphan_link_scoped_to_containing_revision` | a page with two revisions, the link in revision 1 only, revision 2 carrying two *additional* independence groups → the count sees only revision 1's groups | widen the scope to every revision of the page | move the link into revision 2 → revision 2's groups now count and revision 1's do not |
| `m6_orphan_link_scope_is_not_structurally_empty` | a correctly-set-up candidate with 3 groups **admits** | rewrite the scope as `e.edge_type='supports' AND e.src_id IN (…)` (the naive form) → count collapses to 0 | the same fixture with 2 groups rejects, proving the count moves |
| `m6_orphan_link_binding_verifies_anchor_digest` | a revision whose stored `span_digest` no longer matches the live body contributes zero | drop the digest verification (step 4) → the stale revision contributes | the same fixture with an intact digest contributes |
| `m6_orphan_link_binding_uses_d8_key` | `[[Ｒust]]` (fullwidth) and `[[rust]]` in two different revisions both bind to one candidate | bind on the raw target text instead of `normalize_label_key` | `[[rust]]` and `[[haskell]]` do **not** collapse |
| `m6_orphan_link_binding_reads_alias_and_fragment_forms` | `[[Label\|display]]` and `[[Label#heading]]` bind to `label` | bind on capture group 0 rather than the target group | a genuinely different target does not bind |
| `m6_orphan_link_ignores_superseded_revision` | a revision present at version N−1 but absent from `page_version_claims` at N contributes zero | drop the `page_version = pages.version` join condition | the version-N revision containing the link does contribute |
| `m6_orphan_link_supported_check_is_version_bound` | a page whose `page_truth_state.support_status='supported'` names an **older** `page_version` is not an eligible referring page | compare only `support_status`, not the version | the same page with a current-version stamp is eligible |

### Gap B

| Gate | Assertion | Mutation (must fail) | Positive control |
|---|---|---|---|
| `m6_genesis_consumer_seeded_disabled` | after migration 111 the genesis cutover row has `enabled = 0` | change the seed literal `0` → `1` | the two summary consumers are still seeded `1`, so the test is reading a real value |
| `m6_genesis_gate_closed_without_receipt` | with genesis enabled by the manual lever but no reconcile pass, `community_reader_uses_durable` is `false` | delete the `unexplained_drift_count = 0` term from the gate | after a clean reconcile the same call returns `true` |
| `m6_genesis_unknown_consumer_still_fails_closed` | `community_reader_durable_gate_sql("m6_genesis_typo") == "0=1"` | replace the `matches!` with a permissive default | `community_reader_durable_gate_sql(GENESIS)` is **not** `"0=1"` |
| `m6_genesis_reconcile_requires_intent` | with genesis at `enabled = 0`, `reconcile_pending_community_readers` writes no genesis watermark row | drop the `enabled = 1` condition at `:15058` | at `enabled = 1` the same call does write one |
| `m6_genesis_drift_blocks_gate` | a relevant space with no matching `community_publication_receipts` row → `unexplained_drift_count > 0` → gate `false` | count invalid publication proofs as explained rather than unexplained | with the receipt present, gate `true` |
| `m6_consumer_lists_agree` | every element of the consumer set yields non-`"0=1"` SQL **and** passes `is_known_community_reader`; a non-element fails both | add a literal to one list only | holds for the two pre-existing consumers, so it is not vacuous |
| `m6_genesis_not_enabled_by_the_seed_loop` | the migration's `enabled = 1` seed loop contains exactly the two summary consumers | add genesis to that loop | the loop still seeds both summary consumers at 1 |

---

## 6. Test plan — hermetic, no GPU, no LLM

Everything above is a database-state test. Nothing needs the on-device model,
because both gaps are read-side: claim revisions, anchors, support edges,
provenance roots, and community rows are all **inserted directly** by the
fixture rather than produced by running the judge. The existing fixtures already
do exactly this — `crates/wenlan-core/src/db/claim_derivation_test.rs:895`–`:925`
inserts `claims` / `claim_revisions` / `page_version_claims` by hand, and
`crates/wenlan-core/src/db/claim_identity_test.rs:732`–`:745` does the same.

**Placement.** Gap A tests belong in `crates/wenlan-core/src/m6/signals_test.rs`
(the orphan-wikilink signal's own module) with the offset-extractor unit tests in
`crates/wenlan-core/src/sources/obsidian.rs`'s existing `mod tests`. Gap B tests
belong beside the M4 gate's existing coverage in `db.rs`'s test module, next to
whatever already exercises `community_reader_uses_durable`.

**Fixture shape for Gap A** (one helper, reused by all seven gates):

1. a space, a page with a body containing two sentences and a `[[Label]]` in a
   named one;
2. `claims` + `claim_revisions` + `claim_anchors` + `page_version_claims` at the
   page's current version, with `span_start`/`span_end` computed from the fixture
   body so the digest verification passes for real rather than by construction;
3. `page_truth_state` at `supported`, `page_version` = the page's version;
4. N `provenance_roots` with distinct `independence_group_id`, and one grounded
   `relates` edge per root;
5. one `supports` edge per (revision, root) pair, written with `grounded = 0` —
   **matching production**, so the test would catch the §2.3 structural-emptiness
   trap rather than paper over it.

Point 5 is the one place a fixture could quietly lie. A fixture that writes
`grounded = 1` on supports edges makes every Gap A gate pass while production
returns zero. Recommend a single assertion in the fixture helper that the
supports edges it writes carry `grounded = 0`.

**Fixture shape for Gap B:** the existing community-parity test scaffolding —
`space_graph_state`, `communities`, `community_members`,
`community_publication_receipts`, `community_parity_input_state` — plus a call to
`reconcile_community_reader_parity` where the gate must open. No new scaffolding
is needed; the two existing consumers already have all of it.

**Cost.** Every test is an in-memory or temp-file libSQL database. These are L4
CI-safe lib tests, selected whenever the planner includes `wenlan-core`.

---

## 7. Open decisions needing a human or lead ruling

Ordered by blast radius. Q4 is the only one that touches M4's gate invariant.

**Q1 — Gap A: derive at read time, or store the binding in a new table?**
*Recommend: derive at read time.* Zero schema, zero migration, zero backfill,
supersession falls out of `page_version_claims` for free (§2.4), and the read
side is already Rust-filtering labels through the D8 normalizer so the loop has a
home. The stored-table alternative buys SQL-joinability and a saved body scan,
at the cost of a migration, a backfill, and the M5 writer having to know about
M6's normalizer. **This does not touch M4's gate.**

**Q2 — Gap B: what is genesis's reconciliation output?**
*Recommend: the partition symmetric difference — reuse the `summary_buckets`
branch.* Low stakes by construction: `output_delta` is recorded, never gating
(`crates/wenlan-core/src/db.rs:15442`–`:15447`, `:2686`). The
admitted-candidate-set alternative is not merely worse, it is unusable — the
legacy arm is structurally empty, so it would report permanent drift. **This does
not weaken the gate; `output_delta` is not one of its terms.**

**Q3 — Gap B: seed a cutover row at `enabled = 0`, or seed nothing?**
*Recommend: seed it, in migration 111.* Absence and `enabled = 0` are
behaviourally identical (`:10976`), so this is not a correctness question — it
is a fence against the next person extending the `enabled = 1` seed loop at
`:11392` by pattern-match. If the reviewer prefers zero migrations, drop it; the
fence survives as test G-B7. **Either choice is fail-closed.**

**Q4 — Gap B: collapse the consumer literal lists into one constant?**
*Recommend: yes, for the three membership sites (gate `matches!`,
`is_known_community_reader`, reconcile loop), leaving the candidate-query branch
and the `enabled = 1` seed loop explicit.* **This is the one item that edits
M4's gate code.** The invariant it must preserve is that an unrecognised
consumer still returns `"0=1"`, and gate G-B3 asserts exactly that. The
counter-argument is real and deserves a decision rather than my assertion:
touching four-times-reviewed code for a refactor is a risk, and five hand-kept
lists that have not yet drifted may be judged fine as they are. My reason for
recommending the change is the failure mode — a consumer present in
`is_known_community_reader` but absent from the gate is permanently dark, and
"dark" produces no error anywhere.

**Q5 — Gap B: the seam for PR-B.**
*Recommend: one M6-owned wrapper module making the single call into `db.rs`.*
Do **not** relocate the gate SQL out of `db.rs`. Needs a ruling only because
"must not reach into `db.rs`" could be read as requiring the relocation, and the
relocation is the expensive reading of an otherwise cheap constraint.

**Q6 — the consumer literal string.** Proposed `"m6_genesis"`. It is stored in
`community_reader_cutover.consumer` and compared as a SQL literal, so it is
effectively permanent from the first install that seeds it. Worth ten seconds of
naming attention now rather than a rename migration later.

**Q7 — out of scope, flagged so it is not lost.** Under Q1's recommendation
there is no stored binding to go stale, but a genesis *candidate* whose referring
page is later re-derived can still have a stale `genesis_candidate_roots`
fingerprint. That is refresh-readiness territory
(`crates/wenlan-core/src/m6/refresh_readiness.rs`, migration 109) and belongs to
PR-C, not to this follow-up. Naming it here so nobody discovers it as a surprise.

### 7.1 Lead rulings (2026-08-03)

Recommendations accepted except Q4 and Q6. Spec-level calls; the independent
review at the integrated boundary still owns the final word.

- **Q1 → derive at read time. ACCEPTED, and it overturns an earlier ruling.**
  Gap A needs no schema, no migration, no backfill. Consequence: the
  orphan-wikilink signal is buildable in PR-B after all. The earlier call to drop
  it rested on "the binding does not exist"; the binding does not exist *stored*,
  but every input to it does, and §2.1's containment algorithm derives it from
  `page_version_claims` + `claim_anchors` + `pages.content` using the span-digest
  verification the derivation aligner already performs. PR-B claims contract
  item 1 with all four signals. Both `LIKE`-based shortcuts stay rejected for the
  reasons in §2.2.
- **Q2 → reuse the `summary_buckets` branch. ACCEPTED.** `output_delta` is
  recorded and never gating, and the admitted-candidate-set alternative would
  report permanent drift against a structurally empty legacy arm.
- **Q3 → seed the cutover row at `enabled = 0` in migration 111. ACCEPTED.**
  Behaviourally identical to absence, so it costs nothing and fences the next
  person who extends the `enabled = 1` seed loop by pattern-match.
- **Q4 → DECLINED. Do not consolidate the consumer literal lists.** The failure
  mode is real — a consumer known to one list and absent from the gate is
  permanently dark and silent — but the fix does not have to be a refactor of
  code that survived four independent review rounds. Add a test asserting the
  membership lists agree, which catches the same failure with none of the risk.
  We are adding exactly one consumer; a refactor to prevent drift among five
  lists that have not drifted is speculative surface bought with review risk.
- **Q5 → one M6-owned wrapper module. ACCEPTED.** The gate SQL stays in `db.rs`.
  "PR-B must not reach into `db.rs`" was never a demand to relocate it.
- **Q6 → name it `genesis_candidates`, not `m6_genesis`.** The existing
  consumers (`summary_buckets`, `summary_eligibility`) carry no milestone
  prefix and describe what reads. The value is permanent from first seed, and a
  milestone name ages badly — M6 becomes history, the consumer does not.
- **Q7 → agreed, PR-C.** Stale candidate fingerprints are refresh-readiness
  territory; noted, not claimed here.

---

## 8. Size estimate

| Piece | Files | Rough size |
|---|---|---|
| Gap A — offset-carrying wikilink extractor | `sources/obsidian.rs` (+ its `mod tests`) | ~25 lines production, ~40 test |
| Gap A — binding + scope in the orphan-wikilink signal | `m6/signals.rs` | ~90 lines |
| Gap A — gates | `m6/signals_test.rs` | ~250 lines (7 gates + one shared fixture) |
| Gap B — constant, match arms, reconcile loop, output branch | `db.rs` (6 edit sites) | ~30 lines |
| Gap B — migration 111 | `db.rs` | ~25 lines |
| Gap B — M6-owned gate seam | new `m6/community_gate.rs` | ~40 lines |
| Gap B — gates | `db.rs` test module | ~200 lines |
| Consumer-list consolidation (Q4, if ruled yes) | `db.rs` | ~20 lines net |

**Roughly 230 lines of production code and 490 of tests, across 4 files plus one
new module.** Gap A is the larger and more interesting half; Gap B is small,
mechanical, and mostly tests — its cost is review attention on M4's gate, not
volume. Both halves are independently mergeable, and Gap B alone unblocks PR-B2.
