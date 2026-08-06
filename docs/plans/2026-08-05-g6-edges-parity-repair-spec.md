# G6 edges-parity repair — doc-source-page drift (77 missing)

Status: fix spec, ships as its own PR before the Stage 2 spec. Root-cause
investigation: Opus lane, 2026-08-05, reproduced the live watermark
byte-for-byte (expected 2464 / actual 2387 / missing 77 / extra 0 / corrupt 0)
and confirmed all 77 missing edges belong to one page,
`src_4e030cfe8f424bd4` (the doc-enrich source page for
`~/.wenlan/sources/2026-07-23-mcp-surface-consolidation-plan.md`).
Two independent defects fired in the same `replace_source_page_inner`
transaction at ts 1785954012. Blocks Stage 2: the parity oracle must read
drift 0 before the parity machinery retires.

## Root cause (confirmed)

**Defect 1 — 29 edges, over-retire of carried-over evidence.**
`replace_source_page_inner` (`crates/wenlan-core/src/db.rs:45024`): the
`removed` snapshot (db.rs:45162-45206) collects every `(dst_kind, locator)`
the page's current `page_sources` + `page_evidence` imply — external kinds
included — but the subtraction (db.rs:45207-45209) removes only the
*memory*-kind re-asserted sids. An external-kind locator present in BOTH the
old and new evidence sets is never subtracted. The retire loop runs AFTER
`insert_resolved_page_evidence` re-asserts the new edges, and
`dual_write_invalidate_edge` (db.rs:15147) has no refcount guard — so the 29
carried-over chunks were re-asserted and immediately killed in the same
transaction. Introduced by commit `1658e4dc` (PR #482); the intent was right,
the subtraction is incomplete.

**Defect 2 — 48 edges, kind-derivation disagreement.** The sweep's
`page_sources` contributor hard-codes `dst_kind="memory"` (db.rs:19268-19270),
mirroring the one-time backfill (`backfill_edges_from_page_sources`,
db.rs:20675). The only live writer minting cites edges for a page's sources,
`insert_resolved_page_evidence` (db.rs:44472), RESOLVES the kind via
`resolve_page_evidence_source_kind` (`citations.rs:48-70`) — a
`source_agent='folder'` memory resolves to `external_file` →
`dst_kind='external'`. For a doc source page the sweep therefore expects
`cites/page:P/memory:chunk` while the writer minted
`cites/page:P/external:chunk`: a permanently unbacked expectation per row.
Coincides for every normal page (memory both sides), which is why it never
fired elsewhere. Self-heal: none — the ambient sweep is read-only, the
backfills run only from m81/m111, and any re-enrichment of the same document
re-opens the hole.

## Fix targets (three, ruled)

**(a) One kind-resolution rule everywhere.** The sweep's `page_sources`
contributor (db.rs:19253-19284) and `backfill_edges_from_page_sources`
(db.rs:20639-20685) resolve `dst_kind` exactly as
`insert_resolved_page_evidence` does — via the SAME shared helper
(`resolve_page_evidence_source_kind` / `resolve_one_source_kind`), never a
copied match. For every row whose source is a memory the derived id is
unchanged, so this is a no-op on the rest of the corpus (simulation:
expected 2464 → 2417, page_sources drift half disappears entirely).

*Fold-in RETRACTED (review round 13):* the original ruling folded
`replace_page_sources`' retire site into (a). The scoped review proved that
wrong: that site's `page_evidence` prune is memory-kind-only, so a pruned
folder-doc sid's evidence row SURVIVES the prune — the sweep's evidence
contributor therefore still *expects* the edge, and resolving the kind at the
retire makes the site kill a still-backed edge (the same over-retire class as
Defect 1). The pre-existing hard-coded `"memory"` retire is a harmless no-op
there (the memory-kind id was never minted), and prune scope must match
retire scope — the pairing `try_update_page_content` keeps. Ruling: the
retire stays hard-coded `"memory"`; the leg test asserts
`reconcile_edges_parity()` drift 0 AFTER the prune (settling check first: on
the fold-in tree that assertion must fail with drift 1, then pass on the
revert). The site's memory-kind-only evidence prune remains genuinely out of
scope — a stale external evidence row keeps its edge legitimately alive.

**(b) Fix the over-retire in `replace_source_page_inner`.** Mirror the
sibling `try_update_page_content` shape (db.rs:46539-46558): retire loop runs
BEFORE the evidence insert, `removed` computed as old-set-minus-new-set with
kinds resolved by the shared rule, and each retire guarded by the refcount
check (`cites_backed_by_page_citations`, db.rs:46540). End-state invariant: a
locator present in both the old and new sets keeps one active edge across the
replace; a locator only in the old set retires. If the sibling shape
genuinely doesn't fit this call path, the order-independent variant
(post-insert re-read of implied set, subtract, then retire) is the sanctioned
fallback — flag the deviation, don't silently pick it.

**(c) m119 — repair migration: reactivate, don't re-mint.** For every active
`page_evidence` row with a non-NULL locator whose derived edge exists with
`valid_until IS NOT NULL AND superseded_by IS NULL`, clear `valid_until`.
Narrow by construction (only revives edges whose legacy backing still
exists), idempotent, logs the reactivated count. `SCHEMA_VERSION` → 119.
Simulation receipt: with (a) + (c), drift 0, zero new edge rows minted.

## Tests

- RED-controlled integration test per the #482 convention: drive the real
  `write_document_source_page` path twice over a folder-doc source page with
  a growing chunk set (the 29→48 shape), then assert
  `reconcile_edges_parity()` drift 0 AND the specific end state: carried-over
  locators still active, dropped locators retired. This catches both defects;
  a RED control on (b) alone would not, because (a) fires independently.
- m119 migration test family per the established migration-test pattern
  (reactivates a qualifying retired edge; leaves a superseded edge alone;
  idempotent on re-run).
- `replace_page_sources` leg (post-retraction shape): prune of a
  folder-doc-backed row leaves the external-kind edge ACTIVE (its surviving
  `page_evidence` row is live backing) and asserts
  `reconcile_edges_parity()` drift 0 after the prune.

## Post-review addenda (round 13)

- F2 (nit): the `cites_backed_by_page_citations` guard in
  `replace_source_page_inner`'s retire loop can never fire on that path (the
  pages UPDATE NULLs `citations` earlier in the same transaction, and the
  guard matches memory-kind only). Kept as a defensive mirror of the sibling
  shape; the comment must say it cannot fire here, not claim protection.
- PR-body operator note: m119 does not touch `edges_parity_watermark`, so the
  watermark stays stale at 77 until one sweep runs post-migration;
  `reader_uses_edges` fail-closes on staleness. Run one sweep after
  migrating, or the cutover stays blocked for a reason that looks like the
  old bug.

## Out of scope

- The 2026-08-04 "32-row repair" forensics (how the earlier 29 were cleared)
  — settled enough: no memory-kind edge for this page exists in any snapshot,
  so the repair didn't mint them; nothing about it changes the fix targets.
- Stage 2 retirement of the parity machinery itself — separate spec, blocked
  on this PR reaching live drift 0.
- Pre-existing, review-round-13 side-find (`accept_page_merge`, db.rs:47458):
  the `page_links` repoint retires the old `links` edge passing
  `new_edge_id.as_deref()`, which is `None` when the source page has a NULL
  space (the replacement mint is computed only inside the
  `if let Some(space)` arm) — a space-less source page retires bare with no
  replacement, reading as missing-drift on the survivor-target `links` edge.
  Outside m119's reach (cites-only). Separate ticket if a sweep ever shows it.
- m119 risk-surface closure (round 13): all three page-lifecycle paths
  verified — `delete_page` (bulk retire then FK-cascade drops evidence),
  `archive_page` (touches neither), `accept_page_merge` (copies evidence,
  adds edges, supersedes where it retires) — none leaves a retired `cites`
  edge with live `page_evidence`. The reverted fold-in would have been the
  first.
