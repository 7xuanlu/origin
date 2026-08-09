# Page detail redesign: per-claim citations + reading-page structure

- **Date:** 2026-07-03
- **Status:** approved layout (option A "reading page"); boule-debate reviewed
  (approve-with-changes, all six required changes incorporated); pending user review
- **Visual reference:** design-review artifact (option A mockup with annotations); original inspiration mockup provided by user
- **Backend dependency:** `7xuanlu/wenlan` PR #332 "feat: per-claim verified citations for wiki pages" (commit `2f6ee4bd`, merged 2026-07-03, **unreleased** — on `origin/main` only)

## Goal

Turn the wiki page-detail view into a citation-first reading page: every claim in the
body carries an inline source chip with a hover popover, and everything that is not
reading (sources index, backlinks, revision history) folds into a single collapsed
disclosure. Resolves all findings from the 2026-07-03 UI audit of `PageDetail.tsx`.

## Non-goals

- Toolbar redesign (icon-only buttons, native `confirm()` delete stay as-is).
- Edit-mode markdown preview.
- A new home for Unlinked Mentions (deleted from this view; future maintenance
  surface is a separate effort).
- Shell / typography changes outside the page view.
- TLDR pull-quote heuristic rework.

## Context: what the backend now provides (PR #332)

- `Page.citations: PageCitation[]` where `PageCitation = { occurrence: u32, marker: u32,
  source_kind: "memory" | "external_url" | "external_file" | "authored", locator: string,
  score: f64, status: "verified" | "unverified", scope: "sentence" | "paragraph" }`.
  `occurrence` is the k-th `[N]` instance in body order; `marker` is the N.
- Page body markdown carries literal `[N]` markers appended after cited sentences;
  multi-source claims render as `[1][3]`.
- `PageChangelogEntry` gained `citations_summary: Option<String>`
  (e.g. `"3 verified, 1 unverified, 2 stripped"`).
- No new endpoints. Citation resolution for popovers goes through the existing
  `GET /api/pages/{id}/sources` (`PageSourceWithMemory[]`).

## Blocker found and designed around

`app/Cargo.toml:83` pins `wenlan-types = "0.9.2"` (semver-capped to 0.9.x), and the
Tauri commands deserialize daemon responses into that crate's structs
(`get_page`, `app/src/search.rs:2941`, via `GetPageWire { page: Option<Page> }`).
Serde silently drops unknown fields, so the daemon's new `citations` array would
never reach the frontend — even with a new daemon binary.

**Decision: raw-JSON passthrough.** `get_page` and `get_page_revisions` return
`serde_json::Value` instead of typed structs. The Rust layer does not consume
`Page`/revision fields on these paths; TypeScript types the response. This removes
the silent-field-drop bug class for these endpoints permanently and does not block
on a crates.io publish of a newer `wenlan-types`. Other page endpoints stay typed
(their shapes are unchanged by PR #332).

## Design

### 1. Data layer

- **Rust** (`app/src/search.rs`): `get_page` and `get_page_revisions` switch to
  `serde_json::Value` passthrough. 404 handling for `get_page` (distinguishing
  "not found" from real errors) is preserved.
- **TS** (`src/lib/tauri.ts`): add `PageCitation` interface mirroring the wire type;
  `citations?: PageCitation[]` on `Page`; `citations_summary?: string` on
  `PageChangelogEntry`.
- **Queries:** no new react-query queries. Popovers resolve against the existing
  `["page-sources", pageId]` data. The global `["orphan-page-links", 2]` query and
  the `listOrphanLinks` call are deleted from `PageDetail.tsx`.

### 2. Citation rendering pipeline

Backend counting contract (verified against `origin/main`
`crates/wenlan-core/src/citations.rs`, `process_citation_output`): stored content is
already marker-normalized with out-of-range markers stripped; `occurrence` k is the
k-th `\[(\d+)\]` regex match over the raw stored body, in order, with **no code-fence
or inline-code awareness**. In-range markers therefore correspond 1:1 with
`citations` entries in body order. The frontend mirrors this exactly:

- **Counting** runs the same plain `\[(\d+)\]` regex over the **original**
  `page.content`, *before* the existing title-heading and `## Sources` stripping.
  Matches inside fenced/inline code **do** consume occurrence indices (mirroring the
  backend); they are simply not rewritten for display (a rewritten link inside a code
  span would render as literal `[…](…)` garbage — the raw `[N]` is correct display
  for code).
- **Rewriting**: the k-th countable, non-code match becomes `[k](#citation:k)` —
  opaque link text, *never* the locator, so bracket/paren-bearing locators (URLs,
  paths) cannot break markdown link syntax. Same preprocessing idiom `[[wikilinks]]`
  already use (`PageDetail.tsx:320-329`).
- `ContentRenderer`'s existing `a` component override detects the `#citation:` href
  and renders `CitationChip` inline, pulling display text from the `citations[k]`
  data: mono pill showing the locator plus a superscript occurrence number. Long
  locators (URLs, paths) truncate to domain / basename.
- **Mismatch fallback (conservative):** if the regex match count ≠ `citations.length`
  or any `marker` value disagrees with its matched `N`, render **no chips** and
  display-strip all markers instead — misattributed citations are worse than none.
  The Page info diagnosability line (section 5) reports the state.
- **`citations` empty or absent → display-strip `\[\d+\]`** from the rendered content
  (plus double-space collapse), mirroring the backend's `strip_markers`. This is a
  display-only transform; stored content is never modified. This handles the
  verified backend behavior where a user content edit resets `citations` to `[]`
  **without** stripping markers from content
  (`crates/wenlan-server/src/memory_routes.rs:3370-3372`) — without it, every edited
  page renders permanent `[N]` noise. Known ceiling: legitimate `[N]`-shaped prose
  (e.g. reference-style link labels) is also hidden — the exact regex the backend
  itself applies when stripping; acceptable. A backend fix (strip markers on the
  edit path) is worth filing upstream but is not a blocker.

### 3. Popover (`CitationPopover`)

- Opens on hover (~150 ms delay) and keyboard focus. Chip click opens the source.
  On touch, first tap opens the popover, action buttons navigate.
- Content by `source_kind`:
  - `memory`: source title, "Source memory" kind badge, locator + date line,
    ~200-char snippet, "Open memory →" action — navigates via the existing
    `onMemoryClick` path (the currently-dead `#memory:` anchor wiring at
    `PageDetail.tsx:350-354` becomes live).
  - `external_url`: domain badge, locator, "Open in browser" (Tauri shell open).
  - `external_file`: path shown, no snippet, no action.
  - `authored`: "written directly" note, no action.
- `status: "unverified"` → chip renders dashed and muted; popover carries an explicit
  "unverified" line. Unverified citations are **never hidden** — hiding them would
  misrepresent the page's trustworthiness.
- Locator that does not resolve in page-sources data → popover shows the locator and
  "source not available".
- Popover data still loading → chip renders normally; popover shows a skeleton.
- **Accessibility (required):** the chip is a real `<button>` (focusable, visible
  focus ring). Popover uses `role="tooltip"` linked via `aria-describedby`; opens on
  hover and `focus`, closes on `blur`, mouse-out, and **Escape**. Focus stays on the
  chip (the popover is descriptive; its single action is also reachable by
  activating the chip). Popover flips/shifts to stay inside the viewport — use an
  existing positioning dependency if one is already installed, otherwise minimal
  manual flip logic; do not add a new dependency for this alone.

### 4. Page structure (top → bottom)

1. Header (back, serif title, meta line) — unchanged.
2. Actions toolbar — unchanged.
3. TLDR pull-quote — extraction unchanged, but it runs on marker-processed text:
   markers landing inside the extracted first sentence are display-stripped (the
   pull-quote stays plain text; those citations remain reachable via Page info
   sources). Known ceiling: a first-sentence citation gets no inline chip.
4. Body with citation chips.
5. **Related pages**: outbound links (`get_page_links.outbound`) as cards; resolved
   targets clickable, unresolved targets muted and inert (no more dead links styled
   as buttons). Hidden when there are zero outbound links — an empty header here is
   noise, not information.
6. **Page info** disclosure — **always rendered**, collapsed by default, summary line
   with counts ("6 sources · 2 backlinks · 4 revisions"; zero counts are informative
   and replace today's vanishing sections). Expanded groups:
   - **Sources**: compact rows (locator chip, title, date), ordered by first citation
     occurrence, then uncited by recency; unverified tags where applicable. Replaces
     the current full-width Source Memories list.
   - **Backlinks**: inbound links shown by label — raw page-ID strings removed.
   - **Revisions**: existing changelog rows plus a `citations_summary` chip.
7. **Unlinked Mentions: deleted** from this view (today it renders a global list —
   identical on every page, `PageDetail.tsx:361` — under a page-scoped heading).

**Component layout:** new subcomponents in `src/components/memory/page/`:
`CitationChip.tsx`, `CitationPopover.tsx`, `RelatedPages.tsx`, `PageInfo.tsx`.
`PageDetail.tsx` remains the orchestrator (queries, mutations, edit mode) and sheds
roughly 300 lines. `EvidenceCard` is absorbed into `PageInfo`'s source rows.

### 5. Fallback & errors

- Old daemon (response lacks `citations`): no chips; related pages and Page info
  work unchanged. Silent graceful degrade — no version banner on the read path.
  Re-distilling on a new daemon populates citations.
- **Diagnosability line** (muted, one line at the bottom of expanded Page info):
  - citations present → "Citations: N (M unverified)";
  - markers present but citations empty (display-stripped, e.g. after a user edit) →
    "Citations cleared by edit — re-distill to restore";
  - count/marker mismatch fallback triggered → "Citation data mismatched —
    re-distill to repair";
  - no citations and no markers → line omitted.
  Wire limitation (accepted): `citations` uses `skip_serializing_if empty`, so
  "old daemon" and "processed, none found" are indistinguishable — the line reflects
  what is knowable client-side.
- `get_page` failure modes unchanged (404 → "Page not found" state, real errors
  surface).

### 6. Testing

- **Vitest + Testing Library:**
  - Marker-mapping util: occurrence ordering, `[1][3]` runs, fenced/inline-code
    matches counted but not rewritten, count/marker mismatch → strip-all fallback,
    empty-citations → display-strip (edited-page noise case), whitespace collapse
    parity with backend `strip_markers`.
  - `CitationChip` / `CitationPopover`: verified / unverified / unresolved states,
    each `source_kind` variant; keyboard focus opens, Escape closes, focus ring
    visible.
  - `PageDetail` integration: chips render from a fixture page with citations;
    Page info disclosure toggles; diagnosability line per state; orphan-links query
    no longer issued; empty-citations fallback strips markers and renders no chips.
- **Manual:** `git -C ../wenlan pull` (PR #332 is on `origin/main`), `pnpm dev:all`,
  verify chips + popovers on a re-distilled page. No e2e harness exists in this repo —
  known limitation.
- **CI:** `cargo fmt --check`, `clippy -D warnings`, `cargo test`, `tsc -b`, `pnpm test`
  all must pass (enforced by `ci.yml`).

## Dependencies and release gate

- **Dev:** sibling `../wenlan` checkout must be at `origin/main` (≥ commit `2f6ee4bd`)
  for the daemon sidecar to serve citations.
- **Release:** `release.yml` resolves the backend at its **latest GitHub release
  tag**. Shipping this feature requires the backend to tag a release containing
  PR #332 first. Until then the app degrades gracefully (no chips).

## Verified backend facts (were open questions; resolved 2026-07-03 against `origin/main`)

- `PageCitation.locator` **is** the memory `source_id`
  (`crates/wenlan-core/src/citations.rs:386-390`: `locator: m.source_id.clone()`
  from `get_memories_by_source_ids`) — same namespace as
  `PageSource.memory_source_id`. Popover resolution via page-sources is sound; the
  "source not available" fallback remains as belt-and-braces.
- Occurrence counting is a plain `\[(\d+)\]` regex over the raw stored body with no
  code-fence awareness (`process_citation_output`); stored content is
  pre-normalized with out-of-range markers stripped, so in-range markers ↔
  `citations` entries are 1:1 in body order.
- The user-edit path resets `citations` to `[]` but stores content **with markers
  intact** (`crates/wenlan-server/src/memory_routes.rs:3370-3372`) — the reason the
  empty-citations display-strip rule in section 2 exists.

## Alternatives considered

- **B — content + inspector rail** (two-column, provenance always visible): rejected —
  diverges from the reference mockup, largest layout diff, needs a narrow-window
  collapse breakpoint. Can be layered on later if reading mode proves insufficient.
- **C — citations only, sections untouched**: rejected — ships every audit finding
  unfixed (global Unlinked Mentions, raw UUIDs, four look-alike always-open sections).
- **Bumping `wenlan-types`** (crates.io) instead of JSON passthrough: rejected —
  blocks on an unreleased crate publish and re-couples the app to backend release
  cadence.
- **Git dependency / `[patch.crates-io]`** on `wenlan-types` at commit `2f6ee4bd`:
  viable today (no publish needed) — rejected because it couples every app build to
  the backend repo's availability and pins a rev that needs manual churn on each
  backend change, while these endpoints' fields have **no Rust consumer**; typing
  them buys drift exposure without a beneficiary.

Scope honesty on the passthrough: it removes the silent-field-drop class **for
`get_page` and `get_page_revisions` only**. `get_page_sources` stays on the typed
0.9.2 path and remains exposed to additive-field drops — accepted because its shape
is unchanged by PR #332 and a dropped future field there degrades to the popover's
"source not available" state, not silent misrendering.

## Design review record (boule debate, 2026-07-03)

Adversarial 3-lab council (Claude main-loop, Codex gpt-5.5, Gemini 3.1 Pro):
initial tally unanimous approve-with-changes; stake-free judgment
**approve-with-changes, medium confidence, position-stable** across both
counterbalanced orderings. Six required changes, all incorporated above:

1. Empty-citations display-strip (fixes the verified edited-page marker-noise
   defect) — section 2.
2. TLDR marker handling specified — section 4, item 3.
3. Occurrence-counting parity verified against PR #332's code; fence handling
   corrected (count everything, rewrite only non-code) + conservative mismatch
   fallback — section 2 and "Verified backend facts".
4. Popover accessibility (Escape, aria, focus, viewport collision) + opaque link
   text eliminating locator-escaping bugs — sections 2 and 3.
5. Alternatives section repaired (`[patch.crates-io]` acknowledged) and passthrough
   scope claim softened (`get_page_sources` stays typed) — "Alternatives considered".
6. Diagnosability line distinguishing knowable citation states — section 5.

## Audit findings → resolution map

| Audit finding | Resolution |
|---|---|
| Unlinked Mentions shows a global list under a page-scoped heading | Section deleted; global query removed |
| Inbound links expose raw page IDs | Backlinks show labels only |
| Unresolved outbound links styled identically to clickable ones | Muted + inert in Related pages |
| Sources decoupled from the claims they support | Inline chips + popovers |
| Three visually identical always-open sections | One collapsed Page info disclosure |
| Dead `#memory:` anchor wiring | Becomes the popover's "Open memory" path |
| No empty states (sections vanish) | Page info always renders with zero-count summaries; Related pages hides only when genuinely empty |
