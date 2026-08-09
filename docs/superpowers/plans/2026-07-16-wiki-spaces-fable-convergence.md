# Wiki and Spaces Fable convergence implementation plan

> Execute in this worktree with RED-first tests. Preserve unrelated dirty changes. Do not commit, push, publish, change backend/schema, or touch production data.

**Goal:** Bring the current Wiki, Spaces, sidebar, Home heading, and native titlebar into the approved Wenlan hierarchy while representing all three Page review states honestly.

**Architecture:** Persisted Pages continue to come from `list_pages`. Their existing wire `review_status` becomes a quiet inventory state and filter. `page_keep_or_archive` stays a read-only refinement signal on its existing Page. Unpersisted `page_candidate` discovery is shared from the explicit Review run through React Query session cache; opening Wiki never invokes `distill_review` and therefore never performs a hidden write. Existing Space suggestion verbs remain unchanged.

**Stack:** React 19, TypeScript, React Query, Vitest, Playwright, CSS/Tailwind v4, Tauri 2.

## Task 1 — Lock the Page-state contract RED

- Extend `src/lib/tauri.ts` with backward-compatible `creation_kind` and `review_status` Page metadata.
- Add focused tests in `src/components/memory/pages/PagesOverview.test.tsx` for:
  - no redundant `All pages` heading;
  - an `unconfirmed` row marker and filter, without calling it a candidate;
  - a quiet `page_keep_or_archive` attention marker;
  - cached `page_candidate` preview above inventory, linked candidates opening their Page and unlinked candidates requesting Review preview;
  - no `distill_review` invocation on Wiki mount;
  - whole-row Page routing while Space remains the sole secondary row destination.
- Add a Review-panel test proving a successful explicit distill stores the result under one exported session query key.

## Task 2 — Implement Wiki review states

- Extract candidate mapping/presentation helpers into `src/components/memory/pages/pageReviewSignals.ts`.
- In `src/components/memory/DistillReviewPanel.tsx`, cache successful `DistillReviewResponse` without changing its current explicit Review trigger or verbs.
- In `src/components/memory/pages/PagesOverview.tsx`, read only the cached discovery result, fetch only read-only refinement proposals, render the quiet candidate preview when non-empty, and add the `unconfirmed` inventory filter/markers.
- Add localized English, Simplified Chinese, and Traditional Chinese state/filter/action copy to `src/i18n/resources.ts`; retain the existing Review candidate wording where it already fits.
- Remove the redundant inventory heading while retaining a quiet count near the filters.

## Task 3 — Lock and implement sidebar/Home/titlebar RED→GREEN

- Reverse the existing Spaces badge test: primary navigation never exposes per-section suggestion counts.
- Add navigation-style contracts for a full-row quiet indigo selected wash, outer 2px rail, 8px rail-to-icon space, active weight 500, and a distinct focus ring.
- Add visited-history tests and implementation for `src/lib/recentPages.ts` plus `src/components/memory/RecentPages.tsx`.
- Render Recent Pages then Recent Spaces after Sources on every standard main view, at most four actual visits each; omit empty groups and mark the current destination with `aria-current`.
- Change `Today in Wenlan` to 22px serif 500 and lock it in `HomePage.redesign.test.tsx`.
- Give the sidebar toggle a stable geometry hook. Calibrate its vertical position only from a fresh native capture; keep the macOS traffic-light config unchanged unless rendered evidence requires it.

## Task 4 — Lock and implement Spaces inventory RED→GREEN

- Remove the redundant `All spaces` heading while keeping Suggested as the only conditional section heading.
- Keep Suggested Space actions at 12px with 4px 10px padding; Keep is tonal only, Discard quiet.
- Make New Space a quiet outlined action.
- Remove unlabeled memory counts from Recent Spaces.
- Introduce one shared locale-aware date formatter for Page and Space inventory rows.
- Update focused Spaces/Recent tests before production CSS/components.

## Task 5 — Lock and implement Space detail RED→GREEN

- Set metric numerals to 15px mono/tabular and labels to 11px so Page content remains primary.
- Use the same compact suggestion action treatment as the overview.
- Merge separate name/description edit entry points into one edit mode and place Delete inside the overflow menu.
- Preserve title scale `clamp(24px, 1.6vw + 14px, 30px)`, current Page/Entity links, icons, and IA.

## Task 6 — Integration verification

- Run focused Wiki, navigation, recent-history, Spaces, Space-detail, Home, i18n, and Review-flavor Vitest suites.
- Run full `pnpm test`, `pnpm test:i18n`, `pnpm build`, Review web/native build, Rust formatting/tests, and `git diff --check`.
- Launch only the fixture-isolated `Wenlan Review.app`; capture 1280x760 Wiki, Spaces, Space detail, Home/sidebar and 393px compact states without foregrounding the user's workspace.
- Inspect actual traffic-light/toggle alignment, responsive wrapping, CJK copy, focus/current semantics, and the three Page states.
- Ask Fable for a fresh-eye final design verdict from the new captures, apply critical findings, then run independent design and code reviews.
