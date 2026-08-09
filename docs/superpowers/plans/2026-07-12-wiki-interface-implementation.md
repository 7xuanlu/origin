# Wiki interface implementation plan

> **Superseded taxonomy (2026-07-17):** The later approved model removes
> title/body heuristics for Topic, Decision, and Recap. Wiki now exposes the
> durable `Kind` facet `Page | Entity`; only `Page.entity_id` specializes a
> Page as Entity. Decisions, recaps, topics, and future schemas remain Page
> content until the backend supplies an explicit kind. The whole row opens the
> Page. Kind is non-interactive metadata, and Space is the sole secondary row
> destination.

**Outcome:** Replace the current Pages overview with the approved full-width Wiki inventory while preserving Pages as the existing internal route and backend model.

**Non-goals:** No daemon/API changes, no public route migration, no dependency changes, no unrelated cleanup, and no commit or publish operation.

## Implementation sequence

1. Lock the approved information architecture in focused navigation, copy, and Wiki overview tests.
2. Keep `view.kind === "pages"` as the internal seam, but present it as Wiki in every user-facing navigation and overview label.
3. Reorder primary navigation to Home, Wiki, Spaces, Graph, Memories, Sources; render Recent Spaces as a separate section after the primary navigation.
4. Replace the Home/Activity segmented control with one Activity action and keep Quick Capture as the adjacent compose action.
5. Build a responsive full-width Wiki table with Type, Space, and Sort controls; paginate locally over the already-loaded active pages.
6. Leave pages without Space context visually blank. Never render Independent, Optional, Unassigned, No Space, or a placeholder dash.
7. Classify the current wire model conservatively: `entity_id` means Entity; explicit decision/recap signals mean Decision/Recap; all other pages are Topic. Keep classification isolated for replacement by a future backend field.
8. Run focused Vitest tests, i18n tests, TypeScript/build checks, then render the preview and complete visual QA against the approved reference.

## Verification floor

- Focused navigation, Wiki overview, Main header/search, and i18n tests pass.
- `pnpm test:i18n`, `pnpm exec tsc -b`, and `pnpm build` pass.
- The rendered desktop Wiki view matches the approved hierarchy and has no right rail, KPI strip, or empty-Space label.
- `design-qa.md` records the reference, render evidence, and final result.
