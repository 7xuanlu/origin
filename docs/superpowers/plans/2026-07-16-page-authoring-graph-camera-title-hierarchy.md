# Page authoring, Graph camera, and title hierarchy implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Authoring flow superseded (2026-07-17):** Tasks 1 and 2 shipped. The later
> direct Page draft editor plan replaces Tasks 3 and 4's modal composer,
> required-at-submit fields, and `attached_to` notice flow. Do not implement
> those unchecked modal tasks; use
> `2026-07-16-direct-page-draft-editor.md` as the current contract.

**Goal:** Add safe direct Page authoring to Wiki and Space detail, repair Graph framing, and converge top-level title hierarchy.

**Architecture:** A typed `create_page` Tauri bridge wraps the daemon's existing authored-Page contract. One shared `PageComposerDialog` owns form safety and cache invalidation, while Main owns the one-shot attached-to-existing destination notice. Graph framing is reduced to pure camera/node-value helpers used by both runtime trigger points.

**Tech Stack:** React 19, TypeScript, React Query, Vitest, Tauri 2, Rust, Tailwind v4/CSS, react-force-graph-2d.

## Global constraints

- Preserve all unrelated dirty worktree changes.
- Do not change the daemon schema or add dependencies.
- Space remains optional; a Page is first-class and may have no Space.
- Use RED-first tests and inspect each expected failure before production edits.
- Do not commit, push, publish, or touch production data.

---

### Task 1: Lock title and Graph geometry contracts

**Files:**
- Modify: `src/components/memory/pages/wikiSpaceTypography.test.ts`
- Modify: `src/components/memory/ConstellationMap.test.tsx`
- Modify: `src/reviewTauriCore.test.ts`
- Modify: `e2e/fixtures/spacesNavigation.ts`
- Modify: `src/components/memory/ConstellationMap.tsx`
- Modify: `src/components/memory/navigation/navigation-shell.css`
- Modify: `src/components/memory/spaces/spacesInventory.css`
- Modify: `src/components/memory/HomePage.tsx`

**Interfaces:**
- Produces: `graphCameraPlan(nodes)` with explicit `empty | single | fit` results.
- Produces: `graphNodeValue(node)` returning the custom-painted radius squared while `nodeRelSize` stays `1`.

- [x] Write CSS expectations for the shared `34px/1.12/-0.03em` destination token and Home's `16px` title gap.
- [x] Write zero-, one-, and multi-node camera-plan tests plus radius-squared geometry tests.
- [x] Write a Review invariant that every `list_entities_cmd` result resolves through `get_entity_detail_cmd` and the fixture returns all seven entities.
- [x] Run the three focused suites and confirm expected failures from the old `38.4px`, missing helpers, and one-row fixture.
- [x] Implement the minimal shared title declarations, camera helper, runtime calls, and fixture detail rows.
- [x] Rerun the focused suites to green.

### Task 2: Add the typed authored-Page bridge

**Files:**
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/lib/tauri.ts`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `e2e/tauriMock/runtime.ts`
- Modify: `src/reviewTauriCore.test.ts`

**Interfaces:**
- Produces: `CreatePageInput { title: string; content: string; space: string | null }`.
- Produces: `CreatePageResponse { id: string; attached_to?: string | null; warnings: string[] }`.
- Produces: `createPage(input): Promise<CreatePageResponse>` invoking `create_page`.

- [x] Write a TypeScript wrapper test that expects `create_page` with trimmed Title/Content and nullable Space.
- [x] Write a Rust request-serialization test proving `creation_kind = authored`, empty sources, nullable Space, and null workspace/entity/summary.
- [x] Write a Review runtime create/list/get round-trip test.
- [x] Run TypeScript and Rust focused suites and inspect the missing-command failures.
- [x] Implement the typed TS wrapper, Rust command/registration, and isolated Review mutation.
- [x] Rerun focused suites to green.

### Task 3: Build the safe shared Page composer

**Files:**
- Create: `src/components/memory/pages/PageComposerDialog.tsx`
- Create: `src/components/memory/pages/PageComposerDialog.test.tsx`
- Create: `src/components/memory/pages/pageComposer.css`
- Modify: `src/i18n/resources.ts`
- Modify: `src/i18n/resources.test.ts`

**Interfaces:**
- Consumes: `createPage`, `listSpaces`, and React Query's `QueryClient`.
- Produces: `PageComposerDialog({ open, defaultSpace, onClose, onCreated })`.
- Calls: `onCreated({ result, requestedTitle })` only after a successful mutation.

- [ ] Write dialog tests for field order, defaults, no `Optional` copy, autofocus, disabled submitting state, `Cmd/Ctrl+Enter`, inline error preservation, clean Escape, dirty Escape/Cancel confirmation, and ignored backdrop clicks.
- [ ] Run the dialog suite and inspect the missing-component failure.
- [ ] Implement the minimal accessible dialog and shared CSS using Wenlan tokens.
- [ ] Add English, Simplified Chinese, and Traditional Chinese copy with parity coverage.
- [ ] Rerun dialog and i18n suites to green.

### Task 4: Wire Wiki, Space detail, navigation, and attached feedback

**Files:**
- Modify: `src/components/memory/pages/PagesOverview.test.tsx`
- Modify: `src/components/memory/pages/PagesOverview.tsx`
- Modify: `src/components/memory/space-detail/SpaceDetail.dossier.test.tsx`
- Modify: `src/components/memory/space-detail/SpaceDossierHeader.tsx`
- Modify: `src/components/memory/space-detail/space-detail-header.css`
- Modify: `src/components/memory/SpaceDetail.tsx`
- Modify: `src/components/memory/Main.search.test.tsx`
- Modify: `src/components/memory/Main.tsx`
- Modify: `src/components/memory/PageDetail.test.tsx`
- Modify: `src/components/memory/PageDetail.tsx`

**Interfaces:**
- Consumes: `PageComposerDialog` and `CreatePageResponse`.
- Main owns `attachedPageId: string | null` and routes to `attached_to ?? id`.
- PageDetail consumes `showAttachedPageNotice` and `onDismissAttachedPageNotice`.

- [ ] Write Wiki and Space-detail tests for visible outlined `New page` actions and correct default Space.
- [ ] Write Main routing tests for new ID and `attached_to` precedence.
- [ ] Write PageDetail tests for a dismissible `Added to “<loaded title>”` inline banner.
- [ ] Run focused suites and inspect missing-prop/action/banner failures.
- [ ] Wire both entry points, Main's one-shot notice state, destination routing, and PageDetail banner.
- [ ] Rerun focused suites to green.

### Task 5: Integration and rendered verification

**Files:**
- Update only source/tests necessary to resolve fresh verification failures.
- Save screenshots outside the repository under the active visualization directory.

- [ ] Run focused Vitest suites for all four tasks.
- [ ] Run full `pnpm test`, `pnpm test:i18n`, `pnpm build`, `cargo fmt --check --all`, focused/full Rust tests as practical, and `git diff --check`.
- [ ] Clean-restart the fixture-isolated Review server/app without production data.
- [ ] Capture Home, Graph, Wiki, Wiki composer, Space detail, Space-detail composer, and attached-banner states after the final edit.
- [ ] Exercise the actual Review create flow, Page navigation, optional-Space clearing, zero/one/many Graph framing, and console-health checks.
- [ ] Run inline Visual QA Pass A and Pass B against every fresh capture; loop until both return PASS or report the exact residual blocker.
