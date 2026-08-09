# Space Icon Planet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Map-based Space navigation mark with the approved official Tabler Planet while preserving Wenlan's existing navigation geometry, states, and accessibility.

**Architecture:** Keep the change inside the existing `SpaceMark` primitive. Lock the exact two-path Planet geometry in the focused component test and browser E2E, then update the ignored local `DESIGN.md` contract and rendered QA evidence. No new icon package or broader navigation refactor is needed.

**Tech Stack:** React 19, TypeScript 5.7, Vitest 4, Testing Library, Playwright, SVG on Tabler's 24×24 grid.

## Global Constraints

- Use the exact official Tabler Planet paths approved in the runtime preview.
- Keep `14 × 14`, `2px` round stroke, no fill, no animation, and existing Wenlan color tokens.
- Preserve parent-owned accessible name, focus behavior, and `aria-current` state.
- Do not modify other Spaces layout, copy, dependencies, git history, or publishing state.

---

### Task 1: Lock Planet Geometry and Implement SpaceMark

**Files:**
- Modify: `src/components/memory/navigation/PrimaryNavigation.test.tsx`
- Modify: `src/components/memory/navigation/SpaceMark.tsx`
- Modify: `DESIGN.md`

**Interfaces:**
- Consumes: `SpaceMark({ active }: { readonly active: boolean })` and PrimaryNavigation's existing icon slot.
- Produces: `data-space-mark="self-contained-world"` with the exact official Planet paths and unchanged state styling.

- [x] **Step 1: Write the failing component contract**

Require these exact paths and reject the old Map geometry:

```tsx
expect(paths).toEqual([
  "M18.816 13.58c2.292 2.138 3.546 4 3.092 4.9c-.745 1.46 -5.783 -.259 -11.255 -3.838c-5.47 -3.579 -9.304 -7.664 -8.56 -9.123c.464 -.91 2.926 -.444 5.803 .805",
  "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0",
]);
```

- [x] **Step 2: Run RED**

Run: `pnpm vitest run src/components/memory/navigation/PrimaryNavigation.test.tsx`

Expected: the new `self-contained-world` selector and Planet paths fail against the current Map implementation.

- [x] **Step 3: Implement the minimal Planet primitive**

Replace the three Map paths with the two Planet paths, change only the semantic data attribute, and retain the existing SVG footprint, tokens, and accessibility attributes.

- [x] **Step 4: Update the design contract**

Record `A Space is a self-contained world for one body of work`, the exact Planet geometry, the approved preview evidence, and the no-stars/no-fill/no-motion constraints in `DESIGN.md`.

- [x] **Step 5: Run GREEN**

Run: `pnpm vitest run src/components/memory/navigation/PrimaryNavigation.test.tsx`

Expected: `5 passed`.

### Task 2: Update Browser Contract and Verify the Rendered UI

**Files:**
- Modify: `e2e/space-mark.visual.spec.ts`
- Modify: `design-qa.md`
- Regenerate: `.omo/evidence/task-7-spaces-navigation-redesign/space-mark/*`

**Interfaces:**
- Consumes: the implemented `self-contained-world` mark.
- Produces: fresh light, dark, mobile overlay, keyboard-focus, DPR2, and zh-Hant evidence with zero browser errors.

- [x] **Step 1: Update the E2E selector and exact geometry assertions**

Use `[data-space-mark='self-contained-world']`, require two paths, and assert the first path equals the official orbit path.

- [x] **Step 2: Run focused browser verification**

Run: `pnpm exec playwright test e2e/space-mark.visual.spec.ts`

Expected: `1 passed` with fresh screenshots and `space-mark-qa.json` recording two Planet paths.

- [x] **Step 3: Run build and broader tests**

Run: `pnpm build`

Run: `pnpm vitest run --testTimeout=10000`

Expected: build exit `0`; all Vitest files pass with the known single skip.

- [x] **Step 4: Perform two-pass visual QA**

Inspect every fresh screenshot, compare the implementation with the approved Planet preview, and update `design-qa.md` with separate design-system and visual-fidelity verdicts.

- [x] **Step 5: Confirm scope hygiene**

Run: `git diff --check` for tracked edits and equivalent no-index checks for untracked implementation files. Confirm no temporary preview spec remains and no dependency file changed for this icon replacement.

## Self-review

- Spec coverage: exact geometry, semantics, states, accessibility, responsive render, and no-dependency constraints are covered.
- Placeholder scan: no TBD, TODO, or deferred implementation step remains.
- Type consistency: `SpaceMarkProps`, `active`, and the existing PrimaryNavigation interface remain unchanged.
