# Sources Intake Tray Icon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the Sources primary-navigation mark with the approved shallow intake tray receiving three short vertical strokes.

**Architecture:** Keep the mark inline in `PrimaryNavigation`, matching the other single-use 14 px navigation SVGs. Lock the new Sources geometry and the unchanged Home Page layered glyph with focused DOM assertions, then verify the actual sidebar render.

**Tech Stack:** React 19, TypeScript, inline SVG, Vitest, Testing Library, Vite preview.

## Global Constraints

- Preserve the Home Page layered glyph byte-for-byte: `M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5`.
- Change only the Sources navigation icon; do not alter Graph, Memories, Space, Wiki, Home, folders, databases, or integrations.
- Sources must be a noun-like shallow tray with three short vertical input strokes and no arrowheads.
- Keep the SVG decorative; the enclosing Sources button remains the accessible navigation control.
- Do not commit from this detached, already-dirty worktree.

---

### Task 1: Sources intake-tray mark

**Files:**
- Modify: `src/components/memory/navigation/PrimaryNavigation.test.tsx`
- Modify: `src/components/memory/navigation/PrimaryNavigation.tsx`
- Modify: `src/components/memory/HomePage.redesign.test.tsx`

**Interfaces:**
- Consumes: `PrimaryNavigation` and its existing `NavButton` Sources destination.
- Produces: an inline SVG marked `data-navigation-icon="sources-intake-tray"`, with input path `M7 4v6M12 4v6M17 4v6` and tray path `M5 13l1.5 5h11l1.5-5`.

- [x] **Step 1: Write the failing Sources geometry test**

Add a focused test that renders `PrimaryNavigation`, selects the Sources button, and expects:

```tsx
const mark = screen
  .getByRole("button", { name: "Sources" })
  .querySelector<SVGSVGElement>('[data-navigation-icon="sources-intake-tray"]');

expect(mark).not.toBeNull();
expect(Array.from(mark?.querySelectorAll("path") ?? [], (path) => path.getAttribute("d"))).toEqual([
  "M7 4v6M12 4v6M17 4v6",
  "M5 13l1.5 5h11l1.5-5",
]);
expect(mark?.querySelector("polygon, polyline")).toBeNull();
expect(mark).toHaveAttribute("aria-hidden", "true");
expect(mark).toHaveAttribute("height", "14");
expect(mark).toHaveAttribute("width", "14");
```

Remove Sources from `preservedIconGeometry` because it is the one intentionally changed mark.

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
pnpm vitest run src/components/memory/navigation/PrimaryNavigation.test.tsx
```

Expected: FAIL because the current Sources SVG lacks `data-navigation-icon="sources-intake-tray"` and still contains layered geometry.

- [x] **Step 3: Implement the minimal inline SVG**

Replace only the existing Sources `icon` prop with:

```tsx
<svg
  aria-hidden="true"
  data-navigation-icon="sources-intake-tray"
  height="14"
  style={iconStyle}
  viewBox="0 0 24 24"
  width="14"
>
  <path
    d="M7 4v6M12 4v6M17 4v6"
    fill="none"
    stroke="currentColor"
    strokeLinecap="round"
    strokeWidth="1.8"
  />
  <path
    d="M5 13l1.5 5h11l1.5-5"
    fill="none"
    stroke="currentColor"
    strokeLinecap="round"
    strokeLinejoin="round"
    strokeWidth="1.8"
  />
</svg>
```

- [x] **Step 4: Lock the preserved Home Page layered glyph**

In the existing Home Page test with populated pages, inspect the first SVG under `wiki-page-list` and assert its path remains exactly:

```tsx
expect(screen.getByTestId("wiki-page-list").querySelector("svg path")).toHaveAttribute(
  "d",
  "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
);
```

- [x] **Step 5: Verify GREEN and the frontend floor**

Run:

```bash
pnpm vitest run src/components/memory/navigation/PrimaryNavigation.test.tsx src/components/memory/HomePage.redesign.test.tsx
pnpm exec tsc -b
pnpm build
```

Expected: all focused tests pass, TypeScript exits 0, and Vite production build exits 0.

- [x] **Step 6: Render and inspect the sidebar**

Launch the repository's isolated preview fixture, capture the sidebar at the existing desktop viewport, and confirm:

- Sources reads as three downward input strokes entering a shallow tray.
- It has no arrowhead and does not resemble Import.
- Home Page retains the layered Page glyph.
- Sources remains distinct from Graph, Memories, and Space at 14 px.

Save the screenshot under `.omo/evidence/sources-intake-tray-20260715/` and stop the preview server after inspection.
