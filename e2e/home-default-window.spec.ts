// SPDX-License-Identifier: AGPL-3.0-only
//
// The home page has to fit the window the app opens at. That size is not a
// guess: app/tauri.conf.json's main window is 1280x720 with the sidebar
// expanded, so this file measures geometry at exactly that viewport.
//
// Geometry, not pixels, on purpose: the approved snapshots in this suite are
// macOS captures and cannot be regenerated from a Windows or Linux checkout, so
// a screenshot assertion here would be unrunnable for most of the people who
// need it. Rects and scrollWidth are platform-independent.
import { expect, test, type Page as BrowserPage } from "@playwright/test";
import type { Page as KnowledgePage } from "../src/lib/tauri";
import {
  createSpacesNavigationFixture,
  type SpacesNavigationFixture,
} from "./fixtures/spacesNavigation";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

/** app/tauri.conf.json → app.windows[0].{width,height}. */
const DEFAULT_WINDOW = { width: 1280, height: 720 } as const;

/**
 * A library on first run: nothing has been captured, distilled or proposed yet,
 * so the home page renders its empty state and the rail renders "all caught
 * up". Overriding only `pages` on the populated fixture would leave 205
 * memories and a pending review queue behind it, which is a different layout.
 */
function createFirstRunFixture(): SpacesNavigationFixture {
  const populated = createSpacesNavigationFixture();
  return {
    ...populated,
    pages: [] as readonly KnowledgePage[],
    memories: [],
    entities: [],
    entityDetails: [],
    refinements: [],
    distillReview: {
      ...populated.distillReview,
      pending: [],
      stale_pages: [],
      orphan_topics: [],
    },
  };
}

async function openHome(page: BrowserPage, fixture: SpacesNavigationFixture) {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, {
    fixture,
    locale: "en",
    localStorage: { "wenlan-theme": "dark" },
    rawActions: [],
  });
  await page.goto("/");
  await expect(page.getByTestId("wiki-home")).toBeVisible();
  // The rail appears off a ResizeObserver measurement of the container, so the
  // two-column layout lands one frame after the first paint.
  await expect(page.getByTestId("wiki-content-grid")).toBeVisible();
  await page.waitForFunction(() => {
    const grid = document.querySelector('[data-testid="wiki-content-grid"]');
    return !!grid && getComputedStyle(grid).gridTemplateColumns.split(" ").length === 2;
  }, undefined, { timeout: 10_000 });
  return browserErrors;
}

/**
 * Every element's right edge, against the viewport.
 *
 * Content inside a deliberately scrollable box is exempt: a wide table or code
 * block that scrolls within its own container is not a layout defect, and the
 * container itself is still measured on its own pass through this loop.
 */
async function horizontalOverflow(page: BrowserPage) {
  return page.evaluate(() => {
    const doc = document.documentElement;
    const inScrollContainer = (el: Element) => {
      for (let parent = el.parentElement; parent; parent = parent.parentElement) {
        const overflowX = getComputedStyle(parent).overflowX;
        if (overflowX === "auto" || overflowX === "scroll") return true;
      }
      return false;
    };
    const past: string[] = [];
    for (const el of Array.from(document.querySelectorAll("*"))) {
      const box = el.getBoundingClientRect();
      if (box.width === 0 && box.height === 0) continue;
      if (box.right <= window.innerWidth + 0.5) continue;
      if (inScrollContainer(el)) continue;
      const testid = el.getAttribute("data-testid");
      past.push(`${el.tagName.toLowerCase()}${testid ? `[${testid}]` : ""} right=${Math.round(box.right)}`);
    }
    return { documentOverflow: doc.scrollWidth - doc.clientWidth, past };
  });
}

const LIBRARIES: readonly (readonly [string, () => SpacesNavigationFixture])[] = [
  ["a first-run library", createFirstRunFixture],
  ["a library with pages", createSpacesNavigationFixture],
];

for (const [label, makeFixture] of LIBRARIES) {
  test(`home fits the default window with ${label}`, async ({ page }) => {
    await page.setViewportSize({ ...DEFAULT_WINDOW });
    const browserErrors = await openHome(page, makeFixture());

    const { documentOverflow, past } = await horizontalOverflow(page);
    expect(past, "no element may extend past the right edge of the default window").toEqual([]);
    expect(documentOverflow, "the document must not scroll horizontally").toBe(0);

    expect(browserErrors.pageErrors).toEqual([]);
    expect(browserErrors.consoleErrors).toEqual([]);
  });
}

// The ghost row is the empty state's placeholder for pages that do not exist
// yet. It used to be three fixed 280px cards in a scroll strip with no visible
// scrollbar (864px of content), so at this window size the second card was cut
// in half at the column edge and the third was gone.
test("the empty state's ghost cards fit the content column", async ({ page }) => {
  await page.setViewportSize({ ...DEFAULT_WINDOW });
  await openHome(page, createFirstRunFixture());

  const cards = page.locator("[data-ghost-card]");
  await expect(cards).toHaveCount(3);

  const boxes = await cards.evaluateAll((els: readonly Element[]) =>
    els.map((el) => {
      const box = el.getBoundingClientRect();
      return { right: box.right, width: box.width };
    }),
  );
  for (const [index, box] of boxes.entries()) {
    // Comfortably under the ~161px the three tracks resolve to at this
    // viewport, and far above a card collapsed to a sliver.
    expect(box.width, `ghost card ${index} must be a readable placeholder`).toBeGreaterThanOrEqual(96);
    expect(box.right, `ghost card ${index} must not spill past the window`).toBeLessThanOrEqual(
      DEFAULT_WINDOW.width,
    );
  }
  // All three sit on one row, none of them clipped by a scroll container.
  const scrolled = await cards.first().evaluate((el) => {
    const strip = el.parentElement!;
    return strip.scrollWidth - strip.clientWidth;
  });
  expect(scrolled, "the ghost row must not need horizontal scrolling").toBe(0);
});
