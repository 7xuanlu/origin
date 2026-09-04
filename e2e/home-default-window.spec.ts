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
import { createSpacesNavigationFixture } from "./fixtures/spacesNavigation";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

/** app/tauri.conf.json → app.windows[0].{width,height}. */
const DEFAULT_WINDOW = { width: 1280, height: 720 } as const;

async function openHome(page: BrowserPage, pages: readonly unknown[]) {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, {
    fixture: { ...createSpacesNavigationFixture(), pages: pages as never },
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

/** Every element's right edge, against the viewport. */
async function horizontalOverflow(page: BrowserPage) {
  return page.evaluate(() => {
    const doc = document.documentElement;
    const past: string[] = [];
    for (const el of Array.from(document.querySelectorAll("*"))) {
      const box = el.getBoundingClientRect();
      if (box.width === 0 && box.height === 0) continue;
      if (box.right > window.innerWidth + 0.5) {
        const testid = el.getAttribute("data-testid");
        past.push(`${el.tagName.toLowerCase()}${testid ? `[${testid}]` : ""} right=${Math.round(box.right)}`);
      }
    }
    return { documentOverflow: doc.scrollWidth - doc.clientWidth, past };
  });
}

for (const [label, pages] of [
  ["an empty library", [] as readonly unknown[]],
  ["a library with pages", createSpacesNavigationFixture().pages as readonly unknown[]],
] as const) {
  test(`home fits the default window with ${label}`, async ({ page }) => {
    await page.setViewportSize({ ...DEFAULT_WINDOW });
    const browserErrors = await openHome(page, pages);

    const { documentOverflow, past } = await horizontalOverflow(page);
    expect(past, "no element may extend past the right edge of the default window").toEqual([]);
    expect(documentOverflow, "the document must not scroll horizontally").toBe(0);

    // The needs-review rail is the rightmost column, so it is the one that
    // shows a fit problem first.
    const rail = await page.getByTestId("wiki-page-updates").boundingBox();
    expect(rail).not.toBeNull();
    expect(rail!.x + rail!.width).toBeLessThanOrEqual(DEFAULT_WINDOW.width);

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
  await openHome(page, []);

  const grid = await page.getByTestId("wiki-content-grid").boundingBox();
  expect(grid).not.toBeNull();

  const cards = page.locator("[data-ghost-card]");
  await expect(cards).toHaveCount(3);

  const boxes = await cards.evaluateAll((els) =>
    els.map((el) => {
      const b = el.getBoundingClientRect();
      return { left: b.left, right: b.right, width: b.width };
    }),
  );
  for (const [index, box] of boxes.entries()) {
    expect(box.width, `ghost card ${index} must be visible`).toBeGreaterThan(0);
    expect(box.right, `ghost card ${index} must not spill past the window`).toBeLessThanOrEqual(
      DEFAULT_WINDOW.width,
    );
  }
  // All three sit on one row, none of them clipped by a scroll container.
  const scrolled = await page.locator("[data-ghost-card]").first().evaluate((el) => {
    const strip = el.parentElement!;
    return strip.scrollWidth - strip.clientWidth;
  });
  expect(scrolled, "the ghost row must not need horizontal scrolling").toBe(0);
});
