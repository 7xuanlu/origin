// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page as BrowserPage } from "@playwright/test";
import { createSpacesNavigationFixture } from "./fixtures/spacesNavigation";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

const evidenceDirectory = ".omo/evidence/wiki-implementation";

function wikiPages() {
  const defaults = createSpacesNavigationFixture().pages;
  const [topic, entity, decision, recap, ambient, person, policy] = defaults;
  if (!topic || !entity || !decision || !recap || !ambient || !person || !policy) throw new Error("Wiki fixture requires seven base pages");
  return [
    { ...topic, title: "Wenlan product principles", summary: "Guiding principles shaping the future of Wenlan.", space: "Wenlan" },
    { ...entity, title: "Nash Su", summary: "Research partner in distributed cognition.", entity_id: "entity-ada", space: "Research" },
    { ...decision, title: "Why citations stay visible", summary: "Keep citations visible to preserve trust and traceability.", content: "Decision: keep citations visible.", space: "Wenlan" },
    { ...recap, title: "July research recap", summary: "Key findings and open questions from July.", space: "Research" },
    { ...ambient, title: "Ambient memory", summary: "Capturing and surfacing useful context over time.", domain: null, space: null },
    { ...person, title: "Grace Hopper", summary: "Pioneer of programming languages and tooling.", entity_id: "entity-grace", domain: null, space: null },
    { ...policy, title: "Source credibility policy", summary: "Ensure sources are evaluated for credibility and trust.", content: "Decision: evaluate every source.", space: "Wenlan" },
    { ...topic, id: "page-eighth", title: "Memory provenance", summary: "Where durable context comes from.", last_modified: "2026-07-05T12:00:00Z", space: "Research" },
    { ...topic, id: "page-ninth", title: "Local-first architecture", summary: "Private knowledge stays on this device.", last_modified: "2026-07-04T12:00:00Z", space: "Wenlan" },
  ];
}

async function openWiki(page: BrowserPage, locale: "en" | "zh-Hant", theme: "dark" | "light") {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, {
    fixture: { pages: wikiPages() },
    locale,
    localStorage: { "wenlan-theme": theme },
    rawActions: [],
  });
  await page.goto("/");

  if ((page.viewportSize()?.width ?? 0) < 900) {
    await page.getByTitle(locale === "zh-Hant" ? "顯示側邊欄" : "Show sidebar").click();
  }
  const primaryNavigation = page.getByRole("navigation", { name: locale === "zh-Hant" ? "主要導覽" : "Primary navigation" });
  await primaryNavigation.getByRole("button", { name: "Wiki", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  if ((page.viewportSize()?.width ?? 0) < 900) await page.waitForTimeout(250);
  return browserErrors;
}

test("Wiki desktop light matches the approved inventory and its controls work", async ({ page }) => {
  await page.setViewportSize({ width: 1487, height: 1058 });
  const browserErrors = await openWiki(page, "en", "light");

  await expect(page.getByRole("columnheader", { name: "Page" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Kind" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Space" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Updated" })).toBeVisible();
  await expect(page.getByTestId("page-space-page-errors")).toHaveText("Wenlan");
  await expect(page.getByTestId("page-space-page-cjk")).toBeEmpty();
  await expect(page.getByText("Independent", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Optional", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Browse by type", { exact: true })).toHaveCount(0);
  await expect(page.getByText("1–7 of 9", { exact: true })).toBeVisible();

  await page.getByLabel("Kind", { exact: true }).selectOption("entity");
  await expect(page.getByRole("button", { name: "Open Nash Su" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Wenlan product principles" })).toHaveCount(0);
  await page.getByLabel("Kind", { exact: true }).selectOption("all");
  await page.getByLabel("Space", { exact: true }).selectOption("Research");
  await expect(page.getByRole("button", { name: "Open July research recap" })).toBeVisible();
  await page.getByLabel("Space", { exact: true }).selectOption("all");
  await page.getByLabel("Sort", { exact: true }).selectOption("recent");

  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-1487x1058.png`, fullPage: true });

  const typeFilter = page.getByLabel("Kind", { exact: true });
  await typeFilter.focus();
  await expect(typeFilter).toBeFocused();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-filter-focus-1487x1058.png`, fullPage: true });

  await typeFilter.selectOption("entity");
  await expect(page.getByRole("button", { name: "Open Nash Su" })).toBeVisible();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-filtered-entity-1487x1058.png`, fullPage: true });
  await typeFilter.selectOption("all");

  await page.getByRole("button", { name: /^Next/ }).click();
  await expect(page.getByText("8–9 of 9", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Previous" })).toBeEnabled();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-page-2-1487x1058.png`, fullPage: true });
  await page.getByRole("button", { name: "Previous" }).click();

  const firstPageLink = page.getByRole("button", { name: "Open Wenlan product principles" });
  await firstPageLink.hover();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-link-hover-mid-1487x1058.png`, fullPage: true });
  await page.waitForTimeout(175);
  await page.screenshot({ path: `${evidenceDirectory}/wiki-light-link-hover-settled-1487x1058.png`, fullPage: true });
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("Wiki renders in dark mode", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1024 });
  const browserErrors = await openWiki(page, "en", "dark");
  await page.screenshot({ path: `${evidenceDirectory}/wiki-dark-1440x1024.png`, fullPage: true });
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("Wiki keeps Traditional Chinese controls precise at tablet and mobile widths", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  const tabletErrors = await openWiki(page, "zh-Hant", "light");
  await expect(page.getByLabel("類別", { exact: true })).toBeVisible();
  await expect(page.getByLabel("空間", { exact: true })).toBeVisible();
  await expect(page.getByLabel("排序", { exact: true })).toBeVisible();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-zh-hant-768x900.png`, fullPage: true });
  expect(tabletErrors.pageErrors).toEqual([]);
  expect(tabletErrors.consoleErrors).toEqual([]);

  await page.setViewportSize({ width: 375, height: 812 });
  await page.reload();
  await page.getByTitle("顯示側邊欄").click();
  await page.waitForTimeout(250);
  await page.screenshot({ path: `${evidenceDirectory}/wiki-zh-hant-overlay-open-375x812.png`, fullPage: true });
  const navigation = page.getByRole("navigation", { name: "主要導覽" });
  await navigation.getByRole("button", { name: "Wiki", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await page.waitForTimeout(250);

  const mobileSearch = page.getByRole("button", { name: "搜尋", exact: true });
  await mobileSearch.click();
  await expect(mobileSearch).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByPlaceholder("搜尋頁面、記憶、來源...")).toBeFocused();
  await page.screenshot({ path: `${evidenceDirectory}/wiki-zh-hant-search-open-375x812.png`, fullPage: true });
  await page.keyboard.press("Escape");
  await expect(mobileSearch).toHaveAttribute("aria-expanded", "false");
  await page.screenshot({ path: `${evidenceDirectory}/wiki-zh-hant-375x812.png`, fullPage: true });
  expect(tabletErrors.pageErrors).toEqual([]);
  expect(tabletErrors.consoleErrors).toEqual([]);
});

test("Cmd+K event opens the responsive global search instead of focusing a hidden input", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "zh-Hant", rawActions: [] });
  await page.goto("/");

  await page.evaluate(async () => {
    await window.__TAURI_INTERNALS__?.invoke("plugin:event|emit", { event: "toggle-spotlight", payload: null });
  });

  const searchInput = page.getByPlaceholder("搜尋頁面、記憶、來源...");
  await expect(searchInput).toBeVisible();
  await expect(searchInput).toBeFocused();
  const searchShell = searchInput.locator("..");
  await expect(searchShell).toHaveCSS("outline-style", "solid");
  await expect(searchShell).toHaveCSS("outline-width", "2px");
  await expect(searchShell).toHaveCSS("outline-offset", "2px");
  await expect(page.getByRole("button", { name: "搜尋", exact: true })).toHaveAttribute("aria-expanded", "true");
  await page.screenshot({ path: `${evidenceDirectory}/wiki-zh-hant-shortcut-search-375x812.png`, fullPage: true });
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
