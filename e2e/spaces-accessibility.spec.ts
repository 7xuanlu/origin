// SPDX-License-Identifier: AGPL-3.0-only
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { getSpaceEntityButton, openSpaceEntity } from "./helpers/spaceEntity";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";
import { renderedContrast, type ContrastResult } from "./helpers/renderedContrast";

const evidenceDir = path.join(
  process.cwd(),
  ".omo/evidence/task-7-spaces-navigation-redesign/accessibility",
);
const screenshotEvidenceDir = path.join(
  process.cwd(),
  ".omo/evidence/task-7-spaces-navigation-redesign/screenshots",
);

type PhysicalTextMetric = {
  readonly cssBox: { readonly height: number; readonly width: number; readonly x: number; readonly y: number };
  readonly cssFontSize: number;
  readonly devicePixelRatio: number;
  readonly physicalBox: { readonly height: number; readonly width: number };
  readonly physicalFontSize: number;
};

async function physicalTextMetric(locator: Locator): Promise<PhysicalTextMetric> {
  return locator.evaluate((node) => {
    const box = node.getBoundingClientRect();
    const cssFontSize = Number.parseFloat(getComputedStyle(node).fontSize);
    const ratio = devicePixelRatio;
    return {
      cssBox: { height: box.height, width: box.width, x: box.x, y: box.y },
      cssFontSize,
      devicePixelRatio: ratio,
      physicalBox: { height: box.height * ratio, width: box.width * ratio },
      physicalFontSize: cssFontSize * ratio,
    };
  });
}

async function pngDimensions(filePath: string): Promise<{ readonly height: number; readonly width: number }> {
  const bytes = await readFile(filePath);
  if (bytes.toString("ascii", 1, 4) !== "PNG") throw new Error(`Not a PNG: ${filePath}`);
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

async function layoutMetrics(page: Page) {
  return page.evaluate(() => ({
    clientHeight: document.documentElement.clientHeight,
    clientWidth: document.documentElement.clientWidth,
    devicePixelRatio,
    innerHeight,
    innerWidth,
    visualViewport: window.visualViewport
      ? { height: window.visualViewport.height, scale: window.visualViewport.scale, width: window.visualViewport.width }
      : null,
  }));
}

async function settleZoomLayout(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    for (const animation of document.getAnimations()) animation.finish();
    const main = document.querySelector("main");
    if (main) {
      main.scrollLeft = 0;
      main.scrollTop = 0;
    }
  });
}

async function assertNoPageOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    main: (document.querySelector("main")?.scrollWidth ?? 0) - (document.querySelector("main")?.clientWidth ?? 0),
  }));
  expect(overflow.document).toBeLessThanOrEqual(1);
  expect(overflow.main).toBeLessThanOrEqual(1);
}

async function assertNotClipped(locator: Locator): Promise<void> {
  await locator.scrollIntoViewIfNeeded();
  const metrics = await locator.evaluate((node) => {
    const box = node.getBoundingClientRect();
    const element = node as HTMLElement;
    return {
      bottom: box.bottom,
      clientHeight: element.clientHeight,
      clientWidth: element.clientWidth,
      left: box.left,
      right: box.right,
      scrollHeight: element.scrollHeight,
      scrollWidth: element.scrollWidth,
      top: box.top,
      viewportHeight: innerHeight,
      viewportWidth: innerWidth,
    };
  });
  expect(metrics.left).toBeGreaterThanOrEqual(-1);
  expect(metrics.right).toBeLessThanOrEqual(metrics.viewportWidth + 1);
  expect(metrics.top).toBeGreaterThanOrEqual(-1);
  expect(metrics.bottom).toBeLessThanOrEqual(metrics.viewportHeight + 1);
  expect(metrics.scrollWidth - metrics.clientWidth).toBeLessThanOrEqual(2);
  expect(metrics.scrollHeight - metrics.clientHeight).toBeLessThanOrEqual(2);
}

async function openSidebar(page: Page, title = "Show sidebar"): Promise<void> {
  const aside = page.locator("aside.memory-sidebar");
  const overlay = await page.evaluate(() => window.matchMedia("(max-width: 899px)").matches);
  await expect(aside).toHaveCSS("position", overlay ? "fixed" : "relative");
  if (await aside.getAttribute("aria-hidden") === "true") {
    const toggle = page.getByTitle(title);
    await expect(toggle).toBeVisible();
    await toggle.click();
  }
  await expect(aside).toHaveAttribute("aria-hidden", "false");
}

async function openSpaces(page: Page, labels = { navigation: "Primary navigation", spaces: "Spaces", show: "Show sidebar" }): Promise<void> {
  await openSidebar(page, labels.show);
  await page.getByRole("navigation", { name: labels.navigation }).getByRole("button", { name: labels.spaces, exact: true }).click();
  await expect(page.locator(".spaces-overview")).toBeVisible();
}

function spaceOverviewButton(page: Page, name = "Wenlan"): Locator {
  return page.locator(".spaces-overview").getByRole("button", { name, exact: true });
}

async function assertFocusOutline(page: Page): Promise<void> {
  const outline = await page.evaluate(() => {
    const style = getComputedStyle(document.activeElement as Element);
    return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
  });
  expect(outline.style).not.toBe("none");
  expect(outline.width).toBeGreaterThanOrEqual(2);
}

async function focusedElementEvidence(page: Page, step: string) {
  const evidence = await page.evaluate((label) => {
    const node = document.activeElement as HTMLElement | null;
    if (!node) throw new Error(`No active element for ${label}`);
    const style = getComputedStyle(node);
    const labelledBy = node.getAttribute("aria-labelledby")
      ?.split(/\s+/)
      .map((id) => document.getElementById(id)?.textContent?.trim() ?? "")
      .filter(Boolean)
      .join(" ");
    const formLabels = node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement || node instanceof HTMLSelectElement
      ? Array.from(node.labels ?? []).map((item) => item.textContent?.trim() ?? "").filter(Boolean).join(" ")
      : "";
    const exposedText = (current: Node): string => {
      if (current.nodeType === Node.TEXT_NODE) return current.textContent ?? "";
      if (!(current instanceof Element) || current.getAttribute("aria-hidden") === "true") return "";
      return Array.from(current.childNodes).map(exposedText).join(" ");
    };
    const name = node.getAttribute("aria-label")
      ?? (labelledBy || undefined)
      ?? (formLabels || undefined)
      ?? node.getAttribute("title")
      ?? node.getAttribute("placeholder")
      ?? exposedText(node).replace(/\s+/g, " ").trim();
    return {
      accessibleName: name,
      outlineStyle: style.outlineStyle,
      outlineWidth: Number.parseFloat(style.outlineWidth),
      step: label,
      tagName: node.tagName.toLowerCase(),
    };
  }, step);
  expect(evidence.accessibleName, step).not.toBe("");
  expect(evidence.outlineStyle, step).not.toBe("none");
  expect(evidence.outlineWidth, step).toBeGreaterThanOrEqual(2);
  return evidence;
}

async function tabTo(page: Page, target: Locator, maximumSteps = 120): Promise<void> {
  for (let step = 0; step < maximumSteps && !(await target.evaluate((node) => node === document.activeElement)); step += 1) {
    await page.keyboard.press("Tab");
  }
  await expect(target).toBeFocused();
}

test("has no page-level horizontal overflow across all responsive surfaces", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  for (const viewport of [
    { width: 1280, height: 900 },
    { width: 768, height: 900 },
    { width: 375, height: 812 },
  ] as const) {
    await page.setViewportSize(viewport);
    await openSidebar(page);
    await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Home", exact: true }).click();
    await assertNoPageOverflow(page);
    await openSpaces(page);
    await assertNoPageOverflow(page);
    await spaceOverviewButton(page).click();
    await expect(page.locator(".space-dossier-grid")).toBeVisible();
    await assertNoPageOverflow(page);
    await openSpaceEntity(page, "Ada Lovelace");
    await expect(page.locator(".entity-detail-dossier")).toBeVisible();
    await assertNoPageOverflow(page);
  }

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("switches exactly at the management and dossier breakpoints", async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpaces(page);
  const suggested = page.getByTestId("space-row-space-suggested");
  const count = page.getByTestId("space-row-space-wenlan").locator(".spaces-row-count");
  expect((await suggested.evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(2);
  await expect(count).not.toHaveCSS("display", "none");
  await page.setViewportSize({ width: 699, height: 900 });
  expect((await suggested.evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(1);
  await expect(count).toHaveCSS("display", "none");
  const mobileMetadata = page.getByTestId("space-row-space-wenlan").getByTestId("space-mobile-metadata");
  await expect(mobileMetadata).toBeVisible();
  await expect(mobileMetadata.getByText("Pages", { exact: true })).toBeVisible();
  await expect(mobileMetadata.getByText("Memories", { exact: true })).toBeVisible();
  await expect(page.locator("main")).toHaveCSS("padding-left", "32px");
  await page.setViewportSize({ width: 639, height: 900 });
  await expect(page.locator("main")).toHaveCSS("padding-left", "20px");

  await page.setViewportSize({ width: 900, height: 900 });
  await expect(page.locator("main")).toHaveCSS("padding-left", "72px");
  await spaceOverviewButton(page).click();
  const dossierGrid = page.locator(".space-dossier-grid");
  expect((await dossierGrid.evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(2);
  await page.setViewportSize({ width: 899, height: 900 });
  await openSidebar(page);
  await expect(page.getByRole("navigation", { name: "Recent spaces" })).toBeVisible();
  expect((await dossierGrid.evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(1);
  await assertNoPageOverflow(page);
  await page.keyboard.press("Escape");
  await expect(page.locator("aside.memory-sidebar")).toHaveAttribute("aria-hidden", "true");
});

test("meets computed browser contrast on redesigned surfaces in both themes", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  const results: ContrastResult[] = [];

  for (const theme of ["light", "dark"] as const) {
    await page.evaluate((value) => document.documentElement.setAttribute("data-theme", value), theme);
    await openSpaces(page);
    await page.getByRole("button", { name: "Actions for Wenlan" }).click();
    results.push(...await renderedContrast(page, [
      { label: `${theme} spaces metadata`, selector: ".spaces-row-updated", foregroundProperty: "color", minimum: 4.5 },
      { label: `${theme} spaces filter boundary`, selector: ".spaces-filter input", foregroundProperty: "border-top-color", minimum: 3 },
      { label: `${theme} spaces menu boundary`, selector: ".spaces-menu", foregroundProperty: "border-top-color", minimum: 3 },
      { label: `${theme} outlined New Space boundary`, selector: ".spaces-new-action", foregroundProperty: "border-top-color", minimum: 3 },
      { label: `${theme} outlined New Space text`, selector: ".spaces-new-action", foregroundProperty: "color", minimum: 4.5 },
    ]));
    await page.keyboard.press("Escape");

    await spaceOverviewButton(page).click();
    await page.getByRole("button", { name: "Edit space" }).click();
    results.push(...await renderedContrast(page, [
      { label: `${theme} space title editor boundary`, selector: ".space-dossier-title-input", foregroundProperty: "border-bottom-color", minimum: 3 },
      { label: `${theme} description editor boundary`, selector: ".space-dossier-description-editor", foregroundProperty: "border-top-color", minimum: 3 },
      { label: `${theme} space sage action`, selector: ".space-dossier-text-action", foregroundProperty: "color", minimum: 4.5 },
    ]));
    await page.getByRole("button", { name: "Cancel", exact: true }).click();

    await openSpaceEntity(page, "Ada Lovelace");
    await expect(page.locator(".entity-detail-seal")).toBeVisible();
    results.push(...await renderedContrast(page, [
      { label: `${theme} entity seal text`, selector: ".entity-detail-seal", foregroundProperty: "color", minimum: 4.5 },
      { label: `${theme} graph node text`, selector: ".entity-graph-node-name", foregroundProperty: "color", minimum: 4.5 },
      { label: `${theme} relationship text`, selector: ".entity-relation-name", foregroundProperty: "color", minimum: 4.5 },
      { label: `${theme} relationship edge`, selector: ".entity-graph-edges line", backgroundSelector: ".entity-graph", foregroundProperty: "stroke", minimum: 3 },
    ]));
  }

  for (const result of results) expect(result.ratio, result.label).toBeGreaterThanOrEqual(result.minimum);
  await mkdir(evidenceDir, { recursive: true });
  await writeFile(path.join(evidenceDir, "computed-contrast.json"), `${JSON.stringify(results, null, 2)}\n`);
});

test("supports keyboard-only drawer and dossier navigation with visible focus", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await expect(page.locator('[data-primary-navigation-active-marker="true"]')).toHaveCount(1);
  await expect(page.locator('[aria-current="page"] [data-primary-navigation-active-marker="true"]')).toHaveCount(1);
  const toggle = page.getByTitle("Show sidebar");
  await toggle.focus();
  await page.keyboard.press("Enter");
  const aside = page.locator("aside");
  await expect(aside).toHaveAttribute("aria-hidden", "false");
  await expect.poll(() => page.evaluate(() => document.querySelector("aside")?.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(aside).toHaveAttribute("aria-hidden", "true");
  await expect(page.getByTitle("Show sidebar")).toBeFocused();

  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: "Close sidebar" }).click({ position: { x: 360, y: 10 } });
  await expect(aside).toHaveAttribute("aria-hidden", "true");
  await expect(page.getByTitle("Show sidebar")).toBeFocused();

  await page.keyboard.press("Enter");
  const spaces = page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Spaces", exact: true });
  await tabTo(page, spaces, 24);
  await assertFocusOutline(page);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();

  const wenlan = spaceOverviewButton(page);
  await tabTo(page, wenlan);
  await assertFocusOutline(page);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
});

test("reaches management, dossier, graph, ledger, observation, and linked-memory controls by Tab", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpaces(page);

  const filter = page.getByLabel("Filter spaces");
  await tabTo(page, filter);
  await assertFocusOutline(page);
  await page.keyboard.type("wen");
  await expect(spaceOverviewButton(page)).toBeVisible();
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.press("Backspace");

  const menuTrigger = page.getByRole("button", { name: "Actions for Wenlan" });
  await tabTo(page, menuTrigger);
  await page.keyboard.press("Enter");
  const rename = page.getByRole("menuitem", { name: "Rename" });
  await tabTo(page, rename, 8);
  await page.keyboard.press("Escape");
  await expect(menuTrigger).toBeFocused();

  const spaceRow = spaceOverviewButton(page);
  await tabTo(page, spaceRow);
  await page.keyboard.press("Enter");
  await expect(page.locator(".space-dossier")).toBeVisible();
  const archive = page.locator(".space-dossier-archive-trigger");
  await tabTo(page, archive);
  await assertFocusOutline(page);
  await page.keyboard.press("Enter");
  await expect(archive).toHaveAttribute("aria-expanded", "true");
  await page.keyboard.press("Enter");
  await expect(archive).toHaveAttribute("aria-expanded", "false");

  const entity = await getSpaceEntityButton(page, "Ada Lovelace");
  await tabTo(page, entity);
  await page.keyboard.press("Enter");
  await expect(page.locator(".entity-detail-dossier")).toBeVisible();
  const graphNode = page.locator(".entity-graph-node").first();
  await tabTo(page, graphNode);
  await assertFocusOutline(page);
  const ledgerRow = page.locator(".entity-relation-row").first();
  await tabTo(page, ledgerRow, 12);
  await assertFocusOutline(page);
  const addNote = page.getByRole("button", { name: "Add note" });
  await tabTo(page, addNote, 24);
  await page.keyboard.press("Enter");
  await expect(page.getByPlaceholder("New note about Ada Lovelace…")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeVisible();
  const linkedMemory = page.locator(".memory-detail-related-card").first();
  await tabTo(page, linkedMemory);
  await assertFocusOutline(page);
  const back = page.locator(".memory-detail-back");
  await tabTo(page, back);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
});

test("records the durable keyboard focus sequence and representative captures", async ({ page }) => {
  await mkdir(screenshotEvidenceDir, { recursive: true });
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  const sequence = [];

  const primarySpaces = page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Spaces", exact: true });
  await tabTo(page, primarySpaces);
  sequence.push(await focusedElementEvidence(page, "primary navigation Spaces"));
  await page.screenshot({ path: path.join(screenshotEvidenceDir, "focus-primary-navigation-spaces.png"), fullPage: false });
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();

  const filter = page.getByLabel("Filter spaces");
  await tabTo(page, filter);
  sequence.push(await focusedElementEvidence(page, "Spaces filter"));
  await page.screenshot({ path: path.join(screenshotEvidenceDir, "focus-spaces-filter.png"), fullPage: false });

  const menuTrigger = page.getByRole("button", { name: "Actions for Wenlan" });
  await tabTo(page, menuTrigger);
  sequence.push(await focusedElementEvidence(page, "Wenlan row action menu trigger"));
  await page.screenshot({ path: path.join(screenshotEvidenceDir, "focus-spaces-row-action-trigger.png"), fullPage: false });
  await page.keyboard.press("Enter");
  const rename = page.getByRole("menuitem", { name: "Rename" });
  await tabTo(page, rename, 8);
  sequence.push(await focusedElementEvidence(page, "Wenlan Rename menu item"));
  await page.screenshot({ path: path.join(screenshotEvidenceDir, "focus-spaces-rename-menu-item.png"), fullPage: false });
  await page.keyboard.press("Escape");
  await expect(menuTrigger).toBeFocused();
  sequence.push(await focusedElementEvidence(page, "Wenlan row action menu trigger restored"));

  const spaceRow = spaceOverviewButton(page);
  await tabTo(page, spaceRow);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  const spacesBack = page.locator(".space-dossier-parent");
  await tabTo(page, spacesBack);
  sequence.push(await focusedElementEvidence(page, "Space dossier back to Spaces"));
  await page.screenshot({ path: path.join(screenshotEvidenceDir, "focus-space-back-to-spaces.png"), fullPage: false });

  await writeFile(path.join(evidenceDir, "keyboard-sequence.json"), `${JSON.stringify({
    screenshots: [
      path.join(screenshotEvidenceDir, "focus-primary-navigation-spaces.png"),
      path.join(screenshotEvidenceDir, "focus-spaces-filter.png"),
      path.join(screenshotEvidenceDir, "focus-spaces-row-action-trigger.png"),
      path.join(screenshotEvidenceDir, "focus-spaces-rename-menu-item.png"),
      path.join(screenshotEvidenceDir, "focus-space-back-to-spaces.png"),
    ],
    sequence,
  }, null, 2)}\n`);
});

test("preserves the Entity signature and CJK dossiers at 200 percent zoom", async ({ browser, page }) => {
  test.setTimeout(60_000);
  const browserErrors = collectBrowserErrors(page);
  await mkdir(evidenceDir, { recursive: true });
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "zh-Hant", rawActions: [] });
  await page.goto("/");
  await openSpaces(page, { navigation: "主要導覽", spaces: "空間", show: "顯示側邊欄" });
  await spaceOverviewButton(page).click();
  await settleZoomLayout(page);
  const spaceHeading = page.getByRole("heading", { level: 1, name: "Wenlan" });
  const baselineSpaceHeading = await physicalTextMetric(spaceHeading);
  const baselineLayout = await layoutMetrics(page);
  const baselineScreenshotPath = path.join(evidenceDir, "space-zh-Hant-baseline-1280x900.png");
  await page.screenshot({ path: baselineScreenshotPath, fullPage: false });
  const baselineScreenshot = await pngDimensions(baselineScreenshotPath);

  await openSpaceEntity(page, "Ada Lovelace", "zh-Hant");
  await settleZoomLayout(page);
  const baselineEntityHeading = await physicalTextMetric(page.getByRole("heading", { level: 1, name: "Ada Lovelace" }));
  await page.locator(".memory-detail-back").click();
  await expect(spaceHeading).toBeVisible();

  const zoomContext = await browser.newContext({
    deviceScaleFactor: 2,
    viewport: { width: 640, height: 450 },
  });
  const zoomPage = await zoomContext.newPage();
  const zoomBrowserErrors = collectBrowserErrors(zoomPage);
  await installTauriMock(zoomPage, { locale: "zh-Hant", rawActions: [] });
  await zoomPage.goto("/");
  await openSidebar(zoomPage, "顯示側邊欄");
  await zoomPage.getByRole("navigation", { name: "主要導覽" }).getByRole("button", { name: "Wiki", exact: true }).click();
  const pagesHeading = zoomPage.getByRole("heading", { level: 1, name: "Wiki" });
  await expect(pagesHeading).toBeVisible();
  await assertNotClipped(pagesHeading);
  await assertNotClipped(zoomPage.getByRole("button", { name: "開啟 Independent research" }));
  await assertNoPageOverflow(zoomPage);
  await openSpaces(zoomPage, { navigation: "主要導覽", spaces: "空間", show: "顯示側邊欄" });
  await spaceOverviewButton(zoomPage).click();
  await expect(zoomPage.locator("aside.memory-sidebar")).toHaveAttribute("aria-hidden", "true");
  await settleZoomLayout(zoomPage);
  const zoomLayout = await layoutMetrics(zoomPage);
  expect(zoomLayout).toMatchObject({ clientHeight: 450, clientWidth: 640, devicePixelRatio: 2, innerHeight: 450, innerWidth: 640 });
  expect(zoomLayout.visualViewport).toMatchObject({ height: 450, scale: 1, width: 640 });
  await expect(spaceHeading).toHaveCount(1);
  const zoomSpaceHeadingLocator = zoomPage.getByRole("heading", { level: 1, name: "Wenlan" });
  await expect(zoomSpaceHeadingLocator).toHaveCount(1);
  await assertNotClipped(zoomSpaceHeadingLocator);
  const zoomSpaceHeading = await physicalTextMetric(zoomSpaceHeadingLocator);
  expect((await zoomPage.locator(".space-dossier-grid").evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(1);
  await assertNoPageOverflow(zoomPage);
  const spaceZoomPath = path.join(evidenceDir, "space-zh-Hant-zoom-200.png");
  await zoomPage.screenshot({ path: spaceZoomPath, fullPage: false });
  const spaceZoomScreenshot = await pngDimensions(spaceZoomPath);
  expect(spaceZoomScreenshot).toEqual({ width: 1280, height: 900 });

  await openSpaceEntity(zoomPage, "Ada Lovelace", "zh-Hant");
  await settleZoomLayout(zoomPage);
  const connections = zoomPage.getByRole("heading", { name: "關聯" });
  const about = zoomPage.getByRole("heading", { name: "關於" });
  expect(await connections.evaluate((left, right) => Boolean(left.compareDocumentPosition(right as Node) & Node.DOCUMENT_POSITION_FOLLOWING), await about.elementHandle())).toBe(true);
  await expect(zoomPage.getByRole("group", { name: "Ada Lovelace 的關聯圖" })).toBeVisible();
  await expect(zoomPage.getByRole("group", { name: "關聯", exact: true })).toBeVisible();
  await expect(zoomPage.locator(".entity-detail-seal")).toBeVisible();
  const entityHeading = zoomPage.getByRole("heading", { level: 1, name: "Ada Lovelace" });
  await expect(entityHeading).toHaveCount(1);
  await assertNotClipped(entityHeading);
  await assertNotClipped(connections);
  await assertNotClipped(about);
  const zoomEntityHeading = await physicalTextMetric(entityHeading);
  expect((await zoomPage.locator(".page-detail-grid").evaluate((node) => getComputedStyle(node).gridTemplateColumns)).split(" ")).toHaveLength(1);
  await assertNoPageOverflow(zoomPage);
  const entityZoomPath = path.join(evidenceDir, "entity-zh-Hant-zoom-200.png");
  await zoomPage.screenshot({ path: entityZoomPath, fullPage: false });
  const entityZoomScreenshot = await pngDimensions(entityZoomPath);
  expect(entityZoomScreenshot).toEqual({ width: 1280, height: 900 });

  const spaceNormalizedDensity = (zoomSpaceHeading.physicalFontSize / baselineSpaceHeading.physicalFontSize)
    / (zoomSpaceHeading.cssFontSize / baselineSpaceHeading.cssFontSize);
  const entityNormalizedDensity = (zoomEntityHeading.physicalFontSize / baselineEntityHeading.physicalFontSize)
    / (zoomEntityHeading.cssFontSize / baselineEntityHeading.cssFontSize);
  expect(spaceNormalizedDensity).toBeCloseTo(2, 5);
  expect(entityNormalizedDensity).toBeCloseTo(2, 5);
  expect(zoomSpaceHeading.physicalFontSize).toBeGreaterThan(baselineSpaceHeading.physicalFontSize * 1.35);
  expect(zoomEntityHeading.physicalFontSize).toBeGreaterThan(baselineEntityHeading.physicalFontSize * 1.55);

  await writeFile(path.join(evidenceDir, "zoom-proof.json"), `${JSON.stringify({
    baseline: {
      devicePixelRatio: baselineLayout.devicePixelRatio,
      entityHeading: baselineEntityHeading,
      layout: baselineLayout,
      screenshot: baselineScreenshot,
      spaceHeading: baselineSpaceHeading,
    },
    comparison: {
      entityNormalizedDensity,
      entityPhysicalFontRatio: zoomEntityHeading.physicalFontSize / baselineEntityHeading.physicalFontSize,
      spaceNormalizedDensity,
      spacePhysicalFontRatio: zoomSpaceHeading.physicalFontSize / baselineSpaceHeading.physicalFontSize,
    },
    reflow: {
      entityGridColumns: 1,
      sidebarHidden: true,
      spaceGridColumns: 1,
    },
    zoom: {
      devicePixelRatio: zoomLayout.devicePixelRatio,
      entityHeading: zoomEntityHeading,
      layout: zoomLayout,
      screenshot: entityZoomScreenshot,
      screenshots: { entity: entityZoomScreenshot, space: spaceZoomScreenshot },
      spaceHeading: zoomSpaceHeading,
    },
  }, null, 2)}\n`);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
  expect(zoomBrowserErrors.pageErrors).toEqual([]);
  expect(zoomBrowserErrors.consoleErrors).toEqual([]);
  await zoomContext.close();
});

test("removes non-essential transitions under reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 768, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await page.getByTitle("Show sidebar").click();
  await expect(page.locator("aside")).toHaveCSS("transition-duration", "0s");
  await openSpaces(page);
  const newSpace = page.getByRole("button", { name: "New space", exact: true });
  await expect(newSpace).toHaveClass(/spaces-new-action/);
  await expect(newSpace).not.toHaveClass(/spaces-primary-action/);
  await expect(newSpace).toHaveCSS("transition-duration", "0s");
  await spaceOverviewButton(page).click();
  await expect(page.locator(".space-dossier-archive-trigger svg")).toHaveCSS("transition-duration", "0s");
});
