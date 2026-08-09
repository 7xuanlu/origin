// SPDX-License-Identifier: AGPL-3.0-only
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { openSpaceEntity } from "./helpers/spaceEntity";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

const evidenceDir = path.join(
  process.cwd(),
  ".omo/evidence/task-7-spaces-navigation-redesign/space-mark",
);
const screenshotEvidenceDir = path.join(
  process.cwd(),
  ".omo/evidence/task-7-spaces-navigation-redesign/screenshots",
);
const markSelector = "[data-space-mark='self-contained-world']";

async function settle(page: Page): Promise<void> {
  await page.evaluate(async () => {
    await document.fonts.ready;
    for (const animation of document.getAnimations()) animation.finish();
  });
}

async function pngDimensions(filePath: string): Promise<{ readonly height: number; readonly width: number }> {
  const bytes = await readFile(filePath);
  if (bytes.toString("ascii", 1, 4) !== "PNG") throw new Error(`Not a PNG: ${filePath}`);
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

async function capture(page: Page, name: string): Promise<string> {
  await settle(page);
  const filePath = path.join(evidenceDir, `${name}.png`);
  await page.screenshot({ path: filePath, fullPage: false });
  return filePath;
}

function spacesButton(page: Page, navigationName = "Primary navigation", spacesName = "Spaces"): Locator {
  return page
    .getByRole("navigation", { name: navigationName })
    .getByRole("button", { name: spacesName, exact: true });
}

async function markState(mark: Locator) {
  return mark.evaluate((node) => {
    const pathNode = node.querySelector("path");
    const style = getComputedStyle(node);
    return {
      ariaHidden: node.getAttribute("aria-hidden"),
      box: node.getBoundingClientRect().toJSON(),
      color: style.color,
      indigoToken: style.getPropertyValue("--mem-accent-indigo").trim(),
      path: pathNode?.getAttribute("d"),
      paths: Array.from(node.querySelectorAll("path"), (path) => path.getAttribute("d")),
      strokeWidth: pathNode?.getAttribute("stroke-width"),
      tertiaryToken: style.getPropertyValue("--mem-text-tertiary").trim(),
      viewBox: node.getAttribute("viewBox"),
    };
  });
}

test("renders the Planet Space mark across light, dark, mobile, focus, and physical DPR2 states", async ({ browser, page }) => {
  test.setTimeout(90_000);
  await mkdir(evidenceDir, { recursive: true });
  const browserErrors = collectBrowserErrors(page);
  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await settle(page);

  const screenshots: string[] = [];
  const defaultButton = spacesButton(page);
  const defaultMark = defaultButton.locator(markSelector);
  await expect(defaultMark).toHaveAttribute("aria-hidden", "true");
  await expect(defaultMark).toHaveAttribute("height", "14");
  await expect(defaultMark).toHaveAttribute("width", "14");
  await expect(defaultMark).toHaveAttribute("viewBox", "0 0 24 24");
  await expect(page.locator('path[d="M4 5.5h6l2 2h8v11H4z"]')).toHaveCount(0);
  await expect(page.locator('path[d="M3 7l6 -3l6 3l6 -3v13l-6 3l-6 -3l-6 3v-13"]')).toHaveCount(0);
  await expect(defaultButton).not.toHaveAttribute("aria-current", "page");
  const lightDefault = await markState(defaultMark);
  expect(lightDefault).toMatchObject({
    ariaHidden: "true",
    color: "rgb(93, 104, 122)",
    indigoToken: "#5E58C8",
    paths: [
      "M18.816 13.58c2.292 2.138 3.546 4 3.092 4.9c-.745 1.46 -5.783 -.259 -11.255 -3.838c-5.47 -3.579 -9.304 -7.664 -8.56 -9.123c.464 -.91 2.926 -.444 5.803 .805",
      "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0",
    ],
    strokeWidth: "2",
    tertiaryToken: "#5D687A",
    viewBox: "0 0 24 24",
  });
  expect(lightDefault.box).toMatchObject({ height: 14, width: 14 });
  screenshots.push(await capture(page, "space-mark-home-default-light-1280x900"));

  await defaultButton.click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
  const selectedButton = spacesButton(page);
  const selectedMark = selectedButton.locator(markSelector);
  await expect(selectedButton).toHaveAttribute("aria-current", "page");
  await expect(selectedButton).toHaveAccessibleName("Spaces");
  const lightSelected = await markState(selectedMark);
  expect(lightSelected.color).toBe("rgb(94, 88, 200)");
  screenshots.push(await capture(page, "space-mark-spaces-selected-light-1280x900"));

  await page.setViewportSize({ width: 375, height: 812 });
  await page.getByTitle("Show sidebar").click();
  await expect(page.locator("aside.memory-sidebar")).toHaveAttribute("aria-hidden", "false");
  await expect(spacesButton(page)).toHaveAttribute("aria-current", "page");
  await expect(spacesButton(page).locator(markSelector)).toBeVisible();
  screenshots.push(await capture(page, "space-mark-selected-overlay-light-375x812"));

  await page.setViewportSize({ width: 1280, height: 900 });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Home", exact: true }).click();
  const darkDefault = await markState(spacesButton(page).locator(markSelector));
  expect(darkDefault).toMatchObject({
    color: "rgb(150, 155, 173)",
    indigoToken: "#A9AEF2",
    tertiaryToken: "#969BAD",
  });
  screenshots.push(await capture(page, "space-mark-home-default-dark-1280x900"));
  await spacesButton(page).click();
  const darkSelected = await markState(spacesButton(page).locator(markSelector));
  expect(darkSelected.color).toBe("rgb(169, 174, 242)");
  screenshots.push(await capture(page, "space-mark-spaces-selected-dark-1280x900"));

  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  const homeButton = page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Home", exact: true });
  await homeButton.focus();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Wiki", exact: true })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(spacesButton(page)).toBeFocused();
  await expect(spacesButton(page)).toHaveAttribute("aria-current", "page");
  const focusStyle = await spacesButton(page).evaluate((node) => {
    const style = getComputedStyle(node);
    return {
      color: style.outlineColor,
      offset: Number.parseFloat(style.outlineOffset),
      style: style.outlineStyle,
      width: Number.parseFloat(style.outlineWidth),
    };
  });
  expect(focusStyle.style).toBe("solid");
  expect(focusStyle.width).toBe(2);
  expect(focusStyle.offset).toBe(2);
  screenshots.push(await capture(page, "space-mark-selected-keyboard-focus-light-1280x900"));

  await page.getByRole("button", { name: "Wenlan", exact: true }).click();
  await openSpaceEntity(page, "Ada Lovelace");
  await settle(page);
  await page.screenshot({
    path: path.join(screenshotEvidenceDir, "entity-1280x900-graph-label.png"),
    fullPage: false,
  });

  const dpr2Context = await browser.newContext({ deviceScaleFactor: 2, viewport: { width: 1280, height: 900 } });
  const dpr2Page = await dpr2Context.newPage();
  const dpr2Errors = collectBrowserErrors(dpr2Page);
  await installTauriMock(dpr2Page, { locale: "en", rawActions: [] });
  await dpr2Page.goto("/");
  await settle(dpr2Page);
  expect(await dpr2Page.evaluate(() => devicePixelRatio)).toBe(2);
  const dpr2DefaultMark = spacesButton(dpr2Page).locator(markSelector);
  const dpr2DefaultPath = path.join(evidenceDir, "space-mark-default-light-dpr2.png");
  await dpr2DefaultMark.screenshot({ path: dpr2DefaultPath });
  expect(await pngDimensions(dpr2DefaultPath)).toEqual({ width: 28, height: 30 });
  screenshots.push(dpr2DefaultPath);
  await spacesButton(dpr2Page).click();
  const dpr2SelectedMark = spacesButton(dpr2Page).locator(markSelector);
  const dpr2SelectedPath = path.join(evidenceDir, "space-mark-selected-light-dpr2.png");
  await dpr2SelectedMark.screenshot({ path: dpr2SelectedPath });
  expect(await pngDimensions(dpr2SelectedPath)).toEqual({ width: 28, height: 30 });
  screenshots.push(dpr2SelectedPath);
  expect(await markState(dpr2SelectedMark)).toMatchObject({
    ariaHidden: "true",
    box: { height: 14, width: 14 },
    color: "rgb(94, 88, 200)",
    viewBox: "0 0 24 24",
  });
  expect(dpr2Errors.pageErrors).toEqual([]);
  expect(dpr2Errors.consoleErrors).toEqual([]);
  await dpr2Context.close();

  const zhHantContext = await browser.newContext({ deviceScaleFactor: 1, viewport: { width: 1280, height: 900 } });
  const zhHantPage = await zhHantContext.newPage();
  const zhHantErrors = collectBrowserErrors(zhHantPage);
  await installTauriMock(zhHantPage, { locale: "zh-Hant", rawActions: [] });
  await zhHantPage.goto("/");
  await spacesButton(zhHantPage, "主要導覽", "空間").click();
  await zhHantPage.getByRole("button", { name: "Wenlan", exact: true }).click();
  await settle(zhHantPage);
  await zhHantPage.screenshot({
    path: path.join(
      process.cwd(),
      ".omo/evidence/task-7-spaces-navigation-redesign/accessibility/space-zh-Hant-baseline-1280x900.png",
    ),
    fullPage: false,
  });
  expect(zhHantErrors.pageErrors).toEqual([]);
  expect(zhHantErrors.consoleErrors).toEqual([]);
  await zhHantContext.close();

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
  await writeFile(path.join(evidenceDir, "space-mark-qa.json"), `${JSON.stringify({
    consoleErrors: 0,
    dark: { default: darkDefault, selected: darkSelected },
    deviceScaleFactor: { primary: 1, physicalProof: 2 },
    focus: focusStyle,
    light: { default: lightDefault, selected: lightSelected },
    oldFolderPathCount: 0,
    oldMapPathCount: 0,
    screenshots,
  }, null, 2)}\n`);
});
