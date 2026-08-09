// SPDX-License-Identifier: AGPL-3.0-only
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { openSpaceEntity } from "./helpers/spaceEntity";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

const evidenceDir = path.join(
  process.cwd(),
  ".omo/evidence/task-7-spaces-navigation-redesign/accessibility",
);

test("captures Simplified Chinese Space and Entity dossiers at physical DPR2", async ({ browser }) => {
  const context = await browser.newContext({
    deviceScaleFactor: 2,
    viewport: { width: 640, height: 450 },
  });
  const page = await context.newPage();
  const errors = collectBrowserErrors(page);
  await mkdir(evidenceDir, { recursive: true });
  await installTauriMock(page, { locale: "zh-Hans", rawActions: [] });
  await page.goto("/");
  await page.getByTitle("显示侧边栏").click();
  await page
    .getByRole("navigation", { name: "主导航" })
    .getByRole("button", { name: "空间", exact: true })
    .click();
  await page.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 2, name: "最近精炼" })).toBeVisible();
  await expect(page.getByRole("button", { name: "原始记忆 (205)" })).toBeVisible();
  await expect(page.locator("aside.memory-sidebar")).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator(".space-dossier-grid")).toHaveCSS("grid-template-columns", /\d+px/);
  const spacePath = path.join(evidenceDir, "space-zh-Hans-dpr2.png");
  await page.screenshot({ path: spacePath, fullPage: false });

  await openSpaceEntity(page, "Ada Lovelace", "zh-Hans");
  await expect(page.getByRole("heading", { level: 2, name: "关联" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "关于" })).toBeVisible();
  await expect(page.locator(".entity-detail-seal")).toBeVisible();
  const entityPath = path.join(evidenceDir, "entity-zh-Hans-dpr2.png");
  await page.screenshot({ path: entityPath, fullPage: false });

  const accessibilitySnapshot = await page.locator("body").ariaSnapshot();
  expect(accessibilitySnapshot).not.toMatch(/\bIndex\b|索引/iu);
  const metrics = await page.evaluate(() => ({
    devicePixelRatio,
    entityGridColumns: getComputedStyle(document.querySelector(".page-detail-grid") ?? document.body).gridTemplateColumns.split(" ").length,
    innerHeight,
    innerWidth,
  }));
  expect(metrics).toEqual({
    devicePixelRatio: 2,
    entityGridColumns: 1,
    innerHeight: 450,
    innerWidth: 640,
  });
  await writeFile(path.join(evidenceDir, "zh-Hans-dpr2.json"), `${JSON.stringify({
    ...metrics,
    screenshots: { entity: entityPath, space: spacePath },
  }, null, 2)}\n`);
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
  await context.close();
});
