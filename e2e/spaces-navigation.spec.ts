// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { openSpaceEntity } from "./helpers/spaceEntity";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

test("Home -> Spaces -> Space -> Page -> back and Space -> Entity -> back", async ({ page }) => {
  // Given a clean fixture and browser error capture.
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  // When the two primary hierarchy journeys are driven through the rendered shell.
  const primaryNavigation = page.getByRole("navigation", { name: "Primary navigation" });
  await expect(primaryNavigation.locator('[aria-current="page"]')).toHaveCount(1);
  await expect(primaryNavigation.getByRole("button", { name: "Home" })).toHaveAttribute("aria-current", "page");

  await primaryNavigation.getByRole("button", { name: "Wiki", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Independent research" })).toContainText("Independent");
  await expect(primaryNavigation.getByRole("button", { name: "Wiki" })).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: "Open Independent research" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Independent research" })).toBeVisible();
  await expect(primaryNavigation.getByRole("button", { name: "Wiki" })).toHaveAttribute("aria-current", "page");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();

  const main = page.locator("main");
  await main.evaluate((node) => { node.scrollTop = node.scrollHeight; });
  await expect.poll(() => main.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);

  await primaryNavigation.getByRole("button", { name: "Spaces", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
  await expect.poll(() => main.evaluate((node) => node.scrollTop)).toBe(0);
  await expect(page.getByRole("navigation", { name: "Recent spaces" })).toHaveCount(0);
  const wenlanRow = page.getByTestId("space-row-space-wenlan");
  await wenlanRow.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
  const recentSpaces = page.getByRole("navigation", { name: "Recent spaces" });
  const recentWenlan = recentSpaces.getByRole("button", { name: "Wenlan", exact: true });
  await expect(recentWenlan).toBeVisible();
  await expect(recentWenlan).not.toHaveAttribute("aria-current");
  await expect(recentWenlan).not.toHaveAttribute("aria-pressed");
  await wenlanRow.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await expect(recentWenlan).toHaveAttribute("aria-current", "page");

  await page.getByRole("button", { name: /Fixture architecture/ }).first().click();
  await expect(page.getByRole("heading", { level: 1, name: "Fixture architecture" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();

  await openSpaceEntity(page, "Ada Lovelace");
  await expect(page.getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeVisible();
  await page.keyboard.press("Escape");

  // Then history returns to the Space and the browser stayed error-free.
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await expect(primaryNavigation.locator('[aria-current="page"]')).toHaveCount(1);
  await expect(primaryNavigation.getByRole("button", { name: "Spaces", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(recentWenlan).toHaveAttribute("aria-current", "page");
  await expect(recentWenlan).not.toHaveAttribute("aria-pressed");
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
