// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

async function openFixturePage(page: Page): Promise<void> {
  await page.goto("/");
  const navigation = page.getByRole("navigation", { name: "Primary navigation" });
  await navigation.getByRole("button", { name: "Wiki", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await page.getByRole("button", { name: "Open Fixture architecture" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Fixture architecture" })).toBeVisible();
}

async function requestDelete(page: Page): Promise<void> {
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Page actions" }).click();
  await page.getByRole("menuitem", { name: "Delete page" }).click();
}

test("deletes a Wiki Page and removes it from the inventory", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openFixturePage(page);

  await requestDelete(page);

  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Fixture architecture" })).toHaveCount(0);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("keeps a failed Wiki Page deletion visible and retryable", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  controller.failNext("delete_page", "daemon offline");
  await openFixturePage(page);

  await requestDelete(page);

  await expect(page.getByRole("alert")).toHaveText("Could not delete this page. Try again.");
  await expect(page.getByRole("heading", { level: 1, name: "Fixture architecture" })).toBeVisible();

  await requestDelete(page);
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Fixture architecture" })).toHaveCount(0);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
