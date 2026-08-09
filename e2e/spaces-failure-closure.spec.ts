// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { openSpaceEntity } from "./helpers/spaceEntity";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

async function openSpaces(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Spaces", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
}

async function openWenlan(page: Page): Promise<void> {
  await openSpaces(page);
  await page.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
}

test("closes the 899px drawer before history and moves focus safely at the 900px boundary", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  await page.setViewportSize({ width: 899, height: 900 });
  await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    localStorage: { "wenlan-sidebar-collapsed": "true" },
  });
  await page.goto("/");
  await page.getByTitle("Show sidebar").click();
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Spaces", exact: true }).click();
  await page.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();

  await page.getByTitle("Show sidebar").click();
  await page.getByRole("navigation", { name: "Recent spaces" }).getByRole("button", { name: "Wenlan" }).focus();
  await page.keyboard.press("Escape");
  const primarySidebar = page.locator('aside[aria-label="Primary navigation"]');
  await expect(primarySidebar).toHaveAttribute("aria-hidden", "true");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await expect(page.getByTitle("Show sidebar")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();

  await page.getByRole("button", { name: "Wenlan", exact: true }).click();
  await page.getByTitle("Show sidebar").click();
  const recentWenlan = page.getByRole("navigation", { name: "Recent spaces" }).getByRole("button", { name: "Wenlan" });
  await recentWenlan.focus();
  await page.setViewportSize({ width: 900, height: 900 });
  await expect(primarySidebar).toHaveAttribute("aria-hidden", "true");
  await expect.poll(() => page.evaluate(() => document.querySelector('aside[aria-label="Primary navigation"]')?.contains(document.activeElement))).toBe(false);
  await expect(page.getByTitle("Show sidebar")).toBeFocused();
  expect(await page.evaluate(() => localStorage.getItem("wenlan-sidebar-collapsed"))).toBe("true");
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test("turns a non-Error Spaces rejection into a recoverable inline failure", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpaces(page);
  await page.evaluate(() => {
    const internals = window.__TAURI_INTERNALS__;
    if (!internals) throw new Error("Tauri internals are unavailable");
    const originalInvoke = internals.invoke;
    let rejectOnce = true;
    internals.invoke = (command, args, options) => {
      if (rejectOnce && command === "update_space") {
        rejectOnce = false;
        return Promise.reject("plain string rejection");
      }
      return originalInvoke(command, args, options);
    };
  });

  await page.getByRole("button", { name: "Actions for Archive" }).click();
  await page.getByRole("menuitem", { name: "Rename" }).click();
  await page.getByRole("textbox", { name: "Name", exact: true }).fill("Archive Retry");
  await page.getByRole("textbox", { name: "Name", exact: true }).press("Enter");
  await expect(page.getByRole("alert")).toContainText("could not be saved");
  await expect(page.getByRole("alert")).not.toContainText("plain string rejection");
  await expect(page.getByRole("textbox", { name: "Name", exact: true })).toHaveValue("Archive Retry");
  await page.getByRole("textbox", { name: "Name", exact: true }).press("Enter");
  await expect(page.getByRole("button", { name: "Archive Retry", exact: true })).toBeVisible();
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test("retries Entity load, retains a failed observation draft, and retries two-step deletion", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openWenlan(page);

  controller.failNext("get_entity_detail_cmd", "entity offline");
  await openSpaceEntity(page, "Ada Lovelace");
  await expect(page.getByText("Couldn't load this entity.")).toBeVisible();
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeVisible();

  await page.getByRole("button", { name: "Wrote the first published algorithm" }).click();
  const observation = page.getByRole("textbox", { name: "Edit note" });
  await observation.fill("Observation draft survives");
  controller.failNext("update_observation_cmd", "observation write failed");
  await observation.press("Enter");
  await expect(page.getByRole("alert")).toContainText("Couldn't save. Try again.");
  await expect(page.getByRole("textbox", { name: "Edit note" })).toHaveValue("Observation draft survives");
  await page.getByRole("textbox", { name: "Edit note" }).press("Escape");

  controller.failNext("delete_entity_cmd", "delete failed");
  await page.getByRole("button", { name: "Delete entity" }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Couldn't save. Try again.");
  await expect(page.getByRole("heading", { level: 1, name: "Ada Lovelace" })).toBeVisible();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();

  expect(controller.calls().filter(({ command }) => command === "get_entity_detail_cmd").length).toBeGreaterThanOrEqual(2);
  expect(controller.calls().filter(({ command }) => command === "update_observation_cmd")).toHaveLength(1);
  expect(controller.calls().filter(({ command }) => command === "delete_entity_cmd")).toHaveLength(2);
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});
