// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

const RECENT_SPACES_KEY = "wenlan:recent-spaces:v1";

async function openSpaces(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Spaces", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
}

async function openSpace(page: Page, name = "Wenlan"): Promise<void> {
  await openSpaces(page);
  await page
    .getByTestId("space-row-space-wenlan")
    .getByRole("button", { name, exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name })).toBeVisible();
}

async function expectNoIndexLabel(page: Page): Promise<void> {
  const accessibilitySnapshot = await page.locator("body").ariaSnapshot();
  expect(accessibilitySnapshot).not.toMatch(/\bIndex\b|索引/iu);
}

test("opens Home Recent by stable id, prunes missing history, and preserves selection through rename", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  const recentHistory = JSON.stringify({
    version: 1,
    entries: [
      { id: "space-wenlan", name: "Old Wenlan name", visitedAt: Date.now() - 1_000 },
      { id: "space-missing", name: "Missing", visitedAt: Date.now() - 2_000 },
    ],
  });
  await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    localStorage: { [RECENT_SPACES_KEY]: recentHistory },
  });
  await page.goto("/");

  await expectNoIndexLabel(page);
  const recent = page.getByRole("navigation", { name: "Recent spaces" });
  await recent.getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await expect(recent.getByRole("button", { name: "Wenlan", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(recent.getByRole("button", { name: "Wenlan", exact: true })).not.toHaveAttribute("aria-pressed");
  await expect.poll(() => page.evaluate((key) => localStorage.getItem(key), RECENT_SPACES_KEY)).not.toContain("space-missing");
  await expectNoIndexLabel(page);

  await page.keyboard.press("Escape");
  await expect(page.getByLabel("Home overview")).toBeVisible();
  await openSpaces(page);
  const wenlanRow = page.getByTestId("space-row-space-wenlan");
  await wenlanRow.getByRole("button", { name: "Actions for Wenlan" }).click();
  await page.getByRole("menuitem", { name: "Rename" }).click();
  const rename = wenlanRow.getByRole("textbox", { name: "Name", exact: true });
  await rename.fill("Wenlan Core");
  await rename.press("Enter");
  await expect(wenlanRow.getByRole("button", { name: "Wenlan Core", exact: true })).toBeVisible();
  await expect.poll(() => page.evaluate((key) => localStorage.getItem(key), RECENT_SPACES_KEY)).toContain('"id":"space-wenlan","name":"Wenlan Core"');

  await wenlanRow.getByRole("button", { name: "Wenlan Core", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan Core" })).toBeVisible();
  const renamedRecent = page.getByRole("navigation", { name: "Recent spaces" }).getByRole("button", { name: "Wenlan Core" });
  await expect(renamedRecent).toHaveAttribute("aria-current", "page");
  await expect(renamedRecent).not.toHaveAttribute("aria-pressed");
  await expectNoIndexLabel(page);
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test("keeps keyboard editor drafts after failure and preserves raw-memory ordering and delete cancellation", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpace(page);

  const editSpace = page.getByRole("button", { name: "Edit space" });
  await editSpace.focus();
  await editSpace.press("Enter");
  const titleEditor = page.getByRole("textbox", { name: "Wenlan" });
  const descriptionEditor = page.getByRole("textbox", { name: "Edit description" });
  await expect(titleEditor).toBeFocused();
  await titleEditor.fill("Cancelled title");
  await descriptionEditor.fill("Cancelled description");
  await descriptionEditor.press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await expect(page.getByText("Editorial memory system", { exact: true })).toBeVisible();
  await expect(titleEditor).toHaveCount(0);
  await expect(descriptionEditor).toHaveCount(0);

  await editSpace.focus();
  await editSpace.press("Space");
  const failedTitle = page.getByRole("textbox", { name: "Wenlan" });
  const failedDescription = page.getByRole("textbox", { name: "Edit description" });
  await failedTitle.fill("Failed title draft");
  await failedDescription.fill("Failed description draft");
  const updatesBefore = controller.calls().filter(({ command }) => command === "update_space").length;
  controller.failNext("update_space", "identity write failed");
  await failedDescription.press("Meta+Enter");
  await expect(page.getByRole("alert")).toContainText("Could not save this change");
  await expect(page.getByRole("textbox", { name: "Wenlan" })).toHaveValue("Failed title draft");
  await expect(page.getByRole("textbox", { name: "Edit description" })).toHaveValue("Failed description draft");
  const failedUpdates = controller.calls().filter(({ command }) => command === "update_space");
  expect(failedUpdates).toHaveLength(updatesBefore + 1);
  expect(failedUpdates.at(-1)?.args).toEqual({
    description: "Failed description draft",
    name: "Wenlan",
    newName: "Failed title draft",
  });
  await page.getByRole("textbox", { name: "Wenlan" }).press("Escape");
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();

  await page.reload();
  await openSpace(page);

  const rawMemories = page.getByRole("button", { name: "Raw memories (205)" });
  await expect(rawMemories).toHaveAttribute("aria-expanded", "false");
  await rawMemories.click();
  await expect(rawMemories).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("button", { name: "Curated", exact: true }).click();
  await page.getByRole("button", { name: "Oldest first", exact: true }).click();
  await page.getByTitle("Switch to list view").click();
  const memoryCards = page.locator(".space-dossier-archive .group.relative.h-full");
  await expect(memoryCards.nth(1)).toContainText("Fixture memory 199");

  const deleteCallsBefore = controller.calls().filter(({ command }) => command === "delete_space").length;
  await page.getByRole("button", { name: "Actions for Wenlan" }).click();
  await page.getByRole("menuitem", { name: "Delete space" }).click();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("button", { name: "Confirm delete space" })).toHaveCount(0);
  expect(controller.calls().filter(({ command }) => command === "delete_space")).toHaveLength(deleteCallsBefore);
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test("keeps and discards suggested Spaces through their dossier controls", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpaces(page);

  await page.getByTestId("space-row-space-suggested").getByRole("button", { name: "AI Workflows" }).click();
  await page.getByRole("button", { name: "Keep", exact: true }).click();
  await expect(page.getByRole("button", { name: "Keep", exact: true })).toHaveCount(0);
  await page.locator("button.space-dossier-parent").click();

  await page.getByTestId("space-row-space-suggested-2").getByRole("button", { name: "Product Signals" }).click();
  await page.getByRole("button", { name: "Discard", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
  await expect(page.getByTestId("space-row-space-suggested-2")).toHaveCount(0);
  expect(controller.calls().filter(({ command }) => command === "confirm_space")).toHaveLength(1);
  expect(controller.calls().filter(({ command }) => command === "delete_space")).toHaveLength(1);
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test("saves Space title and description through one keyboard-only dossier mutation", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await openSpace(page);

  await page.getByRole("button", { name: "Edit space" }).press("Enter");
  const titleEditor = page.getByRole("textbox", { name: "Wenlan" });
  const descriptionEditor = page.getByRole("textbox", { name: "Edit description" });
  await expect(titleEditor).toBeFocused();
  await titleEditor.fill("Wenlan Detail");
  await descriptionEditor.fill("Updated through keyboard");
  await descriptionEditor.press("Meta+Enter");
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
  await page
    .getByTestId("space-row-space-wenlan")
    .getByRole("button", { name: "Wenlan Detail", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan Detail" })).toBeVisible();
  await expect(page.getByText("Updated through keyboard", { exact: true })).toBeVisible();
  const updates = controller.calls().filter(({ command }) => command === "update_space");
  expect(updates).toHaveLength(1);
  expect(updates[0]?.args).toEqual({
    description: "Updated through keyboard",
    name: "Wenlan",
    newName: "Wenlan Detail",
  });
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});
