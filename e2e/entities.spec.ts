// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

async function openEntities(page: Page): Promise<void> {
  await page.goto("/");
  const navigation = page.getByRole("navigation", { name: "Primary navigation" });
  await navigation.getByRole("button", { name: "Entities", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Entities" })).toBeVisible();
}

test("archives every detected entity matching the current filter, then restores it back to Detected", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openEntities(page);

  // The fixture ships exactly one detected entity (Ada Lovelace, never
  // confirmed) alongside six already-established ones, so Detected starts at
  // one row and no filter needs to be set for "all matching" to mean "all".
  await expect(page.getByRole("tab", { name: "Detected" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Ada Lovelace", exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Archive all matching" }).click();

  const archiveDialog = page.getByRole("dialog");
  await expect(archiveDialog.getByText("Archive 1 detected entity?")).toBeVisible();
  await expect(archiveDialog.getByText("Filter", { exact: true })).toBeVisible();
  await expect(archiveDialog.getByText("Any, any number of memories")).toBeVisible();
  await expect(
    archiveDialog.getByText("Archived entities can be restored from the Archived tab."),
  ).toBeVisible();

  await archiveDialog.getByRole("button", { name: "Archive", exact: true }).click();

  await expect(archiveDialog).toHaveCount(0);
  await expect(page.getByText("No detected entities match")).toBeVisible();

  await page.getByRole("tab", { name: "Archived" }).click();
  await expect(page.getByRole("cell", { name: "Ada Lovelace", exact: true })).toBeVisible();

  await page.getByRole("row", { name: /Ada Lovelace/ }).getByRole("button", { name: "Restore" }).click();

  // Ada was never confirmed before archiving, so she comes back Detected, not
  // Established -- the exact inverse of the archive, not a reset to a fixed
  // state (crates/wenlan-core/src/db.rs: restore only flips `pages.status`).
  await expect(page.getByText("No archived entities")).toBeVisible();
  await page.getByRole("tab", { name: "Detected" }).click();
  await expect(page.getByRole("cell", { name: "Ada Lovelace", exact: true })).toBeVisible();

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
