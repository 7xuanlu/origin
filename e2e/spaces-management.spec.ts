// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";

async function openOverview(page: Page): Promise<void> {
  await page.getByRole("navigation", { name: "Primary navigation" }).getByRole("button", { name: "Spaces", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();
}

function confirmedSpaceRow(page: Page, id: string) {
  return page.getByTestId(`space-row-${id}`);
}

async function openMenu(page: Page, id: string, name: string): Promise<void> {
  await confirmedSpaceRow(page, id).getByRole("button", { name: `Actions for ${name}` }).click();
}

test("manages Spaces, cleans MRU, and preserves data after a rejected mutation", async ({ page }) => {
  // Given a clean typed fixture with command capture.
  const browserErrors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  // When create, filter, star, reorder, rename, failure, suggestions, and delete run.
  await openOverview(page);
  await page.getByRole("button", { name: "New space", exact: true }).click();
  const createName = page.getByRole("textbox", { name: "Name", exact: true });
  await expect(createName).toBeFocused();
  await createName.fill("Journey Space");
  await page.getByLabel("Description").fill("Created through Playwright");
  await createName.evaluate((input) => {
    input.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    input.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
  });
  const journeyRow = confirmedSpaceRow(page, "space-journey-space");
  await expect(journeyRow.getByRole("button", { name: "Journey Space", exact: true })).toBeVisible();

  const filter = page.getByLabel("Filter spaces");
  await filter.fill("  journey  ");
  await expect(journeyRow.getByRole("button", { name: "Journey Space", exact: true })).toBeVisible();
  await expect(confirmedSpaceRow(page, "space-wenlan")).toHaveCount(0);
  await filter.fill("");

  await openMenu(page, "space-journey-space", "Journey Space");
  await page.getByRole("menuitem", { name: "Star" }).click();
  await expect(journeyRow.getByRole("button", { name: "Journey Space", exact: true })).toBeVisible();
  await openMenu(page, "space-journey-space", "Journey Space");
  await page.getByRole("menuitem", { name: "Move up" }).click();
  const movedJourney = journeyRow.getByRole("button", { name: "Journey Space", exact: true });
  const researchRow = confirmedSpaceRow(page, "space-research");
  const shiftedResearch = researchRow.getByRole("button", { name: "Research", exact: true });
  expect(await movedJourney.evaluate((left, right) => Boolean(left.compareDocumentPosition(right as Node) & Node.DOCUMENT_POSITION_FOLLOWING), await shiftedResearch.elementHandle())).toBe(true);
  await openMenu(page, "space-journey-space", "Journey Space");
  await page.getByRole("menuitem", { name: "Rename" }).click();
  const renameName = journeyRow.getByRole("textbox", { name: "Name", exact: true });
  await renameName.fill("Renamed Journey");
  await renameName.press("Enter");
  await expect(journeyRow.getByRole("button", { name: "Renamed Journey", exact: true })).toBeVisible();

  controller.failNext("update_space", "intentional rejection");
  await openMenu(page, "space-journey-space", "Renamed Journey");
  await page.getByRole("menuitem", { name: "Rename" }).click();
  await journeyRow.getByRole("textbox", { name: "Name", exact: true }).fill("Rejected Rename");
  await journeyRow.getByRole("textbox", { name: "Name", exact: true }).press("Enter");
  await expect(page.getByRole("alert")).toContainText("could not be saved");
  await expect(journeyRow.getByRole("textbox", { name: "Name", exact: true })).toHaveValue("Rejected Rename");
  await journeyRow.getByRole("textbox", { name: "Name", exact: true }).press("Escape");
  await expect(journeyRow.getByRole("button", { name: "Renamed Journey", exact: true })).toBeVisible();

  await page.getByTestId("space-row-space-suggested").getByRole("button", { name: "Keep" }).click();
  await page.getByTestId("space-row-space-suggested-2").getByRole("button", { name: "Discard" }).click();
  await expect(page.getByTestId("space-row-space-suggested-2")).toHaveCount(0);

  await researchRow.getByRole("button", { name: "Research", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Research" })).toBeVisible();
  await page.locator("button.space-dossier-parent").click();
  await confirmedSpaceRow(page, "space-wenlan").getByRole("button", { name: "Wenlan", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wenlan" })).toBeVisible();
  await page.locator("button.space-dossier-parent").click();

  await openMenu(page, "space-research", "Research");
  await page.getByRole("menuitem", { name: "Rename" }).click();
  await researchRow.getByRole("textbox", { name: "Name", exact: true }).fill("Research Lab");
  await researchRow.getByRole("textbox", { name: "Name", exact: true }).press("Enter");
  await expect.poll(async () => page.evaluate(() => localStorage.getItem("wenlan:recent-spaces:v1"))).toContain("Research Lab");
  await openMenu(page, "space-research", "Research Lab");
  await page.getByRole("menuitem", { name: "Delete" }).click();
  await researchRow.getByRole("button", { name: "Confirm delete" }).click();
  await expect.poll(async () => page.evaluate(() => localStorage.getItem("wenlan:recent-spaces:v1"))).not.toContain("Research Lab");

  await openMenu(page, "space-journey-space", "Renamed Journey");
  await page.getByRole("menuitem", { name: "Delete" }).click();
  await journeyRow.getByRole("button", { name: "Confirm delete" }).click();

  // Then every state transition is singular, recoverable, and browser-clean.
  const calls = controller.calls();
  expect(calls.filter((call) => call.command === "create_space")).toHaveLength(1);
  expect(calls.filter((call) => call.command === "confirm_space")).toHaveLength(1);
  expect(calls.filter((call) => call.command === "delete_space").length).toBeGreaterThanOrEqual(3);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("recovers from corrupt recent-space storage on reload", async ({ page }) => {
  // Given corrupt persisted recency.
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    localStorage: { "wenlan:recent-spaces:v1": "{corrupt" },
  });

  // When Home reloads.
  await page.goto("/");
  await page.reload();

  // Then invalid history is treated as no visits, so Recents is omitted without browser errors.
  await expect(page.getByLabel("Home overview")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Recent spaces" })).toHaveCount(0);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
