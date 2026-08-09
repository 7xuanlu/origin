// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors } from "./tauriMock";

async function openWiki(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Wiki", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
}

async function openFixtureArchitecture(page: Page): Promise<void> {
  await openWiki(page);
  await page.getByRole("button", { name: "Open Fixture architecture" }).click();
  await expect(
    page.getByRole("heading", { level: 1, name: "Fixture architecture" }),
  ).toBeVisible();
}

async function installRejectedCommandAudit(page: Page): Promise<void> {
  await page.addInitScript(() => {
    Reflect.set(window, "__WENLAN_REVIEW_COMMAND_FAILURES__", []);
  });
}

async function rejectedCommands(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const failures = Reflect.get(window, "__WENLAN_REVIEW_COMMAND_FAILURES__");
    return Array.isArray(failures) ? failures : [];
  });
}

test("keeps every enabled primary destination inside the Review command contract", async ({
  page,
}) => {
  const browserErrors = collectBrowserErrors(page);
  await installRejectedCommandAudit(page);
  await page.goto("/");

  const navigation = page.getByRole("navigation", { name: "Primary navigation" });

  await navigation.getByRole("button", { name: "Wiki", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();

  await navigation.getByRole("button", { name: "Spaces", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();

  await navigation.getByRole("button", { name: "Graph", exact: true }).click();
  await expect(page.getByTestId("atlas-view")).toBeVisible();
  await expect(page.getByText(/^7 entities(?: · \d+ regions?)?$/)).toBeVisible();

  await navigation.getByRole("button", { name: "Memories", exact: true }).click();
  await expect(page.getByRole("region", { name: "Memory list" })).toBeVisible();
  const firstMemory = page.getByRole("article").first();
  await firstMemory.getByRole("button", { name: "Unpin memory" }).click();
  await expect(firstMemory.getByRole("button", { name: "Pin memory" })).toBeVisible();
  await firstMemory.getByRole("button", { name: "Unconfirm memory" }).click();
  await expect(firstMemory.getByRole("button", { name: "Confirm memory" })).toBeVisible();

  await navigation.getByRole("button", { name: "Sources", exact: true }).click();
  await expect(
    page.getByRole("heading", { level: 2, name: "Nothing on the shelf yet" }),
  ).toBeVisible();

  await navigation.getByRole("button", { name: "Home", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Today in Wenlan" })).toBeVisible();

  await page.waitForTimeout(250);
  expect(await rejectedCommands(page)).toEqual([]);
  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("proves Review identity and exercises Wiki Page mutations", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await page.goto("/");

  const fixtureNotice = page.locator('[data-review-environment="fixture-only"]');
  await expect(fixtureNotice).toBeVisible();
  await expect(fixtureNotice).toContainText("TEST DATA");
  await expect(fixtureNotice).toContainText("Fixture data · resets on relaunch");

  await openFixtureArchitecture(page);
  await page.locator('.mem-icon-action[aria-label="Edit page"]').click();
  const editor = page.locator(".page-detail textarea");
  await editor.fill("# Fixture architecture\n\nEdited through the Review-flavor lane.");
  await page.getByRole("button", { name: "Save (Cmd+Enter)" }).click();
  await expect(page.getByText("Edited through the Review-flavor lane.")).toBeVisible();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Re-distill page" }).click();
  await expect(page.getByText("Page re-distilled.", { exact: true })).toBeVisible();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Page actions" }).click();
  await page.getByRole("menuitem", { name: "Delete page" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Open Fixture architecture" }),
  ).toHaveCount(0);

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("creates and publishes a Page draft through the Review runtime", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await page.goto("/");
  await openWiki(page);

  await page.getByRole("button", { name: "New page", exact: true }).click();
  await page.getByRole("textbox", { name: "Title", exact: true }).fill(
    "Review lane authored Page",
  );
  await page.getByRole("textbox", { name: "Content", exact: true }).fill(
    "This Page proves draft creation and publication in the isolated Review runtime.",
  );
  await page.getByRole("button", { name: "Publish", exact: true }).click();

  await expect(
    page.getByRole("heading", { level: 1, name: "Review lane authored Page" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "This Page proves draft creation and publication in the isolated Review runtime.",
    ),
  ).toBeVisible();

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("creates a Space and navigates to its rendered detail", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await page.goto("/");

  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Spaces", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Spaces" })).toBeVisible();

  await page.getByRole("button", { name: "New space", exact: true }).click();
  await page.getByRole("textbox", { name: "Name", exact: true }).fill("Review Lane");
  await page.getByRole("textbox", { name: "Description", exact: true }).fill(
    "Created by the Review-flavor Playwright lane.",
  );
  await page.getByRole("button", { name: "Create", exact: true }).click();

  const row = page.getByTestId("space-row-space-review-lane");
  await expect(row.getByRole("button", { name: "Review Lane", exact: true })).toBeVisible();
  await row.getByRole("button", { name: "Review Lane", exact: true }).click();
  await expect(
    page.getByRole("heading", { level: 1, name: "Review Lane" }),
  ).toBeVisible();
  await expect(page.getByText("Created by the Review-flavor Playwright lane.")).toBeVisible();

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});
