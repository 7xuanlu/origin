// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock } from "./tauriMock";
import { openCanvas } from "./helpers/pageCanvas";

/**
 * Dragging a box's connector into empty space is the canvas's primary way to
 * grow a map, and it only exists as pointer events — jsdom has no layout to
 * drag across and the Tauri webview refuses synthetic ones. Chromium is the
 * only place this interaction can actually be observed.
 */

/** Grab the dot under a box and let go somewhere empty. */
async function dragConnectorToEmptySpace(page: Page): Promise<void> {
  const handle = page.locator(".page-canvas-grow").first();
  await expect(handle).toBeVisible();
  const from = await handle.boundingBox();
  if (!from) throw new Error("connector handle has no box");
  const startX = from.x + from.width / 2;
  const startY = from.y + from.height / 2;

  await page.mouse.move(startX, startY);
  await page.mouse.down();
  // Several moves, not one: React Flow starts a connection on the first move
  // past its drag threshold and needs the pointer to actually travel.
  for (const step of [1, 2, 3, 4]) {
    await page.mouse.move(startX + step * 40, startY + step * 30);
  }
  await page.mouse.up();
}

test("keeps the box dragged out of a connector standing after the pointer is released", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);

  await dragConnectorToEmptySpace(page);

  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  // The regression: the field mounted, React Flow took focus back for the pane,
  // and the empty blur deleted the box a beat later. Wait past that beat.
  await page.waitForTimeout(1000);
  await expect(field).toBeVisible();
  expect(controller.calls().some((call) => call.command === "create_page_map_node")).toBe(false);

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("names the dragged-out box and keeps it on the canvas", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);

  await dragConnectorToEmptySpace(page);

  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  await field.fill("Design notes");
  await field.press("Enter");

  await expect(page.getByRole("button", { name: "Design notes" })).toBeVisible();
  const created = controller.calls().filter((call) => call.command === "create_page_map_node");
  expect(created).toHaveLength(1);

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("keeps the box and its line on screen while a slow daemon catches up", async ({ page }) => {
  const browserErrors = collectBrowserErrors(page);
  await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    // What a real library costs: the map refetch behind a create is seconds, not
    // milliseconds, and the box has to survive that window.
    delays: { get_page_map: 1500, update_page: 400 },
  });
  await openCanvas(page);

  await dragConnectorToEmptySpace(page);
  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();

  // A line joins the box to the one it was dragged from, from the moment the
  // pointer lifts — not only once the server copy lands.
  await expect(page.locator('.react-flow__edge[data-id="tree-__draft__"]')).toHaveCount(1);

  await field.fill("Design notes");
  await field.press("Enter");

  // Straight after commit, mid-flight: the box is still there, named.
  await expect(page.getByRole("button", { name: "Design notes" })).toBeVisible();
  await expect(page.locator('.react-flow__edge[data-id="tree-__draft__"]')).toHaveCount(1);

  // And once the daemon answers, the server's copy takes over without a blink.
  await expect(page.locator('.react-flow__edge[data-id="tree-__draft__"]')).toHaveCount(0, {
    timeout: 15_000,
  });
  await expect(page.getByRole("button", { name: "Design notes" })).toBeVisible();

  expect(browserErrors.pageErrors).toEqual([]);
  expect(browserErrors.consoleErrors).toEqual([]);
});

test("drops the dragged-out box on Escape", async ({ page }) => {
  const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);

  await dragConnectorToEmptySpace(page);
  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  await field.press("Escape");

  await expect(field).toHaveCount(0);
  expect(controller.calls().some((call) => call.command === "create_page_map_node")).toBe(false);
});
