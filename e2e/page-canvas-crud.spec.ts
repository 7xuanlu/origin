// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Locator, type Page } from "@playwright/test";
import { collectBrowserErrors, installTauriMock, type TauriMockController } from "./tauriMock";

/**
 * The canvas advertises a shortcut for every way to change a map, and the
 * shortcut sheet is a promise to the reader. This sweep holds each line of it to
 * that promise in a real browser, because every one of them is pointer or key
 * behaviour that jsdom cannot produce.
 */
async function openCanvas(page: Page): Promise<void> {
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Wiki", exact: true })
    .click();
  await page.getByRole("button", { name: "Open Fixture architecture" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Fixture architecture" })).toBeVisible();
  await page.getByRole("button", { name: "Canvas" }).click();
  await expect(page.getByRole("region", { name: "Canvas for Fixture architecture" })).toBeVisible();
  await expect(box(page, "Storage layer")).toBeVisible();
}

/** A box on the canvas, addressed the way a reader sees it: by its name. */
function box(page: Page, label: string): Locator {
  return page.locator(".react-flow__node").filter({ hasText: label }).first();
}

function surface(page: Page): Locator {
  return page.getByRole("region", { name: "Canvas for Fixture architecture" });
}

/** Somewhere on the pane with no box under it. */
async function emptySpot(page: Page): Promise<{ x: number; y: number }> {
  const pane = await surface(page).boundingBox();
  if (!pane) throw new Error("canvas has no box");
  return { x: pane.x + pane.width - 90, y: pane.y + pane.height - 110 };
}

async function named(page: Page, label: string): Promise<void> {
  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  await field.fill(label);
  await field.press("Enter");
}

function created(controller: TauriMockController): unknown[] {
  return controller.calls().filter((c) => c.command === "create_page_map_node");
}

test.describe("canvas CRUD", () => {
  test("double-click on empty canvas draws a box there", async ({ page }) => {
    const browserErrors = collectBrowserErrors(page);
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    const spot = await emptySpot(page);
    await page.mouse.dblclick(spot.x, spot.y);
    await named(page, "Write path");

    await expect(box(page, "Write path")).toBeVisible();
    expect(created(controller)).toHaveLength(1);
    expect(browserErrors.pageErrors).toEqual([]);
  });

  test("double-click on a box renames it, starting from its current name", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Storage layer").dblclick();
    const field = page.getByRole("textbox", { name: "Section name" });
    await expect(field).toHaveValue("Storage layer");
    await field.fill("Storage engine");
    await field.press("Enter");

    await expect(box(page, "Storage engine")).toBeVisible();
    const renames = controller.calls().filter((c) => c.command === "patch_page_map_node");
    expect(renames).toHaveLength(1);
  });

  test("F2 renames the selected box", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Query path").click();
    await page.keyboard.press("F2");

    await expect(page.getByRole("textbox", { name: "Section name" })).toHaveValue("Query path");
  });

  test("Tab adds a box inside the selected one, Enter adds one beside it", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Storage layer").click();
    await page.keyboard.press("Tab");
    await named(page, "Column types");
    await expect(box(page, "Column types")).toBeVisible();

    await box(page, "Storage layer").click();
    await page.keyboard.press("Enter");
    await named(page, "Cache layer");
    await expect(box(page, "Cache layer")).toBeVisible();

    const calls = created(controller) as { args: { body: { parent_id: string } } }[];
    expect(calls).toHaveLength(2);
    // Inside means a child of the box that was selected; beside means a sibling,
    // so it hangs off that box's own parent.
    expect(calls[0].args.body.parent_id).toBe("n_storage");
    expect(calls[1].args.body.parent_id).toBe("n_root");
  });

  test("Delete removes the selected box, and Escape first drops the selection", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Query path").click();
    await page.keyboard.press("Escape");
    await expect(page.locator(".react-flow__node.selected")).toHaveCount(0);
    await page.keyboard.press("Delete");
    await page.waitForTimeout(500);
    // Nothing was selected, so Delete had nothing to act on.
    expect(controller.calls().filter((c) => c.command === "delete_page_map_node")).toHaveLength(0);
    await expect(box(page, "Query path")).toBeVisible();

    await box(page, "Query path").click();
    await page.keyboard.press("Delete");
    await expect(box(page, "Query path")).toHaveCount(0);
  });

  test("says why the centre box cannot be deleted instead of doing nothing", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Fixture architecture").click();
    await page.keyboard.press("Delete");

    await expect(
      page.getByText("The center box is the page itself. It can't be deleted."),
    ).toBeVisible();
  });

  test("right-click on the pane offers to add, select all, and fit", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    const spot = await emptySpot(page);
    await page.mouse.click(spot.x, spot.y, { button: "right" });

    const menu = page.getByRole("menu", { name: "Canvas actions" });
    await expect(menu.getByRole("menuitem", { name: "New box here" })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: "Select all" })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: "Fit to view" })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
  });

  test("right-click on a box names the cascade before deleting a parent", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    // "Storage layer" has a box inside it, so the verb has to say so.
    await box(page, "Storage layer").click({ button: "right" });
    const menu = page.getByRole("menu", { name: "Canvas actions" });
    await expect(menu.getByRole("menuitem", { name: "Delete, with everything inside" })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: "Add box inside" })).toBeVisible();
    await expect(menu.getByRole("menuitem", { name: "Rename" })).toBeVisible();
    // A section is a heading in the body — there is nothing behind it to open.
    await expect(menu.getByRole("menuitem", { name: "Open" })).toHaveCount(0);

    await page.keyboard.press("Escape");
    // A leaf deletes alone, so it gets the plain verb.
    await box(page, "Query path").click({ button: "right" });
    await expect(page.getByRole("menuitem", { name: "Delete", exact: true })).toBeVisible();
  });

  test("deleting a parent takes what is inside it down too", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Storage layer").click({ button: "right" });
    await page.getByRole("menuitem", { name: "Delete, with everything inside" }).click();

    await expect(box(page, "Storage layer")).toHaveCount(0);
    await expect(box(page, "Blob columns")).toHaveCount(0);
    const removed = controller.calls().filter((c) => c.command === "delete_page_map_node");
    expect(removed).toHaveLength(2);
  });
});

test.describe("canvas Escape", () => {
  /**
   * Escape is shared: the canvas uses it to dismiss its own things and the app
   * uses it to leave a page. Getting the order wrong is not a small bug — it
   * threw the reader out of the page mid-edit, which is why each layer is pinned
   * separately here.
   */
  const onCanvas = (page: Page) =>
    expect(page.getByRole("region", { name: "Canvas for Fixture architecture" })).toBeVisible();

  test("closes the menu and stays on the page", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    const spot = await emptySpot(page);
    await page.mouse.click(spot.x, spot.y, { button: "right" });
    await expect(page.getByRole("menu", { name: "Canvas actions" })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.getByRole("menu", { name: "Canvas actions" })).toHaveCount(0);
    await onCanvas(page);
  });

  test("closes the shortcut sheet opened by keyboard, and stays on the page", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await surface(page).click({ position: { x: 8, y: 8 } });
    await page.keyboard.press("Shift+?");
    const sheet = page.getByRole("note", { name: "Canvas shortcuts" });
    await expect(sheet).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(sheet).toHaveCount(0);
    await onCanvas(page);
  });

  test("drops the selection and stays on the page", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await box(page, "Storage layer").click();
    await expect(page.locator(".react-flow__node.selected")).toHaveCount(1);

    await page.keyboard.press("Escape");
    await expect(page.locator(".react-flow__node.selected")).toHaveCount(0);
    await onCanvas(page);
  });

  test("leaves the page once the canvas has nothing left to close", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await surface(page).click({ position: { x: 8, y: 8 } });
    await page.keyboard.press("Escape");

    // Escape still means "back" — it just has to wait its turn.
    await expect(page.getByRole("heading", { level: 1, name: "Wiki" })).toBeVisible();
  });
});

test.describe("canvas suggestions", () => {
  test("opens on the user's own map and reveals the daemon's guesses on request", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    // A suggestion is not yet part of the map, so it is not on screen.
    await expect(box(page, "Vector search")).toHaveCount(0);
    await page.getByRole("button", { name: "1 suggestion" }).click();
    await expect(box(page, "Vector search")).toBeVisible();
  });

  test("accepts a suggestion into the map", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);
    await page.getByRole("button", { name: "1 suggestion" }).click();

    await box(page, "Vector search").getByRole("button", { name: "Accept" }).click();

    const patched = controller.calls().filter((c) => c.command === "patch_page_map_node") as {
      args: { body: { status?: string } };
    }[];
    expect(patched).toHaveLength(1);
    expect(patched[0].args.body.status).toBe("active");
  });

  test("dismisses a suggestion off the map", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);
    await page.getByRole("button", { name: "1 suggestion" }).click();

    await box(page, "Vector search").getByRole("button", { name: "Dismiss" }).click();

    const patched = controller.calls().filter((c) => c.command === "patch_page_map_node") as {
      args: { body: { status?: string } };
    }[];
    expect(patched[0].args.body.status).toBe("dismissed");
  });
});

test.describe("canvas viewport and keys", () => {
  test("the shortcut sheet opens from its button and from ?", async ({ page }) => {
    await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    const sheet = page.getByRole("note", { name: "Canvas shortcuts" });
    await page.getByRole("button", { name: "Canvas shortcuts" }).click();
    await expect(sheet).toBeVisible();
    await expect(sheet.getByText("Add child")).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(sheet).toHaveCount(0);

    await surface(page).click({ position: { x: 8, y: 8 } });
    await page.keyboard.press("Shift+?");
    await expect(sheet).toBeVisible();
  });

  test("Cmd-A selects every box and the arrow keys move what is selected", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    await surface(page).click({ position: { x: 8, y: 8 } });
    await page.keyboard.press("ControlOrMeta+a");
    await expect(page.locator(".react-flow__node.selected")).toHaveCount(4);

    const before = await box(page, "Storage layer").boundingBox();
    await page.keyboard.press("ArrowRight");
    await expect
      .poll(async () => (await box(page, "Storage layer").boundingBox())?.x)
      .not.toBe(before?.x);

    // Moving a box is a change to the map, so it has to reach the daemon.
    await expect
      .poll(() => controller.calls().filter((c) => c.command === "put_page_map_layout").length, {
        timeout: 10_000,
      })
      .toBeGreaterThan(0);
  });

  test("dragging a box keeps it where it was dropped and saves the position", async ({ page }) => {
    const controller = await installTauriMock(page, { locale: "en", rawActions: [] });
    await openCanvas(page);

    const target = box(page, "Query path");
    const from = await target.boundingBox();
    if (!from) throw new Error("box has no position");
    await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
    await page.mouse.down();
    await page.mouse.move(from.x + from.width / 2 + 120, from.y + from.height / 2 + 60, { steps: 6 });
    await page.mouse.up();

    await expect.poll(async () => (await target.boundingBox())?.x).not.toBe(from.x);
    await expect
      .poll(() => controller.calls().filter((c) => c.command === "put_page_map_layout").length, {
        timeout: 10_000,
      })
      .toBeGreaterThan(0);
    // And it must not snap back once the map is refetched.
    await page.waitForTimeout(1200);
    await expect.poll(async () => (await target.boundingBox())?.x).not.toBe(from.x);
  });
});
