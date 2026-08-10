// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { installTauriMock } from "./tauriMock";

/**
 * The canvas's control surface in every state, both themes, photographed to
 * shots/ so the design can be looked at rather than imagined.
 *
 * It carries the coverage those states have nowhere else — waiting and failing
 * are only reachable with a stalled or broken daemon — plus the one check the
 * pure layout tests cannot make: that boxes measured by the real browser, with
 * the real font, still land clear of each other.
 */
const THEMES = ["light", "dark"] as const;

async function openCanvas(page: Page): Promise<void> {
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Primary navigation" })
    .getByRole("button", { name: "Wiki", exact: true })
    .click();
  await page.getByRole("button", { name: "Open Fixture architecture" }).click();
  await page.getByRole("button", { name: "Canvas" }).click();
  await expect(page.getByRole("region", { name: "Canvas for Fixture architecture" })).toBeVisible();
}

function box(page: Page, label: string) {
  return page.locator(".react-flow__node").filter({ hasText: label }).first();
}

/**
 * A map the size a real page grows to. Four boxes cannot show whether a radial
 * layout is a mind map or a flowchart, so seed the mock straight through its own
 * command surface before the canvas ever asks for it.
 */
async function seedLargeMap(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const add = async (parent: string, label: string): Promise<string> => {
      const result = (await window.__wenlanTauriInvoke("create_page_map_node", {
        pageId: "page-architecture",
        body: { parent_id: parent, label, ref_kind: "section", ref_id: label },
      })) as { node: { id: string } };
      return result.node.id;
    };
    const ingest = await add("n_root", "Ingest pipeline");
    await add(ingest, "Chunking");
    await add(ingest, "Embedding queue");
    const api = await add("n_root", "HTTP surface");
    await add(api, "Auth");
    await add(api, "Rate limits");
    await add("n_root", "Observability");
    await add("n_query", "Reranking");
    await add("n_query", "Filters");
  });
}

for (const theme of THEMES) {
  test.describe(`canvas look — ${theme}`, () => {
    test.use({ colorScheme: theme });

    test("resting, suggestions revealed, sheet, menu, notice, naming", async ({ page }) => {
      await installTauriMock(page, {
        locale: "en",
        rawActions: [],
        localStorage: { "wenlan-theme": theme },
      });
      await openCanvas(page);
      const surface = page.locator(".page-canvas");
      const shot = (name: string) =>
        surface.screenshot({ path: `shots/${theme}-${name}.png` });

      await page.waitForTimeout(400);
      await shot("1-resting");
      await page.screenshot({ path: `shots/${theme}-1-resting-page.png` });

      await page.getByRole("button", { name: /suggestion/ }).click();
      await box(page, "Vector search").hover();
      await page.waitForTimeout(200);
      await shot("2-suggestions");

      await page.getByRole("button", { name: "Canvas shortcuts" }).click();
      await page.waitForTimeout(200);
      await shot("3-shortcuts");
      await page.keyboard.press("Escape");

      await box(page, "Storage layer").click({ button: "right" });
      await page.waitForTimeout(200);
      await shot("4-node-menu");
      await page.keyboard.press("Escape");

      await box(page, "Fixture architecture").click();
      await page.keyboard.press("Delete");
      await page.waitForTimeout(300);
      await shot("5-notice");

      // The other notice, and the one that matters: a real delete going through.
      // Backspace is the delete key on a Mac, it takes the box's whole subtree,
      // and the daemon's tombstones are terminal — so this is the most expensive
      // keystroke here. Until this landed it was also the quietest: the success
      // handler clears the notice area, so permanent destruction said nothing.
      await box(page, "Storage layer").click();
      await page.keyboard.press("Backspace");
      await expect(
        page.getByText("Deleted “Storage layer” and the 1 box inside it. This can't be undone."),
      ).toBeVisible();
      await shot("5-deleted");

      await page.locator(".react-flow__pane").dblclick({ position: { x: 120, y: 90 } });
      await page.waitForTimeout(300);
      await shot("6-naming");
    });

    test("a map the size of a real page", async ({ page }) => {
      await installTauriMock(page, {
        locale: "en",
        rawActions: [],
        localStorage: { "wenlan-theme": theme },
      });
      await page.goto("/");
      await seedLargeMap(page);
      await openCanvas(page);
      await page.waitForTimeout(600);
      await page.locator(".page-canvas").screenshot({ path: `shots/${theme}-9-large.png` });

      // Every box as the browser actually measured it: no two may overlap, and
      // the map has to use the wide axis. Two branches once landed at 90 and 270
      // degrees, which is a legal radial layout and an unreadable mind map.
      const boxes = await page.locator(".react-flow__node").evaluateAll((els) =>
        els.map((el) => el.getBoundingClientRect()).map((r) => ({
          left: r.left, right: r.right, top: r.top, bottom: r.bottom,
        })),
      );
      expect(boxes.length).toBeGreaterThan(10);
      for (let i = 0; i < boxes.length; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
          const a = boxes[i];
          const b = boxes[j];
          const overlaps =
            a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
          expect(overlaps, `box ${i} overlaps box ${j}`).toBe(false);
        }
      }
      const spanX = Math.max(...boxes.map((b) => b.right)) - Math.min(...boxes.map((b) => b.left));
      const spanY = Math.max(...boxes.map((b) => b.bottom)) - Math.min(...boxes.map((b) => b.top));
      expect(spanX).toBeGreaterThan(spanY);
      await page.getByRole("button", { name: /suggestion/ }).click();
      await page.waitForTimeout(400);
      await page.locator(".page-canvas").screenshot({ path: `shots/${theme}-9-large-suggested.png` });
    });

    test("the strokes between boxes are curves, not wires", async ({ page }) => {
      await installTauriMock(page, {
        locale: "en",
        rawActions: [],
        localStorage: { "wenlan-theme": theme },
      });
      await page.goto("/");
      await seedLargeMap(page);
      await openCanvas(page);
      await page.waitForTimeout(600);

      // Each stroke measured against its own chord: how far the drawn path bows
      // away from the straight line between the two boxes, as a fraction of the
      // distance between them. A straight edge — or React Flow's own bezier,
      // which on center-pinned Top/Bottom handles lays an S over the spoke —
      // does not produce a single consistent bow.
      const bows = await page.locator(".react-flow__edge-path").evaluateAll((els) =>
        els.map((el) => {
          const path = el as unknown as SVGPathElement;
          const len = path.getTotalLength();
          const a = path.getPointAtLength(0);
          const b = path.getPointAtLength(len);
          const mid = path.getPointAtLength(len / 2);
          const span = Math.hypot(b.x - a.x, b.y - a.y);
          const off = Math.hypot(mid.x - (a.x + b.x) / 2, mid.y - (a.y + b.y) / 2);
          return span > 1 ? off / span : 0;
        }),
      );
      console.log(`bow per edge: ${bows.map((b) => b.toFixed(3)).join(", ")}`);
      expect(bows.length).toBeGreaterThan(10);
      for (const bow of bows) {
        expect(bow, "a stroke was drawn straight").toBeGreaterThan(0.02);
        // And not so far that it leans into a neighbouring box on a crowded ring.
        expect(bow, "a stroke bowed far enough to cross the map").toBeLessThan(0.08);
      }
    });

    test("loading", async ({ page }) => {
      await installTauriMock(page, {
        locale: "en",
        rawActions: [],
        localStorage: { "wenlan-theme": theme },
        delays: { get_page_map: 6000 },
      });
      // Deliberately NOT openCanvas: it waits for the map region, which is the
      // one thing that does not exist while the canvas is still loading.
      await page.goto("/");
      await page
        .getByRole("navigation", { name: "Primary navigation" })
        .getByRole("button", { name: "Wiki", exact: true })
        .click();
      await page.getByRole("button", { name: "Open Fixture architecture" }).click();
      await page.getByRole("button", { name: "Canvas" }).click();
      await expect(page.getByText("Loading canvas")).toBeVisible();
      await page.screenshot({ path: `shots/${theme}-7-loading.png` });
    });

    test("load failure", async ({ page }) => {
      await installTauriMock(page, {
        locale: "en",
        rawActions: [],
        localStorage: { "wenlan-theme": theme },
        failures: [{ command: "get_page_map", message: "daemon unreachable", times: 99 }],
      });
      await page.goto("/");
      await page
        .getByRole("navigation", { name: "Primary navigation" })
        .getByRole("button", { name: "Wiki", exact: true })
        .click();
      await page.getByRole("button", { name: "Open Fixture architecture" }).click();
      await page.getByRole("button", { name: "Canvas" }).click();
      await expect(page.getByText("Could not load the canvas")).toBeVisible();
      await page.screenshot({ path: `shots/${theme}-8-error.png` });
    });
  });
}
