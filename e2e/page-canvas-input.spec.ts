// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Page } from "@playwright/test";
import { installTauriMock } from "./tauriMock";

/**
 * Where the canvas's pointer and keyboard input actually lands.
 *
 * The map is a graphical surface inside a document that is not, and the seam
 * between them is where input goes wrong: a drag that misses the pane starts a
 * text selection, and a keystroke that misses the surface reaches the browser
 * instead of the map.
 */
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

async function marquees(page: Page): Promise<number> {
  return page.locator(".react-flow__selection").count();
}

test("a rubber-band drag leaves nothing behind", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  const pane = page.locator(".react-flow__pane");
  const area = (await pane.boundingBox())!;

  // Four drags anchored near the bottom-right, each reaching further into the
  // top-left — the gesture from the recording.
  for (let i = 1; i <= 4; i++) {
    await page.mouse.move(area.x + area.width - 40, area.y + area.height - 30);
    await page.mouse.down();
    await page.mouse.move(area.x + area.width - 40 - i * 90, area.y + area.height - 30 - i * 60, {
      steps: 8,
    });
    await page.mouse.up();
    await page.waitForTimeout(120);
    console.log(`after drag ${i}: ${await marquees(page)} marquee(s) in the DOM`);
  }
  await page.locator(".page-canvas").screenshot({ path: "shots/marquee-after.png" });
  expect(await marquees(page)).toBe(0);
});

test("select all picks the boxes, not the page text", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  // Deliberately no click into the canvas first: opening it leaves focus on the
  // Canvas button in the page header, which is where a reader's first keystroke
  // actually lands.
  console.log(
    `focus after opening: ${await page.evaluate(
      () => `${document.activeElement?.tagName}.${document.activeElement?.className}`,
    )}`,
  );
  await page.keyboard.press("Meta+a");
  await page.waitForTimeout(200);
  const selectedText = await page.evaluate(() => window.getSelection()?.toString() ?? "");
  const selectedNodes = await page.locator(".react-flow__node.selected").count();
  console.log(
    `meta+a → ${selectedNodes} node(s) selected, ${selectedText.length} chars of page text highlighted`,
  );
  await page.locator(".page-canvas").screenshot({ path: "shots/select-all.png" });
  expect(selectedText, "meta+a highlighted page text instead of selecting boxes").toBe("");
  expect(selectedNodes).toBeGreaterThan(1);
});

test("a new box stays where it was dropped", async ({ page }) => {
  // A slow map read, or the refetch lands before the box can be measured where
  // it was dropped and the jump is invisible to the test.
  const controller = await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    delays: { get_page_map: 1200 },
  });
  await openCanvas(page);
  const pane = (await page.locator(".react-flow__pane").boundingBox())!;
  const spot = { x: pane.x + pane.width - 110, y: pane.y + pane.height - 130 };

  await page.mouse.dblclick(spot.x, spot.y);
  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  await field.fill("Deploy notes");
  await field.press("Enter");

  const made = page.locator(".react-flow__node").filter({ hasText: "Deploy notes" }).first();
  await expect(made).toBeVisible();
  const center = async () => {
    const r = (await made.boundingBox())!;
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  };
  const dropped = await center();
  // Guard the guard: if this is already the settled position, the test is
  // measuring after the jump and would pass with the bug in.
  expect(
    Math.hypot(dropped.x - spot.x, dropped.y - spot.y),
    "the held box is not where the double-click was",
  ).toBeLessThan(130);

  // Wait out the whole create: heading write, node create, position write, and
  // the map refetch that replaces the held box with the server's copy. That
  // refetch is the moment the box used to jump, because a node the daemon
  // reports as unplaced is given a computed slot on the ring.
  await expect
    .poll(() => controller.calls().filter((c) => c.command === "get_page_map").length, {
      timeout: 15_000,
    })
    .toBeGreaterThan(1);
  await page.waitForTimeout(1500);

  const settled = await center();
  const moved = Math.hypot(settled.x - dropped.x, settled.y - dropped.y);
  console.log(
    `dropped at (${Math.round(dropped.x)},${Math.round(dropped.y)}), settled at (${Math.round(settled.x)},${Math.round(settled.y)}) — moved ${Math.round(moved)}px`,
  );
  // The box grows to fit its label, so its edges shift a little; its center is
  // what the drop placed and what must hold.
  expect(moved, "the new box jumped after it was created").toBeLessThan(24);
});

test("select all from outside the canvas cannot paint over the map", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  // Focus back out to the page around the canvas, the way clicking a header
  // control leaves it, then select all. The browser's own select-all is allowed
  // to highlight the document — it must not reach inside the map, where it
  // paints an opaque block over everything.
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.keyboard.press("Meta+a");
  const selected = await page.evaluate(() => window.getSelection()?.toString() ?? "");
  await page.locator(".page-canvas").screenshot({ path: "shots/select-all-unfocused.png" });
  for (const inside of ["Improve", "suggestion", "Storage layer"]) {
    expect(selected, `select-all reached "${inside}" inside the canvas`).not.toContain(inside);
  }
});

test("the grow dot sits on the side each box faces", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  // A map big enough to put boxes in all four quadrants: the fixture's four
  // boxes sit on one axis, which would leave the up and down cases unchecked.
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
    await add("n_root", "Observability");
    await add("n_query", "Reranking");
  });
  await openCanvas(page);
  await page.waitForTimeout(600);
  await page.keyboard.press("Meta+a");
  await page.locator(".page-canvas").screenshot({ path: "shots/grow-dots.png" });

  // Every box's dot, against the direction that box sits from the page in the
  // middle. On a radial map those directions differ, so a dot pinned to one
  // side is pointing back at the parent for half of them.
  const dots = await page.locator(".react-flow__node").evaluateAll((els) => {
    const root = els.find((el) => el.textContent?.includes("Fixture architecture"));
    const c = (el: Element) => {
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    };
    const origin = c(root!);
    return els
      .filter((el) => el !== root)
      .map((el) => {
        const handle = el.querySelector(".page-canvas-grow")!;
        const side = ["top", "right", "bottom", "left"].find((s) =>
          handle.classList.contains(`react-flow__handle-${s}`),
        );
        const p = c(el);
        return {
          label: (el.textContent ?? "").slice(0, 24),
          side,
          dx: Math.round(p.x - origin.x),
          dy: Math.round(p.y - origin.y),
        };
      });
  });
  console.log(JSON.stringify(dots, null, 1));
  expect(dots.length).toBeGreaterThan(1);
  for (const d of dots) {
    const want =
      Math.abs(d.dx) > Math.abs(d.dy)
        ? d.dx > 0
          ? "right"
          : "left"
        : d.dy > 0
          ? "bottom"
          : "top";
    expect(d.side, `${d.label} sits at (${d.dx},${d.dy}) but its dot is on the ${d.side}`).toBe(
      want,
    );
  }
});

/** The scale the viewport is actually drawn at, read off its own transform. */
async function scale(page: Page): Promise<number> {
  return page.evaluate(() => {
    const vp = document.querySelector(".react-flow__viewport") as HTMLElement;
    return new DOMMatrixReadOnly(getComputedStyle(vp).transform).a;
  });
}

/** A map the size a real page grows to: thirteen boxes over three rings. */
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

test("the map opens at a scale a person can read", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  await page.waitForTimeout(600);
  const small = await scale(page);
  const smallBox = (await page.locator(".react-flow__node").first().boundingBox())!;
  console.log(`4 boxes: opened at ${small.toFixed(3)}, box ${Math.round(smallBox.width)}px wide`);
  // A four-box map used to open at 1.167, drawing every box larger than it was
  // designed. Framing the extent is not worth rendering the design wrong.
  expect(small, "a sparse map opened zoomed past actual size").toBeLessThanOrEqual(1.001);

  await page.goto("/");
  await seedLargeMap(page);
  await openCanvas(page);
  await page.waitForTimeout(600);
  const large = await scale(page);
  console.log(`13 boxes: opened at ${large.toFixed(3)}, 12px label rasterizes at ${(12 * large).toFixed(1)}px`);
  // And a thirteen-box map used to open at 0.563, which draws the 12px label at
  // under 7px. 9px rendered is the derived floor — the smallest defensible size
  // for UI text — and the rest of the map is one pan or one Shift 1 away.
  expect(12 * large, "labels opened too small to read").toBeGreaterThanOrEqual(8.99);
  expect(large, "a dense map opened zoomed past actual size").toBeLessThanOrEqual(1.001);
});

test("the zoom keys work, and the zoom buttons say what they are", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  await page.waitForTimeout(600);
  // Focus stays on the Canvas button in the page header, which is where a
  // reader's first keystroke actually lands.
  const opened = await scale(page);

  await page.keyboard.press("Minus");
  await page.waitForTimeout(400);
  const out = await scale(page);
  await page.keyboard.press("Equal");
  await page.keyboard.press("Equal");
  await page.waitForTimeout(400);
  const inAgain = await scale(page);
  await page.keyboard.press("Shift+Digit0");
  await page.waitForTimeout(400);
  const actual = await scale(page);
  console.log(`opened ${opened.toFixed(3)} → "−" ${out.toFixed(3)} → "++" ${inAgain.toFixed(3)} → "Shift 0" ${actual.toFixed(3)}`);
  expect(out, '"−" did not zoom out').toBeLessThan(opened - 0.01);
  expect(inAgain, '"+" did not zoom back in').toBeGreaterThan(out + 0.01);
  expect(actual, '"Shift 0" did not return to actual size').toBeCloseTo(1, 2);

  // Cmd+plus is the app window's own page zoom, so the canvas keys are bare —
  // which makes them undiscoverable unless the control someone already reached
  // for names them.
  const titles = await page.evaluate(() =>
    [".react-flow__controls-zoomin", ".react-flow__controls-zoomout", ".react-flow__controls-fitview"].map(
      (sel) => document.querySelector(sel)?.getAttribute("title") ?? "",
    ),
  );
  console.log(`control tooltips: ${JSON.stringify(titles)}`);
  expect(titles[0], "the zoom-in button does not name its key").toContain("(+)");
  expect(titles[1], "the zoom-out button does not name its key").toContain("(-)");
  expect(titles[2], "the fit button does not name its key").toContain("Shift 1");
});

test("a real-sized page opens with all of it on screen", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  await seedLargeMap(page);
  await openCanvas(page);
  await page.waitForTimeout(800);

  const view = await page.evaluate(() => {
    const frame = document.querySelector(".page-canvas-surface")!.getBoundingClientRect();
    const boxes = [...document.querySelectorAll(".react-flow__node")].map((el) => ({
      r: el.getBoundingClientRect(),
      label: (el.textContent ?? "").slice(0, 20),
    }));
    return {
      nodes: boxes.length,
      clipped: boxes
        .filter(
          ({ r }) =>
            r.left < frame.left - 1 ||
            r.right > frame.right + 1 ||
            r.top < frame.top - 1 ||
            r.bottom > frame.bottom + 1,
        )
        .map(({ label }) => label),
    };
  });
  console.log(
    `13-box map at ${(await scale(page)).toFixed(3)}: ${view.clipped.length} clipped ${JSON.stringify(view.clipped)}`,
  );
  await page.locator(".page-canvas").screenshot({ path: "shots/opens-complete.png" });
  expect(view.nodes).toBeGreaterThan(10);
  // Circular rings made a roughly square map, which in a frame near 1.8:1 could
  // only open small or open cut. Squashing the rings toward the frame's own shape
  // is what lets a page this size open both readable and whole.
  expect(view.clipped, "boxes were cut off by the frame on open").toEqual([]);
});

test("fit all frames the map the same way however it is asked for", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });

  // Sparse first, and it has to stay first: seeded boxes outlive a `goto`, so a
  // four-box check after `seedLargeMap` silently measures thirteen.
  await openCanvas(page);
  await page.waitForTimeout(600);
  await page.locator(".react-flow__pane").click({ position: { x: 30, y: 30 } });
  await page.keyboard.press("Shift+Digit1");
  await page.waitForTimeout(400);
  const sparse = await scale(page);
  const sparseBox = (await page.locator(".react-flow__node").first().boundingBox())!;
  console.log(`4 boxes: fit all at ${sparse.toFixed(3)}, box ${Math.round(sparseBox.width)}px wide`);
  expect(await page.locator(".react-flow__node").count()).toBeLessThan(6);
  // Fitting a sparse page to the frame means magnifying it to 1.167 — drawing
  // every box half again larger than it was designed.
  expect(sparse, "fit all magnified a sparse map past actual size").toBeLessThanOrEqual(1.001);

  await page.goto("/");
  await seedLargeMap(page);
  await openCanvas(page);
  await page.waitForTimeout(700);
  const opened = await scale(page);

  // The key, then the button, then the key again from a nudged viewport: one
  // command asked for three ways has to land in the same place every time. Each
  // reached React Flow's bare defaults separately before, so opening a page and
  // then pressing fit moved the map for no reason a reader could see.
  await page.locator(".react-flow__pane").click({ position: { x: 30, y: 30 } });
  await page.keyboard.press("Shift+Digit1");
  await page.waitForTimeout(400);
  const byKey = await scale(page);

  await page.keyboard.press("Equal");
  await page.waitForTimeout(300);
  await page.locator(".react-flow__controls-fitview").click();
  await page.waitForTimeout(400);
  const byButton = await scale(page);
  console.log(`13 boxes: opened ${opened.toFixed(3)}, key ${byKey.toFixed(3)}, button ${byButton.toFixed(3)}`);

  expect(byKey, "the fit key reframed a map that was already fitted").toBeCloseTo(opened, 2);
  expect(byButton, "the fit button and the fit key disagree").toBeCloseTo(byKey, 2);
});

test("fit all shows the whole map, however big it is", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");
  // Bigger than the thirteen-box fixture on purpose: React Flow's default zoom
  // floor is 0.5, and a map only reveals that floor once fitting it needs less.
  await page.evaluate(async () => {
    const add = async (parent: string, label: string): Promise<string> => {
      const result = (await window.__wenlanTauriInvoke("create_page_map_node", {
        pageId: "page-architecture",
        body: { parent_id: parent, label, ref_kind: "section", ref_id: label },
      })) as { node: { id: string } };
      return result.node.id;
    };
    for (let b = 0; b < 6; b++) {
      const branch = await add("n_root", `Subsystem number ${b}`);
      for (let c = 0; c < 3; c++) await add(branch, `Component ${b}.${c} detail`);
    }
  });
  await openCanvas(page);
  await page.waitForTimeout(800);
  await page.locator(".react-flow__pane").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("Shift+Digit1");
  await page.waitForTimeout(700);

  const fit = await page.evaluate(() => {
    const frame = document.querySelector(".page-canvas-surface")!.getBoundingClientRect();
    const boxes = [...document.querySelectorAll(".react-flow__node")].map((el) =>
      el.getBoundingClientRect(),
    );
    return {
      nodes: boxes.length,
      outside: boxes.filter(
        (b) =>
          b.left < frame.left - 1 ||
          b.right > frame.right + 1 ||
          b.top < frame.top - 1 ||
          b.bottom > frame.bottom + 1,
      ).length,
    };
  });
  console.log(
    `Shift 1 on ${fit.nodes} boxes at scale ${(await scale(page)).toFixed(3)}: ${fit.outside} outside the frame`,
  );
  expect(fit.nodes).toBeGreaterThan(20);
  // The opening view has its own legibility floor. "Fit all" has no business
  // inheriting one — it is the command that means "show me everything".
  expect(fit.outside, "fit all left boxes outside the frame").toBe(0);
});

test("the zoom keys stay out of the box being renamed", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  await page.waitForTimeout(600);
  const before = await scale(page);

  await page.locator(".react-flow__node").filter({ hasText: "Storage layer" }).first().click();
  await page.keyboard.press("F2");
  const field = page.getByRole("textbox", { name: "Section name" });
  await expect(field).toBeVisible();
  // The key handler runs on the document in the capture phase, so it sees these
  // presses before the field's own handler can stop them — the guard on the
  // event target is the only thing between a hyphen and the viewport moving.
  await field.press("Minus");
  await field.press("Shift+Digit0");
  await page.waitForTimeout(400);
  const typed = await field.inputValue();
  const after = await scale(page);
  console.log(`renaming: field now ${JSON.stringify(typed)}, scale ${before.toFixed(3)} → ${after.toFixed(3)}`);
  expect(after, "a keystroke meant for the name field moved the map").toBeCloseTo(before, 3);
  expect(typed, "the name field did not receive the keystrokes").toContain("-");
});

test("the zoom keys leave the window's own zoom alone, and work off a US layout", async ({
  page,
}) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  await page.waitForTimeout(600);
  const opened = await scale(page);

  // The app window has `zoomHotkeysEnabled`, so Cmd+plus is the webview's page
  // zoom. If the canvas took it too, one press would resize the whole app UI
  // and the map at once.
  await page.keyboard.press("Meta+Equal");
  await page.keyboard.press("Control+Minus");
  await page.waitForTimeout(400);
  const afterModified = await scale(page);
  console.log(`Cmd+= / Ctrl+− : ${opened.toFixed(3)} → ${afterModified.toFixed(3)}`);
  expect(afterModified, "a modified press zoomed the map as well as the window").toBeCloseTo(
    opened,
    2,
  );

  // On a German layout the key printed with a + reports code "BracketRight",
  // so matching the physical US position alone leaves those readers with no
  // zoom key at all. Playwright cannot type a layout it is not using, but the
  // handler is an ordinary document listener, so the press itself is real.
  const german = (key: string, code: string) =>
    page.evaluate(
      ([k, c]) =>
        document.dispatchEvent(
          new KeyboardEvent("keydown", { key: k, code: c, bubbles: true, cancelable: true }),
        ),
      [key, code],
    );
  await german("+", "BracketRight");
  await page.waitForTimeout(400);
  const germanIn = await scale(page);
  await german("-", "Slash");
  await page.waitForTimeout(400);
  const germanOut = await scale(page);
  console.log(
    `German +/− : ${afterModified.toFixed(3)} → ${germanIn.toFixed(3)} → ${germanOut.toFixed(3)}`,
  );
  expect(germanIn, 'the German "+" key did not zoom in').toBeGreaterThan(afterModified + 0.01);
  expect(germanOut, 'the German "-" key did not zoom out').toBeLessThan(germanIn - 0.01);
});

test("a drag that starts on a canvas control selects no text", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await openCanvas(page);
  const surface = page.locator(".page-canvas-surface");
  const area = (await surface.boundingBox())!;

  for (const [name, from] of [
    ["help button (bottom right)", page.getByRole("button", { name: "Canvas shortcuts" })],
    ["suggestion badge (top right)", page.getByRole("button", { name: /suggestion/ })],
  ] as const) {
    const start = (await from.boundingBox())!;
    await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
    await page.mouse.down();
    await page.mouse.move(area.x + 60, area.y + 40, { steps: 12 });
    await page.mouse.up();
    const selected = await page.evaluate(() => window.getSelection()?.toString() ?? "");
    console.log(`${name}: selected ${JSON.stringify(selected.slice(0, 70))}`);
    await page
      .locator(".page-canvas")
      .screenshot({ path: `shots/drag-from-${name.split(" ")[0]}.png` });
    expect(selected, `dragging from the ${name} selected page text`).toBe("");
  }
});
