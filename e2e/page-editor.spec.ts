// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test, type Locator, type Page } from "@playwright/test";
import type {
  Page as WenlanPage,
  UpdatePageInput,
  UpdatePageOutcome,
} from "../src/lib/tauri";
import { createSpacesNavigationFixture } from "./fixtures/spacesNavigation";
import {
  installTauriMock,
  type TauriMockController,
  type TauriMockPageScenario,
} from "./tauriMock";

const PAGE_ID = "page-editor-e2e";
const INITIAL_SOURCE = "# Browser editor fixture\n\nInitial source paragraph.\n";

function pageFixture(content = INITIAL_SOURCE): WenlanPage {
  return {
    id: PAGE_ID,
    title: "Browser editor fixture",
    summary: null,
    content,
    entity_id: null,
    domain: "Wenlan",
    space: "Wenlan",
    source_memory_ids: [],
    version: 7,
    status: "active",
    creation_kind: "authored",
    review_status: "confirmed",
    created_at: "2026-07-19T12:00:00.000Z",
    last_compiled: "2026-07-19T12:00:00.000Z",
    last_modified: "2026-07-19T12:00:00.000Z",
    user_edited: false,
  };
}

async function openPage(
  page: Page,
  fixture = pageFixture(),
  pageScenario?: TauriMockPageScenario,
): Promise<TauriMockController> {
  const defaults = createSpacesNavigationFixture();
  const controller = await installTauriMock(page, {
    locale: "en",
    rawActions: [],
    fixture: { ...defaults, pages: [fixture] },
    pageScenario,
  });
  await page.goto("/");
  await page
    .getByRole("button", { name: `Open ${fixture.title}`, exact: true })
    .click();
  await expect(page.getByRole("heading", { name: fixture.title })).toBeVisible();
  return controller;
}

async function beginEditing(page: Page): Promise<Locator> {
  await page.getByRole("button", { name: "Edit page", exact: true }).first().click();
  const editor = page.getByRole("textbox", { name: "Page editor" });
  await expect(editor).toBeVisible();
  return editor;
}

async function editorSource(editor: Locator): Promise<string> {
  return editor.evaluate((element) => {
    if (element instanceof HTMLTextAreaElement) return element.value;
    type CodeMirrorContent = HTMLElement & {
      cmTile?: {
        root?: {
          view?: {
            state: { doc: { toString(): string } };
          };
        };
      };
    };
    const content = (
      element.classList.contains("cm-content")
        ? element
        : element.querySelector(".cm-content")
    ) as CodeMirrorContent | null;
    const view = content?.cmTile?.root?.view;
    if (!view) throw new Error("CodeMirror view is unavailable");
    return view.state.doc.toString();
  });
}

test("uses one contextual editing view without source-mode controls", async ({ page }) => {
  await openPage(page);

  const editor = await beginEditing(page);

  await expect(
    page.getByRole("radio", { name: "Live Preview", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("radio", { name: "Source mode", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.locator('[data-editor-presentation-mode="writing"]'),
  ).toBeVisible();

  const heading = editor.locator(".cm-line").first();
  await editor.locator(".cm-line").last().click();
  await expect(heading).toHaveText("Browser editor fixture");
  await heading.click();
  await expect(heading).toHaveText("# Browser editor fixture");
});

test("formats through the toolbar and toggles a task without source loss", async ({ page }) => {
  const source = "alpha\n\n- [ ] verify exact source";
  await openPage(page, pageFixture(source));

  const editor = await beginEditing(page);
  const task = editor.locator(".cm-writing-task-checkbox");
  await expect(task).not.toBeChecked();
  await expect(task).toHaveAccessibleName("verify exact source");
  await task.focus();
  await task.press("Space");
  await expect(task).toBeChecked();
  await expect.poll(() => editorSource(editor)).toBe(
    "alpha\n\n- [x] verify exact source",
  );
  await expect(task).toBeFocused();

  await editor.locator(".cm-line").first().click();
  await editor.press("Home");
  await editor.press("Shift+End");
  await page.getByRole("button", { name: "Bold", exact: true }).click();

  await expect(editor).toBeFocused();
  await expect.poll(() => editorSource(editor)).toBe(
    "**alpha**\n\n- [x] verify exact source",
  );
});

test("saves the exact source through a typed CAS request and refetches it", async ({ page }) => {
  const controller = await openPage(page);
  const savedSource =
    "# Browser editor fixture\n\nSaved through the browser route exactly.  \n";
  const editor = await beginEditing(page);

  await editor.fill(savedSource);
  await expect.poll(() => editorSource(editor)).toBe(savedSource);
  await page.getByRole("button", { name: "Save", exact: true }).click();

  await expect(
    page.getByText("Saved through the browser route exactly."),
  ).toBeVisible();
  const updateCalls = controller.calls().filter(
    (call) => call.command === "update_page",
  );
  expect(updateCalls).toHaveLength(1);
  expect(updateCalls[0]?.args).toMatchObject({
    id: PAGE_ID,
    content: savedSource,
    expectedVersion: 7,
    callerId: "wenlan-app",
  } satisfies Omit<UpdatePageInput, "operationId">);
  expect((updateCalls[0]?.args as UpdatePageInput).operationId).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
  );

  const stored = await page.evaluate(async (id) =>
    window.__TAURI_INTERNALS__!.invoke("get_page", { id })
  , PAGE_ID) as WenlanPage;
  expect(stored).toMatchObject({
    id: PAGE_ID,
    content: savedSource,
    version: 8,
    user_edited: true,
  });
});

test("stateful mock replays exact receipts and rejects reused or stale writes", async ({ page }) => {
  await openPage(page);

  const outcomes = await page.evaluate(async (pageId) => {
    const invoke = window.__TAURI_INTERNALS__!.invoke;
    const request: UpdatePageInput = {
      id: pageId,
      content: "# Browser editor fixture\n\nFirst exact write.  \n",
      expectedVersion: 7,
      callerId: "wenlan-app",
      operationId: "11111111-1111-4111-8111-111111111111",
    };
    return {
      first: await invoke("update_page", request),
      replay: await invoke("update_page", request),
      changedReplay: await invoke("update_page", {
        ...request,
        content: `${request.content}changed`,
      }),
      stale: await invoke("update_page", {
        ...request,
        operationId: "22222222-2222-4222-8222-222222222222",
      }),
      page: await invoke("get_page", { id: pageId }),
    };
  }, PAGE_ID);

  expect(outcomes).toMatchObject({
    first: { outcome: "saved" },
    replay: { outcome: "saved" },
    changedReplay: {
      outcome: "conflict",
      message: "operation identity was already used for different page content",
    },
    stale: {
      outcome: "conflict",
      message: "expected version 7; current version is 8",
    },
    page: {
      id: PAGE_ID,
      content: "# Browser editor fixture\n\nFirst exact write.  \n",
      version: 8,
    },
  } satisfies {
    first: UpdatePageOutcome;
    replay: UpdatePageOutcome;
    changedReplay: UpdatePageOutcome;
    stale: UpdatePageOutcome;
    page: Partial<WenlanPage>;
  });
});

test("preserves the local draft when a remote write wins the CAS race", async ({ page }) => {
  const remoteSource = "# Browser editor fixture\n\nRemote writer won.\n";
  await openPage(page, pageFixture(), {
    firstWriteRemoteMutation: { pageId: PAGE_ID, content: remoteSource },
  });
  const editor = await beginEditing(page);
  const localDraft = "# Browser editor fixture\n\nMy unsaved local draft.\n";

  await editor.fill(localDraft);
  await page.getByRole("button", { name: "Save", exact: true }).click();

  const conflict = page.getByRole("alert");
  await expect(conflict).toContainText("This page changed elsewhere.");
  await expect(conflict).toContainText("Your local draft is still here.");
  await expect(page.getByText("Latest source (version 8)")).toBeVisible();
  await expect.poll(() => editorSource(editor)).toBe(localDraft);

  const stored = await page.evaluate(async (id) =>
    window.__TAURI_INTERNALS__!.invoke("get_page", { id })
  , PAGE_ID) as WenlanPage;
  expect(stored.content).toBe(remoteSource);
  expect(stored.version).toBe(8);
});

test("blocks editing below the daemon floor and records the typed diagnostic", async ({ page }) => {
  const controller = await openPage(page, pageFixture(), {
    daemonVersion: "0.14.0",
  });

  await page.getByRole("button", { name: "Edit page", exact: true }).first().click();

  await expect(page.getByRole("alert")).toContainText(
    "Page editing requires stable Wenlan daemon 0.14.1 or later. Running version: 0.14.0.",
  );
  await expect(page.getByRole("textbox", { name: "Page editor" })).toHaveCount(0);
  await expect.poll(() =>
    controller.calls().filter(
      (call) => call.command === "record_page_editor_diagnostic",
    ).map((call) => call.args)
  ).toEqual([{
    event: "daemon_floor_blocked",
    reportedVersion: "0.14.0",
    requiredFloor: "0.14.1",
  }]);
});

test("keeps the local draft when the daemon falls below the save floor", async ({ page }) => {
  await openPage(page, pageFixture(), {
    daemonVersion: "0.14.1",
    saveDaemonVersion: "0.14.0",
  });
  const editor = await beginEditing(page);
  const localDraft =
    "# Browser editor fixture\n\nUnsaved because the daemon was downgraded.\n";

  await editor.fill(localDraft);
  await page.getByRole("button", { name: "Save", exact: true }).click();

  await expect(page.getByRole("alert")).toContainText(
    "Page editing requires stable Wenlan daemon 0.14.1 or later. Running version: 0.14.0.",
  );
  await expect.poll(() => editorSource(editor)).toBe(localDraft);

  const stored = await page.evaluate(async (id) =>
    window.__TAURI_INTERNALS__!.invoke("get_page", { id })
  , PAGE_ID) as WenlanPage;
  expect(stored).toMatchObject({
    id: PAGE_ID,
    content: INITIAL_SOURCE,
    version: 7,
  });
});

test("falls back to the basic editor and saves exact source after module load failure", async ({ page }) => {
  const controller = await openPage(page);
  await page.route(
    "**/src/components/memory/editor/CodeMirrorMarkdownEditor.tsx*",
    (route) => route.abort("failed"),
  );

  const editor = await beginEditing(page);
  await expect(page.getByText("Basic editor active", { exact: true })).toBeVisible();
  await expect(editor).toHaveJSProperty("tagName", "TEXTAREA");
  await expect(page.getByRole("radio")).toHaveCount(0);
  await expect(page.getByRole("group", { name: "Formatting" })).toHaveCount(0);

  const fallbackSource =
    "# Browser editor fixture\n\nSaved exactly through the basic fallback.\n";
  await editor.fill(fallbackSource);
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(
    page.getByText("Saved exactly through the basic fallback."),
  ).toBeVisible();
  await expect.poll(() =>
    controller.calls().filter(
      (call) => call.command === "record_page_editor_diagnostic",
    ).map((call) => call.args)
  ).toEqual([{ event: "editor_fallback", reason: "load" }]);
});

test("gives a long document main-content scrolling while persistence actions remain usable", async ({ page }) => {
  const longSource = [
    "# Browser editor fixture",
    "",
    ...Array.from({ length: 180 }, (_, index) =>
      `${index + 1}. Long source line ${index + 1} stays readable and editable.`
    ),
  ].join("\n");
  await page.setViewportSize({ width: 1280, height: 900 });
  await openPage(page, pageFixture(longSource));

  const main = page.locator("main.memory-main-content");
  await main.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  await expect.poll(() => main.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(0);
  await main.evaluate((element) => {
    element.scrollTop = 0;
  });

  const editor = await beginEditing(page);
  await expect(page.getByRole("radio")).toHaveCount(0);
  const save = page.getByRole("button", { name: "Save", exact: true });
  const cancel = page.getByRole("button", { name: "Cancel", exact: true });
  await expect(save).toBeVisible();
  await expect(save).toBeEnabled();
  await expect(cancel).toBeVisible();
  await expect(cancel).toBeEnabled();

  const scroller = page.locator(
    '[data-markdown-editor-engine="codemirror"] .cm-scroller',
  );
  const geometry = await scroller.evaluate((element) => {
    element.scrollTop = 128;
    const style = window.getComputedStyle(element);
    return {
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      overflowY: style.overflowY,
      scrollTop: element.scrollTop,
    };
  });
  expect(geometry.overflowY).toBe("visible");
  expect(geometry.scrollHeight).toBeLessThanOrEqual(geometry.clientHeight + 1);
  expect(geometry.scrollTop).toBe(0);

  await main.evaluate((element) => {
    element.scrollTop = 240;
  });
  await expect.poll(() => main.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(0);
  await expect(save).toBeVisible();
  await expect(cancel).toBeVisible();

  await save.click();
  await expect(editor).toHaveCount(0);
  await beginEditing(page);
  await cancel.click();
  await expect(page.getByRole("textbox", { name: "Page editor" })).toHaveCount(0);
});

test("keeps persistence actions visible at 375px while the format row owns horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openPage(page);
  const editor = await beginEditing(page);
  const save = page.getByRole("button", { name: "Save", exact: true });
  const cancel = page.getByRole("button", { name: "Cancel", exact: true });
  const formatRow = page.getByRole("group", { name: "Formatting", exact: true });

  await expect(save).toBeVisible();
  await expect(cancel).toBeVisible();
  const persistenceBounds = await Promise.all([save.boundingBox(), cancel.boundingBox()]);
  for (const bounds of persistenceBounds) {
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(375);
  }

  const formatGeometry = await formatRow.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
    overflowX: window.getComputedStyle(element).overflowX,
  }));
  expect(formatGeometry.scrollWidth).toBeGreaterThan(formatGeometry.clientWidth);
  expect(formatGeometry.overflowX).toBe("auto");
  expect(await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))).toEqual({ clientWidth: 375, scrollWidth: 375 });

  const editorBox = await editor.boundingBox();
  expect(editorBox).not.toBeNull();
  expect(editorBox!.x).toBeGreaterThanOrEqual(0);
  expect(editorBox!.x + editorBox!.width).toBeLessThanOrEqual(375);
});

test("renders a focused opaque accent border in the dark editor", async ({ page }) => {
  await openPage(page);
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));

  const editor = await beginEditing(page);
  const editorSurface = page.locator('[data-markdown-editor-engine="codemirror"] .cm-editor');
  const toolbar = page.getByRole("toolbar", { name: "Formatting", exact: true });
  await expect(editorSurface).toBeVisible();
  await expect(toolbar).toBeVisible();

  await editor.focus();
  await expect(editor).toBeFocused();
  await expect(editorSurface).toHaveClass(/cm-focused/);
  expect(await editorSurface.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return {
      accent: window.getComputedStyle(document.documentElement)
        .getPropertyValue("--mem-accent-page")
        .trim(),
      borderTopColor: style.borderTopColor,
    };
  })).toEqual({
    accent: "#8FB3EA",
    borderTopColor: "rgb(143, 179, 234)",
  });
});

test("uses opaque light-theme focus indicators on editor controls and tasks", async ({ page }) => {
  await openPage(
    page,
    pageFixture("# Browser editor fixture\n\n- [ ] Keyboard task\n"),
  );
  const editor = await beginEditing(page);
  const bold = page.getByRole("button", { name: "Bold", exact: true });
  const italic = page.getByRole("button", { name: "Italic", exact: true });
  const task = editor.locator(".cm-writing-task-checkbox");

  await bold.focus();
  await page.keyboard.press("Tab");
  await expect(italic).toBeFocused();
  expect(await italic.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return {
      accent: window.getComputedStyle(document.documentElement)
        .getPropertyValue("--mem-accent-page")
        .trim(),
      outlineColor: style.outlineColor,
    };
  })).toEqual({
    accent: "#5E58C8",
    outlineColor: "rgb(94, 88, 200)",
  });

  await task.focus();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Shift+Tab");
  await expect(task).toBeFocused();
  expect(await task.evaluate((element) =>
    window.getComputedStyle(element).outlineColor
  )).toBe("rgb(94, 88, 200)");
});

test("gives a long document most of the available desktop height", async ({ page }) => {
  const longSource = Array.from(
    { length: 180 },
    (_, index) => `${index + 1}. Long source line ${index + 1}.`,
  ).join("\n");
  await page.setViewportSize({ width: 1280, height: 900 });
  await openPage(page, pageFixture(longSource));
  await beginEditing(page);

  const scroller = page.locator(
    '[data-markdown-editor-engine="codemirror"] .cm-scroller',
  );
  await expect.poll(() =>
    scroller.evaluate((element) => element.clientHeight)
  ).toBeGreaterThanOrEqual(600);
});
