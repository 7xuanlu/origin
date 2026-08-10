// SPDX-License-Identifier: AGPL-3.0-only
import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import PageDetail from "./PageDetail";
import {
  editorViewFromTextbox,
  installCodeMirrorDomPolyfills,
  pressKey,
  replaceDocument,
  selectRange,
} from "./editor/editorTestUtils";

const tauriMocks = vi.hoisted(() => ({
  getPage: vi.fn(),
  getPageSources: vi.fn(),
  listRegisteredSources: vi.fn(),
  getPageLinks: vi.fn(),
  getPageRevisions: vi.fn(),
  getEntityDetail: vi.fn(),
  redistillPage: vi.fn(),
  updatePage: vi.fn(),
  getDaemonVersion: vi.fn(),
  getSystemInfo: vi.fn(),
  recordPageEditorDiagnostic: vi.fn(),
  deletePage: vi.fn(),
  clipboardWrite: vi.fn(),
  exportPageToObsidian: vi.fn(),
}));

vi.mock("../../lib/tauri", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/tauri")>()),
  ...tauriMocks,
}));

const PAGE = {
  id: "page-codemirror-integration",
  title: "Real CodeMirror page",
  summary: null,
  content: "Original exact source.\n",
  entity_id: null,
  domain: "testing",
  source_memory_ids: [],
  version: 7,
  status: "active",
  created_at: "2026-07-20T00:00:00+00:00",
  last_compiled: "2026-07-20T00:00:00+00:00",
  last_modified: "2026-07-20T00:00:00+00:00",
};

beforeAll(() => {
  installCodeMirrorDomPolyfills();
});

beforeEach(() => {
  vi.clearAllMocks();
  tauriMocks.getPage.mockResolvedValue(PAGE);
  tauriMocks.getPageSources.mockResolvedValue([]);
  tauriMocks.listRegisteredSources.mockResolvedValue([]);
  tauriMocks.getPageLinks.mockResolvedValue({ outbound: [], inbound: [] });
  tauriMocks.getPageRevisions.mockResolvedValue({
    page_id: PAGE.id,
    current_version: PAGE.version,
    user_edited: false,
    stale_reason: null,
    entries: [],
  });
  tauriMocks.redistillPage.mockResolvedValue({ status: "ok", updated: true });
  tauriMocks.updatePage.mockResolvedValue({ outcome: "saved" });
  tauriMocks.getDaemonVersion.mockResolvedValue("0.14.1");
  tauriMocks.getSystemInfo.mockResolvedValue({ os: "macos" });
  tauriMocks.recordPageEditorDiagnostic.mockResolvedValue(undefined);
  tauriMocks.deletePage.mockResolvedValue(undefined);
  tauriMocks.clipboardWrite.mockResolvedValue(undefined);
  tauriMocks.exportPageToObsidian.mockResolvedValue({ path: "/tmp/page.md" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("PageDetail with the real MarkdownEditor and CodeMirror", () => {
  it.each([
    ["macos", "Cmd", "Ctrl"],
    ["windows", "Ctrl", "Cmd"],
  ])(
    "renders the %s shortcut modifier in the editor description",
    async (os, modifier, otherModifier) => {
      tauriMocks.getSystemInfo.mockResolvedValue({ os });
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      const user = userEvent.setup();
      render(
        <QueryClientProvider client={client}>
          <PageDetail
            pageId={PAGE.id}
            onBack={vi.fn()}
            onMemoryClick={vi.fn()}
            onPageClick={vi.fn()}
          />
        </QueryClientProvider>,
      );

      expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
      await user.click(screen.getByTitle("Edit page"));

      const description = document.getElementById(
        "page-markdown-editor-description",
      );
      expect(description).toHaveTextContent(
        `Markdown appears at the cursor. Save with ${modifier}+S.`,
      );
      expect(description).toHaveClass("sr-only");
      expect(description).not.toHaveTextContent(otherModifier);
      expect(tauriMocks.getSystemInfo).toHaveBeenCalledOnce();
    },
  );

  it("keeps the local macOS Cmd hint when system info is unavailable", async () => {
    vi.spyOn(window.navigator, "platform", "get").mockReturnValue("MacIntel");
    tauriMocks.getSystemInfo.mockRejectedValue(new Error("system info offline"));
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
    await user.click(screen.getByTitle("Edit page"));

    expect(
      document.getElementById("page-markdown-editor-description"),
    ).toHaveTextContent("Markdown appears at the cursor. Save with Cmd+S.");
    expect(tauriMocks.getSystemInfo).toHaveBeenCalledOnce();
  });

  it("uses one editing view and keeps Markdown syntax contextual", async () => {
    const editablePage = {
      ...PAGE,
      content: "# Heading\n\nalpha\n",
    };
    tauriMocks.getPage.mockResolvedValue(editablePage);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", { name: "Page editor" });
    const view = editorViewFromTextbox(textbox);
    expect(
      screen.queryByRole("radiogroup", { name: "Editing mode" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("radio", { name: "Live Preview" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("radio", { name: "Source mode" }),
    ).not.toBeInTheDocument();
    expect(view.dom).toHaveAttribute("data-editor-presentation-mode", "writing");
    expect(
      document.getElementById("page-markdown-editor-description"),
    ).toHaveTextContent(
      "Markdown appears at the cursor. Save with Cmd+S.",
    );

    act(() => {
      replaceDocument(view, "# Local heading\n\nalpha\n");
      selectRange(view, view.state.doc.length - 2);
    });
    expect(view.state.doc.toString()).toBe("# Local heading\n\nalpha\n");
    expect(view.dom).toHaveAttribute("data-editor-presentation-mode", "writing");
    expect(tauriMocks.updatePage).not.toHaveBeenCalled();
  });

  it("formats through the top toolbar and Save sends the exact live CodeMirror snapshot", async () => {
    const editablePage = { ...PAGE, content: "alpha" };
    tauriMocks.getPage.mockResolvedValue(editablePage);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", { name: "Page editor" });
    const view = editorViewFromTextbox(textbox);
    act(() => selectRange(view, 0, view.state.doc.length));

    await user.click(screen.getByRole("button", { name: "Bold" }));
    expect(view.state.doc.toString()).toBe("**alpha**");
    expect(textbox).toHaveFocus();

    await user.click(screen.getByRole("button", { name: /^Save$/ }));

    await waitFor(() => {
      expect(tauriMocks.updatePage).toHaveBeenCalledWith({
        id: PAGE.id,
        content: "**alpha**",
        expectedVersion: PAGE.version,
        callerId: "wenlan-app",
        operationId: expect.any(String),
      });
    });
  });

  it("saves the exact current CodeMirror document through the PageDetail CAS seam", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <StrictMode>
        <QueryClientProvider client={client}>
          <PageDetail
            pageId={PAGE.id}
            onBack={vi.fn()}
            onMemoryClick={vi.fn()}
            onPageClick={vi.fn()}
          />
        </QueryClientProvider>
      </StrictMode>,
    );

    expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", {
      name: "Page editor",
    });
    await waitFor(() => {
      expect(
        screen.getByRole("textbox", { name: "Page editor" }),
      ).toHaveFocus();
    });
    expect(
      document.getElementById("page-markdown-editor-description"),
    ).toHaveClass("sr-only");
    const exactSource = "  # Exact CodeMirror source  \n\nFinal line\t \n";
    act(() => {
      replaceDocument(editorViewFromTextbox(textbox), exactSource);
      pressKey(textbox, "s", { ctrlKey: true });
    });

    await waitFor(() => {
      expect(tauriMocks.updatePage).toHaveBeenCalledWith({
        id: PAGE.id,
        content: exactSource,
        expectedVersion: PAGE.version,
        callerId: "wenlan-app",
        operationId: expect.any(String),
      });
    });
    await waitFor(() => {
      expect(
        screen.queryByRole("textbox", { name: "Page editor" }),
      ).toBeNull();
    });
  });

  it("preserves an upgrade-blocked draft, offers copy, and keeps Save recoverable", async () => {
    tauriMocks.updatePage.mockResolvedValue({
      outcome: "upgrade_required",
      reportedVersion: "0.14.0",
      requiredFloor: "0.14.1",
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(PAGE.title)).toBeInTheDocument();
    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", {
      name: "Page editor",
    });
    const blockedDraft = "# Draft blocked by the old daemon\n";
    act(() => replaceDocument(editorViewFromTextbox(textbox), blockedDraft));
    await user.click(screen.getByRole("button", { name: /^Save$/ }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Page editing requires stable Wenlan daemon 0.14.1 or later. Running version: 0.14.0.",
    );
    expect(screen.getByRole("button", { name: "Copy my draft" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Retry$/ })).toBeNull();
    expect(screen.getByRole("button", { name: /^Save$/ })).toBeInTheDocument();
    expect(editorViewFromTextbox(textbox).state.doc.toString()).toBe(blockedDraft);

    const editedDraft = `${blockedDraft}\nStill editing locally.\n`;
    act(() => replaceDocument(editorViewFromTextbox(textbox), editedDraft));
    expect(screen.getByRole("alert")).toHaveTextContent("0.14.1");
    await user.click(screen.getByRole("button", { name: "Copy my draft" }));
    expect(tauriMocks.clipboardWrite).toHaveBeenCalledWith(editedDraft);

    await user.click(screen.getByRole("button", { name: /^Save$/ }));
    await waitFor(() => expect(tauriMocks.updatePage).toHaveBeenCalledTimes(2));
  });

  it("uses the editable H1 as the single title and the toolbar as the only action center", async () => {
    const editablePage = {
      ...PAGE,
      content: `# ${PAGE.title}\n\nBody copy.\n`,
    };
    tauriMocks.getPage.mockResolvedValue(editablePage);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    const { container } = render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText(PAGE.title)).toHaveClass("page-detail-title");
    expect(
      screen.getByRole("button", { name: "Copy as context" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Page actions" }),
    ).toBeInTheDocument();

    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", {
      name: "Page editor",
    });

    expect(container.querySelector(".page-detail-title")).toBeNull();
    expect(
      screen.getByRole("heading", { level: 1, name: PAGE.title }),
    ).toHaveClass("sr-only");
    expect(
      screen.getByRole("toolbar", { name: "Formatting" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Copy as context" }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Page actions" })).toBeNull();
    expect(
      document.getElementById("page-markdown-editor-description"),
    ).toHaveClass("sr-only");

    act(() => {
      replaceDocument(
        editorViewFromTextbox(textbox),
        "Body without a matching title.\n",
      );
    });
    expect(container.querySelector(".page-detail-title")).toBeVisible();

    act(() => {
      replaceDocument(editorViewFromTextbox(textbox), editablePage.content);
    });
    expect(container.querySelector(".page-detail-title")).toBeNull();
    expect(
      screen.getByRole("heading", { level: 1, name: PAGE.title }),
    ).toHaveClass("sr-only");

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByText(PAGE.title)).toHaveClass("page-detail-title");
    expect(
      screen.getByRole("button", { name: "Copy as context" }),
    ).toBeInTheDocument();
  });

  it("keeps title suppression tied to the active editor baseline during a remote conflict", async () => {
    const editablePage = {
      ...PAGE,
      content: `# ${PAGE.title}\n\nLocal baseline.\n`,
    };
    tauriMocks.getPage.mockResolvedValue(editablePage);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    const { container } = render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    await screen.findByRole("heading", { level: 1, name: PAGE.title });
    await user.click(screen.getByTitle("Edit page"));
    const textbox = await screen.findByRole("textbox", { name: "Page editor" });
    const localDraft = `# ${PAGE.title}\n\nUnsaved local draft.\n`;
    act(() => replaceDocument(editorViewFromTextbox(textbox), localDraft));

    act(() => {
      client.setQueryData(["page", PAGE.id], {
        ...editablePage,
        content: "Remote source without a matching heading.\n",
        version: PAGE.version + 1,
      });
    });

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "This page changed elsewhere.",
      ),
    );
    expect(container.querySelector(".page-detail-title")).toBeNull();
    expect(
      screen.getByRole("heading", { level: 1, name: PAGE.title }),
    ).toHaveClass("sr-only");
    expect(editorViewFromTextbox(textbox).state.doc.toString()).toBe(localDraft);
  });

  it("keeps the Page title visible while matching-H1 source awaits line-ending normalization", async () => {
    const mixedLineEndingPage = {
      ...PAGE,
      content: `# ${PAGE.title}\r\n\r\nMixed source.\n`,
    };
    tauriMocks.getPage.mockResolvedValue(mixedLineEndingPage);
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const user = userEvent.setup();
    const { container } = render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId={PAGE.id}
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onPageClick={vi.fn()}
        />
      </QueryClientProvider>,
    );

    await screen.findByTitle("Edit page");
    expect(container.querySelector(".page-detail-title")).toBeVisible();
    await user.click(screen.getByTitle("Edit page"));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Line-ending normalization required",
    );
    expect(container.querySelector(".page-detail-title")).toBeVisible();
    expect(
      screen.getByRole("heading", { level: 1, name: PAGE.title }),
    ).not.toHaveClass("sr-only");
    expect(
      screen.queryByRole("textbox", { name: "Page editor" }),
    ).toBeNull();
  });
});
