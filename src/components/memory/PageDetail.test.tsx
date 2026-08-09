// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "./PageDetail";
import { i18n } from "../../i18n";

vi.mock("../../lib/tauri", () => ({
  getPage: vi.fn().mockResolvedValue({
    id: "concept_abc",
    title: "libSQL Architecture",
    summary: "Core database layer for Origin",
    content: "libSQL is the core database layer powering Origin's memory system, chosen for its vector support and SQLite compatibility.\n\nOrigin stores 768-dimensional embeddings in F32_BLOB columns with DiskANN indexing for fast approximate nearest neighbor search. The database also hosts the knowledge graph ([[Entity Graph]]) and full-text search via FTS5 triggers. This architecture enables hybrid retrieval combining vector similarity with keyword matching through [[Reciprocal Rank Fusion]].\n\n## Open Questions\n- Performance at scale beyond 10k memories?\n\n## Sources\n- mem_1 (2026-04-01)\n- mem_2 (2026-04-01)",
    entity_id: "entity_libsql",
    domain: "architecture",
    source_memory_ids: ["mem_1", "mem_2"],
    version: 3,
    status: "active",
    created_at: "2026-04-01T00:00:00+00:00",
    last_compiled: "2026-04-07T12:00:00+00:00",
    last_modified: "2026-04-07T12:00:00+00:00",
  }),
  getPageSources: vi.fn().mockResolvedValue([
    {
      source: { page_id: "page_abc", memory_source_id: "mem_1", linked_at: 1712000000, link_reason: "page_growth" },
      memory: {
        source_id: "mem_1",
        title: "libSQL stores vectors",
        content: "libSQL stores vectors in F32_BLOB columns",
        summary: null,
        memory_type: "fact",
        domain: "architecture",
        source_agent: "claude",
        confidence: 0.9,
        confirmed: false,
        last_modified: 1712000000,
      },
    },
    {
      source: { page_id: "page_abc", memory_source_id: "mem_2", linked_at: 1712100000, link_reason: "page_growth" },
      memory: {
        source_id: "mem_2",
        title: "DiskANN indexing strategy",
        content: "DiskANN provides fast approximate nearest neighbor search",
        summary: null,
        memory_type: "fact",
        domain: "architecture",
        source_agent: "claude-code",
        confidence: 0.85,
        confirmed: true,
        last_modified: 1712100000,
      },
    },
  ]),
  clipboardWrite: vi.fn().mockResolvedValue(undefined),
  exportPagesToObsidian: vi.fn().mockResolvedValue({ exported: 1, skipped: 0, failed: 0 }),
  exportPageToObsidian: vi.fn().mockResolvedValue({ path: "/path/to/file.md" }),
  listRegisteredSources: vi.fn().mockResolvedValue([
    { id: "obsidian-vault", source_type: "obsidian", path: "/Users/test/vault", status: "Active", last_sync: null, file_count: 10, memory_count: 20 },
  ]),
  getPageLinks: vi.fn().mockResolvedValue({ outbound: [], inbound: [] }),
  listOrphanLinks: vi.fn().mockResolvedValue({ min_count: 2, orphan_labels: [] }),
  listPages: vi.fn().mockResolvedValue([]),
  redistillPage: vi.fn().mockResolvedValue({ status: "ok", updated: true }),
  updatePage: vi.fn().mockResolvedValue(undefined),
  deletePage: vi.fn().mockResolvedValue(undefined),
  FACET_COLORS: {},
  STABILITY_TIERS: {},
  getPendingRevision: vi.fn().mockResolvedValue(null),
  acceptPendingRevision: vi.fn(),
  dismissPendingRevision: vi.fn(),
}));

function renderWithQuery(
  ui: React.ReactElement,
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return {
    user: userEvent.setup(),
    ...render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>),
  };
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

async function makeNextPageResolvable() {
  const { getPage } = await import("../../lib/tauri");
  const getPageMock = getPage as ReturnType<typeof vi.fn>;
  const currentPage = await getPage("concept_abc");
  getPageMock.mockImplementation(async (id: string) => {
    if (id !== "concept_next") return currentPage;
    return {
      id: "concept_next",
      title: "Next Page",
      summary: null,
      content: "Next page content.",
      entity_id: null,
      domain: null,
      source_memory_ids: [],
      version: 1,
      status: "active",
      created_at: "2026-04-09T00:00:00+00:00",
      last_compiled: "2026-04-09T12:00:00+00:00",
      last_modified: "2026-04-09T12:00:00+00:00",
    };
  });
  getPageMock.mockClear();
}

describe("PageDetail", () => {
  const defaultProps = {
    pageId: "concept_abc",
    onBack: vi.fn(),
    onMemoryClick: vi.fn(),
    onPageClick: vi.fn(),
  };

  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
  });

  it("renders page title", async () => {
    renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("names and dismisses the destination Page when authored content attaches to it", async () => {
    const onDismissAttachedPageNotice = vi.fn();
    const { user } = renderWithQuery(
      <PageDetail
        {...defaultProps}
        onDismissAttachedPageNotice={onDismissAttachedPageNotice}
        showAttachedPageNotice
      />,
    );

    const notice = await screen.findByRole("status", { name: "Added to “libSQL Architecture”" });
    expect(notice).toHaveTextContent("Added to “libSQL Architecture”");
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onDismissAttachedPageNotice).toHaveBeenCalledOnce();
  });

  it("reports the loaded Page once so visit history records real opens", async () => {
    const onPageLoaded = vi.fn();
    renderWithQuery(<PageDetail {...defaultProps} onPageLoaded={onPageLoaded} />);

    await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" });
    await waitFor(() => expect(onPageLoaded).toHaveBeenCalledWith({
      id: "concept_abc",
      status: "active",
      title: "libSQL Architecture",
    }));
    expect(onPageLoaded).toHaveBeenCalledTimes(1);
  });

  it("renders meta line with distilled time", async () => {
    const { container } = renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText(/Last distilled/)).toBeTruthy();
    expect(await screen.findByText(/from 2 memories/)).toBeTruthy();

    const dateline = container.querySelector(".page-detail-dateline");
    const items = dateline?.querySelectorAll(".page-detail-dateline-item");
    expect(items).toHaveLength(2);
    expect(Array.from(items ?? []).map((item) => item.textContent)).toEqual([
      expect.stringMatching(/^Last distilled \S+(?: \S+)?$/),
      "from 2 memories",
    ]);
    expect(dateline?.querySelector(".page-detail-dateline-separator")).toBeNull();

    const css = readFileSync(resolve("src/index.css"), "utf8");
    expect(css).toMatch(
      /\.page-detail-dateline-item\s*\{[^}]*white-space:\s*nowrap;/s,
    );
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.page-detail-dateline-item\s*\+\s*\.page-detail-dateline-item::before\s*\{[^}]*display:\s*none;/,
    );
  });

  it("renders last distilled info", async () => {
    renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText(/Last distilled/)).toBeTruthy();
  });

  it("renders source memory count", async () => {
    renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByText(/from 2 memories/)).toBeTruthy();
  });

  it("renders copy and export buttons in header", async () => {
    const { container } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    expect(container.querySelector('button[title="Copy as context"]')).toBeTruthy();
    expect(container.querySelector('button[title="Export to Obsidian"]')).toBeTruthy();
  });

  it("offers one mobile primary action and a keyboard-safe overflow instead of bare delete", async () => {
    const { deletePage } = await import("../../lib/tauri");
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user, container } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");

    const mobilePrimary = container.querySelector(".page-detail-primary-action");
    expect(mobilePrimary).toHaveAccessibleName("Edit page");
    expect(container.querySelector('button[title="Delete page"]')).toBeNull();

    const trigger = screen.getByRole("button", { name: "Page actions" });
    await user.click(trigger);
    const menu = screen.getByRole("menu", { name: "Page actions" });
    expect(within(menu).getByRole("menuitem", { name: "Re-distill page" })).toBeInTheDocument();
    expect(within(menu).getByRole("menuitem", { name: "Copy as context" })).toBeInTheDocument();
    expect(within(menu).getByRole("menuitem", { name: "Delete page" })).toBeInTheDocument();

    expect(within(menu).getByRole("menuitem", { name: "Re-distill page" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Page actions" })).toBeNull();
    expect(trigger).toHaveFocus();

    await user.click(trigger);
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("menu", { name: "Page actions" })).toBeNull();

    await user.click(trigger);
    await user.click(screen.getByRole("menuitem", { name: "Delete page" }));
    expect(confirmSpy).toHaveBeenCalledWith("Delete this page?");
    expect(deletePage).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    await user.click(trigger);
    await user.click(screen.getByRole("menuitem", { name: "Delete page" }));
    await waitFor(() => expect(deletePage).toHaveBeenCalledWith("concept_abc"));

    const css = readFileSync(resolve("src/index.css"), "utf8");
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)\s*\{[\s\S]*?\.page-detail-icon-actions\s*\{[^}]*display:\s*none;[\s\S]*?\.page-detail-primary-action\s*\{[^}]*display:\s*inline-flex;/,
    );
    confirmSpy.mockRestore();
  });

  it("returns to the Wiki inventory and invalidates Page queries after deletion", async () => {
    const { deletePage } = await import("../../lib/tauri");
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />, queryClient);

    try {
      await screen.findByText("libSQL Architecture");
      await user.click(screen.getByRole("button", { name: "Page actions" }));
      await user.click(screen.getByRole("menuitem", { name: "Delete page" }));

      await waitFor(() => expect(deletePage).toHaveBeenCalledWith("concept_abc"));
      await waitFor(() => expect(defaultProps.onBack).toHaveBeenCalledOnce());
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["pages"] });
    } finally {
      confirmSpy.mockRestore();
    }
  });

  it("surfaces deletion failures without navigating away and remains retryable", async () => {
    const { deletePage } = await import("../../lib/tauri");
    (deletePage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("daemon offline"));
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    try {
      await screen.findByText("libSQL Architecture");
      await user.click(screen.getByRole("button", { name: "Page actions" }));
      await user.click(screen.getByRole("menuitem", { name: "Delete page" }));

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Could not delete this page. Try again.",
      );
      expect(defaultProps.onBack).not.toHaveBeenCalled();

      await user.click(screen.getByRole("button", { name: "Page actions" }));
      const retry = screen.getByRole("menuitem", { name: "Delete page" });
      expect(retry).toBeEnabled();
      await user.click(retry);

      await waitFor(() => expect(deletePage).toHaveBeenCalledTimes(2));
      await waitFor(() => expect(defaultProps.onBack).toHaveBeenCalledOnce());
    } finally {
      confirmSpy.mockRestore();
    }
  });

  it("does not navigate away from a newer Page when an older deletion finishes", async () => {
    const { deletePage } = await import("../../lib/tauri");
    const pending = deferred<void>();
    (deletePage as ReturnType<typeof vi.fn>).mockReturnValueOnce(pending.promise);
    await makeNextPageResolvable();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender, user } = renderWithQuery(
      <PageDetail {...defaultProps} />,
      queryClient,
    );

    try {
      await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" });
      await user.click(screen.getByRole("button", { name: "Page actions" }));
      await user.click(screen.getByRole("menuitem", { name: "Delete page" }));
      await waitFor(() => expect(deletePage).toHaveBeenCalledWith("concept_abc"));

      rerender(
        <QueryClientProvider client={queryClient}>
          <PageDetail {...defaultProps} pageId="concept_next" />
        </QueryClientProvider>,
      );
      await screen.findByRole("heading", { level: 1, name: "Next Page" });

      await act(async () => pending.resolve());
      expect(defaultProps.onBack).not.toHaveBeenCalled();
      expect(screen.getByRole("heading", { level: 1, name: "Next Page" })).toBeVisible();
    } finally {
      confirmSpy.mockRestore();
    }
  });

  it("does not show an older Page save failure on a newer Page", async () => {
    const { updatePage } = await import("../../lib/tauri");
    const pending = deferred<void>();
    (updatePage as ReturnType<typeof vi.fn>).mockReturnValueOnce(pending.promise);
    await makeNextPageResolvable();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender, user } = renderWithQuery(
      <PageDetail {...defaultProps} />,
      queryClient,
    );

    await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" });
    await user.click(screen.getByTitle("Edit page"));
    const editor = screen.getByRole("textbox");
    await user.clear(editor);
    await user.type(editor, "Revised Page body");
    await user.click(screen.getByRole("button", { name: "Save (Cmd+Enter)" }));
    await waitFor(() =>
      expect(updatePage).toHaveBeenCalledWith("concept_abc", "Revised Page body"),
    );

    rerender(
      <QueryClientProvider client={queryClient}>
        <PageDetail {...defaultProps} pageId="concept_next" />
      </QueryClientProvider>,
    );
    await screen.findByRole("heading", { level: 1, name: "Next Page" });

    await act(async () => {
      pending.reject(new Error("late write failure"));
      await pending.promise.catch(() => undefined);
    });
    expect(screen.queryByText("Could not save this page. Try again.")).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("serializes Cmd+Enter saves so an older body cannot finish last", async () => {
    const { updatePage } = await import("../../lib/tauri");
    const pending = deferred<void>();
    (updatePage as ReturnType<typeof vi.fn>).mockReturnValueOnce(pending.promise);
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" });
    await user.click(screen.getByTitle("Edit page"));
    const editor = screen.getByRole("textbox");
    await user.clear(editor);
    await user.type(editor, "One serialized Page body");

    try {
      fireEvent.keyDown(editor, { key: "Enter", metaKey: true });
      fireEvent.keyDown(editor, { key: "Enter", metaKey: true });

      await waitFor(() =>
        expect(updatePage).toHaveBeenCalledWith(
          "concept_abc",
          "One serialized Page body",
        ),
      );
      expect(updatePage).toHaveBeenCalledTimes(1);
      expect(editor).toBeDisabled();
    } finally {
      await act(async () => pending.resolve());
    }
  });

  it("does not show an older re-distill result on a newer Page", async () => {
    const { redistillPage } = await import("../../lib/tauri");
    const pending = deferred<{ status: "ok"; updated: true }>();
    (redistillPage as ReturnType<typeof vi.fn>).mockReturnValueOnce(pending.promise);
    await makeNextPageResolvable();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender, user } = renderWithQuery(
      <PageDetail {...defaultProps} />,
      queryClient,
    );

    await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" });
    await user.click(screen.getByTitle("Re-distill page"));
    await waitFor(() => expect(redistillPage).toHaveBeenCalledWith("concept_abc"));

    rerender(
      <QueryClientProvider client={queryClient}>
        <PageDetail {...defaultProps} pageId="concept_next" />
      </QueryClientProvider>,
    );
    await screen.findByRole("heading", { level: 1, name: "Next Page" });

    await act(async () => pending.resolve({ status: "ok", updated: true }));
    expect(screen.queryByText("Page re-distilled.")).toBeNull();
  });

  it("distinguishes a Page load failure from not found and retries in place", async () => {
    const { getPage } = await import("../../lib/tauri");
    (getPage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("daemon offline"));
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not load this page.",
    );
    expect(screen.queryByText("Page not found")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" })).toBeVisible();
  });

  it("keeps cached Page content visible when a background refetch fails", async () => {
    const { getPage } = await import("../../lib/tauri");
    const getPageMock = getPage as ReturnType<typeof vi.fn>;
    const cachedPage = await getPage("concept_abc");
    getPageMock.mockClear();
    getPageMock.mockRejectedValueOnce(new Error("background refresh failed"));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    queryClient.setQueryData(["page", "concept_abc"], cachedPage);

    renderWithQuery(<PageDetail {...defaultProps} />, queryClient);

    expect(
      await screen.findByRole("heading", { level: 1, name: "libSQL Architecture" }),
    ).toBeVisible();
    await waitFor(() =>
      expect(
        queryClient.getQueryState(["page", "concept_abc"])?.error,
      ).toEqual(expect.any(Error)),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "libSQL Architecture" }),
    ).toBeVisible();
    expect(screen.queryByText("Could not load this page.")).toBeNull();
  });

  it("keeps the editor open and retryable when saving fails", async () => {
    const { updatePage } = await import("../../lib/tauri");
    (updatePage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("write failed"));
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByTitle("Edit page"));
    const editor = screen.getByRole("textbox");
    await user.clear(editor);
    await user.type(editor, "Revised Page body");
    await user.click(screen.getByRole("button", { name: "Save (Cmd+Enter)" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not save this page. Try again.",
    );
    expect(editor).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Save (Cmd+Enter)" }));
    await waitFor(() => expect(updatePage).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull());
  });

  it("surfaces copy failures and lets the user retry", async () => {
    const { clipboardWrite } = await import("../../lib/tauri");
    (clipboardWrite as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("clipboard denied"));
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByTitle("Copy as context"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not copy this page. Try again.",
    );

    await user.click(screen.getByTitle("Copy as context"));
    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledTimes(2));
    expect(await screen.findByTitle("Copied!")).toBeVisible();
  });

  it("surfaces export failures and lets the user retry", async () => {
    const { exportPageToObsidian } = await import("../../lib/tauri");
    (exportPageToObsidian as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("vault unavailable"));
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByTitle("Export to Obsidian"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not export this page. Try again.",
    );

    await user.click(screen.getByTitle("Export to Obsidian"));
    await waitFor(() => expect(exportPageToObsidian).toHaveBeenCalledTimes(2));
    expect(await screen.findByTitle("Exported!")).toBeVisible();
  });

  it("keeps Page action menu keyboard events out of Main history and manages item focus", async () => {
    const windowEscapeObserver = vi.fn();
    window.addEventListener("keydown", windowEscapeObserver);

    try {
      const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
      await screen.findByText("libSQL Architecture");

      const trigger = screen.getByRole("button", { name: "Page actions" });
      await user.click(trigger);
      const menu = screen.getByRole("menu", { name: "Page actions" });
      const items = within(menu).getAllByRole("menuitem").filter((item) => !item.hasAttribute("disabled"));

      expect(items[0]).toHaveFocus();
      await user.keyboard("{ArrowDown}");
      expect(items[1]).toHaveFocus();
      await user.keyboard("{End}");
      expect(items[items.length - 1]).toHaveFocus();
      await user.keyboard("{ArrowDown}");
      expect(items[0]).toHaveFocus();
      await user.keyboard("{ArrowUp}");
      expect(items[items.length - 1]).toHaveFocus();
      await user.keyboard("{Home}");
      expect(items[0]).toHaveFocus();

      await user.keyboard("{Escape}");
      expect(screen.queryByRole("menu", { name: "Page actions" })).toBeNull();
      expect(trigger).toHaveFocus();
      expect(windowEscapeObserver).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("keydown", windowEscapeObserver);
    }
  });

  it("opens Page actions from its trigger with ArrowDown or ArrowUp at the matching boundary", async () => {
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");

    const trigger = screen.getByRole("button", { name: "Page actions" });
    trigger.focus();
    await user.keyboard("{ArrowDown}");
    let menu = screen.getByRole("menu", { name: "Page actions" });
    let items = within(menu).getAllByRole("menuitem").filter((item) => !item.hasAttribute("disabled"));
    expect(items[0]).toHaveFocus();

    await user.keyboard("{Escape}");
    await user.keyboard("{ArrowUp}");
    menu = screen.getByRole("menu", { name: "Page actions" });
    items = within(menu).getAllByRole("menuitem").filter((item) => !item.hasAttribute("disabled"));
    expect(items[items.length - 1]).toHaveFocus();
  });

  it("focuses the first rendered Page action instead of a CSS-hidden mobile item", async () => {
    const rectsSpy = vi
      .spyOn(HTMLElement.prototype, "getClientRects")
      .mockImplementation(function (this: HTMLElement) {
        return (
          this.classList.contains("page-detail-mobile-menu-item")
            ? []
            : [{}]
        ) as unknown as DOMRectList;
      });

    try {
      const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
      await screen.findByText("libSQL Architecture");

      await user.click(screen.getByRole("button", { name: "Page actions" }));
      expect(screen.getByRole("menuitem", { name: "Delete page" })).toHaveFocus();
    } finally {
      rectsSpy.mockRestore();
    }
  });

  it.each([
    ["en", "Last distilled 5m ago", "from 2 memories", "needs review"],
    ["zh-Hans", "上次精炼：5 分钟前", "来自 2 条记忆", "需要审核"],
    ["zh-Hant", "上次精煉：5 分鐘前", "來自 2 則記憶", "需要審核"],
  ] as const)(
    "localizes the full Page dateline in %s",
    async (locale, distilled, sources, stale) => {
      const { getPage } = await import("../../lib/tauri");
      (getPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        id: "concept_abc",
        title: "Localized Page",
        summary: null,
        content: "Localized page content.",
        entity_id: null,
        domain: null,
        source_memory_ids: ["mem_1", "mem_2"],
        version: 1,
        status: "active",
        created_at: "2026-07-17T11:00:00+00:00",
        last_compiled: "2026-07-17T11:55:00+00:00",
        last_modified: "2026-07-17T11:55:00+00:00",
        stale_reason: "source_conflict",
      });
      const nowSpy = vi.spyOn(Date, "now").mockReturnValue(
        new Date("2026-07-17T12:00:00+00:00").getTime(),
      );
      await i18n.changeLanguage(locale);

      try {
        renderWithQuery(<PageDetail {...defaultProps} />);

        expect(await screen.findByText(distilled)).toBeInTheDocument();
        expect(screen.getByText(sources)).toBeInTheDocument();
        expect(screen.getByText(stale)).toBeInTheDocument();
        if (locale !== "en") {
          expect(screen.queryByText("Last distilled 5m ago")).toBeNull();
          expect(screen.queryByText("from 2 memories")).toBeNull();
          expect(screen.queryByText("needs review")).toBeNull();
        }
      } finally {
        nowSpy.mockRestore();
      }
    },
  );

  it("re-distills the current page and keeps skipped daemon hints visible", async () => {
    const { redistillPage } = await import("../../lib/tauri");
    const hint = "page re-distill needs an LLM in the daemon";
    (redistillPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      status: "skipped",
      updated: false,
      hint,
    });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />, queryClient);

    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByTitle("Re-distill page"));

    expect(redistillPage).toHaveBeenCalledWith("concept_abc");
    expect(await screen.findByText(hint)).toBeTruthy();
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["page", "concept_abc"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["page-revisions", "concept_abc"] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["page-sources", "concept_abc"] });
    expect(defaultProps.onBack).not.toHaveBeenCalled();
  });

  it("confirms before re-distilling a user-edited page", async () => {
    const { getPage, redistillPage } = await import("../../lib/tauri");
    (getPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      id: "concept_abc",
      title: "Edited Page",
      summary: null,
      content: "Edited page prose.",
      entity_id: null,
      domain: null,
      source_memory_ids: [],
      version: 4,
      status: "active",
      created_at: "2026-04-01T00:00:00+00:00",
      last_compiled: "2026-04-07T12:00:00+00:00",
      last_modified: "2026-04-08T12:00:00+00:00",
      user_edited: true,
    });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);

    await screen.findByText("Edited Page");
    await user.click(screen.getByTitle("Re-distill page"));

    expect(confirmSpy).toHaveBeenCalledWith(
      "Re-distill this edited page? The current version stays in page history for recovery.",
    );
    expect(redistillPage).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it("clears re-distill notices when navigating between pages", async () => {
    const { getPage, redistillPage } = await import("../../lib/tauri");
    const hint = "page re-distill needs an LLM in the daemon";
    (redistillPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      status: "skipped",
      updated: false,
      hint,
    });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { user, rerender } = renderWithQuery(<PageDetail {...defaultProps} />, queryClient);

    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByTitle("Re-distill page"));
    expect(await screen.findByText(hint)).toBeTruthy();

    (getPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      id: "concept_next",
      title: "Next Page",
      summary: null,
      content: "Next page content.",
      entity_id: null,
      domain: null,
      source_memory_ids: [],
      version: 1,
      status: "active",
      created_at: "2026-04-09T00:00:00+00:00",
      last_compiled: "2026-04-09T12:00:00+00:00",
      last_modified: "2026-04-09T12:00:00+00:00",
    });
    rerender(
      <QueryClientProvider client={queryClient}>
        <PageDetail {...defaultProps} pageId="concept_next" />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(screen.queryByText(hint)).toBeNull());
  });

  it("gives the icon-only back button a localized accessible name", async () => {
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    const backButton = screen.getByRole("button", { name: "Back" });
    expect(backButton.querySelector("svg")).toBeTruthy();

    await user.click(backButton);
    expect(defaultProps.onBack).toHaveBeenCalledOnce();
  });

  it("renders source memories section with count", async () => {
    renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    expect(screen.getByText(/2 sources/)).toBeInTheDocument();
  });

  it("uses the page-detail grammar class (matches MemoryDetail's dossier pattern)", async () => {
    const { container } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    const topDiv = container.firstElementChild as HTMLElement;
    expect(topDiv.className).toContain("page-detail");
    expect(topDiv.className).not.toContain("h-screen");
  });

  it("shows Page Links from daemon page links without listPages inference", async () => {
    const { getPageLinks, listPages: mockList } = await import("../../lib/tauri");
    (getPageLinks as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      outbound: [{ label: "Entity Graph", target_page_id: "concept_eg" }],
      inbound: [],
    });
    renderWithQuery(<PageDetail {...defaultProps} />);
    expect(await screen.findByLabelText("Related pages")).toBeTruthy();
    const entityEls = await screen.findAllByText("Entity Graph");
    const cardSpan = entityEls.find((el) => el.tagName === "SPAN");
    expect(cardSpan).toBeTruthy();
    expect(mockList).not.toHaveBeenCalled();
  });

  it("hides Page Links when the daemon returns no links", async () => {
    const { listPages: mockList } = await import("../../lib/tauri");
    renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    expect(screen.queryByLabelText("Related pages")).toBeNull();
    expect(screen.getByText(/Page info/i)).toBeInTheDocument();
    expect(mockList).not.toHaveBeenCalled();
  });

  it("hides Page Links section when no daemon links in content", async () => {
    const { getPage } = await import("../../lib/tauri");
    (getPage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      id: "concept_abc",
      title: "Simple Page",
      summary: "No links here",
      content: "A simple page with no wikilinks.\n\n## Open Questions\n- None",
      entity_id: null,
      domain: null,
      source_memory_ids: [],
      version: 1,
      status: "active",
      created_at: "2026-04-01T00:00:00+00:00",
      last_compiled: "2026-04-07T12:00:00+00:00",
      last_modified: "2026-04-07T12:00:00+00:00",
    });
    renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("Simple Page");
    expect(screen.queryByLabelText("Related pages")).toBeNull();
    expect(screen.getByText(/Page info/i)).toBeInTheDocument();
  });

  it("renders one evidence card per source memory after fetch", async () => {
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByText(/Page info/i));
    expect(screen.getAllByTestId("page-info-source-row")).toHaveLength(2);
  });

  it("clicking an evidence card calls onMemoryClick with the right source_id", async () => {
    const { user } = renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL Architecture");
    await user.click(screen.getByText(/Page info/i));
    const row = screen
      .getByText("libSQL stores vectors")
      .closest('[data-testid="page-info-source-row"]')!;
    await user.click(row);
    expect(defaultProps.onMemoryClick).toHaveBeenCalledWith("mem_1");
  });

  it("uses getPageSources (join table) not listMemoriesByIds", async () => {
    const { getPageSources } = await import("../../lib/tauri");
    renderWithQuery(<PageDetail {...defaultProps} />);
    await screen.findByText("libSQL stores vectors");
    expect(getPageSources).toHaveBeenCalledTimes(1);
    expect(getPageSources).toHaveBeenCalledWith("concept_abc");
  });
});
