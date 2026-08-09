// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "../PageDetail";
import * as tauri from "../../../lib/tauri";

vi.mock("../../../lib/tauri");

const MOCK_PAGE: tauri.Page = {
  id: "c1",
  title: "Test Page",
  content: "Some page content",
  summary: "A test summary",
  domain: "testing",
  entity_id: null,
  version: 1,
  status: "active",
  created_at: new Date().toISOString(),
  last_compiled: new Date().toISOString(),
  last_modified: new Date().toISOString(),
  source_memory_ids: [],
};

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("PageDetail export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(tauri.getPage).mockResolvedValue(MOCK_PAGE);
    vi.mocked(tauri.getPageLinks).mockResolvedValue({ outbound: [], inbound: [] });
    vi.mocked(tauri.getPageRevisions).mockResolvedValue({
      page_id: "c1",
      current_version: 1,
      user_edited: false,
      entries: [],
    });
    vi.mocked(tauri.getPageSources).mockResolvedValue([]);
    vi.mocked(tauri.listOrphanLinks).mockResolvedValue({ min_count: 2, orphan_labels: [] });
    vi.mocked(tauri.listPages).mockResolvedValue([MOCK_PAGE]);
  });

  it("disables export button when no obsidian sources exist", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);

    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    await waitFor(() => {
      expect(screen.getByTitle(/add an obsidian source/i)).toBeInTheDocument();
    });

    const btn = screen.getByTitle(/add an obsidian source/i);
    expect(btn).toBeDisabled();
  });

  it("exports directly when exactly one obsidian source exists", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault",
        source_type: "obsidian",
        path: "/Users/test/vault",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
    ]);
    vi.mocked(tauri.exportPageToObsidian).mockResolvedValue({
      path: "/Users/test/vault/Wenlan/pages/Test Page.md",
    });

    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    await waitFor(() => {
      expect(screen.getByTitle("Export to Obsidian")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTitle("Export to Obsidian"));

    await waitFor(() => {
      expect(tauri.exportPageToObsidian).toHaveBeenCalledWith(
        "c1",
        "/Users/test/vault/Wenlan/pages",
      );
    });
  });

  it("shows popover menu when 2+ obsidian sources exist", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault-1",
        source_type: "obsidian",
        path: "/Users/test/vault-one",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
      {
        id: "obsidian-vault-2",
        source_type: "obsidian",
        path: "/Users/test/vault-two",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 10,
      },
    ]);

    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    await waitFor(() => {
      expect(screen.getByTitle("Export to Obsidian")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTitle("Export to Obsidian"));

    await waitFor(() => {
      expect(screen.getByRole("menu")).toBeInTheDocument();
      expect(screen.getByRole("menuitem", { name: "vault-one" })).toBeInTheDocument();
      expect(screen.getByRole("menuitem", { name: "vault-two" })).toBeInTheDocument();
    });
  });

  it("closes the export menu with Escape and returns focus to its trigger", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault-1",
        source_type: "obsidian",
        path: "/Users/test/vault-one",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
      {
        id: "obsidian-vault-2",
        source_type: "obsidian",
        path: "/Users/test/vault-two",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 10,
      },
    ]);

    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    const trigger = await screen.findByTitle("Export to Obsidian");
    fireEvent.click(trigger);
    const menu = await screen.findByRole("menu");

    fireEvent.keyDown(menu, { key: "Escape" });

    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("keeps export-menu Escape out of Main history and supports standard item navigation", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault-1",
        source_type: "obsidian",
        path: "/Users/test/vault-one",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
      {
        id: "obsidian-vault-2",
        source_type: "obsidian",
        path: "/Users/test/vault-two",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 10,
      },
    ]);
    const windowEscapeObserver = vi.fn();
    window.addEventListener("keydown", windowEscapeObserver);

    try {
      const user = userEvent.setup();
      render(
        <PageDetail
          pageId="c1"
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
        />,
        { wrapper },
      );

      const trigger = await screen.findByTitle("Export to Obsidian");
      await user.click(trigger);
      const firstItem = await screen.findByRole("menuitem", { name: "vault-one" });
      const lastItem = screen.getByRole("menuitem", { name: "vault-two" });

      expect(firstItem).toHaveFocus();
      await user.keyboard("{ArrowDown}");
      expect(lastItem).toHaveFocus();
      await user.keyboard("{ArrowDown}");
      expect(firstItem).toHaveFocus();
      await user.keyboard("{ArrowUp}");
      expect(lastItem).toHaveFocus();
      await user.keyboard("{Home}");
      expect(firstItem).toHaveFocus();
      await user.keyboard("{End}");
      expect(lastItem).toHaveFocus();

      await user.keyboard("{Escape}");
      expect(screen.queryByRole("menu")).not.toBeInTheDocument();
      expect(trigger).toHaveFocus();
      expect(windowEscapeObserver).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("keydown", windowEscapeObserver);
    }
  });

  it("opens a multi-vault export menu from its trigger with ArrowDown or ArrowUp", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault-1",
        source_type: "obsidian",
        path: "/Users/test/vault-one",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
      {
        id: "obsidian-vault-2",
        source_type: "obsidian",
        path: "/Users/test/vault-two",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 10,
      },
    ]);
    const user = userEvent.setup();
    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    const trigger = await screen.findByTitle("Export to Obsidian");
    trigger.focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("menuitem", { name: "vault-one" })).toHaveFocus();

    await user.keyboard("{Escape}");
    await user.keyboard("{ArrowUp}");
    expect(screen.getByRole("menuitem", { name: "vault-two" })).toHaveFocus();
  });

  it("exports to selected vault from popover menu", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-vault-1",
        source_type: "obsidian",
        path: "/Users/test/vault-one",
        status: "Active",
        last_sync: null,
        file_count: 10,
        memory_count: 20,
      },
      {
        id: "obsidian-vault-2",
        source_type: "obsidian",
        path: "/Users/test/vault-two",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 10,
      },
    ]);
    vi.mocked(tauri.exportPageToObsidian).mockResolvedValue({
      path: "/Users/test/vault-two/Wenlan/pages/Test Page.md",
    });

    render(
      <PageDetail
        pageId="c1"
        onBack={vi.fn()}
        onMemoryClick={vi.fn()}
      />,
      { wrapper },
    );

    await waitFor(() => {
      expect(screen.getByTitle("Export to Obsidian")).toBeInTheDocument();
    });

    // Open popover
    fireEvent.click(screen.getByTitle("Export to Obsidian"));

    await waitFor(() => {
      expect(screen.getByText("vault-two")).toBeInTheDocument();
    });

    // Click the second vault
    fireEvent.click(screen.getByText("vault-two"));

    await waitFor(() => {
      expect(tauri.exportPageToObsidian).toHaveBeenCalledWith(
        "c1",
        "/Users/test/vault-two/Wenlan/pages",
      );
    });
  });
});
