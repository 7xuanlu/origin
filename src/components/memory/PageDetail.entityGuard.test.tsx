// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "./PageDetail";

vi.mock("../../lib/tauri", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/tauri")>()),
  getPage: vi.fn().mockResolvedValue({
    id: "concept_abc",
    title: "libSQL Architecture",
    summary: "Core database layer",
    content: "libSQL is the core database layer.",
    entity_id: null,
    domain: null,
    source_memory_ids: ["mem_1"],
    version: 1,
    status: "active",
    created_at: "2026-04-01T00:00:00+00:00",
    last_compiled: "2026-04-07T12:00:00+00:00",
    last_modified: "2026-04-07T12:00:00+00:00",
  }),
  getPageSources: vi.fn().mockResolvedValue([]),
  getPageLinks: vi.fn().mockResolvedValue({ outbound: [], inbound: [] }),
  getPageRevisions: vi.fn().mockResolvedValue({ entries: [], user_edited: false }),
  listRegisteredSources: vi.fn().mockResolvedValue([]),
  getEntityDetail: vi.fn().mockResolvedValue(null),
  pageReviewSupported: vi.fn().mockResolvedValue("daemon_unsupported"),
  reviewPage: vi.fn(),
  deletePage: vi.fn(),
}));

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

function renderDetail(onEntityClick = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return {
    onEntityClick,
    user: userEvent.setup(),
    ...render(
      <QueryClientProvider client={client}>
        <PageDetail
          pageId="concept_abc"
          onBack={vi.fn()}
          onMemoryClick={vi.fn()}
          onEntityClick={onEntityClick}
        />
      </QueryClientProvider>,
    ),
  };
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => cleanup());

describe("PageDetail entity-guard delete error", () => {
  it("renders the daemon's guard message with a link to the entity dossier", async () => {
    const { deletePage } = await import("../../lib/tauri");
    (deletePage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error(
        'invoke failed: {"error":"This page belongs to the entity \'Ada Lovelace\'. Archive or delete it from Entities","entity_id":"entity-ada"}',
      ),
    );
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { onEntityClick, user } = renderDetail();

    try {
      await screen.findByText("libSQL Architecture");
      await user.click(screen.getByRole("button", { name: "Page actions" }));
      await user.click(screen.getByRole("menuitem", { name: "Delete page" }));

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("This page belongs to the entity 'Ada Lovelace'");

      const link = screen.getByRole("button", { name: "Open the entity" });
      await user.click(link);
      expect(onEntityClick).toHaveBeenCalledWith("entity-ada");
    } finally {
      confirmSpy.mockRestore();
    }
  });

  it("falls back to the generic delete error when the daemon rejection carries no entity_id", async () => {
    const { deletePage } = await import("../../lib/tauri");
    (deletePage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("daemon offline"));
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { user } = renderDetail();

    try {
      await screen.findByText("libSQL Architecture");
      await user.click(screen.getByRole("button", { name: "Page actions" }));
      await user.click(screen.getByRole("menuitem", { name: "Delete page" }));

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("Could not delete this page. Try again.");
      expect(screen.queryByRole("button", { name: "Open the entity" })).toBeNull();
    } finally {
      confirmSpy.mockRestore();
    }
  });
});
