// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MemoryDetail from "./MemoryDetail";
import * as tauri from "../../lib/tauri";

vi.mock("../../lib/tauri");

const memory: tauri.MemoryItem = {
  source_id: "mem-1",
  title: "Memory",
  content: "A memory",
  summary: null,
  memory_type: "fact",
  domain: null,
  source_agent: null,
  confidence: null,
  confirmed: false,
  pinned: false,
  supersedes: null,
  last_modified: 1,
  chunk_count: 1,
};

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("MemoryDetail enrichment status", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue(memory);
    vi.mocked(tauri.listSpaces).mockResolvedValue([]);
    vi.mocked(tauri.listEntities).mockResolvedValue([]);
    vi.mocked(tauri.listAllTags).mockResolvedValue({
      tags: [],
      document_tags: {},
      categories: [],
      document_categories: {},
    });
    vi.mocked(tauri.setDocumentTags).mockResolvedValue([]);
    vi.mocked(tauri.suggestTags).mockResolvedValue([]);
    vi.mocked(tauri.search).mockResolvedValue([]);
    vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("old daemon"));
    vi.mocked(tauri.getVersionChain).mockResolvedValue([]);
    vi.mocked(tauri.getPendingRevision).mockResolvedValue(null);
    vi.mocked(tauri.acceptPendingRevision).mockResolvedValue({
      target_source_id: "mem-1",
      revision_source_id: "rev-1",
      wrote: true,
    });
    vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
      current_source_id: "mem-1",
      chain_depth: 1,
      entries: [],
    });
  });

  it("shows daemon enrichment status without blocking the memory body", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockResolvedValue({
      source_id: "mem-1",
      summary: "complete",
      steps: [{ step: "classify", status: "done", error: null, attempts: 1 }],
    });

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/enrichment/i)).toBeInTheDocument();
      expect(screen.getByText(/complete/i)).toBeInTheDocument();
    });
  });

  it("keeps a navigable shell while memory detail is loading", () => {
    vi.mocked(tauri.getMemoryDetail).mockReturnValue(
      new Promise<tauri.MemoryItem | null>(() => {}),
    );

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    const dossier = screen.getByRole("main", { name: /memory dossier/i });
    expect(within(dossier).getByRole("button", { name: /back to memories/i })).toBeInTheDocument();
  });

  it("renders the quiet dossier structure without losing core controls", async () => {
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue({
      ...memory,
      title: "Durable design note",
      content: "Primary reading content",
      memory_type: "preference",
      domain: "work",
      source_agent: "codex",
      confirmed: true,
      pinned: true,
      structured_fields: JSON.stringify({ source: "design review" }),
    });
    vi.mocked(tauri.listAllTags).mockResolvedValue({
      tags: ["design", "workflow"],
      document_tags: { "memory::mem-1": ["design"] },
      categories: [],
      document_categories: {},
    });
    vi.mocked(tauri.getEnrichmentStatus).mockResolvedValue({
      source_id: "mem-1",
      summary: "complete",
      steps: [{ step: "classify", status: "done", error: null, attempts: 1 }],
    });
    vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
      current_source_id: "mem-1",
      chain_depth: 2,
      entries: [
        {
          source_id: "mem-1",
          depth: 0,
          title: "Durable design note",
          content_preview: "Current version",
          last_modified: 10,
          delta_summary: "Current dossier copy",
        },
      ],
    });
    vi.mocked(tauri.listEntities).mockResolvedValue([
      {
        id: "entity-1",
        name: "Wenlan",
        entity_type: "project",
        domain: "work",
        source_agent: null,
        confidence: null,
        confirmed: true,
        created_at: 1,
        updated_at: 1,
      },
    ]);
    vi.mocked(tauri.search).mockResolvedValue([
      {
        id: "related-1",
        source_id: "mem-2",
        source: "memory",
        title: "Related memory",
        url: null,
        content: "Related context appears here",
        chunk_index: 0,
        last_modified: 2,
        score: 0.9,
        memory_type: "fact",
      },
    ]);

    const onNavigateEntity = vi.fn();
    const onNavigateMemory = vi.fn();

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={onNavigateEntity}
        onNavigateMemory={onNavigateMemory}
      />,
      { wrapper },
    );

    expect(await screen.findByText("Primary reading content")).toBeInTheDocument();
    const dossier = screen.getByRole("main", { name: /memory dossier/i });
    const backButton = within(dossier).getByRole("button", { name: /back to memories/i });
    const reading = within(dossier).getByRole("region", { name: /memory reading/i });
    // Context rail mounts a tick after the reading column (its related-memories
    // + entities queries resolve separately), so await it — a bare getByRole
    // races on slower CI runners.
    const rail = await within(dossier).findByRole("complementary", { name: /memory context/i });

    expect(backButton.textContent).toBe("");
    expect(backButton.querySelector("svg")).not.toBeNull();
    expect(within(reading).getByText("Primary reading content")).toBeInTheDocument();
    expect(within(dossier).getByRole("button", { name: /edit memory/i })).toBeInTheDocument();
    expect(within(reading).getByRole("button", { name: /edit tags/i })).toBeInTheDocument();
    expect(within(dossier).getByRole("button", { name: /pinned yes/i })).toBeInTheDocument();
    expect(within(dossier).getByRole("button", { name: /confirmed yes/i })).toBeInTheDocument();
    expect(within(reading).getByText(/complete/i)).toBeInTheDocument();
    expect(within(reading).getByText(/current dossier copy/i)).toBeInTheDocument();
    expect(reading).not.toContainElement(rail);
    await userEvent.click(await within(rail).findByRole("button", { name: /wenlan/i }));
    await userEvent.click(await within(rail).findByRole("button", { name: /related context appears here/i }));

    expect(onNavigateEntity).toHaveBeenCalledWith("entity-1");
    expect(onNavigateMemory).toHaveBeenCalledWith("mem-2");
  });

  it("saves edited memory content from the redesigned reading column", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.updateMemory).mockResolvedValue();

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /edit memory/i }));
    const editor = screen.getByRole("textbox");
    expect(screen.getByText("Editing")).toBeInTheDocument();
    expect(editor.closest(".memory-detail-editing-surface")).not.toBeNull();
    await user.clear(editor);
    await user.type(editor, "Updated reading content");
    await user.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(tauri.updateMemory).toHaveBeenCalledWith("mem-1", "Updated reading content");
    });
    expect(screen.queryByText("Editing")).toBeNull();
  });

  it("summarizes long related context until the user expands it", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue({
      ...memory,
      domain: "work",
    });
    vi.mocked(tauri.listEntities).mockResolvedValue(
      Array.from({ length: 5 }, (_, index) => ({
        id: `entity-${index + 1}`,
        name: `Entity ${index + 1}`,
        entity_type: "project",
        domain: "work",
        source_agent: null,
        confidence: null,
        confirmed: true,
        created_at: 1,
        updated_at: 1,
      })),
    );
    vi.mocked(tauri.search).mockResolvedValue(
      Array.from({ length: 5 }, (_, index) => ({
        id: `related-${index + 1}`,
        source_id: `mem-${index + 2}`,
        source: "memory",
        title: `Related memory ${index + 1}`,
        url: null,
        content: `Related memory ${index + 1}`,
        chunk_index: 0,
        last_modified: index + 2,
        score: 0.9,
        memory_type: "fact",
      })),
    );

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    const relatedEntitiesHeading = await screen.findByRole("heading", { name: /related entities/i });
    const relatedEntitiesPanel = relatedEntitiesHeading.closest("section");
    expect(relatedEntitiesPanel).not.toBeNull();
    if (!relatedEntitiesPanel) throw new Error("Missing related entities panel");
    expect(within(relatedEntitiesPanel).getByText("Entity 4")).toBeInTheDocument();
    expect(within(relatedEntitiesPanel).queryByText("Entity 5")).toBeNull();
    const relatedEntitiesShowAll = within(relatedEntitiesPanel).getByRole("button", { name: /show all 5/i });
    expect(relatedEntitiesShowAll).toHaveClass("memory-detail-disclosure-button");
    expect(relatedEntitiesShowAll.querySelector(".memory-detail-disclosure-label")).not.toBeNull();
    expect(relatedEntitiesShowAll.querySelector(".memory-detail-disclosure-count")).not.toBeNull();
    await user.click(relatedEntitiesShowAll);
    expect(within(relatedEntitiesPanel).getByText("Entity 5")).toBeInTheDocument();

    const relatedMemoriesHeading = screen.getByRole("heading", { name: /related memories/i });
    const relatedMemoriesPanel = relatedMemoriesHeading.closest("section");
    expect(relatedMemoriesPanel).not.toBeNull();
    if (!relatedMemoriesPanel) throw new Error("Missing related memories panel");
    expect(within(relatedMemoriesPanel).getByText("Related memory 3")).toBeInTheDocument();
    expect(within(relatedMemoriesPanel).queryByText("Related memory 4")).toBeNull();
    const relatedMemoriesShowAll = within(relatedMemoriesPanel).getByRole("button", { name: /show all 5/i });
    expect(relatedMemoriesShowAll).toHaveClass("memory-detail-disclosure-button");
    expect(relatedMemoriesShowAll.querySelector(".memory-detail-disclosure-label")).not.toBeNull();
    expect(relatedMemoriesShowAll.querySelector(".memory-detail-disclosure-count")).not.toBeNull();
    await user.click(relatedMemoriesShowAll);
    expect(within(relatedMemoriesPanel).getByText("Related memory 5")).toBeInTheDocument();
  });

  it("keeps rendering the memory when enrichment status route is unavailable", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("404"));

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();
    expect(screen.queryByText(/enrichment/i)).toBeNull();
  });

  it("renders daemon memory revision history", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("old daemon"));
    vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
      current_source_id: "mem-1",
      chain_depth: 2,
      entries: [
        {
          source_id: "mem-1",
          depth: 0,
          title: "Current",
          content_preview: "Current version",
          last_modified: 10,
          source_agent: "claude-code",
          supersede_mode: "protected_revision",
          delta_summary: "Clarified wording",
        },
      ],
    });

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText(/revision history/i)).toBeInTheDocument();
    expect(screen.getByText(/clarified wording/i)).toBeInTheDocument();
  });

  it("keeps rendering the memory when memory revisions route is unavailable", async () => {
    vi.mocked(tauri.getEnrichmentStatus).mockRejectedValue(new Error("old daemon"));
    vi.mocked(tauri.getMemoryRevisions).mockRejectedValue(new Error("404"));

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();
    expect(screen.queryByText(/revision history/i)).toBeNull();
  });

  it("refreshes daemon memory revision history after accepting a pending revision", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue({
      ...memory,
      memory_type: "identity",
      confirmed: true,
    });
    vi.mocked(tauri.getPendingRevision).mockResolvedValue({
      source_id: "rev-1",
      content: "Proposed protected update",
      source_agent: "claude-code",
    });

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText(/proposed protected update/i)).toBeInTheDocument();
    expect(tauri.getMemoryRevisions).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /accept/i }));

    await waitFor(() => {
      expect(tauri.acceptPendingRevision).toHaveBeenCalledWith("mem-1");
      expect(tauri.getMemoryRevisions).toHaveBeenCalledTimes(2);
    });
  });

  it("invalidates tag inventory after editing tags", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    vi.mocked(tauri.setDocumentTags).mockResolvedValue(["reviewed"]);

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryDetail
          sourceId="mem-1"
          onBack={vi.fn()}
          onNavigateEntity={vi.fn()}
          onNavigateMemory={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("A memory")).toBeInTheDocument();

    await user.click(screen.getByTitle("Edit tags"));
    await user.type(screen.getByPlaceholderText("Add a tag..."), "reviewed{enter}");

    await waitFor(() => {
      expect(tauri.setDocumentTags).toHaveBeenCalledWith("memory", "mem-1", ["reviewed"]);
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["tags"] });
    });
  });

  it("uses daemon memory revision history instead of the legacy version panel", async () => {
    vi.mocked(tauri.getMemoryDetail).mockResolvedValue({
      ...memory,
      supersedes: "mem-old",
    });
    vi.mocked(tauri.getVersionChain).mockResolvedValue([
      {
        source_id: "mem-old",
        title: "Legacy version",
        content: "Old content",
        memory_type: "fact",
        confirmed: true,
        supersedes: null,
        last_modified: 5,
      },
    ]);
    vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
      current_source_id: "mem-1",
      chain_depth: 2,
      entries: [
        {
          source_id: "mem-1",
          depth: 0,
          title: "Current",
          content_preview: "Current version",
          last_modified: 10,
          delta_summary: "Clarified wording",
        },
      ],
    });

    render(
      <MemoryDetail
        sourceId="mem-1"
        onBack={vi.fn()}
        onNavigateEntity={vi.fn()}
        onNavigateMemory={vi.fn()}
      />,
      { wrapper },
    );

    expect(await screen.findByText(/revision history/i)).toBeInTheDocument();
    expect(screen.getByText(/clarified wording/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(tauri.getVersionChain).toHaveBeenCalledWith("mem-1");
    });
    expect(screen.queryByText(/version history/i)).toBeNull();
    expect(screen.queryByText(/legacy version/i)).toBeNull();
  });
});
