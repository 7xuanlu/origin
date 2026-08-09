// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../../lib/tauri", () => ({
  acceptPendingRevision: vi.fn(),
  agentDisplayName: (slug: string | null) => slug,
  confirmSpace: vi.fn(),
  deleteFileChunks: vi.fn(),
  deleteSpace: vi.fn(),
  dismissPendingRevision: vi.fn(),
  FACET_COLORS: {},
  getNurtureCards: vi.fn().mockResolvedValue([]),
  getPendingRevision: vi.fn().mockResolvedValue(null),
  getSpace: vi.fn(),
  getVersionChain: vi.fn().mockResolvedValue([]),
  listEntities: vi.fn(),
  listMemoriesRich: vi.fn(),
  listPages: vi.fn(),
  pinMemory: vi.fn().mockResolvedValue(undefined),
  setStability: vi.fn(),
  STABILITY_TIERS: {},
  unpinMemory: vi.fn().mockResolvedValue(undefined),
  updateMemory: vi.fn(),
  updateSpace: vi.fn(),
}));

import {
  confirmSpace,
  deleteSpace,
  getSpace,
  listEntities,
  listMemoriesRich,
  listPages,
  updateSpace,
} from "../../../lib/tauri";
import SpaceDetail from "../SpaceDetail";
import { SPACE_DETAIL_TEST_COPY } from "./testTranslation";

const space = {
  id: "s1",
  name: "Origin",
  description: "A working context",
  suggested: false,
  starred: false,
  sort_order: 0,
  memory_count: 1,
  entity_count: 1,
  created_at: 1_000,
  updated_at: 2_000,
};

const memory = {
  source_id: "m1",
  title: "Memory one",
  content: "Body",
  summary: null,
  memory_type: "fact",
  domain: "Origin",
  source_agent: "codex",
  confidence: 0.9,
  confirmed: true,
  pinned: false,
  supersedes: null,
  last_modified: 2_000,
  chunk_count: 1,
};

const entity = {
  id: "e1",
  name: "Lucian",
  entity_type: "person",
  domain: "Origin",
  source_agent: "codex",
  confidence: 0.8,
  confirmed: true,
  created_at: 1_000,
  updated_at: 2_000,
};

const page = {
  id: "p1",
  title: "Page one",
  summary: "Summary",
  content: "Content",
  entity_id: null,
  domain: "Origin",
  source_memory_ids: ["m1"],
  version: 1,
  status: "active",
  created_at: "2026-07-01T00:00:00Z",
  last_compiled: "2026-07-01T00:00:00Z",
  last_modified: "2026-07-02T00:00:00Z",
};

function renderDetail(overrides: Partial<React.ComponentProps<typeof SpaceDetail>> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const props = {
    copy: SPACE_DETAIL_TEST_COPY,
    spaceName: "Origin",
    onBack: vi.fn(),
    onSelectMemory: vi.fn(),
    onSelectPage: vi.fn(),
    onEntityClick: vi.fn(),
    ...overrides,
  };
  const result = render(
    <QueryClientProvider client={client}>
      <SpaceDetail {...props} />
    </QueryClientProvider>,
  );
  return { ...result, props };
}

describe("SpaceDetail existing behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSpace).mockResolvedValue(space);
    vi.mocked(listMemoriesRich).mockResolvedValue([memory]);
    vi.mocked(listEntities).mockResolvedValue([entity]);
    vi.mocked(listPages).mockResolvedValue([page]);
  });

  it("keeps a suggested space through confirmSpace", async () => {
    vi.mocked(getSpace).mockResolvedValue({ ...space, suggested: true });
    renderDetail();

    fireEvent.click(await screen.findByRole("button", { name: "Keep" }));

    await waitFor(() => expect(confirmSpace).toHaveBeenCalledWith("Origin"));
  });

  it("renames the title on Enter", async () => {
    renderDetail();
    fireEvent.click(await screen.findByRole("button", { name: "Edit space" }));
    const input = screen.getByDisplayValue("Origin");

    fireEvent.change(input, { target: { value: "Planning" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() =>
      expect(updateSpace).toHaveBeenCalledWith("Origin", "Planning", "A working context"),
    );
  });

  it("reports a successful load with the stable Space identity", async () => {
    // Given an integrated successful-load callback
    const onSpaceLoaded = vi.fn();

    // When the dossier query resolves
    renderDetail({ onSpaceLoaded });

    // Then the loaded Space is reported once
    await waitFor(() => expect(onSpaceLoaded).toHaveBeenCalledWith(space));
    expect(onSpaceLoaded).toHaveBeenCalledTimes(1);
  });

  it("reports successful rename and delete mutations by stable id", async () => {
    // Given integrated MRU cleanup callbacks
    const onSpaceRenamed = vi.fn();
    const onSpaceDeleted = vi.fn();
    vi.mocked(updateSpace).mockResolvedValue({ ...space, name: "Planning" });
    vi.mocked(deleteSpace).mockResolvedValue(undefined);
    const first = renderDetail({ onSpaceRenamed });

    // When rename succeeds
    fireEvent.click(await screen.findByRole("button", { name: "Edit space" }));
    fireEvent.change(screen.getByDisplayValue("Origin"), { target: { value: "Planning" } });
    fireEvent.keyDown(screen.getByDisplayValue("Planning"), { key: "Enter" });

    // Then the stable id and new name are reported
    await waitFor(() => expect(onSpaceRenamed).toHaveBeenCalledWith({ id: "s1", name: "Planning" }));
    first.unmount();

    // When delete succeeds on a fresh dossier
    renderDetail({ onSpaceDeleted });
    fireEvent.click(await screen.findByRole("button", { name: "Actions for Origin" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Delete space" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm delete space" }));

    // Then the deleted stable id is reported
    await waitFor(() => expect(onSpaceDeleted).toHaveBeenCalledWith("s1"));
  });

  it("edits the description on Cmd+Enter", async () => {
    renderDetail();
    fireEvent.click(await screen.findByRole("button", { name: "Edit space" }));
    const textarea = screen.getByRole("textbox", { name: "Edit description" });

    fireEvent.change(textarea, { target: { value: "New context" } });
    fireEvent.keyDown(textarea, { key: "Enter", metaKey: true });

    await waitFor(() =>
      expect(updateSpace).toHaveBeenCalledWith("Origin", "Origin", "New context"),
    );
  });

  it("opens a page row", async () => {
    const onSelectPage = vi.fn();
    renderDetail({ onSelectPage });

    fireEvent.click(await screen.findByRole("button", { name: /Page one/ }));

    expect(onSelectPage).toHaveBeenCalledWith("p1");
  });

  it("opens an entity chip", async () => {
    const onEntityClick = vi.fn();
    renderDetail({ onEntityClick });

    fireEvent.click(await screen.findByRole("button", { name: "Lucian" }));

    expect(onEntityClick).toHaveBeenCalledWith("e1");
  });

  it("keeps raw memories collapsed and preserves its sort controls", async () => {
    renderDetail();
    expect(screen.queryByText("Memory one")).not.toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "Raw memories (1)" }));
    fireEvent.click(await screen.findByRole("button", { name: "Curated" }));
    fireEvent.click(screen.getByRole("button", { name: /Oldest first/ }));

    expect(screen.getByText("Memory one")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Oldest" })).toBeInTheDocument();
  });

  it("renders distinct loading, error, and missing-space states", async () => {
    vi.mocked(getSpace).mockImplementation(() => new Promise(() => {}));
    const loading = renderDetail();
    expect(screen.getByRole("status")).toHaveTextContent("Loading space");
    loading.unmount();

    vi.mocked(getSpace).mockRejectedValue(new Error("offline"));
    const failed = renderDetail();
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not open space");
    failed.unmount();

    vi.mocked(getSpace).mockResolvedValue(null);
    renderDetail();
    expect(await screen.findByText("Space not found")).toBeInTheDocument();
  });
});
