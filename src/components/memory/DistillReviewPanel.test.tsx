// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import DistillReviewPanel from "./DistillReviewPanel";
import { DISTILL_REVIEW_SESSION_QUERY_KEY } from "./pages/pageReviewSignals";
import {
  EXAMPLE_REVIEW_ITEMS,
  exampleReviewLabel,
  isExampleReviewItem,
  seedReviewExampleCaches,
} from "./reviewExamples";
import type {
  DistillReviewResponse,
  MemoryItem,
  PendingRevisionItem,
} from "../../lib/tauri";

vi.mock("../../lib/tauri", async () => {
  const actual = await vi.importActual<typeof import("../../lib/tauri")>("../../lib/tauri");
  return {
    ...actual,
    distillReview: vi.fn(),
    listPendingRevisions: vi.fn(),
    acceptPendingRevision: vi.fn(),
    dismissPendingRevision: vi.fn(),
    listRefinements: vi.fn(),
    acceptRefinement: vi.fn(),
    rejectRefinement: vi.fn(),
    listUnconfirmedMemories: vi.fn(),
    confirmMemory: vi.fn(),
    deleteMemory: vi.fn(),
    getMemoryDetail: vi.fn(),
    getEntityDetail: vi.fn(),
    redistillPage: vi.fn(),
    search: vi.fn(),
    listRecentChanges: vi.fn(),
  };
});

import {
  distillReview,
  listPendingRevisions,
  acceptPendingRevision,
  dismissPendingRevision,
  listRefinements,
  acceptRefinement,
  rejectRefinement,
  listUnconfirmedMemories,
  confirmMemory,
  deleteMemory,
  getMemoryDetail,
  getEntityDetail,
  redistillPage,
  search,
  listRecentChanges,
} from "../../lib/tauri";

function renderPanel(props: Partial<React.ComponentProps<typeof DistillReviewPanel>> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onBack = props.onBack ?? vi.fn();
  const onPageClick = props.onPageClick ?? vi.fn();
  const onMemoryClick = props.onMemoryClick ?? vi.fn();
  const user = userEvent.setup();
  render(
    <QueryClientProvider client={client}>
      <DistillReviewPanel onBack={onBack} onPageClick={onPageClick} onMemoryClick={onMemoryClick} />
    </QueryClientProvider>,
  );
  return { user, client, onBack, onPageClick, onMemoryClick };
}

function truncateForTest(value: string, max: number): string {
  if (max <= 0) return "";
  if (value.length <= max) return value;
  if (max <= 3) return ".".repeat(max);
  return `${value.slice(0, max - 3).trimEnd()}...`;
}

function memory(overrides: Partial<MemoryItem> & Pick<MemoryItem, "source_id" | "title" | "content">): MemoryItem {
  return {
    summary: null,
    memory_type: null,
    domain: null,
    source_agent: null,
    confidence: null,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: 1_760_000_000,
    chunk_count: 1,
    ...overrides,
  };
}

function revision(
  overrides: Partial<PendingRevisionItem> & Pick<PendingRevisionItem, "target_source_id" | "revision_content">,
): PendingRevisionItem {
  return {
    revision_source_id: `${overrides.target_source_id}_rev`,
    source_agent: "claude-code",
    last_modified: 1_760_000_000,
    ...overrides,
  };
}

const fallbackSource =
  "Fallback content label should appear when no title or entity exists, and this preview detail must stay visible in the review panel source list even when it becomes the cluster title.";
const fallbackSecondSource =
  "A second fallback source confirms the preview keeps the first two contents.";

const reviewPayload: DistillReviewResponse = {
  pages_created: 0,
  scoped: false,
  created_ids: [],
  pending: [
    {
      source_ids: ["mem_1", "mem_2"],
      contents: [
        "This is a detailed source memory about temporal page refresh behavior.",
        "A second source memory adds routing context for the distill review panel.",
      ],
      entity_id: "entity_temporal",
      entity_name: "Temporal refresh",
      space: "Engineering",
      estimated_tokens: 180,
      centroid_embedding: [0.1, 0.2],
      existing_page_id: "page_temporal",
      existing_page_title: "Temporal page refresh",
      new_memory_count: 1,
    },
    {
      source_ids: ["mem_3"],
      contents: [fallbackSource, fallbackSecondSource],
      estimated_tokens: 80,
    },
  ],
  stale_pages: [
    {
      page_id: "page_stale",
      title: "Retrieval Pipeline",
      summary: "Source memories changed after the page compiled.",
      source_memory_ids: ["mem_old"],
      sources_updated_count: 3,
      stale_reason: "source_updated",
      user_edited: false,
    },
  ],
  stale_truncated: true,
  orphan_topics: [{ label: "Vector clocks", count: 4 }],
};

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  vi.mocked(distillReview).mockResolvedValue(reviewPayload);
  vi.mocked(listRecentChanges).mockResolvedValue([]);
  vi.mocked(listPendingRevisions).mockResolvedValue([]);
  vi.mocked(listRefinements).mockResolvedValue({ proposals: [] });
  vi.mocked(listUnconfirmedMemories).mockResolvedValue([]);
  vi.mocked(confirmMemory).mockResolvedValue(undefined);
  vi.mocked(deleteMemory).mockResolvedValue(undefined);
  vi.mocked(acceptPendingRevision).mockResolvedValue({
    target_source_id: "mem_target",
    revision_source_id: "mem_target_rev",
    wrote: true,
  });
  vi.mocked(dismissPendingRevision).mockResolvedValue({
    target_source_id: "mem_target",
    wrote: true,
  });
  vi.mocked(acceptRefinement).mockResolvedValue({ id: "prop_1", action_applied: "entity_merge" });
  vi.mocked(rejectRefinement).mockResolvedValue({ id: "prop_1" });
  vi.mocked(getMemoryDetail).mockResolvedValue(
    memory({ source_id: "mem_target", title: "Target memory", content: "Prefers npm for installs" }),
  );
  vi.mocked(search).mockResolvedValue([]);
  vi.mocked(redistillPage).mockResolvedValue({
    status: "ok",
    updated: true,
    hint: null,
  } as any);
  vi.mocked(getEntityDetail).mockResolvedValue({
    entity: {
      id: "ent_1",
      name: "Visual Studio Code",
      entity_type: "tool",
      domain: null,
      source_agent: null,
      confidence: null,
      confirmed: true,
      created_at: 0,
      updated_at: 0,
    },
    observations: [],
    relations: [],
  });
});

describe("DistillReviewPanel", () => {
  it("loads the page review once on mount", async () => {
    const { client } = renderPanel();

    await waitFor(() => {
      expect(distillReview).toHaveBeenCalledTimes(1);
    });
    expect(client.getQueryData(DISTILL_REVIEW_SESSION_QUERY_KEY)).toEqual(reviewPayload);
    expect(client.getQueryDefaults(DISTILL_REVIEW_SESSION_QUERY_KEY)).toMatchObject({
      gcTime: Infinity,
      staleTime: Infinity,
    });
    expect(await screen.findByRole("heading", { name: "Review" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^refresh$/i })).toBeInTheDocument();
  });

  it("renders page review sections after loading", async () => {
    renderPanel();

    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "New page candidates" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Pages with new sources" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "New topics" })).toBeInTheDocument();
    expect(screen.getByText(/1 new source/)).toBeInTheDocument();
    expect(screen.getByText(truncateForTest(fallbackSource, 72))).toBeInTheDocument();
    expect(screen.getByText("Retrieval Pipeline")).toBeInTheDocument();
    expect(screen.getByText(/first 10 stale pages/i)).toBeInTheDocument();
    expect(screen.getByText("Vector clocks")).toBeInTheDocument();
    expect(screen.getByText(/4 mentions/)).toBeInTheDocument();
  });

  it("renders source previews even when a fallback label comes from the first source", async () => {
    renderPanel();

    expect(await screen.findByText(truncateForTest(fallbackSource, 72))).toBeInTheDocument();
    expect(screen.getByText(truncateForTest(fallbackSource, 140))).toBeInTheDocument();
    expect(screen.getByText(fallbackSecondSource)).toBeInTheDocument();
  });

  it("opens a stale page card and refreshes it through redistill", async () => {
    const { user, onPageClick } = renderPanel();

    await user.click(
      await screen.findByRole("button", { name: /Review Retrieval Pipeline/ }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Page refresh")).toBeInTheDocument();
    expect(within(dialog).getByText("3 sources updated")).toBeInTheDocument();
    // No dismiss verb exists for stale pages, so the dialog never offers one.
    expect(within(dialog).queryByRole("button", { name: "Dismiss" })).toBeNull();

    await user.click(within(dialog).getByRole("button", { name: "Open page" }));
    expect(onPageClick).toHaveBeenCalledWith("page_stale");

    await user.click(within(dialog).getByRole("button", { name: "Refresh page" }));
    await waitFor(() => {
      expect(redistillPage).toHaveBeenCalledWith("page_stale");
    });
  });

  it("keeps the last successful result visible after refresh failure", async () => {
    vi.mocked(distillReview)
      .mockResolvedValueOnce(reviewPayload)
      .mockRejectedValueOnce(new Error("HTTP POST /api/distill returned 500"));
    const { user } = renderPanel();

    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(await screen.findByText(/HTTP POST \/api\/distill returned 500/)).toBeInTheDocument();
    expect(screen.getByText("Temporal page refresh")).toBeInTheDocument();
  });

  it("keeps manual refresh available after the initial load", async () => {
    const { user } = renderPanel();

    await screen.findByText("Temporal page refresh");
    await user.click(screen.getByRole("button", { name: /^refresh$/i }));

    expect(distillReview).toHaveBeenCalledTimes(2);
  });
});

describe("DistillReviewPanel review queue", () => {
  it("shows all-caught-up only when the queue and discovery are both empty", async () => {
    vi.mocked(distillReview).mockResolvedValue({
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    });
    renderPanel();

    expect(await screen.findByRole("heading", { name: "All caught up" })).toBeInTheDocument();
  });

  it("does not claim all-caught-up while discovery items exist", async () => {
    renderPanel();

    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "All caught up" })).toBeNull();
  });

  it("does not claim all-caught-up when the review queue fails to load, and retry reuses the existing refresh", async () => {
    vi.mocked(distillReview).mockResolvedValue({
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    });
    vi.mocked(listRefinements).mockRejectedValue(
      new Error("HTTP GET /api/refinery/queue returned 500"),
    );

    const { user } = renderPanel();

    expect(
      await screen.findByRole("heading", { name: "Couldn't load review items." }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "All caught up" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(listRefinements).toHaveBeenCalledTimes(2);
    expect(distillReview).toHaveBeenCalledTimes(2);
  });

  it("shows the queue plus a quiet partial-load notice, no retry button, when items exist alongside a queue error", async () => {
    vi.mocked(distillReview).mockResolvedValue({
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    });
    vi.mocked(listRefinements).mockRejectedValue(
      new Error("HTTP GET /api/refinery/queue returned 500"),
    );
    vi.mocked(listPendingRevisions).mockResolvedValue([
      revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
    ]);

    renderPanel();

    expect(await screen.findByRole("heading", { name: "Memory revisions" })).toBeInTheDocument();
    expect(
      await screen.findByText(/Some review items couldn't load/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Couldn't load review items." })).toBeNull();
  });

  it("renders revision cards with a pending count", async () => {
    vi.mocked(listPendingRevisions).mockResolvedValue([
      revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
    ]);
    renderPanel();

    expect(await screen.findByRole("heading", { name: "Memory revisions" })).toBeInTheDocument();
    expect(screen.getByText("1 to decide")).toBeInTheDocument();
    // The card titles itself with the target memory's real name once fetched.
    expect(await screen.findByRole("button", { name: /Review Target memory/ })).toBeInTheDocument();
  });

  it("opens the diff dialog from a card and shows stripped and added words", async () => {
    vi.mocked(listPendingRevisions).mockResolvedValue([
      revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
    ]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: /Review Target memory/ }));

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByRole("heading", { name: "Target memory" })).toBeInTheDocument();
    await waitFor(() => {
      const dels = dialog.querySelectorAll("del");
      const inss = dialog.querySelectorAll("ins");
      expect([...dels].some((el) => el.textContent?.includes("npm"))).toBe(true);
      expect([...inss].some((el) => el.textContent?.includes("pnpm"))).toBe(true);
    });
    // 1 revision + 1 stale page + 2 page candidates + 1 topic share the list.
    expect(within(dialog).getByText("1 of 5")).toBeInTheDocument();
  });

  it("approves the last revision and shows the caught-up pane when nothing else is queued", async () => {
    vi.mocked(distillReview).mockResolvedValue({
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    });
    vi.mocked(listPendingRevisions)
      .mockResolvedValueOnce([
        revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
      ])
      .mockResolvedValue([]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: /Review Target memory/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(acceptPendingRevision).toHaveBeenCalledWith("mem_target");
    });
    expect(await within(dialog).findByText("Every pending change has been reviewed.")).toBeInTheDocument();
  });

  it("advances into discovery items after the last decision resolves", async () => {
    vi.mocked(listPendingRevisions)
      .mockResolvedValueOnce([
        revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
      ])
      .mockResolvedValue([]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: /Review Target memory/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(acceptPendingRevision).toHaveBeenCalledWith("mem_target");
    });
    // The stale page now precedes discovery items in the dialog list.
    expect(
      await within(dialog).findByRole("heading", { name: "Retrieval Pipeline" }),
    ).toBeInTheDocument();
  });

  it("dismisses a revision through the dialog", async () => {
    vi.mocked(listPendingRevisions)
      .mockResolvedValueOnce([
        revision({ target_source_id: "mem_target", revision_content: "Prefers pnpm for installs" }),
      ])
      .mockResolvedValue([]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: /Review Target memory/ }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Dismiss" }));

    await waitFor(() => {
      expect(dismissPendingRevision).toHaveBeenCalledWith("mem_target");
    });
  });

  it("advances to the next item after approving", async () => {
    const first = revision({ target_source_id: "mem_a", revision_content: "First revision content" });
    const second = revision({ target_source_id: "mem_b", revision_content: "Second revision content" });
    vi.mocked(listPendingRevisions)
      .mockResolvedValueOnce([first, second])
      .mockResolvedValue([second]);
    vi.mocked(getMemoryDetail).mockImplementation(async (sourceId: string) =>
      memory({ source_id: sourceId, title: `Title ${sourceId}`, content: "old content" }),
    );
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: /Review Title mem_a/ }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByRole("heading", { name: "Title mem_a" })).toBeInTheDocument();
    // 2 revisions + 1 stale page + 2 page candidates + 1 topic share the list.
    expect(within(dialog).getByText("1 of 6")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    expect(await within(dialog).findByRole("heading", { name: "Title mem_b" })).toBeInTheDocument();
    await waitFor(() => {
      expect(acceptPendingRevision).toHaveBeenCalledWith("mem_a");
    });
  });

  it("renders refinement proposals and approves through the daemon verb", async () => {
    vi.mocked(listRefinements)
      .mockResolvedValueOnce({
        proposals: [
          {
            id: "prop_1",
            action: "entity_merge",
            source_ids: ["ent_1", "ent_2"],
            payload: { action: "entity_merge", existing_id: "ent_1", new_id: "ent_2", similarity: 0.94 },
            confidence: 0.94,
            created_at: "2026-07-09T00:00:00Z",
          },
        ],
      })
      .mockResolvedValue({ proposals: [] });
    const { user } = renderPanel();

    expect(await screen.findByRole("heading", { name: "Merges & suggestions" })).toBeInTheDocument();
    // Both entity ids resolve through the same mocked detail, so the card
    // titles itself with the fetched names instead of the bare kind label.
    await user.click(
      await screen.findByRole("button", { name: /look like the same entity/ }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("94% confidence")).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(acceptRefinement).toHaveBeenCalledWith("prop_1");
    });
  });

  it("keeps new-memory captures off the review page", async () => {
    vi.mocked(listUnconfirmedMemories).mockResolvedValue([
      {
        kind: "memory",
        id: "mem-capture",
        title: "User prefers pnpm over npm",
        snippet: "Stated while setting up the monorepo.",
        timestamp_ms: 1_782_365_080_000,
        badge: { kind: "needs_review" },
      },
    ]);
    renderPanel();

    expect(await screen.findByText("Temporal page refresh")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "New memories" })).toBeNull();
    expect(screen.queryByText("User prefers pnpm over npm")).toBeNull();
    expect(screen.queryByText(/to decide/)).toBeNull();
    expect(confirmMemory).not.toHaveBeenCalled();
    expect(deleteMemory).not.toHaveBeenCalled();
  });

  it("opens a read-only dialog from a new page candidate card", async () => {
    const { user, onPageClick } = renderPanel();

    await user.click(
      await screen.findByRole("button", { name: "Review Temporal page refresh" }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByRole("heading", { name: "Temporal page refresh" }),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("New page")).toBeInTheDocument();
    expect(within(dialog).getByText(/next compile pass/)).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(within(dialog).queryByRole("button", { name: "Dismiss" })).toBeNull();
    // 1 stale page + 2 page candidates + 1 topic; the stale page comes first.
    expect(within(dialog).getByText("2 of 4")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Open page" }));
    expect(onPageClick).toHaveBeenCalledWith("page_temporal");
  });

  it("opens a read-only dialog from a new topic card", async () => {
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: "Review Vector clocks" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("New topic")).toBeInTheDocument();
    expect(
      within(dialog).getByText("4 mentions · no page covers it yet"),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText("Mentioned across memories, but no page covers it yet."),
    ).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Approve" })).toBeNull();
    expect(within(dialog).queryByRole("button", { name: "Dismiss" })).toBeNull();
    // No search hits (default mock) — the evidence pane says so honestly
    // instead of rendering nothing.
    expect(within(dialog).getByText("Mentioned in")).toBeInTheDocument();
    expect(
      await within(dialog).findByText("No related memories found."),
    ).toBeInTheDocument();
  });

  it("shows search evidence and its honesty caveat in a new topic dialog", async () => {
    vi.mocked(search).mockResolvedValue([
      {
        id: "mem_vc-search",
        content: "Vector clocks order events across replicas without a shared clock.",
        source: "memory",
        source_id: "mem_vc",
        title: "Vector clocks note",
        url: null,
        chunk_index: 0,
        last_modified: 1_760_000_000,
        score: 0.9,
      },
    ]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: "Review Vector clocks" }));

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("Vector clocks note")).toBeInTheDocument();
    expect(
      within(dialog).getByText(/Found by search/),
    ).toBeInTheDocument();
  });

  it("opens a read-only dialog from a new entity suggestion card and surfaces search evidence", async () => {
    vi.mocked(listRefinements)
      .mockResolvedValueOnce({
        proposals: [
          {
            id: "prop_entity",
            action: "suggest_entity",
            source_ids: ["mem_x", "mem_y"],
            payload: { action: "suggest_entity", name_hint: "Tauri" },
            confidence: 0.8,
            created_at: "2026-07-09T00:00:00Z",
          },
        ],
      })
      .mockResolvedValue({ proposals: [] });
    vi.mocked(search).mockResolvedValue([
      {
        id: "mem_x-search",
        content: "Tauri wraps a Rust backend with a web frontend.",
        source: "memory",
        source_id: "mem_x",
        title: "Tauri overview",
        url: null,
        chunk_index: 0,
        last_modified: 1_760_000_000,
        score: 0.9,
      },
    ]);
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: "Review Tauri" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "Tauri" })).toBeInTheDocument();
    expect(within(dialog).getByText("Entity suggestion")).toBeInTheDocument();
    expect(await within(dialog).findByText("Tauri overview")).toBeInTheDocument();
    expect(
      within(dialog).getByText(/Found by search/),
    ).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("hides a topic through the dialog, persists it to localStorage, and keeps it hidden across a refresh", async () => {
    const { user } = renderPanel();

    await user.click(await screen.findByRole("button", { name: "Review Vector clocks" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Hide" }));

    await waitFor(() => {
      expect(screen.queryByRole("heading", { name: "New topics" })).toBeNull();
    });
    const stored = JSON.parse(localStorage.getItem("wenlan.review.hidden.v1") ?? "[]");
    expect(stored).toHaveLength(1);
    expect(stored[0]).toMatchObject({ key: "topic:Vector clocks", kind: "topic" });

    // Refresh re-fetches the same distill payload — unlike resolvedStaleIds,
    // the hide survives a fresh lastResult instead of resetting with it.
    await user.click(screen.getByRole("button", { name: /^refresh$/i }));
    await waitFor(() => {
      expect(distillReview).toHaveBeenCalledTimes(2);
    });
    expect(screen.queryByRole("heading", { name: "New topics" })).toBeNull();
  });

  it("restores a hidden item from the Hidden footer", async () => {
    localStorage.setItem(
      "wenlan.review.hidden.v1",
      JSON.stringify([
        { key: "topic:Vector clocks", label: "Vector clocks", kind: "topic", at: Date.now() },
      ]),
    );
    const { user } = renderPanel();

    await screen.findByText("Temporal page refresh");
    expect(screen.queryByRole("heading", { name: "New topics" })).toBeNull();

    await user.click(await screen.findByRole("button", { name: /1 hidden/ }));
    await user.click(await screen.findByRole("button", { name: "Restore" }));

    expect(await screen.findByRole("heading", { name: "New topics" })).toBeInTheDocument();
    expect(screen.getByText("Vector clocks")).toBeInTheDocument();
    const stored = JSON.parse(localStorage.getItem("wenlan.review.hidden.v1") ?? "[]");
    expect(stored).toHaveLength(0);
  });

  it("renders the revisions section's empty state when nothing is pending", async () => {
    renderPanel();

    await screen.findByRole("heading", { name: "Memory revisions" });
    expect(screen.getByText("Nothing waiting for approval.")).toBeInTheDocument();
    expect(
      screen.getByText(/before\/after diff lands here first/),
    ).toBeInTheDocument();
  });

  it("renders the Recent revisions section from listRecentChanges", async () => {
    vi.mocked(listRecentChanges).mockResolvedValue([
      {
        page_id: "page_recent",
        title: "Threads Metrics And Postmortem Schema",
        change_kind: "revised",
        changed_at_ms: Date.now() - 2 * 60 * 60 * 1000,
      },
    ]);
    const { user, onPageClick } = renderPanel();

    await screen.findByRole("heading", { name: "Recent revisions" });
    const row = await screen.findByRole("button", {
      name: /Threads Metrics And Postmortem Schema/,
    });
    await user.click(row);
    expect(onPageClick).toHaveBeenCalledWith("page_recent");
  });
});

describe("DistillReviewPanel review filter", () => {
  const emptyDistill: DistillReviewResponse = {
    pages_created: 0,
    scoped: false,
    created_ids: [],
    pending: [],
    stale_pages: [],
    stale_truncated: false,
    orphan_topics: [],
  };

  it("shows only All and Revisions chips when the queue is empty", async () => {
    vi.mocked(distillReview).mockResolvedValue(emptyDistill);
    renderPanel();

    const group = await screen.findByRole("group", { name: "Filter reviews" });
    expect(within(group).getByRole("button", { name: /^All/ })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: /^Revisions/ })).toBeInTheDocument();
    expect(within(group).queryByRole("button", { name: /^Conflicts/ })).toBeNull();
    expect(within(group).queryByRole("button", { name: /^Pages/ })).toBeNull();
    expect(within(group).queryByRole("button", { name: /^Merges/ })).toBeNull();
    expect(within(group).queryByRole("button", { name: /^Candidates/ })).toBeNull();
    expect(within(group).queryByRole("button", { name: /^Topics/ })).toBeNull();
  });

  it("filters to a single category, hides other sections, and Show all returns to all after the last item resolves", async () => {
    localStorage.setItem(
      "wenlan.review.hidden.v1",
      JSON.stringify([{ key: "topic:Ghost topic", label: "Ghost topic", kind: "topic", at: Date.now() }]),
    );
    vi.mocked(distillReview).mockResolvedValue(emptyDistill);
    vi.mocked(listRefinements)
      .mockResolvedValueOnce({
        proposals: [
          {
            id: "prop_conflict",
            action: "relation_conflict",
            source_ids: ["ent_a", "ent_b"],
            payload: {
              action: "relation_conflict",
              existing_id: "ent_a",
              new_id: "ent_b",
              from: "Alice",
              to: "Bob",
              old_type: "manages",
              new_type: "reports_to",
            },
            confidence: 0.7,
            created_at: "2026-07-09T00:00:00Z",
          },
        ],
      })
      .mockResolvedValue({ proposals: [] });
    const { user } = renderPanel();

    // Under "all": recent revisions and the hidden footer both show.
    expect(await screen.findByRole("heading", { name: "Recent revisions" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /1 hidden/ })).toBeInTheDocument();

    const group = screen.getByRole("group", { name: "Filter reviews" });
    await user.click(within(group).getByRole("button", { name: /^Conflicts/ }));

    expect(screen.getByRole("heading", { name: "Contradictions & conflicts" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Memory revisions" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Recent revisions" })).toBeNull();
    expect(screen.queryByRole("button", { name: /1 hidden/ })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Review Alice → Bob" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(acceptRefinement).toHaveBeenCalledWith("prop_conflict");
    });
    expect(await screen.findByText("Nothing in this category right now.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show all" }));

    expect(
      within(screen.getByRole("group", { name: "Filter reviews" })).getByRole("button", {
        name: /^All/,
      }),
    ).toHaveAttribute("aria-pressed", "true");
  });
});

// REVIEW_EXAMPLES_ENABLED is false under Vitest (MODE === "test"), so the
// panel never renders these — covered here as plain unit tests of the
// exported pure functions/fixtures instead.
describe("reviewExamples", () => {
  it("flags only example: ids", () => {
    for (const item of EXAMPLE_REVIEW_ITEMS) {
      expect(isExampleReviewItem(item)).toBe(true);
    }
    expect(
      isExampleReviewItem({
        kind: "revision",
        id: "mem_target",
        targetSourceId: "mem_target",
        revisionSourceId: "mem_target_rev",
        content: "content",
        agent: null,
        timestampMs: null,
      }),
    ).toBe(false);
  });

  it("ships exactly one revision and one detect_contradiction refinement sample", () => {
    expect(EXAMPLE_REVIEW_ITEMS).toHaveLength(2);
    const [revision, refinement] = EXAMPLE_REVIEW_ITEMS;
    expect(revision.kind).toBe("revision");
    expect(refinement.kind).toBe("refinement");
    expect(refinement.kind === "refinement" && refinement.action).toBe(
      "detect_contradiction",
    );
    for (const item of EXAMPLE_REVIEW_ITEMS) {
      expect(item.id.startsWith("example:")).toBe(true);
    }
  });

  it("labels each sample with its human title", () => {
    const [revision, refinement] = EXAMPLE_REVIEW_ITEMS;
    expect(exampleReviewLabel(revision)).toBe("Coffee routine");
    expect(exampleReviewLabel(refinement)).toBe("Standup schedule (updated)");
  });

  it("seeds every id the two dialogs read, with no stale gcTime/staleTime", () => {
    const client = new QueryClient();
    seedReviewExampleCaches(client);

    const [revision, refinement] = EXAMPLE_REVIEW_ITEMS;
    expect(revision.kind).toBe("revision");
    const coffeeId = revision.kind === "revision" ? revision.targetSourceId : "";
    expect(refinement.kind).toBe("refinement");
    const [standupNewId, standupOldId] =
      refinement.kind === "refinement" ? refinement.sourceIds : ["", ""];

    for (const sourceId of [coffeeId, standupNewId, standupOldId]) {
      const memory = client.getQueryData<MemoryItem>(["memory-detail", sourceId]);
      expect(memory?.source_id).toBe(sourceId);
      expect(memory?.content.length).toBeGreaterThan(0);

      const summary = client.getQueryData<{ name: string; text: string }>([
        "review-summary",
        "memory",
        sourceId,
      ]);
      expect(summary?.name.length).toBeGreaterThan(0);

      const chain = client.getQueryData<{ chain_depth: number; entries: unknown[] }>([
        "memory-revisions",
        sourceId,
      ]);
      expect(chain).toEqual({ current_source_id: sourceId, chain_depth: 0, entries: [] });
    }
  });
});
