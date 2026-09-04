// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { i18n } from "../../i18n";
import HomePage from "./HomePage";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

vi.mock("../../lib/tauri", async () => {
  const actual = await vi.importActual<typeof import("../../lib/tauri")>("../../lib/tauri");
  return {
    ...actual,
    listRecentRetrievals: vi.fn(),
    listRecentPages: vi.fn(),
    listRecentConcepts: vi.fn(),
    listRecentMemories: vi.fn(),
    listUnconfirmedMemories: vi.fn(),
    listPages: vi.fn(),
    listConcepts: vi.fn(),
    listRecentChanges: vi.fn(),
    listEntities: vi.fn(),
    getMemoryStats: vi.fn(),
    getProfile: vi.fn(),
    getPendingContradictions: vi.fn(),
    dismissContradiction: vi.fn(),
    confirmMemory: vi.fn(),
    deleteMemory: vi.fn(),
    listPendingRevisions: vi.fn(),
    acceptPendingRevision: vi.fn(),
    dismissPendingRevision: vi.fn(),
    listRefinements: vi.fn(),
    acceptRefinement: vi.fn(),
    rejectRefinement: vi.fn(),
    getMemoryDetail: vi.fn(),
    getEntityDetail: vi.fn(),
    getPage: vi.fn(),
    getPageSources: vi.fn(),
    getMemoryRevisions: vi.fn(),
    listOnboardingMilestones: vi.fn(),
    acknowledgeOnboardingMilestone: vi.fn(),
    getApiKey: vi.fn(),
    getExternalLlm: vi.fn(),
    getOnDeviceModel: vi.fn(),
  };
});

import * as tauri from "../../lib/tauri";

function renderHome(
  props: {
    onOpenDistillReview?: () => void;
    onSelectPage?: (pageId: string) => void;
    onCreatePage?: (space: string | null) => void;
    onOpenIntelligenceSettings?: () => void;
  } = {},
) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <HomePage
        onNavigateMemory={() => {}}
        onNavigateStream={() => {}}
        onNavigateLog={() => {}}
        onNavigateGraph={() => {}}
        onOpenDistillReview={props.onOpenDistillReview}
        onSelectPage={props.onSelectPage}
        onCreatePage={props.onCreatePage ?? (() => {})}
        onOpenIntelligenceSettings={props.onOpenIntelligenceSettings ?? (() => {})}
      />
    </QueryClientProvider>,
  );
}

/**
 * The daemon latches `intelligence-ready` on the first successful inference
 * and never clears it, so it says nothing about what is configured now.
 */
function intelligenceReadyMilestone(): tauri.MilestoneRecord {
  return {
    id: "intelligence-ready",
    first_triggered_at: Math.floor(Date.now() / 1000) - 3_600,
    acknowledged_at: null,
    payload: null,
  };
}

/** A saved Anthropic key — the daemon returns it masked. */
function withAnthropicKey() {
  vi.mocked(tauri.getApiKey).mockResolvedValue("sk-ant-...abcd");
}

const nowIso = new Date().toISOString();

function page(overrides: Partial<tauri.Page> & Pick<tauri.Page, "id" | "title">): tauri.Page {
  return {
    summary: null,
    content: "",
    entity_id: null,
    domain: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: nowIso,
    last_compiled: nowIso,
    last_modified: nowIso,
    ...overrides,
  };
}

beforeEach(async () => {
  await i18n.changeLanguage("en");
  localStorage.clear();
  vi.clearAllMocks();
  vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([]);
  vi.mocked(tauri.listRecentPages).mockResolvedValue([]);
  vi.mocked(tauri.listRecentConcepts).mockResolvedValue([]);
  vi.mocked(tauri.listRecentMemories).mockResolvedValue([]);
  vi.mocked(tauri.listUnconfirmedMemories).mockResolvedValue([]);
  vi.mocked(tauri.listPages).mockResolvedValue([]);
  vi.mocked(tauri.listConcepts).mockResolvedValue([]);
  vi.mocked(tauri.listRecentChanges).mockResolvedValue([]);
  vi.mocked(tauri.getPageSources).mockResolvedValue([]);
  vi.mocked(tauri.getMemoryRevisions).mockResolvedValue({
    current_source_id: "mem-target",
    chain_depth: 1,
    entries: [],
  } as any);
  vi.mocked(tauri.listEntities).mockResolvedValue([]);
  vi.mocked(tauri.listOnboardingMilestones).mockResolvedValue([]);
  vi.mocked(tauri.acknowledgeOnboardingMilestone).mockResolvedValue(undefined);
  // Default: no provider of any kind, and all three queries answer.
  vi.mocked(tauri.getApiKey).mockResolvedValue(null);
  vi.mocked(tauri.getExternalLlm).mockResolvedValue([null, null]);
  vi.mocked(tauri.getOnDeviceModel).mockResolvedValue({
    loaded: null,
    selected: null,
    models: [],
  });
  vi.mocked(tauri.getMemoryStats).mockResolvedValue({
    total: 0,
    new_today: 0,
    confirmed: 0,
    domains: [],
  } as any);
  vi.mocked(tauri.getProfile).mockResolvedValue(null);
  vi.mocked(tauri.confirmMemory).mockResolvedValue(undefined);
  vi.mocked(tauri.deleteMemory).mockResolvedValue(undefined);
  vi.mocked(tauri.dismissContradiction).mockResolvedValue({ source_id: "mem-new", wrote: true });
  vi.mocked(tauri.listPendingRevisions).mockResolvedValue([]);
  vi.mocked(tauri.acceptPendingRevision).mockResolvedValue({
    target_source_id: "mem-target",
    revision_source_id: "mem-revision",
    wrote: true,
  });
  vi.mocked(tauri.dismissPendingRevision).mockResolvedValue({
    target_source_id: "mem-target",
    wrote: true,
  });
  vi.mocked(tauri.listRefinements).mockResolvedValue({ proposals: [] });
  vi.mocked(tauri.acceptRefinement).mockResolvedValue({
    id: "ref-merge",
    action_applied: "entity_merge",
  });
  vi.mocked(tauri.rejectRefinement).mockResolvedValue({ id: "ref-merge" });
  vi.mocked(tauri.getMemoryDetail).mockResolvedValue({
    source_id: "mem-target",
    title: "Target memory",
    content: "The durable original wording from the daemon.",
    summary: null,
    memory_type: null,
    domain: null,
    source_agent: null,
    confidence: null,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: 1_782_365_000,
    chunk_count: 1,
  } as any);
  vi.mocked(tauri.getEntityDetail).mockResolvedValue({
    entity: {
      id: "ent-a",
      name: "Wenlan",
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
  } as any);
  vi.mocked(tauri.getPendingContradictions).mockResolvedValue([
    {
      id: "contra-1",
      existing_content: "First claim",
      new_content: "Second claim",
      new_source_id: "mem-new",
      existing_source_id: "mem-existing",
    } as any,
  ]);
});

describe("HomePage redesign", () => {
  it.each([
    ["en", "Home overview", "Index"],
    ["zh-Hans", "首页概览", "索引"],
    ["zh-Hant", "首頁概覽", "索引"],
  ] as const)(
    "labels the Home overview without exposing Index in %s",
    async (locale, overviewLabel, indexLabel) => {
      // Given a populated Home page in the selected locale
      await i18n.changeLanguage(locale);
      vi.mocked(tauri.listPages).mockResolvedValue([
        page({ id: `page-${locale}`, title: "Localized page" }),
      ]);

      // When Home renders its overview metrics
      renderHome();
      const overview = await screen.findByTestId("wiki-index-summary");

      // Then the accessible name describes Home and never exposes Index
      expect(overview).toHaveAttribute("aria-label", overviewLabel);
      expect(overview).not.toHaveAttribute("aria-label", indexLabel);
      expect(screen.queryByText(indexLabel)).not.toBeInTheDocument();
    },
  );

  it("uses wiki pages as the primary home surface when pages exist without activity", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({
        id: "page-architecture",
        title: "Wenlan app architecture",
        domain: "Projects",
        summary: "How the desktop app, daemon, and page compiler fit together.",
        source_memory_ids: ["m1", "m2", "m3", "m4"],
        version: 3,
      }),
      page({
        id: "page-policy",
        title: "Codex workflow policy",
        domain: "Decisions",
        source_memory_ids: ["m5", "m6"],
        version: 2,
      }),
    ]);

    renderHome();

    const todayHeading = await screen.findByRole("heading", { name: "Today in Wenlan", level: 1 });
    expect(todayHeading).toHaveStyle({
      fontFamily: "var(--mem-font-heading)",
      fontSize: "var(--mem-destination-title-size)",
      fontWeight: "500",
      letterSpacing: "-0.03em",
      lineHeight: "1.12",
    });
    expect(todayHeading.parentElement).toHaveStyle({ marginBottom: "16px" });
    expect(tauri.listPages).toHaveBeenCalledWith("active", undefined, 500, 0);
    expect(screen.getByTestId("wiki-home")).toHaveStyle({ display: "grid" });
    expect(screen.getByTestId("wiki-index-summary")).toHaveAttribute("aria-label", "Home overview");
    expect(within(screen.getByTestId("wiki-context-rail")).queryByText("Index")).toBeNull();
    expect(screen.getByTestId("wiki-context-rail")).not.toHaveTextContent("Recently active");
    expect(screen.getByTestId("wiki-context-pages")).toHaveTextContent("2");
    expect(screen.getByTestId("wiki-context-updated-today")).toHaveTextContent(/^2 updated today$/);
    // The review count lives only in the needs-review rail pill now.
    expect(screen.queryByTestId("wiki-context-needs-review")).toBeNull();
    expect(screen.queryByTestId("wiki-space-filter-row")).toBeNull();
    expect(screen.queryByTestId("wiki-recent-spaces")).toBeNull();
    expect(screen.queryByText("Wiki pages")).toBeNull();
    expect(screen.queryByText("Compiled pages, links, and sources your agents can traverse.")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Recent Space" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Recently refined" })).toBeNull();
    expect(screen.getByText("Wenlan app architecture")).toBeInTheDocument();
    expect(screen.getByTestId("wiki-page-list").querySelector("svg path")).toHaveAttribute(
      "d",
      "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
    );
    expect(screen.getByText("4 sources")).toBeInTheDocument();
    expect(screen.getAllByText("updated today").length).toBeGreaterThan(0);
    expect(screen.queryByText("Key facts")).toBeNull();
    expect(screen.queryByText("Related pages")).toBeNull();
    expect(screen.queryByText("Related sources")).toBeNull();
    expect(screen.queryByText("source-backed")).toBeNull();
  });

  it("counts only knowledge pages in the pages metric, never entity shadow pages", async () => {
    // Given two knowledge pages (one updated today) and one entity shadow page
    // updated today — the daemon's browse list returns all three by contract
    const lastWeek = new Date(Date.now() - 7 * 86_400_000).toISOString();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-architecture", title: "Wenlan app architecture", creation_kind: "distilled" }),
      page({ id: "page-policy", title: "Codex workflow policy", last_modified: lastWeek }),
      page({ id: "shadow-lucian", title: "Lucian", creation_kind: "entity", entity_id: "ent-1" }),
    ]);

    // When Home renders its overview metrics
    renderHome();
    await screen.findByTestId("wiki-context-rail");

    // Then the pages total and its updated-today chip exclude the shadow page
    expect(screen.getByTestId("wiki-context-pages")).toHaveTextContent("2");
    expect(screen.getByTestId("wiki-context-pages")).not.toHaveTextContent("3");
    expect(screen.getByTestId("wiki-context-updated-today")).toHaveTextContent(/^1 updated today$/);
  });

  it("keeps today, index, articles, and review items in the expected reading order", async () => {
    const rectSpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      bottom: 0,
      height: 720,
      left: 0,
      right: 1000,
      top: 0,
      width: 1000,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-architecture", title: "Wenlan app architecture", source_memory_ids: ["m1", "m2", "m3"] }),
      page({ id: "page-policy", title: "Codex workflow policy", source_memory_ids: ["m4"] }),
    ]);

    try {
      renderHome();

      await screen.findByTestId("wiki-home");
      const todayHeading = screen.getByTestId("wiki-today-heading");
      const contextRail = screen.getByTestId("wiki-context-rail");
      const pageList = screen.getByTestId("wiki-page-list");
      const pageUpdates = screen.getByTestId("wiki-page-updates");

      expect(todayHeading.compareDocumentPosition(contextRail) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(contextRail.compareDocumentPosition(pageList) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(pageList.compareDocumentPosition(pageUpdates) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

      expect(todayHeading).not.toHaveTextContent("2 pages");
      expect(screen.getByTestId("wiki-context-pages")).toHaveTextContent("2");

      expect(screen.getByTestId("wiki-index-summary")).toHaveAttribute("aria-label", "Home overview");
      // Review items stay in the page-updates section, not the rail (the
      // needs-review metric label is allowed here).
      expect(within(contextRail).queryByText("Wenlan app architecture")).toBeNull();
      expect(within(contextRail).queryByText("Index")).toBeNull();

      expect(screen.getByTestId("wiki-page-list")).toHaveStyle({ borderTopStyle: "none" });
    } finally {
      rectSpy.mockRestore();
    }
  });

  it("moves the dateline into the Today heading action slot", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", last_modified: nowIso }),
    ]);

    renderHome();

    const todayHeading = await screen.findByTestId("wiki-today-heading");
    expect(within(todayHeading).getByTestId("wiki-context-latest")).toHaveTextContent("updated today");
    // The dateline lives in the heading now, not among the context rail cells.
    expect(within(screen.getByTestId("wiki-context-rail")).queryByTestId("wiki-context-latest")).toBeNull();
  });

  it("opens wiki page rows from the home index", async () => {
    const onSelectPage = vi.fn();
    const user = userEvent.setup();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({
        id: "page-architecture",
        title: "Wenlan app architecture",
        domain: "Projects",
        source_memory_ids: ["m1", "m2", "m3"],
        version: 2,
      }),
    ]);

    renderHome({ onSelectPage });

    await user.click(await screen.findByRole("button", { name: /open Wenlan app architecture/i }));

    expect(onSelectPage).toHaveBeenCalledWith("page-architecture");
  });

  it("does not duplicate space navigation on the home index", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({
        id: "page-architecture",
        title: "Wenlan app architecture",
        domain: "Projects",
      }),
      page({
        id: "page-policy",
        title: "Codex workflow policy",
        domain: "Decisions",
      }),
    ]);

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.queryByTestId("wiki-space-filter-row")).toBeNull();
    expect(screen.queryByRole("button", { name: /open Projects space/i })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Spaces" })).toBeNull();
    expect(screen.queryByLabelText("Recent spaces")).toBeNull();
  });

  it("does not expose recent spaces from the home context rail", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({
        id: "page-architecture",
        title: "Wenlan app architecture",
        domain: "Projects",
        last_modified: "2026-06-30T12:00:00Z",
      }),
      page({
        id: "page-policy",
        title: "Codex workflow policy",
        domain: "Decisions",
        last_modified: "2026-06-29T12:00:00Z",
      }),
    ]);

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.queryByTestId("wiki-recent-spaces")).toBeNull();
    expect(screen.getByTestId("wiki-context-rail")).not.toHaveTextContent("Recently active");
  });

  it("counts every queue item in needs review and only today's pages in updated today", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-a", title: "A" }),
      page({ id: "page-b", title: "B" }),
      page({ id: "page-c", title: "C" }),
      page({ id: "page-d", title: "D", last_modified: "2026-06-01T12:00:00Z" }),
      page({ id: "page-e", title: "E", last_modified: "2026-06-01T12:00:00Z" }),
    ]);
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-a",
        revision_source_id: "mem-a-rev",
        revision_content: "First proposed wording",
        source_agent: "claude-code",
        last_modified: 1_782_365_076,
        target_kind: "memory" as const,
      },
      {
        target_source_id: "mem-b",
        revision_source_id: "mem-b-rev",
        revision_content: "Second proposed wording",
        source_agent: "claude-code",
        last_modified: 1_782_365_077,
        target_kind: "memory" as const,
      },
      {
        target_source_id: "mem-c",
        revision_source_id: "mem-c-rev",
        revision_content: "Third proposed wording",
        source_agent: "claude-code",
        last_modified: 1_782_365_078,
        target_kind: "memory" as const,
      },
    ]);
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-merge",
          action: "entity_merge",
          source_ids: ["ent-a", "ent-b"],
          payload: { action: "entity_merge", existing_id: "ent-a", new_id: "ent-b", similarity: 0.86 },
          confidence: 0.86,
          created_at: nowIso,
        },
      ],
    });

    renderHome({ onOpenDistillReview: vi.fn() });

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    // The rail's list slices to 3 items; the heading pill still counts all 4.
    const rail = await screen.findByTestId("wiki-page-updates");
    await within(rail).findByText("4");
    await within(rail).findByText("Review all →");
    expect(screen.queryByTestId("wiki-context-needs-review")).toBeNull();
    expect(screen.getByTestId("wiki-context-updated-today")).toHaveTextContent(/^3 updated today$/);
    // All three revision rows title themselves with the fetched memory name.
    await waitFor(() => {
      expect(screen.getAllByText("Target memory")).toHaveLength(3);
    });
    expect(screen.queryByText("Entity merge")).toBeNull();
  });

  it("does not navigate to the synthetic Unsorted page bucket", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-unsorted", title: "Unassigned page" }),
    ]);

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.getByText("Unassigned page")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /open Unsorted space/i })).toBeNull();
    expect(screen.queryByText("Unsorted")).toBeNull();
  });

  it("does not render traversal paths on the home surface", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-wenlan", title: "Wenlan app architecture", domain: "Projects" }),
    ]);
    vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([
      {
        timestamp_ms: Date.now(),
        agent_name: "claude-code",
        query: "wiki home",
        page_titles: ["Wenlan app architecture", "Codex workflow policy"],
        page_ids: ["page-wenlan", "page-policy"],
        memory_snippets: [],
      },
    ]);

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.queryByTestId("wiki-traversal-paths")).toBeNull();
    expect(screen.queryByText("Traversal paths")).toBeNull();
  });

  it("keeps the needs-review rail secondary to the index", async () => {
    const onOpenDistillReview = vi.fn();
    const user = userEvent.setup();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-target",
        revision_source_id: "mem-revision",
        revision_content: "The durable updated wording from the daemon.",
        source_agent: "claude-code",
        last_modified: 1_782_365_076,
        target_kind: "memory" as const,
      },
    ]);

    renderHome({ onOpenDistillReview });

    const contextRail = await screen.findByTestId("wiki-context-rail");
    expect(screen.getByTestId("wiki-index-summary")).toHaveAttribute("aria-label", "Home overview");
    expect(within(contextRail).queryByText("Index")).toBeNull();
    expect(contextRail).not.toHaveTextContent("Recently active");
    expect(within(contextRail).queryByText(/durable updated wording/)).toBeNull();

    const pageUpdates = screen.getByTestId("wiki-page-updates");
    expect(pageUpdates).toHaveTextContent("Needs review");
    await within(pageUpdates).findByText(/The durable updated wording/);
    expect(pageUpdates).toHaveTextContent("Memory revision");
    // The rail meta is now "kind · age" per the mockup; the proposing agent
    // stays in the review dialog, not the rail row.
    expect(pageUpdates).not.toHaveTextContent("proposed by");
    expect(pageUpdates).not.toHaveTextContent("Current page");
    expect(pageUpdates).toHaveTextContent("Review all →");

    await user.click(within(pageUpdates).getByRole("button", { name: /review all/i }));

    expect(onOpenDistillReview).toHaveBeenCalledTimes(1);
  });

  it("opens the page review route from the home maintenance area", async () => {
    const onOpenDistillReview = vi.fn();
    const user = userEvent.setup();

    renderHome({ onOpenDistillReview });

    await user.click(await screen.findByRole("button", { name: /review page changes/i }));

    expect(onOpenDistillReview).toHaveBeenCalledTimes(1);
  });

  it("never renders a greeting screen, with or without pages", async () => {
    vi.mocked(tauri.getProfile).mockResolvedValue({
      id: "p1",
      name: "Lucian",
      display_name: null,
      email: null,
      bio: null,
      avatar_path: null,
      created_at: 0,
      updated_at: 0,
    } as any);

    // Empty library: one home, no salutation.
    const emptyHome = renderHome();
    await screen.findByTestId("wiki-home");
    expect(screen.queryByTestId("greeting")).toBeNull();
    expect(screen.queryByText(/Good (morning|afternoon|evening)/)).toBeNull();
    expect(screen.queryByText(/your library holds/)).toBeNull();
    emptyHome.unmount();

    // Populated library: the same one home.
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-architecture", title: "Wenlan app architecture" }),
    ]);
    renderHome();
    await screen.findByTestId("wiki-page-list");
    expect(screen.queryByTestId("greeting")).toBeNull();
    expect(screen.queryByText(/Good (morning|afternoon|evening)/)).toBeNull();
  });

  it("does NOT render ProfileNarrativeCompact on home", async () => {
    const now = Date.now();
    vi.mocked(tauri.listRecentConcepts).mockResolvedValue([
      { kind: "concept", id: "c1", title: "A", snippet: "s", timestamp_ms: now, badge: { kind: "new" } },
    ] as any);
    renderHome();
    // Settle React Query before asserting absence.
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByText(/^Updated/i)).toBeNull();
  });

  it("renders the retrievals list with known agent names", async () => {
    vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([
      {
        timestamp_ms: Date.now(),
        agent_name: "claude-code",
        query: "positioning",
        page_titles: ["Origin positioning", "Daemon architecture"],
        page_ids: ["concept_pos", "concept_arch"],
        memory_snippets: [],
      },
    ]);
    renderHome();
    expect(await screen.findByTestId("retrievals")).toBeInTheDocument();
    expect(screen.getByText(/Where AI looked/i)).toBeInTheDocument();
    expect(screen.getByText(/Claude Code/)).toBeInTheDocument();
    expect(screen.getByText(/Origin positioning/)).toBeInTheDocument();
  });

  it("filters unknown agents out of the retrievals list", async () => {
    vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([
      {
        timestamp_ms: Date.now(),
        agent_name: "unknown",
        query: "anything",
        page_titles: ["Should not appear"],
        page_ids: [],
        memory_snippets: [],
      },
    ]);
    renderHome();
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByTestId("retrievals")).toBeNull();
    expect(screen.queryByText(/Should not appear/)).toBeNull();
  });

  it("does NOT render contradiction resolver on home", async () => {
    renderHome();
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByTestId("contradiction-resolver")).toBeNull();
  });

  it("does not use recent activity as review items", async () => {
    const now = Date.now();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);
    vi.mocked(tauri.listRecentPages).mockResolvedValue([
      { kind: "concept", id: "c1", title: "Flagged concept", snippet: "s", timestamp_ms: now, badge: { kind: "needs_review" } },
      { kind: "concept", id: "c2", title: "Fresh concept", snippet: "s", timestamp_ms: now - 500, badge: { kind: "new" } },
    ] as any);
    vi.mocked(tauri.listRecentMemories).mockResolvedValue([
      { kind: "memory", id: "m1", title: "Refined memory", snippet: "s", timestamp_ms: now - 1000, badge: { kind: "refined" } },
    ] as any);
    renderHome();
    const strip = await screen.findByTestId("worth-a-glance");
    await within(strip).findByText(/All caught up/);
    expect(strip.textContent).not.toContain("Flagged concept");
    expect(strip.textContent).not.toContain("Fresh concept");
    expect(strip.textContent).not.toContain("Refined memory");
  });

  it("opens the review dialog from a rail revision and approves it", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listPendingRevisions)
      .mockResolvedValueOnce([
        {
          target_source_id: "mem-target",
          revision_source_id: "mem-revision",
          revision_content: "The durable updated wording from the daemon.",
          source_agent: "claude-code",
          last_modified: 1_782_365_076,
          target_kind: "memory" as const,
        },
      ])
      .mockResolvedValue([]);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);

    renderHome();

    // The rail row titles itself with the target memory's real name.
    await user.click(
      await screen.findByRole("button", { name: /Review Target memory/ }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByRole("heading", { name: "Target memory" })).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(tauri.acceptPendingRevision).toHaveBeenCalledWith("mem-target");
    });
  });

  it("keeps new memories out of the needs-review rail — inflow, not decisions", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.listUnconfirmedMemories).mockResolvedValue([
      {
        kind: "memory",
        id: "mem-capture",
        title: "User prefers pnpm over npm",
        snippet: "Stated while setting up the monorepo.",
        timestamp_ms: 1_782_365_080_000,
        badge: { kind: "needs_review" },
      },
    ]);

    renderHome({ onOpenDistillReview: vi.fn() });

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    // Captures are already-live inflow, not a chore: no "to confirm" pill…
    expect(
      screen.queryByTestId("wiki-context-new-memories"),
    ).not.toBeInTheDocument();
    // …and the decisions rail stays caught up and never lists the capture.
    const rail = screen.getByTestId("wiki-page-updates");
    await within(rail).findByText(/All caught up/);
    expect(
      within(rail).queryByText("User prefers pnpm over npm"),
    ).not.toBeInTheDocument();
  });

  it("opens the dialog on the decisions the rail counted, not the merged queue", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-a",
        revision_source_id: "mem-a-rev",
        revision_content: "Proposed wording",
        source_agent: "claude-code",
        last_modified: 1_782_365_076,
        target_kind: "memory" as const,
      },
    ]);
    vi.mocked(tauri.listUnconfirmedMemories).mockResolvedValue([
      {
        kind: "memory",
        id: "mem-capture-1",
        title: "First new memory",
        snippet: "Captured a moment ago.",
        timestamp_ms: 1_782_365_080_000,
        badge: { kind: "needs_review" },
      },
      {
        kind: "memory",
        id: "mem-capture-2",
        title: "Second new memory",
        snippet: "Captured a moment ago.",
        timestamp_ms: 1_782_365_081_000,
        badge: { kind: "needs_review" },
      },
    ]);

    renderHome({ onOpenDistillReview: vi.fn() });

    const rail = await screen.findByTestId("wiki-page-updates");
    await within(rail).findByText("1");
    await user.click(await screen.findByRole("button", { name: /Target memory/ }));

    // The header counter walks the rail's one decision; the two captures in
    // the merged queue never make it "1 of 3" beside a badge that says 1.
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("1 of 1")).toBeInTheDocument();
    expect(within(dialog).queryByText("First new memory")).not.toBeInTheDocument();
  });

  it("opens the contradiction dialog with before/after panes and resolves it", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-contra",
          action: "detect_contradiction",
          source_ids: ["mem-new", "mem-old"],
          payload: { action: "detect_contradiction" },
          confidence: 0.78,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.getMemoryDetail).mockImplementation(
      async (sourceId: string) =>
        ({
          source_id: sourceId,
          title: sourceId === "mem-new" ? "New memory" : "Old memory",
          content:
            sourceId === "mem-new"
              ? "The project stores data in redb."
              : "The project stores data in SQLite.",
          summary: null,
          memory_type: null,
          domain: null,
          source_agent: null,
          confidence: null,
          confirmed: true,
          pinned: false,
          supersedes: null,
          last_modified: 1_782_365_000,
          chunk_count: 1,
        }) as any,
    );

    renderHome();

    // Both memory names load into the rail title.
    await user.click(
      await screen.findByRole("button", { name: /contradicts/ }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Existing memory")).toBeInTheDocument();
    expect(within(dialog).getByText("New memory — newer")).toBeInTheDocument();
    // Existing pane keeps the old wording; the new pane shows the replacement.
    await within(dialog).findByText(/SQLite/);
    await within(dialog).findByText(/redb/);
    expect(
      within(dialog).getByRole("button", { name: "Keep both" }),
    ).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Resolve" }));
    await waitFor(() =>
      expect(tauri.acceptRefinement).toHaveBeenCalledWith("ref-contra"),
    );
    expect(tauri.rejectRefinement).not.toHaveBeenCalled();
  });

  it("opens the page-merge strip-off dossier and merges it", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-page-merge",
          action: "page_merge",
          source_ids: ["page-keep", "page-absorb"],
          payload: {
            action: "page_merge",
            left_page_id: "page-keep",
            right_page_id: "page-absorb",
            source_overlap: 5,
            source_overlap_ratio: 1.0,
          },
          confidence: 1.0,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.getPage).mockImplementation(
      async (id: string) =>
        page({
          id,
          title: id === "page-keep" ? "Surviving page" : "Absorbed page",
        }) as any,
    );
    // The retiring page's 5 sources are a strict subset of the kept page's 6,
    // so the dossier ledger yields no unique retiring sources → safe verdict.
    vi.mocked(tauri.getPageSources).mockImplementation(async (id: string) => {
      const ids =
        id === "page-keep"
          ? ["m1", "m2", "m3", "m4", "m5", "m6"]
          : ["m1", "m2", "m3", "m4", "m5"];
      return ids.map((m) => ({
        source: { page_id: id, memory_source_id: m, linked_at: 0 },
        memory: null,
      })) as any;
    });

    renderHome();

    // The rail title carries both page names and the merge direction.
    await user.click(
      await screen.findByRole("button", { name: /“Surviving page” absorbs “Absorbed page”/ }),
    );

    const dialog = await screen.findByRole("dialog");
    // New "strip-off" dossier: kept/retiring framing + the "nothing lost" verdict.
    await within(dialog).findByText(/Nothing unique is lost/i);
    expect(within(dialog).getByText("Kept page")).toBeInTheDocument();
    expect(within(dialog).getByText("Retiring page")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /Merge pages/i }));
    await waitFor(() =>
      expect(tauri.acceptRefinement).toHaveBeenCalledWith("ref-page-merge"),
    );
  });

  it("orders the review queue revisions first, then conflicts, then page items", async () => {
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-target",
        revision_source_id: "mem-revision",
        revision_content: "Revised memory wording",
        source_agent: "claude-code",
        last_modified: 1_782_365_000,
      },
    ] as any);
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-merge",
          action: "entity_merge",
          source_ids: ["ent-a", "ent-b"],
          payload: null,
          confidence: 0.86,
          created_at: nowIso,
        },
        {
          id: "ref-contra",
          action: "detect_contradiction",
          source_ids: ["mem-new", "mem-old"],
          payload: null,
          confidence: 0.78,
          created_at: nowIso,
        },
        {
          id: "ref-page-merge",
          action: "page_merge",
          source_ids: ["page-keep", "page-absorb"],
          payload: null,
          confidence: 1.0,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.getMemoryDetail).mockImplementation(
      async (sourceId: string) =>
        ({
          source_id: sourceId,
          title:
            sourceId === "mem-new"
              ? "New claim"
              : sourceId === "mem-old"
                ? "Old claim"
                : "Target memory",
          content: "content",
          summary: null,
          memory_type: null,
          domain: null,
          source_agent: null,
          confidence: null,
          confirmed: true,
          pinned: false,
          supersedes: null,
          last_modified: 1_782_365_000,
          chunk_count: 1,
        }) as any,
    );

    renderHome();

    const strip = await screen.findByTestId("worth-a-glance");
    await within(strip).findAllByText(/Page merge/);
    // Rail shows the top 3 of the ranked queue: revisions > conflicts > pages.
    // Titles resolve asynchronously from the fetched names, so wait for them.
    await waitFor(() => {
      const labels = within(strip)
        .getAllByRole("button")
        .map((button) => button.getAttribute("aria-label"));
      expect(labels).toEqual([
        "Review Target memory",
        "Review “New claim” contradicts “Old claim”",
        "Review Page merge",
      ]);
    });
  });

  it("shows em-dash totals for memories and entities while their queries are pending", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    // Leave getMemoryStats/listEntities pending — the masthead degrades to
    // an em dash rather than a spinner.
    vi.mocked(tauri.getMemoryStats).mockImplementation(() => new Promise(() => {}));
    vi.mocked(tauri.listEntities).mockImplementation(() => new Promise(() => {}));

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.getByTestId("wiki-context-memories")).toHaveTextContent("—");
    expect(screen.getByTestId("wiki-context-entities")).toHaveTextContent("—");
  });

  it("shows memories and entities totals with their today-deltas once loaded", async () => {
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.getMemoryStats).mockResolvedValue({
      total: 1204,
      new_today: 18,
      confirmed: 1190,
      domains: [],
    } as any);
    vi.mocked(tauri.listEntities).mockResolvedValue(
      Array.from({ length: 87 }, (_, i) => ({
        id: `ent-${i}`,
        name: `Entity ${i}`,
        entity_type: "tool",
        domain: null,
        source_agent: null,
        confidence: null,
        confirmed: true,
        created_at: 0,
        updated_at: 0,
      })) as any,
    );

    renderHome();

    await screen.findByRole("heading", { name: "Today in Wenlan" });

    expect(screen.getByTestId("wiki-context-memories")).toHaveTextContent("1204");
    expect(await screen.findByTestId("wiki-context-memories-delta")).toHaveTextContent("+18 today");
    expect(await screen.findByTestId("wiki-context-entities")).toHaveTextContent("87");
  });

  it("shows a before/after relation pair for relation conflicts and approves", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-relation",
          action: "relation_conflict",
          source_ids: ["rel-new", "rel-old"],
          payload: {
            action: "relation_conflict",
            existing_id: "rel-old",
            new_id: "rel-new",
            from: "Lucian",
            to: "Zed",
            old_type: "EVALUATES",
            new_type: "USES_DAILY",
          },
          confidence: 0.82,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);

    renderHome();

    // Relation conflicts title themselves from the payload's endpoints.
    await user.click(
      await screen.findByRole("button", { name: "Review Lucian → Zed" }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Current relation")).toBeInTheDocument();
    expect(within(dialog).getByText("Proposed relation")).toBeInTheDocument();
    expect(within(dialog).getByText(/EVALUATES/)).toBeInTheDocument();
    expect(within(dialog).getByText(/USES_DAILY/)).toBeInTheDocument();
    // The relation ids must never be fetched as memories.
    expect(tauri.getMemoryDetail).not.toHaveBeenCalledWith("rel-new");

    await user.click(within(dialog).getByRole("button", { name: "Approve" }));
    await waitFor(() =>
      expect(tauri.acceptRefinement).toHaveBeenCalledWith("ref-relation"),
    );
  });

  it("offers only dismiss for proposals the daemon cannot accept", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-suggest",
          action: "suggest_entity",
          source_ids: ["mem-a"],
          payload: { action: "suggest_entity", name_hint: "Zed Editor" },
          confidence: 0.7,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);

    renderHome();

    await user.click(
      await screen.findByRole("button", { name: "Review Zed Editor" }),
    );

    const dialog = await screen.findByRole("dialog");
    // The name hint appears as both the dialog heading and the body chip.
    await within(dialog).findAllByText("Zed Editor");
    expect(
      within(dialog).queryByRole("button", { name: "Approve" }),
    ).not.toBeInTheDocument();
    // Enter must not fire the blocked accept verb either.
    await user.keyboard("{Enter}");
    expect(tauri.acceptRefinement).not.toHaveBeenCalled();

    await user.click(within(dialog).getByRole("button", { name: "Dismiss" }));
    await waitFor(() =>
      expect(tauri.rejectRefinement).toHaveBeenCalledWith("ref-suggest"),
    );
  });

  it("opens the page keep-or-archive dialog with archive/keep actions", async () => {
    const user = userEvent.setup();
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-archive",
          action: "page_keep_or_archive",
          source_ids: ["memory-evidence"],
          payload: {
            action: "page_keep_or_archive",
            page_id: "page-thin",
            source_count: 1,
          },
          confidence: 1.0,
          created_at: nowIso,
        },
      ],
    } as any);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page" }),
    ]);
    vi.mocked(tauri.getPage).mockImplementation(async (id: string) => (
      id === "page-thin"
        ? page({
          id: "page-thin",
          title: "Thin scratch page",
          summary: "One lonely source.",
        }) as any
        : null
    ));

    renderHome();

    await user.click(
      await screen.findByRole("button", { name: /Review Thin scratch page/ }),
    );

    expect(tauri.getPage).toHaveBeenCalledWith("page-thin");
    expect(tauri.getPage).not.toHaveBeenCalledWith("memory-evidence");

    const dialog = await screen.findByRole("dialog");
    // The page title appears as both the dialog heading and the body pane.
    expect(await within(dialog).findAllByText("Thin scratch page")).toHaveLength(2);
    expect(
      within(dialog).getByRole("button", { name: "Archive" }),
    ).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Keep page" }));
    await waitFor(() =>
      expect(tauri.rejectRefinement).toHaveBeenCalledWith("ref-archive"),
    );
    expect(tauri.acceptRefinement).not.toHaveBeenCalled();
  });

  it("surfaces refinement proposals in the needs-review rail", async () => {
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-merge",
          action: "entity_merge",
          source_ids: ["mem-a", "mem-b"],
          payload: {
            action: "entity_merge",
            existing_id: "ent-a",
            new_id: "ent-b",
            similarity: 0.86,
          },
          confidence: 0.86,
          created_at: "2026-06-26T00:00:00Z",
        },
      ],
    });
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);

    renderHome();

    const strip = await screen.findByTestId("worth-a-glance");
    // Rail meta is kind + age only — the bare confidence "%" moved to the
    // dialog, where it's labeled (see DistillReviewPanel "94% confidence").
    await within(strip).findByText(/Entity merge/);
  });

  it("does not render inline approval actions in the needs-review rail", async () => {
    vi.mocked(tauri.listRefinements).mockResolvedValue({
      proposals: [
        {
          id: "ref-merge",
          action: "entity_merge",
          source_ids: ["mem-a", "mem-b"],
          payload: { action: "entity_merge", existing_id: "ent-a", new_id: "ent-b", similarity: 0.86 },
          confidence: 0.86,
          created_at: nowIso,
        },
      ],
    });
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-target",
        revision_source_id: "mem-revision",
        revision_content: "The durable updated wording from the daemon.",
        source_agent: "claude-code",
        last_modified: 1_782_365_076,
        target_kind: "memory" as const,
      },
    ]);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);

    renderHome();

    const rail = await screen.findByTestId("worth-a-glance");
    await within(rail).findByText(/The durable updated wording/);
    // Approve/Dismiss live in the review dialog, never inline in the rail.
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Accept" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Dismiss" })).not.toBeInTheDocument();
    expect(tauri.acceptPendingRevision).not.toHaveBeenCalled();
    expect(tauri.acceptRefinement).not.toHaveBeenCalled();
  });

  it("retrieval card with archived concept shows archived badge and does not navigate", async () => {
    const onSelectPage = vi.fn();
    // Event has page_ids: [] simulating an archived concept (no active match found at read time)
    vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([
      {
        timestamp_ms: Date.now(),
        agent_name: "claude-code",
        query: "origin arch",
        page_titles: ["Origin Architecture"],
        page_ids: [],
        memory_snippets: [],
      },
    ]);
    renderHome({ onSelectPage });
    // Wait for the retrievals section to render
    await screen.findByTestId("retrievals");
    // The archived badge should be visible
    expect(screen.getByTitle("This page has been archived")).toBeInTheDocument();
    // Clicking should not navigate because page_ids is empty
    const item = screen.getByTestId("retrieval-item");
    await userEvent.click(item);
    expect(onSelectPage).not.toHaveBeenCalled();
  });

  it("does not flash the empty state while the pages query is still loading", async () => {
    let resolvePages!: (pages: tauri.Page[]) => void;
    vi.mocked(tauri.listPages).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePages = resolve;
        }),
    );

    renderHome();

    // The query has not resolved yet — its data defaults to `[]`, which must
    // not be read as "no pages" and paint the empty state over a library that
    // in fact has pages.
    expect(screen.queryByTestId("wiki-page-empty")).toBeNull();
    expect(screen.queryByTestId("wiki-home")).toBeNull();

    resolvePages([
      page({ id: "page-architecture", title: "Wenlan app architecture" }),
    ]);

    expect(await screen.findByTestId("wiki-home")).toBeInTheDocument();
    expect(screen.queryByTestId("wiki-page-empty")).toBeNull();
    expect(screen.getByTestId("wiki-page-list")).toBeInTheDocument();
  });

  it("renders one wiki home with the not-ready empty state and both actions wired", async () => {
    const onCreatePage = vi.fn();
    const onOpenIntelligenceSettings = vi.fn();
    const user = userEvent.setup();
    // No provider configured, but the milestone is latched from an inference
    // that ran before the key was removed. The live queries win: page
    // synthesis cannot run today, so the promise must not appear.
    vi.mocked(tauri.listOnboardingMilestones).mockResolvedValue([
      intelligenceReadyMilestone(),
    ]);

    renderHome({ onCreatePage, onOpenIntelligenceSettings });

    // The one home surface renders — heading, rail, and the empty page slot.
    await screen.findByRole("heading", { name: "Today in Wenlan" });
    expect(screen.getByTestId("wiki-context-rail")).toBeInTheDocument();
    const empty = await screen.findByTestId("wiki-page-empty");
    expect(screen.queryByTestId("wiki-page-list")).toBeNull();

    // The copy names the precondition instead of promising a deadline.
    await waitFor(() =>
      expect(empty).toHaveTextContent(
        /Wenlan compiles pages from your memories once a local model or an API key is turned on/,
      ),
    );
    expect(empty).toHaveTextContent("No pages yet.");
    expect(empty).not.toHaveTextContent(/usually within a day/);

    // Both actions are live, not decoration.
    await user.click(within(empty).getByRole("button", { name: "Turn on a model" }));
    expect(onOpenIntelligenceSettings).toHaveBeenCalledTimes(1);

    await user.click(within(empty).getByRole("button", { name: "Write a page" }));
    expect(onCreatePage).toHaveBeenCalledTimes(1);
    expect(onCreatePage).toHaveBeenCalledWith(null);

    // The ghost outlines still preview what a compiled page will look like,
    // and the section is announced by its own heading.
    expect(
      within(empty).getByText("Pages will appear here as Wenlan finds patterns."),
    ).toBeInTheDocument();
    expect(
      within(empty).getByRole("heading", { name: "No pages yet." }),
    ).toBeInTheDocument();
  });

  it("promises compiled pages when a provider is configured, milestone or not", async () => {
    const onCreatePage = vi.fn();
    const onOpenIntelligenceSettings = vi.fn();
    const user = userEvent.setup();
    // A key is saved but has never served an inference, so the milestone has
    // not latched. The provider is nonetheless on: pages really are coming.
    vi.mocked(tauri.listOnboardingMilestones).mockResolvedValue([]);
    withAnthropicKey();

    renderHome({ onCreatePage, onOpenIntelligenceSettings });

    const empty = await screen.findByTestId("wiki-page-empty");
    await waitFor(() =>
      expect(empty).toHaveTextContent(
        /Wenlan compiles pages as patterns emerge in your memories, usually within a day of regular use/,
      ),
    );
    expect(empty).not.toHaveTextContent(/an API key is turned on/);

    // Nothing to turn on, so only the write action is offered.
    expect(within(empty).queryByRole("button", { name: "Turn on a model" })).toBeNull();
    await user.click(within(empty).getByRole("button", { name: "Write a page" }));
    expect(onCreatePage).toHaveBeenCalledWith(null);
    expect(onOpenIntelligenceSettings).not.toHaveBeenCalled();
  });

  it.each([
    ["an on-device model the daemon has loaded", () => {
      vi.mocked(tauri.getOnDeviceModel).mockResolvedValue({
        loaded: "qwen3-1.7b",
        selected: "qwen3-1.7b",
        models: [
          {
            id: "qwen3-1.7b",
            display_name: "Qwen3 1.7B",
            param_count: "1.7B",
            ram_required_gb: 4,
            file_size_gb: 1.2,
            cached: true,
          },
        ],
      });
    }],
    ["a saved external endpoint", () => {
      vi.mocked(tauri.getExternalLlm).mockResolvedValue([
        "http://127.0.0.1:11434/v1",
        "llama3.1",
      ]);
    }],
  ] as const)("counts %s as a configured provider", async (_label, configure) => {
    configure();

    renderHome();

    const empty = await screen.findByTestId("wiki-page-empty");
    await waitFor(() =>
      expect(empty).toHaveTextContent(/as patterns emerge in your memories/),
    );
    expect(within(empty).queryByRole("button", { name: "Turn on a model" })).toBeNull();
  });

  it("commits to neither variant while the provider queries are unresolved", async () => {
    // The key query never settles: the answer to "is a provider on" is
    // unknown, so neither the promise nor the turn-on button may appear.
    vi.mocked(tauri.getApiKey).mockImplementation(() => new Promise(() => {}));

    renderHome();

    const empty = await screen.findByTestId("wiki-page-empty");
    expect(empty).toHaveTextContent("No pages yet.");
    expect(empty).not.toHaveTextContent(/an API key is turned on/);
    expect(empty).not.toHaveTextContent(/as patterns emerge in your memories/);
    expect(within(empty).queryByRole("button", { name: "Turn on a model" })).toBeNull();
    // The action that is true in every state still renders immediately.
    expect(within(empty).getByRole("button", { name: "Write a page" })).toBeInTheDocument();
    expect(
      within(empty).getByText("Pages will appear here as Wenlan finds patterns."),
    ).toBeInTheDocument();
  });

  it("treats a library of nothing but entity shadow pages as empty", async () => {
    // The daemon's browse list carries a shadow page per entity by contract.
    // The rail already excludes them from Pages; the empty-state switch must
    // agree, or the home shows "PAGES 0" beside a list of shadow pages.
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({
        id: "shadow-lucian",
        title: "Lucian",
        creation_kind: "entity",
        entity_id: "ent-1",
      }),
    ]);

    renderHome();

    expect(await screen.findByTestId("wiki-page-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("wiki-page-list")).toBeNull();
    expect(screen.getByTestId("wiki-context-pages")).toHaveTextContent("0");
    expect(screen.queryByText("Lucian")).toBeNull();
  });

  it("keeps Where AI looked on the home surface when there are no pages yet", async () => {
    vi.mocked(tauri.listRecentRetrievals).mockResolvedValue([
      {
        timestamp_ms: Date.now(),
        agent_name: "claude-code",
        query: "positioning",
        page_titles: ["Wenlan positioning"],
        page_ids: ["page-positioning"],
        memory_snippets: [],
      },
    ]);

    renderHome();

    const home = await screen.findByTestId("wiki-home");
    expect(await screen.findByTestId("wiki-page-empty")).toBeInTheDocument();
    const retrievals = within(home).getByTestId("retrievals");
    expect(within(retrievals).getByText(/Where AI looked/i)).toBeInTheDocument();
    expect(within(retrievals).getByText(/Claude Code/)).toBeInTheDocument();
  });

  it("shows the page list and no empty state once pages exist", async () => {
    const onCreatePage = vi.fn();
    const onOpenIntelligenceSettings = vi.fn();
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-architecture", title: "Wenlan app architecture" }),
    ]);

    renderHome({ onCreatePage, onOpenIntelligenceSettings });

    await screen.findByTestId("wiki-page-list");
    expect(screen.queryByTestId("wiki-page-empty")).toBeNull();
    expect(screen.queryByRole("button", { name: "Turn on a model" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Write a page" })).toBeNull();
    expect(
      screen.getByRole("button", { name: /open Wenlan app architecture/i }),
    ).toBeInTheDocument();
  });

  it("shows a load-error state with retry in the needs-review rail when the queue fetch fails and the queue is empty, never All caught up", async () => {
    vi.mocked(tauri.listRefinements).mockRejectedValue(new Error("daemon can't deserialize a proposal variant"));
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);

    renderHome();

    const rail = await screen.findByTestId("worth-a-glance");
    await within(rail).findByText(/Couldn't load review items/);
    expect(within(rail).queryByText(/All caught up/)).not.toBeInTheDocument();

    await userEvent.click(within(rail).getByRole("button", { name: "Try again" }));
    expect(tauri.listRefinements).toHaveBeenCalledTimes(2);
  });

  it("shows the queue items plus a quiet partial-load notice when one source fails but others have data", async () => {
    vi.mocked(tauri.listRefinements).mockRejectedValue(new Error("daemon can't deserialize a proposal variant"));
    vi.mocked(tauri.listPendingRevisions).mockResolvedValue([
      {
        target_source_id: "mem-target",
        revision_source_id: "mem-revision",
        revision_content: "The durable updated wording from the daemon.",
        source_agent: "claude-code",
        last_modified: 1_782_365_076,
        target_kind: "memory" as const,
      },
    ]);
    vi.mocked(tauri.listPages).mockResolvedValue([
      page({ id: "page-current", title: "Current page", domain: "Projects" }),
    ]);

    renderHome();

    const rail = await screen.findByTestId("worth-a-glance");
    await within(rail).findByText(/The durable updated wording/);
    await within(rail).findByText(/Some review items couldn't load/);
    // Self-heals via the 30s refetch — no retry button while items are shown.
    expect(within(rail).queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });
});
