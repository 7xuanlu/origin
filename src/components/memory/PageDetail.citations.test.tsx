// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PageCitation } from "../../lib/tauri";
import PageDetail from "./PageDetail";

const tauriMocks = vi.hoisted(() => ({
  getPage: vi.fn(),
  getPageSources: vi.fn(),
  listRegisteredSources: vi.fn(),
  getPageLinks: vi.fn(),
  getPageRevisions: vi.fn(),
  redistillPage: vi.fn(),
  updatePage: vi.fn(),
  deletePage: vi.fn(),
  clipboardWrite: vi.fn(),
  exportPageToObsidian: vi.fn(),
}));

vi.mock("../../lib/tauri", () => ({
  ...tauriMocks,
  FACET_COLORS: {},
  STABILITY_TIERS: {},
}));

const cite = (
  occurrence: number,
  marker: number,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator: `mem-${marker}`,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const BASE_PAGE = {
  id: "page-1",
  title: "Cited Page",
  summary: null,
  content:
    "# Cited Page\n\nIntro sentence stands alone. The daemon is local-first.[1] It uses libSQL.[2]",
  entity_id: null,
  domain: "testing",
  source_memory_ids: ["mem-1", "mem-2"],
  version: 1,
  status: "active",
  created_at: "2026-06-26T00:00:00+00:00",
  last_compiled: "2026-06-26T00:00:00+00:00",
  last_modified: "2026-06-26T00:00:00+00:00",
  citations: [cite(1, 1), cite(2, 2, { status: "unverified" })],
};

const SOURCES = [
  {
    source: { page_id: "page-1", memory_source_id: "mem-1", linked_at: 0 },
    memory: {
      source_id: "mem-1",
      title: "Local-first decision",
      content: "We keep the daemon local-first.",
      summary: null,
      memory_type: "memory",
      domain: null,
      source_agent: "claude-code",
      confidence: null,
      confirmed: true,
      pinned: false,
      supersedes: null,
      last_modified: 1_700_000_000,
      chunk_count: 1,
    },
  },
];

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const props = {
    pageId: "page-1",
    onBack: vi.fn(),
    onMemoryClick: vi.fn(),
    onPageClick: vi.fn(),
  };
  render(
    <QueryClientProvider client={client}>
      <PageDetail {...props} />
    </QueryClientProvider>,
  );
  return { props, user: userEvent.setup() };
}

beforeEach(() => {
  vi.clearAllMocks();
  tauriMocks.getPage.mockResolvedValue(BASE_PAGE);
  tauriMocks.getPageSources.mockResolvedValue(SOURCES);
  tauriMocks.listRegisteredSources.mockResolvedValue([]);
  tauriMocks.getPageLinks.mockResolvedValue({ outbound: [], inbound: [] });
  tauriMocks.getPageRevisions.mockResolvedValue({
    page_id: "page-1",
    current_version: 1,
    user_edited: false,
    stale_reason: null,
    entries: [],
  });
  tauriMocks.redistillPage.mockResolvedValue({ status: "ok", updated: true });
});

describe("PageDetail citations", () => {
  it("renders one chip per citation and no raw markers in the body", async () => {
    renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    const chip1 = await screen.findByRole("button", { name: /mem-1/ });
    const chip2 = screen.getByRole("button", { name: /mem-2/ });
    expect(chip1).toHaveAttribute("data-status", "verified");
    expect(chip2).toHaveAttribute("data-status", "unverified");
    expect(screen.queryByText(/\[1\]/)).toBeNull();
  });

  it("resolves the popover from page-sources and opens the memory", async () => {
    const { props, user } = renderPage();
    const chip = await screen.findByRole("button", { name: /mem-1/ });
    fireEvent.focus(chip);
    // Scoped to the popover: PageInfo's (closed, but DOM-present) Sources row
    // for the same memory carries the identical title text, and plain
    // getByText/findByText don't filter on visibility — only toBeVisible()
    // understands a closed <details>. Disambiguate by container instead.
    const popover = await screen.findByRole("tooltip");
    expect(within(popover).getByText("Local-first decision")).toBeInTheDocument();
    await user.click(within(popover).getByRole("button", { name: /Open memory/ }));
    expect(props.onMemoryClick).toHaveBeenCalledWith("mem-1");
  });

  it("explains a locator missing from page-sources", async () => {
    renderPage();
    const chip = await screen.findByRole("button", { name: /mem-2/ });
    fireEvent.focus(chip);
    expect(await screen.findByText(/no longer exists/i)).toBeInTheDocument();
  });

  it("display-strips markers when citations were cleared by an edit", async () => {
    tauriMocks.getPage.mockResolvedValue({ ...BASE_PAGE, citations: undefined });
    const { user } = renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    expect(screen.getByText(/It uses libSQL\./)).toBeInTheDocument();
    expect(screen.queryByText(/\[2\]/)).toBeNull();
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citations cleared by edit — re-distill to restore"),
    ).toBeInTheDocument();
  });

  it("falls back to strip-all on count mismatch and reports it", async () => {
    tauriMocks.getPage.mockResolvedValue({ ...BASE_PAGE, citations: [cite(1, 1)] });
    const { user } = renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    expect(screen.queryByText(/\[1\]/)).toBeNull();
    await user.click(screen.getByText(/Page info/i));
    expect(
      screen.getByText("Citation data mismatched — re-distill to repair"),
    ).toBeInTheDocument();
  });

  it("keeps the TLDR pull-quote free of markers and citation links", async () => {
    // First `.\s` sentence boundary lands AFTER marker [1], so the extracted
    // pull-quote contains a rewritten citation link that must be stripped.
    tauriMocks.getPage.mockResolvedValue({
      ...BASE_PAGE,
      content:
        "# Cited Page\n\nThe daemon is local-first.[1] It stays fast under load. Second paragraph here.[2]",
    });
    renderPage();
    expect(await screen.findByText("Cited Page")).toBeInTheDocument();
    const quote = screen.getByText(/It stays fast under load\./);
    expect(quote.textContent).not.toMatch(/\[\d+\]/);
    expect(quote.textContent).not.toContain("#citation");
    // Known ceiling (spec §4.3): the first-sentence citation gets no inline
    // chip; the second citation still renders in the body.
    expect(screen.queryByRole("button", { name: /mem-1/ })).toBeNull();
    expect(screen.getByRole("button", { name: /mem-2/ })).toBeInTheDocument();
  });
});
