// SPDX-License-Identifier: AGPL-3.0-only
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../../lib/tauri", () => ({
  acceptPendingRevision: vi.fn(), agentDisplayName: (slug: string | null) => slug,
  confirmSpace: vi.fn(), createPage: vi.fn(), deleteFileChunks: vi.fn(), deleteSpace: vi.fn(),
  dismissPendingRevision: vi.fn(), FACET_COLORS: {}, getNurtureCards: vi.fn().mockResolvedValue([]),
  getPendingRevision: vi.fn().mockResolvedValue(null), getSpace: vi.fn(),
  getVersionChain: vi.fn().mockResolvedValue([]), listEntities: vi.fn(),
  listMemoriesRich: vi.fn(), listPages: vi.fn(), listSpaces: vi.fn(), pinMemory: vi.fn(),
  setStability: vi.fn(), STABILITY_TIERS: {}, unpinMemory: vi.fn(),
  updateMemory: vi.fn(), updateSpace: vi.fn(),
}));

import type { Entity, MemoryItem, Page, Space } from "../../../lib/tauri";
import { getSpace, listEntities, listMemoriesRich, listPages, listSpaces } from "../../../lib/tauri";
import SpaceDetail from "../SpaceDetail";
import { SPACE_DETAIL_TEST_COPY } from "./testTranslation";

const baseSpace: Space = {
  id: "s1", name: "Wenlan", description: "Editorial memory", suggested: false,
  starred: true, sort_order: 0, memory_count: 250, entity_count: 8,
  created_at: 1_700_000_000, updated_at: 1_700_000_000,
};

function makePage(id: string, title: string, lastModified: string, staleReason?: string): Page {
  return {
    id, title, summary: `${title} summary`, content: title, entity_id: null,
    domain: "Wenlan", source_memory_ids: [`m-${id}`], version: 1, status: "active",
    created_at: "2026-07-01T00:00:00Z", last_compiled: lastModified,
    last_modified: lastModified, ...(staleReason ? { stale_reason: staleReason } : {}),
  };
}

function makeEntity(id: string, name: string, confirmed: boolean, confidence: number, updatedAt: number): Entity {
  return {
    id, name, entity_type: "topic", domain: "Wenlan", source_agent: "codex",
    confidence, confirmed, created_at: 1_700_000_000, updated_at: updatedAt,
  };
}

const memory: MemoryItem = {
  source_id: "m1", title: "Latest raw memory", content: "Body", summary: null,
  memory_type: "fact", domain: "Wenlan", source_agent: "codex", confidence: 0.9,
  confirmed: true, pinned: false, supersedes: null, last_modified: 2_000, chunk_count: 1,
};

function renderDetail(overrides: Partial<React.ComponentProps<typeof SpaceDetail>> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const props = {
    copy: SPACE_DETAIL_TEST_COPY, spaceName: "Wenlan", onBack: vi.fn(), onSelectMemory: vi.fn(),
    onSelectPage: vi.fn(), onEntityClick: vi.fn(), onReviewAll: vi.fn(), ...overrides,
  };
  render(<QueryClientProvider client={client}><SpaceDetail {...props} /></QueryClientProvider>);
  return props;
}

describe("SpaceDetail editorial dossier", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSpace).mockResolvedValue(baseSpace);
    vi.mocked(listMemoriesRich).mockResolvedValue([memory]);
    vi.mocked(listEntities).mockResolvedValue([]);
    vi.mocked(listPages).mockResolvedValue([]);
    vi.mocked(listSpaces).mockResolvedValue([baseSpace]);
  });

  it("opens a Page editor for the current Space before the overflow action", async () => {
    const onCreatePage = vi.fn();
    renderDetail({ onCreatePage });

    const newPage = await screen.findByRole("button", { name: "New page" });
    const overflow = screen.getByRole("button", { name: "Actions for Wenlan" });
    expect(newPage.compareDocumentPosition(overflow) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);

    fireEvent.click(newPage);
    expect(onCreatePage).toHaveBeenCalledWith("Wenlan");
    expect(screen.queryByRole("dialog", { name: "New page" })).not.toBeInTheDocument();
  });

  it("loads the full active page cap and renders exact quiet metrics", async () => {
    const pages = Array.from({ length: 1_000 }, (_, index) =>
      makePage(`p${index}`, `Page ${String(index).padStart(4, "0")}`, "2026-07-09T20:00:00Z"),
    );
    vi.mocked(listPages).mockResolvedValue(pages);
    renderDetail();

    expect(await screen.findByRole("heading", { level: 1, name: "Wenlan" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByText("1,000+")).toBeInTheDocument();
    expect(screen.getAllByText("250")).not.toHaveLength(0);
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getAllByText("Jul 9, 2026")).not.toHaveLength(0);
    expect(listPages).toHaveBeenCalledWith("active", "Wenlan", 1_000);

    const css = readFileSync(resolve("src/components/memory/space-detail/space-detail-header.css"), "utf8");
    expect(css).toMatch(/\.space-dossier-metrics dd\s*\{[^}]*font:\s*15px var\(--mem-font-mono\)[^}]*font-variant-numeric:\s*tabular-nums/s);
    expect(css).toMatch(/\.space-dossier-metrics dt\s*\{[^}]*font:\s*11px var\(--mem-font-mono\)/s);
  });

  it("puts each quiet metric label before its value", async () => {
    renderDetail();

    await screen.findByRole("heading", { level: 1, name: "Wenlan" });
    const metrics = document.querySelector(".space-dossier-metrics");
    expect(metrics).not.toBeNull();
    expect(
      Array.from(metrics?.children ?? []).map((metric) =>
        Array.from(metric.children).map((child) => child.tagName),
      ),
    ).toEqual([
      ["DT", "DD"],
      ["DT", "DD"],
      ["DT", "DD"],
      ["DT", "DD"],
    ]);
  });

  it("uses compact tonal actions for a suggested Space", async () => {
    vi.mocked(getSpace).mockResolvedValue({ ...baseSpace, suggested: true });
    renderDetail();

    const keep = await screen.findByRole("button", { name: "Keep" });
    const discard = screen.getByRole("button", { name: "Discard" });
    expect(keep).toHaveClass("space-dossier-suggestion-action", "space-dossier-suggestion-keep");
    expect(discard).toHaveClass("space-dossier-suggestion-action", "space-dossier-suggestion-discard");

    const css = readFileSync(resolve("src/components/memory/space-detail/space-detail-header.css"), "utf8");
    expect(css).toMatch(/\.space-dossier-suggestion-action\s*\{[^}]*font-size:\s*12px[^}]*padding:\s*4px 10px/s);
    expect(css).toMatch(/\.space-dossier-suggestion-keep\s*\{[^}]*background:\s*transparent[^}]*color:\s*var\(--mem-accent-indigo\)/s);
    expect(css).toMatch(/\.space-dossier-suggestion-keep:hover\s*\{[^}]*background:\s*var\(--mem-indigo-bg\)/s);
  });

  it("orders and caps recently refined pages with a stable title tie-break", async () => {
    vi.mocked(listPages).mockResolvedValue([
      makePage("old", "Old", "2026-07-01T00:00:00Z"),
      makePage("b", "Beta", "2026-07-10T00:00:00Z"),
      makePage("a", "Alpha", "2026-07-10T00:00:00Z"),
      makePage("c", "Charlie", "2026-07-09T00:00:00Z"),
      makePage("d", "Delta", "2026-07-08T00:00:00Z"),
      makePage("e", "Echo", "2026-07-07T00:00:00Z"),
      makePage("bad", "Invalid", "not-a-date"),
    ]);
    const onSelectPage = vi.fn();
    renderDetail({ onSelectPage });

    const region = await screen.findByRole("region", { name: "Recently refined" });
    const rows = within(region).getAllByRole("button");
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining("Alpha"), expect.stringContaining("Beta"),
      expect.stringContaining("Charlie"), expect.stringContaining("Delta"),
      expect.stringContaining("Echo"),
    ]);
    expect(region).not.toHaveTextContent("Invalid");
    expect(document.body).not.toHaveTextContent("NaN");
    fireEvent.click(rows[0]);
    expect(onSelectPage).toHaveBeenCalledWith("a");
  });

  it("prioritizes three review pages and opens the global review queue", async () => {
    vi.mocked(listPages).mockResolvedValue([
      makePage("u1", "Updated first", "2026-07-10T00:00:00Z", "source_updated"),
      makePage("c1", "Conflict old", "2026-07-01T00:00:00Z", "source_conflict"),
      makePage("c2", "Conflict new", "2026-07-09T00:00:00Z", "source_conflict"),
      makePage("u2", "Updated second", "2026-07-08T00:00:00Z", "source_updated"),
      makePage("blank", "Blank", "2026-07-11T00:00:00Z", "   "),
    ]);
    const onReviewAll = vi.fn();
    renderDetail({ onReviewAll });

    const region = await screen.findByRole("region", { name: "Needs review" });
    const rows = within(region).getAllByRole("button");
    expect(rows.slice(0, 3).map((row) => row.textContent)).toEqual([
      expect.stringContaining("Conflict new"), expect.stringContaining("Conflict old"),
      expect.stringContaining("Updated first"),
    ]);
    expect(region).not.toHaveTextContent("Blank");
    const reviewAll = within(region).getByRole("button", { name: "Review all" });
    expect(reviewAll).toHaveClass("space-dossier-text-action", "space-dossier-text-action-review");
    fireEvent.click(reviewAll);
    expect(onReviewAll).toHaveBeenCalledTimes(1);

    const css = readFileSync(resolve("src/components/memory/space-detail/space-detail.css"), "utf8");
    expect(css).toMatch(
      /\.space-dossier-text-action-review\s*\{[^}]*color:\s*var\(--mem-accent-indigo\)/s,
    );
  });

  it("caps sorted key entities at six and expands in place", async () => {
    vi.mocked(listEntities).mockResolvedValue([
      makeEntity("u", "Unconfirmed", false, 1, 9_999),
      makeEntity("z", "Zulu", true, 0.9, 1_000), makeEntity("a", "Alpha", true, 0.9, 1_000),
      makeEntity("b", "Beta", true, 0.8, 3_000), makeEntity("c", "Charlie", true, 0.8, 2_000),
      makeEntity("d", "Delta", true, 0.7, 4_000), makeEntity("e", "Echo", true, 0.6, 5_000),
      makeEntity("f", "Foxtrot", true, 0.5, 6_000),
    ]);
    renderDetail();

    const region = await screen.findByRole("region", { name: "Key entities" });
    expect(within(region).queryByRole("button", { name: "Foxtrot" })).not.toBeInTheDocument();
    const viewAll = within(region).getByRole("button", { name: "View all 8" });
    expect(viewAll).toHaveClass("space-dossier-text-action");
    expect(viewAll).not.toHaveClass("space-dossier-text-action-review");
    fireEvent.click(viewAll);
    expect(within(region).getByRole("button", { name: "Foxtrot" })).toBeInTheDocument();
    expect(within(region).getAllByRole("button").slice(0, 3).map((row) => row.textContent)).toEqual(["Alpha", "Zulu", "Beta"]);
  });

  it("keeps the raw archive collapsed and discloses a 200-of-N limit", async () => {
    vi.mocked(listMemoriesRich).mockResolvedValue(Array.from({ length: 200 }, (_, index) => ({
      ...memory, source_id: `m${index}`, title: index === 0 ? memory.title : `Memory ${index}`,
    })));
    renderDetail();

    const region = await screen.findByRole("region", { name: "Raw memories" });
    expect(region).toHaveTextContent("Showing the latest 200 of 250 memories");
    expect(within(region).queryByText("Latest raw memory")).not.toBeInTheDocument();
    fireEvent.click(within(region).getByRole("button", { name: "Raw memories (250)" }));
    expect(await within(region).findByText("Latest raw memory")).toBeInTheDocument();
    expect(within(region).getByRole("button", { name: "Curated" })).toBeInTheDocument();
  }, 15_000);

  it("places the archive after the two-column dossier content", async () => {
    renderDetail();
    const recent = await screen.findByRole("region", { name: "Recently refined" });
    const review = screen.getByRole("region", { name: "Needs review" });
    const entities = screen.getByRole("region", { name: "Key entities" });
    const archive = screen.getByRole("region", { name: "Raw memories" });
    expect(recent.compareDocumentPosition(review) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(review.compareDocumentPosition(entities) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(entities.compareDocumentPosition(archive) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(document.querySelector(".space-dossier-grid")).toBeInTheDocument();
  });
});
