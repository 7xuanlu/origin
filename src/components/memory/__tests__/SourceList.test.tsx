// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SourceList from "../SourceList";
import * as tauri from "../../../lib/tauri";
import type { RegisteredSource } from "../../../lib/tauri";

vi.mock("../../../lib/tauri");

const SOURCES: RegisteredSource[] = [
  {
    id: "obsidian-second-brain",
    source_type: "obsidian",
    path: "/Users/test/second-brain",
    status: "Active",
    last_sync: 1712700000,
    file_count: 2146,
    memory_count: 3118,
  },
  {
    id: "directory-papers",
    source_type: "directory",
    path: "/Users/test/papers",
    status: { Error: "sync failed" },
    last_sync: null,
    file_count: 42,
    memory_count: 87,
  },
  {
    id: "directory-books",
    source_type: "directory",
    path: "/Users/test/books",
    status: { Unavailable: "root missing" },
    last_sync: null,
    file_count: 3,
    memory_count: 12,
  },
];

function renderSourceList(onNavigateSources = vi.fn()) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <SourceList onNavigateSources={onNavigateSources} />
    </QueryClientProvider>,
  );
  return onNavigateSources;
}

describe("SourceList", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue(SOURCES);
  });

  it("renders a Sources section with one row per registered source and its memory count", async () => {
    renderSourceList();

    expect(await screen.findByText("second-brain")).toBeInTheDocument();
    expect(screen.getByText("papers")).toBeInTheDocument();
    expect(screen.getByText("books")).toBeInTheDocument();
    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("3118")).toBeInTheDocument();
  });

  it("renders a spine per source, tallest for the source with the most memories", async () => {
    renderSourceList();

    await screen.findByText("second-brain");
    const spines = screen.getAllByTestId("source-spine");
    expect(spines).toHaveLength(3);

    const heights = spines.map((el) => parseInt(el.style.height, 10));
    // second-brain (3118 memories) caps the scale at 14px; the others sit above the 8px floor
    expect(heights[0]).toBe(14);
    expect(Math.max(...heights.slice(1))).toBeLessThan(14);
    expect(Math.min(...heights)).toBeGreaterThanOrEqual(8);
  });

  it("marks non-active sources with their sync status", async () => {
    renderSourceList();

    await screen.findByText("papers");
    expect(screen.getByTitle(/Sync error/)).toBeInTheDocument();
    expect(screen.getByTitle(/Unavailable/)).toBeInTheDocument();
    // Active sources carry no warning marker
    expect(screen.getByTitle("/Users/test/second-brain")).toBeInTheDocument();
  });

  it("navigates to source management when a row is clicked", async () => {
    const user = userEvent.setup();
    const onNavigateSources = renderSourceList();

    await user.click(await screen.findByText("second-brain"));

    expect(onNavigateSources).toHaveBeenCalledTimes(1);
  });

  it("navigates to source management from the header add button", async () => {
    const user = userEvent.setup();
    const onNavigateSources = renderSourceList();

    await screen.findByText("second-brain");
    await user.click(screen.getByTitle("Add source"));

    expect(onNavigateSources).toHaveBeenCalledTimes(1);
  });

  it("shows an add-source affordance when no sources are registered", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);
    const user = userEvent.setup();
    const onNavigateSources = renderSourceList();

    const add = await screen.findByRole("button", { name: /add folder or vault/i });
    await user.click(add);

    expect(onNavigateSources).toHaveBeenCalledTimes(1);
  });

  it("collapses the section when the header is clicked", async () => {
    const user = userEvent.setup();
    renderSourceList();

    await screen.findByText("second-brain");
    await user.click(screen.getByText("Sources"));

    expect(screen.queryByText("second-brain")).not.toBeInTheDocument();
  });
});
