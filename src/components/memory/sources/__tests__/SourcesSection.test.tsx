import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SourcesSection from "../SourcesSection";
import * as tauri from "../../../../lib/tauri";

vi.mock("../../../../lib/tauri");

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("SourcesSection", () => {
  beforeEach(() => {
    vi.mocked(tauri.getKnowledgePath).mockResolvedValue(
      "/Users/test/Origin/knowledge",
    );
    vi.mocked(tauri.countKnowledgeFiles).mockResolvedValue(12);
  });

  it("renders empty state when no obsidian sources", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);

    render(<SourcesSection />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("No sources yet")).toBeInTheDocument();
    });
    expect(
      screen.getByRole("button", { name: /add your first source/i }),
    ).toBeInTheDocument();
  });

  it("renders source rows when obsidian sources exist", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([
      {
        id: "obsidian-second-brain",
        source_type: "obsidian",
        path: "/Users/test/second-brain",
        status: "Active",
        last_sync: 1712700000,
        file_count: 2146,
        memory_count: 3118,
      },
    ]);

    render(<SourcesSection />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("second-brain")).toBeInTheDocument();
    });
    expect(screen.getByText(/2,146 files/)).toBeInTheDocument();
    expect(screen.getByText(/3,118 memories/)).toBeInTheDocument();
  });

  it("filters out non-obsidian sources", async () => {
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
      {
        id: "directory-old",
        source_type: "directory",
        path: "/Users/test/old-docs",
        status: "Active",
        last_sync: null,
        file_count: 5,
        memory_count: 5,
      },
    ]);

    render(<SourcesSection />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("vault")).toBeInTheDocument();
    });
    expect(screen.queryByText("old-docs")).not.toBeInTheDocument();
  });

  it("renders knowledge directory block with path and count", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);

    render(<SourcesSection />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText(/Origin\/knowledge/)).toBeInTheDocument();
    });
    expect(screen.getByText(/12 concept files/)).toBeInTheDocument();
  });
});
