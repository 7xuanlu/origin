// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConceptDetail from "../ConceptDetail";
import * as tauri from "../../../lib/tauri";

vi.mock("../../../lib/tauri");

const MOCK_CONCEPT: tauri.Concept = {
  id: "c1",
  title: "Test Concept",
  content: "Some concept content",
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

describe("ConceptDetail export", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(tauri.getConcept).mockResolvedValue(MOCK_CONCEPT);
    vi.mocked(tauri.listConcepts).mockResolvedValue([MOCK_CONCEPT]);
  });

  it("disables export button when no obsidian sources exist", async () => {
    vi.mocked(tauri.listRegisteredSources).mockResolvedValue([]);

    render(
      <ConceptDetail
        conceptId="c1"
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
    vi.mocked(tauri.exportConceptToObsidian).mockResolvedValue(
      "/Users/test/vault/Origin/concepts/Test Concept.md",
    );

    render(
      <ConceptDetail
        conceptId="c1"
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
      expect(tauri.exportConceptToObsidian).toHaveBeenCalledWith(
        "c1",
        "/Users/test/vault/Origin/concepts",
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
      <ConceptDetail
        conceptId="c1"
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
      expect(screen.getByText("vault-one")).toBeInTheDocument();
      expect(screen.getByText("vault-two")).toBeInTheDocument();
    });
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
    vi.mocked(tauri.exportConceptToObsidian).mockResolvedValue(
      "/Users/test/vault-two/Origin/concepts/Test Concept.md",
    );

    render(
      <ConceptDetail
        conceptId="c1"
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
      expect(tauri.exportConceptToObsidian).toHaveBeenCalledWith(
        "c1",
        "/Users/test/vault-two/Origin/concepts",
      );
    });
  });
});
