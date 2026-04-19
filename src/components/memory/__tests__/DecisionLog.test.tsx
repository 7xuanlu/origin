// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import DecisionLog from "../DecisionLog";
import { listDecisions, listDecisionDomains } from "../../../lib/tauri";

const MOCK_DECISIONS = [
  {
    source_id: "d1",
    title: "Use libSQL over Neo4j",
    content: "Keep libSQL — no dedicated graph database",
    summary: null,
    memory_type: "decision",
    domain: "architecture",
    source_agent: "claude-code",
    confidence: 0.9,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: 1744041600,
    chunk_count: 1,
    structured_fields: JSON.stringify({
      decision: "Keep libSQL",
      context: "Evaluated Neo4j and FalkorDB before deciding libSQL sufficient.",
      alternatives_considered: "Neo4j,FalkorDB,ArcadeDB",
      reversible: false,
    }),
  },
  {
    source_id: "d2",
    title: "Ship wiki-style concepts",
    content: "Use encyclopedia prose format for concepts",
    summary: null,
    memory_type: "decision",
    domain: "product",
    source_agent: "claude-code",
    confidence: 0.8,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: 1744128000,
    chunk_count: 1,
    structured_fields: JSON.stringify({
      decision: "Wiki prose format",
      context: "Bullet lists felt robotic. Prose reads better.",
      alternatives_considered: "bullet lists,structured cards",
      reversible: true,
    }),
  },
];

vi.mock("../../../lib/tauri", () => ({
  listDecisions: vi.fn(),
  listDecisionDomains: vi.fn(),
}));

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

const noop = () => {};

describe("DecisionLog", () => {
  beforeEach(() => {
    vi.mocked(listDecisions).mockResolvedValue(MOCK_DECISIONS as any);
    vi.mocked(listDecisionDomains).mockResolvedValue(["architecture", "product"]);
  });

  it("renders decision entries after loading", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    expect(await screen.findByText(/Decisions/)).toBeInTheDocument();
    expect(await screen.findByText("Use libSQL over Neo4j")).toBeInTheDocument();
    expect(await screen.findByText("Ship wiki-style concepts")).toBeInTheDocument();
  });

  it("shows domain filter pills", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    // Wait for data to load via a domain pill
    const archPill = await screen.findByRole("button", { name: "architecture" });
    expect(archPill).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "product" })).toBeInTheDocument();
    // "All" pill always present
    expect(screen.getByRole("button", { name: "All" })).toBeInTheDocument();
  });

  it("shows context preview in collapsed state", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    expect(await screen.findByText(/Evaluated Neo4j/)).toBeInTheDocument();
  });

  it("shows alternatives when entry is expanded", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    // Wait for entries to load
    const entry = await screen.findByText("Use libSQL over Neo4j");
    // Click to expand
    fireEvent.click(entry.closest("[data-testid='decision-entry']")!);
    expect(await screen.findByText("Neo4j")).toBeInTheDocument();
    expect(screen.getByText("FalkorDB")).toBeInTheDocument();
    expect(screen.getByText("ArcadeDB")).toBeInTheDocument();
  });

  it("calls onBack when back button is clicked", async () => {
    const onBack = vi.fn();
    render(wrap(<DecisionLog onBack={onBack} onSelectMemory={noop} onSelectConcept={noop} />));
    await screen.findByText(/Decisions/);
    fireEvent.click(screen.getByLabelText("Go back"));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("shows reversibility indicator when expanded", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    // Expand the second entry which has reversible: true
    const entry = await screen.findByText("Ship wiki-style concepts");
    fireEvent.click(entry.closest("[data-testid='decision-entry']")!);
    expect(await screen.findByText(/Reversible/)).toBeInTheDocument();
  });

  it("filters by domain when pill is clicked", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    const architecturePill = await screen.findByRole("button", { name: "architecture" });
    fireEvent.click(architecturePill);
    await waitFor(() => {
      // After clicking domain pill, listDecisions should be called with that domain
      expect(listDecisions).toHaveBeenCalledWith("architecture", 100);
    });
  });

  it("shows count in header", async () => {
    render(wrap(<DecisionLog onBack={noop} onSelectMemory={noop} onSelectConcept={noop} />));
    // 2 decisions in mock data
    expect(await screen.findByText("2")).toBeInTheDocument();
  });
});
