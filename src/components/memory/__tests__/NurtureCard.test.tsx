import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("../../../lib/tauri", () => ({
  correctMemory: vi.fn(),
  updateMemory: vi.fn(),
}));

import NurtureCard from "../NurtureCard";

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

const mockMemory = {
  source_id: "test-1",
  title: "Test Memory",
  content: "Chose Google Play Internal Testing over EAS Internal Distribution for Ruru beta distribution because it offers free testing tracks",
  summary: null as string | null,
  memory_type: "decision" as string | null,
  domain: "ruru" as string | null,
  source_agent: "claude-code" as string | null,
  confidence: 0.7 as number | null,
  confirmed: false,
  stability: "new" as const,
  pinned: false,
  supersedes: null as string | null,
  last_modified: Date.now() / 1000,
  chunk_count: 1,
  structured_fields: JSON.stringify({ decision: "Google Play Internal Testing over EAS" }) as string | null,
};

describe("NurtureCard", () => {
  it("shows type badge and domain", () => {
    render(<NurtureCard memory={mockMemory} onConfirm={vi.fn()} onDismiss={vi.fn()} onDelete={vi.fn()} />, { wrapper });
    expect(screen.getByText("decision")).toBeInTheDocument();
    expect(screen.getByText("ruru")).toBeInTheDocument();
  });

  it("shows structured field as smart summary when available", () => {
    render(<NurtureCard memory={mockMemory} onConfirm={vi.fn()} onDismiss={vi.fn()} onDelete={vi.fn()} />, { wrapper });
    expect(screen.getByText(/Google Play Internal Testing over EAS/)).toBeInTheDocument();
  });

  it("calls onConfirm when Yes is clicked", () => {
    const onConfirm = vi.fn();
    render(<NurtureCard memory={mockMemory} onConfirm={onConfirm} onDismiss={vi.fn()} onDelete={vi.fn()} />, { wrapper });
    fireEvent.click(screen.getByText("Yes, that's right"));
    expect(onConfirm).toHaveBeenCalledWith("test-1");
  });

  it("expands correction flow on Not quite", () => {
    render(<NurtureCard memory={mockMemory} onConfirm={vi.fn()} onDismiss={vi.fn()} onDelete={vi.fn()} />, { wrapper });
    fireEvent.click(screen.getByText(/Not quite/));
    expect(screen.getByPlaceholderText(/Describe what's wrong/)).toBeInTheDocument();
  });

  it("shows expand toggle when content is longer than summary", () => {
    render(<NurtureCard memory={mockMemory} onConfirm={vi.fn()} onDismiss={vi.fn()} onDelete={vi.fn()} />, { wrapper });
    // The structured_fields provides a shorter summary, so "more" should appear
    expect(screen.getByText(/more/)).toBeInTheDocument();
  });
});
