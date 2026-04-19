// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { MemoryItem } from "../../lib/tauri";

vi.mock("../../lib/tauri", () => ({
  FACET_COLORS: {
    identity: "bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30",
    preference: "bg-purple-500/20 text-purple-700 dark:text-purple-400 border-purple-500/30",
    decision: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30",
    fact: "bg-zinc-500/20 text-zinc-700 dark:text-zinc-400 border-zinc-500/30",
    goal: "bg-green-500/20 text-green-700 dark:text-green-400 border-green-500/30",
  },
  STABILITY_TIERS: {
    identity: "protected",
    preference: "protected",
    fact: "standard",
    decision: "standard",
    goal: "ephemeral",
  },
  getPendingRevision: vi.fn().mockResolvedValue(null),
  acceptPendingRevision: vi.fn(),
  dismissPendingRevision: vi.fn(),
}));

import MemoryCard from "./MemoryCard";

function makeMemory(overrides: Partial<MemoryItem> = {}): MemoryItem {
  return {
    source_id: overrides.source_id ?? "mem-1",
    title: overrides.title ?? "Test memory title",
    content: overrides.content ?? "Test content",
    summary: overrides.summary ?? null,
    memory_type: overrides.memory_type ?? "preference",
    domain: overrides.domain ?? null,
    source_agent: overrides.source_agent ?? null,
    confidence: overrides.confidence ?? null,
    confirmed: overrides.confirmed ?? false,
    pinned: overrides.pinned ?? false,
    supersedes: overrides.supersedes ?? null,
    last_modified: overrides.last_modified ?? Date.now() / 1000,
    chunk_count: overrides.chunk_count ?? 1,
    access_count: overrides.access_count ?? 0,
    is_recap: overrides.is_recap ?? false,
  };
}

const defaultProps = {
  onConfirm: vi.fn(),
  onDelete: vi.fn(),
  expandedChain: false,
  onToggleChain: vi.fn(),
  versionChain: [],
};

describe("MemoryCard insight variant", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders content without type badge", () => {
    const memory = makeMemory({ title: "Prefers dark mode", memory_type: "preference" });
    render(<MemoryCard memory={memory} variant="insight" {...defaultProps} />);

    expect(screen.getByText("Prefers dark mode")).toBeInTheDocument();
    expect(screen.queryByText("preference")).not.toBeInTheDocument();
  });

  it("shows human-readable provenance", () => {
    const memory = makeMemory({
      title: "Likes TypeScript",
      source_agent: "claude-code",
      confirmed: true,
    });
    render(<MemoryCard memory={memory} variant="insight" {...defaultProps} />);

    expect(screen.getByText(/From Claude Code/)).toBeInTheDocument();
    expect(screen.getByText(/confirmed/)).toBeInTheDocument();
  });

  it("does not render confirm dot", () => {
    const memory = makeMemory({ confirmed: false });
    render(<MemoryCard memory={memory} variant="insight" {...defaultProps} />);

    expect(screen.queryByTestId("confirm-dot")).not.toBeInTheDocument();
  });

  it("renders full variant with type badge by default", () => {
    const memory = makeMemory({ memory_type: "preference" });
    render(<MemoryCard memory={memory} {...defaultProps} />);

    expect(screen.getByText("preference")).toBeInTheDocument();
  });
});
