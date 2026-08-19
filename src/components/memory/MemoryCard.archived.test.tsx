// SPDX-License-Identifier: AGPL-3.0-only
//
// An 'archive'-mode superseder keeps its predecessor *visible but muted*. The
// list readers used to drop such rows entirely, so a memory replaced by a
// decision was findable nowhere in the app while agents kept recalling it.
// These tests pin the muted-but-present rendering in both grid and list views.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { MemoryItem } from "../../lib/tauri";

vi.mock("../../lib/tauri", () => ({
  FACET_COLORS: {
    decision: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30",
    fact: "bg-zinc-500/20 text-zinc-700 dark:text-zinc-400 border-zinc-500/30",
  },
  STABILITY_TIERS: { fact: "standard", decision: "standard" },
  getPendingRevision: vi.fn().mockResolvedValue(null),
  acceptPendingRevision: vi.fn(),
  dismissPendingRevision: vi.fn(),
  agentDisplayName: (slug: string | null) => slug,
}));

import MemoryCard from "./MemoryCard";
import { getPendingRevision } from "../../lib/tauri";

const mockGetPendingRevision = vi.mocked(getPendingRevision);

function makeMemory(overrides: Partial<MemoryItem> = {}): MemoryItem {
  return {
    source_id: "mem-old",
    title: "Ship the beta on Friday",
    content: "Ship the beta on Friday",
    summary: null,
    memory_type: "fact",
    domain: null,
    source_agent: null,
    confidence: null,
    confirmed: false,
    pinned: false,
    supersedes: null,
    last_modified: 0,
    chunk_count: 1,
    access_count: 0,
    is_recap: false,
    ...overrides,
  };
}

const defaultProps = {
  onConfirm: vi.fn(),
  onDelete: vi.fn(),
  expandedChain: false,
  onToggleChain: vi.fn(),
  versionChain: [],
};

describe("archive-superseded memories stay visible but muted", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPendingRevision.mockResolvedValue(null);
  });

  it("fades and labels the grid card", () => {
    const { container } = render(
      <MemoryCard memory={makeMemory({ is_archived: true })} {...defaultProps} />,
    );

    expect(screen.getByText("Ship the beta on Friday")).toBeInTheDocument();
    expect(screen.getByText("archived")).toBeInTheDocument();
    expect(container.firstElementChild).toHaveStyle({ opacity: "0.55" });
  });

  it("leaves an unarchived grid card alone", () => {
    const { container } = render(
      <MemoryCard memory={makeMemory({ is_archived: false })} {...defaultProps} />,
    );

    expect(screen.queryByText("archived")).not.toBeInTheDocument();
    expect(container.firstElementChild).not.toHaveStyle({ opacity: "0.55" });
  });

  it("fades and labels the list row, so grid and list agree", () => {
    render(
      <MemoryCard
        memory={makeMemory({ is_archived: true })}
        {...defaultProps}
        presentation="parent-list"
      />,
    );

    const row = screen.getByRole("article", { name: /Ship the beta on Friday/i });
    expect(row).toHaveStyle({ opacity: "0.55" });
    expect(screen.getByText("archived")).toBeInTheDocument();
  });

  it("leaves an unarchived list row alone", () => {
    render(
      <MemoryCard
        memory={makeMemory({ is_archived: false })}
        {...defaultProps}
        presentation="parent-list"
      />,
    );

    const row = screen.getByRole("article", { name: /Ship the beta on Friday/i });
    expect(row).not.toHaveStyle({ opacity: "0.55" });
    expect(screen.queryByText("archived")).not.toBeInTheDocument();
  });
});
