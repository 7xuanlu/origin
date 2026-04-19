// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { MilestoneToaster } from "../MilestoneToaster";
import type { MilestoneRecord } from "../../../lib/tauri";

const recentRecall: MilestoneRecord = {
  id: "first-recall",
  first_triggered_at: Math.floor(Date.now() / 1000) - 10,
  acknowledged_at: null,
  payload: { agent: "claude" },
};

const oldRecall: MilestoneRecord = {
  ...recentRecall,
  first_triggered_at: Math.floor(Date.now() / 1000) - 60 * 60 * 48, // 48h old
};

vi.mock("../../../lib/tauri", () => ({
  listOnboardingMilestones: vi.fn(),
  acknowledgeOnboardingMilestone: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

const { listOnboardingMilestones } = await import("../../../lib/tauri");
const mockList = vi.mocked(listOnboardingMilestones);

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("MilestoneToaster", () => {
  it("renders a toast for a recent unacknowledged toast-channel milestone", async () => {
    mockList.mockResolvedValueOnce([recentRecall]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/recalled for you/i)).toBeInTheDocument(),
    );
  });

  it("does not render a toast for milestones older than 24h", async () => {
    mockList.mockResolvedValueOnce([oldRecall]);
    const { container } = render(<MilestoneToaster />, { wrapper });
    await waitFor(() => expect(mockList).toHaveBeenCalled());
    expect(container.textContent).not.toMatch(/recalled for you/i);
  });

  it("renders a toast for first-memory", async () => {
    const firstMemory: MilestoneRecord = {
      id: "first-memory",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: null,
    };
    mockList.mockResolvedValueOnce([firstMemory]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/first memory saved/i)).toBeInTheDocument(),
    );
  });

  it("renders quoted preview + source attribution for first-memory with payload", async () => {
    const firstMemory: MilestoneRecord = {
      id: "first-memory",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: {
        memory_id: "mem_abc",
        source: "claude",
        preview: "I prefer Rust for CLI tools because of compile-time safety.",
      },
    };
    mockList.mockResolvedValueOnce([firstMemory]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(
        screen.getByText(/I prefer Rust for CLI tools/i),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText(/— claude/i)).toBeInTheDocument();
  });

  it("omits source attribution when first-memory source is empty", async () => {
    const firstMemory: MilestoneRecord = {
      id: "first-memory",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: {
        memory_id: "mem_abc",
        source: "", // empty string — should be treated as missing
        preview: "Fresh note from the daemon.",
      },
    };
    mockList.mockResolvedValueOnce([firstMemory]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/Fresh note from the daemon/i)).toBeInTheDocument(),
    );
    // No "— <source>" attribution line should appear.
    expect(document.body.textContent).not.toMatch(/— \b/);
  });

  it("renders agent subtitle for second-agent", async () => {
    const secondAgent: MilestoneRecord = {
      id: "second-agent",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: { agent: "cursor" },
    };
    mockList.mockResolvedValueOnce([secondAgent]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/A second AI/i)).toBeInTheDocument(),
    );
    expect(
      screen.getByText(/cursor joined.*memories follow you across tools/i),
    ).toBeInTheDocument();
  });

  it("renders static subtitle for intelligence-ready", async () => {
    const ready: MilestoneRecord = {
      id: "intelligence-ready",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: null,
    };
    mockList.mockResolvedValueOnce([ready]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/on-device intelligence/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/run locally/i)).toBeInTheDocument();
  });

  it("renders rephrased first-recall copy", async () => {
    const recall: MilestoneRecord = {
      id: "first-recall",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: { agent: "claude" },
    };
    mockList.mockResolvedValueOnce([recall]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(
        screen.getByText(/origin just recalled for you/i),
      ).toBeInTheDocument(),
    );
    expect(screen.getByText(/called by claude/i)).toBeInTheDocument();
  });

  it("quotes the recalled excerpt when first-recall payload has preview", async () => {
    const recall: MilestoneRecord = {
      id: "first-recall",
      first_triggered_at: Math.floor(Date.now() / 1000) - 5,
      acknowledged_at: null,
      payload: {
        agent: "claude-code",
        preview: "Origin uses Cloudflare quick tunnels.",
      },
    };
    mockList.mockResolvedValueOnce([recall]);
    render(<MilestoneToaster />, { wrapper });
    await waitFor(() =>
      expect(screen.getByText(/Cloudflare quick tunnels/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/— claude-code/i)).toBeInTheDocument();
    // Plain "Called by ..." fallback should NOT render when preview is present.
    expect(document.body.textContent).not.toMatch(/called by/i);
  });
});
