// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useMilestones } from "../useMilestones";

vi.mock("../../../lib/tauri", () => ({
  listOnboardingMilestones: vi.fn().mockResolvedValue([]),
  acknowledgeOnboardingMilestone: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("useMilestones", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns empty array on fresh install", async () => {
    const { result } = renderHook(() => useMilestones(), { wrapper });
    await waitFor(() => expect(result.current.milestones).toEqual([]));
  });

  it("exposes an acknowledge mutation", async () => {
    const { result } = renderHook(() => useMilestones(), { wrapper });
    await waitFor(() => expect(result.current.milestones).toBeDefined());
    expect(typeof result.current.acknowledge).toBe("function");
  });
});
