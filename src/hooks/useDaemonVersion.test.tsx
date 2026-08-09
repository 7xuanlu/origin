// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const mocks = vi.hoisted(() => ({ getDaemonVersion: vi.fn() }));
vi.mock("../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/tauri")>();
  return { ...actual, getDaemonVersion: mocks.getDaemonVersion };
});

import { useDaemonVersion } from "./useDaemonVersion";

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("useDaemonVersion", () => {
  // NOTE: reset runs in afterEach, not beforeEach. Resetting the same vi.fn()
  // instance in beforeEach — immediately before it is reconfigured with
  // mockRejectedValue and consumed by a react-query queryFn — deterministically
  // triggers a false-positive "unhandled rejection" failure in this project's
  // pinned vitest 4.1.5 + @tanstack/react-query 5.100.9 + React 19 combination,
  // independent of retry setting, wait strategy, or queryFn wrapping (isolated
  // via ~12 minimal repros). Resetting after each test instead avoids the race
  // while preserving the same test-isolation guarantee.
  afterEach(() => mocks.getDaemonVersion.mockReset());

  it("reports 0.13+ as supporting external key and hot-swap", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(result.current.version).toBe("0.13.0"));
    expect(result.current.supportsExternalKey).toBe(true);
    expect(result.current.supportsHotSwap).toBe(true);
  });

  it("reports 0.12 as not supporting either", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(result.current.version).toBe("0.12.0"));
    expect(result.current.supportsExternalKey).toBe(false);
    expect(result.current.supportsHotSwap).toBe(false);
  });

  it("is conservative when the health fetch fails", async () => {
    mocks.getDaemonVersion.mockRejectedValue(new Error("daemon down"));
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(mocks.getDaemonVersion).toHaveBeenCalled());
    expect(result.current.version).toBeNull();
    expect(result.current.supportsExternalKey).toBe(false);
    expect(result.current.supportsHotSwap).toBe(false);
  });
});
