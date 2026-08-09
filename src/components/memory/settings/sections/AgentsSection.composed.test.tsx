// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AgentsSection from "./AgentsSection";

// Unlike AgentsSection.test.tsx, this file does NOT stub out
// RemoteAccessPanel/ClientSetupList — it renders the real composition so the
// no-auth warning consolidation (redesign spec §6/§8) is proven against the
// actual composed Settings screen, not a component that merely intends to be
// the sole renderer.
vi.mock("../../../../lib/tauri", () => ({
  listAgents: vi.fn().mockResolvedValue([]),
  updateAgent: vi.fn().mockResolvedValue(null),
  deleteAgent: vi.fn().mockResolvedValue(null),
  detectMcpClients: vi.fn().mockResolvedValue([]),
  writeMcpConfig: vi.fn().mockResolvedValue(undefined),
  installClientPlugin: vi.fn().mockResolvedValue(undefined),
  clipboardWrite: vi.fn().mockResolvedValue(undefined),
  getRemoteAccessStatus: vi.fn().mockResolvedValue({
    status: "connected",
    tunnel_url: "https://x.trycloudflare.com",
    token: "t",
    relay_url: null,
  }),
  getWireState: vi.fn().mockResolvedValue({
    daemon: { base_url: "", reachable: true, version: null, error: null },
    mcp_binary: { command: "", args: [], candidates: [] },
    clients: [],
  }),
  testRemoteMcpConnection: vi.fn().mockResolvedValue({ ok: true }),
  toggleRemoteAccess: vi.fn().mockResolvedValue(undefined),
}));

function renderAgentsSection() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AgentsSection />
    </QueryClientProvider>,
  );
}

describe("AgentsSection composed — no-auth warning single-source proof", () => {
  it("the no-auth warning renders exactly once across the whole composed Settings → Agents screen", async () => {
    renderAgentsSection();

    // Remote Access section has loaded (status: connected) — the warning is
    // unconditional once RemoteAccessPanel has rendered.
    await screen.findByText("Claude.ai");

    expect(screen.getAllByText(/no authentication for Claude\.ai and ChatGPT/)).toHaveLength(1);
  });
});
