// SPDX-License-Identifier: AGPL-3.0-only
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../../../i18n";
import type { AgentConnection } from "../../../../lib/tauri";

const mocks = vi.hoisted(() => ({
  listAgents: vi.fn(),
  updateAgent: vi.fn(),
  deleteAgent: vi.fn(),
  detectMcpClients: vi.fn(),
}));
vi.mock("../../../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../lib/tauri")>();
  return { ...actual, ...mocks };
});

// The web/remote surfaces are proven in their own files; here we isolate the
// roster (grouping, disclosure, aggregate controls).
vi.mock("../../RemoteAccessPanel", () => ({ RemoteAccessPanel: () => <div /> }));
vi.mock("../../../connect/ClientSetupList", () => ({ default: () => <div /> }));

import AgentsSection from "./AgentsSection";

function agent(name: string, agent_type: string, overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: name,
    name,
    display_name: null,
    agent_type,
    description: null,
    enabled: true,
    trust_level: "full",
    last_seen_at: null,
    memory_count: 0,
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

function renderAgentsSection() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AgentsSection />
    </QueryClientProvider>,
  );
}

describe("AgentsSection", () => {
  afterEach(() => Object.values(mocks).forEach((m) => m.mockReset()));
  beforeEach(() => {
    mocks.listAgents.mockResolvedValue([agent("codex", "mcp", { memory_count: 3 })]);
    mocks.updateAgent.mockResolvedValue(null);
    mocks.deleteAgent.mockResolvedValue(null);
    mocks.detectMcpClients.mockResolvedValue([]);
  });

  // Every registered identity of one physical tool folds into a single row;
  // the identities live behind the disclosure. Codex ships three canonical
  // IDs that must coalesce into one "Codex" row.
  it("coalesces a tool's identities into one row, revealing them on disclosure", async () => {
    mocks.listAgents.mockResolvedValue([
      agent("codex", "mcp", { memory_count: 3, last_seen_at: 100 }),
      agent("codex-mcp-client", "", { memory_count: 2, last_seen_at: 300 }),
      agent("codex-ulw-loop", "", { memory_count: 1, last_seen_at: 200 }),
    ]);
    renderAgentsSection();

    // Exactly one family row — the display name renders once, with a
    // "3 identities" chip and the aggregate memory count (3 + 2 + 1).
    expect(await screen.findByText("Codex")).toBeInTheDocument();
    expect(screen.getAllByText("Codex")).toHaveLength(1);
    expect(screen.getByText("3 identities")).toBeInTheDocument();
    expect(screen.getByText("6 memories")).toBeInTheDocument();

    // Collapsed: the per-identity controls are not mounted.
    expect(screen.queryByRole("button", { name: "codex-ulw-loop" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Show identities" }));

    // Expanded: one identity subrow per canonical ID (found by its own
    // enable Toggle's aria-label).
    for (const id of ["codex", "codex-mcp-client", "codex-ulw-loop"]) {
      expect(screen.getByRole("button", { name: id })).toBeInTheDocument();
    }
  });

  // The aggregate trust Select shows "Mixed" when identities disagree, and
  // picking a real value writes it to every identity.
  it("renders Mixed for divergent trust and applies a pick to every identity", async () => {
    mocks.listAgents.mockResolvedValue([
      agent("codex", "mcp", { trust_level: "full" }),
      agent("codex-ulw-loop", "", { trust_level: "review" }),
    ]);
    renderAgentsSection();

    await screen.findByText("Codex");
    // The disabled "Mixed" option is the selected placeholder.
    expect(screen.getByText("Mixed")).toBeInTheDocument();

    // Only the aggregate Select is mounted (disclosure collapsed).
    await userEvent.selectOptions(screen.getByRole("combobox"), "full");

    expect(mocks.updateAgent).toHaveBeenCalledTimes(2);
    expect(mocks.updateAgent).toHaveBeenCalledWith("codex", { trustLevel: "full" });
    expect(mocks.updateAgent).toHaveBeenCalledWith("codex-ulw-loop", { trustLevel: "full" });
  });

  // Primitives migration: the delete flow is a two-step ConfirmActionButton,
  // now living inside each identity's disclosure subrow.
  it("requires a second click before deleteAgent fires", async () => {
    const user = userEvent.setup();
    renderAgentsSection();

    await user.click(await screen.findByRole("button", { name: "Show identities" }));
    await user.click(screen.getByRole("button", { name: "Delete Codex" }));
    expect(mocks.deleteAgent).not.toHaveBeenCalled();

    await user.click(await screen.findByText("Confirm"));
    expect(mocks.deleteAgent).toHaveBeenCalled();
    expect(mocks.deleteAgent.mock.calls[0][0]).toBe("codex");
  });

  // S3: the trust legend renders each level as a `Tag` (tone="neutral", no
  // accent border) — `Tag`'s signature `rounded-full` class proves the swap.
  it("renders each trust level in the legend as a Tag, not the old accent badge", async () => {
    renderAgentsSection();

    await screen.findByText("Codex");
    const fullTags = Array.from(document.querySelectorAll<HTMLElement>(".rounded-full")).filter(
      (el) => el.textContent === "Full",
    );
    expect(fullTags).toHaveLength(1);
    expect(fullTags[0].style.border).toBe("");
  });

  // A configured client whose family already has a connected identity folds
  // in as a restart note on that family's row — not a separate pending row.
  it("shows a pending client's restart note on the family row when the family is already connected", async () => {
    mocks.listAgents.mockResolvedValue([agent("codex", "mcp", { memory_count: 3 })]);
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: true, already_configured: true },
    ]);
    renderAgentsSection();

    const codexRow = (await screen.findByText("Codex")).closest("div.px-5") as HTMLElement;
    expect(within(codexRow).getByText("Restart Codex to activate")).toBeInTheDocument();
  });
});
