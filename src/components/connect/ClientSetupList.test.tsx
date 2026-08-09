// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";
import { resources } from "../../i18n/resources";

const mocks = vi.hoisted(() => ({
  detectMcpClients: vi.fn(),
  writeMcpConfig: vi.fn(),
  installClientPlugin: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import ClientSetupList from "./ClientSetupList";

const CLIENTS = [
  { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: true, already_configured: false },
  { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: true, already_configured: false },
  { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: true, already_configured: false },
  { name: "Claude Desktop", client_type: "claude_desktop", config_path: "~/Library/.../config.json", detected: true, already_configured: false },
  { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: true, already_configured: false },
];

function renderList(qc: QueryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })) {
  return render(
    <QueryClientProvider client={qc}>
      <ClientSetupList />
    </QueryClientProvider>,
  );
}

/** Finds the row shell (ClientRow's `rounded-xl` wrapper div) for a given
 *  client name, so assertions can be scoped `within` a single card instead
 *  of matching anywhere in the document. */
function rowFor(name: string) {
  return screen.getByText(name).closest("div.rounded-xl") as HTMLElement;
}

async function clickSetUp(name: string) {
  await userEvent.click(within(rowFor(name)).getByRole("button", { name: "Set up" }));
}

describe("ClientSetupList — one Set up button, two different jobs behind it", () => {
  afterEach(() => Object.values(mocks).forEach((m) => m.mockReset()));
  beforeEach(() => {
    mocks.detectMcpClients.mockResolvedValue(CLIENTS);
    mocks.writeMcpConfig.mockResolvedValue(undefined);
    mocks.installClientPlugin.mockResolvedValue(undefined);
  });

  it("every detected client gets the same one-click Set up — no slash commands, no copy-a-prompt", async () => {
    renderList();
    const setUps = await screen.findAllByRole("button", { name: "Set up" });
    expect(setUps).toHaveLength(CLIENTS.length);

    expect(screen.queryByRole("button", { name: "Copy setup prompt" })).not.toBeInTheDocument();
    expect(screen.queryByText("Show terminal commands")).not.toBeInTheDocument();
    expect(screen.queryByText("Advanced")).not.toBeInTheDocument();
    expect(screen.queryByText(/plugin marketplace add/)).not.toBeInTheDocument();
  });

  // The invariant, from the Settings side. Claude Code's and Codex's Wenlan
  // plugins declare their own `mcpServers`, so writing an MCP config on top of
  // installing the plugin would register Wenlan twice.
  it("claude_code installs the plugin and never writes an MCP config", async () => {
    renderList();
    await screen.findByText("Claude Code");
    await clickSetUp("Claude Code");

    expect(mocks.installClientPlugin).toHaveBeenCalledWith("claude_code");
    expect(mocks.writeMcpConfig).not.toHaveBeenCalled();
  });

  it("codex_cli installs the plugin and never writes an MCP config", async () => {
    renderList();
    await screen.findByText("Codex CLI");
    await clickSetUp("Codex CLI");

    expect(mocks.installClientPlugin).toHaveBeenCalledWith("codex_cli");
    expect(mocks.writeMcpConfig).not.toHaveBeenCalled();
  });

  it("a non-plugin client still takes the config-write path", async () => {
    renderList();
    await screen.findByText("Cursor");
    await clickSetUp("Cursor");

    expect(mocks.writeMcpConfig).toHaveBeenCalledWith("cursor");
    expect(mocks.installClientPlugin).not.toHaveBeenCalled();
  });

  it("shipped copy never references .mcpb or .codex-plugin — the DOM and every locale", async () => {
    const { container } = renderList();
    await screen.findByText("Claude Code");
    expect(container.textContent).not.toContain(".mcpb");
    expect(container.textContent).not.toContain(".codex-plugin");

    // en is only one of three shipped locales — scan every connectMatrix
    // string in every locale, not just what happened to render in en.
    for (const [locale, bundle] of Object.entries(resources)) {
      const connectMatrix = (bundle.translation as Record<string, unknown>).connectMatrix as Record<
        string,
        string
      >;
      for (const [key, value] of Object.entries(connectMatrix)) {
        expect(value, `${locale}.connectMatrix.${key}`).not.toContain(".mcpb");
        expect(value, `${locale}.connectMatrix.${key}`).not.toContain(".codex-plugin");
      }
    }
  });

  it("undetected clients show Not installed, not a Set up button", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: false, already_configured: false },
      { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: false, already_configured: false },
    ]);
    renderList();
    for (const name of ["Claude Code", "Codex CLI"]) {
      await screen.findByText(name);
      const row = rowFor(name);
      expect(within(row).getByText("Not installed")).toBeInTheDocument();
      expect(within(row).queryByRole("button", { name: "Set up" })).not.toBeInTheDocument();
    }
  });

  it("a failed Set up shows the error in the danger-text token, not a raw Tailwind color", async () => {
    mocks.writeMcpConfig.mockRejectedValue(new Error("permission denied"));
    renderList();
    await screen.findByText("Cursor");
    await clickSetUp("Cursor");

    const errorEl = await screen.findByRole("alert");
    expect(errorEl).toHaveTextContent(/permission denied/);
    expect(errorEl).toHaveStyle({ color: "var(--mem-status-danger-text)" });
    expect(errorEl.className).not.toContain("text-red-500");
  });

  it("a failed plugin install surfaces its reason too — the CLI-not-found case", async () => {
    mocks.installClientPlugin.mockRejectedValue(new Error("Codex CLI not found"));
    renderList();
    await screen.findByText("Codex CLI");
    await clickSetUp("Codex CLI");

    expect(await screen.findByRole("alert")).toHaveTextContent(/Codex CLI not found/);
  });

  // Configured clients are already shown in the Connected group above —
  // repeating them here (with nothing left to do) was the duplication the
  // user vetoed. Mixed detected list proves the filter, not just the empty case.
  it("hides already-configured clients — only clients with something left to do render", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: true, already_configured: true },
      { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: true, already_configured: false },
    ]);
    renderList();

    await screen.findByText("Cursor");
    expect(screen.queryByText("Claude Code")).not.toBeInTheDocument();
  });

  it("shows an all-connected note when every detected client is already configured", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: true, already_configured: true },
    ]);
    renderList();

    expect(await screen.findByText("Every detected tool is already connected")).toBeInTheDocument();
  });

  // Coherence pass: the roster above is the single source of truth for what
  // is connected. A client whose tool family already has an identity there is
  // hidden here even when its OWN config file reads unconfigured (the live
  // Cursor identities did not carry `already_configured`). Family, not the
  // per-file flag, decides.
  it("hides a detected-but-unconfigured client whose family is already connected, still shows others", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: true, already_configured: false },
      { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: true, already_configured: false },
    ]);
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <ClientSetupList connectedFamilies={new Set(["cursor"])} />
      </QueryClientProvider>,
    );

    // Gemini's family is not connected → it still offers a Set up button.
    await screen.findByText("Gemini CLI");
    expect(screen.queryByText("Cursor")).not.toBeInTheDocument();
  });
});
