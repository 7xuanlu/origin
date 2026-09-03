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
import { NO, YES, unreadable } from "../../test/readings";

const CLIENTS = [
  { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
  { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
  { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
  { name: "Claude Desktop", client_type: "claude_desktop", config_path: "~/Library/.../config.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
  { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
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
    mocks.writeMcpConfig.mockResolvedValue([]);
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
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: NO, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
      { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: NO, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
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
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: YES, already_configured: YES, has_raw_entry: YES, has_raw_duplicate: NO, has_plugin: NO },
      { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
    ]);
    renderList();

    await screen.findByText("Cursor");
    expect(screen.queryByText("Claude Code")).not.toBeInTheDocument();
  });

  it("shows an all-connected note when every detected client is already configured", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: YES, already_configured: YES, has_raw_entry: YES, has_raw_duplicate: NO, has_plugin: NO },
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
      { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
      { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: YES, already_configured: NO, has_raw_entry: NO, has_raw_duplicate: NO, has_plugin: NO },
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
  // ── Round 5, defect 4: a look that failed is not a tool that is absent ──

  // `detect_mcp_clients` used to answer this with a bare `false`, so a config
  // the OS would not hand over and a config that genuinely has no Wenlan entry
  // arrived here as the same value. This row is what that produced.
  it("a detection that could not be made says so - it never says Not installed", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "~/.cursor/mcp.json",
        detected: unreadable("Access is denied. (os error 5)"),
        already_configured: NO,
        has_raw_entry: NO,
        has_raw_duplicate: NO,
        has_plugin: NO,
      },
    ]);
    renderList();

    await screen.findByText("Cursor");
    const row = rowFor("Cursor");
    expect(within(row).queryByText("Not installed")).not.toBeInTheDocument();
    expect(within(row).getByText("Could not check")).toBeInTheDocument();
    // The reason reaches the user, not just the fact that something failed.
    expect(within(row).getByText(/Access is denied/)).toBeInTheDocument();
    // And the action survives: withholding Set up would be the same false
    // negative, expressed as a missing button instead of a wrong label.
    expect(within(row).getByRole("button", { name: "Set up" })).toBeInTheDocument();
  });

  // The hide filter was `!client.already_configured`, which is FALSY-negated to
  // true for a read that failed - so an unreadable config was hidden from the
  // one list whose job is showing what still needs doing, under the heading
  // "Every detected tool is already connected".
  it("a configured-state that could not be read is still listed, marked unknown", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "~/.claude.json",
        detected: YES,
        already_configured: unreadable("Access is denied. (os error 5)"),
        has_raw_entry: NO,
        has_raw_duplicate: NO,
        has_plugin: NO,
      },
    ]);
    renderList();

    await screen.findByText("Claude Code");
    expect(screen.queryByText("Every detected tool is already connected")).not.toBeInTheDocument();
    const row = rowFor("Claude Code");
    // Regex: the chip appends the reason after the label, and the reason is
    // the half a boolean could never carry.
    expect(within(row).getByText(/Setup state unknown/)).toBeInTheDocument();
    expect(within(row).getByText(/Access is denied/)).toBeInTheDocument();
    // Not the green "Configured" chip: nothing was read that says so.
    expect(within(row).queryByText("Configured")).not.toBeInTheDocument();
  });

  // ── Round 6, D6a ────────────────────────────────────────────────────

  /** A Claude Desktop whose chat-side plugin scan FAILED. `claude_desktop` is
   *  the client this matters for: it has a plugin surface AND its "Set up"
   *  writes a raw `mcpServers` entry, so an unread plugin state plus a write
   *  is the plugin+raw double registration Diagnostics has a warning box for.
   */
  const desktopWithUnreadPlugin = [
    {
      name: "Claude Desktop",
      client_type: "claude_desktop",
      config_path: "~/Library/.../claude_desktop_config.json",
      detected: YES,
      // The config file itself read fine and holds no raw entry…
      has_raw_entry: NO,
      has_raw_duplicate: NO,
      // …but the plugin half could not be read, so the OR is unknown.
      has_plugin: unreadable("Access is denied. (os error 5)"),
      already_configured: unreadable("Access is denied. (os error 5)"),
    },
  ];

  it("a write that could duplicate a registration is offered as a decision, not as the default action", async () => {
    mocks.detectMcpClients.mockResolvedValue(desktopWithUnreadPlugin);
    renderList();

    await screen.findByText("Claude Desktop");
    const row = rowFor("Claude Desktop");

    // SEES: not the same button a measured `no` gets.
    expect(within(row).queryByRole("button", { name: "Set up" })).not.toBeInTheDocument();
    const button = within(row).getByRole("button", { name: "Set up anyway" });

    // SEES: what could not be read, and what writing anyway would do.
    expect(
      within(row).getByText(/could not check whether this tool is already connected/i),
    ).toBeInTheDocument();
    expect(within(row).getByText(/second registration/i)).toBeInTheDocument();
    // The OS's own words, not just "something went wrong". `getAllByText`:
    // the unknown-state chip carries the same reason, and both should.
    expect(within(row).getAllByText(/Access is denied/).length).toBeGreaterThan(0);

    // ANNOUNCES: the button carries that line, so the risk is read out with
    // the action rather than sitting several nodes away from it.
    const describedBy = button.getAttribute("aria-describedby");
    expect(describedBy).toBe("client-row-desc-claude_desktop");
    expect(document.getElementById(describedBy!)).toHaveTextContent(/second registration/i);

    // CLICKS: still possible. Withholding the action would be the same false
    // negative wearing a different coat — the user may well know the plugin
    // is not installed.
    await userEvent.click(button);
    expect(mocks.writeMcpConfig).toHaveBeenCalledWith("claude_desktop");
  });

  it("a MEASURED plugin absence keeps the ordinary unqualified Set up", async () => {
    mocks.detectMcpClients.mockResolvedValue([
      { ...desktopWithUnreadPlugin[0], has_plugin: NO, already_configured: NO },
    ]);
    renderList();

    await screen.findByText("Claude Desktop");
    const row = rowFor("Claude Desktop");
    expect(within(row).getByRole("button", { name: "Set up" })).toBeInTheDocument();
    expect(within(row).queryByText(/second registration/i)).not.toBeInTheDocument();
  });

  it("a plugin-install client is never qualified — its route writes no raw entry", async () => {
    // `claude_code` goes through installClientPlugin, which installs the
    // plugin rather than adding a second registration beside it, so an unread
    // plugin state cannot produce a duplicate through this button.
    mocks.detectMcpClients.mockResolvedValue([
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "~/.claude.json",
        detected: YES,
        has_raw_entry: NO,
        has_raw_duplicate: NO,
        has_plugin: unreadable("Access is denied. (os error 5)"),
        already_configured: unreadable("Access is denied. (os error 5)"),
      },
    ]);
    renderList();

    await screen.findByText("Claude Code");
    const row = rowFor("Claude Code");
    expect(within(row).getByRole("button", { name: "Set up" })).toBeInTheDocument();
    expect(within(row).queryByText(/second registration/i)).not.toBeInTheDocument();
  });

  // ── Round 6, D3's boundary defect, at the pixel ──────────────────────

  it("a write that skipped an input it never determined says so, and still counts as done", async () => {
    mocks.writeMcpConfig.mockResolvedValue([
      {
        input: "WENLAN_MCP_DEV_BIN",
        blocked: "WENLAN_MCP_DEV_BIN",
        error: "environment variable was not valid Unicode",
      },
    ]);
    renderList();

    await screen.findByText("Cursor");
    await clickSetUp("Cursor");

    const row = rowFor("Cursor");
    expect(within(row).getByText(/WENLAN_MCP_DEV_BIN could not be determined/))
      .toBeInTheDocument();
    expect(within(row).getByText(/not valid Unicode/)).toBeInTheDocument();
    // Not an error: the write SUCCEEDED. It is the completeness of the search
    // behind it that is being reported.
    expect(within(row).queryByRole("alert")).not.toBeInTheDocument();
  });

  it("a write with every input determined reports nothing extra", async () => {
    mocks.writeMcpConfig.mockResolvedValue([]);
    renderList();

    await screen.findByText("Cursor");
    await clickSetUp("Cursor");

    const row = rowFor("Cursor");
    expect(within(row).queryByText(/could not be determined/)).not.toBeInTheDocument();
  });
});
