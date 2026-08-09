import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within, fireEvent, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SetupWizard, displayedStatuses } from "./SetupWizard";
import { i18n } from "../i18n";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

vi.mock("../lib/tauri", () => ({
  importMemories: vi.fn(),
  shouldShowWizard: vi.fn().mockResolvedValue(true),
  setSetupCompleted: vi.fn().mockResolvedValue(undefined),
  detectMcpClients: vi.fn().mockResolvedValue([]),
  writeMcpConfig: vi.fn().mockResolvedValue(undefined),
  installClientPlugin: vi.fn().mockResolvedValue(undefined),
  listAgents: vi.fn().mockResolvedValue([]),
  getWenlanMcpEntry: vi.fn().mockResolvedValue({ command: "npx", args: ["-y", "wenlan-mcp"] }),
  getRemoteAccessStatus: vi.fn().mockResolvedValue({ status: "off" }),
  toggleRemoteAccess: vi.fn().mockResolvedValue({ status: "off" }),
  rotateRemoteToken: vi.fn().mockResolvedValue("new-token"),
  testRemoteMcpConnection: vi.fn().mockResolvedValue({ ok: true, latency_ms: 42, error: null }),
  clipboardWrite: vi.fn().mockResolvedValue(undefined),
  getApiKey: vi.fn().mockResolvedValue(null),
  setApiKey: vi.fn().mockResolvedValue(undefined),
  getModelChoice: vi.fn().mockResolvedValue([null, null]),
  setModelChoice: vi.fn().mockResolvedValue(undefined),
  getOnDeviceModel: vi.fn().mockResolvedValue({
    loaded: null,
    selected: "qwen3-4b-instruct-2507",
    models: [{
      id: "qwen3-4b-instruct-2507",
      display_name: "Qwen3 4B",
      param_count: "4B",
      ram_required_gb: 8,
      file_size_gb: 2.7,
      cached: false,
    }],
  }),
  downloadOnDeviceModel: vi.fn().mockResolvedValue(undefined),
  onDeviceModelDownloadBytes: vi.fn().mockResolvedValue(null),
  getSystemInfo: vi.fn().mockResolvedValue({
    total_ram_gb: 16,
    available_ram_gb: 10,
    has_metal: true,
    has_cuda: false,
    os: "macOS",
    arch: "arm64",
    recommended_builtin: "qwen3-4b-instruct-2507",
  }),
  getDaemonVersion: vi.fn().mockResolvedValue("0.12.0"),
  daemonMeetsFloor: vi.fn().mockReturnValue(false),
  getExternalLlm: vi.fn().mockResolvedValue([null, null]),
  setExternalLlm: vi.fn().mockResolvedValue(undefined),
  testExternalLlm: vi.fn().mockResolvedValue({ response: "pong" }),
  listExternalModels: vi.fn().mockResolvedValue([]),
  getExternalLlmKeyConfigured: vi.fn().mockResolvedValue(false),
  getResolvedRouting: vi.fn().mockResolvedValue(null),
  setSourcePin: vi.fn().mockResolvedValue(undefined),
  detectObsidianVaults: vi.fn().mockResolvedValue([]),
  addSource: vi.fn().mockResolvedValue({
    id: "src-1",
    source_type: "obsidian",
    path: "/Users/x/Vaults/Work",
    status: "idle",
    last_sync: null,
    file_count: 0,
    memory_count: 0,
  }),
  syncRegisteredSource: vi.fn().mockResolvedValue({
    files_found: 0,
    ingested: 0,
    skipped: 0,
    errors: 0,
  }),
  getWireState: vi.fn().mockResolvedValue({
    daemon: { base_url: "http://127.0.0.1:7878", reachable: true, version: "0.12.0", error: null },
    mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
    clients: [],
  }),
  storeMemory: vi.fn().mockResolvedValue({ source_id: "mem_setup-check" }),
  listRecentMemories: vi.fn().mockResolvedValue([
    {
      kind: "memory",
      id: "mem_setup-check",
      title: "Wenlan setup check",
      snippet: null,
      timestamp_ms: Date.now(),
      badge: { kind: "none" },
    },
  ]),
  // Echoes the id back in a MemoryItem-shaped object so it always matches
  // whatever storeMemory resolved with, by default — tests that need a
  // mismatch (or a missing memory) override this per-test.
  getMemoryDetail: vi.fn().mockImplementation((sourceId: string) =>
    Promise.resolve({
      source_id: sourceId,
      title: "Wenlan setup check",
      content: "Wenlan setup check: confirms this device can store and recall a memory.",
      summary: null,
      memory_type: null,
      domain: null,
      source_agent: "wenlan-setup",
      confidence: null,
      confirmed: false,
      pinned: false,
      supersedes: null,
      last_modified: Date.now(),
      chunk_count: 0,
    }),
  ),
  deleteMemory: vi.fn().mockResolvedValue(undefined),
}));

import {
  detectMcpClients,
  writeMcpConfig,
  installClientPlugin,
  listAgents,
  setApiKey,
  getWireState,
  getOnDeviceModel,
  downloadOnDeviceModel,
  onDeviceModelDownloadBytes,
  getResolvedRouting,
  setSourcePin,
  detectObsidianVaults,
  addSource,
  syncRegisteredSource,
  storeMemory,
  listRecentMemories,
  getMemoryDetail,
  deleteMemory,
} from "../lib/tauri";
import { deriveOnboardingPins, DoneStep } from "./SetupWizard";

/** An agent write that lands AFTER the wizard was entered — the only kind that
 *  proves the config we just wrote actually works. `wizardEnteredAt` is stamped
 *  at mount, so a future timestamp is the reliable way to say "since then". */
const FRESH = () => Math.floor(Date.now() / 1000) + 60;
/** An agent write from a previous install. Proof of nothing about this run. */
const STALE = () => Math.floor(Date.now() / 1000) - 3600;

function renderWizard(
  props: {
    onComplete?: () => void;
    initialStep?: "welcome" | "intelligence-choice" | "import" | "connect" | "setting-up" | "done";
    initialPendingModelId?: string | null;
    initialPendingImportPick?: { sourceType: "obsidian" | "directory"; path: string; label: string } | null;
  } = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  const onComplete = props.onComplete ?? vi.fn();
  return {
    onComplete,
    ...render(
      <QueryClientProvider client={queryClient}>
        <SetupWizard
          onComplete={onComplete}
          initialStep={props.initialStep}
          initialPendingModelId={props.initialPendingModelId}
          initialPendingImportPick={props.initialPendingImportPick}
        />
      </QueryClientProvider>,
    ),
  };
}

describe("SetupWizard", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (installClientPlugin as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (setApiKey as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (getWireState as ReturnType<typeof vi.fn>).mockResolvedValue({
      daemon: { base_url: "http://127.0.0.1:7878", reachable: true, version: "0.12.0", error: null },
      mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
      clients: [],
    });
    (detectObsidianVaults as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (onDeviceModelDownloadBytes as ReturnType<typeof vi.fn>).mockResolvedValue(null);
    // Default to LEGACY (no routing endpoint) so existing done-step tests wire
    // nothing; the onboarding-pin tests override getResolvedRouting per case.
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue(null);
    (setSourcePin as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (storeMemory as ReturnType<typeof vi.fn>).mockResolvedValue({ source_id: "mem_setup-check" });
    (listRecentMemories as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        kind: "memory",
        id: "mem_setup-check",
        title: "Wenlan setup check",
        snippet: null,
        timestamp_ms: Date.now(),
        badge: { kind: "none" },
      },
    ]);
    (getMemoryDetail as ReturnType<typeof vi.fn>).mockImplementation((sourceId: string) =>
      Promise.resolve({
        source_id: sourceId,
        title: "Wenlan setup check",
        content: "Wenlan setup check: confirms this device can store and recall a memory.",
        summary: null,
        memory_type: null,
        domain: null,
        source_agent: "wenlan-setup",
        confidence: null,
        confirmed: false,
        pinned: false,
        supersedes: null,
        last_modified: Date.now(),
        chunk_count: 0,
      }),
    );
    (deleteMemory as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
  });

  it("renders Welcome step by default", () => {
    renderWizard();
    expect(screen.getByText("Welcome to Wenlan")).toBeInTheDocument();
    const tagline = screen.getByText("A living knowledge base your AI tools build as they work.");
    expect(tagline).toBeInTheDocument();
    // Wiki-pages-first: the welcome step must say what Wenlan produces
    // (source-cited pages), not just gesture at "understanding".
    const body = screen.getByText(
      "Your AI tools write what they learn into source-cited pages that refresh between sessions.",
    );
    expect(body).toBeInTheDocument();
    expect(screen.getByText("Your memories live on this machine.")).toBeInTheDocument();
    // R3 typography ladder: on-scale size only, never an off-scale 15px.
    expect(tagline).toHaveStyle({ fontSize: "var(--mem-text-lg)" });
    expect(body).toHaveStyle({ fontSize: "var(--mem-text-lg)" });
  });

  it("renders Welcome step in Simplified Chinese when selected", async () => {
    await i18n.changeLanguage("zh-Hans");
    renderWizard();

    expect(screen.getByText("欢迎使用文澜")).toBeInTheDocument();
    expect(screen.getByText("你的记忆保存在这台设备上。")).toBeInTheDocument();
  });

  it('advances from Welcome to intelligence choice on "Get started" click', () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    expect(screen.getByText("Choose how Wenlan thinks")).toBeInTheDocument();
  });

  it("lets users save an API key from the intelligence step", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Cloud model"));

    fireEvent.change(screen.getByPlaceholderText("sk-ant-api03-..."), {
      target: { value: "sk-ant-test-key" },
    });
    fireEvent.click(screen.getByText("Save"));

    await waitFor(() => {
      expect(setApiKey).toHaveBeenCalledWith("sk-ant-test-key");
    });
  });

  it("intelligence step offers device, cloud, and local server", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));

    // The on-device tile is selected by default, so its pane's <h3> shares
    // the tile's exact text — query at the role level to hit the tiles only.
    expect(screen.getByRole("button", { name: /On-device model/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Cloud model/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Your own local server/ })).toBeInTheDocument();
  });

  it("recommends on-device — the same tile that's selected by default, not cloud", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));

    const deviceButton = screen.getByRole("button", { name: /On-device model/ });
    const cloudButton = screen.getByRole("button", { name: "Cloud model" });

    expect(deviceButton).toHaveAttribute("aria-pressed", "true");
    expect(within(deviceButton).getByText("Recommended")).toBeInTheDocument();
    expect(within(cloudButton).queryByText("Recommended")).not.toBeInTheDocument();
  });

  it("cloud pane offers only the Anthropic key card — no dead cloud-vendor picker (§5.2 honesty fix)", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Cloud model"));

    expect(screen.getByPlaceholderText("sk-ant-api03-...")).toBeInTheDocument();
    // The 7 keyed cloud vendors the daemon can't actually authenticate on
    // this (sub-0.13) daemon must never appear — the pill row exists (it now
    // always includes Anthropic, unified with the other cloud vendors), but
    // on a closed gate Anthropic is the only chip in it.
    expect(screen.queryByText("OpenAI")).not.toBeInTheDocument();
    expect(screen.queryByText("Groq")).not.toBeInTheDocument();
  });

  it("local pane offers the local-server card scoped to keyless presets", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Your own local server"));

    expect(await screen.findByRole("heading", { name: "Your own local server" })).toBeInTheDocument();
    // No key field for any preset in this pane (none of ollama/lmstudio/custom
    // require a key) — the keyed-vendor story is gone from the wizard too.
    expect(screen.queryByLabelText("API key")).not.toBeInTheDocument();
  });

  it("import step offers chat history and vault side by side", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Continue"));

    expect(screen.getByText("Bring what you already know")).toBeInTheDocument();
    expect(screen.getByText("Chat history")).toBeInTheDocument();
    expect(screen.getByText("Import chat history")).toBeInTheDocument();
    // VaultConnectCard (Task 6), rendered inline as the second path. It carries
    // its own title — no separate wrapper heading duplicates it.
    expect(screen.getByText("Connect a notes folder")).toBeInTheDocument();
    expect(screen.queryByText("Obsidian vault / notes folder")).not.toBeInTheDocument();
  });

  it("shows chat-history guidance after choosing the chat path and routes directly to connect", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Continue"));

    fireEvent.click(screen.getByText("Import chat history"));

    expect(screen.getByText("Import Memories")).toBeInTheDocument();
    expect(screen.getByText(/Settings > Sources/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Connect your AI tools")).toBeInTheDocument();
    });
  });

  it("only detected tools render a row — an undetected tool is invisible, not a disabled row", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "/path/to/claude.json",
        detected: false,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    // "Found on this Mac" renders identically in both the loading skeleton
    // and the loaded-data branch, so it can't signal load completion by
    // itself — wait on "Cursor", which only exists once data has loaded.
    await screen.findByText("Cursor");

    expect(screen.getByText("Found on this Mac")).toBeInTheDocument();
    expect(screen.queryByText("Claude Code")).not.toBeInTheDocument();
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
    // The old per-row "Detected"/"Install first" pill duplicated the section
    // header; the redesign has no such pill at all.
    expect(screen.queryByText("Detected")).not.toBeInTheDocument();
    expect(screen.queryByText("Install first")).not.toBeInTheDocument();
  });

  it("no detected tools at all shows the empty-state copy, not an empty section", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "/path/to/claude.json",
        detected: false,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    await waitFor(() => {
      expect(screen.getByText("No AI tools found on this Mac.")).toBeInTheDocument();
    });
    expect(screen.queryByText("Found on this Mac")).not.toBeInTheDocument();
  });

  // ── Dot 4 asks, dot 5 acts ────────────────────────────────────────────
  // Connect writes nothing. Every mutation happens on Setting up, one visible
  // row at a time.

  it("connect writes nothing — Continue only carries the selection to Setting up", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    const cursorCheckbox = await screen.findByRole("checkbox", { name: "Cursor" });
    await waitFor(() => expect(cursorCheckbox).toBeChecked());

    // Nothing has been written while the user was merely looking at the list.
    expect(writeMcpConfig).not.toHaveBeenCalled();
    expect(installClientPlugin).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    // (d) A non-plugin client is set up with writeMcpConfig — on dot 5.
    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("cursor");
    });
    expect(screen.getByText("Setting up")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByTestId("task-status-cursor")).toHaveTextContent("Configured");
    });
    expect(installClientPlugin).not.toHaveBeenCalledWith("cursor");
  });

  it("unchecking a tool drops it from the work list — it gets no row and no write", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
      {
        name: "Windsurf",
        client_type: "windsurf",
        config_path: "/path/to/windsurf.json",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    const windsurfCheckbox = await screen.findByRole("checkbox", { name: "Windsurf" });
    await waitFor(() => expect(windsurfCheckbox).toBeChecked());
    fireEvent.click(windsurfCheckbox);
    expect(windsurfCheckbox).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("cursor");
    });
    expect(writeMcpConfig).not.toHaveBeenCalledWith("windsurf");
    expect(screen.queryByTestId("task-status-windsurf")).not.toBeInTheDocument();
  });

  // (e) A failed row is inert, not fatal. It shows its reason and stays failed;
  // the user is never trapped in setup because one editor's config was
  // read-only.
  it("a failed row shows its error, stays failed, and never blocks Continue", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("permission denied"));

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    // Settle first: wait for the row to actually reach its failed state before
    // asserting anything about what is *not* blocked.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-cursor")).toHaveTextContent("Couldn't set up");
    });

    const error = screen.getByRole("alert");
    expect(error).toHaveTextContent(/permission denied/i);
    expect(error).toHaveStyle({ color: "var(--mem-status-danger-text)" });

    const continueButton = screen.getByRole("button", { name: "Continue" });
    expect(continueButton).toBeEnabled();

    // And it really does move on.
    fireEvent.click(continueButton);
    await waitFor(() => {
      expect(screen.getByText("Wenlan is ready.")).toBeInTheDocument();
    });
  });

  // (b) THE invariant. Claude Code's Wenlan plugin declares its own
  // `mcpServers`, so writing `~/.claude.json` on top of installing the plugin
  // would register the Wenlan server TWICE. Plugin, never a raw MCP entry.
  it("claude_code is set up with installClientPlugin — and its config is never written", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "/path/to/claude.json",
        detected: true,
        already_configured: false,
      },
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    // Claude Code is an ordinary checkbox now — no slash commands, no
    // "go type this in your terminal".
    const claudeCheckbox = await screen.findByRole("checkbox", { name: "Claude Code" });
    await waitFor(() => expect(claudeCheckbox).toBeChecked());

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(installClientPlugin).toHaveBeenCalledWith("claude_code");
    });
    expect(writeMcpConfig).not.toHaveBeenCalledWith("claude_code");

    // The neighbouring GUI client still takes the config-write path — the rule
    // is per-client, not a blanket switch.
    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("cursor");
    });
    expect(installClientPlugin).not.toHaveBeenCalledWith("cursor");

    // The runtime row leads unconditionally, then config rows sort ahead of
    // plugin rows — ascending by how long the row actually takes, so the
    // rail inks top-to-bottom instead of the slowest row sitting at the top.
    const rows = within(screen.getByTestId("setting-up-tasks")).getAllByTestId(/^task-status-/);
    expect(rows.map((el) => el.getAttribute("data-testid"))).toEqual([
      "task-status-daemon",
      "task-status-cursor",
      "task-status-claude_code",
    ]);
  });

  // (c) Same invariant for Codex — and one row covering both surfaces that
  // share ~/.codex/config.toml. It must not read as "ChatGPT connected":
  // ChatGPT's chat assistant only speaks remote HTTPS MCP and is NOT covered.
  it("codex_cli is set up with installClientPlugin, on one row named for both Codex CLI and ChatGPT desktop", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Codex CLI",
        client_type: "codex_cli",
        config_path: "/path/to/config.toml",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Codex CLI" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(installClientPlugin).toHaveBeenCalledWith("codex_cli");
    });
    expect(writeMcpConfig).not.toHaveBeenCalledWith("codex_cli");

    expect(screen.getByText("Codex CLI & ChatGPT desktop")).toBeInTheDocument();
    expect(
      screen.getByText(
        "One setup covers both, including the Codex pane in ChatGPT desktop. ChatGPT's chat assistant is not covered.",
      ),
    ).toBeInTheDocument();
  });

  // Rows are ordered ascending by how long they actually take, so the rail
  // inks top-to-bottom in the common case instead of stalling on a slow row
  // near the top while everything below it has long since finished. Plugin
  // rows (claude_code) clone from GitHub — seconds; config rows (cursor) are
  // a local file write — milliseconds; import is disk + ingest; the model is
  // a multi-GB download, the slowest row on the screen.
  it("rows are ordered daemon, config tools, plugin tools, import, then model — ascending by expected duration", async () => {
    (detectObsidianVaults as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "/path/to/claude.json",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    // Step 2: keep the default on-device model pick so a model row exists.
    await waitFor(() =>
      expect(screen.getByTestId("on-device-model-deferred-note")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText("Continue"));

    // Step 3: pick the vault so an import row exists.
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Work Notes"));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    // Step 4: both detected tools start checked — take both.
    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    const rows = within(screen.getByTestId("setting-up-tasks")).getAllByTestId(/^task-status-/);
    expect(rows.map((el) => el.getAttribute("data-testid"))).toEqual([
      "task-status-daemon",
      "task-status-cursor",
      "task-status-claude_code",
      "task-status-import",
      "task-status-on-device-model",
    ]);
  });

  // (f) The returning-user bug. VerifyStep used to call onNext() from an effect
  // the moment it saw ANY past agent write, so anyone who had ever used Wenlan
  // shot straight past this step. Only a write made SINCE the wizard was
  // entered feeds the Done step's connected-agent chips.
  it("a pre-existing agent write does NOT advance past Setting up, and does not appear as a connected agent", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Claude", display_name: "Claude", last_seen_at: STALE(), memory_count: 5 },
    ]);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    // Settle on a real state change first — otherwise the "did not advance"
    // assertions below could pass simply by running before anything happened.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-cursor")).toHaveTextContent("Configured");
    });

    expect(screen.getByText("Setting up")).toBeInTheDocument();
    expect(screen.queryByText("You're all set.")).not.toBeInTheDocument();
    expect(screen.queryByText("Wenlan is ready.")).not.toBeInTheDocument();

    // An old write proves nothing about the config that was written seconds
    // ago — it must not show up as a connected agent on Done.
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });
    expect(screen.queryByText("Claude")).not.toBeInTheDocument();
  });

  it("renders skip-path Done copy without a back button", async () => {
    renderWizard();

    fireEvent.click(screen.getByText("Get started"));
    // Skip every choosing step: no model, no import, no tools.
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Bring what you already know")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Connect your AI tools")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Skip"));

    // Even with nothing chosen, the runtime row still runs and proves itself —
    // it's unconditional, not gated on any pick.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("Wenlan is ready.")).toBeInTheDocument();
    });

    // readyBody1 must not lead with "Memories will appear..." — wiki pages
    // first, memory second (the memories are kept "behind" the pages).
    expect(screen.queryByText(/^Memories will appear/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "Use your AI tools normally. As they work, Wenlan turns what they learn into source-cited pages — and keeps the memories behind them. You can always return to Settings to connect more tools.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Pages refresh between sessions, so the next one starts where the last ended."),
    ).toBeInTheDocument();

    expect(screen.queryByText("Back")).not.toBeInTheDocument();
  });

  // Redesign spec §4/§12: Remote Access left the wizard entirely — it's a
  // Settings-only surface now, alongside the single consolidated no-auth
  // warning (see RemoteAccessPanel.test.tsx and AgentsSection.test.tsx).
  // The wizard instead points at Settings → Agents for it and every other
  // web tool (assertion lives in the settingsPointer test below).
  it("does not render Remote Access — it lives only in Settings now", async () => {
    renderWizard({ initialStep: "connect" });
    await waitFor(() => {
      expect(screen.getByText("Connect your AI tools")).toBeInTheDocument();
    });
    expect(screen.queryByText(/Share with web-based AI tools/i)).not.toBeInTheDocument();
  });

  it("points at Settings → Agents for Claude.ai, ChatGPT, and anything not listed", async () => {
    renderWizard({ initialStep: "connect" });
    await waitFor(() => {
      expect(
        screen.getByText(
          "Claude.ai, ChatGPT, and more tools can be connected any time in Settings → Agents.",
        ),
      ).toBeInTheDocument();
    });
  });

  // StepShell (§4.0, defect 5): the primary CTA lives in a fixed action bar
  // that is a sibling of the scrollable content column, never a descendant
  // of it — so it is visible by construction regardless of content height,
  // instead of relying on the tallest step happening to fit in 720px.
  it("StepShell: the primary CTA lives outside the scrollable content, never inside it", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    // "local-server variant" — one of the taller intelligence-choice panes.
    fireEvent.click(screen.getByText("Your own local server"));
    await screen.findByRole("heading", { name: "Your own local server" });

    const continueButton = screen.getByText("Continue");
    const scrollMain = screen.getByTestId("wizard-scroll-main");
    const actionBar = screen.getByTestId("wizard-action-bar");

    expect(scrollMain.contains(actionBar)).toBe(false);
    expect(actionBar.contains(continueButton)).toBe(true);
    expect(scrollMain.contains(continueButton)).toBe(false);
  });

  // (a) The bug this redesign fixes: an already-configured client's checkbox
  // used to be `disabled`, so a user could not opt out of a tool Wenlan had
  // already touched. It starts checked, but it stays fully interactive.
  it("an already-configured client starts checked but stays interactive — it can be unchecked, and then it is left alone", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: true,
      },
      {
        name: "Windsurf",
        client_type: "windsurf",
        config_path: "/path/to/windsurf.json",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    const cursorCheckbox = await screen.findByRole("checkbox", { name: "Cursor" });
    await waitFor(() => expect(cursorCheckbox).toBeChecked());

    // Exactly one badge, on Cursor's row — Windsurf isn't configured yet and
    // must not render a second one. The badge is the affordance; `disabled`
    // never is.
    expect(screen.getAllByText("Configured")).toHaveLength(1);
    expect(screen.getByText("Already set up. Uncheck to leave it as it is.")).toBeInTheDocument();
    expect(cursorCheckbox).toBeEnabled();

    // And unchecking actually takes: the box clears, and Setting up gives it
    // no row and no write.
    fireEvent.click(cursorCheckbox);
    expect(cursorCheckbox).not.toBeChecked();

    const windsurfCheckbox = screen.getByRole("checkbox", { name: "Windsurf" });
    await waitFor(() => expect(windsurfCheckbox).toBeChecked());
    expect(windsurfCheckbox).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("windsurf");
    });
    expect(writeMcpConfig).not.toHaveBeenCalledWith("cursor");
    expect(screen.queryByTestId("task-status-cursor")).not.toBeInTheDocument();
  });

  it("done: agent ids that resolve to the same display name collapse to one chip; raw ids never render", async () => {
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "codex-ulw-loop", display_name: "Codex", last_seen_at: FRESH(), memory_count: 1 },
      { name: "codex-mcp-client", display_name: "Codex", last_seen_at: FRESH(), memory_count: 1 },
    ]);

    renderWizard({ initialStep: "setting-up" });

    // Settle on the daemon row finishing first — it's the only visible signal
    // that a render has happened at all before we act on it.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });

    // The fresh-agent notification (listAgents → onConnected) can land either
    // before or after the Continue click — wait for it rather than asserting
    // synchronously right after the Done heading appears.
    await waitFor(() => {
      expect(screen.getByText("Codex")).toBeInTheDocument();
    });
    expect(screen.queryByText("codex-ulw-loop")).not.toBeInTheDocument();
    expect(screen.queryByText("codex-mcp-client")).not.toBeInTheDocument();
  });

  // Regression: ConnectStep used to feed resolveAgentDisplayName the Rust
  // *friendly* name ("Gemini CLI") instead of the canonical client_type
  // slug ("gemini_cli"). Since prettifySlug only splits on `-`/`_`, a
  // friendly name with a space collapses to one token and gets mangled
  // ("Gemini cli"). The write path moved to SettingUpStep, so this now
  // exercises the ConnectStep → SettingUpStep → DoneStep chain.
  it("done: tools set up on dot 5 show correctly-capitalized names, not mangled friendly names", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Claude Desktop",
        client_type: "claude_desktop",
        config_path: "/path/to/claude_desktop_config.json",
        detected: true,
        already_configured: false,
      },
      {
        name: "Gemini CLI",
        client_type: "gemini_cli",
        config_path: "/path/to/gemini/settings.json",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Gemini CLI" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByTestId("task-status-gemini_cli")).toHaveTextContent("Configured");
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });

    expect(screen.getByText("Claude Desktop")).toBeInTheDocument();
    expect(screen.getByText("Gemini CLI")).toBeInTheDocument();
    expect(screen.queryByText("Claude desktop")).not.toBeInTheDocument();
    expect(screen.queryByText("Gemini cli")).not.toBeInTheDocument();
  });

  // Same bug, the other call site: a client that's *already configured*
  // (so onConnected fires from ConnectStep's mount effect, not from a row's
  // completion) must also resolve from client_type.
  it("done: an already-configured tool's name also resolves from client_type, not the friendly Rust name", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Gemini CLI",
        client_type: "gemini_cli",
        config_path: "/path/to/gemini/settings.json",
        detected: true,
        already_configured: true,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    await waitFor(() => {
      expect(screen.getByText("Configured")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("Setting up")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });

    expect(screen.getByText("Gemini CLI")).toBeInTheDocument();
    expect(screen.queryByText("Gemini cli")).not.toBeInTheDocument();
  });

  it("intelligence choice tiles signal selection via aria-pressed, not color alone", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));

    const deviceButton = screen.getByRole("button", { name: /On-device model/ });
    const cloudButton = screen.getByRole("button", { name: "Cloud model" });

    expect(deviceButton).toHaveAttribute("aria-pressed", "true");
    expect(cloudButton).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(cloudButton);

    expect(cloudButton).toHaveAttribute("aria-pressed", "true");
    expect(deviceButton).toHaveAttribute("aria-pressed", "false");
  });

  // Connect is selection-only: no config snippet, no copy-a-config affordance,
  // nothing that writes or hands the user a payload to paste.
  it("connect offers no config snippet — the step asks, it never acts", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    expect(screen.queryByText(/mcpServers/)).not.toBeInTheDocument();
    expect(screen.queryByText("Using another MCP client? Show config")).not.toBeInTheDocument();
  });

  it("done: caps connected-agent chips at 6 with a +N overflow chip", async () => {
    const ids = ["tool-a", "tool-b", "tool-c", "tool-d", "tool-e", "tool-f", "tool-g", "tool-h"];
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue(
      ids.map((name) => ({ name, last_seen_at: FRESH(), memory_count: 1 })),
    );

    renderWizard({ initialStep: "setting-up" });

    // Settle on the daemon row finishing first — it's the only visible signal
    // that a render has happened at all before we act on it.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });

    // Unrecognized slugs are prettified (word-split + title-case), not shown raw.
    // The fresh-agent notification can land either before or after the
    // Continue click, so wait for the first chip rather than asserting
    // synchronously right after the Done heading appears.
    const firstPrettified = "Tool A";
    await waitFor(() => {
      expect(screen.getByText(firstPrettified)).toBeInTheDocument();
    });
    for (const id of ids.slice(0, 6)) {
      const prettified = id
        .split("-")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
      expect(screen.getByText(prettified)).toBeInTheDocument();
    }
    expect(screen.getByText("+2 more")).toBeInTheDocument();
    expect(screen.queryByText("Tool G")).not.toBeInTheDocument();
    expect(screen.queryByText("Tool H")).not.toBeInTheDocument();
  });

  // Fix-back: the same DoneStep renders for the in-app connect-agent re-run
  // (Main.tsx mounts <SetupWizard initialStep="connect"> post-onboarding). Its
  // pin-wiring effect must NOT re-derive and overwrite an existing user's pins
  // — only the full first-onboarding run (no initialStep → wireRouting) wires.
  // Even with a fully configured pool (which would otherwise write), the re-run
  // leaves pins alone.
  it("connect-agent re-run (initialStep=connect) never wires pins and shows no routing summary", async () => {
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue({
      everyday: { source: "basic", model: null, mode: "auto", pin: null },
      synthesis: { source: "none", model: null, mode: "auto", pin: null },
      pool: {
        anthropic: { configured: true, everyday_model: null, synthesis_model: null },
        external: null,
        on_device: { selected: "qwen3-4b-instruct-2507", loaded: true },
      },
    });
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    // A fresh agent drives the done screen to the full "all set" branch.
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "cursor", last_seen_at: FRESH(), memory_count: 1 },
    ]);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await screen.findByText("You're all set.");
    // Flush any effect continuation so a wrongful write/summary would surface.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(setSourcePin).not.toHaveBeenCalled();
    expect(screen.queryByText(/Everyday tasks:/)).not.toBeInTheDocument();
    // Re-run entry (wireRouting=false): neither the summary nor the no-model line.
    expect(
      screen.queryByText(/Your wiki updates whenever your AI tools use Wenlan/),
    ).not.toBeInTheDocument();
  });

  // The full first-onboarding run is the ONLY path that wires, and the render
  // site `wireRouting={!initialStep}` is the load-bearing link the DoneStep-
  // direct tests can't cover (they pass the prop explicitly). Drive
  // welcome→done end-to-end with a configured pool and prove the pins land +
  // the summary renders. Mutation-proof: force `wireRouting={false}` at the
  // render site (which kills the feature in production) → exactly this fails.
  it("full onboarding run (welcome→done) writes the derived pins and shows the summary", async () => {
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue({
      everyday: { source: "basic", model: null, mode: "auto", pin: null },
      synthesis: { source: "none", model: null, mode: "auto", pin: null },
      pool: {
        anthropic: { configured: true, everyday_model: null, synthesis_model: null },
        external: null,
        on_device: { selected: "qwen3-4b-instruct-2507", loaded: true },
      },
    });

    renderWizard(); // no initialStep → full run → wireRouting=true

    // welcome → intelligence (default device model, committed on Continue)
    fireEvent.click(screen.getByText("Get started"));
    await waitFor(() =>
      expect(screen.getByTestId("on-device-model-deferred-note")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText("Continue"));
    // intelligence → import → connect → setting-up (skip the optional steps)
    await waitFor(() => expect(screen.getByText("Bring what you already know")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Skip"));
    await waitFor(() => expect(screen.getByText("Connect your AI tools")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Skip"));
    // setting-up → done
    await waitFor(() =>
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    // DoneStep's effect runs on the full run: derived pins written, summary shown.
    await waitFor(() => expect(setSourcePin).toHaveBeenCalledWith("on_device", "anthropic"));
    expect(
      await screen.findByText(
        "Everyday tasks: On-device. Page synthesis: Anthropic. Change this anytime in Settings → Intelligence.",
      ),
    ).toBeInTheDocument();
    // Exclusivity: a wired routing shows the summary, never the no-model line.
    expect(
      screen.queryByText(/Your wiki updates whenever your AI tools use Wenlan/),
    ).not.toBeInTheDocument();
  });

  // ── Round 2: steps 2-4 collect only; step 5 does + proves everything ────

  it("intelligence step 2 records the on-device model choice but does not download it — only step 5 does", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));

    // Default device mode, default model — no explicit interaction needed.
    // Wait for proof the catalog resolved and the choice was already
    // reported upward (the deferred note only renders once `currentId` is
    // populated) — asserting the absence of a call before that point would
    // pass trivially regardless of whether the code is correct.
    await waitFor(() =>
      expect(screen.getByTestId("on-device-model-deferred-note")).toBeInTheDocument(),
    );
    expect(downloadOnDeviceModel).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("Continue"));
    await waitFor(() => expect(screen.getByText("Bring what you already know")).toBeInTheDocument());
    expect(downloadOnDeviceModel).not.toHaveBeenCalled();
  });

  it("import step 3 records the vault pick but does not import it — only step 5 does", async () => {
    (detectObsidianVaults as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);

    renderWizard({ initialStep: "import" });
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Work Notes"));

    await waitFor(() =>
      expect(screen.getByText("Wenlan will import this when setup finishes.")).toBeInTheDocument(),
    );
    expect(addSource).not.toHaveBeenCalled();
    expect(syncRegisteredSource).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByText("Connect your AI tools")).toBeInTheDocument());
    expect(addSource).not.toHaveBeenCalled();
  });

  it("on-device model: step 5 downloads it and proves loaded, not just that the download call resolved", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });

    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    // Wait for the catalog query to resolve and populate a real model id
    // before committing — clicking Continue before this settles would carry
    // a null pick, same as if the user had skipped.
    await waitFor(() =>
      expect(screen.getByTestId("on-device-model-deferred-note")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText("Continue")); // commit the default device model
    await waitFor(() => expect(screen.getByText("Bring what you already know")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Skip"));
    await waitFor(() => expect(screen.getByText("Connect your AI tools")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(downloadOnDeviceModel).toHaveBeenCalledWith("qwen3-4b-instruct-2507");
    });
    // The download call resolved (the mock always resolves), but `loaded` is
    // still null and the model isn't cached — the row must read
    // "Downloading…", never "done", on the POST alone.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Downloading…");
    });

    // Now the daemon reports the model actually loaded.
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: "qwen3-4b-instruct-2507",
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: true,
      }],
    });

    await waitFor(
      () => {
        expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Ready");
      },
      { timeout: 4000 },
    );
  });

  // Previously unreachable: entering directly at "setting-up" with a pending
  // model id lets the DOWNLOADING vs LOADING split (daemon-reported `cached`)
  // be tested without driving a real multi-minute download through the wizard.
  it("model row shows the DOWNLOADING status and size when the daemon reports the model not yet cached", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(
        screen.getByText("Fetching 2.7 GB. Leave it running; it keeps going if you continue."),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Downloading…");
  });

  it("model row shows the LOADING status once the daemon reports the model cached but not yet loaded", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: true,
      }],
    });

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(screen.getByText("Downloaded. Loading it into memory now.")).toBeInTheDocument();
    });
    expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Loading…");
  });

  // REACHABILITY, not just behavior. Every other test here mocks
  // downloadOnDeviceModel as resolving immediately — but the real command is a
  // single blocking request that only returns once the download AND engine init
  // have finished, minutes later. The model catalog poll used to be gated on
  // that promise resolving, so for the whole download `modelPoll` was undefined,
  // `modelEntry` with it, and the progress bar could never render in the app —
  // while every mocked test rendered it fine. This drives the state the user
  // actually sits in: the download PENDING, forever.
  it("renders byte progress while the download request is still pending (never resolves)", async () => {
    (downloadOnDeviceModel as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });
    (onDeviceModelDownloadBytes as ReturnType<typeof vi.fn>).mockResolvedValue(1_400_000_000);

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(screen.getByText("1.4 GB downloaded")).toBeInTheDocument();
    });
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "52");
  });

  // Nothing on this step is a gate, but a model download made the screen feel
  // like one until the walk-away note said otherwise. It must track whether
  // any row is actually running, not render unconditionally.
  // Absence only — the name used to promise a positive check this test never
  // makes, and it passed with the note deleted outright. The presence half is
  // the next test over; this one exists to catch a note that never goes away.
  it("hides the walk-away note once every row has settled", async () => {
    (getWireState as ReturnType<typeof vi.fn>).mockResolvedValue({
      daemon: { base_url: "http://127.0.0.1:7878", reachable: true, version: "0.12.0", error: null },
      mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
      clients: [],
    });

    renderWizard({ initialStep: "setting-up" });

    // Positive assertion first: settle on "done" before asserting the note's
    // absence, so the absence check can't pass just because nothing rendered yet.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    expect(screen.queryByTestId("setting-up-walk-away")).not.toBeInTheDocument();
  });

  it("shows the walk-away note while the model row is still downloading", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Downloading…");
    });
    expect(screen.getByTestId("setting-up-walk-away")).toBeInTheDocument();
  });

  // ── Byte-based download progress + ETA ──────────────────────────────────
  // The daemon's own download call reports nothing until it finishes, but the
  // hf-hub cache's in-progress `.part` file gives a real byte count. These
  // tests drive that signal through `onDeviceModelDownloadBytes` and check
  // the exact rendered strings — never just "a progress element exists".

  it("no bytes available: falls back to exactly today's copy, with no progressbar rendered", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });
    (onDeviceModelDownloadBytes as ReturnType<typeof vi.fn>).mockResolvedValue(null);

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(
        screen.getByText("Fetching 2.7 GB. Leave it running; it keeps going if you continue."),
      ).toBeInTheDocument();
    });
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("bytes available: renders the downloaded amount, then the derived ETA once a second sample lands", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });
    // 500 MB, then 520 MB one second later — a clean 20 MB/s so the expected
    // ETA comes out to a round number: (2.7GB - 0.52GB) / 20MB/s = 109s -> 2min.
    (onDeviceModelDownloadBytes as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(500_000_000)
      .mockResolvedValueOnce(520_000_000)
      .mockResolvedValue(520_000_000);

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    // One sample only: a real numerator, but not yet two points to derive a
    // rate from, so no ETA yet — never a fabricated one from a single sample.
    await waitFor(() => {
      expect(screen.getByText("0.5 GB downloaded")).toBeInTheDocument();
    });
    let bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "19"); // round(500e6 / 2.7e9 * 100)

    vi.useFakeTimers();
    try {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
    } finally {
      vi.useRealTimers();
    }

    await waitFor(() => {
      expect(screen.getByText("0.5 GB downloaded · about 2 min left")).toBeInTheDocument();
    });
    bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "19"); // round(520e6 / 2.7e9 * 100)
    expect(bar).toHaveAttribute("aria-valuetext", "0.5 GB downloaded · about 2 min left");
  });

  it("bytes at or past the estimated total: the bar clamps below 100%, never claims done off the estimate alone", async () => {
    (getOnDeviceModel as ReturnType<typeof vi.fn>).mockResolvedValue({
      loaded: null,
      selected: "qwen3-4b-instruct-2507",
      models: [{
        id: "qwen3-4b-instruct-2507",
        display_name: "Qwen3 4B",
        param_count: "4B",
        ram_required_gb: 8,
        file_size_gb: 2.7,
        cached: false,
      }],
    });
    // Past the registry's rounded-up 2.7 GB estimate — this happens because
    // the estimate is a hand-rounded number, not the real blob size.
    (onDeviceModelDownloadBytes as ReturnType<typeof vi.fn>).mockResolvedValue(3_000_000_000);

    renderWizard({ initialStep: "setting-up", initialPendingModelId: "qwen3-4b-instruct-2507" });

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeInTheDocument();
    });
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "97");
    expect(bar.firstElementChild).toHaveStyle({ width: "97%" });
    // Still just "downloaded", never "done" — only the real `loaded` poll
    // (asserted elsewhere) is allowed to flip the row's own status to Ready.
    expect(screen.getByTestId("task-status-on-device-model")).toHaveTextContent("Downloading…");
  });

  it("import: step 5 runs addSource + syncRegisteredSource and shows the real SyncStats it gets back, never a fabricated count", async () => {
    (detectObsidianVaults as ReturnType<typeof vi.fn>).mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);
    (addSource as ReturnType<typeof vi.fn>).mockResolvedValue({
      id: "src-1",
      source_type: "obsidian",
      path: "/Users/x/Vaults/Work Notes",
      status: "idle",
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    });
    (syncRegisteredSource as ReturnType<typeof vi.fn>).mockResolvedValue({
      files_found: 1260,
      ingested: 1247,
      skipped: 13,
      errors: 0,
    });

    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Skip")); // skip intelligence — isolate the import row
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Work Notes"));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByText("Connect your AI tools")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(addSource).toHaveBeenCalledWith("obsidian", "/Users/x/Vaults/Work Notes");
    });
    await waitFor(() => {
      expect(syncRegisteredSource).toHaveBeenCalledWith("src-1");
    });
    await waitFor(() => {
      expect(screen.getByTestId("task-status-import")).toHaveTextContent("Imported");
    });
    expect(screen.getByText("1247 indexed, 13 skipped")).toBeInTheDocument();
  });

  it("the runtime row reflects daemon.reachable — a down daemon fails the row with its own error, and Retry re-checks it", async () => {
    (getWireState as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        daemon: { base_url: "http://127.0.0.1:7878", reachable: false, version: null, error: "connection refused" },
        mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
        clients: [],
      })
      .mockResolvedValueOnce({
        daemon: { base_url: "http://127.0.0.1:7878", reachable: true, version: "0.12.0", error: null },
        mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
        clients: [],
      });

    renderWizard({ initialStep: "setting-up" });

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Couldn't set up");
    });
    expect(screen.getByText("connection refused")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("task-retry-daemon"));

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    expect(screen.queryByText("connection refused")).not.toBeInTheDocument();
  });

  // Reachable is not the same as working: the row must prove the actual
  // write path, not just that the daemon answered a ping. It stores a marked
  // probe memory, reads it straight back BY ID, then cleans up after itself.
  // The read-back must never depend on the recent-memories feed: the import
  // row runs concurrently and can flood that feed with real ingested
  // memories between the store and the read, evicting the probe from a
  // fixed-size window and failing this row on exactly the installs that
  // have data to import.
  it("the runtime row proves the write path by id — a flood of unrelated recent memories does not red the row", async () => {
    (storeMemory as ReturnType<typeof vi.fn>).mockResolvedValue({ source_id: "mem_probe-1" });
    // Simulates the exact race this fix closes: the import row raced ahead
    // and pushed enough real memories that the probe would have fallen out
    // of any fixed-size "recent" window. The row must not care — it never
    // reads this feed at all anymore.
    (listRecentMemories as ReturnType<typeof vi.fn>).mockResolvedValue(
      ["mem_flood-1", "mem_flood-2", "mem_flood-3", "mem_flood-4", "mem_flood-5"].map((id) => ({
        kind: "memory",
        id,
        title: "Imported note",
        snippet: null,
        timestamp_ms: Date.now(),
        badge: { kind: "none" },
      })),
    );

    renderWizard({ initialStep: "setting-up" });

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });

    // The row proves it, not just pings it: content is a self-identifying
    // setup check, and the read-back is a real lookup by the id the store
    // call returned — not a read off a feed a concurrent import can flood.
    expect(storeMemory).toHaveBeenCalledWith(
      expect.objectContaining({ content: expect.stringContaining("setup check") }),
    );
    expect(getMemoryDetail).toHaveBeenCalledWith("mem_probe-1");
    await waitFor(() => {
      expect(deleteMemory).toHaveBeenCalledWith("mem_probe-1");
    });
  });

  it("the runtime row fails when the read-back by id comes back null — reachable is not enough", async () => {
    (storeMemory as ReturnType<typeof vi.fn>).mockResolvedValue({ source_id: "mem_probe-2" });
    // The read-back comes back empty: the write never lands, or lands
    // somewhere the daemon can't find it by id either way.
    (getMemoryDetail as ReturnType<typeof vi.fn>).mockResolvedValue(null);

    renderWizard({ initialStep: "setting-up" });

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Couldn't set up");
    });
    expect(
      screen.getByText(/Stored a test memory but couldn't read it back\./),
    ).toBeInTheDocument();
    // Never cleaned up: cleanup only runs once the round trip has actually
    // proved the pipeline works.
    expect(deleteMemory).not.toHaveBeenCalled();
  });

  it("a delete that fails once then succeeds on retry leaves the runtime row done, with no warning", async () => {
    (storeMemory as ReturnType<typeof vi.fn>).mockResolvedValue({ source_id: "mem_probe-3" });
    (deleteMemory as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error("delete failed"))
      .mockResolvedValueOnce(undefined);

    renderWizard({ initialStep: "setting-up" });

    await waitFor(() => {
      expect(deleteMemory).toHaveBeenCalledTimes(2);
    });
    expect(deleteMemory).toHaveBeenNthCalledWith(1, "mem_probe-3");
    expect(deleteMemory).toHaveBeenNthCalledWith(2, "mem_probe-3");
    expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    expect(
      screen.queryByText(
        "Couldn't remove the test memory — you can delete it from your knowledge base.",
      ),
    ).not.toBeInTheDocument();
  });

  it("a delete that fails twice (the retry also fails) leaves the runtime row done, with the cleanup warning surfaced", async () => {
    (storeMemory as ReturnType<typeof vi.fn>).mockResolvedValue({ source_id: "mem_probe-4" });
    (deleteMemory as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("delete failed"));

    renderWizard({ initialStep: "setting-up" });

    await waitFor(() => {
      expect(deleteMemory).toHaveBeenCalledTimes(2);
    });
    expect(deleteMemory).toHaveBeenNthCalledWith(1, "mem_probe-4");
    expect(deleteMemory).toHaveBeenNthCalledWith(2, "mem_probe-4");

    // The round trip already proved the pipeline works, so a cleanup
    // failure — even after a retry — must never red the row.
    expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    expect(screen.queryByText("delete failed")).not.toBeInTheDocument();

    const warningText =
      "Couldn't remove the test memory — you can delete it from your knowledge base.";
    await waitFor(() => {
      expect(screen.getByText(warningText)).toBeInTheDocument();
    });
    // Announced accessibly the same way `errors` already are: the status
    // span describes itself via the warning paragraph's id.
    const statusSpan = screen.getByTestId("task-status-daemon");
    const describedBy = statusSpan.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    const warningEl = document.getElementById(describedBy!);
    expect(warningEl).toHaveTextContent(warningText);
  });

  // The rows run concurrently and independently. A dead daemon is the single
  // most likely failure on a real machine — it must not cancel, skip, or defer
  // the sibling rows. If someone ever "tidies" the kickoff into a sequential
  // await loop, every tool silently stops being configured whenever the daemon
  // happens to be down, and nothing else in this suite would notice.
  it("a down daemon does not stop the tools — sibling rows still run, and the runtime row is listed first", async () => {
    (getWireState as ReturnType<typeof vi.fn>).mockResolvedValue({
      daemon: {
        base_url: "http://127.0.0.1:7878",
        reachable: false,
        version: null,
        error: "connection refused",
      },
      mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
      clients: [],
    });
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Couldn't set up");
    });

    // The tool got configured anyway — that is the whole point.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-cursor")).toHaveTextContent("Configured");
    });
    expect(writeMcpConfig).toHaveBeenCalledWith("cursor");

    // And the runtime row comes first, because its failure is the explanation
    // for anything else that goes wrong further down the list.
    const ids = Array.from(
      screen.getByTestId("setting-up-tasks").querySelectorAll("[data-testid^='task-status-']"),
    ).map((el) => el.getAttribute("data-testid"));
    expect(ids).toEqual(["task-status-daemon", "task-status-cursor"]);
  });

  // The user's actual complaint: step 5 looked like step 4 — "boxes by
  // boxes". This pins the redesign so a regression can't silently bring the
  // card treatment back. Guarded by a preceding positive assertion so it
  // cannot pass before the rows exist at all.
  it("the task rows are not cards — no card background, border, or radius on any row", async () => {
    renderWizard({ initialStep: "setting-up" });

    await screen.findByTestId("setting-up-tasks");

    const rows = screen.getAllByTestId(/^task-row-/);
    expect(rows.length).toBeGreaterThan(0);

    for (const row of rows) {
      expect(row.style.backgroundColor).toBe("");
      expect(row.style.border).toBe("");
      expect(row.style.borderRadius).toBe("");
    }
  });

  // A daemon is not "Configured" — it runs. One flat done-word for every row
  // kind said the wrong thing about the runtime, the model and the import.
  // Caught by looking at the running app, not by any test, so pin it now.
  it("each finished row says what it actually finished — the runtime runs, it is not 'Configured'", async () => {
    (getWireState as ReturnType<typeof vi.fn>).mockResolvedValue({
      daemon: {
        base_url: "http://127.0.0.1:7878",
        reachable: true,
        version: "0.13.0",
        error: null,
      },
      mcp_binary: { command: "wenlan-mcp", args: [], candidates: [] },
      clients: [],
    });
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    renderWizard({ initialStep: "connect" });

    await screen.findByRole("checkbox", { name: "Cursor" });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => {
      expect(screen.getByTestId("task-status-daemon")).toHaveTextContent("Running");
    });
    expect(screen.getByTestId("task-status-daemon")).not.toHaveTextContent("Configured");

    // And a client that IS configured stops claiming it is still being added.
    await waitFor(() => {
      expect(screen.getByTestId("task-status-cursor")).toHaveTextContent("Configured");
    });
    expect(screen.getByText("Wenlan is in Cursor's configuration file.")).toBeInTheDocument();
  });
});

// Pure display-gating rule behind the setting-up rail: a row's terminal
// state only reveals once every row above it is also terminal, so
// concurrent, out-of-order completions still SHOW as a serial top-to-bottom
// sweep even though the underlying work never was serialized.
describe("displayedStatuses", () => {
  const rows = [{ id: "a" }, { id: "b" }, { id: "c" }];

  it("a terminal row below a still-running row displays as running", () => {
    const statuses = { a: "running", b: "done", c: "pending" } as const;
    expect(displayedStatuses(rows, statuses)).toEqual({
      a: "running",
      b: "running",
      c: "pending",
    });
  });

  it("once the row above goes terminal, the row below reveals its real status", () => {
    const statuses = { a: "done", b: "done", c: "pending" } as const;
    expect(displayedStatuses(rows, statuses)).toEqual({
      a: "done",
      b: "done",
      c: "pending",
    });
  });

  it("a failed row above does not block the reveal below it — failed is terminal too", () => {
    const statuses = { a: "failed", b: "done", c: "running" } as const;
    expect(displayedStatuses(rows, statuses)).toEqual({
      a: "failed",
      b: "done",
      c: "running",
    });
  });

  it("a pending row is unaffected either way — non-terminal statuses never change", () => {
    const statuses = { a: "pending", b: "done", c: "done" } as const;
    expect(displayedStatuses(rows, statuses)).toEqual({
      a: "pending",
      b: "running",
      c: "running",
    });
  });
});

// ── Onboarding routing wiring ───────────────────────────────────────────
// First-onboarding writes explicit per-job pins so the defaults are visible,
// not silent. Pure derivation is unit-tested; the effect (feature-detect →
// write → summary) is tested through the rendered Done step.

describe("deriveOnboardingPins", () => {
  const pool = (
    over: Partial<Parameters<typeof deriveOnboardingPins>[0]> = {},
  ): Parameters<typeof deriveOnboardingPins>[0] => ({
    anthropic: { configured: false, everyday_model: null, synthesis_model: null },
    external: null,
    on_device: null,
    ...over,
  });

  it("on-device only → both jobs pin on_device", () => {
    expect(
      deriveOnboardingPins(pool({ on_device: { selected: "qwen3-4b-instruct-2507", loaded: true } })),
    ).toEqual({ everyday: "on_device", synthesis: "on_device" });
  });

  it("cloud key + on-device model → everyday on_device, synthesis anthropic", () => {
    expect(
      deriveOnboardingPins(
        pool({
          anthropic: { configured: true, everyday_model: null, synthesis_model: null },
          on_device: { selected: "qwen3-4b-instruct-2507", loaded: true },
        }),
      ),
    ).toEqual({ everyday: "on_device", synthesis: "anthropic" });
  });

  it("external only, no on-device model → both jobs pin external", () => {
    expect(
      deriveOnboardingPins(pool({ external: { endpoint: "http://localhost:11434/v1", model: "llama3.2" } })),
    ).toEqual({ everyday: "external", synthesis: "external" });
  });

  it("both providers, no on-device → everyday anthropic, synthesis prefers anthropic", () => {
    expect(
      deriveOnboardingPins(
        pool({
          anthropic: { configured: true, everyday_model: null, synthesis_model: null },
          external: { endpoint: "https://api.openai.com/v1", model: "gpt-5.2" },
        }),
      ),
    ).toEqual({ everyday: "anthropic", synthesis: "anthropic" });
  });

  it("nothing configured → no pins (caller writes nothing)", () => {
    expect(deriveOnboardingPins(pool())).toEqual({ everyday: null, synthesis: null });
  });
});

// The wiring effect lives in DoneStep and only runs on the full first-onboarding
// run (wireRouting), which the wizard reaches from `welcome` with no initialStep.
// The test architecture jumps to steps via initialStep (→ wireRouting=false), so
// the write/feature-detect cases render DoneStep directly with wireRouting=true;
// a separate wizard-level test proves the re-run path passes wireRouting=false.
describe("DoneStep onboarding routing wiring (wireRouting=true)", () => {
  type Pool = Parameters<typeof deriveOnboardingPins>[0];
  const routing = (pool: Pool) => ({
    everyday: { source: "basic", model: null, mode: "auto", pin: null },
    synthesis: { source: "none", model: null, mode: "auto", pin: null },
    pool,
  });

  function renderDone(wireRouting: boolean) {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <DoneStep
          wireRouting={wireRouting}
          hideDots={false}
          importResult={null}
          connectedAgents={[]}
          onComplete={vi.fn()}
        />
      </QueryClientProvider>,
    );
  }

  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("en");
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (setSourcePin as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    // Legacy by default; the pinned cases override per test.
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue(null);
  });

  it("pinned daemon: writes the derived pins and shows the wired-routing summary", async () => {
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue(
      routing({
        anthropic: { configured: true, everyday_model: null, synthesis_model: null },
        external: null,
        on_device: { selected: "qwen3-4b-instruct-2507", loaded: true },
      }),
    );
    renderDone(true);

    await waitFor(() => expect(setSourcePin).toHaveBeenCalledWith("on_device", "anthropic"));
    expect(
      await screen.findByText(
        "Everyday tasks: On-device. Page synthesis: Anthropic. Change this anytime in Settings → Intelligence.",
      ),
    ).toBeInTheDocument();
    // Exclusivity: a wired routing shows the summary, never the no-model line.
    expect(
      screen.queryByText(/Your wiki updates whenever your AI tools use Wenlan/),
    ).not.toBeInTheDocument();
  });

  it("legacy daemon: never writes a pin and shows no routing summary", async () => {
    renderDone(true);

    await screen.findByText("Open Wenlan");
    await waitFor(() => expect(getResolvedRouting).toHaveBeenCalled());
    // Flush the feature-detect continuation so a wrongful write/summary would show.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(setSourcePin).not.toHaveBeenCalled();
    expect(screen.queryByText(/Everyday tasks:/)).not.toBeInTheDocument();
    // Concluded without wiring (legacy daemon) → the no-model assurance shows.
    expect(
      await screen.findByText(/Your wiki updates whenever your AI tools use Wenlan/),
    ).toBeInTheDocument();
  });

  it("pinned daemon but nothing configured: writes no pin and shows no summary", async () => {
    (getResolvedRouting as ReturnType<typeof vi.fn>).mockResolvedValue(
      routing({
        anthropic: { configured: false, everyday_model: null, synthesis_model: null },
        external: null,
        on_device: null,
      }),
    );
    renderDone(true);

    await screen.findByText("Open Wenlan");
    await waitFor(() => expect(getResolvedRouting).toHaveBeenCalled());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(setSourcePin).not.toHaveBeenCalled();
    expect(screen.queryByText(/Everyday tasks:/)).not.toBeInTheDocument();
    // Concluded without wiring (nothing configured) → the no-model assurance shows.
    expect(
      await screen.findByText(/Your wiki updates whenever your AI tools use Wenlan/),
    ).toBeInTheDocument();
  });
});
