import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SetupWizard } from "./SetupWizard";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

vi.mock("../lib/tauri", () => ({
  importMemories: vi.fn(),
  shouldShowWizard: vi.fn().mockResolvedValue(true),
  setSetupCompleted: vi.fn().mockResolvedValue(undefined),
  detectMcpClients: vi.fn().mockResolvedValue([]),
  writeMcpConfig: vi.fn().mockResolvedValue(undefined),
  listAgents: vi.fn().mockResolvedValue([]),
  getOriginMcpEntry: vi.fn().mockResolvedValue({ command: "npx", args: ["-y", "origin-mcp"] }),
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
  getSystemInfo: vi.fn().mockResolvedValue({
    total_ram_gb: 16,
    available_ram_gb: 10,
    has_metal: true,
    has_cuda: false,
    os: "macOS",
    arch: "arm64",
    recommended_builtin: "qwen3-4b-instruct-2507",
  }),
}));

import {
  detectMcpClients,
  writeMcpConfig,
  listAgents,
  setApiKey,
} from "../lib/tauri";

function renderWizard(
  props: {
    onComplete?: () => void;
    initialStep?: "welcome" | "intelligence-choice" | "import" | "connect" | "verify" | "done";
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
        <SetupWizard onComplete={onComplete} initialStep={props.initialStep} />
      </QueryClientProvider>,
    ),
  };
}

describe("SetupWizard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (setApiKey as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);
  });

  it("renders Welcome step by default", () => {
    renderWizard();
    expect(screen.getByText("Welcome to Origin")).toBeInTheDocument();
    expect(screen.getByText(/Where understanding compounds/i)).toBeInTheDocument();
    expect(screen.getByText("Everything stays on your device")).toBeInTheDocument();
  });

  it('advances from Welcome to intelligence choice on "Get started" click', () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    expect(screen.getByText("Choose how Origin thinks")).toBeInTheDocument();
  });

  it("lets users save an API key from the intelligence step", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Use my API key"));

    fireEvent.change(screen.getByPlaceholderText("sk-ant-..."), {
      target: { value: "sk-ant-test-key" },
    });
    fireEvent.click(screen.getByText("Save"));

    await waitFor(() => {
      expect(setApiKey).toHaveBeenCalledWith("sk-ant-test-key");
    });
  });

  it("shows chat-history guidance in the import step and routes directly to connect", async () => {
    renderWizard();
    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Continue"));

    expect(screen.getByText("Import Memories")).toBeInTheDocument();
    expect(screen.getByText(/Settings > Sources/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Choose tools to connect")).toBeInTheDocument();
    });
  });

  it("connect step separates detected and supported safe tools", async () => {
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

    await waitFor(() => {
      expect(screen.getByText("Detected on your Mac")).toBeInTheDocument();
      expect(screen.getByText("Supported tools")).toBeInTheDocument();
    });

    expect(screen.getByText("Cursor")).toBeInTheDocument();
    expect(screen.getByText("Claude Code")).toBeInTheDocument();
    const [cursorCheckbox, claudeCheckbox] = screen.getAllByRole("checkbox");
    expect(cursorCheckbox).toBeEnabled();
    expect(claudeCheckbox).toBeDisabled();
  });

  it("connects selected detected tools on continue", async () => {
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

    await waitFor(() => {
      expect(screen.getByText("Cursor")).toBeInTheDocument();
    });

    const [cursorCheckbox] = screen.getAllByRole("checkbox");
    await waitFor(() => {
      expect(cursorCheckbox).toBeChecked();
    });

    fireEvent.click(screen.getByText("Continue"));

    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("cursor");
      expect(screen.getByText("Waiting for your first agent...")).toBeInTheDocument();
    });
  });

  it("stays on connect step when MCP setup fails", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);
    (writeMcpConfig as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("setup failed"));

    renderWizard({ initialStep: "connect" });

    await waitFor(() => {
      expect(screen.getByText("Cursor")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Continue"));

    await waitFor(() => {
      expect(writeMcpConfig).toHaveBeenCalledWith("cursor");
      expect(screen.getByText("Choose tools to connect")).toBeInTheDocument();
      expect(screen.queryByText("Waiting for your first agent...")).not.toBeInTheDocument();
      expect(screen.getByText(/setup failed/i)).toBeInTheDocument();
    });
  });

  it("verify step skips waiting UX when agents already wrote in the past", async () => {
    (listAgents as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Claude",
        client_type: "claude_code",
        first_seen_at: Math.floor(Date.now() / 1000) - 86400,
        last_seen_at: Math.floor(Date.now() / 1000) - 3600,
        memory_count: 5,
      },
    ]);

    renderWizard({ initialStep: "verify" });

    await waitFor(() => {
      expect(screen.getByText("You're all set.")).toBeInTheDocument();
    });

    expect(screen.getByText("Claude")).toBeInTheDocument();
    expect(screen.queryByText("Back")).not.toBeInTheDocument();
  });

  it("renders skip-path Done copy without a back button", async () => {
    renderWizard();

    fireEvent.click(screen.getByText("Get started"));
    fireEvent.click(screen.getByText("Continue"));
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Choose tools to connect")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Skip"));
    fireEvent.click(screen.getByText("Skip"));

    await waitFor(() => {
      expect(screen.getByText("Origin is ready.")).toBeInTheDocument();
    });

    expect(screen.queryByText("Back")).not.toBeInTheDocument();
  });

  it("Connect step renders RemoteAccessPanel (compact)", async () => {
    renderWizard({ initialStep: "connect" });
    await waitFor(() => {
      expect(screen.getByText(/Share with web-based AI tools/i)).toBeInTheDocument();
    });
  });

  it("Connect step remote toggle invokes toggleRemoteAccess", async () => {
    const { toggleRemoteAccess } = await import("../lib/tauri");
    renderWizard({ initialStep: "connect" });
    await screen.findByText(/Share with web-based AI tools/i);
    fireEvent.click(screen.getByRole("switch"));
    await waitFor(() => {
      expect(toggleRemoteAccess).toHaveBeenCalled();
    });
  });
});
