import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { i18n } from "../../i18n";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

const mocks = vi.hoisted(() => ({
  toggleRemoteAccess: vi.fn(),
  getRemoteAccessStatus: vi.fn(),
  getWireState: vi.fn(),
  installClientPlugin: vi.fn(),
  testRemoteMcpConnection: vi.fn(),
  clipboardWrite: vi.fn(),
}));
vi.mock("../../lib/tauri", () => mocks);

import { RemoteAccessPanel } from "./RemoteAccessPanel";

/** A WireState whose claude_code client has (or lacks) the connector. Pass
 *  `null` for an empty client list (also reads as not-installed). */
function wireState(hasPlugin: boolean | null) {
  return {
    daemon: { base_url: "", reachable: true, version: null, error: null },
    mcp_binary: { command: "", args: [], candidates: [] },
    clients:
      hasPlugin === null
        ? []
        : [
            {
              client_type: "claude_code",
              name: "Claude Code",
              detected: true,
              config_path: "~/.claude.json",
              has_raw_entry: false,
              has_raw_duplicate: false,
              has_plugin: hasPlugin,
              route: "plugin",
            },
          ],
  };
}

function renderPanel() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<RemoteAccessPanel />, { wrapper: Wrapper });
}

const CONNECTED = {
  status: "connected" as const,
  tunnel_url: "https://example.trycloudflare.com",
  token: "secret-token",
  relay_url: null,
};

describe("RemoteAccessPanel", () => {
  beforeEach(() => {
    mocks.getRemoteAccessStatus.mockResolvedValue({ status: "off" });
    mocks.toggleRemoteAccess.mockResolvedValue({ status: "starting" });
    mocks.testRemoteMcpConnection.mockResolvedValue({ ok: true, latency_ms: 42, error: null });
    mocks.clipboardWrite.mockResolvedValue(undefined);
    mocks.installClientPlugin.mockResolvedValue(undefined);
    mocks.getWireState.mockResolvedValue(wireState(null));
  });

  afterEach(async () => {
    vi.clearAllMocks();
    vi.useRealTimers();
    await i18n.changeLanguage("en");
  });

  // "off" is a setting the user chose, not something the app probed — no chip.
  it("renders the Web access title and no status chip when off", async () => {
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Web access")).toBeInTheDocument();
    });
    expect(screen.queryByText("Off")).not.toBeInTheDocument();
    // No disclosure to expand any more — Test/Reconnect only exist when up.
    expect(screen.queryByRole("button", { name: "View relay URL" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reconnect" })).not.toBeInTheDocument();
  });

  it("states the no-auth URL boundary, exactly once", async () => {
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Web access")).toBeInTheDocument();
    });
    expect(screen.getByText(/no authentication/i)).toBeInTheDocument();
    expect(screen.getByText(/anyone with the URL can access Wenlan/i)).toBeInTheDocument();
    expect(screen.getByText(/turn Remote Access off when unused/i)).toBeInTheDocument();
    expect(screen.getAllByText(/no authentication for Claude\.ai and ChatGPT/)).toHaveLength(1);
  });

  it("clicking the toggle calls toggleRemoteAccess(true)", async () => {
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Web access")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { pressed: false }));
    await waitFor(() => {
      expect(mocks.toggleRemoteAccess).toHaveBeenCalledWith(true);
    });
  });

  it("renders 'Connecting…' when starting", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({ status: "starting" });
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText(/Connecting/i)).toBeInTheDocument();
    });
  });

  // Test connection + Reconnect live inline in the status row now, not behind
  // a disclosure — they appear as soon as the relay is connected.
  it("Test connection reports latency inline when connected", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue(CONNECTED);
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /Test connection/i }));
    await waitFor(() => {
      expect(mocks.testRemoteMcpConnection).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getByText(/Connected \(42 ms\)/i)).toBeInTheDocument();
    });
  });

  it("Test connection failure surfaces the error inline", async () => {
    mocks.testRemoteMcpConnection.mockResolvedValue({ ok: false, latency_ms: null, error: "timeout after 5s" });
    mocks.getRemoteAccessStatus.mockResolvedValue(CONNECTED);
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /Test connection/i }));
    await waitFor(() => {
      expect(screen.getByText(/timeout after 5s/i)).toBeInTheDocument();
    });
  });

  it("error state surfaces the verbatim daemon error and offers Retry + Reconnect", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "error",
      error: "connection refused: dial tcp 127.0.0.1:7878",
    });
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("connection refused: dial tcp 127.0.0.1:7878")).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reconnect" })).toBeInTheDocument();
  });

  // ── Claude.ai row ─────────────────────────────────────────────────────
  // has_plugin true → Ready, nothing to do: no Set up button, no steps.
  it("Claude.ai row: connector installed shows Ready and offers no setup", async () => {
    mocks.getWireState.mockResolvedValue(wireState(true));
    renderPanel();
    expect(await screen.findByText("Ready")).toBeInTheDocument();
    expect(
      screen.getByText(/chats on claude\.ai reach your memory while web access is on/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set up" })).not.toBeInTheDocument();
    expect(screen.queryByText("Set up manually")).not.toBeInTheDocument();
  });

  // has_plugin false → the one-click Set up installs the connector.
  it("Claude.ai row: no connector shows a Set up button that installs the Claude plugin", async () => {
    mocks.getWireState.mockResolvedValue(wireState(false));
    renderPanel();
    const setUp = await screen.findByRole("button", { name: "Set up" });
    // Manual fallback is available alongside the one-click path.
    expect(screen.getByText("Set up manually")).toBeInTheDocument();
    fireEvent.click(setUp);
    await waitFor(() => {
      expect(mocks.installClientPlugin).toHaveBeenCalledWith("claude_code");
    });
  });

  // Wire query failed → we can't tell if the connector exists, so only the
  // manual steps show (a one-click install could double-register).
  it("Claude.ai row: an unreadable wire state offers manual steps only, no install button", async () => {
    mocks.getWireState.mockRejectedValue(new Error("daemon down"));
    renderPanel();
    expect(await screen.findByText("Set up manually")).toBeInTheDocument();
    expect(screen.getByText("Step 1 — Add Wenlan to claude.ai")).toBeInTheDocument();
    // The install button shows during the pending window, then the rejection
    // resolves it away — wait for the settled (error) state before asserting.
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Set up" })).not.toBeInTheDocument();
    });
  });

  // ── ChatGPT row ───────────────────────────────────────────────────────
  it("ChatGPT row: prompts to turn on web access when off, with no steps or URL", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({ status: "off" });
    renderPanel();
    expect(await screen.findByText("Turn on web access to connect ChatGPT.")).toBeInTheDocument();
    expect(screen.queryByText(/In ChatGPT, open Settings/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy URL" })).not.toBeInTheDocument();
  });

  // The relay URL's one and only home is the ChatGPT row (getAllByText length
  // 1 catches both a missing URL and a duplicated one).
  it("ChatGPT row: connected shows the steps + URL, and the URL appears exactly once", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "t",
      relay_url: "https://relay.example/abc",
    });
    renderPanel();
    await waitFor(() => {
      expect(screen.getAllByText("https://relay.example/abc")).toHaveLength(1);
    });
    expect(screen.getByText(/In ChatGPT, open Settings/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Copy URL" })).toHaveLength(1);
  });

  it("ChatGPT row: Copy URL writes the /mcp-suffixed tunnel URL when there is no relay", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue(CONNECTED);
    renderPanel();
    const copyBtn = await screen.findByRole("button", { name: "Copy URL" });
    fireEvent.click(copyBtn);
    await waitFor(() => {
      expect(mocks.clipboardWrite).toHaveBeenCalledWith("https://example.trycloudflare.com/mcp");
    });
    expect(screen.getByText("Copied")).toBeInTheDocument();
  });

  // ── i18n + hygiene ────────────────────────────────────────────────────
  it("off state renders the translated title in zh-Hans, no English fallback, no chip", async () => {
    await i18n.changeLanguage("zh-Hans");
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText("网页访问")).toBeInTheDocument();
    });
    expect(screen.queryByText("Web access")).not.toBeInTheDocument();
    expect(screen.queryByText("关闭")).not.toBeInTheDocument();
  });

  it("has no raw #ef4444 or color: white left in the source", async () => {
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const filePath = path.join(process.cwd(), "src/components/memory/RemoteAccessPanel.tsx");
    const source = await fs.readFile(filePath, "utf-8");
    expect(source).not.toMatch(/#ef4444/i);
    expect(source).not.toMatch(/color:\s*["']white["']/i);
  });
});
