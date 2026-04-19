import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { RemoteAccessPanel } from "./RemoteAccessPanel";

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

vi.mock("../../lib/tauri", () => ({
  toggleRemoteAccess: vi.fn().mockResolvedValue({ status: "starting" }),
  getRemoteAccessStatus: vi.fn().mockResolvedValue({ status: "off" }),
  rotateRemoteToken: vi.fn().mockResolvedValue("new-token"),
  testRemoteMcpConnection: vi.fn().mockResolvedValue({ ok: true, latency_ms: 42, error: null }),
  clipboardWrite: vi.fn().mockResolvedValue(undefined),
}));

import {
  toggleRemoteAccess,
  getRemoteAccessStatus,
  rotateRemoteToken,
  testRemoteMcpConnection,
  clipboardWrite,
} from "../../lib/tauri";

function renderPanel(mode: "compact" | "full") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<RemoteAccessPanel mode={mode} />, { wrapper: Wrapper });
}

describe("RemoteAccessPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "off" });
    (toggleRemoteAccess as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "starting" });
    (rotateRemoteToken as ReturnType<typeof vi.fn>).mockResolvedValue("new-token");
    (testRemoteMcpConnection as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      latency_ms: 42,
      error: null,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders 'Off' status when disabled", async () => {
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText(/Off/i)).toBeInTheDocument();
    });
  });

  it("renders 'Connecting…' when starting", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({ status: "starting" });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText(/Connecting/i)).toBeInTheDocument();
    });
  });

  it("renders 'Connected' and URL when connected", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: "https://relay.origin.dev/abcdef/mcp",
    });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    expect(screen.getByText("https://relay.origin.dev/abcdef/mcp")).toBeInTheDocument();
  });

  it("clicking toggle calls toggleRemoteAccess", async () => {
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText(/Off/i)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("switch"));
    await waitFor(() => {
      expect(toggleRemoteAccess).toHaveBeenCalledWith(true);
    });
  });

  it("Copy URL button shows 'Copied!' briefly", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    const copyBtn = screen.getByRole("button", { name: /^Copy URL$/i });
    fireEvent.click(copyBtn);
    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalledWith("https://example.trycloudflare.com/mcp");
    });
    expect(screen.getByText(/Copied!/i)).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(2100);
    });
    await waitFor(() => {
      expect(screen.queryByText(/Copied!/i)).not.toBeInTheDocument();
    });
  });

  it("Test connection button calls testRemoteMcpConnection and shows 'Connected (NNNms)'", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    const testBtn = screen.getByRole("button", { name: /Test connection/i });
    fireEvent.click(testBtn);
    await waitFor(() => {
      expect(testRemoteMcpConnection).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getByText(/Connected \(42ms\)/i)).toBeInTheDocument();
    });
  });

  it("Test connection failure shows error message", async () => {
    (testRemoteMcpConnection as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      latency_ms: null,
      error: "timeout after 5s",
    });
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: /Test connection/i }));
    await waitFor(() => {
      expect(screen.getByText(/timeout after 5s/i)).toBeInTheDocument();
    });
  });

  it("compact mode does NOT render Token section", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("compact");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    expect(screen.queryByText(/Token/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Rotate$/i })).not.toBeInTheDocument();
  });

  it("full mode renders Token section and Rotate button", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("full");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    expect(screen.getByText(/Token/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Rotate$/i })).toBeInTheDocument();
  });

  it("Rotate requires two clicks (confirm pattern)", async () => {
    (getRemoteAccessStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "connected",
      tunnel_url: "https://example.trycloudflare.com",
      token: "secret-token",
      relay_url: null,
    });
    renderPanel("full");
    await waitFor(() => {
      expect(screen.getByText("Connected")).toBeInTheDocument();
    });
    const rotate = screen.getByRole("button", { name: /^Rotate$/i });
    fireEvent.click(rotate);
    // Label should change after first click
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Click again to confirm/i })).toBeInTheDocument();
    });
    expect(rotateRemoteToken).not.toHaveBeenCalled();
    // Second click should fire rotateRemoteToken
    fireEvent.click(screen.getByRole("button", { name: /Click again to confirm/i }));
    await waitFor(() => {
      expect(rotateRemoteToken).toHaveBeenCalledTimes(1);
    });
  });
});
