// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider, onlineManager } from "@tanstack/react-query";
import App from "./App";

const eventListeners = vi.hoisted(
  () => new Map<string, (event: { payload: unknown }) => void>(),
);
const quitGuardMock = vi.hoisted(
  () => vi.fn<() => Promise<boolean>>(),
);
const quitWenlanFullMock = vi.hoisted(
  () => vi.fn<() => Promise<void>>(),
);
const cancelGuardedQuitRequestMock = vi.hoisted(
  () => vi.fn<() => Promise<void>>(),
);
const hideWindowMock = vi.hoisted(() => vi.fn<() => Promise<void>>());
const showWindowMock = vi.hoisted(() => vi.fn<() => Promise<void>>());
const focusWindowMock = vi.hoisted(() => vi.fn<() => Promise<void>>());
const emitMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/event", () => ({
  emit: emitMock,
  listen: vi.fn((event: string, handler: (event: { payload: unknown }) => void) => {
    eventListeners.set(event, handler);
    return Promise.resolve(() => eventListeners.delete(event));
  }),
}));

vi.mock("./lib/tauri", () => ({
  cancelGuardedQuitRequest: cancelGuardedQuitRequestMock,
  quitWenlanFull: quitWenlanFullMock,
  shouldShowWizard: vi.fn(),
  setSetupCompleted: vi.fn().mockResolvedValue(undefined),
  setTrafficLightsVisible: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("./lib/resizeWindow", () => ({
  resizeWindow: vi.fn(),
  resizeWindowCentered: vi.fn(),
}));

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    setAlwaysOnTop: vi.fn(),
    isVisible: vi.fn().mockResolvedValue(true),
    hide: hideWindowMock,
    show: showWindowMock,
    setFocus: focusWindowMock,
    setSize: vi.fn().mockResolvedValue(undefined),
    setPosition: vi.fn().mockResolvedValue(undefined),
    scaleFactor: vi.fn().mockResolvedValue(1),
  }),
  LogicalSize: vi.fn(),
  LogicalPosition: vi.fn(),
  currentMonitor: vi.fn().mockResolvedValue(null),
}));

// Heavy real children — swap for markers so this test only pins App's own
// wizard-vs-home branching, not Main's or SetupWizard's internals.
vi.mock("./components/memory/Main", () => ({
  default: (props: {
    onRegisterQuitGuard?: (guard: (() => Promise<boolean>) | null) => void;
  }) => {
    props.onRegisterQuitGuard?.(quitGuardMock);
    return (
      <div data-testid="home-main">
        <input aria-label="Draft title" />
      </div>
    );
  },
}));

vi.mock("./components/SetupWizard", () => ({
  default: () => <div data-testid="setup-wizard">wizard</div>,
}));

// EntityDetail transitively imports AtlasView → sigma, whose dist touches
// WebGL2RenderingContext at module scope — jsdom has no such global, so the
// real import crashes this whole suite before a single test runs.
vi.mock("./components/memory/EntityDetail", () => ({
  default: () => null,
}));

vi.mock("./components/onboarding/MilestoneToaster", () => ({
  MilestoneToaster: () => null,
}));

vi.mock("./components/UpdaterDialog", () => ({
  default: () => null,
}));

import { shouldShowWizard } from "./lib/tauri";

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App - first-run wizard gate", () => {
  beforeEach(() => {
    eventListeners.clear();
    emitMock.mockReset().mockResolvedValue(undefined);
    focusWindowMock.mockReset().mockResolvedValue(undefined);
    hideWindowMock.mockReset().mockResolvedValue(undefined);
    quitGuardMock.mockReset().mockResolvedValue(true);
    cancelGuardedQuitRequestMock.mockReset().mockResolvedValue(undefined);
    quitWenlanFullMock.mockReset().mockResolvedValue(undefined);
    showWindowMock.mockReset().mockResolvedValue(undefined);
    vi.mocked(shouldShowWizard).mockReset();
  });

  it("renders Home when shouldShowWizard resolves false", async () => {
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();

    expect(await screen.findByTestId("home-main")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-wizard")).not.toBeInTheDocument();
  });

  it("renders SetupWizard when shouldShowWizard resolves true", async () => {
    vi.mocked(shouldShowWizard).mockResolvedValue(true);
    renderApp();

    expect(await screen.findByTestId("setup-wizard")).toBeInTheDocument();
    expect(screen.queryByTestId("home-main")).not.toBeInTheDocument();
  });

  // The gate asks the daemon over localhost IPC, so it must still answer on a
  // machine with no network. Under react-query's default "online" networkMode
  // an offline browser PAUSES the query (fetchStatus "paused", never
  // "fetching"), which would strand the gate with no data and no error.
  it("still answers the gate when the machine is offline", async () => {
    onlineManager.setOnline(false);
    try {
      vi.mocked(shouldShowWizard).mockResolvedValue(true);
      renderApp();

      expect(await screen.findByTestId("setup-wizard")).toBeInTheDocument();
      expect(screen.queryByTestId("home-main")).not.toBeInTheDocument();
    } finally {
      onlineManager.setOnline(true);
    }
  });

  // The daemon's first-run install is async (app/src/lib.rs) and can race this
  // query past its own retry budget. App.tsx's per-query retry (5 attempts,
  // exponential backoff capped at 3s, ~12s total) overrides main.tsx's global
  // retry:false, so this exercises the REAL production retry policy end to
  // end rather than a shortened test double — hence the generous timeout.
  it(
    "fails closed to SetupWizard when shouldShowWizard rejects (daemon unreachable)",
    async () => {
      vi.mocked(shouldShowWizard).mockRejectedValue(new Error("connection refused"));
      renderApp();

      expect(
        await screen.findByTestId("setup-wizard", {}, { timeout: 20000 }),
      ).toBeInTheDocument();
      expect(screen.queryByTestId("home-main")).not.toBeInTheDocument();
      // Proves retries actually happened, not just a single failed attempt.
      expect(vi.mocked(shouldShowWizard).mock.calls.length).toBeGreaterThan(1);
    },
    25000,
  );

  it("waits for the active editor guard before an explicit quit reaches Tauri", async () => {
    let resolveFlush!: (saved: boolean) => void;
    quitGuardMock.mockReturnValue(new Promise((resolve) => {
      resolveFlush = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });
    expect(quitGuardMock).toHaveBeenCalledTimes(1);
    expect(quitWenlanFullMock).not.toHaveBeenCalled();

    await act(async () => {
      resolveFlush(true);
    });
    await waitFor(() => expect(quitWenlanFullMock).toHaveBeenCalledTimes(1));
  });

  it("waits for the native window to hide before starting the draft flush", async () => {
    let resolveHide!: () => void;
    hideWindowMock.mockReturnValue(new Promise<void>((resolve) => {
      resolveHide = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });
    expect(hideWindowMock).toHaveBeenCalledTimes(1);
    expect(quitGuardMock).not.toHaveBeenCalled();

    await act(async () => {
      resolveHide();
    });
    await waitFor(() => expect(quitGuardMock).toHaveBeenCalledTimes(1));
  });

  it("keeps the native app hidden after flushing while teardown is still pending", async () => {
    let resolveTeardown!: () => void;
    quitWenlanFullMock.mockReturnValue(new Promise<void>((resolve) => {
      resolveTeardown = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });
    await waitFor(() => expect(quitWenlanFullMock).toHaveBeenCalledTimes(1));

    expect(hideWindowMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveTeardown();
    });
  });

  it("aborts quit and reveals the editor when its pending draft cannot be saved", async () => {
    quitGuardMock.mockResolvedValueOnce(false).mockResolvedValueOnce(true);
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });
    expect(quitWenlanFullMock).not.toHaveBeenCalled();
    expect(hideWindowMock).toHaveBeenCalledTimes(1);
    expect(showWindowMock).toHaveBeenCalledTimes(1);
    expect(focusWindowMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledTimes(1);
    expect(emitMock).not.toHaveBeenCalledWith("quit-cancelled");

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });
    await waitFor(() => expect(quitWenlanFullMock).toHaveBeenCalledTimes(1));
  });

  it("unlocks and reveals the app when native teardown rejects", async () => {
    quitWenlanFullMock.mockRejectedValue(new Error("shutdown failed"));
    hideWindowMock.mockImplementationOnce(async () => {
      if (document.activeElement instanceof HTMLElement) {
        document.activeElement.blur();
      }
    });
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    const title = await screen.findByRole("textbox", { name: "Draft title" });
    title.focus();
    expect(title).toHaveFocus();

    await act(async () => {
      eventListeners.get("quit-requested")?.({ payload: null });
      await Promise.resolve();
    });

    await waitFor(() => expect(showWindowMock).toHaveBeenCalledTimes(1));
    expect(hideWindowMock).toHaveBeenCalledTimes(1);
    expect(focusWindowMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledTimes(1);
    expect(emitMock).not.toHaveBeenCalledWith("quit-cancelled");
    expect(title).toHaveFocus();
  });
});
