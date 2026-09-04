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
const acknowledgeGuardedQuitRequestMock = vi.hoisted(
  () => vi.fn<(requestId: number, deliveryId: number) => Promise<boolean>>(),
);
const cancelGuardedQuitRequestMock = vi.hoisted(
  () => vi.fn<(requestId: number, deliveryId: number) => Promise<boolean>>(),
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
  acknowledgeGuardedQuitRequest: acknowledgeGuardedQuitRequestMock,
  cancelGuardedQuitRequest: cancelGuardedQuitRequestMock,
  quitWenlanFull: quitWenlanFullMock,
  shouldShowWizard: vi.fn(),
  setSetupCompleted: vi.fn().mockResolvedValue(undefined),
  setTrafficLightsVisible: vi.fn().mockResolvedValue(undefined),
}));

// No setSize/setPosition/scaleFactor/currentMonitor here on purpose: App no
// longer touches the window geometry, so mocking those would only hide a
// regression that re-added the launch-time resize.
vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    setAlwaysOnTop: vi.fn(),
    isVisible: vi.fn().mockResolvedValue(true),
    hide: hideWindowMock,
    show: showWindowMock,
    setFocus: focusWindowMock,
  }),
}));

// The real ladder is ~162s of wall clock by design (it has to outlast the
// Rust health loop). Running it here would be a three-minute unit test, so
// this file substitutes a two-attempt policy and asserts only the branching
// it owns. The production schedule and its budget are pinned as arithmetic in
// src/lib/bootRetryPolicy.test.ts.
vi.mock("./lib/bootRetryPolicy", () => ({
  ATTEMPT_TIMEOUT_MS: 5000,
  RUST_HEALTH_LOOP_BUDGET_MS: 152_200,
  BOOT_QUERY_RETRY: 1,
  bootQueryRetryDelay: () => 10,
  bootQueryBudgetMs: () => 10_010,
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

vi.mock("./components/RuntimeOverlays", () => ({
  RuntimeOverlays: ({ variant = "main" }: { variant?: string }) => (
    <div data-testid="runtime-overlays" data-variant={variant}>runtime overlays</div>
  ),
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
import { resources } from "./i18n/resources";

const STARTING_RUNTIME = resources.en.translation.common.startingRuntime;

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

function dispatchQuit(requestId = 1, deliveryId = 1) {
  eventListeners.get("quit-requested")?.({
    payload: { requestId, deliveryId },
  });
}

describe("App - first-run wizard gate", () => {
  beforeEach(() => {
    eventListeners.clear();
    emitMock.mockReset().mockResolvedValue(undefined);
    focusWindowMock.mockReset().mockResolvedValue(undefined);
    hideWindowMock.mockReset().mockResolvedValue(undefined);
    quitGuardMock.mockReset().mockResolvedValue(true);
    acknowledgeGuardedQuitRequestMock.mockReset().mockResolvedValue(true);
    cancelGuardedQuitRequestMock.mockReset().mockResolvedValue(true);
    quitWenlanFullMock.mockReset().mockResolvedValue(undefined);
    showWindowMock.mockReset().mockResolvedValue(undefined);
    vi.mocked(shouldShowWizard).mockReset();
  });

  it("renders Home when shouldShowWizard resolves false", async () => {
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();

    expect(await screen.findByTestId("home-main")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-wizard")).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime-overlays")).toHaveAttribute("data-variant", "main");
  });

  it("renders SetupWizard when shouldShowWizard resolves true", async () => {
    vi.mocked(shouldShowWizard).mockResolvedValue(true);
    renderApp();

    expect(await screen.findByTestId("setup-wizard")).toBeInTheDocument();
    expect(screen.getByTestId("runtime-overlays")).toHaveAttribute(
      "data-variant",
      "updater-only",
    );
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

  // A cold start can leave the window with nothing to show for seconds. The
  // gate must say what it is waiting for rather than holding an empty window,
  // and it must not pre-empt that wait with the first-run wizard.
  it("shows the starting-runtime status while the wizard query is pending", async () => {
    let resolveWizard!: (showWizard: boolean) => void;
    vi.mocked(shouldShowWizard).mockReturnValue(
      new Promise<boolean>((resolve) => {
        resolveWizard = resolve;
      }),
    );
    renderApp();

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent(STARTING_RUNTIME);
    expect(screen.queryByTestId("setup-wizard")).not.toBeInTheDocument();
    expect(screen.queryByTestId("home-main")).not.toBeInTheDocument();
    expect(screen.getByTestId("runtime-overlays")).toHaveAttribute(
      "data-variant",
      "updater-only",
    );

    // …and it is a waiting state, not a terminal one.
    await act(async () => {
      resolveWizard(false);
    });
    expect(await screen.findByTestId("home-main")).toBeInTheDocument();
    expect(screen.queryByText(STARTING_RUNTIME)).not.toBeInTheDocument();
  });

  // The daemon's first-run install is async (app/src/lib.rs) and can race this
  // query. App.tsx's per-query retry overrides main.tsx's global retry:false,
  // and this pins that the query retries at all and then fails CLOSED. How
  // MANY times and for how long is the policy module's contract, tested there.
  it("fails closed to SetupWizard when shouldShowWizard rejects (daemon unreachable)", async () => {
    vi.mocked(shouldShowWizard).mockRejectedValue(new Error("connection refused"));
    renderApp();

    expect(await screen.findByTestId("setup-wizard")).toBeInTheDocument();
    expect(screen.queryByTestId("home-main")).not.toBeInTheDocument();
    // Proves retries actually happened, not just a single failed attempt.
    expect(vi.mocked(shouldShowWizard).mock.calls.length).toBeGreaterThan(1);
  });

  it("waits for the active editor guard before an explicit quit reaches Tauri", async () => {
    let resolveFlush!: (saved: boolean) => void;
    quitGuardMock.mockReturnValue(new Promise((resolve) => {
      resolveFlush = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      dispatchQuit();
      await Promise.resolve();
    });
    expect(acknowledgeGuardedQuitRequestMock).toHaveBeenCalledWith(1, 1);
    expect(quitGuardMock).toHaveBeenCalledTimes(1);
    expect(quitWenlanFullMock).not.toHaveBeenCalled();

    await act(async () => {
      resolveFlush(true);
    });
    await waitFor(() => expect(quitWenlanFullMock).toHaveBeenCalledTimes(1));
  });

  it("does not hide or flush until the native quit delivery is acknowledged", async () => {
    let resolveAcknowledgement!: (acknowledged: boolean) => void;
    acknowledgeGuardedQuitRequestMock.mockReturnValue(new Promise((resolve) => {
      resolveAcknowledgement = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      dispatchQuit(3, 1);
      await Promise.resolve();
    });
    expect(hideWindowMock).not.toHaveBeenCalled();
    expect(quitGuardMock).not.toHaveBeenCalled();

    await act(async () => {
      resolveAcknowledgement(true);
    });
    await waitFor(() => expect(quitGuardMock).toHaveBeenCalledTimes(1));
  });

  it("ignores a stale native quit delivery without hiding or flushing", async () => {
    acknowledgeGuardedQuitRequestMock.mockResolvedValue(false);
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      dispatchQuit(3, 9);
      await Promise.resolve();
    });

    expect(hideWindowMock).not.toHaveBeenCalled();
    expect(quitGuardMock).not.toHaveBeenCalled();
    expect(quitWenlanFullMock).not.toHaveBeenCalled();
  });

  it("acknowledges a liveness probe without starting a second draft flush", async () => {
    let resolveFlush!: (saved: boolean) => void;
    quitGuardMock.mockReturnValue(new Promise((resolve) => {
      resolveFlush = resolve;
    }));
    vi.mocked(shouldShowWizard).mockResolvedValue(false);
    renderApp();
    await screen.findByTestId("home-main");

    await act(async () => {
      dispatchQuit(4, 1);
      await Promise.resolve();
    });
    await waitFor(() => expect(quitGuardMock).toHaveBeenCalledTimes(1));

    await act(async () => {
      dispatchQuit(4, 2);
      await Promise.resolve();
    });
    await waitFor(() => {
      expect(acknowledgeGuardedQuitRequestMock).toHaveBeenNthCalledWith(2, 4, 2);
    });
    expect(quitGuardMock).toHaveBeenCalledTimes(1);

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
      dispatchQuit();
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
      dispatchQuit();
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
      dispatchQuit();
      await Promise.resolve();
    });
    expect(quitWenlanFullMock).not.toHaveBeenCalled();
    expect(hideWindowMock).toHaveBeenCalledTimes(1);
    expect(showWindowMock).toHaveBeenCalledTimes(1);
    expect(focusWindowMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledWith(1, 1);
    expect(emitMock).not.toHaveBeenCalledWith("quit-cancelled");

    await act(async () => {
      dispatchQuit(2, 1);
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
      dispatchQuit();
      await Promise.resolve();
    });

    await waitFor(() => expect(showWindowMock).toHaveBeenCalledTimes(1));
    expect(hideWindowMock).toHaveBeenCalledTimes(1);
    expect(focusWindowMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledTimes(1);
    expect(cancelGuardedQuitRequestMock).toHaveBeenCalledWith(1, 1);
    expect(emitMock).not.toHaveBeenCalledWith("quit-cancelled");
    expect(title).toHaveFocus();
  });
});
