// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useState, useEffect, useRef, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { emit, listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  acknowledgeGuardedQuitRequest,
  cancelGuardedQuitRequest,
  quitWenlanFull,
  setTrafficLightsVisible,
  shouldShowWizard,
  setSetupCompleted,
} from "./lib/tauri";
import { BOOT_QUERY_RETRY, bootQueryRetryDelay } from "./lib/bootRetryPolicy";
import Main from "./components/memory/Main";
import SetupWizard from "./components/SetupWizard";
import { RuntimeOverlays } from "./components/RuntimeOverlays";

export default function App() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: showWizard, isPending: wizardPending, isError: wizardError } = useQuery({
    queryKey: ["shouldShowWizard"],
    queryFn: shouldShowWizard,
    staleTime: Infinity,
    // Overrides main.tsx's global retry:false — the first-run daemon install
    // (app/src/lib.rs) is spawned async and races this query, so it needs to
    // survive that window instead of failing on the first miss.
    // The schedule and, more importantly, the total budget live in
    // ./lib/bootRetryPolicy: it has to outlast the Rust health loop's own
    // ~152s wait for this same daemon, or the fail-closed branch below shows
    // the first-run wizard to a configured user while startup is still
    // waiting. Each attempt is separately bounded to 5s (app/src/api.rs).
    retry: BOOT_QUERY_RETRY,
    retryDelay: bootQueryRetryDelay,
    // This is a Tauri IPC call to a daemon on localhost, not a network request.
    // The default "online" mode would PAUSE it whenever navigator.onLine is
    // false (fetchStatus "paused", never "fetching"), so an offline machine
    // would strand the gate below with no data and no error.
    networkMode: "always",
  });

  async function handleWizardComplete() {
    await setSetupCompleted(true);
    queryClient.invalidateQueries({ queryKey: ["shouldShowWizard"] });
  }

  const [migration, setMigration] = useState<{ current: number; total: number; phase: string } | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [selectedPageId, setSelectedPageId] = useState<string | null>(null);
  const quitGuardRef = useRef<(() => Promise<boolean>) | null>(null);
  const quitAttemptRef = useRef<{
    requestId: number;
  } | null>(null);

  const registerQuitGuard = useCallback((guard: (() => Promise<boolean>) | null) => {
    quitGuardRef.current = guard;
  }, []);

  // Signal backend that the webview has loaded, so it can focus the already
  // visible main window after the frontend is ready.
  useEffect(() => {
    emit("app-ready");
  }, []);

  // Native app-menu and tray quits arrive here before Rust shuts down the
  // daemon. This lets the active editor bypass its debounce and make the last
  // keystroke durable. A failed save aborts quit and reveals the existing retry
  // surface instead of silently dropping text.
  useEffect(() => {
    const cancelQuitRequest = async (requestId: number, deliveryId: number) => {
      try {
        await cancelGuardedQuitRequest(requestId, deliveryId);
      } catch {
        // A failed cancellation signal must not hide the editor recovery path.
      }
    };
    const revealMainWindow = async (focusTarget: HTMLElement | null) => {
      try {
        const win = getCurrentWindow();
        await win.show();
        await win.setFocus();
      } catch {
        // The safe fallback is still to leave the process running.
      }
      await Promise.resolve();
      if (focusTarget?.isConnected) focusTarget.focus();
    };
    const unlisten = listen<{ requestId: number; deliveryId: number }>(
      "quit-requested",
      (event) => {
        const { requestId, deliveryId } = event.payload;
        void (async () => {
          let acknowledged = false;
          try {
            acknowledged = await acknowledgeGuardedQuitRequest(requestId, deliveryId);
          } catch {
            return;
          }
          if (!acknowledged) return;

          const activeAttempt = quitAttemptRef.current;
          if (activeAttempt?.requestId === requestId) return;
          // Reachable only in a narrow race: an attempt that already cancelled
          // itself (a rejected hide(), or a failed save) has returned the
          // coordinator to Idle, but quitAttemptRef is not cleared until the
          // finally microtask runs. A Cmd-Q landing inside that gap opens a new
          // request that we acknowledge and then discard here, so the press
          // appears to do nothing. It is not silent: discarding counts as a
          // refusal in the Rust coordinator, so a user who keeps pressing hits
          // the escape hatch and gets out on the third press.
          if (activeAttempt) {
            await cancelQuitRequest(requestId, deliveryId);
            return;
          }

          const focusTarget = document.activeElement instanceof HTMLElement
            ? document.activeElement
            : null;
          const attempt = (async () => {
            try {
              await getCurrentWindow().hide();
            } catch {
              await cancelQuitRequest(requestId, deliveryId);
              await revealMainWindow(focusTarget);
              return;
            }
            let persisted = false;
            try {
              persisted = quitGuardRef.current ? await quitGuardRef.current() : true;
            } catch {
              persisted = false;
            }
            if (!persisted) {
              await cancelQuitRequest(requestId, deliveryId);
              await revealMainWindow(focusTarget);
              return;
            }
            await quitWenlanFull();
          })();
          const trackedAttempt = { requestId };
          quitAttemptRef.current = trackedAttempt;
          void attempt
            .catch(async () => {
              await cancelQuitRequest(requestId, deliveryId);
              await revealMainWindow(focusTarget);
            })
            .finally(() => {
              if (quitAttemptRef.current === trackedAttempt) quitAttemptRef.current = null;
            });
        })();
      },
    );
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Embedding migration progress overlay
  useEffect(() => {
    const unlisten1 = listen<{ current: number; total: number; phase: string }>(
      'migration-progress',
      (event) => setMigration(event.payload)
    );
    const unlisten2 = listen('migration-complete', () => setMigration(null));
    return () => {
      unlisten1.then(f => f());
      unlisten2.then(f => f());
    };
  }, []);

  // Spotlight mode is retired — Home is the only reachable page, so traffic
  // lights are always visible and the window is never always-on-top.
  useEffect(() => {
    setTrafficLightsVisible(true).catch(() => {});
    getCurrentWindow().setAlwaysOnTop(false);
  }, []);

  // Cmd+K: summon main window and focus the header search input.
  // (Spotlight page mode is retired — the event name is kept to avoid a Rust
  // shortcut-registration change.)
  useEffect(() => {
    const unlisten = listen("toggle-spotlight", async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      if (!(await win.isVisible())) {
        await win.show();
      }
      await win.setFocus();
      // Give Main a tick to mount, then signal it to focus the search input.
      await new Promise((r) => setTimeout(r, 30));
      await emit("focus-search");
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Cmd+Shift+K: show Memory page (no-op now that Home is the only page —
  // the listener is kept so the Rust-side shortcut registration is untouched).
  useEffect(() => {
    const unlisten = listen("show-memory", () => {});
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Navigate to memory detail (cross-window event)
  useEffect(() => {
    const unlisten = listen<{ sourceId: string }>("navigate-to-memory", async (event) => {
      const { sourceId } = event.payload;
      if (sourceId) {
        setSelectedMemoryId(sourceId);
        setSelectedPageId(null);
        // Ensure main window is visible and focused
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        const win = getCurrentWindow();
        await win.show();
        await win.setFocus();
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // The window is born at its final size and centered by Tauri
  // (app/tauri.conf.json: 1280x720, "center": true). Nothing here resizes or
  // moves it afterwards — a mount-time resize made the window visibly shrink
  // and jump a couple of seconds after launch, every launch.

  // isPending, not isLoading: isLoading is (isPending && isFetching), which goes
  // false whenever the query is paused rather than fetching — that would fall
  // through to Home with no answer. isPending is true until we actually have one.
  let body: ReactNode;
  if (wizardPending) {
    // The daemon can take seconds to answer on a cold start. Say so, instead
    // of holding an empty window. --bg-primary is what index.html paints
    // before React mounts (src/__tests__/firstPaint.test.ts pins the two to
    // the same hex), so this panel takes over that background without a flash.
    body = (
      <div
        className="flex flex-col items-center justify-center w-screen h-screen bg-[var(--bg-primary)] text-[var(--text-secondary)]"
        role="status"
        aria-live="polite"
      >
        <p className="text-lg">{t("common.startingRuntime")}</p>
      </div>
    );
  } else if (showWizard || wizardError) {
    // ponytail: fail CLOSED. If the daemon is still unreachable after retries,
    // show the wizard rather than silently falling through to Home — an
    // existing user whose daemon is dead for 15s+ sees the wizard too, but its
    // step-5 task thread already surfaces "daemon isn't reachable" + Retry,
    // which is the intended repair surface for that tradeoff.
    body = <SetupWizard onComplete={handleWizardComplete} />;
  } else if (migration) {
    const pct = migration.total > 0 ? Math.round((migration.current / migration.total) * 100) : 0;
    body = (
      <div className="flex flex-col items-center justify-center h-screen bg-zinc-950 text-zinc-200">
        <p className="text-lg mb-4">{migration.phase}</p>
        <div className="w-64 h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
        <p className="text-sm text-zinc-500 mt-2">{migration.current} / {migration.total}</p>
      </div>
    );
  } else {
    body = (
      <div className="w-screen min-h-screen bg-[var(--bg-secondary)]">
        <Main
          initialMemoryId={selectedMemoryId}
          initialPageId={selectedPageId}
          onRegisterQuitGuard={registerQuitGuard}
          onBackFromDetail={() => { setSelectedMemoryId(null); setSelectedPageId(null); }}
        />
      </div>
    );
  }

  return (
    <>
      {body}
      <RuntimeOverlays
        variant={
          wizardPending || showWizard || wizardError || migration
            ? "updater-only"
            : "main"
        }
      />
    </>
  );
}
