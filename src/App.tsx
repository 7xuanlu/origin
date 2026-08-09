// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { emit, listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { resizeWindow, resizeWindowCentered } from "./lib/resizeWindow";
import {
  cancelGuardedQuitRequest,
  quitWenlanFull,
  setTrafficLightsVisible,
  shouldShowWizard,
  setSetupCompleted,
  type IndexedFileInfo,
} from "./lib/tauri";
import { markProcessing, clearProcessing } from "./lib/processingStore";
import { recordCapture } from "./lib/captureHeartbeat";
import Spotlight from "./components/Spotlight";
import RecapDetail from "./components/RecapDetail";
import EntityDetail from "./components/memory/EntityDetail";
import Main from "./components/memory/Main";
import SetupWizard from "./components/SetupWizard";
import { RuntimeOverlays } from "./components/RuntimeOverlays";

const MEMORY_WIDTH = 1280;
const MEMORY_HEIGHT = 720;

type Page = "spotlight" | "home" | "memory" | "recap" | "entity";

export default function App() {
  const queryClient = useQueryClient();
  const { data: showWizard, isPending: wizardPending, isError: wizardError } = useQuery({
    queryKey: ["shouldShowWizard"],
    queryFn: shouldShowWizard,
    staleTime: Infinity,
    // Overrides main.tsx's global retry:false — the first-run daemon install
    // (app/src/lib.rs) is spawned async and races this query, so it needs to
    // survive that window (~12s) instead of failing on the first miss.
    retry: 5,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 3000),
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
  const [page, setPage] = useState<Page>("home");
  const [selectedSnapshot, setSelectedSnapshot] = useState<IndexedFileInfo | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [selectedPageId, setSelectedPageId] = useState<string | null>(null);
  const [initialView, setInitialView] = useState<"import" | null>(null);
  const [prevPage, setPrevPage] = useState<Page>("spotlight");
  const quitGuardRef = useRef<(() => Promise<boolean>) | null>(null);
  const quitAttemptRef = useRef<Promise<void> | null>(null);

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
    const cancelQuitRequest = async () => {
      try {
        await cancelGuardedQuitRequest();
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
    const unlisten = listen("quit-requested", () => {
      if (quitAttemptRef.current) return;
      const focusTarget = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
      const attempt = (async () => {
        try {
          await getCurrentWindow().hide();
        } catch {
          await cancelQuitRequest();
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
          await cancelQuitRequest();
          await revealMainWindow(focusTarget);
          return;
        }
        await quitWenlanFull();
      })();
      quitAttemptRef.current = attempt;
      void attempt.catch(async () => {
        await cancelQuitRequest();
        await revealMainWindow(focusTarget);
      }).finally(() => {
        if (quitAttemptRef.current === attempt) quitAttemptRef.current = null;
      });
    });
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

  // Global capture-event → processingStore bridge (persists across page navigation)
  useEffect(() => {
    const unlisten = listen<{ source: string; source_id: string; processing: boolean }>("capture-event", (event) => {
      const { source, source_id, processing } = event.payload;
      if (source_id) {
        recordCapture(source);
        if (processing) markProcessing(source_id);
        else clearProcessing(source_id);
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Spotlight mode: hide traffic lights + always on top; Memory mode: reverse
  useEffect(() => {
    const isSpotlight = page === "spotlight";
    setTrafficLightsVisible(!isSpotlight).catch(() => {});
    import("@tauri-apps/api/window").then(({ getCurrentWindow }) => {
      getCurrentWindow().setAlwaysOnTop(isSpotlight);
    });
  }, [page]);

  // Cmd+K: summon main window and focus the header search input.
  // (Spotlight page mode is retired — the event name is kept to avoid a Rust
  // shortcut-registration change. The Spotlight component is still in the tree
  // but unreachable via normal navigation.)
  useEffect(() => {
    const unlisten = listen("toggle-spotlight", async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      setPage("home");
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

  // Cmd+Shift+K: show Memory page
  useEffect(() => {
    const unlisten = listen("show-memory", () => {
      setPage("home");
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Navigate to memory detail (cross-window event)
  useEffect(() => {
    const unlisten = listen<{ sourceId: string }>("navigate-to-memory", async (event) => {
      const { sourceId } = event.payload;
      if (sourceId) {
        setSelectedMemoryId(sourceId);
        setSelectedPageId(null);
        setInitialView(null);
        setPage("home");
        // Ensure main window is visible and focused
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        const win = getCurrentWindow();
        await win.show();
        await win.setFocus();
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Resize window based on current page
  const prevPageRef = useRef<Page | null>(null); // null = first mount
  useEffect(() => {
    const fullSizePages = ["memory", "home", "recap", "entity"];
    if (fullSizePages.includes(page)) {
      const isFirstMount = prevPageRef.current === null;
      const comingFromSpotlight = prevPageRef.current === "spotlight";
      if (isFirstMount || comingFromSpotlight) {
        resizeWindowCentered(MEMORY_WIDTH, MEMORY_HEIGHT);
      } else {
        resizeWindow(MEMORY_WIDTH, MEMORY_HEIGHT);
      }
    }
    prevPageRef.current = page;
  }, [page]);

  // isPending, not isLoading: isLoading is (isPending && isFetching), which goes
  // false whenever the query is paused rather than fetching — that would fall
  // through to Home with no answer. isPending is true until we actually have one.
  if (wizardPending) {
    return <div className="w-screen min-h-screen bg-[var(--bg-secondary)]" />;
  }

  // ponytail: fail CLOSED. If the daemon is still unreachable after retries,
  // show the wizard rather than silently falling through to Home — an
  // existing user whose daemon is dead for 15s+ sees the wizard too, but its
  // step-5 task thread already surfaces "daemon isn't reachable" + Retry,
  // which is the intended repair surface for that tradeoff.
  if (showWizard || wizardError) {
    return <SetupWizard onComplete={handleWizardComplete} />;
  }

  if (migration) {
    const pct = migration.total > 0 ? Math.round((migration.current / migration.total) * 100) : 0;
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-zinc-950 text-zinc-200">
        <p className="text-lg mb-4">{migration.phase}</p>
        <div className="w-64 h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
        <p className="text-sm text-zinc-500 mt-2">{migration.current} / {migration.total}</p>
      </div>
    );
  }

  return (
    <div className="w-screen min-h-screen bg-[var(--bg-secondary)]">
      {page === "spotlight" && (
        <Spotlight
          onOpenMemory={() => { setSelectedPageId(null); setPage("home"); }}
          onOpenPage={(pageId) => { setSelectedPageId(pageId); setSelectedMemoryId(null); setInitialView(null); setPage("home"); }}
          onOpenRecap={(snap) => { setSelectedPageId(null); setSelectedSnapshot(snap); setPrevPage("spotlight"); setPage("recap"); }}
          onEntityClick={(id) => { setSelectedPageId(null); setSelectedEntityId(id); setPrevPage("spotlight"); setPage("entity"); }}
        />
      )}
      {page === "home" && (
        <Main
          initialMemoryId={selectedMemoryId}
          initialPageId={selectedPageId}
          initialView={initialView}
          onRegisterQuitGuard={registerQuitGuard}
          onBackFromDetail={() => { setSelectedMemoryId(null); setSelectedPageId(null); setPage("home"); }}
        />
      )}
      {page === "recap" && selectedSnapshot && (
        <RecapDetail snapshot={selectedSnapshot} onBack={() => setPage(prevPage)} />
      )}
      {page === "entity" && selectedEntityId && (
        <div className="h-screen overflow-y-auto">
          <EntityDetail
            key={selectedEntityId}
            entityId={selectedEntityId}
            onBack={() => setPage(prevPage)}
            onEntityClick={(id) => setSelectedEntityId(id)}
            onMemoryClick={(sid) => { setSelectedMemoryId(sid); setSelectedPageId(null); setInitialView(null); setPage("home"); }}
          />
        </div>
      )}
      <RuntimeOverlays />
    </div>
  );
}
