// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { startListening, stopListening, onClipboardChange } from "tauri-plugin-clipboard-x-api";
import { emit, listen } from "@tauri-apps/api/event";
import { resizeWindow, resizeWindowCentered } from "./lib/resizeWindow";
import { ingestClipboard, getClipboardEnabled, shouldSkipClipboardChange, setTrafficLightsVisible, shouldShowWizard, setSetupCompleted, type IndexedFileInfo } from "./lib/tauri";
import { triggerAmbient, getAmbientMode, setAmbientMode } from "./lib/ambient";

// Expose ambient controls on window for devtools testing
// Also expose diagnostic helpers
import { invoke } from "@tauri-apps/api/core";
const ambientDiag = async () => {
  const { getAllWebviewWindows } = await import("@tauri-apps/api/webviewWindow");
  const wins = await getAllWebviewWindows();
  console.log("All windows:", wins.map(w => `${w.label} (visible: pending)`));
  for (const w of wins) {
    const vis = await w.isVisible();
    console.log(`  ${w.label}: visible=${vis}`);
  }
};
Object.assign(window, { triggerAmbient, getAmbientMode, setAmbientMode, ambientDiag, invoke, emit });
import { markProcessing, clearProcessing } from "./lib/processingStore";
import { recordCapture } from "./lib/captureHeartbeat";
import Spotlight from "./components/Spotlight";
import RecapDetail from "./components/RecapDetail";
import IdentityDetail from "./components/memory/IdentityDetail";
import Main from "./components/memory/Main";
import SetupWizard from "./components/SetupWizard";
import { MilestoneToaster } from "./components/onboarding/MilestoneToaster";

const MEMORY_WIDTH = 1280;
const MEMORY_HEIGHT = 720;

// Start clipboard watcher once at module level — must not run inside React
// effects because StrictMode double-invokes them, creating duplicate watchers.
// stopListening() first to clear any stale watcher from a previous HMR cycle.
const clipboardReady = stopListening().catch(() => {}).then(() => startListening());

// Clean up watcher when Vite hot-reloads this module
if (import.meta.hot) {
  import.meta.hot.dispose(() => { stopListening(); });
}

type Page = "spotlight" | "home" | "memory" | "recap" | "entity";

export default function App() {
  const queryClient = useQueryClient();
  const { data: showWizard, isLoading: wizardLoading } = useQuery({
    queryKey: ["shouldShowWizard"],
    queryFn: shouldShowWizard,
    staleTime: Infinity,
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
  const [initialView, setInitialView] = useState<"import" | null>(null);
  const [prevPage, setPrevPage] = useState<Page>("spotlight");
  const clipboardEnabledRef = useRef(true);

  // Signal backend that the webview has loaded, so it can show the window
  // without a white flash.
  useEffect(() => {
    emit("app-ready");
  }, []);

  // Keep clipboard enabled state in sync
  useEffect(() => {
    getClipboardEnabled().then((enabled) => {
      clipboardEnabledRef.current = enabled;
    });
  }, [page]); // re-check when leaving settings

  // Clipboard change listener
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    async function setup() {
      await clipboardReady;
      if (cancelled) return;
      unlisten = await onClipboardChange(async (result) => {
        if (cancelled) return;
        if (!clipboardEnabledRef.current) return;
        if (shouldSkipClipboardChange()) return;
        const text = result.text?.value;
        if (text && text.trim().length >= 4) {
          try {
            const chunks = await ingestClipboard(text);
            if (chunks > 0) {
              const firstLine = text.trim().split("\n")[0];
              const summary = firstLine.length > 50 ? firstLine.slice(0, 50) + "..." : firstLine;
              await emit("capture-event", { source: "clipboard", summary, chunks });
            }
          } catch (e) {
            console.error("Failed to ingest clipboard:", e);
          }
        }
      });
      if (cancelled) unlisten?.();
    }

    setup();
    return () => { cancelled = true; unlisten?.(); };
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

  // Navigate to memory detail from ambient overlay (cross-window event)
  useEffect(() => {
    const unlisten = listen<{ sourceId: string }>("navigate-to-memory", async (event) => {
      const { sourceId } = event.payload;
      if (sourceId) {
        setSelectedMemoryId(sourceId);
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

  if (wizardLoading) {
    return <div className="w-screen min-h-screen bg-[var(--bg-secondary)]" />;
  }

  if (showWizard) {
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
          onOpenMemory={() => setPage("home")}
          onOpenRecap={(snap) => { setSelectedSnapshot(snap); setPrevPage("spotlight"); setPage("recap"); }}
          onEntityClick={(id) => { setSelectedEntityId(id); setPrevPage("spotlight"); setPage("entity"); }}
        />
      )}
      {page === "home" && (
        <Main
          initialMemoryId={selectedMemoryId}
          initialView={initialView}
          onBackFromDetail={() => { setSelectedMemoryId(null); setPage("home"); }}
        />
      )}
      {page === "recap" && selectedSnapshot && (
        <RecapDetail snapshot={selectedSnapshot} onBack={() => setPage(prevPage)} />
      )}
      {page === "entity" && selectedEntityId && (
        <IdentityDetail
          entityId={selectedEntityId}
          onBack={() => setPage(prevPage)}
          onEntityClick={(id) => setSelectedEntityId(id)}
          onMemoryClick={(sid) => { setSelectedMemoryId(sid); setInitialView(null); setPage("home"); }}
        />
      )}
      <MilestoneToaster />
    </div>
  );
}
