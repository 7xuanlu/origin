// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import {
  listMemoriesRich,
  getMemoryStats,
  searchEntities,
  searchConcepts,
  deleteFileChunks,
} from "../../lib/tauri";
import ActivityFeed from "./ActivityFeed";
import { useSearch } from "../../hooks/useSearch";
import IdentityDetail from "./IdentityDetail";
import ProfilePage from "./ProfilePage";
import MemoryStream from "./MemoryStream";
import type { SortMode } from "./MemoryStream";
import HomePage from "./HomePage";
import ConstellationMap from "./ConstellationMap";
import MemoryStatusBar from "./MemoryStatusBar";
import MemorySearchResult from "./MemorySearchResult";
import MemoryDetail from "./MemoryDetail";
import ConceptDetail from "./ConceptDetail";
import SettingsPage from "./SettingsPage";
import { ImportView } from "./ImportView";
import { SetupWizard } from "../SetupWizard";
import ViewToggle from "../ViewToggle";
import Sidebar, { SidebarToggleButton } from "./Sidebar";
import SettingsSidebar, { type SettingsSection } from "./settings/SettingsSidebar";
import SpaceDetail from "./SpaceDetail";
import DecisionLog from "./DecisionLog";
import MemoryCard from "./MemoryCard";

interface MainProps {
  initialMemoryId?: string | null;
  initialView?: "import" | null;
  onBackFromDetail?: () => void;
}

type View = { kind: "home" } | { kind: "stream" } | { kind: "activity" } | { kind: "recaps" } | { kind: "entity"; entityId: string } | { kind: "profile" } | { kind: "memory"; sourceId: string } | { kind: "settings"; section?: SettingsSection } | { kind: "import" } | { kind: "connect-agent" } | { kind: "space"; spaceName: string } | { kind: "graph" } | { kind: "concept"; conceptId: string } | { kind: "decisions" };

const SIDEBAR_KEY = "origin-sidebar-collapsed";

export default function Main({ initialMemoryId, initialView, onBackFromDetail }: MainProps) {
  const queryClient = useQueryClient();
  const [view, setView] = useState<View>(
    initialMemoryId ? { kind: "memory", sourceId: initialMemoryId }
    : initialView === "import" ? { kind: "import" }
    : { kind: "home" },
  );
  const [viewHistory, setViewHistory] = useState<View[]>([]);
  const [activeTab, setActiveTab] = useState<"home" | "activity">("home");

  // Respond to initialView prop changes after mount (e.g. resume banner click)
  useEffect(() => {
    if (initialView === "import") {
      setView({ kind: "import" });
    }
  }, [initialView]);

  // Navigate forward — pushes current view onto history stack
  const navigateTo = (next: View) => {
    setViewHistory((prev) => [...prev, view]);
    setView(next);
  };

  // Navigate back — pops from history stack, falls back to activeTab
  const navigateBack = () => {
    setViewHistory((prev) => {
      if (prev.length === 0) {
        setView({ kind: activeTab });
        return prev;
      }
      const popped = prev[prev.length - 1];
      setView(popped);
      return prev.slice(0, -1);
    });
  };
  const [statusMessage, _setStatusMessage] = useState<string | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>("recent");
  const [stabilityFilter, setStabilityFilter] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) === "true"; } catch { return false; }
  });
  const { query, setQuery, results } = useSearch("memory");

  const [debouncedEntityQuery, setDebouncedEntityQuery] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedEntityQuery(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  const { data: entityResults = [] } = useQuery({
    queryKey: ["searchEntities", debouncedEntityQuery],
    queryFn: () => searchEntities(debouncedEntityQuery, 5),
    enabled: debouncedEntityQuery.length > 0,
  });

  const { data: conceptResults = [] } = useQuery({
    queryKey: ["searchConcepts", debouncedEntityQuery],
    queryFn: () => searchConcepts(debouncedEntityQuery, 5),
    enabled: debouncedEntityQuery.length > 0,
  });

  const toggleSidebar = () => {
    setSidebarCollapsed((v) => {
      const next = !v;
      try { localStorage.setItem(SIDEBAR_KEY, String(next)); } catch {}
      return next;
    });
  };

  const { data: memories = [] } = useQuery({
    queryKey: ["memories"],
    queryFn: () => listMemoriesRich(undefined, undefined, undefined, 200),
    refetchInterval: 5000,
  });

  const { data: _stats } = useQuery({
    queryKey: ["memoryStats"],
    queryFn: getMemoryStats,
    refetchInterval: 10000,
  });

  // Listen for capture events
  useEffect(() => {
    const unlisten = listen<{ source: string }>("capture-event", () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
      queryClient.invalidateQueries({ queryKey: ["homeStats"] });
      queryClient.invalidateQueries({ queryKey: ["recentChanges"] });
      queryClient.invalidateQueries({ queryKey: ["recentRetrievals"] });
      queryClient.invalidateQueries({ queryKey: ["recentConceptItems"] });
      queryClient.invalidateQueries({ queryKey: ["recentMemoryItems"] });
      queryClient.invalidateQueries({ queryKey: ["unconfirmedMemories"] });
      queryClient.invalidateQueries({ queryKey: ["home-recaps"] });
      queryClient.invalidateQueries({ queryKey: ["home-memories"] });
      queryClient.invalidateQueries({ queryKey: ["briefing"] });
      queryClient.invalidateQueries({ queryKey: ["contradictions"] });
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
    });
    return () => { unlisten.then((f) => f()); };
  }, [queryClient]);

  // Status message hidden — impact stats shown on home page instead

  // Cmd+K global shortcut (fired from App.tsx) — focus the header search input.
  useEffect(() => {
    const unlisten = listen("focus-search", () => {
      const searchInput = document.querySelector<HTMLInputElement>('input[placeholder="Search..."]');
      searchInput?.focus();
      searchInput?.select();
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "/" && !e.metaKey && !e.ctrlKey) {
        const active = document.activeElement;
        if (active?.tagName === "INPUT" || active?.tagName === "TEXTAREA") return;
        e.preventDefault();
        const searchInput = document.querySelector<HTMLInputElement>('input[placeholder="Search..."]');
        searchInput?.focus();
      }
      if (e.key === "Escape") {
        if (query) {
          setQuery("");
        } else if (view.kind === "entity" || view.kind === "memory" || view.kind === "settings" || view.kind === "import" || view.kind === "graph" || view.kind === "concept" || view.kind === "decisions") {
          navigateBack();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [query, view]);

  // Close dropdowns on outside click
  const handleEntityClick = (entityId: string) => {
    if (entityId === "__create_profile__") {
      navigateTo({ kind: "profile" });
    } else if (entityId.startsWith("memory:")) {
      navigateTo({ kind: "memory", sourceId: entityId.replace("memory:", "") });
    } else {
      navigateTo({ kind: "entity", entityId });
    }
  };

  return (
    <div
      className="w-full h-screen flex flex-col"
      style={{ backgroundColor: "var(--mem-bg)", color: "var(--mem-text)" }}
    >
      {/* Full-width header */}
      <header
        className="relative flex items-center gap-3 shrink-0"
        style={{
          height: 52,
          paddingLeft: 82,
          paddingRight: 20,
          background: sidebarCollapsed
            ? "transparent"
            : "linear-gradient(to right, var(--mem-sidebar) 240px, transparent 240px)",
        }}
        data-tauri-drag-region
      >
        <SidebarToggleButton collapsed={sidebarCollapsed} onToggle={toggleSidebar} />
        <div className="flex-1" data-tauri-drag-region />

          {/* Right actions */}
          <div className="flex items-center gap-1.5 shrink-0">
            <ViewToggle
              active={view.kind === "activity" ? "activity" : "home"}
              onSwitch={(v) => {
                setActiveTab(v);
                setView({ kind: v });
                setViewHistory([]);
              }}
            />
            {/* Quick Capture */}
            <button
              onClick={async () => {
                const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");
                const win = await WebviewWindow.getByLabel("quick-capture");
                if (!win) return;
                await win.show();
                await win.setFocus();
              }}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              style={{ color: "var(--mem-text-secondary)" }}
              title="Quick Capture (⌘⇧N)"
            >
              <svg className="w-[16px] h-[16px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
            </button>
            {/* Settings */}
            <button
              onClick={() => navigateTo({ kind: "settings" })}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              style={{ color: "var(--mem-text-secondary)" }}
            >
              <svg className="w-[16px] h-[16px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.11 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          </div>

          {/* Search — absolutely centered */}
          <div
            className="absolute flex items-center"
            style={{ left: "50%", transform: "translateX(-50%)" }}
          >
            <div
              className="flex items-center gap-2 rounded-md px-3 py-[6px]"
              style={{
                width: 480,
                backgroundColor: "var(--mem-sidebar)",
                border: "1px solid var(--mem-border)",
              }}
            >
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: "var(--mem-text-tertiary)" }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search..."
              className="flex-1 bg-transparent outline-none"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text)",
              }}
              spellCheck={false}
              autoComplete="off"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="shrink-0"
                style={{ color: "var(--mem-text-tertiary)" }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            )}
            </div>
          </div>
      </header>

      {/* Sidebar + Content row */}
      <div className="flex flex-1 overflow-hidden">
        {view.kind === "settings" ? (
          <SettingsSidebar
            collapsed={sidebarCollapsed}
            active={view.section ?? "intelligence"}
            onSelect={(section) => setView({ kind: "settings", section })}
            onNavigateHome={() => {
              setView({ kind: "home" });
              setActiveTab("home");
              setViewHistory([]);
            }}
          />
        ) : (
          <Sidebar
            collapsed={sidebarCollapsed}
            onSelectSpace={(name) => {
              if (name) {
                navigateTo({ kind: "space", spaceName: name });
              } else {
                setView({ kind: "home" });
                setActiveTab("home");
                setViewHistory([]);
              }
            }}
            onEntityClick={handleEntityClick}
            onNavigateLog={() => { setView({ kind: "stream" }); setViewHistory([]); }}
            onNavigateHome={() => { setView({ kind: "home" }); setActiveTab("home"); setViewHistory([]); }}
            onNavigateDecisions={() => navigateTo({ kind: "decisions" })}
            onNavigateGraph={() => navigateTo({ kind: "graph" })}
          />
        )}

        {/* Main content */}
        <main className={`flex-1 ${view.kind === "graph" ? "overflow-hidden p-0" : "overflow-y-auto pb-7"}`} style={view.kind === "graph" ? undefined : { paddingLeft: "72px", paddingRight: "72px", paddingTop: "56px" }}>
          {/* Search results overlay */}
          {query ? (
            (results.length > 0 || entityResults.length > 0 || conceptResults.length > 0) ? (
              <div className="flex flex-col gap-2">
                {conceptResults.length > 0 && (
                  <>
                    <p
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                      }}
                    >
                      Concepts
                    </p>
                    {conceptResults.map((c) => (
                      <div
                        key={c.id}
                        className="rounded-lg px-4 py-3 cursor-pointer transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                        style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
                        onClick={() => { setQuery(""); navigateTo({ kind: "concept", conceptId: c.id }); }}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: "var(--mem-accent-concept)" }} />
                          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)" }}>
                            {c.title}
                          </span>
                          {c.domain && (
                            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}>
                              {c.domain}
                            </span>
                          )}
                        </div>
                        {c.summary && (
                          <p
                            style={{
                              fontFamily: "var(--mem-font-body)",
                              fontSize: "12px",
                              color: "var(--mem-text-secondary)",
                              marginTop: 4,
                              marginLeft: 16,
                              lineHeight: 1.5,
                            }}
                          >
                            {c.summary}
                          </p>
                        )}
                      </div>
                    ))}
                  </>
                )}
                {results.length > 0 && (
                  <>
                    <p
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                        marginTop: conceptResults.length > 0 ? 12 : 0,
                      }}
                    >
                      {results.length} memor{results.length !== 1 ? "ies" : "y"} for &ldquo;{query}&rdquo;
                    </p>
                    {results.map((r) => (
                      <MemorySearchResult key={r.id} result={r} query={query} onClick={(sid) => { setQuery(""); navigateTo({ kind: "memory", sourceId: sid }); }} />
                    ))}
                  </>
                )}
                {entityResults.length > 0 && (
                  <>
                    <p
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                        marginTop: (results.length > 0 || conceptResults.length > 0) ? 12 : 0,
                      }}
                    >
                      Entities
                    </p>
                    {entityResults.map((r) => (
                      <div
                        key={r.entity.id}
                        className="rounded-lg px-4 py-3 cursor-pointer transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                        style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
                        onClick={() => { setQuery(""); handleEntityClick(r.entity.id); }}
                      >
                        <div className="flex items-center gap-2.5">
                          <span
                            className="text-[10px] font-medium px-1.5 py-0.5 rounded-full"
                            style={{ backgroundColor: "rgba(52, 211, 153, 0.15)", color: "rgb(52, 211, 153)" }}
                          >
                            {r.entity.entity_type}
                          </span>
                          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)" }}>
                            {r.entity.name}
                          </span>
                          {r.entity.domain && (
                            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}>
                              {r.entity.domain}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            ) : (
              <p
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                0 results for &ldquo;{query}&rdquo;
              </p>
            )
          ) : view.kind === "import" ? (
            <ImportView
              onBack={navigateBack}
              onComplete={(_source, _result) => { setView({ kind: "stream" }); setViewHistory([]); }}
            />
          ) : view.kind === "settings" ? (
            <SettingsPage
              section={view.section ?? "intelligence"}
              onBack={navigateBack}
              onSetupAgent={() => navigateTo({ kind: "connect-agent" })}
              onImport={() => navigateTo({ kind: "import" })}
            />
          ) : view.kind === "profile" ? (
            <ProfilePage
              onBack={navigateBack}
              onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "decisions" ? (
            <DecisionLog
              onBack={navigateBack}
              onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onSelectConcept={(id) => navigateTo({ kind: "concept", conceptId: id })}
            />
          ) : view.kind === "space" ? (
            <SpaceDetail
              spaceName={view.spaceName}
              onBack={navigateBack}
              onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onSelectConcept={(id) => navigateTo({ kind: "concept", conceptId: id })}
              onEntityClick={handleEntityClick}
            />
          ) : view.kind === "memory" ? (
            <MemoryDetail
              sourceId={view.sourceId}
              onBack={initialMemoryId && onBackFromDetail && viewHistory.length === 0 ? onBackFromDetail : navigateBack}
              onNavigateEntity={handleEntityClick}
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "connect-agent" ? (
            <SetupWizard
              initialStep="connect"
              onComplete={navigateBack}
            />
          ) : view.kind === "entity" ? (
            <IdentityDetail
              entityId={view.entityId}
              onBack={navigateBack}
              onEntityClick={handleEntityClick}
              onMemoryClick={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "concept" ? (
            <ConceptDetail
              conceptId={view.conceptId}
              onBack={navigateBack}
              onMemoryClick={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onConceptClick={(id) => navigateTo({ kind: "concept", conceptId: id })}
            />
          ) : view.kind === "home" ? (
            <HomePage
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onNavigateStream={() => navigateTo({ kind: "recaps" })}
              onNavigateLog={() => navigateTo({ kind: "stream" })}
              onNavigateGraph={() => navigateTo({ kind: "graph" })}
              onSelectConcept={(id) => navigateTo({ kind: "concept", conceptId: id })}
            />
          ) : view.kind === "activity" ? (
            <ActivityFeed
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "recaps" ? (
            <RecapsList
              onBack={navigateBack}
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "graph" ? (
            <div style={{ position: "relative", width: "100%", height: "100%" }}>
              <ConstellationMap fullScreen onNodeClick={handleEntityClick} />
              <button
                onClick={navigateBack}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                style={{
                  position: "absolute", top: 12, left: 12, zIndex: 10,
                  color: "var(--mem-text-secondary)", fontSize: 12, fontFamily: "var(--mem-font-body)",
                  background: "var(--mem-surface)", border: "1px solid var(--mem-border)",
                  cursor: "pointer", backdropFilter: "blur(8px)",
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
                Back
              </button>
            </div>
          ) : (
            <>
              <button onClick={() => navigateTo({ kind: "home" })} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] mb-3" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
              </button>
              <h2 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "24px", fontWeight: 500, color: "var(--mem-text)", margin: "0 0 12px 0" }}>Memory Log</h2>
              <MemoryStream
                memories={memories}
                selectedDomain={null}
                sortMode={sortMode}
                onSortChange={setSortMode}
                stabilityFilter={stabilityFilter}
                onStabilityFilterChange={setStabilityFilter}
                onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              />
            </>
          )}
        </main>
      </div>

      {/* Status bar */}
      <MemoryStatusBar message={statusMessage} />
    </div>
  );
}

function RecapsList({ onBack, onNavigateMemory }: { onBack: () => void; onNavigateMemory: (sid: string) => void }) {
  const queryClient = useQueryClient();
  const { data: recaps = [] } = useQuery({
    queryKey: ["all-recaps"],
    queryFn: async () => {
      const all = await listMemoriesRich(undefined, undefined, undefined, 200);
      return all.filter((m) => m.is_recap === true);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (sourceId: string) => deleteFileChunks("memory", sourceId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["all-recaps"] }),
  });

  return (
    <div>
      <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] mb-3" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0 }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
      </button>
      <h2 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "20px", fontWeight: 400, color: "var(--mem-text)", margin: "0 0 16px 0" }}>Recaps</h2>
      <h2
        className="mb-4"
        style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" as const, color: "var(--mem-accent-indigo)" }}
      >
        All Recaps ({recaps.length})
      </h2>
      <div className="flex flex-col">
        {recaps.map((recap) => (
          <MemoryCard
            key={recap.source_id}
            memory={recap}
            onConfirm={() => {}}
            onDelete={(sid) => deleteMutation.mutate(sid)}
            expandedChain={false}
            onToggleChain={() => {}}
            versionChain={[]}
            onClick={onNavigateMemory}
          />
        ))}
      </div>
    </div>
  );
}
