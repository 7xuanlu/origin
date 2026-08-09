// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { useTranslation } from "react-i18next";
import {
  listMemoriesRich,
  listSpaces,
  getMemoryStats,
  openFile,
  searchEntities,
  searchPages,
  type Page,
  type SearchResult,
  type Space,
} from "../../lib/tauri";
import ActivityFeed from "./ActivityFeed";
import { useSearch } from "../../hooks/useSearch";
import EntityDetail from "./EntityDetail";
import MemoryStream from "./MemoryStream";
import type { SortMode } from "./MemoryStream";
import HomePage from "./HomePage";
import AtlasView from "./AtlasView";
import MemoryStatusBar from "./MemoryStatusBar";
import MemorySearchResult from "./MemorySearchResult";
import MemoryDetail from "./MemoryDetail";
import PageDetail from "./PageDetail";
import DistillReviewPanel from "./DistillReviewPanel";
import SettingsPage from "./SettingsPage";
import { ImportView } from "./ImportView";
import { SetupWizard } from "../SetupWizard";
import Sidebar, { SidebarHeaderDivider, SidebarToggleButton } from "./Sidebar";
import SettingsSidebar from "./settings/SettingsSidebar";
import SpaceDetail from "./SpaceDetail";
import { SpacesOverview } from "./spaces";
import { PagesOverview } from "./pages/PagesOverview";
import {
  PageDraftEditor,
  type PageDraftEditorHandle,
} from "./pages/PageDraftEditor";
import SourcesView from "./SourcesView";
import DecisionLog from "./DecisionLog";
import { RecapsList } from "./RecapsList";
import AboutWenlanDialog from "./AboutWenlanDialog";
import { readPreference, writePreference } from "../../lib/preferenceStorage";
import { searchResultTarget } from "../../lib/searchResultTarget";
import { recordRecentPageVisit } from "../../lib/recentPages";
import { deleteRecentSpace, recordRecentSpaceVisit, renameRecentSpace } from "../../lib/recentSpaces";
import { createSpaceDetailCopy, createSpacesOverviewLabels } from "./navigation/copy";
import { activeNavigationForView, type View } from "./navigation/viewState";
import { ReviewEnvironmentBadge } from "./navigation/ReviewEnvironmentBadge";
import { useResponsiveSidebar } from "./navigation/useResponsiveSidebar";
import "./navigation/navigation-shell.css";

interface MainProps {
  initialMemoryId?: string | null;
  initialPageId?: string | null;
  initialView?: "import" | null;
  onBackFromDetail?: () => void;
  onRegisterQuitGuard?: (guard: (() => Promise<boolean>) | null) => void;
}
const SIDEBAR_KEY = "wenlan-sidebar-collapsed";
const LEGACY_SIDEBAR_KEY = "origin-sidebar-collapsed";

function scrollDestinationKey(view: View): string {
  switch (view.kind) {
    case "entity":
      return `entity:${view.entityId}`;
    case "memory":
      return `memory:${view.sourceId}`;
    case "page":
      return `page:${view.pageId}`;
    case "page-draft":
      return `page-draft:${view.draftId ?? "new"}:${view.space ?? "none"}`;
    case "settings":
      return `settings:${view.section ?? "general"}`;
    case "space":
      return `space:${view.spaceId ?? view.spaceName}`;
    default:
      return view.kind;
  }
}

export default function Main({
  initialMemoryId,
  initialPageId,
  initialView,
  onBackFromDetail,
  onRegisterQuitGuard,
}: MainProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const mainContentRef = useRef<HTMLElement>(null);
  const pageDraftEditorRef = useRef<PageDraftEditorHandle>(null);
  const pendingDraftNavigationRef = useRef<{
    readonly action: (sourceView: View) => void;
    readonly token: symbol;
  } | null>(null);
  const draftNavigationFlushRef = useRef<Promise<boolean> | null>(null);
  const pendingDraftSearchCancelRef = useRef<(() => void) | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);
  const externalMemoryIdRef = useRef<string | null>(initialMemoryId ?? null);
  const [view, setView] = useState<View>(
    initialMemoryId ? { kind: "memory", sourceId: initialMemoryId }
    : initialPageId ? { kind: "page", pageId: initialPageId }
    : initialView === "import" ? { kind: "import" }
    : { kind: "home" },
  );
  const viewRef = useRef(view);
  viewRef.current = view;
  const [viewHistory, setViewHistory] = useState<View[]>([]);
  const [activeTab, setActiveTab] = useState<"home" | "activity">("home");
  const [aboutOpen, setAboutOpen] = useState(false);
  const [recentPagesRevision, setRecentPagesRevision] = useState(0);
  const [recentSpacesRevision, setRecentSpacesRevision] = useState(0);
  const [mobileSearchOpen, setMobileSearchOpen] = useState(false);
  const [pendingDraftSearchQuery, setPendingDraftSearchQuery] = useState<string | null>(null);
  const viewScrollDestination = scrollDestinationKey(view);

  const prepareForQuit = useCallback(async (): Promise<boolean> => {
    const editor = pageDraftEditorRef.current;
    if (viewRef.current.kind !== "page-draft" || !editor) return true;
    return editor.flush();
  }, []);

  useEffect(() => {
    onRegisterQuitGuard?.(prepareForQuit);
    return () => onRegisterQuitGuard?.(null);
  }, [onRegisterQuitGuard, prepareForQuit]);

  useLayoutEffect(() => {
    if (!mainContentRef.current) return;
    mainContentRef.current.scrollLeft = 0;
    mainContentRef.current.scrollTop = 0;
  }, [viewScrollDestination]);

  const afterPageDraftFlush = (action: (sourceView: View) => void): (() => void) => {
    const editor = pageDraftEditorRef.current;
    if (view.kind !== "page-draft" || !editor) {
      action(view);
      return () => {};
    }

    const token = Symbol("draft-navigation");
    pendingDraftNavigationRef.current = { action, token };
    const cancel = () => {
      if (pendingDraftNavigationRef.current?.token === token) {
        pendingDraftNavigationRef.current = null;
      }
    };
    if (draftNavigationFlushRef.current) return cancel;

    const flush = editor.flush();
    draftNavigationFlushRef.current = flush;
    void flush.then(
      (saved) => {
        const pending = pendingDraftNavigationRef.current;
        pendingDraftNavigationRef.current = null;
        if (saved && pending) {
          const identity = editor.getIdentity();
          const sourceView: View = {
            ...view,
            draftId: identity.draftId ?? undefined,
          };
          if (sourceView !== view) setView(sourceView);
          pending.action(sourceView);
        }
      },
      () => {
        pendingDraftNavigationRef.current = null;
      },
    ).finally(() => {
      if (draftNavigationFlushRef.current === flush) {
        draftNavigationFlushRef.current = null;
      }
    });
    return cancel;
  };

  // Respond to externally requested destinations only after the current draft is durable.
  useEffect(() => {
    if (initialView === "import") {
      return afterPageDraftFlush(() => setView({ kind: "import" }));
    }
  }, [initialView]);

  useEffect(() => {
    if (initialPageId) {
      return afterPageDraftFlush(() => setView({ kind: "page", pageId: initialPageId }));
    }
  }, [initialPageId]);

  useEffect(() => {
    if (initialMemoryId) {
      return afterPageDraftFlush(() => {
        externalMemoryIdRef.current = initialMemoryId;
        setViewHistory([]);
        setView({ kind: "memory", sourceId: initialMemoryId });
      });
    } else if (externalMemoryIdRef.current) {
      return afterPageDraftFlush(() => {
        externalMemoryIdRef.current = null;
        setViewHistory([]);
        setView({ kind: activeTab });
      });
    }
  }, [initialMemoryId, activeTab]);

  // Navigate forward — pushes current view onto history stack
  const navigateTo = (next: View) => {
    afterPageDraftFlush((sourceView) => {
      setViewHistory((prev) => [...prev, sourceView]);
      setView(next);
    });
  };

  const navigateHome = () => {
    afterPageDraftFlush(() => {
      setView({ kind: "home" });
      setActiveTab("home");
      setViewHistory([]);
    });
  };

  const navigateSpaces = (create: boolean) => {
    afterPageDraftFlush(() => {
      setView(create ? { kind: "spaces", create: true } : { kind: "spaces" });
      setViewHistory([]);
    });
  };

  const navigatePages = () => {
    afterPageDraftFlush(() => {
      setView({ kind: "pages" });
      setViewHistory([]);
    });
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
    return readPreference(SIDEBAR_KEY, LEGACY_SIDEBAR_KEY) === "true";
  });
  const { query, setQuery, results } = useSearch();
  const handleSearchQueryChange = (nextQuery: string) => {
    if (!query && nextQuery && view.kind === "page-draft") {
      pendingDraftSearchCancelRef.current?.();
      setPendingDraftSearchQuery(nextQuery);
      pendingDraftSearchCancelRef.current = afterPageDraftFlush(() => {
        pendingDraftSearchCancelRef.current = null;
        setQuery(nextQuery);
        setPendingDraftSearchQuery(null);
      });
      return;
    }
    pendingDraftSearchCancelRef.current?.();
    pendingDraftSearchCancelRef.current = null;
    setPendingDraftSearchQuery(null);
    setQuery(nextQuery);
  };
  const displayedSearchQuery = pendingDraftSearchQuery ?? query;
  const memoryResults = results.filter((result) => searchResultTarget(result).kind === "copy");
  const sourceResults = results.filter((result) => searchResultTarget(result).kind === "file");

  useEffect(() => {
    if (mobileSearchOpen) searchInputRef.current?.focus();
  }, [mobileSearchOpen]);

  useEffect(() => {
    if (view.kind !== "page-draft") setPendingDraftSearchQuery(null);
  }, [view.kind]);

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
    queryKey: ["searchPages", debouncedEntityQuery],
    queryFn: () => searchPages(debouncedEntityQuery, 5),
    enabled: debouncedEntityQuery.length > 0,
  });

  const toggleSidebar = () => {
    setSidebarCollapsed((v) => {
      const next = !v;
      writePreference(SIDEBAR_KEY, String(next));
      return next;
    });
  };
  const responsiveSidebar = useResponsiveSidebar(sidebarCollapsed, toggleSidebar, sidebarToggleRef);
  const standardSidebarMounted = view.kind !== "settings" && view.kind !== "connect-agent";
  const activeNavigation = activeNavigationForView(view);
  const spacesOverviewLabels = createSpacesOverviewLabels(t);
  const spaceDetailCopy = createSpaceDetailCopy(t);
  const { data: spaces } = useQuery({ queryKey: ["spaces"], queryFn: listSpaces });

  const refreshRecentSpaces = useCallback(() => {
    setRecentSpacesRevision((revision) => revision + 1);
  }, []);
  const handlePageLoaded = useCallback((page: Pick<Page, "id" | "status" | "title">) => {
    recordRecentPageVisit(page);
    setRecentPagesRevision((revision) => revision + 1);
  }, []);
  const handleSpaceLoaded = useCallback((space: Space) => {
    const runtime = spaces === undefined
      ? undefined
      : { spaces: [space, ...spaces.filter((current) => current.id !== space.id)] };
    recordRecentSpaceVisit(space, runtime);
    setView((current) => current.kind === "space" && current.spaceName === space.name
      ? { ...current, spaceId: space.id }
      : current);
    refreshRecentSpaces();
  }, [refreshRecentSpaces, spaces]);
  const handleSpaceRenamed = useCallback((space: Pick<Space, "id" | "name">) => {
    const runtime = spaces === undefined
      ? undefined
      : {
          spaces: spaces.map((current) => current.id === space.id
            ? { ...current, name: space.name }
            : current),
        };
    renameRecentSpace(space, runtime);
    setView((current) => current.kind === "space" && current.spaceId === space.id
      ? { ...current, spaceName: space.name }
      : current);
    refreshRecentSpaces();
  }, [refreshRecentSpaces, spaces]);
  const handleSpaceDeleted = useCallback((spaceId: string) => {
    const runtime = spaces === undefined
      ? undefined
      : { spaces: spaces.filter(({ id }) => id !== spaceId) };
    deleteRecentSpace(spaceId, runtime);
    refreshRecentSpaces();
  }, [refreshRecentSpaces, spaces]);

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
      setMobileSearchOpen(true);
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
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
        setMobileSearchOpen(true);
        searchInputRef.current?.focus();
      }
      if (e.key === "Escape") {
        if (responsiveSidebar.presentation === "overlay" && responsiveSidebar.open) return;
        if (query) {
          setQuery("");
        } else if (mobileSearchOpen) {
          setMobileSearchOpen(false);
        } else if (view.kind === "page-draft") {
          // PageDraftEditor owns Escape so it can await the same flush gate as Back.
          return;
        } else if (view.kind === "entity" || view.kind === "memory" || view.kind === "settings" || view.kind === "import" || view.kind === "graph" || view.kind === "page" || view.kind === "space" || view.kind === "distill-review" || view.kind === "decisions") {
          navigateBack();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mobileSearchOpen, query, responsiveSidebar.open, responsiveSidebar.presentation, view]);

  // Close dropdowns on outside click
  const handleEntityClick = (entityId: string) => {
    if (entityId === "__create_profile__") {
      navigateTo({ kind: "settings", section: "general" });
    } else if (entityId.startsWith("memory:")) {
      navigateTo({ kind: "memory", sourceId: entityId.replace("memory:", "") });
    } else if (entityId.startsWith("page:")) {
      navigateTo({ kind: "page", pageId: entityId.replace("page:", "") });
    } else {
      navigateTo({ kind: "entity", entityId });
    }
  };

  const openSearchResult = async (result: SearchResult) => {
    setQuery("");
    setMobileSearchOpen(false);
    const target = searchResultTarget(result);
    if (target.kind === "page") {
      navigateTo({ kind: "page", pageId: target.pageId });
    } else if (target.kind === "file") {
      await openFile(target.url);
    } else {
      navigateTo({ kind: "memory", sourceId: result.source_id });
    }
  };

  return (
    <div
      className="memory-shell flex h-screen w-full flex-col"
      style={{ backgroundColor: "var(--mem-bg)", color: "var(--mem-text)" }}
    >
      {/* Full-width header */}
      <header
        className="relative flex items-center gap-3 shrink-0"
        style={{
          height: 52,
          paddingLeft: 82,
          paddingRight: 20,
          background: responsiveSidebar.presentation === "desktop" && !responsiveSidebar.collapsed
            ? "linear-gradient(to right, var(--mem-sidebar) 240px, transparent 240px)"
            : "transparent",
        }}
        data-tauri-drag-region
      >
        <SidebarHeaderDivider
          visible={responsiveSidebar.presentation === "desktop" && !responsiveSidebar.collapsed}
        />
        <SidebarToggleButton collapsed={responsiveSidebar.collapsed} onToggle={responsiveSidebar.toggle} ref={sidebarToggleRef} />
        {(!standardSidebarMounted || !responsiveSidebar.open) && <ReviewEnvironmentBadge compact />}
        <div className="flex-1" data-tauri-drag-region />

          {/* Right actions */}
          <div className="flex items-center gap-1.5 shrink-0">
            <button
              aria-expanded={mobileSearchOpen}
              aria-label={t("main.searchButton")}
              className="lg:hidden rounded-md p-1.5 transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              onClick={() => setMobileSearchOpen((open) => !open)}
              style={{ color: "var(--mem-text-secondary)" }}
              type="button"
            >
              <svg aria-hidden="true" fill="none" height="16" stroke="currentColor" viewBox="0 0 24 24" width="16">
                <path d="m21 21-4.35-4.35m2.35-5.65a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
              </svg>
            </button>
            <button
              aria-current={view.kind === "activity" ? "page" : undefined}
              className="flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              onClick={() => {
                afterPageDraftFlush(() => {
                  setActiveTab("activity");
                  setView({ kind: "activity" });
                  setViewHistory([]);
                });
              }}
              style={{ color: view.kind === "activity" ? "var(--mem-text)" : "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)", fontSize: "12px" }}
              type="button"
            >
              <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 24 24" width="16">
                <path d="M3 12a9 9 0 109-9 9.75 9.75 0 00-6.74 2.74L3 8M3 3v5h5M12 7v5l3 2" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" />
              </svg>
              <span>{t("main.activity")}</span>
            </button>
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
              title={t("main.quickCaptureTitle")}
            >
              <svg className="w-[16px] h-[16px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
            </button>
          </div>

          {/* Search — absolutely centered */}
          <div
            className={`${mobileSearchOpen ? "flex" : "hidden"} absolute left-4 right-4 top-[56px] z-50 items-center lg:flex lg:left-1/2 lg:right-auto lg:top-auto lg:-translate-x-1/2`}
          >
            <div
              className="flex w-full items-center gap-2 rounded-md px-3 py-[6px] shadow-lg focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--mem-accent-page)] lg:w-[clamp(220px,40vw,480px)] lg:shadow-none"
              style={{
                backgroundColor: "var(--mem-sidebar)",
                border: "1px solid var(--mem-control-border)",
              }}
            >
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: "var(--mem-text-tertiary)" }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={searchInputRef}
              data-wenlan-search-input
              value={displayedSearchQuery}
              onChange={(e) => handleSearchQueryChange(e.target.value)}
              placeholder={t("main.searchPlaceholder")}
              className="flex-1 bg-transparent outline-none"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text)",
              }}
              spellCheck={false}
              autoComplete="off"
            />
            {displayedSearchQuery && (
              <button
                onClick={() => {
                  pendingDraftSearchCancelRef.current?.();
                  pendingDraftSearchCancelRef.current = null;
                  setPendingDraftSearchQuery(null);
                  setQuery("");
                }}
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
            active={view.section ?? "general"}
            onSelect={(section) => setView({ kind: "settings", section })}
            onNavigateHome={navigateHome}
          />
        ) : view.kind === "connect-agent" ? null : (
          <Sidebar
            activeNavigation={activeNavigation}
            collapsed={responsiveSidebar.collapsed}
            currentPageId={view.kind === "page" ? view.pageId : null}
            currentSpaceId={view.kind === "space" ? view.spaceId : null}
            onEntityClick={handleEntityClick}
            onNavigateLog={() => {
              afterPageDraftFlush(() => {
                setView({ kind: "stream" });
                setViewHistory([]);
              });
            }}
            onNavigatePages={navigatePages}
            onNavigateHome={navigateHome}
            onNavigateGraph={() => navigateTo({ kind: "graph" })}
            onNavigateSources={() => navigateTo({ kind: "sources" })}
            onNavigateSpaces={navigateSpaces}
            onNavigateSettings={() => navigateTo({ kind: "settings", section: "general" })}
            onOpenAbout={() => setAboutOpen(true)}
            onRequestClose={responsiveSidebar.close}
            onSelectPage={(page) => navigateTo({ kind: "page", pageId: page.id })}
            onSelectSpace={(space) => navigateTo({ kind: "space", spaceId: space.id, spaceName: space.name })}
            open={responsiveSidebar.open}
            presentation={responsiveSidebar.presentation}
            recentPagesRevision={recentPagesRevision}
            recentSpacesRevision={recentSpacesRevision}
          />
        )}

        {/* Main content */}
        <main ref={mainContentRef} className={`flex-1 ${view.kind === "graph" || view.kind === "sources" ? "min-w-0 overflow-hidden p-0" : "memory-main-content overflow-y-auto"}`}>
          {/* Search results overlay */}
          {query ? (
            (memoryResults.length > 0 || sourceResults.length > 0 || entityResults.length > 0 || conceptResults.length > 0) ? (
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
                      {t("main.search.pages")}
                    </p>
                    {conceptResults.map((c) => (
                      <div
                        key={c.id}
                        className="rounded-lg px-4 py-3 cursor-pointer transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                        style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
                        onClick={() => { setQuery(""); navigateTo({ kind: "page", pageId: c.id }); }}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: "var(--mem-accent-page)" }} />
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
                {memoryResults.length > 0 && (
                  <>
                    <p
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                        marginTop: conceptResults.length > 0 ? 12 : 0,
                      }}
                    >
                      {t("main.search.memories", { count: memoryResults.length, query })}
                    </p>
                    {memoryResults.map((r) => (
                      <MemorySearchResult key={r.id} result={r} query={query} onClick={() => void openSearchResult(r)} />
                    ))}
                  </>
                )}
                {sourceResults.length > 0 && (
                  <>
                    <p
                      style={{
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: "11px",
                        color: "var(--mem-text-tertiary)",
                        marginTop: (memoryResults.length > 0 || conceptResults.length > 0) ? 12 : 0,
                      }}
                    >
                      {t("main.search.sources")}
                    </p>
                    {sourceResults.map((r) => (
                      <MemorySearchResult key={r.id} result={r} query={query} onClick={() => void openSearchResult(r)} />
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
                        marginTop: (memoryResults.length > 0 || sourceResults.length > 0 || conceptResults.length > 0) ? 12 : 0,
                      }}
                    >
                      {t("main.search.entities")}
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
                            style={{ backgroundColor: "color-mix(in srgb, var(--mem-accent-sage) 15%, transparent)", color: "var(--mem-accent-sage)" }}
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
                {t("main.search.noResults", { query })}
              </p>
            )
          ) : view.kind === "import" ? (
            <ImportView
              onBack={navigateBack}
              onComplete={(_source, _result) => { setView({ kind: "stream" }); setViewHistory([]); }}
            />
          ) : view.kind === "settings" ? (
            <SettingsPage
              section={view.section ?? "general"}
              onBack={navigateBack}
              onSetupAgent={() => navigateTo({ kind: "connect-agent" })}
              onImport={() => navigateTo({ kind: "import" })}
            />
          ) : view.kind === "decisions" ? (
            <DecisionLog
              onBack={navigateBack}
              onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onSelectPage={(id) => navigateTo({ kind: "page", pageId: id })}
            />
          ) : view.kind === "pages" ? (
            <PagesOverview
              onCreatePage={(space) => navigateTo({ kind: "page-draft", space })}
              onSelectDraft={(draftId, space) => navigateTo({
                kind: "page-draft",
                draftId,
                space,
              })}
              onSelectPage={(id) => navigateTo({ kind: "page", pageId: id })}
              onSelectSpace={(spaceName) => navigateTo({ kind: "space", spaceId: null, spaceName })}
            />
          ) : view.kind === "spaces" ? (
            <SpacesOverview
              createIntent={view.create}
              labels={spacesOverviewLabels}
              onCreateIntentHandled={() => setView({ kind: "spaces" })}
              onSelectSpace={(spaceName) => navigateTo({ kind: "space", spaceId: null, spaceName })}
              onSpaceDeleted={handleSpaceDeleted}
              onSpaceRenamed={handleSpaceRenamed}
            />
          ) : view.kind === "space" ? (
            <SpaceDetail
              copy={spaceDetailCopy}
              spaceName={view.spaceName}
              onBack={() => setView({ kind: "spaces" })}
              onCreatePage={(space) => navigateTo({ kind: "page-draft", space })}
              onReviewAll={() => navigateTo({ kind: "distill-review" })}
              onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onSelectPage={(id) => navigateTo({ kind: "page", pageId: id })}
              onEntityClick={handleEntityClick}
              onSpaceDeleted={handleSpaceDeleted}
              onSpaceLoaded={handleSpaceLoaded}
              onSpaceRenamed={handleSpaceRenamed}
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
            <EntityDetail
              key={view.entityId}
              entityId={view.entityId}
              onBack={navigateBack}
              onEntityClick={handleEntityClick}
              onMemoryClick={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "page-draft" ? (
            <PageDraftEditor
              draftId={view.draftId}
              onBack={() => {
                if (responsiveSidebar.presentation === "overlay" && responsiveSidebar.open) {
                  responsiveSidebar.close();
                  return;
                }
                navigateBack();
              }}
              onEscapeBeforeLeave={() => {
                if (
                  responsiveSidebar.presentation === "overlay"
                  && responsiveSidebar.open
                ) {
                  responsiveSidebar.close();
                  return true;
                }
                return false;
              }}
              onOpenExisting={(pageId) => {
                afterPageDraftFlush(() => setView({ kind: "page", pageId }));
              }}
              onPublished={(pageId) => setView({ kind: "page", pageId })}
              ref={pageDraftEditorRef}
              space={view.space}
            />
          ) : view.kind === "page" ? (
            <PageDetail
              pageId={view.pageId}
              onBack={navigateBack}
              onMemoryClick={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onPageLoaded={handlePageLoaded}
              onPageClick={(id) => navigateTo({ kind: "page", pageId: id })}
              onEntityClick={handleEntityClick}
            />
          ) : view.kind === "distill-review" ? (
            <DistillReviewPanel
              onBack={navigateBack}
              onPageClick={(id) => navigateTo({ kind: "page", pageId: id })}
              onMemoryClick={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
            />
          ) : view.kind === "home" ? (
            <HomePage
              onNavigateMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
              onNavigateStream={() => navigateTo({ kind: "recaps" })}
              onNavigateLog={() => navigateTo({ kind: "stream" })}
              onNavigateGraph={() => navigateTo({ kind: "graph" })}
              onSelectPage={(id) => navigateTo({ kind: "page", pageId: id })}
              onOpenDistillReview={() => navigateTo({ kind: "distill-review" })}
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
          ) : view.kind === "sources" ? (
            <SourcesView
              onManageSources={() => navigateTo({ kind: "settings", section: "sources" })}
            />
          ) : view.kind === "graph" ? (
            <div style={{ position: "relative", width: "100%", height: "100%" }}>
              <AtlasView onNodeClick={handleEntityClick} onBack={navigateBack} />
            </div>
          ) : (
            <>
              <button onClick={() => navigateTo({ kind: "home" })} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] mb-3" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
              </button>
              <h2 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "24px", fontWeight: 500, color: "var(--mem-text)", margin: "0 0 12px 0" }}>{t("main.memories")}</h2>
              <MemoryStream
                memories={memories}
                selectedDomain={null}
                sortMode={sortMode}
                onSortChange={setSortMode}
                stabilityFilter={stabilityFilter}
                onStabilityFilterChange={setStabilityFilter}
                onSelectMemory={(sid) => navigateTo({ kind: "memory", sourceId: sid })}
                presentation="parent-list"
              />
            </>
          )}
        </main>
      </div>

      <AboutWenlanDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />

      {/* Status bar */}
      <MemoryStatusBar message={statusMessage} />
    </div>
  );
}
