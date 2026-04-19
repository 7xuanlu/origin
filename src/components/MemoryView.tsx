// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect, useRef, useCallback, useMemo, useSyncExternalStore } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import {
  listIndexedFiles,
  listActivities,
  deleteFileChunks,
  deleteByTimeRange,
  deleteBulk,
  getCaptureStats,
  listAllTags,
  setDocumentSpace,
  openFile,
  clipboardWrite,
  rebuildActivities,
  listMemoriesRich,
  FACET_COLORS,
  type IndexedFileInfo,
  type ActivitySummary,
  type MemoryItem,
} from "../lib/tauri";
import { useSearch } from "../hooks/useSearch";
import TagEditor from "./TagEditor";
import QuickCapture from "./QuickCapture";
import ResultList from "./ResultList";
import SpaceIcon from "./SpaceIcon";
import { isProcessing, subscribe, getSnapshot } from "../lib/processingStore";
import { subscribe as heartbeatSubscribe, getLastCapture } from "../lib/captureHeartbeat";
import { getCollapsedSections, toggleSection, expandSection } from "../lib/collapsedSections";
import ViewToggle from "./ViewToggle";
import ProfilePage from "./memory/ProfilePage";
import SettingsPage from "./memory/SettingsPage";
import Sidebar, { SidebarToggleButton } from "./memory/Sidebar";

const SOURCE_LABELS: Record<string, string> = {
  local_files: "File",
  clipboard: "Clipboard",
  manual: "Capture",
  screen_capture: "Screen",
  ambient: "Ambient",
  focus_capture: "Focus",
  hotkey_capture: "Capture",
  snip_capture: "Snip",
  quick_thought: "Thought",
  webpage: "Webpage",
  context: "Capture",
  memory: "Memory",
};

const SOURCE_COLORS: Record<string, string> = {
  local_files: "bg-blue-500/15 text-blue-400",
  clipboard: "bg-amber-500/15 text-amber-400",
  manual: "bg-purple-500/15 text-purple-400",
  screen_capture: "bg-green-500/15 text-green-400",
  ambient: "bg-teal-500/15 text-teal-400",
  focus_capture: "bg-amber-500/15 text-amber-400",
  hotkey_capture: "bg-purple-500/15 text-purple-400",
  snip_capture: "bg-rose-500/15 text-rose-400",
  quick_thought: "bg-pink-500/15 text-pink-400",
  webpage: "bg-cyan-500/15 text-cyan-400",
  context: "bg-green-500/15 text-green-400",
  memory: "bg-violet-500/15 text-violet-400",
};

const SPACE_COLOR_MAP: Record<string, string> = {
  sky: "bg-sky-500/15 text-sky-400",
  pink: "bg-pink-500/15 text-pink-400",
  violet: "bg-violet-500/15 text-violet-400",
  orange: "bg-orange-500/15 text-orange-400",
  rose: "bg-rose-500/15 text-rose-400",
  zinc: "bg-zinc-500/15 text-zinc-400",
  gray: "bg-zinc-500/10 text-zinc-500",
};


type TimePeriod = "Today" | "Yesterday" | "This Week" | "Older";

/** Return the start/end boundaries (epoch seconds) for each time period. */
function getTimePeriodBounds(): Record<TimePeriod, { start: number; end: number }> {
  const now = new Date();
  const nowSec = Math.floor(now.getTime() / 1000);
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const yesterdayStart = todayStart - 86400;
  const dayOfWeek = now.getDay();
  const daysSinceMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const weekStart = todayStart - daysSinceMonday * 86400;

  return {
    Today: { start: todayStart, end: nowSec },
    Yesterday: { start: yesterdayStart, end: todayStart - 1 },
    "This Week": { start: weekStart, end: yesterdayStart - 1 },
    Older: { start: 0, end: weekStart - 1 },
  };
}

/** Group items into time-period buckets based on a timestamp extractor. */
function groupByTimePeriod<T>(
  items: T[],
  getTimestamp: (item: T) => number,
): { period: TimePeriod; items: T[] }[] {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const yesterdayStart = todayStart - 86400;
  // Start of this week (Monday)
  const dayOfWeek = now.getDay();
  const daysSinceMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const weekStart = todayStart - daysSinceMonday * 86400;

  const buckets: Record<TimePeriod, T[]> = {
    Today: [],
    Yesterday: [],
    "This Week": [],
    Older: [],
  };

  for (const item of items) {
    const ts = getTimestamp(item);
    if (ts >= todayStart) {
      buckets.Today.push(item);
    } else if (ts >= yesterdayStart) {
      buckets.Yesterday.push(item);
    } else if (ts >= weekStart) {
      buckets["This Week"].push(item);
    } else {
      buckets.Older.push(item);
    }
  }

  const order: TimePeriod[] = ["Today", "Yesterday", "This Week", "Older"];
  return order
    .filter((p) => buckets[p].length > 0)
    .map((p) => ({ period: p, items: buckets[p] }));
}

/** Format a timestamp for display within a time period context. */
function formatActivityTime(ts: number, period: TimePeriod): string {
  const date = new Date(ts * 1000);
  const time = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  switch (period) {
    case "Today":
    case "Yesterday":
      return time;
    case "This Week":
      return `${date.toLocaleDateString([], { weekday: "short" })} ${time}`;
    case "Older":
      return `${date.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
  }
}

/** Fallback: convert internal snake_case source to title case for display. */
function formatSourceLabel(source: string): string {
  return source
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const HEARTBEAT_COLORS: Record<string, string> = {
  ambient: "#2dd4bf",
  focus: "#fbbf24",
  focus_capture: "#fbbf24",
  hotkey: "#c084fc",
  hotkey_capture: "#c084fc",
  snip: "#fb7185",
  snip_capture: "#fb7185",
  clipboard: "#4ade80",
  quick_thought: "#f472b6",
};

const HEARTBEAT_LABELS: Record<string, string> = {
  ambient: "Ambient",
  focus: "Focus",
  focus_capture: "Focus",
  hotkey: "Capture",
  hotkey_capture: "Capture",
  snip: "Snip",
  snip_capture: "Snip",
  clipboard: "Clipboard",
  quick_thought: "Thought",
};

function formatRelativeTime(ms: number): string {
  const secs = Math.floor(ms / 1000);
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
}

function CaptureHeartbeat() {
  const last = useSyncExternalStore(heartbeatSubscribe, getLastCapture);
  const [, tick] = useState(0);

  // Tick every second to update relative time
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  // Pulse animation on new capture
  const [pulse, setPulse] = useState(false);
  const prevTimestamp = useRef<number | null>(null);
  useEffect(() => {
    if (last && last.timestamp !== prevTimestamp.current) {
      prevTimestamp.current = last.timestamp;
      setPulse(true);
      const id = setTimeout(() => setPulse(false), 600);
      return () => clearTimeout(id);
    }
  }, [last]);

  if (!last) return null;

  const elapsed = Date.now() - last.timestamp;
  const isIdle = elapsed > 60_000;
  const color = HEARTBEAT_COLORS[last.source] ?? "#a1a1aa";
  const label = HEARTBEAT_LABELS[last.source] ?? last.source;

  return (
    <div className="flex items-center gap-1.5 mr-2" title={`Last capture: ${label}`}>
      <span
        className="w-1.5 h-1.5 rounded-full shrink-0 transition-opacity"
        style={{
          backgroundColor: isIdle ? "var(--text-tertiary)" : color,
          opacity: isIdle ? 0.4 : 1,
          animation: pulse ? "heartbeat-pulse 0.6s ease-out" : "none",
        }}
      />
      <span className="text-[11px] text-[var(--text-tertiary)]">
        {isIdle ? "Idle" : `${label} \u00B7 ${formatRelativeTime(elapsed)}`}
      </span>
      <style>{`
        @keyframes heartbeat-pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(2); opacity: 0.6; }
          100% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

interface MemoryViewProps {
  onBack: () => void;
  onSelectFile: (file: IndexedFileInfo) => void;
  onSelectRecap: (recap: IndexedFileInfo) => void;
  onSelectMemory?: (sourceId: string) => void;
  onImport?: () => void;
}

const SIDEBAR_KEY = "origin-sidebar-collapsed";

export default function MemoryView({ onBack, onSelectFile, onSelectRecap, onSelectMemory, onImport }: MemoryViewProps) {
  const queryClient = useQueryClient();
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem(SIDEBAR_KEY) === "true"; } catch { return false; }
  });
  const [_selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const toggleSidebar = () => {
    setSidebarCollapsed((v) => {
      const next = !v;
      try { localStorage.setItem(SIDEBAR_KEY, String(next)); } catch {}
      return next;
    });
  };
  const [showProfile, setShowProfile] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [activeSpaceFilter, _setActiveSpaceFilter] = useState<string | null>(null);
  const [activeTagFilter, setActiveTagFilter] = useState<string | null>(null);
  const [sourceTypeFilter, setSourceTypeFilter] = useState<string | null>(null);
  const [activitySortMode, setActivitySortMode] = useState<"recent" | "oldest">("recent");
  const [actSortOpen, setActSortOpen] = useState(false);
  const [actFilterOpen, setActFilterOpen] = useState(false);
  const actSortRef = useRef<HTMLDivElement>(null);
  const actFilterRef = useRef<HTMLDivElement>(null);
  const [editingTagsFor, setEditingTagsFor] = useState<string | null>(null);
  const [editingSpaceFor, setEditingSpaceFor] = useState<string | null>(null);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(getCollapsedSections);
  const [showQuickCapture, setShowQuickCapture] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectedRecapIds, setSelectedRecapIds] = useState<Set<string>>(new Set());
  const [confirmAction, setConfirmAction] = useState<{
    label: string;
    onConfirm: () => void;
  } | null>(null);
  const { query, setQuery, results, isLoading } = useSearch();
  const searchInputRef = useRef<HTMLInputElement>(null);

  const hasSelection = selectedIds.size > 0 || selectedRecapIds.size > 0;
  const totalSelected = selectedIds.size + selectedRecapIds.size;

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    setSelectedRecapIds(new Set());
  }, []);

  // Reset selection when search results change
  useEffect(() => { setSelectedIndex(0); }, [results]);

  // Close sort/filter dropdowns on outside click
  useEffect(() => {
    if (!actSortOpen && !actFilterOpen) return;
    const handler = (e: MouseEvent) => {
      if (actSortOpen && actSortRef.current && !actSortRef.current.contains(e.target as Node)) setActSortOpen(false);
      if (actFilterOpen && actFilterRef.current && !actFilterRef.current.contains(e.target as Node)) setActFilterOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [actSortOpen, actFilterOpen]);

  // Escape key clears selection or dismisses confirmation
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        if (confirmAction) setConfirmAction(null);
        else if (hasSelection) clearSelection();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [hasSelection, confirmAction, clearSelection]);

  const showToast = useCallback((message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 2000);
  }, []);

  const openResult = useCallback(async (index: number) => {
    const result = results[index];
    if (!result?.url) return;
    try {
      await openFile(result.url);
      showToast("Opened file");
    } catch (err) {
      showToast(`Failed to open file: ${err}`);
    }
  }, [results, showToast]);

  const copyResult = useCallback(async (index: number) => {
    const result = results[index];
    if (!result) return;
    try {
      const text = `${result.title}\n\n${result.content}${result.url ? `\n\nSource: ${result.url}` : ""}`;
      await clipboardWrite(text);
      showToast("Copied to clipboard");
    } catch (err) {
      showToast(`Failed to copy: ${err}`);
    }
  }, [results, showToast]);

  const handleSearchKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        if (e.metaKey || e.ctrlKey) {
          // copy all
        } else {
          openResult(selectedIndex);
        }
        break;
      case "Escape":
        e.preventDefault();
        setQuery("");
        searchInputRef.current?.blur();
        break;
    }
  }, [results.length, selectedIndex, openResult, setQuery]);

  // pointermove on the document is the only reliable way to track hover in
  // WebKit/Tauri — mouseleave doesn't fire when the DOM changes mid-hover
  // (e.g. during the 5-second refetch). Walk up from the pointer target to
  // find the nearest [data-row-id] ancestor; clear if none found.
  useEffect(() => {
    function clearHover() {
      setHoveredId(null);
    }

    function handlePointerMove(e: PointerEvent) {
      let el: Element | null = document.elementFromPoint(e.clientX, e.clientY);
      while (el) {
        const id = el.getAttribute("data-row-id");
        if (id) {
          setHoveredId((prev) => (prev === id ? prev : id));
          return;
        }
        el = el.parentElement;
      }
      clearHover();
    }

    document.addEventListener("pointermove", handlePointerMove);
    document.documentElement.addEventListener("pointerleave", clearHover);
    window.addEventListener("blur", clearHover);
    document.addEventListener("visibilitychange", clearHover);

    return () => {
      document.removeEventListener("pointermove", handlePointerMove);
      document.documentElement.removeEventListener("pointerleave", clearHover);
      window.removeEventListener("blur", clearHover);
      document.removeEventListener("visibilitychange", clearHover);
    };
  }, []);

  const { data: indexedFiles = [] } = useQuery({
    queryKey: ["indexedFiles"],
    queryFn: listIndexedFiles,
    refetchInterval: 5000,
  });

  const { data: activities = [] } = useQuery({
    queryKey: ["activities"],
    queryFn: listActivities,
    refetchInterval: 5000,
  });

  // Imported memories — query as MemoryItem for time-period grouping
  const { data: allMemories = [] } = useQuery({
    queryKey: ["memories"],
    queryFn: () => listMemoriesRich(),
    refetchInterval: 5000,
  });

  // Auto-rebuild activities when most captures are orphaned (no matching activity)
  const [rebuilding, setRebuilding] = useState(false);
  useEffect(() => {
    const captureFiles = indexedFiles.filter(f => f.source !== "local_files");
    if (captureFiles.length < 5 || rebuilding) return;
    // Quick check: how many captures fall within any activity's range?
    const grouped = captureFiles.filter(f =>
      activities.some(a =>
        f.last_modified >= a.started_at - 10 &&
        (a.is_live || f.last_modified <= a.ended_at + 10)
      )
    );
    // If >60% of captures are orphaned, rebuild activity boundaries
    if (grouped.length < captureFiles.length * 0.4) {
      setRebuilding(true);
      rebuildActivities()
        .then(() => {
          queryClient.invalidateQueries({ queryKey: ["activities"] });
        })
        .catch((e) => console.error("[MemoryView] rebuild failed:", e))
        .finally(() => setRebuilding(false));
    }
  }, [activities, indexedFiles, rebuilding, queryClient]);

  // Auto-expand live activities (remove from collapsed set + persist)
  useEffect(() => {
    const liveIds = activities.filter((a: ActivitySummary) => a.is_live).map((a: ActivitySummary) => a.id);
    if (liveIds.length === 0) return;
    setCollapsedGroups((prev) => {
      const toExpand = liveIds.filter((id: string) => prev.has(id));
      if (toExpand.length === 0) return prev;
      for (const id of toExpand) expandSection(id);
      const next = new Set(prev);
      for (const id of toExpand) next.delete(id);
      return next;
    });
  }, [activities]);

  // Invalidate queries on capture events (processing store fed by App-level listener)
  useEffect(() => {
    const unlisten = listen("capture-event", () => {
      queryClient.invalidateQueries({ queryKey: ["indexedFiles"] });
      queryClient.invalidateQueries({ queryKey: ["activities"] });
      queryClient.invalidateQueries({ queryKey: ["captureStats"] });
    });
    return () => { unlisten.then((f) => f()); };
  }, [queryClient]);

  // Subscribe to processing store changes so file cards re-render
  useSyncExternalStore(subscribe, getSnapshot);

  // Recaps live on Home only — Activity view shows captures/files, not recaps
  const recaps: typeof indexedFiles = [];

  const { data: captureStats } = useQuery({
    queryKey: ["captureStats"],
    queryFn: getCaptureStats,
    refetchInterval: 5000,
  });

  const { data: tagData } = useQuery({
    queryKey: ["tags"],
    queryFn: listAllTags,
    refetchInterval: 5000,
  });

  // Space color lookup (legacy — old space model removed, always returns default)
  function getSpaceColor(_spaceId: string): string {
    return SPACE_COLOR_MAP.zinc;
  }

  const invalidateAfterDelete = () => {
    queryClient.invalidateQueries({ queryKey: ["indexedFiles"] });
    queryClient.invalidateQueries({ queryKey: ["activities"] });
    queryClient.invalidateQueries({ queryKey: ["tags"] });
    queryClient.invalidateQueries({ queryKey: ["captureStats"] });
  };

  const deleteChunksMutation = useMutation({
    mutationFn: ({ source, sourceId }: { source: string; sourceId: string }) =>
      deleteFileChunks(source, sourceId),
    onSuccess: invalidateAfterDelete,
  });

  const deleteTimeRangeMutation = useMutation({
    mutationFn: ({ start, end }: { start: number; end: number }) =>
      deleteByTimeRange(start, end),
    onMutate: () => setConfirmAction(null),
    onSettled: invalidateAfterDelete,
  });

  const deleteBulkMutation = useMutation({
    mutationFn: (items: { source: string; sourceId: string }[]) =>
      deleteBulk(items),
    onMutate: () => {
      clearSelection();
      setConfirmAction(null);
    },
    onSettled: invalidateAfterDelete,
  });

  const isDeleting =
    deleteChunksMutation.isPending ||
    deleteTimeRangeMutation.isPending ||
    deleteBulkMutation.isPending;

  const setSpaceMutation = useMutation({
    mutationFn: ({ source, sourceId, spaceId }: { source: string; sourceId: string; spaceId: string }) =>
      setDocumentSpace(source, sourceId, spaceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      queryClient.invalidateQueries({ queryKey: ["tags"] });
    },
  });

  function matchesFilter(file: IndexedFileInfo): boolean {
    // Space filter
    if (activeSpaceFilter) {
      const key = `${file.source}::${file.source_id}`;
      const docSpace = ({} as Record<string, string>)[key];
      if (docSpace !== activeSpaceFilter) return false;
    }
    // Tag section filter (new browseable Tags section)
    if (activeTagFilter) {
      const key = `${file.source}::${file.source_id}`;
      const docTags = tagData?.document_tags[key] ?? [];
      if (!docTags.includes(activeTagFilter)) return false;
    }
    // Source type filter
    if (sourceTypeFilter) {
      if (file.source !== sourceTypeFilter) return false;
    }
    return true;
  }

  const { grouped, filteredUngrouped } = useMemo(() => {
    // Small buffer (seconds) to absorb timing gaps between document
    // timestamps (set at bundle creation) and activity boundaries (set
    // after the upsert completes).  Prevents the first capture of each
    // new session from becoming orphaned.
    const GROUPING_BUFFER = 10;
    // Max distance (seconds) for snapping orphaned captures to the
    // nearest activity in the second pass.  Matches ACTIVITY_GAP_SECS
    // so captures that lost their activity (restart, migration) still
    // get grouped.
    const SNAP_LIMIT = 1800;

    const grouped = activities.map((a: ActivitySummary) => ({
      activity: a,
      children: [] as typeof indexedFiles,
    }));

    // Pass 1: match files to activities within a tight buffer
    const orphaned: typeof indexedFiles = [];
    const ungrouped: typeof indexedFiles = [];
    for (const file of indexedFiles) {
      // Local files go to "Other"; imported memories skip — rendered via allMemories query
      if (file.source === "memory") continue;
      if (file.source === "local_files") {
        ungrouped.push(file);
        continue;
      }
      const match = grouped.find(
        g => file.last_modified >= g.activity.started_at - GROUPING_BUFFER &&
             (g.activity.is_live || file.last_modified <= g.activity.ended_at + GROUPING_BUFFER)
      );
      if (match) match.children.push(file);
      else orphaned.push(file);
    }

    // Pass 2: snap remaining captures to the nearest activity (within
    // SNAP_LIMIT).  Handles documents whose original activity was lost
    // due to app restarts or migration.
    for (const file of orphaned) {
      let nearest: typeof grouped[number] | null = null;
      let minDist = Infinity;
      for (const g of grouped) {
        const dStart = Math.abs(file.last_modified - g.activity.started_at);
        const dEnd = g.activity.is_live ? dStart : Math.abs(file.last_modified - g.activity.ended_at);
        const dist = Math.min(dStart, dEnd);
        if (dist < minDist) { minDist = dist; nearest = g; }
      }
      if (nearest && minDist <= SNAP_LIMIT) nearest.children.push(file);
      else ungrouped.push(file);
    }

    // Apply filters and sort recaps first within each group
    for (const group of grouped) {
      group.children = group.children.filter(matchesFilter);
      group.children.sort((a, b) => {
        const aRecap = (a as any).is_recap ? 1 : 0;
        const bRecap = (b as any).is_recap ? 1 : 0;
        if (aRecap !== bRecap) return bRecap - aRecap; // recaps first
        return b.last_modified - a.last_modified;
      });
    }
    const filteredUngrouped = ungrouped.filter(matchesFilter);

    return { grouped, filteredUngrouped };
  }, [indexedFiles, activities, activeSpaceFilter, activeTagFilter, sourceTypeFilter, tagData]);

  function toggleSelected(sourceId: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(sourceId)) next.delete(sourceId);
      else next.add(sourceId);
      return next;
    });
  }

  function toggleRecapSelected(snapshotId: string) {
    setSelectedRecapIds((prev) => {
      const next = new Set(prev);
      if (next.has(snapshotId)) next.delete(snapshotId);
      else next.add(snapshotId);
      return next;
    });
  }

  function handleBulkDelete() {
    const fileItems = indexedFiles
      .filter((f) => selectedIds.has(f.source_id))
      .map((f) => ({ source: f.source, sourceId: f.source_id }));
    const recapCount = selectedRecapIds.size;
    const total = fileItems.length + recapCount;
    if (total === 0) return;

    const parts: string[] = [];
    if (fileItems.length > 0) parts.push(`${fileItems.length} file${fileItems.length !== 1 ? "s" : ""}`);
    if (recapCount > 0) parts.push(`${recapCount} recap${recapCount !== 1 ? "s" : ""}`);

    setConfirmAction({
      label: `Delete ${parts.join(" and ")}?`,
      onConfirm: async () => {
        const recapItems = recaps
          .filter((r) => selectedRecapIds.has(r.source_id))
          .map((r) => ({ source: r.source, sourceId: r.source_id }));
        const allItems = [...fileItems, ...recapItems];
        if (allItems.length > 0) deleteBulkMutation.mutate(allItems);
      },
    });
  }

  function handleActivityDelete(activity: ActivitySummary, children: IndexedFileInfo[]) {
    // Expand the delete range to cover ALL files grouped under this
    // activity (including orphans snapped via SNAP_LIMIT).
    const childTimes = children.map(c => c.last_modified);
    const start = Math.min(activity.started_at, ...childTimes);
    const end = Math.max(activity.ended_at, ...childTimes);
    setConfirmAction({
      label: `Delete session (${formatActivityTime(activity.started_at, "Today")} \u2013 ${formatActivityTime(activity.ended_at, "Today")})?`,
      onConfirm: () => deleteTimeRangeMutation.mutate({ start, end }),
    });
  }

  function handleTimePeriodDelete(period: TimePeriod, itemCount: number) {
    const bounds = getTimePeriodBounds();
    const { start, end } = bounds[period];
    setConfirmAction({
      label: `Delete all ${itemCount} item${itemCount !== 1 ? "s" : ""} from "${period}"?`,
      onConfirm: () => deleteTimeRangeMutation.mutate({ start, end }),
    });
  }

  function renderFileCard(file: typeof indexedFiles[number]) {
    const isHovered = hoveredId === file.source_id;
    const isSelected = selectedIds.has(file.source_id);
    const docTagKey = `${file.source}::${file.source_id}`;
    const fileTags = tagData?.document_tags[docTagKey] ?? [];
    const isEditing = editingTagsFor === file.source_id;
    const isEditingSpace = editingSpaceFor === file.source_id;
    const fileProcessing = file.processing || isProcessing(file.source_id);
    const showActions = isHovered || isSelected;
    return (
      <div
        key={file.source_id}
        data-row-id={file.source_id}
        className={`relative bg-[var(--bg-secondary)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.2)] p-4 min-h-[110px] flex flex-col justify-between transition-colors ${isHovered ? "bg-[var(--overlay-subtle)]" : ""} ${isSelected ? "ring-1 ring-[var(--accent)]" : ""}`}
      >
        {/* Top-right: select checkbox */}
        <button
          onClick={(e) => { e.stopPropagation(); toggleSelected(file.source_id); }}
          className={`absolute top-2.5 right-2.5 z-10 flex items-center justify-center p-0.5 transition-opacity ${showActions ? "opacity-100" : "opacity-0"}`}
        >
          <div
            className={`w-[16px] h-[16px] rounded border transition-colors flex items-center justify-center ${
              isSelected
                ? "bg-[var(--accent)] border-[var(--accent)]"
                : "border-[var(--text-tertiary)]/60 hover:border-[var(--text-secondary)]"
            }`}
          >
            {isSelected && (
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>
        </button>
        <button onClick={() => onSelectMemory ? onSelectMemory(file.source_id) : onSelectFile(file)} className="min-w-0 text-left pr-12">
          <span className="text-[14px] font-medium leading-snug text-[var(--text-primary)] line-clamp-2 hover:text-[var(--accent)] transition-colors">
            {file.title}
          </span>
        </button>
        {/* Tag pills */}
        {fileTags.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap mt-1.5">
            {fileTags.map((tag) => (
              <span
                key={tag}
                className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-[var(--mem-accent-indigo)]/15 text-[var(--mem-accent-indigo)]"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        {/* AI processing indicator */}
        {fileProcessing && (
          <div className="flex items-center gap-1.5 mt-2">
            <span className="w-1.5 h-1.5 bg-[var(--mem-accent-indigo)] rounded-full animate-pulse" />
            <span className="text-[10px] text-[var(--mem-text-tertiary)]">AI processing</span>
          </div>
        )}
        <div className="flex items-end justify-between mt-3">
          <div className="flex items-center gap-2">
            <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${SOURCE_COLORS[file.source] ?? "bg-zinc-500/15 text-zinc-400"}`}>
              {SOURCE_LABELS[file.source] ?? formatSourceLabel(file.source)}
            </span>
            {file.memory_type && (
              <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${FACET_COLORS[file.memory_type] ?? "bg-zinc-500/15 text-zinc-400"}`}>
                {file.memory_type}
              </span>
            )}
            {/* Space badge */}
            {(() => {
              const key = `${file.source}::${file.source_id}`;
              const spaceId = ({} as Record<string, string>)[key];
              const space = ([] as { id: string; name: string; icon: string; color: string }[]).find((s) => s.id === spaceId);
              if (!space) return null;
              return (
                <span
                  className={`relative inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded cursor-pointer ${getSpaceColor(space.id)}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditingSpaceFor(isEditingSpace ? null : file.source_id);
                  }}
                >
                  <SpaceIcon icon={space.icon} size={11} />
                  {space.name}
                </span>
              );
            })()}
            <span className="text-[11px] text-[var(--mem-text-tertiary)]">
              {file.chunk_count} chunk{file.chunk_count !== 1 ? "s" : ""}
            </span>
          </div>
          {/* Tag button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setEditingTagsFor(isEditing ? null : file.source_id);
            }}
            className={`shrink-0 text-[var(--mem-text-tertiary)] hover:text-[var(--mem-accent-indigo)] transition-opacity ${isHovered || isEditing ? "opacity-70 hover:opacity-100" : "opacity-0"}`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A2 2 0 013 12V7a4 4 0 014-4z" />
            </svg>
          </button>
        </div>
        {/* Tag editor popover */}
        {isEditing && (
          <TagEditor
            source={file.source}
            sourceId={file.source_id}
            lastModified={file.last_modified}
            currentTags={fileTags}
            allTags={tagData?.tags ?? []}
            onClose={() => setEditingTagsFor(null)}
            onTagsChanged={() => queryClient.invalidateQueries({ queryKey: ["tags"] })}
          />
        )}
        {/* Space override dropdown */}
        {isEditingSpace && (
          <SpaceDropdown
            spaces={[] as { id: string; name: string; icon: string; color: string }[]}
            current={({} as Record<string, string>)[`${file.source}::${file.source_id}`]}
            onSelect={(spaceId) => {
              setSpaceMutation.mutate({
                source: file.source,
                sourceId: file.source_id,
                spaceId,
              });
              setEditingSpaceFor(null);
            }}
            onClose={() => setEditingSpaceFor(null)}
          />
        )}
      </div>
    );
  }

  const isRenderable = (_f: typeof indexedFiles[number]) => true; // All sources renderable — recaps are first-class memories
  const totalCount = indexedFiles.filter(isRenderable).length;
  const hasActiveFilters = activeSpaceFilter !== null || activeTagFilter !== null || sourceTypeFilter !== null;
  const mergedTimePeriods = (() => {
    const periods = groupByTimePeriod(
      grouped.filter(g => g.children.some(isRenderable) || (g.activity.is_live && !hasActiveFilters)),
      g => g.activity.started_at,
    ).map(({ period, items }) => ({ period, periodItems: items, memories: [] as MemoryItem[] }));
    // Bucket memories into matching time periods
    const memoryPeriods = groupByTimePeriod(allMemories, m => m.last_modified);
    for (const { period, items } of memoryPeriods) {
      const existing = periods.find(p => p.period === period);
      if (existing) {
        existing.memories = items;
      } else {
        periods.push({ period, periodItems: [], memories: items });
      }
    }
    const order: TimePeriod[] = ["Today", "Yesterday", "This Week", "Older"];
    periods.sort((a, b) => order.indexOf(a.period) - order.indexOf(b.period));
    if (activitySortMode === "oldest") periods.reverse();
    return periods;
  })();

  return (
    <div
      className="w-full h-screen flex flex-col"
      style={{
        backgroundColor: "var(--mem-bg)",
        color: "var(--mem-text)",
        // Bridge old CSS vars to --mem-*
        "--bg-primary": "var(--mem-bg)",
        "--bg-secondary": "var(--mem-surface)",
        "--text-primary": "var(--mem-text)",
        "--text-secondary": "var(--mem-text-secondary)",
        "--text-tertiary": "var(--mem-text-tertiary)",
        "--accent": "var(--mem-accent-indigo)",
        "--separator": "var(--mem-border)",
        "--overlay-subtle": "var(--mem-hover)",
        "--overlay-hover": "var(--mem-hover-strong)",
      } as React.CSSProperties}
    >
      {/* Full-width header */}
      <div
        className="relative flex items-center gap-3 shrink-0"
        style={{
          height: 52,
          borderBottom: "1px solid var(--mem-border)",
          paddingLeft: 82,
          paddingRight: 20,
        }}
        data-tauri-drag-region
      >
        <SidebarToggleButton collapsed={sidebarCollapsed} onToggle={toggleSidebar} />
        <div className="flex-1" data-tauri-drag-region />
        {/* Right actions */}
        <CaptureHeartbeat />
        {/* Right actions — same structure as MemoryPage: ViewToggle + settings */}
        <div className="flex items-center gap-2 shrink-0">
          <ViewToggle active="activity" onSwitch={(v) => { if (v !== "activity") onBack(); }} />
          <button
            onClick={() => setShowSettings(true)}
            className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
            style={{ color: "var(--mem-text-secondary)" }}
            title="Settings"
          >
            <svg className="w-[16px] h-[16px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.11 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>

        {/* Search + Add — absolutely centered, same position in both modes */}
        <div
          className="absolute flex items-center gap-2"
          style={{ left: "50%", transform: "translateX(-50%)" }}
        >
          <button
            onClick={() => setShowQuickCapture(true)}
            className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
            style={{ color: "var(--mem-text-secondary)" }}
            title="Quick Capture"
          >
            <svg className="w-[16px] h-[16px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          {onImport && (
            <button
              onClick={onImport}
              className="p-1.5 rounded-md hover:bg-[var(--mem-hover-strong)] text-[var(--mem-text-secondary)] hover:text-[var(--mem-text)]"
              title="Import Memories"
            >
              <svg className="w-[16px] h-[16px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </button>
          )}
          <div
            className="flex items-center gap-2 rounded-md px-3 py-[7px]"
            style={{
              width: 480,
              backgroundColor: "var(--mem-sidebar)",
              border: "1px solid var(--mem-border)",
            }}
            onKeyDown={handleSearchKeyDown}
          >
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ color: "var(--mem-text-tertiary)" }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={searchInputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search activity..."
              className="flex-1 bg-transparent outline-none"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text)",
              }}
              spellCheck={false}
              autoComplete="off"
            />
            {isLoading && (
              <div className="w-3.5 h-3.5 border-2 rounded-full animate-spin shrink-0" style={{ borderColor: "var(--mem-text-tertiary)", borderTopColor: "var(--mem-accent-indigo)" }} />
            )}
            {query.length > 0 && !isLoading && (
              <button onClick={() => setQuery("")} style={{ color: "var(--mem-text-tertiary)" }} className="shrink-0">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Sidebar + Content row */}
      <div className="flex flex-1 overflow-hidden">
      <Sidebar
        collapsed={sidebarCollapsed}
        onSelectSpace={(name) => { setSelectedDomain(name); }}
        onEntityClick={(id) => { if (id === "__create_profile__") setShowProfile(true); }}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
      {/* Profile detail */}
      {showSettings ? (
        <div className="flex-1 overflow-y-auto px-10 py-7">
          <SettingsPage onBack={() => setShowSettings(false)} />
        </div>
      ) : showProfile ? (
        <div className="flex-1 overflow-y-auto px-10 py-7">
          <ProfilePage onBack={() => setShowProfile(false)} />
        </div>
      ) : query.length > 0 ? (
        <div className="flex-1 overflow-y-auto px-10 py-7">
          {results.length === 0 && !isLoading ? (
            <div className="flex items-center justify-center py-12 text-sm" style={{ color: "var(--mem-text-tertiary)" }}>
              No captures found
            </div>
          ) : (
            <ResultList
              results={results}
              selectedIndex={selectedIndex}
              query={query}
              onSelect={setSelectedIndex}
              onOpen={openResult}
              onCopy={copyResult}
            />
          )}
        </div>
      ) : null}

      {/* Toast notification */}
      {toast && (
        <div className="absolute top-16 right-4 text-white text-sm px-3 py-1.5 rounded-lg shadow-lg animate-fade-in z-50" style={{ backgroundColor: "var(--mem-accent-indigo)" }}>
          {toast}
        </div>
      )}

      {/* Content — hidden when searching or viewing profile */}
      <div className={`flex-1 overflow-y-auto px-10 py-7 space-y-8 ${query.length > 0 || showProfile || showSettings ? "hidden" : ""}`}>
        {/* Toolbar: Stats + Recaps label + Sort + Filter — single row */}
        <div className="flex items-center gap-2">
          {captureStats && totalCount > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              {Object.entries(captureStats)
                .filter(([, count]) => count > 0)
                .map(([source, count]) => (
                  <span key={source} className={`text-[11px] font-semibold px-3 py-1 rounded-full ${SOURCE_COLORS[source] ?? "bg-zinc-500/15 text-zinc-400"}`}>
                    {count} {SOURCE_LABELS[source] ?? formatSourceLabel(source)}{count !== 1 ? "s" : ""}
                  </span>
                ))}
              {recaps.length > 0 && (
                <span
                  className="text-[11px] font-semibold px-3 py-1 rounded-full"
                  style={{ backgroundColor: "var(--mem-indigo-bg)", color: "var(--mem-accent-indigo)" }}
                >
                  {recaps.length} Recap{recaps.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          )}

          <span className="flex-1" />

          {/* Sort dropdown */}
          <div className="relative" ref={actSortRef}>
            <button
              onClick={() => { setActSortOpen(!actSortOpen); setActFilterOpen(false); }}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              style={{ color: activitySortMode !== "recent" ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)" }}
              title="Sort"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 4v16M7 20l-3-3M7 20l3-3M17 20V4M17 4l-3 3M17 4l3 3" />
              </svg>
            </button>
            {actSortOpen && (
              <div
                className="absolute right-0 top-9 z-50 rounded-lg shadow-xl py-1 min-w-[140px]"
                style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
              >
                {([
                  { value: "recent" as const, label: "Recent first" },
                  { value: "oldest" as const, label: "Oldest first" },
                ]).map(({ value, label }) => (
                  <button
                    key={value}
                    className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      color: activitySortMode === value ? "var(--mem-text)" : "var(--mem-text-secondary)",
                    }}
                    onClick={() => { setActivitySortMode(value); setActSortOpen(false); }}
                  >
                    {activitySortMode === value && <span className="mr-1.5">&#10003;</span>}
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Filter by source type dropdown */}
          <div className="relative" ref={actFilterRef}>
            <button
              onClick={() => { setActFilterOpen(!actFilterOpen); setActSortOpen(false); }}
              className="p-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
              style={{ color: sourceTypeFilter ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)" }}
              title="Filter by type"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 4h18l-7 8.5V18l-4 2v-7.5L3 4z" />
              </svg>
              {sourceTypeFilter && (
                <span
                  className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full"
                  style={{ backgroundColor: "var(--mem-accent-indigo)" }}
                />
              )}
            </button>
            {actFilterOpen && (
              <div
                className="absolute right-0 top-9 z-50 rounded-lg shadow-xl py-1 min-w-[160px]"
                style={{ backgroundColor: "var(--mem-surface)", border: "1px solid var(--mem-border)" }}
              >
                <button
                  className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    color: !sourceTypeFilter ? "var(--mem-text)" : "var(--mem-text-secondary)",
                  }}
                  onClick={() => { setSourceTypeFilter(null); setActFilterOpen(false); }}
                >
                  {!sourceTypeFilter && <span className="mr-1.5">&#10003;</span>}
                  All types
                </button>
                {Object.entries(captureStats ?? {})
                  .filter(([, count]) => count > 0)
                  .map(([source]) => (
                    <button
                      key={source}
                      className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        color: sourceTypeFilter === source ? "var(--mem-text)" : "var(--mem-text-secondary)",
                      }}
                      onClick={() => { setSourceTypeFilter(source); setActFilterOpen(false); }}
                    >
                      {sourceTypeFilter === source && <span className="mr-1.5">&#10003;</span>}
                      {SOURCE_LABELS[source] ?? formatSourceLabel(source)}
                    </button>
                  ))}
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3">

          {/* Recaps — editorial timeline strip (matches MemoryStream) */}
          {recaps.length > 0 && (
            <div>
              <div
                className="flex items-center gap-2 mb-2.5"
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "10px",
                  fontWeight: 600,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase" as const,
                  color: "var(--mem-accent-indigo)",
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                </svg>
                Recaps
                <span style={{ color: "var(--mem-text-tertiary)", fontWeight: 400 }}>
                  ({recaps.length})
                </span>
              </div>

              <div className="relative">
                {/* Fade edges */}
                <div
                  className="pointer-events-none absolute left-0 top-0 bottom-0 w-5 z-10"
                  style={{ background: "linear-gradient(to right, var(--mem-bg), transparent)" }}
                />
                <div
                  className="pointer-events-none absolute right-0 top-0 bottom-0 w-5 z-10"
                  style={{ background: "linear-gradient(to left, var(--mem-bg), transparent)" }}
                />

                <div
                  className="flex gap-3 overflow-x-auto pb-1 px-0.5"
                  style={{ scrollbarWidth: "none", msOverflowStyle: "none" } as React.CSSProperties}
                >
                  {recaps.map((recap: IndexedFileInfo, idx: number) => {
                    const isRecapSelected = selectedRecapIds.has(recap.source_id);
                    const isFirst = idx === 0;
                    const diff = Math.floor(Date.now() / 1000) - recap.last_modified;
                    const relTime = diff < 60 ? "just now" : diff < 3600 ? `${Math.floor(diff / 60)}m ago` : diff < 86400 ? `${Math.floor(diff / 3600)}h ago` : `${Math.floor(diff / 86400)}d ago`;
                    return (
                      <div
                        key={recap.source_id}
                        onClick={() => onSelectRecap(recap)}
                        className={`group/recap shrink-0 rounded-lg cursor-pointer transition-all duration-150 hover:brightness-110 ${isRecapSelected ? "ring-1 ring-[var(--mem-accent-indigo)]" : ""}`}
                        style={{
                          width: isFirst ? 280 : 220,
                          backgroundColor: "var(--mem-indigo-bg)",
                          border: "1px solid var(--mem-border)",
                          borderLeft: "3px solid var(--mem-accent-indigo)",
                          animation: `mem-fade-up 400ms cubic-bezier(0.16, 1, 0.3, 1) both`,
                          animationDelay: `${idx * 60}ms`,
                        }}
                      >
                        {/* Select checkbox */}
                        <button
                          onClick={(e) => { e.stopPropagation(); toggleRecapSelected(recap.source_id); }}
                          className={`absolute top-2 right-2 z-10 flex items-center justify-center p-0.5 transition-opacity ${isRecapSelected ? "opacity-100" : "opacity-0 group-hover/recap:opacity-100"}`}
                        >
                          <div className={`w-[16px] h-[16px] rounded border transition-colors flex items-center justify-center ${
                            isRecapSelected ? "bg-[var(--mem-accent-indigo)] border-[var(--mem-accent-indigo)]" : "border-[var(--mem-text-tertiary)]/60 hover:border-[var(--mem-text-secondary)]"
                          }`}>
                            {isRecapSelected && (
                              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                              </svg>
                            )}
                          </div>
                        </button>

                        {/* Time + latest badge */}
                        <div className="flex items-center gap-2 px-3 pt-2.5 pb-1">
                          <span
                            style={{
                              fontFamily: "var(--mem-font-mono)",
                              fontSize: isFirst ? "11px" : "10px",
                              fontWeight: 600,
                              color: "var(--mem-accent-indigo)",
                            }}
                          >
                            {relTime}
                          </span>
                          {isFirst && (
                            <span
                              className="ml-auto"
                              style={{
                                fontFamily: "var(--mem-font-mono)",
                                fontSize: "9px",
                                fontWeight: 600,
                                letterSpacing: "0.06em",
                                textTransform: "uppercase" as const,
                                color: "var(--mem-accent-indigo)",
                                opacity: 0.6,
                              }}
                            >
                              Latest
                            </span>
                          )}
                        </div>

                        {/* Summary */}
                        <div className="px-3 pb-2.5">
                          <p
                            className="line-clamp-2 mt-0.5"
                            style={{
                              fontFamily: "var(--mem-font-heading)",
                              fontSize: isFirst ? "13px" : "12px",
                              fontWeight: 400,
                              fontStyle: "italic",
                              lineHeight: "1.5",
                              color: "var(--mem-text)",
                            }}
                          >
                            {recap.summary ?? recap.title}
                          </p>

                          {/* Domain + dot connector */}
                          <div className="flex items-center gap-1.5 mt-2">
                            {recap.domain && (
                              <span
                                className="px-1.5 py-0.5 rounded"
                                style={{
                                  fontFamily: "var(--mem-font-mono)",
                                  fontSize: "9px",
                                  fontWeight: 500,
                                  backgroundColor: "var(--mem-indigo-bg)",
                                  color: "var(--mem-accent-indigo)",
                                }}
                              >
                                {recap.domain}
                              </span>
                            )}
                            <span className="flex-1" />
                            <div
                              className="rounded-full"
                              style={{
                                width: isFirst ? 7 : 5,
                                height: isFirst ? 7 : 5,
                                backgroundColor: "var(--mem-accent-indigo)",
                                opacity: 0.4,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}


          {/* Tags — browseable horizontal pills with counts */}
          {(() => {
            const tagCounts = new Map<string, number>();
            for (const tags of Object.values(tagData?.document_tags ?? {})) {
              for (const tag of tags) {
                tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
              }
            }
            const sortedTags = [...tagCounts.entries()].sort((a, b) => b[1] - a[1]);
            if (sortedTags.length === 0) return null;
            return (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13px] tracking-wide text-[var(--mem-text-secondary)] font-semibold uppercase">
                    Tags
                  </h3>
                  {activeTagFilter && (
                    <button
                      onClick={() => setActiveTagFilter(null)}
                      className="text-[10px] text-[var(--mem-text-tertiary)] hover:text-[var(--mem-accent-indigo)] transition-colors"
                    >
                      Clear filter
                    </button>
                  )}
                </div>
                <div className="relative">
                  <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-6 z-10" style={{ background: "linear-gradient(to right, var(--mem-bg), transparent)" }} />
                  <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-6 z-10" style={{ background: "linear-gradient(to left, var(--mem-bg), transparent)" }} />
                  <div className="flex gap-2 overflow-x-auto pb-2 px-1 scrollbar-none" style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}>
                    {sortedTags.map(([tag, count]) => (
                      <button
                        key={tag}
                        onClick={() => setActiveTagFilter(activeTagFilter === tag ? null : tag)}
                        className={`shrink-0 text-[11px] font-medium px-3 py-1.5 rounded-full border transition-all duration-200 ${
                          activeTagFilter === tag
                            ? "bg-[var(--mem-accent-indigo)]/20 text-[var(--mem-accent-indigo)] border-[var(--mem-accent-indigo)]/25 shadow-sm"
                            : "bg-[var(--mem-sidebar)] text-[var(--mem-text)] border-[var(--mem-border)] hover:bg-[var(--mem-hover-strong)] hover:border-[var(--mem-text-tertiary)]"
                        }`}
                      >
                        {tag}
                        <span className={`ml-1.5 ${activeTagFilter === tag ? "text-[var(--mem-accent-indigo)]/60" : "text-[var(--mem-text-secondary)]"}`}>
                          {count}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}

        </div>

        {totalCount === 0 && activities.length === 0 && (
          <div className="bg-[var(--bg-secondary)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.2)] px-6 py-14 text-center text-sm text-[var(--text-tertiary)]">
            <svg className="w-8 h-8 mx-auto mb-3 text-[var(--text-tertiary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
            No memories indexed yet
          </div>
        )}

        {/* Activities + ungrouped files merged by time period */}
        {mergedTimePeriods.map(({ period, periodItems, memories: periodMemories }) => {
            const periodFileCount = periodItems.reduce((sum, g) => sum + g.children.filter(isRenderable).length, 0) + periodMemories.length;
            return (
            <div key={period} className="space-y-5">
              <div className="flex items-center gap-1.5 px-1 group/period">
                <h4 className="text-[13px] tracking-wide text-[var(--text-secondary)] font-semibold uppercase">
                  {period}
                </h4>
                {!hasSelection && periodFileCount > 0 && (
                  <button
                    onClick={() => handleTimePeriodDelete(period, periodFileCount)}
                    disabled={isDeleting}
                    className="opacity-0 group-hover/period:opacity-60 hover:!opacity-100 transition-all duration-150 text-[var(--text-tertiary)] hover:text-red-400/90 p-0.5 -ml-0.5 rounded"
                    title={`Clear ${period.toLowerCase()}`}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
              {periodItems.length > 0 && periodItems.map((group) => {
                const isExpanded = !collapsedGroups.has(group.activity.id);
                const toggleGroup = () => {
                  setCollapsedGroups(toggleSection(group.activity.id));
                };
                return (
                  <div key={group.activity.id} className="space-y-3">
                    <div className="flex items-start gap-3 px-1 group/session">
                      <div
                        onClick={toggleGroup}
                        className="flex items-start gap-3 flex-1 min-w-0 cursor-pointer"
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") toggleGroup(); }}
                      >
                        <svg
                          className={`w-4 h-4 mt-1 shrink-0 text-[var(--text-tertiary)] transition-transform duration-150 ${isExpanded ? "rotate-90" : ""}`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <div className="flex-1 min-w-0 space-y-1.5">
                          <div className="flex items-center gap-3">
                            <span className="text-base font-semibold text-[var(--text-primary)] tracking-wide">
                              {formatActivityTime(group.activity.started_at, period)} – {formatActivityTime(group.activity.ended_at, period)}
                            </span>
                            {group.activity.is_live && (
                              <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-full bg-green-500/15 text-green-400">
                                Live
                              </span>
                            )}
                            <span className="text-[11px] text-[var(--text-tertiary)] ml-auto">
                              {group.children.filter(isRenderable).length} item{group.children.filter(isRenderable).length !== 1 ? "s" : ""}
                            </span>
                          </div>
                          {group.activity.app_names.length > 0 && (
                            <div className="flex items-center gap-1.5 flex-wrap">
                              {group.activity.app_names.map((app, i) => (
                                <span key={app} className="flex items-center gap-1.5">
                                  {i > 0 && (
                                    <span className="text-[10px] text-[var(--text-tertiary)]">→</span>
                                  )}
                                  <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-full bg-indigo-500/15 text-indigo-400">
                                    {app}
                                  </span>
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                      {!hasSelection && (
                        <button
                          onClick={() => handleActivityDelete(group.activity, group.children)}
                          disabled={isDeleting}
                          className="shrink-0 mt-1 opacity-0 group-hover/session:opacity-60 hover:!opacity-100 transition-all duration-150 text-[var(--text-tertiary)] hover:text-red-400/90 p-0.5 rounded"
                          title="Clear session"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                    </div>
                    {isExpanded && (
                      <div className="grid grid-cols-4 gap-4 items-start">
                        {group.children.filter(isRenderable).length === 0 ? (
                          <div className="col-span-4 bg-[var(--bg-secondary)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.2)] px-5 py-6 text-sm text-[var(--text-tertiary)] text-center italic">
                            No indexed items yet
                          </div>
                        ) : (
                          group.children.filter(isRenderable).map((child) => renderFileCard(child))
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
              {/* Imported memories in this time period — grid cards, click opens detail */}
              {periodMemories.length > 0 && (
                <div className="grid grid-cols-4 gap-4 items-start">
                  {periodMemories.map((mem) => {
                    const file = indexedFiles.find(f => f.source_id === mem.source_id);
                    return file ? renderFileCard(file) : null;
                  })}
                </div>
              )}

            </div>
            );
        })}

        {/* Ungrouped items — always at the bottom, below all time periods */}
        {filteredUngrouped.filter(isRenderable).length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-3 px-1">
              <h4 className="text-[13px] tracking-wide text-[var(--text-secondary)] font-semibold uppercase">
                Other
              </h4>
              <span className="text-[11px] text-[var(--text-tertiary)]">
                {filteredUngrouped.filter(isRenderable).length} item{filteredUngrouped.filter(isRenderable).length !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-4 items-start">
              {filteredUngrouped.filter(isRenderable).map((file) => renderFileCard(file))}
            </div>
          </div>
        )}
      </div>

      {/* Floating selection action bar */}
      {hasSelection && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-[var(--bg-secondary)]/90 backdrop-blur-xl border border-[var(--overlay-border)] rounded-full pl-4 pr-1.5 py-1.5 shadow-[0_8px_32px_rgba(0,0,0,0.3)] z-40">
          <span className="text-[13px] text-[var(--text-secondary)] mr-1">
            {totalSelected} selected
          </span>
          {totalSelected > 0 && (
            <button
              onClick={handleBulkDelete}
              disabled={isDeleting}
              className="text-[13px] font-medium text-red-400 hover:text-red-300 disabled:opacity-50 transition-colors px-3 py-1 rounded-full hover:bg-red-500/10"
            >
              Delete
            </button>
          )}
          <button
            onClick={clearSelection}
            className="text-[var(--text-tertiary)] hover:text-[var(--text-primary)] transition-colors p-1.5 rounded-full hover:bg-[var(--overlay-hover)]"
            title="Cancel selection"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* Confirmation overlay */}
      {confirmAction && (
        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[var(--bg-secondary)] border border-[var(--overlay-border)] rounded-2xl px-6 py-5 mx-6 max-w-xs w-full shadow-[0_24px_64px_rgba(0,0,0,0.4)]">
            <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">{confirmAction.label}</p>
            <p className="text-[12px] text-[var(--text-tertiary)] mt-1 mb-5">This action cannot be undone.</p>
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setConfirmAction(null)}
                disabled={isDeleting}
                className="text-[13px] px-3.5 py-1.5 rounded-lg text-[var(--text-secondary)] hover:bg-[var(--overlay-hover)] transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmAction.onConfirm}
                disabled={isDeleting}
                className="text-[13px] font-medium px-3.5 py-1.5 rounded-lg bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors disabled:opacity-50"
              >
                {isDeleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Capture modal */}
      <QuickCapture
        isOpen={showQuickCapture}
        onClose={() => setShowQuickCapture(false)}
      />
    </div>
    </div>
    </div>
  );
}

// ── Space dropdown ────────────────────────────────────────────────

function SpaceDropdown({
  spaces,
  current,
  onSelect,
  onClose,
}: {
  spaces: { id: string; name: string; icon?: string; pinned?: boolean }[];
  current?: string;
  onSelect: (spaceId: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute left-4 bottom-12 z-50 bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-lg shadow-xl py-1 min-w-[160px]"
    >
      {spaces
        .filter((s) => s.pinned)
        .map((space) => (
          <button
            key={space.id}
            onClick={() => onSelect(space.id)}
            className={`w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 hover:bg-[var(--overlay-hover)] ${
              current === space.id ? "text-[var(--accent)]" : "text-[var(--text-primary)]"
            }`}
          >
            <SpaceIcon icon={space.icon ?? "sparkles"} size={13} />
            <span>{space.name}</span>
            {current === space.id && <span className="ml-auto text-[var(--accent)]">&#10003;</span>}
          </button>
        ))}
    </div>
  );
}
