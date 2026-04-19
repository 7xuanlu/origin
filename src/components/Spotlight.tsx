// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useCallback, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { resizeWindow } from "../lib/resizeWindow";
import { useSearch } from "../hooks/useSearch";
import { openFile, clipboardWrite, getWorkingMemory, getSessionSnapshots, searchEntities, type IndexedFileInfo } from "../lib/tauri";
import { useEscapeToHide } from "../hooks/useShortcut";
import SearchInput from "./SearchInput";
import ResultList from "./ResultList";
import StatusBar from "./StatusBar";

const WINDOW_WIDTH = 580;

const SOURCE_FILTERS = [
  { key: undefined, label: "All" },
  { key: "local_files", label: "Files" },
  { key: "manual", label: "Captured" },
  { key: "clipboard", label: "Clipboard" },
  { key: "screen_capture", label: "Screen" },
  { key: "session_snapshot", label: "Recaps" },
] as const;

const SOURCE_BADGE_STYLES: Record<string, { bg: string; text: string }> = {
  ambient: { bg: "var(--badge-screen)", text: "var(--badge-screen-text)" },
  focus_capture: { bg: "var(--badge-clipboard)", text: "var(--badge-clipboard-text)" },
  hotkey_capture: { bg: "var(--badge-capture)", text: "var(--badge-capture-text)" },
  snip_capture: { bg: "var(--badge-capture)", text: "var(--badge-capture-text)" },
  clipboard: { bg: "var(--badge-clipboard)", text: "var(--badge-clipboard-text)" },
  quick_thought: { bg: "var(--badge-capture)", text: "var(--badge-capture-text)" },
  session_snapshot: { bg: "var(--badge-recap)", text: "var(--badge-recap-text)" },
};

function relativeTime(ts: number): string {
  const diff = Math.floor(Date.now() / 1000) - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

interface SpotlightProps {
  onOpenMemory: () => void;
  onOpenRecap: (recap: IndexedFileInfo) => void;
  onEntityClick: (entityId: string) => void;
}

export default function Spotlight({ onOpenMemory, onOpenRecap, onEntityClick }: SpotlightProps) {
  const [sourceFilter, setSourceFilter] = useState<string | undefined>(
    undefined,
  );
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const { query, setQuery, results, isLoading } = useSearch(sourceFilter);
  const containerRef = useRef<HTMLDivElement>(null);

  const { data: workingMemory = [] } = useQuery({
    queryKey: ["workingMemory"],
    queryFn: getWorkingMemory,
    refetchInterval: 5000,
  });

  const { data: snapshots = [] } = useQuery({
    queryKey: ["sessionSnapshots"],
    queryFn: () => getSessionSnapshots(5),
    refetchInterval: 30000,
  });

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

  useEscapeToHide();

  // Resize the native window to match content height.
  // Debounced to avoid rapid-fire IPC calls on every state change.
  const lastHeightRef = useRef(58);
  useEffect(() => {
    const timer = setTimeout(() => {
      const el = containerRef.current;
      if (!el) return;
      const height = Math.max(58, Math.ceil(el.scrollHeight));
      if (height !== lastHeightRef.current) {
        lastHeightRef.current = height;
        resizeWindow(WINDOW_WIDTH, height);
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [query, results, isLoading, workingMemory, snapshots, entityResults]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [results]);

  const showToast = useCallback((message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 2000);
  }, []);

  const openResult = useCallback(
    async (index: number) => {
      const result = results[index];
      if (!result) return;
      if (result.url) {
        try {
          await openFile(result.url);
          showToast("Opened file");
        } catch (err) {
          showToast(`Failed to open file: ${err}`);
        }
      } else {
        // No URL (e.g. session snapshots) — copy content instead
        const text = `${result.title}\n\n${result.content}`;
        await clipboardWrite(text);
        showToast("Copied to clipboard");
      }
    },
    [results, showToast],
  );

  const copyResult = useCallback(
    async (index: number) => {
      const result = results[index];
      if (!result) return;

      try {
        const text = `${result.title}\n\n${result.content}${result.url ? `\n\nSource: ${result.url}` : ""}`;
        await clipboardWrite(text);
        showToast("Copied to clipboard");
      } catch (err) {
        showToast(`Failed to copy: ${err}`);
      }
    },
    [results, showToast],
  );

  const copyAll = useCallback(async () => {
    if (results.length === 0) return;

    try {
      const text = results
        .map(
          (r, i) =>
            `[${i + 1}] ${r.title}\n${r.content}${r.url ? `\nSource: ${r.url}` : ""}`,
        )
        .join("\n\n---\n\n");

      await clipboardWrite(text);
      showToast("All results copied");
    } catch (err) {
      showToast(`Failed to copy: ${err}`);
    }
  }, [results, showToast]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
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
            copyAll();
          } else {
            openResult(selectedIndex);
          }
          break;
        case "c":
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            copyResult(selectedIndex);
          }
          break;
      }
    },
    [results.length, selectedIndex, openResult, copyResult, copyAll],
  );

  const hasResults = results.length > 0;
  const hasEntityResults = entityResults.length > 0;
  const hasRecents = query.length === 0 && (snapshots.length > 0 || workingMemory.length > 0);
  const showChrome = hasResults || hasEntityResults || query.length > 0 || hasRecents;

  return (
    <div
      ref={containerRef}
      className="w-full flex flex-col bg-[var(--bg-secondary)] overflow-hidden"
      onKeyDown={handleKeyDown}
    >
      {/* Search input */}
      <SearchInput value={query} onChange={setQuery} isLoading={isLoading} onOpenMemory={onOpenMemory} />

      {/* Filter tabs — shown as soon as user starts typing */}
      {query.length > 0 && (
        <div className="flex items-center justify-between px-4 py-1.5 border-t border-[var(--separator)]/40">
          <div className="flex items-center gap-0.5">
            {SOURCE_FILTERS.map((filter) => (
              <button
                key={filter.label}
                onClick={() => setSourceFilter(filter.key)}
                className={`px-2.5 py-1 text-[12px] rounded-md transition-all duration-150 ${
                  sourceFilter === filter.key
                    ? "bg-[var(--accent)]/12 text-[var(--accent)] font-medium"
                    : "text-[var(--text-tertiary)] hover:text-[var(--text-secondary)] hover:bg-[var(--overlay-subtle)]"
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results or Working Memory */}
      {query.length > 0 && results.length === 0 && !hasEntityResults && !isLoading ? (
        <div className="flex flex-col items-center justify-center py-10 gap-2 animate-[fade-in_150ms_ease-out]">
          <svg className="w-8 h-8 text-[var(--text-tertiary)]/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span className="text-[13px] text-[var(--text-tertiary)]">No results found</span>
        </div>
      ) : (hasResults || hasEntityResults) ? (
        <div className="relative max-h-[400px] overflow-y-auto">
          {/* Top separator */}
          <div className="h-px bg-[var(--separator)]/40" />

          {hasResults && (
            <div className="animate-[fade-in_100ms_ease-out]">
              <ResultList
                results={results}
                selectedIndex={selectedIndex}
                query={query}
                onSelect={setSelectedIndex}
                onOpen={openResult}
                onCopy={copyResult}
              />
            </div>
          )}
          {hasEntityResults && (
            <div className="animate-[fade-in_120ms_ease-out]">
              <div className="px-4 pt-3 pb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)]/70">
                  Entities
                </span>
              </div>
              {entityResults.map((r) => (
                <div
                  key={r.entity.id}
                  className="group w-full text-left px-4 py-2.5 hover:bg-[var(--overlay-subtle)] transition-colors duration-100 cursor-pointer"
                  onClick={() => onEntityClick(r.entity.id)}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full" style={{ background: 'var(--badge-entity)', color: 'var(--badge-entity-text)' }}>
                      {r.entity.entity_type}
                    </span>
                    <span className="text-[13px] font-medium text-[var(--text-primary)] truncate flex-1">
                      {r.entity.name}
                    </span>
                    {r.entity.domain && (
                      <span className="text-[10px] text-[var(--text-tertiary)]/70">
                        {r.entity.domain}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Scroll fade gradient */}
          <div className="pointer-events-none sticky bottom-0 h-6 bg-gradient-to-t from-[var(--bg-secondary)] to-transparent" />
        </div>
      ) : hasRecents ? (
        <div className="relative max-h-[400px] overflow-y-auto">
          {/* Top separator */}
          <div className="h-px bg-[var(--separator)]/40" />

          {/* Recaps section */}
          {snapshots.length > 0 && (
            <div className="animate-[fade-in_100ms_ease-out]">
              <div className="px-4 pt-3 pb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)]/70">
                  Recaps
                </span>
              </div>
              {snapshots.map((snap) => (
                <div
                  key={snap.id}
                  className="group w-full text-left px-4 py-2.5 hover:bg-[var(--overlay-subtle)] transition-colors duration-100 cursor-pointer"
                  onClick={() => onOpenRecap({
                    source: "session_snapshot",
                    source_id: snap.id,
                    title: snap.summary,
                    summary: snap.summary,
                    chunk_count: snap.capture_count,
                    last_modified: snap.ended_at,
                    memory_type: "recap",
                  } satisfies IndexedFileInfo)}
                >
                  <div className="flex items-center gap-2.5 mb-1">
                    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full" style={{ background: 'var(--badge-recap)', color: 'var(--badge-recap-text)' }}>
                      Recap
                    </span>
                    <span className="text-[13px] font-medium text-[var(--text-primary)] truncate flex-1">
                      {snap.summary}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        const text = `${snap.summary}\n\nApps: ${snap.primary_apps.join(", ")}\nCaptures: ${snap.capture_count}${snap.tags.length > 0 ? `\nTags: ${snap.tags.join(", ")}` : ""}`;
                        clipboardWrite(text);
                        showToast("Copied recap");
                      }}
                      className="text-[10px] text-[var(--text-tertiary)] hover:text-[var(--text-primary)] opacity-0 group-hover:opacity-100 transition-opacity duration-150 shrink-0"
                    >
                      Copy
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5 pl-0.5">
                    <span className="text-[11px] text-[var(--text-secondary)]">
                      {snap.primary_apps.join(", ")}
                    </span>
                    <span className="text-[11px] text-[var(--text-tertiary)]/50">·</span>
                    <span className="text-[11px] text-[var(--text-tertiary)]">
                      {relativeTime(snap.ended_at)}
                    </span>
                    <span className="text-[11px] text-[var(--text-tertiary)]/50">·</span>
                    <span className="text-[11px] text-[var(--text-tertiary)]">
                      {snap.capture_count} capture{snap.capture_count !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {snap.tags.length > 0 && (
                    <div className="flex items-center gap-1 mt-1.5 pl-0.5 flex-wrap">
                      {snap.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] px-1.5 py-0.5 rounded-full bg-[var(--bg-tertiary)]/60 text-[var(--text-tertiary)]"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Just Now section */}
          {workingMemory.length > 0 && (
            <div className="animate-[fade-in_120ms_ease-out]">
              <div className="px-4 pt-3 pb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)]/70">
                  Just Now
                </span>
              </div>
              {workingMemory.slice(0, 3).map((entry) => {
                const badge = SOURCE_BADGE_STYLES[entry.source] ?? { bg: "var(--overlay-active)", text: "var(--text-tertiary)" };
                return (
                <button
                  key={entry.source_id}
                  onClick={() => setQuery(`${entry.app_name} ${entry.window_title}`)}
                  className="w-full text-left px-4 py-2.5 hover:bg-[var(--overlay-subtle)] transition-colors duration-100 last:border-b-0"
                >
                  <div className="flex items-center gap-2.5 mb-1">
                    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full" style={{ background: badge.bg, color: badge.text }}>
                      {entry.source === "ambient" ? "Screen" : entry.source === "clipboard" ? "Clip" : entry.source === "quick_thought" ? "Note" : "Capture"}
                    </span>
                    <span className="text-[13px] font-medium text-[var(--text-primary)] truncate">
                      {entry.app_name}
                    </span>
                    <span className="text-[11px] text-[var(--text-tertiary)] ml-auto shrink-0">
                      {relativeTime(entry.timestamp)}
                    </span>
                  </div>
                  <div className="text-[12px] text-[var(--text-secondary)] truncate pl-0.5">
                    {entry.window_title}
                  </div>
                  {entry.text_snippet && (
                    <div className="text-[11px] text-[var(--text-tertiary)] truncate mt-0.5 pl-0.5">
                      {entry.text_snippet.split("\n")[0]}
                    </div>
                  )}
                </button>
                );
              })}
            </div>
          )}

          {/* Scroll fade gradient */}
          <div className="pointer-events-none sticky bottom-0 h-6 bg-gradient-to-t from-[var(--bg-secondary)] to-transparent" />
        </div>
      ) : null}

      {/* Toast notification */}
      {toast && (
        <div className="absolute top-4 right-4 bg-[var(--accent)] text-white text-[12px] px-3 py-1.5 rounded-lg shadow-lg animate-[fade-in_100ms_ease-out]">
          {toast}
        </div>
      )}

      {/* Status bar — only rendered when there's content */}
      {showChrome && (
        <StatusBar resultCount={results.length} />
      )}

    </div>
  );
}
