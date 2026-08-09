// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { listRegisteredSources, type SyncStatusStr } from "../../lib/tauri";

const SOURCES_STORAGE_KEY = "sidebar:sourcesCollapsed";
const TTL_MS = 7 * 86_400_000;

function useSourcesCollapsed() {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      const raw = localStorage.getItem(SOURCES_STORAGE_KEY);
      if (!raw) return false;
      const { value, ts } = JSON.parse(raw);
      if (Date.now() - ts > TTL_MS) {
        localStorage.removeItem(SOURCES_STORAGE_KEY);
        return false;
      }
      return !!value;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(SOURCES_STORAGE_KEY, JSON.stringify({ value: collapsed, ts: Date.now() }));
    } catch {
      // localStorage full or disabled. Non-fatal.
    }
  }, [collapsed]);
  return [collapsed, setCollapsed] as const;
}

function folderName(p: string): string {
  return p.split("/").filter(Boolean).pop() || p;
}

/** Collapse the daemon's SyncStatus enum (unit and tuple variants) to a label, or null when Active. */
function statusLabel(status: SyncStatusStr): string | null {
  if (status === "Active") return null;
  if (status === "Paused") return "Paused";
  if (typeof status === "object" && "Unavailable" in status) return "Unavailable";
  return "Sync error";
}

const STATUS_COLORS: Record<string, string> = {
  Paused: "var(--mem-accent-amber)",
  "Sync error": "#ef4444",
  Unavailable: "var(--mem-text-tertiary)",
};

/** Spine height in px: 8px floor, 14px for the largest source. */
function spineHeight(memoryCount: number, maxMemories: number): number {
  return 8 + Math.round(6 * (memoryCount / maxMemories));
}

interface SourceListProps {
  /** Opens Settings › Sources, where sources are added, synced, and removed. */
  onNavigateSources: () => void;
}

export default function SourceList({ onNavigateSources }: SourceListProps) {
  const [collapsed, setCollapsed] = useSourcesCollapsed();

  const { data: sources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: listRegisteredSources,
    refetchInterval: 10000,
  });

  const maxMemories = Math.max(...sources.map((s) => s.memory_count), 1);

  return (
    <div className="flex flex-col gap-0.5">
      {/* Section label: clickable to collapse */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        aria-expanded={!collapsed}
        aria-controls="source-list-items"
        className="flex items-center justify-between px-2 mb-1 rounded hover:bg-[var(--mem-hover)] transition-colors duration-150 w-full"
        style={{ background: "none", border: "none", cursor: "pointer" }}
      >
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            fontWeight: 600,
            letterSpacing: "0.05em",
            textTransform: "uppercase" as const,
            color: "var(--mem-text-tertiary)",
          }}
        >
          Sources
        </span>
        <div className="flex items-center gap-2">
          <span
            role="button"
            title="Add source"
            onClick={(e) => { e.stopPropagation(); onNavigateSources(); }}
            className="flex items-center justify-center rounded transition-colors duration-150"
            style={{ color: "var(--mem-text-tertiary)", cursor: "pointer" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--mem-text-secondary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--mem-text-tertiary)")}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
              <path d="M6 2 L6 10 M2 6 L10 6" />
            </svg>
          </span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              transform: collapsed ? "rotate(-90deg)" : "rotate(0deg)",
              transition: "transform 150ms",
              color: "var(--mem-text-tertiary)",
            }}
            aria-hidden="true"
          >
            <polyline points="3,4.5 6,7.5 9,4.5" />
          </svg>
        </div>
      </button>

      {!collapsed && (
        <div id="source-list-items">
          {sources.map((s) => {
            const label = statusLabel(s.status);
            return (
              <button
                key={s.id}
                onClick={onNavigateSources}
                title={label ? `${s.path} · ${label}` : s.path}
                className="w-full flex items-center justify-between px-2 py-1.5 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  color: "var(--mem-text-secondary)",
                }}
              >
                <span className="flex min-w-0 items-center gap-1 truncate">
                  {/* Book spine: height = this source's share of ingested
                      memories, color = sync status. The shelf of the wiki. */}
                  <span className="shrink-0 w-3.5 flex items-center justify-center">
                    <span
                      data-testid="source-spine"
                      className="inline-block"
                      style={{
                        width: 3,
                        borderRadius: 1,
                        height: spineHeight(s.memory_count, maxMemories),
                        backgroundColor: label
                          ? STATUS_COLORS[label]
                          : "var(--mem-accent-indigo)",
                        opacity: label ? 0.9 : 0.55,
                      }}
                    />
                  </span>
                  <span className="truncate">{folderName(s.path)}</span>
                </span>
                <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", opacity: 0.5 }}>
                  {s.memory_count || ""}
                </span>
              </button>
            );
          })}

          {sources.length === 0 && (
            <button
              onClick={onNavigateSources}
              className="w-full rounded-md border border-dashed border-[var(--mem-border)] px-2 py-1.5 text-left transition-colors duration-150 hover:border-[var(--mem-accent-indigo)] hover:text-[var(--mem-accent-indigo)]"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              + Add folder or vault
            </button>
          )}
        </div>
      )}
    </div>
  );
}
