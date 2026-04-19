// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { convertFileSrc } from "@tauri-apps/api/core";
import {
  getProfile,
  listPinnedMemories,
  FACET_COLORS,
  MEMORY_FACETS,
  type MemoryType,
} from "../../lib/tauri";

interface ProfileSummaryCardProps {
  onClick: () => void;
}

const COLLAPSED_KEY = "profile-summary-collapsed";

function readCollapsed(): boolean {
  try {
    const raw = localStorage.getItem(COLLAPSED_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.value === "boolean" && typeof parsed.ts === "number") {
      // Expire after 7 days
      if (Date.now() - parsed.ts > 7 * 24 * 60 * 60 * 1000) {
        localStorage.removeItem(COLLAPSED_KEY);
        return false;
      }
      return parsed.value;
    }
    return false;
  } catch {
    return false;
  }
}

function writeCollapsed(value: boolean) {
  localStorage.setItem(COLLAPSED_KEY, JSON.stringify({ value, ts: Date.now() }));
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + "\u2026";
}

export default function ProfileSummaryCard({ onClick }: ProfileSummaryCardProps) {
  const [collapsed, setCollapsed] = useState(readCollapsed);

  const { data: profile } = useQuery({
    queryKey: ["profile"],
    queryFn: getProfile,
  });

  const { data: pinnedMemories = [] } = useQuery({
    queryKey: ["pinned-memories"],
    queryFn: listPinnedMemories,
  });

  if (!profile) return null;

  const displayName = profile.display_name || profile.name;
  const initials = displayName
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const avatarUrl = profile.avatar_path ? convertFileSrc(profile.avatar_path) : null;

  const toggleCollapsed = (e: React.MouseEvent) => {
    e.stopPropagation();
    const next = !collapsed;
    setCollapsed(next);
    writeCollapsed(next);
  };

  // Count memories by type
  const typeCounts: Partial<Record<MemoryType, number>> = {};
  for (const mem of pinnedMemories) {
    const t = (mem.memory_type ?? "fact") as MemoryType;
    typeCounts[t] = (typeCounts[t] ?? 0) + 1;
  }

  return (
    <div
      className="mb-4 rounded-lg"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
      }}
    >
      {/* Header row — always visible */}
      <div
        className="flex items-center gap-3 p-3"
        style={{ cursor: "pointer" }}
        onClick={onClick}
      >
        {/* Avatar */}
        {avatarUrl ? (
          <img
            src={avatarUrl}
            alt={displayName}
            style={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              objectFit: "cover",
              flexShrink: 0,
            }}
          />
        ) : (
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              fontFamily: "var(--mem-font-heading)",
              fontSize: "15px",
              color: "white",
              fontWeight: 500,
            }}
          >
            {initials}
          </div>
        )}

        {/* Name */}
        <span
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "15px",
            fontWeight: 500,
            color: "var(--mem-text)",
            flex: 1,
            minWidth: 0,
          }}
        >
          {displayName}
        </span>

        {/* Chevron toggle */}
        <button
          onClick={toggleCollapsed}
          className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{ color: "var(--mem-text-tertiary)", flexShrink: 0 }}
          title={collapsed ? "Expand" : "Collapse"}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            style={{
              transform: collapsed ? "rotate(-90deg)" : "rotate(0deg)",
              transition: "transform 200ms ease",
            }}
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
      </div>

      {/* Expanded content */}
      {!collapsed && pinnedMemories.length > 0 && (
        <div style={{ borderTop: "1px solid var(--mem-border)" }}>
          {/* Chip cloud */}
          <div className="flex flex-wrap gap-1.5 px-3 py-3">
            {pinnedMemories.map((mem) => (
              <span
                key={mem.source_id}
                className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${FACET_COLORS[mem.memory_type ?? "fact"]}`}
                style={{ fontFamily: "var(--mem-font-body)" }}
                title={mem.content}
              >
                {truncate(mem.content, 40)}
              </span>
            ))}
          </div>

          {/* Footer — type counts + "View all" */}
          <div
            className="flex items-center justify-between px-3 py-2"
            style={{ borderTop: "1px solid var(--mem-border)" }}
          >
            <div
              className="flex items-center gap-2 flex-wrap"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "10px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {MEMORY_FACETS.filter((f) => typeCounts[f.type]).map((f) => (
                <span key={f.type}>
                  {typeCounts[f.type]} {f.label.toLowerCase()}
                </span>
              ))}
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                onClick();
              }}
              className="transition-colors duration-150 hover:underline"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-secondary)",
                flexShrink: 0,
              }}
            >
              View all &rarr;
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
