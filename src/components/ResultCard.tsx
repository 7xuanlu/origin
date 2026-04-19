// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import type { SearchResult } from "../lib/tauri";
import { highlightTerms, relevanceLabel } from "../lib/highlight";

interface ResultCardProps {
  result: SearchResult;
  query: string;
  isSelected: boolean;
  onSelect: () => void;
  onOpen: () => void;
  onCopy: () => void;
}

const SOURCE_LABELS: Record<string, string> = {
  local_files: "File",
  clipboard: "Clipboard",
  manual: "Capture",
  screen_capture: "Screen",
  session_snapshot: "Recap",
};

const SOURCE_STYLES: Record<string, { bg: string; text: string }> = {
  local_files: { bg: "var(--badge-file)", text: "var(--badge-file-text)" },
  clipboard: { bg: "var(--badge-clipboard)", text: "var(--badge-clipboard-text)" },
  manual: { bg: "var(--badge-capture)", text: "var(--badge-capture-text)" },
  screen_capture: { bg: "var(--badge-screen)", text: "var(--badge-screen-text)" },
  session_snapshot: { bg: "var(--badge-recap)", text: "var(--badge-recap-text)" },
};

const TIER_COLORS = {
  strong: "text-[var(--badge-entity-text)]",
  good: "text-[var(--text-secondary)]",
  faint: "text-[var(--text-tertiary)]/60",
} as const;

export default function ResultCard({
  result,
  query,
  isSelected,
  onSelect,
  onOpen,
  onCopy,
}: ResultCardProps) {
  const [hovered, setHovered] = useState(false);
  const [copied, setCopied] = useState(false);

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation();
    onCopy();
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }
  const sourceLabel = SOURCE_LABELS[result.source] ?? result.source.replace("_", " ");
  const sourceStyle = SOURCE_STYLES[result.source] ?? { bg: "var(--overlay-active)", text: "var(--text-tertiary)" };

  const chunkTypeLabel = result.language
    ? result.language
    : result.chunk_type ?? null;

  const snippet =
    result.content.length > 300
      ? result.content.substring(0, 300) + "…"
      : result.content;

  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 mx-1 cursor-pointer transition-all duration-100 rounded-lg ${
        isSelected
          ? "bg-[var(--accent)]/10"
          : "hover:bg-[var(--overlay-subtle)]"
      }`}
      onClick={onSelect}
      onDoubleClick={onOpen}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex-1 min-w-0">
        {/* Title row */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[13px] font-medium text-[var(--text-primary)] truncate">
            {result.title}
          </span>
        </div>

        {/* Badges row */}
        <div className="flex items-center gap-1.5 mt-1 flex-wrap">
          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full" style={{ background: sourceStyle.bg, color: sourceStyle.text }}>
            {sourceLabel}
          </span>
          {chunkTypeLabel && (
            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-[var(--overlay-active)] text-[var(--text-tertiary)]">
              {chunkTypeLabel}
            </span>
          )}
          {result.chunk_index > 0 && (
            <span className="text-[10px] text-[var(--text-tertiary)]/60">
              chunk {result.chunk_index}
            </span>
          )}
        </div>

        {/* Content */}
        <p className="text-[12px] text-[var(--text-secondary)] mt-1.5 leading-relaxed line-clamp-3 font-mono whitespace-pre-wrap break-words">
          {highlightTerms(snippet, query, "bg-[var(--highlight-match)] text-[var(--text-primary)] rounded-sm px-0.5")}
        </p>
      </div>

      {/* Score + copy */}
      <div className="flex flex-col items-end gap-1.5 shrink-0 pt-0.5">
        {(() => {
          const rel = relevanceLabel(result.score);
          return (
            <span className={`text-[10px] ${TIER_COLORS[rel.tier]}`}>
              {rel.text}
            </span>
          );
        })()}
        <button
          onClick={handleCopy}
          className={`text-[10px] transition-all duration-150 ${
            copied
              ? "text-green-400 opacity-100"
              : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]"
          } ${hovered || isSelected ? "opacity-100" : "opacity-0"}`}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  );
}
