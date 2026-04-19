// SPDX-License-Identifier: AGPL-3.0-only
import type { SearchResult } from "../../lib/tauri";
import { FACET_COLORS } from "../../lib/tauri";
import { highlightTerms, hasKeywordMatch, relevanceLabel } from "../../lib/highlight";

interface MemorySearchResultProps {
  result: SearchResult;
  query: string;
  onClick?: (sourceId: string) => void;
}

function timeAgo(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

const MARK_CLASS =
  "bg-[rgba(212,136,74,0.18)] text-[var(--mem-text)] rounded-sm px-[1px]";

export default function MemorySearchResult({
  result,
  query,
  onClick,
}: MemorySearchResultProps) {
  const isKeyword = hasKeywordMatch(result.content, query);
  const facetType = result.memory_type ?? null;
  const facetColor = facetType
    ? FACET_COLORS[facetType] ?? "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
    : null;

  const snippet =
    result.content.length > 280
      ? result.content.substring(0, 280) + "…"
      : result.content;

  const rel = relevanceLabel(result.score);

  return (
    <div
      onClick={() => onClick?.(result.source_id)}
      className="group px-4 py-3 rounded-lg transition-colors duration-150 hover:bg-[var(--mem-hover)] cursor-pointer"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
      }}
    >
      {/* Content with highlighting */}
      <p
        className="leading-relaxed"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "14px",
          color: "var(--mem-text)",
          lineHeight: "1.6",
        }}
      >
        {highlightTerms(snippet, query, MARK_CLASS)}
      </p>

      {/* Metadata row */}
      <div
        className="flex items-center gap-2 mt-2 flex-wrap"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "11px",
          color: "var(--mem-text-tertiary)",
        }}
      >
        {/* Match type indicator */}
        <span
          className="px-1.5 py-0.5 rounded text-[10px] font-medium"
          style={{
            backgroundColor: isKeyword
              ? "rgba(212, 136, 74, 0.12)"
              : "rgba(123, 123, 232, 0.12)",
            color: isKeyword
              ? "var(--mem-accent-warm)"
              : "var(--mem-accent-indigo)",
          }}
        >
          {isKeyword ? "keyword" : "semantic"}
        </span>

        {/* Facet badge */}
        {facetType && facetColor && (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${facetColor}`}
          >
            {facetType}
          </span>
        )}

        {/* Source agent */}
        {result.source_agent && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-500/10 text-cyan-400">
            {result.source_agent}
          </span>
        )}

        {/* Domain */}
        {result.domain && (
          <>
            <span style={{ opacity: 0.4 }}>&middot;</span>
            <span>{result.domain}</span>
          </>
        )}

        {/* Timestamp */}
        <span style={{ opacity: 0.4 }}>&middot;</span>
        <span>{timeAgo(result.last_modified)}</span>

        {/* Relevance label — right-aligned */}
        <span className="ml-auto" style={{
          color: rel.tier === "strong"
            ? "var(--mem-accent-sage)"
            : rel.tier === "good"
              ? "var(--mem-text-secondary)"
              : "var(--mem-text-tertiary)",
        }}>
          {rel.text}
        </span>
      </div>
    </div>
  );
}
