// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import type {
  PageChangelogEntry,
  PageCitation,
  PageLinkInbound,
  PageSourceWithMemory,
} from "../../../lib/tauri";
import type { CitationState } from "../../../lib/pageCitations";
import { prettyAgent, relativeMs, sourceKindLabel } from "./format";

interface PageInfoProps {
  sourceCount: number;
  sources: PageSourceWithMemory[] | undefined;
  inbound: PageLinkInbound[];
  revisions: PageChangelogEntry[];
  citations: PageCitation[] | undefined;
  citationState: CitationState;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
}

// Content chips already carry per-claim provenance; inside this panel the
// long lists stay capped so the unique sections read without scrolling.
const SOURCES_SHOWN = 5;
const REVISIONS_SHOWN = 3;

const showAllStyle = {
  fontFamily: "var(--mem-font-mono)",
  fontSize: "10px",
  color: "var(--mem-text-secondary)",
  cursor: "pointer",
  padding: "4px 2px",
  background: "none",
  border: "none",
  textAlign: "left",
} as const;

const groupHeading = {
  fontFamily: "var(--mem-font-mono)",
  fontSize: "10px",
  fontWeight: 600,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
} as const;

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Cited rows first (by first occurrence), then uncited by recency. */
function sortSourceRows(
  rows: PageSourceWithMemory[],
  citations: PageCitation[] | undefined,
): PageSourceWithMemory[] {
  const firstOccurrence = new Map<string, number>();
  for (const c of [...(citations ?? [])].sort((a, b) => a.occurrence - b.occurrence)) {
    if (c.source_kind === "memory" && !firstOccurrence.has(c.locator)) {
      firstOccurrence.set(c.locator, c.occurrence);
    }
  }
  return [...rows].sort((a, b) => {
    const ao = firstOccurrence.get(a.source.memory_source_id);
    const bo = firstOccurrence.get(b.source.memory_source_id);
    if (ao != null && bo != null) return ao - bo;
    if (ao != null) return -1;
    if (bo != null) return 1;
    return (b.memory?.last_modified ?? 0) - (a.memory?.last_modified ?? 0);
  });
}

export default function PageInfo({
  sourceCount,
  sources,
  inbound,
  revisions,
  citations,
  citationState,
  onMemoryClick,
  onPageClick,
}: PageInfoProps) {
  const rows = sortSourceRows(
    (sources ?? []).filter((s) => s.memory !== null),
    citations,
  );
  const [showAllSources, setShowAllSources] = useState(false);
  const [showAllRevisions, setShowAllRevisions] = useState(false);
  const visibleRows = showAllSources ? rows : rows.slice(0, SOURCES_SHOWN);
  // Daemon order varies; newest revision is what the panel is opened for.
  const revisionsDesc = [...revisions].sort((a, b) => b.version - a.version);
  const visibleRevisions = showAllRevisions
    ? revisionsDesc
    : revisionsDesc.slice(0, REVISIONS_SHOWN);
  const unverifiedLocators = new Set(
    (citations ?? []).filter((c) => c.status === "unverified").map((c) => c.locator),
  );
  const unverifiedCount = (citations ?? []).filter(
    (c) => c.status === "unverified",
  ).length;
  const diagnosability =
    citationState === "cited"
      ? `Citations: ${(citations ?? []).length} (${unverifiedCount} unverified)`
      : citationState === "stripped-empty"
        ? "Citations cleared by edit — re-distill to restore"
        : citationState === "stripped-mismatch"
          ? "Citation data mismatched — re-distill to repair"
          : null;

  return (
    <details
      aria-label="Page info"
      className="rounded-lg"
      style={{ border: "1px solid var(--mem-border)" }}
    >
      <summary
        className="flex items-center gap-2 px-4 py-3 cursor-pointer select-none list-none"
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: "var(--mem-text-tertiary)",
        }}
      >
        <span>Page info</span>
        <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>
          {plural(inbound.length, "backlink")} · {plural(revisions.length, "revision")}{" "}
          · {plural(sourceCount, "source")}
        </span>
      </summary>
      <div className="flex flex-col gap-4 px-4 pb-4">
        {inbound.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Backlinks
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {inbound.map((link, idx) => (
                <button
                  key={`${link.source_page_id}-${idx}`}
                  onClick={() => onPageClick?.(link.source_page_id)}
                  className="rounded-md px-2.5 py-1.5 transition-colors duration-150 cursor-pointer hover:bg-[var(--mem-hover)]"
                  style={{
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    fontWeight: 500,
                    color: "var(--mem-text)",
                  }}
                >
                  {link.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {revisions.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Revisions
            </h4>
            <div className="flex flex-col gap-1.5">
              {visibleRevisions.map((entry) => {
                const incomingCount = entry.incoming_source_ids?.length ?? 0;
                return (
                  <div
                    key={`${entry.version}-${entry.at}`}
                    className="rounded-lg px-3 py-2"
                    style={{
                      backgroundColor: "var(--mem-surface)",
                      border: "1px solid var(--mem-border)",
                    }}
                  >
                    <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "11px",
                          fontWeight: 600,
                          color: "var(--mem-accent-page)",
                        }}
                      >
                        v{entry.version}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "12px",
                          color: "var(--mem-text-secondary)",
                        }}
                      >
                        {entry.edited_by}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                        }}
                      >
                        {relativeMs(entry.at * 1000)}
                      </span>
                      {incomingCount > 0 && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-text-tertiary)",
                          }}
                        >
                          {incomingCount} incoming{" "}
                          {incomingCount === 1 ? "memory" : "memories"}
                        </span>
                      )}
                      {entry.citations_summary && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-text-tertiary)",
                            background: "var(--mem-hover)",
                            padding: "1px 5px",
                            borderRadius: "3px",
                          }}
                        >
                          {entry.citations_summary}
                        </span>
                      )}
                    </div>
                    {entry.delta_summary && (
                      <p
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "13px",
                          color: "var(--mem-text)",
                          lineHeight: "1.5",
                        }}
                      >
                        {entry.delta_summary}
                      </p>
                    )}
                  </div>
                );
              })}
              {revisions.length > REVISIONS_SHOWN && !showAllRevisions && (
                <button
                  type="button"
                  style={showAllStyle}
                  onClick={() => setShowAllRevisions(true)}
                >
                  Show all {revisions.length} revisions
                </button>
              )}
            </div>
          </div>
        )}
        {rows.length > 0 && (
          <div>
            <h4 className="mb-1" style={groupHeading}>
              Sources
            </h4>
            <ul>
              {visibleRows.map((row, idx) => {
                const mem = row.memory!;
                const locator = row.source.memory_source_id;
                return (
                  <li
                    key={locator}
                    data-testid="page-info-source-row"
                    className="py-2 px-2 transition-colors duration-150 hover:bg-[var(--mem-hover)]"
                    style={{
                      borderBottom:
                        idx === visibleRows.length - 1
                          ? "none"
                          : "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
                      cursor: "pointer",
                    }}
                    onClick={() => onMemoryClick(locator)}
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                          background: "var(--mem-hover)",
                          padding: "1px 5px",
                          borderRadius: "3px",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {locator}
                      </span>
                      {mem.title && (
                        <span
                          className="truncate"
                          style={{
                            fontFamily: "var(--mem-font-heading)",
                            fontSize: "13px",
                            fontWeight: 500,
                            color: "var(--mem-text)",
                          }}
                        >
                          {mem.title}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      {mem.last_modified && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-body)",
                            fontSize: "11px",
                            color: "var(--mem-text-tertiary)",
                          }}
                        >
                          {relativeMs(mem.last_modified * 1000)}
                        </span>
                      )}
                      <span
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: "11px",
                          color: "var(--mem-text-secondary)",
                        }}
                      >
                        {prettyAgent(mem.source_agent)}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mem-font-mono)",
                          fontSize: "10px",
                          color: "var(--mem-text-tertiary)",
                        }}
                      >
                        {sourceKindLabel(mem)}
                      </span>
                      {mem.version != null && mem.version > 1 && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-accent-blue, #60a5fa)",
                          }}
                        >
                          v{mem.version}
                        </span>
                      )}
                      {unverifiedLocators.has(locator) && (
                        <span
                          style={{
                            fontFamily: "var(--mem-font-mono)",
                            fontSize: "10px",
                            color: "var(--mem-accent-amber)",
                          }}
                        >
                          unverified
                        </span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
            {rows.length > SOURCES_SHOWN && !showAllSources && (
              <button
                type="button"
                style={showAllStyle}
                onClick={() => setShowAllSources(true)}
              >
                Show all {rows.length} sources
              </button>
            )}
          </div>
        )}
        {diagnosability && (
          <p
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {diagnosability}
          </p>
        )}
      </div>
    </details>
  );
}
