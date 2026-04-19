// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  listDecisions,
  listDecisionDomains,
  type MemoryItem,
} from "../../lib/tauri";

interface DecisionLogProps {
  onBack: () => void;
  onSelectMemory: (sourceId: string) => void;
  onSelectConcept: (conceptId: string) => void;
}

function parseStructured(mem: MemoryItem): {
  context?: string;
  alternatives?: string[];
  reversible?: boolean;
} {
  if (!mem.structured_fields) return {};
  try {
    const sf = JSON.parse(mem.structured_fields);
    return {
      context: sf.context,
      alternatives: sf.alternatives_considered
        ? sf.alternatives_considered
            .split(",")
            .map((s: string) => s.trim())
            .filter(Boolean)
        : undefined,
      reversible: sf.reversible,
    };
  } catch {
    return {};
  }
}

function formatMonthGroup(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
  });
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function groupByMonth(decisions: MemoryItem[]): Map<string, MemoryItem[]> {
  const groups = new Map<string, MemoryItem[]>();
  for (const d of decisions) {
    const key = formatMonthGroup(d.last_modified);
    const arr = groups.get(key) ?? [];
    arr.push(d);
    groups.set(key, arr);
  }
  return groups;
}

// ── Domain filter pills ────────────────────────────────────────────────

function DomainPills({
  domains,
  selected,
  onSelect,
}: {
  domains: string[];
  selected: string | null;
  onSelect: (d: string | null) => void;
}) {
  const pillBase: React.CSSProperties = {
    fontFamily: "var(--mem-font-body)",
    fontSize: "12px",
    padding: "4px 10px",
    borderRadius: "9999px",
    border: "1px solid var(--mem-border)",
    cursor: "pointer",
    transition: "all 150ms ease",
    whiteSpace: "nowrap",
  };

  return (
    <div className="flex gap-1.5 flex-wrap" style={{ marginBottom: "16px" }}>
      <button
        onClick={() => onSelect(null)}
        style={{
          ...pillBase,
          backgroundColor: selected === null ? "rgba(240, 198, 116, 0.2)" : "transparent",
          color: selected === null ? "rgba(240, 198, 116, 1)" : "var(--mem-text-tertiary)",
          borderColor: selected === null ? "rgba(240, 198, 116, 0.4)" : "var(--mem-border)",
        }}
      >
        All
      </button>
      {domains.map((d) => (
        <button
          key={d}
          onClick={() => onSelect(d === selected ? null : d)}
          style={{
            ...pillBase,
            backgroundColor: d === selected ? "rgba(240, 198, 116, 0.2)" : "transparent",
            color: d === selected ? "rgba(240, 198, 116, 1)" : "var(--mem-text-tertiary)",
            borderColor: d === selected ? "rgba(240, 198, 116, 0.4)" : "var(--mem-border)",
          }}
        >
          {d}
        </button>
      ))}
    </div>
  );
}

// ── Single decision entry ──────────────────────────────────────────────

// Returns a React fragment with TWO grid cells (timeline + content) — meant to live inside a parent grid
function DecisionEntryRow({
  memory,
  expanded,
  onToggle,
  onSelectMemory,
  isLast: _isLast,
}: {
  memory: MemoryItem;
  expanded: boolean;
  onToggle: () => void;
  onSelectMemory: (sourceId: string) => void;
  isLast: boolean;
}) {
  const { context, alternatives, reversible } = parseStructured(memory);

  // Use title if it looks reasonable, otherwise fall back to truncated content.
  // A "bad" title starts with punctuation/numbers or is very short fragments.
  const titleUsable = memory.title && memory.title.length > 5 && /^[A-Za-z]/.test(memory.title);
  const headline = titleUsable ? memory.title : memory.content.split('\n')[0];

  return (
    <>
      {/* Timeline cell: spine + dot */}
      <div
        className="relative"
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
        }}
      >
        {/* Spine — runs full height of this cell, centered */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: "50%",
            transform: "translateX(-50%)",
            width: "2px",
            backgroundColor: "var(--mem-border)",
          }}
        />
        {/* Dot — vertically centered on the title line.
            Card has p-3 (12px) top padding. Title is 14px at 1.4 line-height = 19.6px.
            Center of title = 12px + 9.8px ≈ 22px. Dot is 10px tall, so top = 22 - 5 = 17px. */}
        <div
          style={{
            position: "relative",
            marginTop: "17px",
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            backgroundColor: expanded ? "rgba(240, 198, 116, 0.9)" : "var(--mem-border)",
            border: "2px solid var(--mem-surface)",
            transition: "background-color 200ms ease",
            zIndex: 1,
            flexShrink: 0,
          }}
        />
      </div>

      {/* Content cell */}
      <div
        data-testid="decision-entry"
        onClick={onToggle}
        className="pb-4 cursor-pointer group"
      >
        <div
          className="rounded-lg p-3 transition-colors duration-150"
          style={{
            backgroundColor: expanded ? "var(--mem-surface-raised)" : "transparent",
            border: expanded ? "1px solid var(--mem-border)" : "1px solid transparent",
          }}
        >
          {/* Headline — up to 2 lines, then clamp */}
          <div
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "14px",
              color: "var(--mem-text)",
              lineHeight: "1.4",
              marginBottom: "4px",
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical" as const,
              overflow: "hidden",
            }}
          >
            {headline}
          </div>

        {/* Description preview (2 lines) — structured context if available, otherwise content */}
        {(() => {
          const description = context || memory.content;
          // Don't show description if it's identical to the headline
          if (!description || description === headline) return null;
          return (
            <div
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-secondary)",
                lineHeight: "1.5",
                display: "-webkit-box",
                WebkitLineClamp: expanded ? undefined : 2,
                WebkitBoxOrient: "vertical" as const,
                overflow: expanded ? "visible" : "hidden",
              }}
            >
              {description}
            </div>
          );
        })()}

        {/* Meta row: domain tag, date, source agent */}
        <div className="flex items-center gap-2 mt-2" style={{ flexWrap: "wrap" }}>
          {memory.domain && (
            <span
              className="px-1.5 py-0.5 rounded text-xs"
              style={{
                backgroundColor: "rgba(240, 198, 116, 0.15)",
                color: "rgba(240, 198, 116, 0.9)",
                border: "1px solid rgba(240, 198, 116, 0.25)",
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
              }}
            >
              {memory.domain}
            </span>
          )}
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {formatDate(memory.last_modified)}
          </span>
          {memory.source_agent && (
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {memory.source_agent}
            </span>
          )}
        </div>

        {/* ── Expanded section ──────────────────────────────────────── */}
        {expanded && (
          <div style={{ marginTop: "12px", borderTop: "1px solid var(--mem-border)", paddingTop: "12px" }}>
            {/* Alternatives considered */}
            {alternatives && alternatives.length > 0 && (
              <div style={{ marginBottom: "10px" }}>
                <div
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "10px",
                    color: "var(--mem-text-tertiary)",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    marginBottom: "6px",
                  }}
                >
                  Alternatives considered
                </div>
                <div className="flex gap-1.5 flex-wrap">
                  {alternatives.map((alt) => (
                    <span
                      key={alt}
                      className="px-2 py-0.5 rounded text-xs"
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        fontSize: "12px",
                        color: "var(--mem-text-tertiary)",
                        textDecoration: "line-through",
                        backgroundColor: "var(--mem-surface)",
                        border: "1px solid var(--mem-border)",
                      }}
                    >
                      {alt}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Reversibility indicator */}
            {reversible !== undefined && (
              <div className="flex items-center gap-1.5">
                <span
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    backgroundColor: reversible ? "rgba(134, 239, 172, 0.8)" : "rgba(252, 165, 165, 0.8)",
                    display: "inline-block",
                  }}
                />
                <span
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: "11px",
                    color: "var(--mem-text-tertiary)",
                  }}
                >
                  {reversible ? "Reversible" : "Not easily reversible"}
                </span>
              </div>
            )}

            {/* Open full detail link */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSelectMemory(memory.source_id);
              }}
              className="mt-2 text-xs transition-colors duration-150 hover:underline"
              style={{
                fontFamily: "var(--mem-font-body)",
                color: "rgba(240, 198, 116, 0.8)",
                background: "none",
                border: "none",
                padding: 0,
                cursor: "pointer",
              }}
            >
              View full memory
            </button>
          </div>
        )}
        </div>
      </div>
    </>
  );
}

// ── Main component ─────────────────────────────────────────────────────

export default function DecisionLog({ onBack, onSelectMemory, onSelectConcept: _onSelectConcept }: DecisionLogProps) {
  const [domainFilter, setDomainFilter] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: decisions = [] } = useQuery({
    queryKey: ["decisions", domainFilter],
    queryFn: () => listDecisions(domainFilter ?? undefined, 100),
  });

  const { data: domains = [] } = useQuery({
    queryKey: ["decisionDomains"],
    queryFn: listDecisionDomains,
  });

  const monthGroups = groupByMonth(decisions);

  return (
    <div className="flex flex-col gap-4 max-w-2xl mx-auto py-4">
      {/* Back + Header */}
      <div>
        <button
          onClick={onBack}
          aria-label="Go back"
          className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{
            color: "var(--mem-text-tertiary)",
            background: "none",
            border: "none",
            cursor: "pointer",
            lineHeight: 0,
            marginBottom: "12px",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
        </button>

        <div className="flex items-center gap-3">
          <h1
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "20px",
              color: "var(--mem-text)",
              margin: 0,
              fontWeight: 500,
            }}
          >
            Decisions
          </h1>
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "12px",
              color: "var(--mem-text-tertiary)",
              backgroundColor: "var(--mem-surface)",
              padding: "2px 8px",
              borderRadius: "9999px",
              border: "1px solid var(--mem-border)",
            }}
          >
            {decisions.length}
          </span>
        </div>
      </div>

      {/* Domain filter pills */}
      <DomainPills domains={domains} selected={domainFilter} onSelect={setDomainFilter} />

      {/* Timeline */}
      {decisions.length === 0 ? (
        <div
          className="text-center py-12"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          No decisions recorded yet
        </div>
      ) : (
        <div>
          {Array.from(monthGroups.entries()).map(([month, items]) => (
            <div key={month} style={{ marginBottom: "24px" }}>
              {/* Whole month group as a single grid — spine is continuous */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "24px 1fr",
                }}
              >
                {/* Month header spans into content column */}
                <div /> {/* empty timeline cell for header row */}
                <div
                  style={{
                    fontFamily: "var(--mem-font-heading)",
                    fontSize: "13px",
                    color: "var(--mem-text-tertiary)",
                    marginBottom: "12px",
                  }}
                >
                  {month}
                </div>

                {/* Entries — timeline column + content column per entry */}
                {items.map((mem, idx) => (
                  <DecisionEntryRow
                    key={mem.source_id}
                    memory={mem}
                    expanded={expandedId === mem.source_id}
                    onToggle={() =>
                      setExpandedId(expandedId === mem.source_id ? null : mem.source_id)
                    }
                    onSelectMemory={onSelectMemory}
                    isLast={idx === items.length - 1}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
