// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect } from "react";
import { FACET_COLORS, STABILITY_TIERS, type MemoryItem, type MemoryVersionItem, type PendingRevision, getPendingRevision, acceptPendingRevision, dismissPendingRevision } from "../../lib/tauri";
import ContentRenderer from "./ContentRenderer";

const AGENT_DISPLAY: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude": "Claude",
  "chatgpt": "ChatGPT",
  "cursor": "Cursor",
};

interface MemoryCardProps {
  memory: MemoryItem;
  onConfirm: (sourceId: string, confirmed: boolean) => void;
  onDelete: (sourceId: string) => void;
  expandedChain: boolean;
  onToggleChain: () => void;
  versionChain: MemoryVersionItem[];
  onPin?: (sourceId: string) => void;
  onUnpin?: (sourceId: string) => void;
  onClick?: (sourceId: string) => void;
  style?: React.CSSProperties;
  variant?: "full" | "insight";
  lineClamp?: number;
  hideBorderBottom?: boolean;
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

export default function MemoryCard({
  memory,
  onConfirm,
  onDelete,
  expandedChain: _expandedChain,
  onToggleChain: _onToggleChain,
  versionChain: _versionChain,
  onPin,
  onUnpin,
  onClick,
  style,
  variant = "full",
  lineClamp,
  hideBorderBottom,
}: MemoryCardProps) {
  const [deleting, setDeleting] = useState(false);
  const [pendingRevision, setPendingRevision] = useState<PendingRevision | null>(null);

  const facetType = memory.memory_type ?? "fact";
  const isRecap = memory.is_recap === true;
  const tier = STABILITY_TIERS[facetType] ?? "ephemeral";
  const confidence = memory.confidence ?? 0;
  const isConfirmed = memory.stability === "confirmed" || (!memory.stability && memory.confirmed);
  const stability = memory.stability ?? (memory.confirmed ? "confirmed" : "new");

  // Fetch pending revision for Protected confirmed memories
  useEffect(() => {
    if (tier === "protected" && isConfirmed) {
      getPendingRevision(memory.source_id).then(setPendingRevision).catch(() => {});
    }
  }, [tier, isConfirmed, memory.source_id]);

  const handleAcceptRevision = async () => {
    if (!pendingRevision) return;
    await acceptPendingRevision(memory.source_id);
    setPendingRevision(null);
    window.dispatchEvent(new CustomEvent("memory-updated"));
  };

  const handleDismissRevision = async () => {
    if (!pendingRevision) return;
    await dismissPendingRevision(memory.source_id);
    setPendingRevision(null);
  };
  const isNew = (Date.now() / 1000 - memory.last_modified) < 86400;
  const isSuperseded = memory.supersede_mode === "archive" && memory.supersedes != null;

  // Prefer source_text (original prose) over content (may be pipe-delimited structured fields)
  const displayText = memory.source_text || memory.summary || memory.content;

  // Maturity tier: raw → structured → distilled
  const hasStructuredFields = memory.structured_fields != null && memory.structured_fields !== "{}";
  const hasEntityLink = memory.entity_id != null;
  const hasClassification = memory.memory_type != null && memory.memory_type !== "fact";
  const isDistilled = memory.supersedes != null && memory.supersede_mode !== "archive";

  type MaturityTier = "raw" | "structured" | "distilled";
  const maturity: MaturityTier = isDistilled
    ? "distilled"
    : (hasClassification && (hasStructuredFields || hasEntityLink))
      ? "structured"
      : "raw";
  const borderColor = isRecap
    ? "var(--mem-accent-indigo)"
    : isConfirmed
      ? "var(--mem-accent-warm)"
      : maturity === "distilled"
        ? "var(--mem-distilled-border)"
        : confidence > 0.5
          ? "var(--mem-accent-amber)"
          : "transparent";

  if (deleting) return null;

  // ── Recap card: editorial digest treatment ──
  if (isRecap) {
    return (
      <div
        className="group relative"
        style={{
          borderLeft: `3px solid var(--mem-accent-indigo)`,
          ...style,
        }}
      >
        <div
          className="py-3.5 pr-4 transition-colors duration-150 cursor-pointer"
          style={{
            paddingLeft: "20px",
            backgroundColor: "var(--mem-indigo-bg)",
            borderBottom: "1px solid var(--mem-border)",
          }}
          onClick={() => onClick?.(memory.source_id)}
        >
          {/* Recap header with icon */}
          <div className="flex items-center gap-2 mb-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--mem-accent-indigo)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "10px",
                fontWeight: 500,
                letterSpacing: "0.05em",
                textTransform: "uppercase" as const,
                color: "var(--mem-accent-indigo)",
              }}
            >
              Activity Recap
            </span>
            <span style={{ flex: 1 }} />
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "10px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {timeAgo(memory.last_modified)}
            </span>
          </div>

          {/* Recap title */}
          <p
            className="leading-relaxed line-clamp-2"
            style={{
              color: "var(--mem-text)",
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              lineHeight: "1.6",
            }}
          >
            {memory.title}
          </p>

          {/* Domain + burst meta */}
          <div
            className="flex items-center gap-2 mt-2 flex-wrap"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {memory.domain && (
              <span
                className="px-1.5 py-0.5 rounded"
                style={{ backgroundColor: "var(--mem-indigo-bg)", color: "var(--mem-accent-indigo)" }}
              >
                {memory.domain}
              </span>
            )}
            {memory.source_agent && (
              <span>via {memory.source_agent}</span>
            )}
            {/* Extract memory count from content header */}
            {memory.content?.match(/(\d+) memories/)?.[0] && (
              <span>{memory.content.match(/(\d+) memories/)![0]}</span>
            )}
          </div>
        </div>

        {/* Hover delete */}
        <div
          className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-150"
        >
          <button
            onClick={() => { setDeleting(true); onDelete(memory.source_id); }}
            className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
            style={{ color: "var(--mem-text-tertiary)" }}
            title="Delete recap"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  // ── Standard memory card ──
  return (
    <div
      className="group relative h-full flex flex-col"
      style={{
        borderLeft: `3px solid ${borderColor}`,
        opacity: isSuperseded
          ? 0.55
          : stability === "confirmed" || maturity === "distilled"
            ? 1
            : stability === "learned"
              ? 0.85
              : Math.max(0.4, Math.min(0.85, 0.4 + confidence * 0.45)),
        animation: isNew ? "mem-shimmer 8s ease-in-out infinite" : undefined,
        ...style,
      }}
    >
      <div
        className="py-4 pr-4 transition-colors duration-150 flex-1"
        style={{
          paddingLeft: "20px",
          backgroundColor: isConfirmed ? "var(--mem-confirm-bg)"
            : maturity === "distilled" ? "var(--mem-distilled-bg)"
            : "transparent",
          borderBottom: hideBorderBottom ? "none" : "1px solid var(--mem-border)",
        }}
      >
        {/* Confirm dot */}
        <div className="flex items-start gap-3">
          {variant === "full" && (
            <button
              data-testid="confirm-dot"
              onClick={() => onConfirm(memory.source_id, !isConfirmed)}
              className="mt-1.5 flex-shrink-0 w-3 h-3 rounded-full border transition-all duration-300 hover:scale-110"
              style={{
                borderColor: stability === "confirmed"
                  ? "var(--mem-accent-warm)"
                  : stability === "learned"
                  ? "var(--mem-accent-sage)"
                  : "var(--mem-accent-amber)",
                backgroundColor: stability === "confirmed"
                  ? "var(--mem-accent-warm)"
                  : stability === "learned"
                  ? "color-mix(in srgb, var(--mem-accent-sage) 40%, transparent)"
                  : "transparent",
              }}
              title={stability === "confirmed" ? "Unconfirm" : "Confirm this memory"}
            />
          )}

          <div className="flex-1 min-w-0">
            <p
              onClick={() => onClick?.(memory.source_id)}
              className={`cursor-pointer ${lineClamp ? `line-clamp-${lineClamp}` : "truncate"}`}
              style={{
                color: "var(--mem-text)",
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                fontWeight: 400,
                lineHeight: "1.5",
              }}
            >
              {(() => {
                // Insight variant and title-only cases: keep existing behavior
                if (variant === "insight" && memory.title) return memory.title;
                if (!lineClamp) return memory.title || displayText;
                if (!memory.title || !displayText) return memory.title || displayText;

                // Use smart preview for the content portion
                const cleanTitle = memory.title.replace(/\.{2,}$/, "").trim();
                if (displayText.startsWith(cleanTitle)) {
                  return (
                    <ContentRenderer
                      content={displayText}
                      structuredFields={memory.structured_fields}
                      variant="card"
                    />
                  );
                }
                return (
                  <>
                    {memory.title} —{" "}
                    <ContentRenderer
                      content={displayText}
                      structuredFields={memory.structured_fields}
                      variant="card"
                    />
                  </>
                );
              })()}
            </p>

            {/* Metadata line */}
            {variant === "insight" ? (
              <div
                className="flex items-center gap-1.5 mt-2 flex-wrap"
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "10px",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                <span>
                  From {AGENT_DISPLAY[memory.source_agent ?? ""] ?? "Manual"}
                </span>
                {isConfirmed && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span>confirmed {"\u2713"}</span>
                  </>
                )}
                {isDistilled && !isConfirmed && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span>Refined from earlier memories</span>
                  </>
                )}
                {!isConfirmed && !isDistilled && confidence !== null && confidence < 0.5 && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span>not yet confirmed</span>
                  </>
                )}
                <span style={{ opacity: 0.4 }}>&middot;</span>
                <span>{timeAgo(memory.last_modified)}</span>
              </div>
            ) : (
              <div
                className="flex items-center gap-1.5 mt-2 flex-wrap"
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "10px",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                {/* Facet badge -- distilled cards show "distilled" instead */}
                {maturity === "distilled" ? (
                  <span
                    className="px-1 py-px rounded text-[9px] font-medium border"
                    style={{
                      backgroundColor: "var(--mem-indigo-bg)",
                      color: "var(--mem-accent-indigo)",
                      borderColor: "rgba(123,123,232,0.2)",
                    }}
                  >
                    distilled
                  </span>
                ) : (
                  <span className={`px-1 py-px rounded text-[9px] font-medium border ${FACET_COLORS[facetType]}`}>
                    {facetType}
                  </span>
                )}
                {/* Pin indicator */}
                {memory.pinned && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span className="text-amber-400">{"\u2605"} pinned</span>
                  </>
                )}
                <span style={{ opacity: 0.4 }}>&middot;</span>
                <span>{isNew ? "new" : timeAgo(memory.last_modified)}</span>
                {(memory.version ?? 1) > 1 && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span style={{ color: "var(--mem-text-tertiary)" }}>v{memory.version}</span>
                  </>
                )}
                {(memory.access_count ?? 0) >= 3 && (
                  <>
                    <span style={{ opacity: 0.4 }}>&middot;</span>
                    <span style={{ color: "var(--mem-accent-indigo)" }}>
                      used {memory.access_count}&times;
                    </span>
                  </>
                )}
              </div>
            )}


            {/* Pending revision badge */}
            {pendingRevision && (
              <div
                className="mt-2 rounded-md px-3 py-2 text-xs"
                style={{
                  backgroundColor: "rgba(245, 158, 11, 0.1)",
                  border: "1px solid rgba(245, 158, 11, 0.3)",
                  color: "rgb(245, 158, 11)",
                }}
              >
                <div className="font-medium mb-1">
                  Proposed update{pendingRevision.source_agent ? ` from ${pendingRevision.source_agent}` : ""}
                </div>
                <p className="line-clamp-2 mb-2" style={{ color: "var(--mem-text-secondary)" }}>
                  {pendingRevision.content}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleAcceptRevision}
                    className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
                    style={{
                      backgroundColor: "rgba(245, 158, 11, 0.2)",
                      color: "rgb(245, 158, 11)",
                    }}
                  >
                    Accept
                  </button>
                  <button
                    onClick={handleDismissRevision}
                    className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
                    style={{
                      backgroundColor: "rgba(161, 161, 170, 0.15)",
                      color: "var(--mem-text-tertiary)",
                    }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Action buttons (hover) */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150 shrink-0">
            {variant === "insight" ? (
              <button
                onClick={() => onClick?.(memory.source_id)}
                className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                style={{ color: "var(--mem-text-tertiary)", fontSize: "16px", lineHeight: 1 }}
                title="More actions"
              >
                &middot;&middot;&middot;
              </button>
            ) : (
              <>
                {/* Pin/Unpin button */}
                {(onPin || onUnpin) && (
                  memory.pinned ? (
                    <button
                      onClick={() => onUnpin?.(memory.source_id)}
                      className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                      style={{ color: "var(--mem-text-tertiary)" }}
                      title="Unpin memory"
                    >
                      <span className="text-amber-400 text-xs font-medium">{"\u2605"} Pinned</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => onPin?.(memory.source_id)}
                      className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                      style={{ color: "var(--mem-text-tertiary)" }}
                      title="Pin memory"
                    >
                      <span className="text-zinc-400 text-xs font-medium">{"\u2606"} Pin</span>
                    </button>
                  )
                )}

                {/* Delete button */}
                <button
                  onClick={() => {
                    setDeleting(true);
                    onDelete(memory.source_id);
                  }}
                  className="p-1 rounded transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
                  style={{ color: "var(--mem-text-tertiary)" }}
                  title="Delete memory"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
                  </svg>
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
