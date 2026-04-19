// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import type { RetrievalEvent } from "../../lib/tauri";

interface Props {
  events: RetrievalEvent[];
  /** Navigate to a concept by its stable ID. Replaces the old title-lookup path. */
  onSelectConceptById?: (conceptId: string) => void;
  onViewRecaps?: () => void;
}

const KNOWN_AGENTS: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude-desktop": "Claude Desktop",
  cursor: "Cursor",
  "chatgpt-mcp": "ChatGPT",
  chatgpt: "ChatGPT",
  "gemini-cli": "Gemini CLI",
  windsurf: "Windsurf",
  zed: "Zed",
};

function isTrustedAgent(name: string | null | undefined): boolean {
  if (!name) return false;
  const trimmed = name.trim().toLowerCase();
  if (!trimmed) return false;
  if (
    trimmed === "unknown" ||
    trimmed === "anonymous" ||
    trimmed === "(unknown)"
  ) {
    return false;
  }
  return true;
}

function prettyAgent(name: string): string {
  const key = name.trim().toLowerCase();
  return KNOWN_AGENTS[key] ?? name;
}

function relative(ms: number): string {
  const delta = Date.now() - ms;
  const mins = Math.floor(delta / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return days === 1 ? "yesterday" : `${days}d ago`;
}

const SECTION_TITLE_STYLE: React.CSSProperties = {
  fontFamily: "var(--mem-font-heading)",
  fontSize: 19,
  fontWeight: 400,
  color: "var(--mem-text)",
  letterSpacing: "-0.005em",
  lineHeight: 1.2,
};

const SECTION_SUB_STYLE: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 12,
  fontStyle: "italic",
  color: "var(--mem-text-tertiary)",
  marginTop: 2,
};

export function RetrievalsList({ events, onSelectConceptById, onViewRecaps }: Props) {
  const trusted = events.filter((e) => isTrustedAgent(e.agent_name));
  if (!trusted.length) return null;

  return (
    <section data-testid="retrievals">
      <h2 style={SECTION_TITLE_STYLE}>Where AI looked</h2>
      <p style={SECTION_SUB_STYLE} className="mb-3">
        recent assistants pulling from your library
      </p>
      <ul className="space-y-2">
        {trusted.map((e, i) => (
          <RetrievalItem key={i} event={e} onSelectConceptById={onSelectConceptById} />
        ))}
      </ul>
      {onViewRecaps && (
        <button
          type="button"
          onClick={onViewRecaps}
          className="mt-3 transition-colors duration-150"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            color: "var(--mem-text-secondary)",
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--mem-text)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--mem-text-secondary)")}
        >
          View all recaps &rarr;
        </button>
      )}
    </section>
  );
}

function RetrievalItem({
  event,
  onSelectConceptById,
}: {
  event: RetrievalEvent;
  onSelectConceptById?: (conceptId: string) => void;
}) {
  const [hover, setHover] = useState(false);
  // Prefer concept_ids (stable) for navigation; fall back to positional index
  // which maps to concept_titles[0] when no id is available (legacy events).
  const conceptIds = (event.concept_ids ?? []).filter(Boolean);
  const concepts = event.concept_titles.filter(Boolean);
  const memories = (event.memory_snippets ?? []).filter(Boolean);
  const hasContent = concepts.length > 0 || memories.length > 0;

  // The first navigable concept ID for this event. Legacy events (no concept_ids)
  // produce an empty string, which the click handler guards against.
  const primaryConceptId = conceptIds[0] ?? "";

  // Cards with results are clickable (productive retrieval).
  // Cards with no results are informational only (dry run).
  if (hasContent) {
    return (
      <li
        data-testid="retrieval-item"
        className="rounded-lg border px-4 py-3 transition-colors"
        role="button"
        tabIndex={0}
        style={{
          backgroundColor: hover ? "var(--mem-hover)" : "transparent",
          borderColor: "var(--mem-border)",
          cursor: primaryConceptId ? "pointer" : "default",
        }}
        onClick={() => {
          if (primaryConceptId) onSelectConceptById?.(primaryConceptId);
        }}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && primaryConceptId) {
            e.preventDefault();
            onSelectConceptById?.(primaryConceptId);
          }
        }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        <RetrievalItemBody event={event} concepts={concepts} memories={memories} archived={!primaryConceptId && concepts.length > 0} />
      </li>
    );
  }

  // Dry-run: informational only, muted opacity, no pointer.
  return (
    <li
      data-testid="retrieval-item"
      className="rounded-lg border px-4 py-3"
      style={{
        backgroundColor: "var(--mem-surface)",
        borderColor: "var(--mem-border)",
        opacity: 0.75,
      }}
    >
      <RetrievalItemBody event={event} concepts={concepts} memories={memories} />
    </li>
  );
}

function RetrievalItemBody({
  event,
  concepts,
  memories,
  archived,
}: {
  event: RetrievalEvent;
  concepts: string[];
  memories: string[];
  archived?: boolean;
}) {
  const hasContent = concepts.length > 0 || memories.length > 0;
  return (
    <>
      <div className="flex items-baseline gap-2 mb-1.5">
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 11,
            color: "var(--mem-text-tertiary)",
            whiteSpace: "nowrap",
          }}
        >
          {relative(event.timestamp_ms)}
        </span>
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 11,
            fontWeight: 500,
            color: "var(--mem-text-secondary)",
            whiteSpace: "nowrap",
          }}
        >
          {prettyAgent(event.agent_name)}
        </span>
        {event.query && (
          <span
            className="flex-1 truncate"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 11,
              fontStyle: "italic",
              color: "var(--mem-text-tertiary)",
            }}
          >
            on "{event.query}"
          </span>
        )}
      </div>
      {hasContent ? (
        <div className="flex items-start gap-2">
          <p
            className="line-clamp-2 flex-1"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: 14,
              fontWeight: 500,
              color: "var(--mem-text)",
              lineHeight: 1.4,
            }}
          >
            {concepts.length > 0
              ? concepts.map((t) => `"${t}"`).join(" · ")
              : memories.map((m) => `"${m}"`).join(" · ")}
          </p>
          {archived && (
            <span
              title="This concept has been archived"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: 10,
                color: "var(--mem-text-tertiary)",
                background: "var(--mem-hover)",
                padding: "2px 6px",
                borderRadius: "3px",
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
            >
              archived
            </span>
          )}
        </div>
      ) : (
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            fontStyle: "italic",
            color: "var(--mem-text-tertiary)",
          }}
        >
          searched, found nothing relevant
        </p>
      )}
    </>
  );
}
