// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  listAgentActivity,
  listAgents,
  type AgentActivityItem,
} from "../../lib/tauri";
import { resolveAgentDisplayName } from "../../lib/agents";

interface ActivityFeedProps {
  onNavigateMemory: (sourceId: string) => void;
}

function relativeTime(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

type TimeGroup = "Today" | "Yesterday" | "This Week" | "Older";

function getTimeGroup(ts: number): TimeGroup {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const yesterdayStart = todayStart - 86400;
  const weekStart = todayStart - 6 * 86400;

  if (ts >= todayStart) return "Today";
  if (ts >= yesterdayStart) return "Yesterday";
  if (ts >= weekStart) return "This Week";
  return "Older";
}

function accentColor(action: string): string {
  // Distinct accent per MCP tool / backend action type
  switch (action) {
    case "store":
      return "var(--mem-accent-indigo)"; // remember
    case "search":
      return "var(--mem-accent-glow)"; // recall
    case "read":
      return "var(--mem-accent-sage)"; // context
    case "refine":
      return "var(--mem-accent-amber)"; // refinement / merge
    case "forget":
      return "var(--mem-accent-warm)"; // destructive
    case "concept_create":
    case "concept_grow":
      return "var(--mem-accent-concept)"; // knowledge compilation
    case "steep":
      return "var(--mem-accent-amber)"; // steeping — background consolidation
    default:
      return "var(--mem-accent-indigo)";
  }
}

function naturalLanguage(item: AgentActivityItem): string {
  const count = item.memory_titles.length;
  const memoryWord = count === 1 ? "memory" : "memories";

  switch (item.action) {
    case "read":
      if (count > 0) {
        return `pulled ${count} ${memoryWord} into context`;
      }
      return "loaded your context";
    case "search":
      if (item.query) {
        return count > 0
          ? `recalled ${count} ${memoryWord}`
          : "searched your memory";
      }
      return "searched your memory";
    case "store":
      if (count > 0) {
        return `remembered ${count} ${memoryWord}`;
      }
      return "stored a memory";
    case "refine":
      if (count > 0) {
        return `refined your ${item.memory_titles[0] ? `"${truncate(item.memory_titles[0], 40)}"` : "memory"}`;
      }
      return "updated a memory with new reasoning";
    case "forget":
      // Memory is gone after delete, so memory_titles will be empty —
      // title is carried in `detail` instead (see handle_delete_memory).
      return "forgot a memory";
    case "concept_grow":
      // Concept title carried in `detail` (concepts aren't in the memories table,
      // so the title lookup in list_agent_activity won't find them).
      return "grew a concept";
    case "concept_create":
      return count > 0
        ? `compiled ${count} ${memoryWord} into a new concept`
        : "created a new concept";
    case "steep":
      // The backend headline IS the natural language text for steep events.
      // Written by classify_* functions in refinery.rs.
      return item.detail ?? "steeped your memories in the background";
    default:
      return item.detail ?? "interacted with your memory";
  }
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max).trimEnd() + "…";
}

function capitalize(s: string): string {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/// Extract the concept title out of the activity `detail` field. Concept
/// events carry their title quoted inside `detail` because concept titles
/// don't live in the `memories` table and can't be resolved by the activity
/// lister's title lookup (see memory_routes.rs `handle_delete_memory` and
/// post_ingest `grow_concept` / refinery `distill_concepts`). Format:
///   `grew "React Hooks"`
///   `created "React Hooks" from 5 memories`
function extractConceptTitle(detail: string | null): string | null {
  if (!detail) return null;
  const m = detail.match(/"([^"]+)"/);
  return m ? m[1] : null;
}

/// Short, human-readable label for the action — used in filter pills and as
/// a prominent tag on each entry. Keep aligned with `naturalLanguage()` above.
function actionLabel(action: string): string {
  switch (action) {
    case "store":
      return "Remember";
    case "search":
      return "Recall";
    case "read":
      return "Context";
    case "refine":
      return "Refine";
    case "forget":
      return "Forget";
    case "concept_create":
      return "New concept";
    case "concept_grow":
      return "Grow concept";
    case "steep":
      return "Steep";
    default:
      return action;
  }
}

function groupActivities(items: AgentActivityItem[]): [TimeGroup, AgentActivityItem[]][] {
  const groups = new Map<TimeGroup, AgentActivityItem[]>();
  const order: TimeGroup[] = ["Today", "Yesterday", "This Week", "Older"];

  for (const item of items) {
    const group = getTimeGroup(item.timestamp);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)!.push(item);
  }

  return order.filter((g) => groups.has(g)).map((g) => [g, groups.get(g)!]);
}

// Parse comma-separated memory_ids string into array
function parseMemoryIds(memoryIds: string | null): string[] {
  if (!memoryIds) return [];
  return memoryIds.split(",").map((s) => s.trim()).filter(Boolean);
}

// ── Filter dropdown ───────────────────────────────────────────────────
// Matches MemoryStream's toolbar select style exactly (transparent bg,
// --mem-border 1px 6px-radius, --mem-font-body 12px, indigo accent when
// active). Keep in sync if MemoryStream's toolbar chrome changes.

function FilterSelect({
  label,
  items,
  selected,
  renderLabel,
  onSelect,
}: {
  label: string;
  items: string[];
  selected: string | null;
  renderLabel?: (item: string) => string;
  onSelect: (value: string | null) => void;
}) {
  const display = renderLabel ?? ((s: string) => s);
  return (
    <select
      value={selected ?? ""}
      onChange={(e) => onSelect(e.target.value || null)}
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: "12px",
        color: selected ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)",
        backgroundColor: "transparent",
        border: "1px solid var(--mem-border)",
        borderRadius: "6px",
        padding: "5px 8px",
        cursor: "pointer",
        outline: "none",
      }}
      aria-label={label}
    >
      <option value="">{label}</option>
      {items.map((a) => (
        <option key={a} value={a}>
          {display(a)}
        </option>
      ))}
    </select>
  );
}

export default function ActivityFeed({ onNavigateMemory }: ActivityFeedProps) {
  const { data: activities = [] } = useQuery({
    queryKey: ["agentActivity"],
    queryFn: () => listAgentActivity(100),
    refetchInterval: 15000,
    staleTime: 30000,
  });

  // Connected Agents — used to resolve friendly display names. User overrides
  // here always beat the built-in KNOWN_CLIENTS registry.
  const { data: connectedAgents = [] } = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    staleTime: 60000,
  });

  const [agentFilter, setAgentFilter] = useState<string | null>(null);
  const [actionFilter, setActionFilter] = useState<string | null>(null);

  // Derive unique agent + action lists from what's actually in the feed —
  // ordered by activity count (descending) so the most-active pills sit leftmost.
  // Diagnostic fallback names written by the daemon when no real agent could
  // be resolved. They're not user-facing agents — hide them from the filter.
  // See `extract_agent_name` in `crates/origin-server/src/memory_routes.rs`.
  const FALLBACK_AGENT_NAMES = new Set([
    "api",
    "unknown",
    "system",
    "test-agent",
    "smoke-test",
  ]);

  const agents = useMemo(() => {
    const counts = new Map<string, number>();
    // Seed from observed activity (the 100-event window).
    for (const a of activities) {
      if (FALLBACK_AGENT_NAMES.has(a.agent_name)) continue;
      counts.set(a.agent_name, (counts.get(a.agent_name) ?? 0) + 1);
    }
    // Union with registered agents — ensures agents you've explicitly set up
    // appear in the filter even when they haven't written anything within the
    // last 100 events (e.g. Cursor with 2 events from three weeks ago used to
    // disappear from the dropdown because pagination clipped it out).
    for (const c of connectedAgents) {
      if (FALLBACK_AGENT_NAMES.has(c.name)) continue;
      if (!counts.has(c.name)) counts.set(c.name, 0);
    }
    return [...counts.entries()]
      .sort((a, b) => {
        // Agents with real activity sort first (by count desc), then zero-count
        // registered agents alphabetically.
        if (b[1] !== a[1]) return b[1] - a[1];
        return a[0].localeCompare(b[0]);
      })
      .map(([name]) => name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activities, connectedAgents]);

  const actions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const a of activities) {
      counts.set(a.action, (counts.get(a.action) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([name]) => name);
  }, [activities]);

  // Reset filters if their selected value no longer appears in the feed
  // (e.g., paginated out after a refetch).
  useEffect(() => {
    if (agentFilter !== null && !agents.includes(agentFilter)) {
      setAgentFilter(null);
    }
  }, [agentFilter, agents]);
  useEffect(() => {
    if (actionFilter !== null && !actions.includes(actionFilter)) {
      setActionFilter(null);
    }
  }, [actionFilter, actions]);

  const visibleActivities = activities.filter((a) => {
    if (agentFilter && a.agent_name !== agentFilter) return false;
    if (actionFilter && a.action !== actionFilter) return false;
    return true;
  });

  if (activities.length === 0) {
    return (
      <div
        className="flex flex-col items-center justify-center h-full"
        style={{ minHeight: 300 }}
      >
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "14px",
            color: "var(--mem-text-tertiary)",
            textAlign: "center",
            maxWidth: 320,
            lineHeight: 1.6,
          }}
        >
          Your AI tools will appear here as they interact with your memory.
        </p>
      </div>
    );
  }

  const grouped = groupActivities(visibleActivities);

  const agentLabelForMessage = agentFilter
    ? resolveAgentDisplayName(agentFilter, connectedAgents)
    : null;
  const emptyReason = actionFilter
    ? `No ${actionLabel(actionFilter).toLowerCase()} activity${
        agentLabelForMessage ? ` from ${agentLabelForMessage}` : ""
      } in the last 100 events.`
    : agentLabelForMessage
      ? `No activity from ${agentLabelForMessage} in the last 100 events.`
      : null;

  const hasActionFilter = actions.length > 1;
  const hasAgentFilter = agents.length > 1;
  const showToolbar = hasActionFilter || hasAgentFilter;

  return (
    <div className="flex flex-col">
      {/* Toolbar — right-aligned dropdowns, same pattern as MemoryStream. */}
      {showToolbar && (
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "center",
            gap: 8,
            marginBottom: 20,
          }}
        >
          {hasActionFilter && (
            <FilterSelect
              label="Action"
              items={actions}
              selected={actionFilter}
              renderLabel={actionLabel}
              onSelect={setActionFilter}
            />
          )}
          {hasAgentFilter && (
            <FilterSelect
              label="Agent"
              items={agents}
              selected={agentFilter}
              renderLabel={(id) => resolveAgentDisplayName(id, connectedAgents)}
              onSelect={setAgentFilter}
            />
          )}
        </div>
      )}
      {grouped.length === 0 && emptyReason ? (
        <div
          className="px-4 py-8 text-center rounded-lg flex flex-col items-center gap-3"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-tertiary)",
            border: "1px solid var(--mem-border)",
          }}
        >
          <span>{emptyReason}</span>
          <button
            onClick={() => {
              setActionFilter(null);
              setAgentFilter(null);
            }}
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              fontWeight: 500,
              color: "var(--mem-accent-indigo)",
              background: "transparent",
              border: "1px solid var(--mem-accent-indigo)",
              borderRadius: "6px",
              padding: "5px 12px",
              cursor: "pointer",
            }}
          >
            Clear filters
          </button>
        </div>
      ) : null}
      <div className="flex flex-col gap-8">
      {grouped.map(([group, items]) => (
        <section key={group}>
          <h3
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              fontWeight: 600,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              color: "var(--mem-text-tertiary)",
              marginBottom: 16,
            }}
          >
            {group}
          </h3>

          <div className="flex flex-col" style={{ gap: 28 }}>
            {items.map((item, i) => (
              <ActivityEntry
                key={item.id}
                item={item}
                index={i}
                connectedAgents={connectedAgents}
                onNavigateMemory={onNavigateMemory}
              />
            ))}
          </div>
        </section>
      ))}
      </div>
    </div>
  );
}

function ActivityEntry({
  item,
  index,
  connectedAgents,
  onNavigateMemory,
}: {
  item: AgentActivityItem;
  index: number;
  connectedAgents: import("../../lib/tauri").AgentConnection[];
  onNavigateMemory: (sourceId: string) => void;
}) {
  const color = accentColor(item.action);
  const memoryIds = parseMemoryIds(item.memory_ids);
  const isConceptEvent =
    item.action === "concept_create" || item.action === "concept_grow";
  const conceptTitle = isConceptEvent ? extractConceptTitle(item.detail) : null;
  const agentDisplay = resolveAgentDisplayName(item.agent_name, connectedAgents);

  return (
    <div
      className="flex gap-3"
      style={{
        animation: `mem-fade-up 0.3s ease-out ${index * 0.03}s both`,
      }}
    >
      {/* Colored dot */}
      <div className="flex flex-col items-center pt-1.5">
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: color }}
        />
      </div>

      {/* Content */}
      <div className="flex flex-col gap-1 min-w-0 flex-1">
        {/* Action label (prominent) + timestamp.
            Action is the primary axis — this is what the user cares about. */}
        <div className="flex items-baseline gap-2">
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              fontWeight: 600,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              color: color,
            }}
          >
            {actionLabel(item.action)}
          </span>
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              color: "var(--mem-text-tertiary)",
              marginLeft: "auto",
              whiteSpace: "nowrap",
            }}
          >
            {relativeTime(item.timestamp)}
          </span>
        </div>

        {/* Action phrase — the sentence the user reads first. */}
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "14px",
            fontWeight: 500,
            color: "var(--mem-text)",
            lineHeight: 1.4,
          }}
        >
          {capitalize(naturalLanguage(item))}
        </span>

        {/* Agent attribution — secondary. Shows the friendly display name
            (e.g. "Claude Code") while the filter still matches the canonical
            technical ID (e.g. "claude-code"). */}
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "11px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          by {agentDisplay}
        </span>

        {/* Search query */}
        {item.action === "search" && item.query && (
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              fontStyle: "italic",
              color: "var(--mem-text-tertiary)",
            }}
          >
            &ldquo;{item.query}&rdquo;
          </span>
        )}

        {/* Concept tag — highlighted, distinct from memory pills.
            Concept events carry their title in `detail` (concepts don't live
            in the `memories` table so the title lookup can't resolve them).
            We render a pill with the concept accent border and a subtle
            distilled-bg so it's unmistakable amongst the grey memory pills. */}
        {conceptTitle && (
          <div className="flex flex-wrap gap-1.5" style={{ marginTop: 4 }}>
            <span
              className="inline-flex items-center gap-2 rounded-full"
              style={{
                padding: "5px 12px 5px 10px",
                backgroundColor: "var(--mem-distilled-bg)",
                border: "1px solid var(--mem-accent-concept)",
                boxShadow: "0 0 12px var(--mem-shimmer-color)",
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ backgroundColor: "var(--mem-accent-concept)" }}
              />
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "9px",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "var(--mem-accent-concept)",
                }}
              >
                Concept
              </span>
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                }}
              >
                {truncate(conceptTitle, 48)}
              </span>
            </span>
          </div>
        )}

        {/* Memory title pills — the source memories that drove the action.
            For concept events these are the seed memories. */}
        {item.memory_titles.length > 0 && (
          <div className="flex flex-wrap gap-1.5" style={{ marginTop: 2 }}>
            {item.memory_titles.map((title, ti) => {
              const memId = memoryIds[ti];
              return (
                <button
                  key={ti}
                  onClick={memId ? () => onNavigateMemory(memId) : undefined}
                  className="rounded-full px-2.5 py-0.5 transition-colors duration-150"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "11px",
                    color: "var(--mem-text-secondary)",
                    backgroundColor: "var(--mem-surface)",
                    border: "1px solid var(--mem-border)",
                    cursor: memId ? "pointer" : "default",
                  }}
                  onMouseEnter={(e) => {
                    if (memId) {
                      (e.target as HTMLElement).style.borderColor = color;
                      (e.target as HTMLElement).style.color = "var(--mem-text)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    (e.target as HTMLElement).style.borderColor = "var(--mem-border)";
                    (e.target as HTMLElement).style.color = "var(--mem-text-secondary)";
                  }}
                >
                  {truncate(title, 50)}
                </button>
              );
            })}
          </div>
        )}

        {/* Detail text for non-concept actions that carry their subject in
            `detail` (rather than via memory_titles lookup).
            - refine: legacy refine detail
            - forget: memory row deleted, title preserved in detail
            Concept events get the highlighted concept tag above instead. */}
        {(item.action === "refine" || item.action === "forget") &&
          item.detail && (
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {item.detail}
            </span>
          )}
      </div>
    </div>
  );
}
