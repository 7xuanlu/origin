// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  listAgentActivity,
  listAgents,
  type AgentActivityItem,
} from "../../lib/tauri";
import { resolveAgentDisplayName } from "../../lib/agents";
import { Select } from "./settings/primitives";

interface ActivityFeedProps {
  onNavigateMemory: (sourceId: string) => void;
}

function relativeTime(ts: number, t: TFunction, language: string): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return t("activity.relative.justNow");
  if (diff < 3600) return t("activity.relative.minutesAgo", { count: Math.floor(diff / 60) });
  if (diff < 86400) return t("activity.relative.hoursAgo", { count: Math.floor(diff / 3600) });
  if (diff < 604800) return t("activity.relative.daysAgo", { count: Math.floor(diff / 86400) });
  return new Date(ts * 1000).toLocaleDateString(language);
}

type TimeGroup = "today" | "yesterday" | "thisWeek" | "older";

function getTimeGroup(ts: number): TimeGroup {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const yesterdayStart = todayStart - 86400;
  const weekStart = todayStart - 6 * 86400;

  if (ts >= todayStart) return "today";
  if (ts >= yesterdayStart) return "yesterday";
  if (ts >= weekStart) return "thisWeek";
  return "older";
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
    case "page_create":
    case "page_grow":
      return "var(--mem-accent-page)"; // knowledge compilation
    case "steep":
      return "var(--mem-accent-amber)"; // steeping — background consolidation
    default:
      return "var(--mem-accent-indigo)";
  }
}

function jsonRecord(value: string | null): Record<string, unknown> | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function stringField(record: Record<string, unknown> | null, field: string): string | null {
  const value = record?.[field];
  return typeof value === "string" ? value : null;
}

function refinementActionLabel(action: string | null, t: TFunction): string {
  switch (action) {
    case "entity_merge":
      return t("activity.refinementActions.entityMerge");
    case "relation_conflict":
      return t("activity.refinementActions.relationConflict");
    case "detect_contradiction":
      return t("activity.refinementActions.detectContradiction");
    case "suggest_entity":
      return t("activity.refinementActions.suggestEntity");
    case "dedup_merge":
      return t("activity.refinementActions.dedupMerge");
    default:
      return t("activity.refinementActions.unknown");
  }
}

function refinementStatusLabel(status: string | null, t: TFunction): string {
  switch (status) {
    case "dismissed":
      return t("activity.refinementStatuses.dismissed");
    case "resolved":
      return t("activity.refinementStatuses.resolved");
    case "awaiting_review":
      return t("activity.refinementStatuses.awaitingReview");
    case "auto_applied":
      return t("activity.refinementStatuses.autoApplied");
    default:
      return t("activity.refinementStatuses.updated");
  }
}

function localizedSteepHeadline(detail: string, t: TFunction, language: string): string {
  let match = detail.match(/^Wenlan resolved a memory contradiction$/);
  if (match) return t("activity.natural.steepResolvedContradictions", { count: 1 });

  match = detail.match(/^Wenlan resolved (\d+) memory contradictions$/);
  if (match) {
    return t("activity.natural.steepResolvedContradictions", {
      count: Number(match[1]),
    });
  }

  match = detail.match(/^Wenlan refreshed a page with new information$/);
  if (match) return t("activity.natural.steepRefreshPage");

  const legacyPageUnit = "con" + "cept";

  match = detail.match(
    new RegExp(`^Wenlan refreshed (\\d+) ${legacyPageUnit}s with new information$`),
  );
  if (match) {
    return t("activity.natural.steepRefreshPages", {
      count: Number(match[1]),
    });
  }

  match = detail.match(/^Wenlan steeped your memories into a new page$/);
  if (match) return t("activity.natural.steepNewPage");

  match = detail.match(
    new RegExp(`^Wenlan steeped your memories into (\\d+) new ${legacyPageUnit}s$`),
  );
  if (match) {
    return t("activity.natural.steepNewPages", {
      count: Number(match[1]),
    });
  }

  match = detail.match(/^Wenlan steeped a recent activity burst into a recap$/);
  if (match) return t("activity.natural.steepRecap", { count: 1 });

  match = detail.match(/^Wenlan steeped (\d+) recent activity bursts into recaps$/);
  if (match) return t("activity.natural.steepRecap", { count: Number(match[1]) });

  return language.startsWith("en") ? detail : t("activity.natural.steep");
}

function naturalLanguage(item: AgentActivityItem, t: TFunction, language: string): string {
  const count = item.memory_titles.length;

  switch (item.action) {
    case "read":
      if (count > 0) {
        return t("activity.natural.readWithMemories", { count });
      }
      return t("activity.natural.read");
    case "search":
      if (item.query) {
        return count > 0
          ? t("activity.natural.searchWithMemories", { count })
          : t("activity.natural.search");
      }
      return t("activity.natural.search");
    case "store":
      if (count > 0) {
        return t("activity.natural.storeWithMemories", { count });
      }
      return t("activity.natural.store");
    case "refine":
      if (count > 0) {
        const subject = item.memory_titles[0]
          ? `"${truncate(item.memory_titles[0], 40)}"`
          : t("activity.memory", { count: 1 });
        return t("activity.natural.refineTitle", { title: subject });
      }
      return t("activity.natural.refine");
    case "forget":
      // Memory is gone after delete, so memory_titles will be empty —
      // title is carried in `detail` instead (see handle_delete_memory).
      return t("activity.natural.forget");
    case "page_grow":
      // Page title carried in `detail` (pages aren't in the memories table,
      // so the title lookup in list_agent_activity won't find them).
      return t("activity.natural.pageGrow");
    case "page_create":
      return count > 0
        ? t("activity.natural.pageCreateWithMemories", { count })
        : t("activity.natural.pageCreate");
    case "page_skip_user_edited":
      return t("activity.natural.pageSkipUserEdited");
    case "entity_create":
      return t("activity.natural.entityCreate");
    case "relation_create":
      return t("activity.natural.relationCreate");
    case "relation_supersede_auto":
      return t("activity.natural.relationSupersedeAuto");
    case "observation_add":
      return t("activity.natural.observationAdd");
    case "steep":
      // The backend headline IS the natural language text for steep events.
      // Written by classify_* functions in refinery.rs.
      return item.detail
        ? localizedSteepHeadline(item.detail, t, language)
        : t("activity.natural.steep");
    case "refinement_resolve": {
      const payload = jsonRecord(item.detail);
      return t("activity.natural.refinementResolveWithStatus", {
        action: refinementActionLabel(stringField(payload, "action"), t),
        status: refinementStatusLabel(stringField(payload, "new_status"), t),
      });
    }
    case "refinement_apply": {
      const payload = jsonRecord(item.detail);
      return t("activity.natural.refinementApply", {
        action: refinementActionLabel(stringField(payload, "action"), t),
      });
    }
    case "revision_accept":
      return t("activity.natural.revisionAccept");
    case "revision_dismiss":
      return t("activity.natural.revisionDismiss");
    case "contradiction_dismiss":
      return t("activity.natural.contradictionDismiss");
    default:
      return language.startsWith("en") && item.detail
        ? item.detail
        : t("activity.natural.fallback");
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

function sentenceCaseForLocale(s: string, language: string): string {
  return language.startsWith("en") ? capitalize(s) : s;
}

/// Extract the page title out of the activity `detail` field. Page
/// events carry their title quoted inside `detail` because page titles
/// don't live in the `memories` table and can't be resolved by the activity
/// lister's title lookup. Format:
///   `grew "React Hooks"`
///   `created "React Hooks" from 5 memories`
function extractPageTitle(detail: string | null): string | null {
  if (!detail) return null;
  const m = detail.match(/"([^"]+)"/);
  return m ? m[1] : null;
}

/// Short, human-readable label for the action — used in filter pills and as
/// a prominent tag on each entry. Keep aligned with `naturalLanguage()` above.
function actionLabel(action: string, t: TFunction): string {
  switch (action) {
    case "store":
      return t("activity.actions.store");
    case "search":
      return t("activity.actions.search");
    case "read":
      return t("activity.actions.read");
    case "refine":
      return t("activity.actions.refine");
    case "forget":
      return t("activity.actions.forget");
    case "page_create":
      return t("activity.actions.pageCreate");
    case "page_grow":
      return t("activity.actions.pageGrow");
    case "page_skip_user_edited":
      return t("activity.actions.pageSkipUserEdited");
    case "entity_create":
      return t("activity.actions.entityCreate");
    case "relation_create":
      return t("activity.actions.relationCreate");
    case "relation_supersede_auto":
      return t("activity.actions.relationSupersedeAuto");
    case "observation_add":
      return t("activity.actions.observationAdd");
    case "steep":
      return t("activity.actions.steep");
    case "refinement_resolve":
      return t("activity.actions.refinementResolve");
    case "refinement_apply":
      return t("activity.actions.refinementApply");
    case "revision_accept":
      return t("activity.actions.revisionAccept");
    case "revision_dismiss":
      return t("activity.actions.revisionDismiss");
    case "contradiction_dismiss":
      return t("activity.actions.contradictionDismiss");
    default:
      return action;
  }
}

function groupActivities(items: AgentActivityItem[]): [TimeGroup, AgentActivityItem[]][] {
  const groups = new Map<TimeGroup, AgentActivityItem[]>();
  const order: TimeGroup[] = ["today", "yesterday", "thisWeek", "older"];

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
// Uses the shared `Select` primitive. `w-fit` wrapper is required: Select's
// wrapper span and inner <select> are both `w-full`, so it needs a `w-fit`
// ancestor to avoid stretching to fill this flex-end toolbar row.

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
    <div className="w-fit shrink-0">
      <Select
        size="sm"
        value={selected ?? ""}
        onChange={(e) => onSelect(e.target.value || null)}
        aria-label={label}
      >
        <option value="">{label}</option>
        {items.map((a) => (
          <option key={a} value={a}>
            {display(a)}
          </option>
        ))}
      </Select>
    </div>
  );
}

export default function ActivityFeed({ onNavigateMemory }: ActivityFeedProps) {
  const { t, i18n } = useTranslation();
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
          {t("activity.empty.noActivity")}
        </p>
      </div>
    );
  }

  const grouped = groupActivities(visibleActivities);

  const agentLabelForMessage = agentFilter
    ? resolveAgentDisplayName(agentFilter, connectedAgents)
    : null;
  const emptyReason = actionFilter
    ? agentLabelForMessage
      ? t("activity.empty.noActionFrom", {
          action: actionLabel(actionFilter, t).toLocaleLowerCase(i18n.language),
          agent: agentLabelForMessage,
        })
      : t("activity.empty.noAction", {
          action: actionLabel(actionFilter, t).toLocaleLowerCase(i18n.language),
        })
    : agentLabelForMessage
      ? t("activity.empty.noAgent", { agent: agentLabelForMessage })
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
              label={t("activity.filters.action")}
              items={actions}
              selected={actionFilter}
              renderLabel={(action) => actionLabel(action, t)}
              onSelect={setActionFilter}
            />
          )}
          {hasAgentFilter && (
            <FilterSelect
              label={t("activity.filters.agent")}
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
            {t("activity.filters.clear")}
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
            {t(`activity.groups.${group}`)}
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
  const { t, i18n } = useTranslation();
  const color = accentColor(item.action);
  const memoryIds = parseMemoryIds(item.memory_ids);
  const isPageEvent =
    item.action === "page_create" || item.action === "page_grow";
  const pageTitle = isPageEvent ? extractPageTitle(item.detail) : null;
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
            {actionLabel(item.action, t)}
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
            {relativeTime(item.timestamp, t, i18n.language)}
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
          {sentenceCaseForLocale(naturalLanguage(item, t, i18n.language), i18n.language)}
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
          {t("activity.byAgent", { agent: agentDisplay })}
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

        {/* Page tag — highlighted, distinct from memory pills.
            Page events carry their title in `detail` (pages don't live
            in the `memories` table so the title lookup can't resolve them).
            We render a pill with the page accent border and a subtle
            distilled-bg so it's unmistakable amongst the grey memory pills. */}
        {pageTitle && (
          <div className="flex flex-wrap gap-1.5" style={{ marginTop: 4 }}>
            <span
              className="inline-flex items-center gap-2 rounded-full"
              style={{
                padding: "5px 12px 5px 10px",
                backgroundColor: "var(--mem-distilled-bg)",
                border: "1px solid var(--mem-accent-page)",
                boxShadow: "0 0 12px var(--mem-shimmer-color)",
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ backgroundColor: "var(--mem-accent-page)" }}
              />
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "9px",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "var(--mem-accent-page)",
                }}
              >
                {t("activity.pageTag")}
              </span>
              <span
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                }}
              >
                {truncate(pageTitle, 48)}
              </span>
            </span>
          </div>
        )}

        {/* Memory title pills — the source memories that drove the action.
            For page events these are the seed memories. */}
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

        {/* Detail text for non-page actions that carry their subject in
            `detail` (rather than via memory_titles lookup).
            - refine: legacy refine detail
            - forget: memory row deleted, title preserved in detail
            Page events get the highlighted page tag above instead. */}
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
