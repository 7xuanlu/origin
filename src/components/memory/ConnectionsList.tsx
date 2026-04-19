// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  listConcepts,
  listEntities,
  listRecentChanges,
  type Concept,
  type Entity,
} from "../../lib/tauri";

interface Props {
  onSelectConcept?: (conceptId: string) => void;
  // TODO: wire onSelectEntity once entity-detail navigation exists
  onSelectEntity?: (entityId: string) => void;
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

type InsightCategory = "momentum" | "stabilizing" | "recurring";

interface Insight {
  key: string;
  category: InsightCategory;
  nounPhrase: string;
  suffix: string;
  sortMs: number;
  timestampMs?: number;
  conceptId?: string;
  entityId?: string;
}

function relativeDate(isoOrMs: string | number): string {
  const ms =
    typeof isoOrMs === "number" ? isoOrMs : Date.parse(isoOrMs as string);
  if (isNaN(ms)) return "";
  const delta = Date.now() - ms;
  const days = Math.floor(delta / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  return weeks === 1 ? "1w ago" : `${weeks}w ago`;
}

function buildInsights(
  concepts: Concept[],
  entities: Entity[],
): Insight[] {
  const now = Date.now();
  const fourteenDaysMs = 14 * 86_400_000;

  // Category 1: Gathered momentum
  // Concepts created in the last 14 days with >= 4 source memories.
  const momentumInsights: Insight[] = concepts
    .filter((c) => {
      const createdMs = Date.parse(c.created_at);
      return (
        !isNaN(createdMs) &&
        now - createdMs <= fourteenDaysMs &&
        c.source_memory_ids.length >= 4
      );
    })
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
    .slice(0, 2)
    .map((c) => ({
      key: `momentum-${c.id}`,
      category: "momentum" as InsightCategory,
      nounPhrase: c.title,
      suffix: ` gathered ${c.source_memory_ids.length} memories`,
      sortMs: Date.parse(c.created_at),
      timestampMs: Date.parse(c.created_at),
      conceptId: c.id,
    }));

  // Category 2: Stabilizing
  // Concepts where version >= 3. Show top 2 by version descending, then last_modified.
  const stabilizingInsights: Insight[] = concepts
    .filter((c) => c.version >= 3)
    .sort((a, b) => {
      if (b.version !== a.version) return b.version - a.version;
      return Date.parse(b.last_modified) - Date.parse(a.last_modified);
    })
    .slice(0, 2)
    .map((c) => ({
      key: `stabilizing-${c.id}`,
      category: "stabilizing" as InsightCategory,
      nounPhrase: c.title,
      suffix: ` refined ${c.version} times, pattern stabilizing`,
      sortMs: Date.parse(c.last_modified),
      timestampMs: Date.parse(c.last_modified),
      conceptId: c.id,
    }));

  // Category 3: Recurring theme
  // Entities that appear as entity_id in >= 2 active concepts.
  const entityConceptMap = new Map<string, Concept[]>();
  for (const c of concepts) {
    if (c.entity_id) {
      const existing = entityConceptMap.get(c.entity_id) ?? [];
      existing.push(c);
      entityConceptMap.set(c.entity_id, existing);
    }
  }

  const entityById = new Map(entities.map((e) => [e.id, e]));

  const recurringInsights: Insight[] = Array.from(entityConceptMap.entries())
    .filter(([, linkedConcepts]) => linkedConcepts.length >= 2)
    .sort(([, a], [, b]) => b.length - a.length)
    .slice(0, 2)
    .map(([entityId, linkedConcepts]) => {
      const entity = entityById.get(entityId);
      const entityName = entity?.name ?? entityId;
      const firstConcept = linkedConcepts[0];
      return {
        key: `recurring-${entityId}`,
        category: "recurring" as InsightCategory,
        nounPhrase: entityName,
        suffix: ` across ${linkedConcepts.length} concepts`,
        sortMs: Date.parse(firstConcept.last_modified),
        entityId,
        conceptId: firstConcept.id,
      };
    });

  // Mix categories and cap at 5, prefer most recent first within each group.
  const all = [
    ...momentumInsights,
    ...stabilizingInsights,
    ...recurringInsights,
  ].sort((a, b) => b.sortMs - a.sortMs);

  return all.slice(0, 5);
}

export function ConnectionsList({ onSelectConcept, onSelectEntity }: Props) {
  const { data: concepts = [] } = useQuery({
    queryKey: ["connections-concepts"],
    queryFn: () => listConcepts("active", undefined, 30),
    refetchInterval: 30_000,
  });

  // listRecentChanges is loaded so react-query doesn't double-fetch with RefiningList
  useQuery({
    queryKey: ["recentChanges"],
    queryFn: () => listRecentChanges(20),
    refetchInterval: 30_000,
  });

  const { data: entities = [] } = useQuery({
    queryKey: ["connections-entities"],
    queryFn: () => listEntities(undefined, undefined),
    refetchInterval: 60_000,
  });

  const insights = buildInsights(concepts, entities);

  if (!insights.length) return null;

  return (
    <section data-testid="connections">
      <h2 style={SECTION_TITLE_STYLE}>What Origin learned</h2>
      <p style={SECTION_SUB_STYLE} className="mb-3">
        patterns and themes the refinery surfaced this week
      </p>
      <ul>
        {insights.map((insight, i) => (
          <InsightRow
            key={insight.key}
            insight={insight}
            isLast={i === insights.length - 1}
            onSelectConcept={onSelectConcept}
            onSelectEntity={onSelectEntity}
          />
        ))}
      </ul>
    </section>
  );
}

function InsightRow({
  insight,
  isLast,
  onSelectConcept,
  onSelectEntity,
}: {
  insight: Insight;
  isLast: boolean;
  onSelectConcept?: (conceptId: string) => void;
  onSelectEntity?: (entityId: string) => void;
}) {
  const [hover, setHover] = useState(false);

  function handleClick() {
    if (insight.category === "recurring") {
      if (onSelectEntity && insight.entityId) {
        onSelectEntity(insight.entityId);
      } else if (onSelectConcept && insight.conceptId) {
        onSelectConcept(insight.conceptId);
      }
    } else if (onSelectConcept && insight.conceptId) {
      onSelectConcept(insight.conceptId);
    }
  }

  const clickable = insight.category === "recurring"
    ? Boolean(onSelectEntity ?? onSelectConcept)
    : Boolean(onSelectConcept);

  const prefix = insight.category === "recurring" ? "You've returned to " : "";

  return (
    <li
      data-testid="insight-row"
      className="py-3 px-2 transition-colors duration-150"
      style={{
        backgroundColor: hover ? "var(--mem-hover)" : "transparent",
        borderBottom: isLast
          ? "none"
          : "1px solid color-mix(in srgb, var(--mem-border) 60%, transparent)",
        cursor: clickable ? "pointer" : "default",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={handleClick}
    >
      <div className="flex items-baseline gap-2">
        <span className="flex-1 truncate">
          {prefix && (
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: 14,
                color: "var(--mem-text-secondary)",
              }}
            >
              {prefix}
            </span>
          )}
          <span
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: 14,
              fontWeight: 500,
              color: "var(--mem-text)",
            }}
          >
            {insight.nounPhrase}
          </span>
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 14,
              color: "var(--mem-text-secondary)",
            }}
          >
            {insight.suffix}
          </span>
        </span>
        {insight.timestampMs != null && !isNaN(insight.timestampMs) && (
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 11,
              color: "var(--mem-text-tertiary)",
              whiteSpace: "nowrap",
            }}
          >
            {relativeDate(insight.timestampMs)}
          </span>
        )}
      </div>
    </li>
  );
}
