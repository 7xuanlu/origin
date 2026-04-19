// SPDX-License-Identifier: AGPL-3.0-only
import { useMemo, useState } from "react";
import type { Concept, ConceptChange } from "../../lib/tauri";

interface Props {
  changes: ConceptChange[];
  concepts?: Concept[];
  onSelectConcept?: (conceptId: string) => void;
}

function explanationFor(
  change: ConceptChange,
  concept: Concept | undefined,
): string {
  switch (change.change_kind) {
    case "created": {
      const n = concept?.source_memory_ids.length;
      return n != null && n >= 2 ? `distilled from ${n} memories` : "newly distilled";
    }
    case "revised": {
      if (!concept) return "refined as evidence settled";
      // Show a peek at the current summary so the user sees what the concept now asserts.
      let raw = (concept.summary?.trim() || "");
      if (!raw) {
        // Fall back to content, stripping leading markdown heading.
        raw = (concept.content?.trim() || "").replace(/^#+\s+[^\n]*\n+/, "");
      }
      const text = raw.replace(/\s+/g, " ").trim();
      if (!text) {
        return concept.version >= 2 ? `refined to version ${concept.version}` : "refined as evidence settled";
      }
      const peek = text.length > 90 ? text.slice(0, 90).trimEnd() + "..." : text;
      return `now reads: "${peek}"`;
    }
    case "merged": {
      const n = concept?.source_memory_ids.length;
      return n != null && n >= 2
        ? `steeped together from ${n} sources`
        : "steeped together";
    }
  }
}

function relative(ms: number): string {
  const delta = Date.now() - ms;
  const days = Math.floor(delta / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  return `${days}d ago`;
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

const MS_48H = 48 * 60 * 60 * 1000;
const TOP_N = 5;

function recencyWeight(now: number, ms: number): number {
  const days = Math.floor((now - ms) / (24 * 60 * 60 * 1000));
  if (days <= 0) return 1.0;
  if (days === 1) return 0.7;
  if (days >= 2) return 0.4;
  return 0;
}

function entityCentrality(concept: Concept, allConcepts: Concept[]): number {
  if (!concept.entity_id) return 1;
  const refs = allConcepts.filter((c) => c.entity_id === concept.entity_id).length;
  return 1 + refs / 10;
}

function scoreOf(
  c: ConceptChange,
  concept: Concept,
  allConcepts: Concept[],
  now: number,
): number {
  // resolved_contradiction_bonus: no data path today; default 1.0 with a TODO comment.
  const resolvedContradictionBonus = 1.0; // TODO: detect when refinement cleared a needs_review flag
  const sourceScore = Math.log(concept.source_memory_ids.length + 1);
  return sourceScore * recencyWeight(now, c.changed_at_ms) * entityCentrality(concept, allConcepts) * resolvedContradictionBonus;
}

function substantiveGate(c: ConceptChange, concept: Concept): boolean {
  switch (c.change_kind) {
    case "created":
      return concept.source_memory_ids.length >= 4;
    case "revised":
      return concept.version >= 3;
    case "merged":
      return concept.source_memory_ids.length >= 3;
  }
}

export function RefiningList({ changes, concepts, onSelectConcept }: Props) {
  const conceptById = useMemo(
    () => new Map((concepts ?? []).map((c) => [c.id, c])),
    [concepts],
  );

  const conceptIdSet = useMemo(
    () => new Set((concepts ?? []).map((c) => c.id)),
    [concepts],
  );

  const scored = useMemo(() => {
    const now = Date.now();

    // Gate 1: Dedupe by concept_id, keep most recent changed_at_ms.
    const deduped = new Map<string, ConceptChange>();
    for (const c of changes) {
      const existing = deduped.get(c.concept_id);
      if (!existing || c.changed_at_ms > existing.changed_at_ms) {
        deduped.set(c.concept_id, c);
      }
    }

    const candidates: Array<{ change: ConceptChange; concept: Concept; score: number }> = [];
    for (const c of deduped.values()) {
      // Gate 2: Fresh (within 48h).
      if (now - c.changed_at_ms > MS_48H) continue;

      // Gate 3: Touched (concept_id appears in the recentConcepts lookup).
      if (!conceptIdSet.has(c.concept_id)) continue;

      const concept = conceptById.get(c.concept_id);
      if (!concept) continue;

      // Gate 4: Substantive.
      if (!substantiveGate(c, concept)) continue;

      const score = scoreOf(c, concept, concepts ?? [], now);
      candidates.push({ change: c, concept, score });
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, TOP_N);
  }, [changes, concepts, conceptById, conceptIdSet]);

  if (scored.length === 0) return null;

  return (
    <section data-testid="refining">
      <h2 style={SECTION_TITLE_STYLE}>Refining</h2>
      <p style={SECTION_SUB_STYLE} className="mb-3">
        quality settling in as you keep working
      </p>
      <ul>
        {scored.map(({ change, concept }, index) => (
          <RefiningItem
            key={change.concept_id}
            change={change}
            concept={concept}
            onSelectConcept={onSelectConcept}
            isLast={index === scored.length - 1}
          />
        ))}
      </ul>
    </section>
  );
}

function RefiningItem({
  change,
  concept,
  onSelectConcept,
  isLast,
}: {
  change: ConceptChange;
  concept: Concept | undefined;
  onSelectConcept?: (conceptId: string) => void;
  isLast: boolean;
}) {
  const [hover, setHover] = useState(false);
  const clickable = Boolean(onSelectConcept);
  const explanation = explanationFor(change, concept);

  return (
    <li
      data-testid={`refining-item-${change.change_kind}`}
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
      onClick={() => onSelectConcept?.(change.concept_id)}
    >
      <div className="flex items-baseline gap-3">
        <span
          className="flex-1 truncate"
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: 14,
            fontWeight: 500,
            color: "var(--mem-text)",
          }}
        >
          {change.title}
        </span>
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: 11,
            color: "var(--mem-text-tertiary)",
            whiteSpace: "nowrap",
          }}
        >
          {relative(change.changed_at_ms)}
        </span>
      </div>
      <p
        className="truncate"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: 12,
          fontStyle: "italic",
          color: "var(--mem-text-secondary)",
          marginTop: 2,
        }}
      >
        {explanation}
      </p>
    </li>
  );
}
