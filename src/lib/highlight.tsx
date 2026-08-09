// SPDX-License-Identifier: AGPL-3.0-only
import type { ReactNode } from "react";

/** Human-readable relevance label from an RRF score. */
export function relevanceLabel(score: number): { text: string; tier: "strong" | "good" | "faint" } {
  // RRF scores: top-1 both signals ≈ 0.033, single signal ≈ 0.017, weak ≈ 0.005
  if (score >= 0.025) return { text: "strong match", tier: "strong" };
  if (score >= 0.012) return { text: "good match", tier: "good" };
  if (score >= 0.005) return { text: "match", tier: "faint" };
  return { text: "related", tier: "faint" };
}

/**
 * Highlights query terms in text with <mark> elements.
 * Returns the original text if no terms match.
 *
 * @param markClass - CSS classes for the <mark> element
 */
export function highlightTerms(
  text: string,
  query: string,
  markClass = "bg-[rgba(74,222,128,0.2)] text-inherit rounded-sm",
): ReactNode {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);
  if (terms.length === 0) return text;

  const escaped = terms.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    if (terms.some((t) => part.toLowerCase() === t.toLowerCase())) {
      return (
        <mark key={i} className={markClass}>
          {part}
        </mark>
      );
    }
    return part;
  });
}

/**
 * Checks whether any query term appears literally in the text.
 * Used to distinguish keyword matches from pure semantic matches.
 */
export function hasKeywordMatch(text: string, query: string): boolean {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);
  if (terms.length === 0) return false;
  const lower = text.toLowerCase();
  return terms.some((t) => lower.includes(t));
}
