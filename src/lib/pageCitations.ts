// SPDX-License-Identifier: AGPL-3.0-only
// Display-side mirror of the backend's per-claim citation contract
// (7xuanlu/wenlan crates/wenlan-core/src/citations.rs). Occurrence k is the
// k-th plain \[(\d+)\] regex match over the raw stored body — deliberately no
// code-fence/wikilink awareness, because the backend counts the same way and
// parity is the contract. Matches inside code consume occurrence indices but
// keep their raw [N] display (a rewritten link inside a code span would render
// as literal garbage). All transforms are display-only.
import type { PageCitation } from "./tauri";

export const CITATION_ANCHOR_PREFIX = "#citation:";

export type CitationState = "cited" | "stripped-empty" | "stripped-mismatch" | "none";

export interface ProcessedCitations {
  /** Body with markers rewritten to [k](#citation:k), or display-stripped. */
  content: string;
  state: CitationState;
  /** 1-based occurrence -> citation; empty unless state === "cited". */
  byOccurrence: Map<number, PageCitation>;
}

const MARKER_RE = /\[(\d+)\]/g;

/** Character ranges covered by fenced blocks or inline code spans. */
function codeRanges(content: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  const fence = /^```.*$/gm;
  let openAt: number | null = null;
  let m: RegExpExecArray | null;
  while ((m = fence.exec(content)) !== null) {
    if (openAt === null) {
      openAt = m.index;
    } else {
      ranges.push([openAt, m.index + m[0].length]);
      openAt = null;
    }
  }
  if (openAt !== null) ranges.push([openAt, content.length]);
  const inline = /`[^`\n]*`/g;
  while ((m = inline.exec(content)) !== null) {
    const start = m.index;
    if (!ranges.some(([s, e]) => start >= s && start < e)) {
      ranges.push([start, start + m[0].length]);
    }
  }
  return ranges;
}

/** Mirror the backend's strip_markers: remove [N], collapse doubled spaces. */
function stripMarkers(text: string): string {
  return text.replace(/\[\d+\]/g, "").replace(/ {2,}/g, " ");
}

/** Remove rewritten citation links from plain-text contexts (TLDR pull-quote). */
export function stripCitationLinks(text: string): string {
  return text
    .replace(/\[\d+\]\(#citation:\d+\)/g, "")
    .replace(/ {2,}/g, " ")
    .trim();
}

export function processCitations(
  content: string,
  citations: PageCitation[] | undefined,
): ProcessedCitations {
  const matches = [...content.matchAll(MARKER_RE)];
  if (matches.length === 0 && (!citations || citations.length === 0)) {
    return { content, state: "none", byOccurrence: new Map() };
  }
  if (!citations || citations.length === 0) {
    // Verified backend behavior: a user content edit resets citations to []
    // but stores the markers verbatim — without this strip every edited page
    // renders permanent [N] noise.
    return { content: stripMarkers(content), state: "stripped-empty", byOccurrence: new Map() };
  }
  const byOccurrence = new Map<number, PageCitation>();
  for (const c of citations) byOccurrence.set(c.occurrence, c);
  // Conservative fallback: any count or marker disagreement means the mapping
  // is untrustworthy — misattributed citations are worse than none.
  const mismatch =
    matches.length !== citations.length ||
    matches.some((m, i) => byOccurrence.get(i + 1)?.marker !== Number(m[1]));
  if (mismatch) {
    return { content: stripMarkers(content), state: "stripped-mismatch", byOccurrence: new Map() };
  }
  const ranges = codeRanges(content);
  const inCode = (idx: number) => ranges.some(([s, e]) => idx >= s && idx < e);
  let out = "";
  let last = 0;
  matches.forEach((m, i) => {
    const at = m.index ?? 0;
    if (inCode(at)) return; // counted, but displayed raw
    const k = i + 1;
    out += content.slice(last, at) + `[${k}](${CITATION_ANCHOR_PREFIX}${k})`;
    last = at + m[0].length;
  });
  out += content.slice(last);
  return { content: out, state: "cited", byOccurrence };
}

/** Short chip label: memory ids as-is, URLs to hostname, paths to basename. */
export function citationDisplayLabel(c: PageCitation): string {
  if (c.source_kind === "external_url") {
    try {
      return new URL(c.locator).hostname;
    } catch {
      return c.locator.slice(0, 24);
    }
  }
  if (c.source_kind === "external_file") {
    return c.locator.split("/").filter(Boolean).pop() ?? c.locator;
  }
  if (c.source_kind === "authored") return "authored";
  return c.locator;
}
