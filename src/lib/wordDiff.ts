// SPDX-License-Identifier: AGPL-3.0-only

/** One run of a word-level diff. `text` preserves the original whitespace. */
export interface DiffSegment {
  kind: "same" | "del" | "ins";
  text: string;
}

/** Tokens are a word plus its leading whitespace, so joins reproduce the input. */
function tokenize(text: string): string[] {
  return text.match(/\s*\S+/g) ?? [];
}

/**
 * Word-level LCS diff between two strings. Whitespace-only changes are treated
 * as equal; segment order at a divergence is deletions first, then insertions.
 */
export function diffWords(before: string, after: string): DiffSegment[] {
  const a = tokenize(before);
  const b = tokenize(after);
  const n = a.length;
  const m = b.length;
  // ponytail: O(n·m) table is fine for memory-sized text; past ~1M cells fall
  // back to whole-string replace rather than allocating a huge matrix.
  if (n * m > 1_000_000) {
    const out: DiffSegment[] = [];
    if (before) out.push({ kind: "del", text: before });
    if (after) out.push({ kind: "ins", text: after });
    return out;
  }

  const key = (token: string) => token.trim();
  const lcs: number[][] = Array.from({ length: n + 1 }, () =>
    new Array<number>(m + 1).fill(0),
  );
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] =
        key(a[i]) === key(b[j])
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const out: DiffSegment[] = [];
  const push = (kind: DiffSegment["kind"], text: string) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ kind, text });
  };

  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (key(a[i]) === key(b[j])) {
      push("same", a[i]);
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      push("del", a[i]);
      i++;
    } else {
      push("ins", b[j]);
      j++;
    }
  }
  while (i < n) push("del", a[i++]);
  while (j < m) push("ins", b[j++]);
  return out;
}

/** Word counts for the added / removed runs of a diff. */
export function diffWordCounts(segments: DiffSegment[]): {
  added: number;
  removed: number;
} {
  let added = 0;
  let removed = 0;
  for (const segment of segments) {
    const words = segment.text.trim().split(/\s+/).filter(Boolean).length;
    if (segment.kind === "ins") added += words;
    if (segment.kind === "del") removed += words;
  }
  return { added, removed };
}
