// SPDX-License-Identifier: AGPL-3.0-only

export type ContentShape =
  | "structured"
  | "key-value"
  | "list"
  | "mixed"
  | "prose"
  | "single-fact";

const KV_PATTERN = /^\s*[\w][\w\s]{0,30}:\s+\S/;
const PIPE_KV_PATTERN = /^\s*[\w][\w\s]{0,30}\s*\|\s*\S/;
const LIST_PATTERN = /^\s*(?:[-*]|\d+\.)\s+\S/;

function isKvLine(line: string): boolean {
  if (line.trimStart().startsWith("**")) return false;
  return KV_PATTERN.test(line) || PIPE_KV_PATTERN.test(line);
}

function isListLine(line: string): boolean {
  return LIST_PATTERN.test(line);
}

/**
 * Pre-normalize single-line content into multi-line text so the classifier
 * can detect structure. Handles three patterns common in agent-generated memories:
 * pipe-delimited fields, inline numbering, and long unbroken prose.
 */
export function normalizeContent(text: string): string {
  // Normalize each line independently, then rejoin.
  // get_memory_detail concatenates chunks with \n, so each line may be
  // a long single-line chunk that needs its own normalization.
  if (text.includes("\n")) {
    return text.split("\n").map((line) => normalizeLine(line)).join("\n");
  }
  return normalizeLine(text);
}

function normalizeLine(text: string): string {
  if (!text.trim()) return text;

  // Pipe-delimited: "key: val | key: val" or "key | val | key | val"
  // Require >=2 pipe characters to avoid splitting prose with a single |
  const pipeCount = (text.match(/\|/g) || []).length;
  if (pipeCount >= 2) {
    return text.split("|").map((s) => s.trim()).join("\n");
  }

  // Inline numbering: "(1) item, (2) item" or "1) item, 2) item"
  const parenNumMatches = text.match(/\(\d+\)/g);
  if (parenNumMatches && parenNumMatches.length >= 2) {
    const parts = text.split(/\(\d+\)\s*/);
    const lines: string[] = [];
    if (parts[0].trim()) {
      // Keep the prefix as-is (preserve trailing colon for labels like "Origin needs:")
      lines.push(parts[0].replace(/,\s*$/, "").trim());
    }
    for (let i = 1; i < parts.length; i++) {
      if (parts[i].trim()) {
        lines.push("- " + parts[i].replace(/,\s*$/, "").trim());
      }
    }
    return lines.join("\n");
  }

  // Bare numbered items: "1) item, 2) item, 3) item"
  const bareNumMatches = text.match(/\b\d+\)/g);
  if (bareNumMatches && bareNumMatches.length >= 2) {
    const parts = text.split(/\b\d+\)\s*/);
    const lines: string[] = [];
    if (parts[0].trim()) {
      lines.push(parts[0].replace(/,\s*$/, "").trim());
    }
    for (let i = 1; i < parts.length; i++) {
      if (parts[i].trim()) {
        lines.push("- " + parts[i].replace(/,\s*$/, "").trim());
      }
    }
    return lines.join("\n");
  }

  // Long prose: >200 chars with no structure — insert paragraph breaks
  if (text.length > 200) {
    // Split on period + space + uppercase (sentence boundary), keeping the period
    const rawSentences = text.split(/(?<=\.)\s+(?=[A-Z])/);
    if (rawSentences.length >= 3) {
      const paragraphs: string[] = [];
      let current: string[] = [];
      for (const sentence of rawSentences) {
        current.push(sentence);
        if (current.join(" ").length > 100 && current.length >= 2) {
          paragraphs.push(current.join(" "));
          current = [];
        }
      }
      if (current.length > 0) {
        paragraphs.push(current.join(" "));
      }
      if (paragraphs.length > 1) {
        return paragraphs.join("\n\n");
      }
    }
  }

  return text;
}

export function classifyContent(
  text: string,
  structuredFields?: string | null
): ContentShape {
  if (structuredFields && structuredFields !== "{}") {
    try {
      const parsed = JSON.parse(structuredFields);
      if (typeof parsed === "object" && parsed !== null && Object.keys(parsed).length > 0) {
        return "structured";
      }
    } catch {
      // Not valid JSON, fall through
    }
  }

  const trimmed = text.trim();
  const lines = trimmed.split("\n").filter((l) => l.trim().length > 0);

  if (lines.length === 0) return "single-fact";

  const kvCount = lines.filter(isKvLine).length;
  const listCount = lines.filter(isListLine).length;
  const structuredCount = kvCount + listCount;
  const nonStructuredCount = lines.length - structuredCount;

  // Mixed: has structured lines (list/kv) AND unstructured prose lines
  if (structuredCount >= 2 && nonStructuredCount >= 1) return "mixed";

  if (kvCount >= 2 && kvCount / lines.length > 0.5) return "key-value";
  if (listCount >= 2 && listCount / lines.length > 0.5) return "list";

  if (trimmed.includes("\n\n") || trimmed.length > 200) return "prose";

  return "single-fact";
}

export function prepareForRender(
  text: string,
  shape: ContentShape
): string {
  if (shape === "structured" || shape === "list" || shape === "prose" || shape === "single-fact") {
    return text;
  }

  return text
    .split("\n")
    .map((line) => {
      if (line.trimStart().startsWith("**")) return line;

      const pipeMatch = line.match(/^\s*([\w][\w\s]{0,30}?)\s*\|\s*(.+)$/);
      if (pipeMatch) {
        return `**${pipeMatch[1].trim()}:** ${pipeMatch[2].trim()}`;
      }

      const colonMatch = line.match(/^\s*([\w][\w\s]{0,30}?):\s+(.+)$/);
      if (colonMatch) {
        return `**${colonMatch[1].trim()}:** ${colonMatch[2].trim()}`;
      }

      return line;
    })
    .join("\n");
}

export type PreviewResult =
  | string
  | { key: string; value: string }
  | null;

export function extractPreview(
  text: string,
  shape: ContentShape
): PreviewResult {
  const trimmed = text.trim();

  switch (shape) {
    case "structured":
      return null;

    case "single-fact":
      return trimmed;

    case "key-value": {
      const lines = trimmed.split("\n");
      for (const line of lines) {
        const pipeMatch = line.match(/^\s*([\w][\w\s]{0,30}?)\s*\|\s*(.+)$/);
        if (pipeMatch) return { key: pipeMatch[1].trim(), value: pipeMatch[2].trim() };
        const colonMatch = line.match(/^\s*([\w][\w\s]{0,30}?):\s+(.+)$/);
        if (colonMatch) return { key: colonMatch[1].trim(), value: colonMatch[2].trim() };
      }
      return trimmed.split("\n")[0] ?? trimmed;
    }

    case "list": {
      const firstItem = trimmed.split("\n").find((l) => isListLine(l));
      if (!firstItem) return trimmed.split("\n")[0] ?? trimmed;
      return firstItem.replace(/^\s*(?:[-*]|\d+\.)\s+/, "");
    }

    case "prose": {
      const sentenceEnd = trimmed.indexOf(". ");
      if (sentenceEnd > 0 && sentenceEnd <= 120) {
        return trimmed.slice(0, sentenceEnd + 1);
      }
      if (trimmed.length <= 120) return trimmed;
      return trimmed.slice(0, 120) + "...";
    }

    case "mixed": {
      const firstLine = trimmed.split("\n").find((l) => l.trim().length > 0);
      return firstLine?.trim() ?? trimmed;
    }
  }
}
