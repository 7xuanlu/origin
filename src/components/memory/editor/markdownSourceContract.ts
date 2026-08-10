export type LineEndingProfile =
  | { kind: "none"; separator: "\n" }
  | { kind: "lf"; separator: "\n" }
  | { kind: "crlf"; separator: "\r\n" }
  | { kind: "mixed" }
  | { kind: "lone-cr" };

export interface MarkdownSourceProfile {
  bom: boolean;
  lineEndings: LineEndingProfile;
}

export interface PreparedMarkdownSource {
  editorDocument: string;
  profile: MarkdownSourceProfile;
  canEditLosslessly: boolean;
}

export type MarkdownDraftValidation =
  | { valid: true; content: string }
  | { valid: false; reason: "empty" };

const BOM = "\uFEFF";

function toLogicalLf(source: string): string {
  return source.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
}

function detectLineEndings(source: string): LineEndingProfile {
  if (/\r(?!\n)/u.test(source)) {
    return { kind: "lone-cr" };
  }

  const hasCrlf = source.includes("\r\n");
  const hasLf = /(^|[^\r])\n/u.test(source);

  if (hasCrlf && hasLf) {
    return { kind: "mixed" };
  }
  if (hasCrlf) {
    return { kind: "crlf", separator: "\r\n" };
  }
  if (hasLf) {
    return { kind: "lf", separator: "\n" };
  }
  return { kind: "none", separator: "\n" };
}

export function prepareMarkdownSource(raw: string): PreparedMarkdownSource {
  const bom = raw.startsWith(BOM);
  const source = bom ? raw.slice(BOM.length) : raw;
  const lineEndings = detectLineEndings(source);

  return {
    editorDocument: toLogicalLf(source),
    profile: { bom, lineEndings },
    canEditLosslessly:
      lineEndings.kind !== "mixed" && lineEndings.kind !== "lone-cr",
  };
}

export function serializeMarkdownSource(
  editorDocument: string,
  profile: MarkdownSourceProfile,
): string {
  if (
    profile.lineEndings.kind === "mixed" ||
    profile.lineEndings.kind === "lone-cr"
  ) {
    throw new Error(
      "Mixed or lone-CR Markdown must be explicitly normalized before editing.",
    );
  }

  const logicalDocument = toLogicalLf(editorDocument);
  const source =
    profile.lineEndings.separator === "\r\n"
      ? logicalDocument.replaceAll("\n", "\r\n")
      : logicalDocument;

  return profile.bom ? BOM + source : source;
}

export function normalizeMarkdownSourceToLf(raw: string): string {
  const bom = raw.startsWith(BOM);
  const source = bom ? raw.slice(BOM.length) : raw;
  const normalized = toLogicalLf(source);

  return bom ? BOM + normalized : normalized;
}

export function validateMarkdownDraft(
  content: string,
): MarkdownDraftValidation {
  return content.trim()
    ? { valid: true, content }
    : { valid: false, reason: "empty" };
}
