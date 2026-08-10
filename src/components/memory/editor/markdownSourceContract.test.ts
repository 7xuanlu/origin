import { describe, expect, it } from "vitest";

import {
  normalizeMarkdownSourceToLf,
  prepareMarkdownSource,
  serializeMarkdownSource,
  validateMarkdownDraft,
} from "./markdownSourceContract";

describe("markdown source contract", () => {
  it.each([
    ["empty", ""],
    ["single line", "# Heading"],
    ["uniform LF", "# Heading\n\nBody\n"],
    ["uniform CRLF", "# Heading\r\n\r\nBody\r\n"],
    ["UTF-8 BOM with LF", "\uFEFF# Heading\n\nBody\n"],
    ["UTF-8 BOM with CRLF", "\uFEFF# Heading\r\n\r\nBody\r\n"],
    ["leading and trailing whitespace", "  # Heading  \n\nBody\t \n\n"],
  ])("round trips editable %s source exactly", (_label, source) => {
    const prepared = prepareMarkdownSource(source);

    expect(prepared.canEditLosslessly).toBe(true);
    expect(
      serializeMarkdownSource(prepared.editorDocument, prepared.profile),
    ).toBe(source);
  });

  it("uses a BOM-free logical-LF document for a uniform CRLF source", () => {
    const prepared = prepareMarkdownSource(
      "\uFEFF# Heading\r\n\r\nBody\r\n",
    );

    expect(prepared).toEqual({
      editorDocument: "# Heading\n\nBody\n",
      profile: {
        bom: true,
        lineEndings: { kind: "crlf", separator: "\r\n" },
      },
      canEditLosslessly: true,
    });
  });

  it("makes pasted line breaks adopt a CRLF source profile", () => {
    const prepared = prepareMarkdownSource("first\r\nsecond\r\n");
    const editedDocument =
      prepared.editorDocument +
      "LF paste\nCRLF paste\r\nlone CR paste\rlast";

    expect(
      serializeMarkdownSource(editedDocument, prepared.profile),
    ).toBe(
      "first\r\nsecond\r\nLF paste\r\nCRLF paste\r\nlone CR paste\r\nlast",
    );
  });

  it("keeps an LF source uniformly LF after mixed pasted line breaks", () => {
    const prepared = prepareMarkdownSource("first\nsecond\n");
    const editedDocument =
      prepared.editorDocument +
      "LF paste\nCRLF paste\r\nlone CR paste\rlast";

    expect(
      serializeMarkdownSource(editedDocument, prepared.profile),
    ).toBe(
      "first\nsecond\nLF paste\nCRLF paste\nlone CR paste\nlast",
    );
  });

  it.each([
    ["mixed LF and CRLF", "first\r\nsecond\nthird", "mixed"],
    ["lone CR", "first\rsecond", "lone-cr"],
  ] as const)(
    "blocks %s source until normalization is explicit",
    (_label, source, kind) => {
      const prepared = prepareMarkdownSource(source);

      expect(prepared.canEditLosslessly).toBe(false);
      expect(prepared.profile.lineEndings.kind).toBe(kind);
      expect(() =>
        serializeMarkdownSource(prepared.editorDocument, prepared.profile),
      ).toThrow();

      const normalized = normalizeMarkdownSourceToLf(source);
      const normalizedPrepared = prepareMarkdownSource(normalized);

      expect(normalizedPrepared.canEditLosslessly).toBe(true);
      expect(normalizedPrepared.profile.lineEndings).toEqual({
        kind: "lf",
        separator: "\n",
      });
      expect(
        serializeMarkdownSource(
          normalizedPrepared.editorDocument,
          normalizedPrepared.profile,
        ),
      ).toBe(normalized);
    },
  );

  it("validates with trimmed content but returns the exact valid source", () => {
    const source = "  # Heading  \n\nBody\t \n";

    expect(validateMarkdownDraft(source)).toEqual({
      valid: true,
      content: source,
    });
    expect(validateMarkdownDraft(" \n\t\r\n ")).toEqual({
      valid: false,
      reason: "empty",
    });
  });

  it("has positive controls for final-newline, CRLF, and BOM loss", () => {
    const source = "\uFEFF# Heading\r\n\r\nBody\r\n";
    const prepared = prepareMarkdownSource(source);
    const roundTrip = serializeMarkdownSource(
      prepared.editorDocument,
      prepared.profile,
    );

    expect(roundTrip).toBe(source);
    expect(roundTrip.slice(0, -2)).not.toBe(source);
    expect(roundTrip.replaceAll("\r\n", "\n")).not.toBe(source);
    expect(roundTrip.slice(1)).not.toBe(source);
  });

  it.each([
    [
      "Obsidian source syntax",
      [
        "\uFEFF---",
        "aliases: [文瀾, \"Wenlan Wiki\"]",
        "cssclasses:",
        "  - source-first",
        "folded: >",
        "  keeps YAML order and comments",
        "# property comment",
        "---",
        "",
        "> [!NOTE] Callout",
        "> Preserve [[Page#Heading|alias]], ![[embed.png]], and ^block-id.",
        "",
        "%%private comment%% ==highlight== #nested/tag",
        "",
      ].join("\n"),
    ],
    [
      "GFM and raw HTML with CRLF",
      [
        "| A | B |",
        "| - | - |",
        "| ~~old~~ | [ref][id] |",
        "",
        "- [x] done",
        "- [ ] open",
        "",
        "```ts",
        "const source = \"unchanged\";",
        "```",
        "",
        "<details><summary>Raw HTML</summary>body</details>",
        "",
        "[id]: https://example.com \"title\"",
        "",
      ].join("\r\n"),
    ],
    [
      "math, Unicode, and Wenlan-like markers",
      [
        "# 中文、かな、한글、emoji 🧠",
        "",
        "Inline $E = mc^2$ and block:",
        "",
        "$$",
        "\\int_0^1 x^2\\,dx",
        "$$",
        "",
        "Combining: e\u0301; citation-like [1]; memory link #memory:mem_1.",
        "",
      ].join("\n"),
    ],
  ])("round trips the %s compatibility corpus without interpretation", (_name, source) => {
    const prepared = prepareMarkdownSource(source);

    expect(prepared.canEditLosslessly).toBe(true);
    expect(
      serializeMarkdownSource(prepared.editorDocument, prepared.profile),
    ).toBe(source);
  });
});
