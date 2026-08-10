// SPDX-License-Identifier: AGPL-3.0-only
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { forceParsing } from "@codemirror/language";
import { EditorSelection, EditorState, type RangeSet } from "@codemirror/state";
import { Decoration, EditorView } from "@codemirror/view";
import { installCodeMirrorDomPolyfills } from "./editorTestUtils";
import { prepareMarkdownSource, serializeMarkdownSource } from "./markdownSourceContract";
import {
  setWritingCompositionActive,
  writingPresentation,
} from "./writingPresentation";

beforeAll(() => installCodeMirrorDomPolyfills());

const liveViews: EditorView[] = [];

afterEach(() => {
  for (const view of liveViews.splice(0)) view.destroy();
  document.body.replaceChildren();
});

function makeView(
  doc: string,
  options: {
    selection?: EditorSelection;
    language?: boolean;
    onDocumentUpdate?: () => void;
  } = {},
): EditorView {
  const parent = document.createElement("div");
  document.body.append(parent);
  const extensions = [
    ...(options.language === false
      ? []
      : [
          markdown({
            base: markdownLanguage,
            addKeymap: false,
            pasteURLAsLink: false,
            completeHTMLTags: false,
          }),
        ]),
    writingPresentation(),
    ...(options.onDocumentUpdate
      ? [
          EditorView.updateListener.of((update) => {
            if (update.docChanged) options.onDocumentUpdate?.();
          }),
        ]
      : []),
  ];
  const view = new EditorView({
    parent,
    state: EditorState.create({
      doc,
      selection: options.selection,
      extensions,
    }),
  });
  liveViews.push(view);
  return view;
}

interface PresentationRange {
  from: number;
  to: number;
}

function rangesFromSet(set: RangeSet<Decoration>, docLength: number): PresentationRange[] {
  const ranges: PresentationRange[] = [];
  set.between(0, docLength, (from, to, value) => {
    if (
      value instanceof Decoration &&
      (value.spec as { writingConceal?: boolean }).writingConceal
    ) {
      ranges.push({ from, to });
    }
  });
  return ranges.sort((left, right) => left.from - right.from || left.to - right.to);
}

function concealedDecorationRanges(view: EditorView): PresentationRange[] {
  const ranges: PresentationRange[] = [];
  for (const source of view.state.facet(EditorView.decorations)) {
    const set = typeof source === "function" ? source(view) : source;
    ranges.push(...rangesFromSet(set, view.state.doc.length));
  }
  return ranges.sort((left, right) => left.from - right.from || left.to - right.to);
}

function concealedAtomicRanges(view: EditorView): PresentationRange[] {
  const ranges: PresentationRange[] = [];
  for (const source of view.state.facet(EditorView.atomicRanges)) {
    ranges.push(...rangesFromSet(source(view), view.state.doc.length));
  }
  return ranges.sort((left, right) => left.from - right.from || left.to - right.to);
}

function rangeFor(source: string, text: string, offset = 0): PresentationRange {
  const from = source.indexOf(text, offset);
  if (from < 0) throw new Error(`Expected ${JSON.stringify(text)} in source`);
  return { from, to: from + text.length };
}

describe("writingPresentation", () => {
  it("renders common parsed Markdown as live preview and keeps unknown syntax literal", () => {
    const source = [
      "# Heading #   ",
      "",
      "**bold** and *italic* and `code`",
      "- bullet",
      "1. ordered",
      "- [ ] task",
      "",
      "> quote",
      "",
      "[link](target)",
      "",
      "---",
      "",
      "~~strike~~",
      "",
      "[[Wiki]] ![[embed]]",
      "::: malformed ** marker",
      "away",
    ].join("\n");
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    const bold = source.indexOf("**bold**");
    const italic = source.indexOf("*italic*");
    const code = source.indexOf("`code`");
    const bullet = source.indexOf("- bullet");
    const ordered = source.indexOf("1. ordered");
    const task = source.indexOf("- [ ] task");
    const quote = source.indexOf("> quote");
    const link = source.indexOf("[link](target)");
    const horizontalRule = source.indexOf("---");
    const strike = source.indexOf("~~strike~~");
    const expected = [
      { from: 0, to: 2 },
      rangeFor(source, "#", source.indexOf("Heading") + "Heading".length),
      { from: bold, to: bold + 2 },
      { from: bold + 6, to: bold + 8 },
      { from: italic, to: italic + 1 },
      { from: italic + 7, to: italic + 8 },
      { from: code, to: code + 1 },
      { from: code + 5, to: code + 6 },
      { from: bullet, to: bullet + 2 },
      { from: ordered, to: ordered + 3 },
      { from: task, to: task + 2 },
      { from: task + 2, to: task + 5 },
      { from: quote, to: quote + 2 },
      { from: link, to: link + 1 },
      { from: link + 5, to: link + "[link](target)".length },
      { from: horizontalRule, to: horizontalRule + 3 },
      { from: strike, to: strike + 2 },
      { from: strike + 8, to: strike + 10 },
    ];

    expect(view.dom).toHaveAttribute("data-editor-presentation-mode", "writing");
    expect(concealedDecorationRanges(view)).toEqual(expected);
    expect(concealedAtomicRanges(view)).toEqual(expected);
    expect(view.state.doc.toString()).toBe(source);
    expect(view.contentDOM.textContent).toContain("Heading    ");
    expect(view.contentDOM.textContent).toContain("bold and italic and code");
    expect(view.contentDOM.textContent).toContain("[[Wiki]] ![[embed]]");
    expect(view.contentDOM.textContent).toContain("::: malformed ** marker");
    expect(view.contentDOM.textContent).not.toContain("[link](target)");
    expect(view.contentDOM.textContent).not.toContain("> quote");
    expect(view.contentDOM.querySelector('[data-writing-list-marker="bullet"]'))
      .toHaveTextContent("•");
    expect(view.contentDOM.querySelector('[data-writing-list-marker="ordered"]'))
      .toHaveTextContent("1.");
    expect(view.contentDOM.querySelector(".cm-writing-task-checkbox"))
      .toHaveAttribute("type", "checkbox");
    expect(view.contentDOM.querySelector(".cm-writing-blockquote")).not.toBeNull();
    expect(view.contentDOM.querySelector(".cm-writing-link")).toHaveTextContent("link");
    expect(view.contentDOM.querySelector(".cm-writing-horizontal-rule")).not.toBeNull();
    expect(view.contentDOM.querySelector(".cm-writing-strikethrough"))
      .toHaveTextContent("strike");
  });

  it("reveals every construct on the active line and restores concealment without a document update", () => {
    const source = "# Heading\n\n**bold** and *italic*\naway";
    const onDocumentUpdate = vi.fn();
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
      onDocumentUpdate,
    });
    const initial = concealedDecorationRanges(view);
    const inlineLine = source.indexOf("bold") + 2;

    view.dispatch({ selection: EditorSelection.single(inlineLine) });

    expect(concealedDecorationRanges(view)).toEqual([{ from: 0, to: 2 }]);
    expect(view.contentDOM.textContent).toContain("**bold** and *italic*");
    expect(view.state.doc.toString()).toBe(source);
    expect(onDocumentUpdate).not.toHaveBeenCalled();

    view.dispatch({ selection: EditorSelection.single(source.length) });

    expect(concealedDecorationRanges(view)).toEqual(initial);
    expect(view.state.doc.toString()).toBe(source);
    expect(onDocumentUpdate).not.toHaveBeenCalled();
  });

  it("reveals complete constructs intersected by a selection", () => {
    const source = "# Heading\n\n**bold** and *italic*\naway";
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    view.dispatch({
      selection: EditorSelection.single(
        source.indexOf("Heading"),
        source.indexOf("italic") + 3,
      ),
    });

    expect(concealedDecorationRanges(view)).toEqual([]);
    expect(view.state.doc.toString()).toBe(source);
  });

  it("reveals the active quoted line without exposing neighboring Markdown", () => {
    const source = [
      "> first",
      "> second",
      "",
      "- bullet",
      "",
      "[link](target)",
      "away",
    ].join("\n");
    const secondQuote = source.indexOf("> second");
    const bullet = source.indexOf("- bullet");
    const link = source.indexOf("[link](target)");
    const view = makeView(source, {
      selection: EditorSelection.single(secondQuote + 4),
    });

    expect(concealedDecorationRanges(view)).toEqual([
      { from: 0, to: 2 },
      { from: bullet, to: bullet + 2 },
      { from: link, to: link + 1 },
      { from: link + 5, to: link + "[link](target)".length },
    ]);
    expect(view.state.doc.toString()).toBe(source);
  });

  it("presents every line of a multi-line blockquote", () => {
    const source = "> first\n> second\n\naway";
    const secondQuote = source.indexOf("> second");
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(concealedDecorationRanges(view)).toEqual([
      { from: 0, to: 2 },
      { from: secondQuote, to: secondQuote + 2 },
    ]);
    expect(view.contentDOM.querySelectorAll(".cm-writing-blockquote")).toHaveLength(2);
    expect(view.state.doc.toString()).toBe(source);
  });

  it("reveals all source during composition and restores it without changing document or selection", () => {
    const source = "**bold**\naway";
    const view = makeView(source, { selection: EditorSelection.single(source.length) });
    const selection = view.state.selection;
    const beforeRanges = concealedDecorationRanges(view);

    view.dispatch({ effects: setWritingCompositionActive.of(true) });

    expect(concealedDecorationRanges(view)).toEqual([]);
    expect(view.contentDOM.textContent).toContain("**bold**");
    expect(view.state.doc.toString()).toBe(source);
    expect(view.state.selection.eq(selection)).toBe(true);

    view.dispatch({ effects: setWritingCompositionActive.of(false) });

    expect(concealedDecorationRanges(view)).toEqual(beforeRanges);
    expect(view.state.doc.toString()).toBe(source);
    expect(view.state.selection.eq(selection)).toBe(true);
  });

  it("toggles a rendered task checkbox through one source update and restores keyboard focus", () => {
    const source = "- [ ] verify exact source\naway";
    const onDocumentUpdate = vi.fn();
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
      onDocumentUpdate,
    });
    const selection = view.state.selection;
    const checkbox = view.contentDOM.querySelector<HTMLInputElement>(
      ".cm-writing-task-checkbox",
    );

    expect(checkbox).not.toBeNull();
    expect(checkbox).not.toBeChecked();
    expect(checkbox).toHaveProperty("tabIndex", 0);
    expect(checkbox).toHaveAccessibleName("verify exact source");
    checkbox?.focus();
    expect(checkbox).toHaveFocus();
    checkbox?.click();

    expect(view.state.doc.toString()).toBe("- [x] verify exact source\naway");
    expect(view.state.selection.eq(selection)).toBe(true);
    expect(onDocumentUpdate).toHaveBeenCalledTimes(1);
    expect(
      view.contentDOM.querySelector(".cm-writing-task-checkbox"),
    ).toHaveFocus();
  });

  it("uses readable task text for the rendered checkbox name", () => {
    const source =
      "- [ ] **verify** [exact source](https://wenlan.app) with `tests`\naway";
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(view.contentDOM.querySelector(".cm-writing-task-checkbox"))
      .toHaveAccessibleName("verify exact source with tests");
    expect(view.state.doc.toString()).toBe(source);
  });

  it("renders a safe CommonMark image without changing its source", () => {
    const source = "![Wenlan graph](https://wenlan.app/graph.png)\naway";
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });
    const image = view.contentDOM.querySelector<HTMLImageElement>(
      ".cm-writing-image",
    );

    expect(image).not.toBeNull();
    expect(image).toHaveAttribute("src", "https://wenlan.app/graph.png");
    expect(image).toHaveAttribute("alt", "Wenlan graph");
    expect(image).toHaveAttribute("loading", "lazy");
    expect(view.contentDOM.textContent).not.toContain("![Wenlan graph]");
    expect(view.state.doc.toString()).toBe(source);
  });

  it("keeps unsafe or unresolved image destinations as editable source", () => {
    const source = [
      "![local](./attachment.png)",
      "![unsafe](javascript:alert(1))",
      "away",
    ].join("\n");
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(view.contentDOM.querySelector(".cm-writing-image")).toBeNull();
    expect(view.contentDOM.textContent).toContain("![local](./attachment.png)");
    expect(view.contentDOM.textContent).toContain("![unsafe](javascript:alert(1))");
    expect(view.state.doc.toString()).toBe(source);
  });

  it("presents an Obsidian callout while preserving its Markdown", () => {
    const source = "> [!NOTE] Exact source\n> Body stays editable\n\naway";
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(view.contentDOM.querySelectorAll(".cm-writing-callout")).toHaveLength(2);
    expect(view.contentDOM.querySelector(".cm-writing-callout-badge"))
      .toHaveTextContent("Note");
    expect(view.contentDOM.textContent).not.toContain("[!NOTE]");
    expect(view.contentDOM.textContent).not.toContain("> Body");
    expect(view.contentDOM.textContent).toContain("Body stays editable");
    expect(view.state.doc.toString()).toBe(source);
  });

  it("reuses the horizontal-rule widget when unrelated selection state changes", () => {
    const source = "---\n\nfirst\nsecond";
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });
    const initialRule = view.contentDOM.querySelector(".cm-writing-horizontal-rule");

    view.dispatch({ selection: EditorSelection.single(source.indexOf("first")) });

    expect(initialRule).not.toBeNull();
    expect(view.contentDOM.querySelector(".cm-writing-horizontal-rule")).toBe(initialRule);
  });

  it("presents a parsed fenced code block without changing its source", () => {
    const source = "```ts\nconst value = 1;\n```\naway";
    const closingFence = source.indexOf("```", 3);
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(concealedDecorationRanges(view)).toEqual([
      { from: 0, to: "```ts".length },
      { from: closingFence, to: closingFence + 3 },
    ]);
    expect(view.contentDOM.textContent).not.toContain("```ts");
    expect(view.contentDOM.querySelectorAll(".cm-writing-fenced-code")).toHaveLength(3);
    expect(view.state.doc.toString()).toBe(source);
  });

  it("reveals fenced-code source when the caret is on its closing line", () => {
    const source = "```ts\nconst value = 1;\n```\naway";
    const closingFence = source.lastIndexOf("```");
    const view = makeView(source, {
      selection: EditorSelection.single(closingFence + 1),
    });

    expect(concealedDecorationRanges(view)).toEqual([]);
    expect(view.contentDOM.textContent).toContain("```ts");
    expect(view.contentDOM.textContent).toContain("```");
    expect(view.state.doc.toString()).toBe(source);
  });

  it("keeps unsupported and unparsed source visible and losslessly serializable", () => {
    const rawSource = [
      "\uFEFF---",
      "key: value",
      "---",
      "<div>html</div>",
      "$$ math $$",
      "![[embed]] [[wiki]]",
      "```md",
      "# literal",
      "```",
      "::: malformed ** marker  ",
      "",
    ].join("\r\n");
    const prepared = prepareMarkdownSource(rawSource);
    const parsed = makeView(prepared.editorDocument, {
      selection: EditorSelection.single(prepared.editorDocument.length),
    });
    const unparsedSource = "# Heading and **bold**";
    const unparsed = makeView(unparsedSource, {
      language: false,
      selection: EditorSelection.single(unparsedSource.length),
    });

    expect(concealedDecorationRanges(parsed)).toEqual([]);
    expect(
      serializeMarkdownSource(parsed.state.doc.toString(), prepared.profile),
    ).toBe(rawSource);
    expect(parsed.contentDOM.textContent).toContain("![[embed]] [[wiki]]");
    expect(parsed.contentDOM.textContent).toContain("```md");
    expect(concealedDecorationRanges(unparsed)).toEqual([]);
    expect(unparsed.contentDOM.textContent).toContain("# Heading and **bold**");
    expect(unparsed.state.doc.toString()).toBe(unparsedSource);
  });

  it("builds concealment only for currently visible ranges in a large document", () => {
    const source = Array.from({ length: 10_000 }, (_, index) => `**line ${index}**`).join("\n");
    const view = makeView(source, {
      selection: EditorSelection.single(0),
    });
    expect(forceParsing(view, view.visibleRanges[0]?.to, 100)).toBe(true);
    const concealed = concealedDecorationRanges(view);

    expect(concealed.length).toBeGreaterThan(0);
    expect(concealed.length).toBeLessThan(20_000);
    expect(
      concealed.every((range) =>
        view.visibleRanges.some(
          (visible) => range.from < visible.to && range.to > visible.from,
        ),
      ),
    ).toBe(true);
  });

  it("presents parsed syntax on the first paint of a very long visible line", () => {
    const source = `**${"visible".repeat(2_000)}**\n\naway`;
    const view = makeView(source, {
      selection: EditorSelection.single(source.length),
    });

    expect(view.contentDOM.querySelector(".cm-writing-strong")).not.toBeNull();
    expect(view.contentDOM.textContent).not.toContain("**visible");
    expect(view.state.doc.toString()).toBe(source);
  });
});
