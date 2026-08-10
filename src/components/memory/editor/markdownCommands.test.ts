// SPDX-License-Identifier: AGPL-3.0-only
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { history, redoDepth, undoDepth } from "@codemirror/commands";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
import { EditorSelection, EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import type { MarkdownEditorCommand } from "./MarkdownEditor";
import { installCodeMirrorDomPolyfills } from "./editorTestUtils";
import { runMarkdownCommand } from "./markdownCommands";

beforeAll(() => installCodeMirrorDomPolyfills());

const liveViews: EditorView[] = [];

afterEach(() => {
  for (const view of liveViews.splice(0)) view.destroy();
});

function makeView(
  doc: string,
  ranges: ReadonlyArray<readonly [anchor: number, head: number]>,
  mainIndex = 0,
  readOnly = false,
): EditorView {
  const view = new EditorView({
    state: EditorState.create({
      doc,
      selection: EditorSelection.create(
        ranges.map(([anchor, head]) => EditorSelection.range(anchor, head)),
        mainIndex,
      ),
      extensions: [
        EditorState.allowMultipleSelections.of(true),
        EditorState.readOnly.of(readOnly),
        history(),
        markdown({
          base: markdownLanguage,
          addKeymap: false,
          pasteURLAsLink: false,
          completeHTMLTags: false,
        }),
      ],
    }),
  });
  liveViews.push(view);
  return view;
}

function selectionPairs(view: EditorView): Array<[number, number]> {
  return view.state.selection.ranges.map((range) => [range.anchor, range.head]);
}

describe("runMarkdownCommand", () => {
  it.each([
    ["bold", "alpha", "**alpha**", [2, 7]],
    ["italic", "alpha", "*alpha*", [1, 6]],
    ["inlineCode", "alpha", "`alpha`", [1, 6]],
    ["link", "alpha", "[alpha]()", [8, 8]],
  ] as const)("%s wraps one safe selection", (command, source, expected, next) => {
    const view = makeView(source, [[0, source.length]]);

    expect(runMarkdownCommand(view, command)).toBe(true);

    expect(view.state.doc.toString()).toBe(expected);
    expect(selectionPairs(view)).toEqual([next]);
  });

  it.each([
    ["bold", "****", [2, 2]],
    ["italic", "**", [1, 1]],
    ["inlineCode", "``", [1, 1]],
    ["link", "[]()", [1, 1]],
  ] as const)("%s inserts a deterministic empty-selection skeleton", (command, expected, next) => {
    const view = makeView("", [[0, 0]]);

    expect(runMarkdownCommand(view, command)).toBe(true);

    expect(view.state.doc.toString()).toBe(expected);
    expect(selectionPairs(view)).toEqual([next]);
  });

  it("unwraps only a parser-recognized adjacent delimiter pair", () => {
    const view = makeView("**alpha**", [[2, 7]]);

    expect(runMarkdownCommand(view, "bold")).toBe(true);

    expect(view.state.doc.toString()).toBe("alpha");
    expect(selectionPairs(view)).toEqual([[0, 5]]);
  });

  it("formats multiple ranges in one undoable transaction and preserves main selection", () => {
    const view = makeView("one two", [[0, 3], [4, 7]], 1);

    expect(runMarkdownCommand(view, "bold")).toBe(true);
    expect(view.state.doc.toString()).toBe("**one** **two**");
    expect(selectionPairs(view)).toEqual([[2, 5], [10, 13]]);
    expect(view.state.selection.mainIndex).toBe(1);
    expect(undoDepth(view.state)).toBe(1);

    expect(runMarkdownCommand(view, "undo")).toBe(true);
    expect(view.state.doc.toString()).toBe("one two");
    expect(runMarkdownCommand(view, "redo")).toBe(true);
    expect(view.state.doc.toString()).toBe("**one** **two**");
    expect(redoDepth(view.state)).toBe(0);
  });

  it.each([
    ["inlineCode", "a`b"],
    ["inlineCode", "a\nb"],
    ["link", "a]b"],
    ["link", "a\\b"],
    ["link", "a\nb"],
    ["bold", " alpha"],
    ["bold", "alpha "],
    ["italic", "a\nb"],
  ] as const)("%s is an exact no-op for unsafe selection %j", (command, source) => {
    const view = makeView(source, [[0, source.length]]);
    const before = view.state.selection;

    expect(runMarkdownCommand(view, command)).toBe(false);

    expect(view.state.doc.toString()).toBe(source);
    expect(view.state.selection.eq(before)).toBe(true);
    expect(undoDepth(view.state)).toBe(0);
  });

  it.each([
    ["heading1", "  alpha\n\nbeta", "  # alpha\n\n# beta"],
    ["heading3", "## alpha", "### alpha"],
    ["heading3", "##   Heading", "###   Heading"],
    ["paragraph", "## alpha", "alpha"],
    ["paragraph", "##   Heading", "  Heading"],
    ["blockquote", "alpha\nbeta", "> alpha\n> beta"],
    ["bulletList", "  alpha\n\n  beta", "  - alpha\n\n  - beta"],
    ["numberedList", "alpha\nbeta", "1. alpha\n2. beta"],
  ] as const)("%s transforms selected logical lines", (command, source, expected) => {
    const view = makeView(source, [[source.length, 0]]);

    expect(runMarkdownCommand(view, command)).toBe(true);

    expect(view.state.doc.toString()).toBe(expected);
    expect(view.state.selection.main.anchor).toBe(expected.length);
    expect(view.state.selection.main.head).toBe(0);
  });

  it("preserves repeated tabs when the pinned parser treats heading-like source as literal", () => {
    const source = "##\t\tHeading";
    const paragraph = makeView(source, [[0, source.length]]);
    const heading = makeView(source, [[0, source.length]]);

    expect(runMarkdownCommand(paragraph, "paragraph")).toBe(false);
    expect(paragraph.state.doc.toString()).toBe(source);

    expect(runMarkdownCommand(heading, "heading1")).toBe(true);
    expect(heading.state.doc.toString()).toBe(`# ${source}`);
  });

  it.each([
    ["blockquote", "> alpha\n> beta", "alpha\nbeta"],
    ["bulletList", "- alpha\n- beta", "alpha\nbeta"],
    ["numberedList", "1. alpha\n2. beta", "alpha\nbeta"],
  ] as const)("%s toggles a uniformly recognized block back off", (command, source, expected) => {
    const view = makeView(source, [[0, source.length]]);

    expect(runMarkdownCommand(view, command)).toBe(true);
    expect(view.state.doc.toString()).toBe(expected);
  });

  it("deduplicates two cursors that target the same logical line", () => {
    const view = makeView("alpha", [[0, 0], [3, 3]], 1);

    expect(runMarkdownCommand(view, "bulletList")).toBe(true);

    expect(view.state.doc.toString()).toBe("- alpha");
    expect(selectionPairs(view)).toEqual([[2, 2], [5, 5]]);
    expect(view.state.selection.mainIndex).toBe(1);
  });

  it("does not change blank selected lines or neighboring list items", () => {
    const source = "- neighbor\n\nchild\n- neighbor two";
    const childFrom = source.indexOf("child");
    const view = makeView(source, [[childFrom, childFrom + "child".length]]);

    expect(runMarkdownCommand(view, "bulletList")).toBe(true);

    expect(view.state.doc.toString()).toBe("- neighbor\n\n- child\n- neighbor two");
  });

  it("returns false for a line command over an empty document", () => {
    const view = makeView("", [[0, 0]]);

    expect(runMarkdownCommand(view, "heading1")).toBe(false);
    expect(view.state.doc.toString()).toBe("");
    expect(undoDepth(view.state)).toBe(0);
  });

  it("rejects every command while the state is read-only", () => {
    const view = makeView("alpha", [[0, 5]], 0, true);

    const commands: MarkdownEditorCommand[] = ["bold", "heading1", "undo", "redo"];
    for (const command of commands) expect(runMarkdownCommand(view, command)).toBe(false);
    expect(view.state.doc.toString()).toBe("alpha");
  });
});
