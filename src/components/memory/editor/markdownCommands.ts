// SPDX-License-Identifier: AGPL-3.0-only
import { redo, undo } from "@codemirror/commands";
import { ensureSyntaxTree } from "@codemirror/language";
import {
  EditorSelection,
  type ChangeSet,
  type ChangeSpec,
  type EditorState,
  type SelectionRange,
} from "@codemirror/state";
import type { EditorView } from "@codemirror/view";
import type { MarkdownEditorCommand } from "./MarkdownEditor";

type InlineCommand = "bold" | "italic" | "inlineCode" | "link";
type LineCommand =
  | "paragraph"
  | "heading1"
  | "heading2"
  | "heading3"
  | "blockquote"
  | "bulletList"
  | "numberedList";

interface Wrapper {
  markerLength: number;
}

interface SelectedLine {
  number: number;
  from: number;
  to: number;
  text: string;
  indent: string;
  body: string;
}

interface HeadingPrefix {
  markerLength: number;
  separatorLength: 0 | 1;
}

export function runMarkdownCommand(
  view: EditorView,
  command: MarkdownEditorCommand,
): boolean {
  if (view.state.readOnly || view.compositionStarted) return false;
  if (command === "undo") return undo(view);
  if (command === "redo") return redo(view);
  if (isLineCommand(command)) return runLineCommand(view, command);
  if (!isInlineCommand(command)) return false;

  const ranges = view.state.selection.ranges;
  if (!ranges.every((range) => inlineSelectionIsSafe(view.state, range, command))) {
    return false;
  }

  const transaction = view.state.changeByRange((range) =>
    inlineRangeChange(view.state, range, command),
  );
  view.dispatch(transaction);
  view.focus();
  return true;
}

function isInlineCommand(command: MarkdownEditorCommand): command is InlineCommand {
  return (
    command === "bold" ||
    command === "italic" ||
    command === "inlineCode" ||
    command === "link"
  );
}

function isLineCommand(command: MarkdownEditorCommand): command is LineCommand {
  return (
    command === "paragraph" ||
    command === "heading1" ||
    command === "heading2" ||
    command === "heading3" ||
    command === "blockquote" ||
    command === "bulletList" ||
    command === "numberedList"
  );
}

function isHeadingCommand(
  command: LineCommand,
): command is "paragraph" | "heading1" | "heading2" | "heading3" {
  return (
    command === "paragraph" ||
    command === "heading1" ||
    command === "heading2" ||
    command === "heading3"
  );
}

function inlineSelectionIsSafe(
  state: EditorState,
  range: SelectionRange,
  command: InlineCommand,
): boolean {
  if (range.empty) return true;
  const selected = state.sliceDoc(range.from, range.to);
  if (command === "inlineCode") {
    return !selected.includes("`") && !hasLineBreak(selected);
  }
  if (command === "link") {
    return !selected.includes("]") && !selected.includes("\\") && !hasLineBreak(selected);
  }
  return !hasLineBreak(selected) && selected.trim() === selected;
}

function hasLineBreak(value: string): boolean {
  return value.includes("\n") || value.includes("\r");
}

function inlineRangeChange(
  state: EditorState,
  range: SelectionRange,
  command: InlineCommand,
): { range: SelectionRange; changes: ChangeSpec | readonly ChangeSpec[] } {
  const wrapper = recognizedWrapper(state, range, command);
  if (wrapper) {
    return {
      changes: [
        { from: range.from - wrapper.markerLength, to: range.from },
        { from: range.to, to: range.to + wrapper.markerLength },
      ],
      range: EditorSelection.range(
        range.anchor - wrapper.markerLength,
        range.head - wrapper.markerLength,
      ),
    };
  }

  if (range.empty) {
    const [insert, caretOffset] = emptySkeleton(command);
    return {
      changes: { from: range.from, to: range.to, insert },
      range: EditorSelection.cursor(range.from + caretOffset),
    };
  }

  const [open, close] = inlineDelimiters(command);
  if (command === "link") {
    return {
      changes: [
        { from: range.from, insert: open },
        { from: range.to, insert: close },
      ],
      range: EditorSelection.cursor(range.to + open.length + close.length - 1),
    };
  }
  return {
    changes: [
      { from: range.from, insert: open },
      { from: range.to, insert: close },
    ],
    range: EditorSelection.range(range.anchor + open.length, range.head + open.length),
  };
}

function emptySkeleton(command: InlineCommand): readonly [insert: string, caretOffset: number] {
  if (command === "bold") return ["****", 2];
  if (command === "italic") return ["**", 1];
  if (command === "inlineCode") return ["``", 1];
  return ["[]()", 1];
}

function inlineDelimiters(command: InlineCommand): readonly [open: string, close: string] {
  if (command === "bold") return ["**", "**"];
  if (command === "italic") return ["*", "*"];
  if (command === "inlineCode") return ["`", "`"];
  return ["[", "]()"];
}

function recognizedWrapper(
  state: EditorState,
  range: SelectionRange,
  command: InlineCommand,
): Wrapper | null {
  if (command === "link") return null;
  const candidates =
    command === "bold"
      ? (["**", "__"] as const)
      : command === "italic"
        ? (["*", "_"] as const)
        : (["`"] as const);
  const parentName =
    command === "bold" ? "StrongEmphasis" : command === "italic" ? "Emphasis" : "InlineCode";

  for (const marker of candidates) {
    const from = range.from - marker.length;
    const to = range.to + marker.length;
    if (from < 0 || to > state.doc.length) continue;
    if (state.sliceDoc(from, range.from) !== marker) continue;
    if (state.sliceDoc(range.to, to) !== marker) continue;
    if (hasExactSyntaxNode(state, parentName, from, to)) {
      return { markerLength: marker.length };
    }
  }
  return null;
}

function hasExactSyntaxNode(
  state: EditorState,
  name: string,
  from: number,
  to: number,
): boolean {
  const tree = ensureSyntaxTree(state, to, 100);
  if (!tree) return false;
  let found = false;
  tree.iterate({
    from,
    to,
    enter(node) {
      if (node.name === name && node.from === from && node.to === to) {
        found = true;
        return false;
      }
      return undefined;
    },
  });
  return found;
}

function runLineCommand(view: EditorView, command: LineCommand): boolean {
  const lines = selectedLines(view.state);
  const nonEmpty = lines.filter((line) => line.body.length > 0);
  if (nonEmpty.length === 0) return false;

  const changes = lineChanges(view.state, nonEmpty, command);
  if (changes.length === 0) return false;
  const changeSet = view.state.changes(changes);
  const selection = mapSelection(view.state.selection, changeSet);
  view.dispatch({ changes: changeSet, selection });
  view.focus();
  return true;
}

function selectedLines(state: EditorState): SelectedLine[] {
  const numbers = new Set<number>();
  for (const range of state.selection.ranges) {
    const first = state.doc.lineAt(range.from).number;
    let lastPosition = range.to;
    if (range.to > range.from && range.to < state.doc.length) {
      const lastLine = state.doc.lineAt(range.to);
      if (range.to === lastLine.from) lastPosition -= 1;
    }
    const last = state.doc.lineAt(lastPosition).number;
    for (let number = first; number <= last; number += 1) numbers.add(number);
  }

  return [...numbers]
    .sort((left, right) => left - right)
    .map((number) => {
      const line = state.doc.line(number);
      const indent = line.text.match(/^[ \t]*/)?.[0] ?? "";
      return {
        number,
        from: line.from,
        to: line.to,
        text: line.text,
        indent,
        body: line.text.slice(indent.length),
      };
    });
}

function lineChanges(
  state: EditorState,
  lines: readonly SelectedLine[],
  command: LineCommand,
): ChangeSpec[] {
  if (isHeadingCommand(command)) {
    return headingChanges(state, lines, command);
  }

  const remove = lines.every((line) => recognizedBlockPrefix(state, line, command) !== null);
  let orderedIndex = 0;
  return lines.map((line) => {
    const prefix = recognizedBlockPrefix(state, line, command);
    const at = line.from + line.indent.length;
    if (remove && prefix) return { from: at, to: at + prefix.length };

    const insert =
      command === "blockquote"
        ? "> "
        : command === "bulletList"
          ? "- "
          : `${(orderedIndex += 1)}. `;
    return { from: at, insert };
  });
}

function headingChanges(
  state: EditorState,
  lines: readonly SelectedLine[],
  command: "paragraph" | "heading1" | "heading2" | "heading3",
): ChangeSpec[] {
  const level = command === "paragraph" ? 0 : Number(command.charAt(command.length - 1));
  const changes: ChangeSpec[] = [];
  for (const line of lines) {
    const prefix = recognizedHeadingPrefix(state, line);
    const at = line.from + line.indent.length;
    if (level === 0) {
      if (prefix) {
        changes.push({
          from: at,
          to: at + prefix.markerLength + prefix.separatorLength,
        });
      }
      continue;
    }
    if (prefix) {
      changes.push({
        from: at,
        to: at + prefix.markerLength,
        insert: "#".repeat(level),
      });
    } else {
      changes.push({ from: at, insert: `${"#".repeat(level)} ` });
    }
  }
  return changes;
}

function recognizedHeadingPrefix(
  state: EditorState,
  line: SelectedLine,
): HeadingPrefix | null {
  const match = /^(#{1,6})(?=[ \t]|$)/.exec(line.body);
  if (!match) return null;
  const markerFrom = line.from + line.indent.length;
  const markerLength = match[1].length;
  if (
    !hasSyntaxNodeAt(
      state,
      /^HeaderMark$/,
      markerFrom,
      markerFrom + markerLength,
    )
  ) {
    return null;
  }
  const separator = line.body.charAt(markerLength);
  return {
    markerLength,
    separatorLength: separator === " " || separator === "\t" ? 1 : 0,
  };
}

function recognizedBlockPrefix(
  state: EditorState,
  line: SelectedLine,
  command: "blockquote" | "bulletList" | "numberedList",
): string | null {
  const markerFrom = line.from + line.indent.length;
  if (command === "blockquote") {
    const match = /^>[ \t]?/.exec(line.body);
    return match && hasSyntaxNodeAt(state, /^QuoteMark$/, markerFrom, markerFrom + 1)
      ? match[0]
      : null;
  }
  if (command === "bulletList") {
    const match = /^[-+*][ \t]+/.exec(line.body);
    return match && hasSyntaxNodeAt(state, /^ListMark$/, markerFrom, markerFrom + 1)
      ? match[0]
      : null;
  }
  const match = /^\d{1,9}[.)][ \t]+/.exec(line.body);
  const markerTo = match ? markerFrom + match[0].trimEnd().length : markerFrom;
  return match && hasSyntaxNodeAt(state, /^ListMark$/, markerFrom, markerTo)
    ? match[0]
    : null;
}

function hasSyntaxNodeAt(
  state: EditorState,
  name: RegExp,
  from: number,
  to: number,
): boolean {
  const tree = ensureSyntaxTree(state, to, 100);
  if (!tree) return false;
  let found = false;
  tree.iterate({
    from,
    to,
    enter(node) {
      if (name.test(node.name) && node.from === from && node.to <= to) {
        found = true;
        return false;
      }
      return undefined;
    },
  });
  return found;
}

function mapSelection(selection: EditorSelection, changes: ChangeSet): EditorSelection {
  return EditorSelection.create(
    selection.ranges.map((range) => {
      if (range.empty) {
        const position = changes.mapPos(range.head, 1);
        return EditorSelection.cursor(
          position,
          range.assoc,
          range.bidiLevel ?? undefined,
          range.goalColumn ?? undefined,
        );
      }
      const mapEndpoint = (position: number): number =>
        changes.mapPos(position, position === range.from ? -1 : 1);
      return EditorSelection.range(
        mapEndpoint(range.anchor),
        mapEndpoint(range.head),
        range.goalColumn ?? undefined,
        range.bidiLevel ?? undefined,
      );
    }),
    selection.mainIndex,
  );
}
